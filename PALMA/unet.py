# %%
# basic imports
import random
import numpy as np
import os
import argparse

# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# libraries for loading image, plotting 
import cv2
# import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from utils import train_validate_model     # train validate function
from utils import IoU
from utils import train_validate_model     # train validate function

from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim

import segmentation_models_pytorch as smp


# %% [markdown]
# ## 1. Dataset

# %%

# Now replace RGB to integer values to be used as labels.
#Find pixels with combination of RGB for the above defined arrays...
#if matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    impervious = [255, 255, 255]
    building = [0, 0, 255]
    vegetation = [0, 255, 255]
    tree = [0, 255, 0]
    car = [255, 255, 0]
    clutter = [255, 0, 0]

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == impervious,axis=-1)] = 0
    label_seg [np.all(label==building,axis=-1)] = 1
    label_seg [np.all(label==vegetation,axis=-1)] = 2
    label_seg [np.all(label==tree,axis=-1)] = 3
    label_seg [np.all(label==car,axis=-1)] = 4
    label_seg [np.all(label==clutter,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

# %%


preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

# %%

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['impervious', 'building', 'vegetation', 'tree', 'car', 'clutter']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            patch_size=512
    ):
        self.im_ids = os.listdir(images_dir) 
        # self.im_ids = list(filter(lambda x: x.endswith('11_RGB.tif'), self.im_ids))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.im_ids]
        self.mask_ids = os.listdir(masks_dir) 
        # self.mask_ids = list(filter(lambda x: x.endswith('11_label.tif'), self.mask_ids))
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        self.dims = (patch_size, patch_size)
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.dims, interpolation=cv2.INTER_NEAREST)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # cv2 reads image as BGR, change to RGB
        mask = cv2.resize(mask, self.dims, interpolation=cv2.INTER_NEAREST)
        mask = rgb_to_2D_label(mask)
        # print(self.images_fps[i])
        
        # # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        # if len(self.class_values) < len(self.CLASSES):
        #     mask = np.c_[np.zeros((np.shape(mask)[0], np.shape(mask)[1], 1)), mask] # add column to make everything not in selected classes background
        mask = torch.from_numpy(mask).long()
        
        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
            
        return image, mask
        
    def __len__(self):
        return len(self.im_ids)

# %% [markdown]
# #### Dataloaders
# 
# - Dataloaders help load data in batches
# - We'll need to define separate dataloaders for training, validation and test sets
# 
# 

# %%

def load_datasets(data_dir):
    x_train_dir = os.path.join(data_dir, 'rgb')
    y_train_dir = os.path.join(data_dir, 'label')

    x_valid_dir = os.path.join(data_dir, 'rgb_valid')
    y_valid_dir = os.path.join(data_dir, 'label_valid')

    x_test_dir = os.path.join(data_dir, 'rgb_test')
    y_test_dir = os.path.join(data_dir, 'label_test')

    CLASSES=['impervious', 'building', 'vegetation', 'tree', 'car', 'clutter']

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        # augmentation=get_training_augmentation(), 
        preprocessing=preprocess,
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        # augmentation=get_validation_augmentation(), 
        preprocessing=preprocess,
        classes=CLASSES,
    )

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        # augmentation=get_validation_augmentation(), 
        preprocessing=preprocess,
        classes=CLASSES,
    )

    return train_dataset, valid_dataset, test_dataset

def make_loader(train_set, val_set, test_set):

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)#, num_workers=4)
    valid_loader = DataLoader(val_set, batch_size=1, shuffle=False)#, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)#, num_workers=4)




# %% [markdown]
# ## 2. Network

# %%
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1, padding = 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride =1, padding = 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

# %%
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, layer_channels):
        super(UNetEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        # Double Convolution blocks
        for num_channels in layer_channels:
            self.encoder.append(double_conv(in_channels, num_channels))
            in_channels = num_channels
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Pass input image through Encoder blocks
        # and return outputs at each stage
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections			
			
			
class UNetDecoder(nn.Module):
    def __init__(self, layer_channels):
        super(UNetDecoder, self).__init__()
        self.decoder = nn.ModuleList()

        # Decoder layer Double Convolution blocks
        # and upsampling blocks
        self.decoder = nn.ModuleList()        
        for num_channels in reversed(layer_channels):
            self.decoder.append(nn.ConvTranspose2d(num_channels*2, num_channels, kernel_size=2, stride=2))
            self.decoder.append(double_conv(num_channels*2, num_channels))
        
    
    def forward(self, x, skip_connections):
        for idx in range(0, len(self.decoder), 2):
            # upsample output and reduce channels by 2
            x = self.decoder[idx](x)
            
            # if skip connection shape doesn't match, resize
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # concatenate and pass through double_conv block
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        return x		



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layer_channels):
        super(UNet, self).__init__()
        
        # Encoder and decoder modules
        self.encoder = UNetEncoder(in_channels, layer_channels)
        self.decoder = UNetDecoder(layer_channels)

        # conv layer to transition from encoder to decoder and 
        # 1x1 convolution to reduce num channels to out_channels
        self.bottleneck = double_conv(layer_channels[-1], layer_channels[-1]*2)
        self.final_conv = nn.Conv2d(layer_channels[0], out_channels, kernel_size=1)
        
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder blocks
        encoder_output, skip_connections = self.encoder(x)

        # transition between encoder and decoder
        x = self.bottleneck(encoder_output)

        # we need the last skip connection first
        # so reversing the list 
        skip_connections = skip_connections[::-1]

        # Decoder blocks
        x = self.decoder(x, skip_connections)

        # final 1x1 conv to match input size
        return self.final_conv(x)          	




# %% [markdown]
# # 4. Evaluate : Evaluate the model on Test Data and visualize results 

# TODO: store performance on test data 


# %%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='UNet_baseline', help='name of the model as it should be saved')
    parser.add_argument('--data_path', type=str, default='/scratch/tmp/j_sten07/data', help='path were the input data is stored')
    parser.add_argument('--output_path', type=str, default='/scratch/tmp/j_sten07/output/', help='path to directory where the output should be stored')
    opt = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_datasets(opt.data_path)

    train_loader, val_loader, test_loader = make_loader(train_dataset, val_dataset, test_dataset)

    # TODO: check if empty_cache() is necessary 
    # torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # reference : https://smp.readthedocs.io/en/latest/losses.html
    # criterion = smp.losses.DiceLoss('multiclass', log_loss = True, smooth=1.0)
    criterion = smp.losses.JaccardLoss('multiclass', log_loss = False, smooth=0.0)

    # MODEL HYPERPARAMETERS
    # TODO: make input arguments
    N_EPOCHS = 20
    NUM_CLASSES = 6
    MAX_LR = 3e-4
    # MODEL_NAME = 'UNet_baseline_jaccardloss'

    # create model, optimizer, lr_scheduler and pass to training function
    model = UNet(in_channels=3, out_channels=NUM_CLASSES, layer_channels=[64, 128, 256, 512]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS, steps_per_epoch = len(train_loader), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos')

    _ = train_validate_model(model, N_EPOCHS, opt.name, criterion, optimizer, 
                         device, train_loader, val_loader, IoU, 'metrices',
                         NUM_CLASSES, lr_scheduler = None, output_path = opt.output_path)
