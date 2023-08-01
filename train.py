# %%
# basic imports
import random
import numpy as np
import os
import argparse

# libraries for loading image, plotting 
import cv2
# import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from PALMA.utils import train_validate_model     # train validate function
from PALMA.utils import IoU
from PALMA.utils import train_validate_model     # train validate function

from models import UNet, segformer

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


def make_loader(train_set, val_set, test_set, train_batch=4, val_batch=2, train_worker=0, val_worker=0):

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=train_worker)
    valid_loader = DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=val_worker)
    test_loader = DataLoader(test_set, batch_size=val_batch, shuffle=False, num_workers=val_worker)
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='name of the model as it should be saved')
    parser.add_argument('--data_path', type=str, default='/scratch/tmp/j_sten07/data', help='path were the input data is stored')
    parser.add_argument('--output_path', type=str, default='/scratch/tmp/j_sten07/output/', help='path to directory where the output should be stored')
    parser.add_argument('--model', choices=['UNet', 'segformer'], default='UNet', help="the model architecture that should be trained; choose from 'UNet' and 'segformer'")
    parser.add_argument('--epochs', type=int, default=20, help='epochs the model should be trained')
    parser.add_argument('--loss_function', type=str, choices=['dice', 'jaccard'], default='jaccard')
    parser.add_argument('--lr', type=float, default=3e-4, help='maximum learning rate')
    parser.add_argument('--train_batch', type=int, default=4, help='batch size for training data')
    parser.add_argument('--val_batch', type=int, default=2, help='batch size for validation data')
    parser.add_argument('--train_worker', type=int, default=0, help='number of workers for training data')
    parser.add_argument('--val_worker', type=int, default=0, help='number of workers for validation data')
    opt = parser.parse_args()

    # load dataset and create data loader
    train_dataset, val_dataset, test_dataset = load_datasets(opt.data_path)
    train_loader, val_loader, test_loader = make_loader(train_dataset, val_dataset, test_dataset, opt.train_batch, opt.val_batch, opt.train_worker, opt.val_worker)

    # TODO: check if empty_cache() is necessary 
    # torch.cuda.empty_cache()

    # check if gpu is available and set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # set loss function
    # reference : https://smp.readthedocs.io/en/latest/losses.html
    if opt.loss_function == 'jaccard':
        criterion = smp.losses.JaccardLoss('multiclass', log_loss = False, smooth=0.0)
    if opt.loss_function == 'dice':
        criterion = smp.losses.DiceLoss('multiclass', log_loss = False, smooth=0.0)
    

    # MODEL HYPERPARAMETERS
    N_EPOCHS = opt.epochs
    NUM_CLASSES = 6
    MAX_LR = opt.lr

    # create model
    if opt.model == 'UNet':
        model = UNet(in_channels=3, out_channels=NUM_CLASSES, layer_channels=[64, 128, 256, 512]).to(device)
    if opt.model == 'segformer':
        model = segformer(in_channels=3, num_classes=NUM_CLASSES).to(device)
    
    # set model name
    if not opt.name == None:
        modelname = opt.name
    else:
        modelname = f"{opt.model}_{opt.epochs}epochs_{opt.loss_function}loss_{opt.lr}lr"
    
    # create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS, steps_per_epoch = len(train_loader), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos')

    # run model training with given arguments
    _ = train_validate_model(model, N_EPOCHS, modelname, criterion, optimizer, 
                         device, train_loader, val_loader, IoU, 'metrices',
                         NUM_CLASSES, lr_scheduler = None, output_path = opt.output_path)
