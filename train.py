# %%
# basic imports
import argparse

from utils import train_validate_model     # train validate function
from utils import IoU

from utils import load_datasets
from utils import make_loader

from models import UNet, segformer

import torch
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim

import segmentation_models_pytorch as smp



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='name of the model as it should be saved')
    parser.add_argument('--data_path', type=str, default='/scratch/tmp/j_sten07/data', help='path were the input data is stored')
    parser.add_argument('--output_path', type=str, default='/scratch/tmp/j_sten07/output', help='path to directory where the output should be stored')
    parser.add_argument('--model', choices=['unet', 'segformer'], default='unet', help="the model architecture that should be trained; choose from 'UNet' and 'segformer'")
    parser.add_argument('--epochs', type=int, default=20, help='epochs the model should be trained')
    parser.add_argument('--loss_function', type=str, choices=['dice', 'jaccard'], default='jaccard')
    parser.add_argument('--lr', type=float, default=3e-4, help='maximum learning rate')
    parser.add_argument('--train_batch', type=int, default=4, help='batch size for training data')
    parser.add_argument('--val_batch', type=int, default=2, help='batch size for validation data')
    parser.add_argument('--train_worker', type=int, default=0, help='number of workers for training data')
    parser.add_argument('--val_worker', type=int, default=0, help='number of workers for validation data')
    parser.add_argument('--stop_threshold', type=int, default=-1, help='number of epochs without improvement in validation loss after that the training should be stopped')
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='wether to use the implemented learning rate scheduler or not')
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
    if opt.model == 'unet':
        model = UNet(in_channels=3, out_channels=NUM_CLASSES, layer_channels=[64, 128, 256, 512]).to(device)
    if opt.model == 'segformer':
        model = segformer(in_channels=3, num_classes=NUM_CLASSES).to(device)
    
    # set model name
    if not opt.name == None:
        modelname = opt.name
    else:
        modelname = f"{opt.model}_{opt.epochs}epochs_{opt.loss_function}loss_{opt.lr}lr_lrscheduler{opt.lr_scheduler}_{opt.train_batch}batches"
    
    # create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    if opt.lr_scheduler: 
        lr_scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS, steps_per_epoch = len(train_loader), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos')
    else:
        lr_scheduler = None

    # run model training with given arguments
    _ = train_validate_model(model, N_EPOCHS, modelname, criterion, optimizer, 
                         device, train_loader, val_loader, IoU, 
                         NUM_CLASSES, lr_scheduler = lr_scheduler, output_path = opt.output_path, early_stop=opt.stop_threshold)
