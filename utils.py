# basic imports
import os
import math
from datetime import datetime
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from collections import namedtuple
from PIL import Image

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
# import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler


###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

norms = {
    'imagenet': {'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)},
    'potsdam': {'mean':(0.349, 0.371, 0.347), 'std':(0.1196, 0.1164, 0.1197)},
    'potsdam_irrg': {'mean':(0.3823, 0.3625, 0.3364), 'std':(0.1172, 0.1167, 0.1203)},
    'floodnet': {'mean':(0.4159, 0.4499, 0.3466), 'std':(0.1297, 0.1197, 0.1304)},
    'vaihingen': {'mean':(0.4731, 0.3206, 0.3182), 'std':(0.1970, 0.1306, 0.1276)},
}

def normalize_images(dataset):
    if dataset in norms.keys():
        return transforms.Compose([
                    transforms.Normalize(mean=np.array(norms[dataset]['mean']), std=np.array(norms[dataset]['std'])) # Image Net mean and std
                ])
    else:
        return transforms.Compose([
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225)) # Image Net mean and std
                ])

augmentation = torch.nn.Sequential(
    # transforms.ToTensor(),
    transforms.ColorJitter(
        brightness=0.5, 
        contrast=1, 
        saturation=0.1, 
        hue=0.5
    )
)

# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
def inverse_transform(dataset):
    if dataset in norms.keys():
        return transforms.Compose([
            transforms.Normalize(-np.array(norms[dataset]['mean'])/np.array(norms[dataset]['std']), 1/np.array(norms[dataset]['std']))
    ])
    else:
        return transforms.Compose([
            transforms.Normalize((-0.485/0.229, -0.56/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])



###################################
# METRIC CLASS DEFINITION
###################################

class meanIoU:
    """ Class to find the mean IoU using confusion matrix approach """    
    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, y_preds, labels):
        """ Function finds the IoU for the input batch
        and add batch metrics to overall metrics """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def compute(self):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        # print(hist)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
        
class IoU:
    """ Class to find the mean IoU using confusion matrix approach """    
    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        # mask only valid labels (this step should be irrelevant usually)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # calculate correctness of segementation by assigning numbers and count them
        # e.g. for 6 classes [0:5], 
            # 7 is a class 2 pixel segemented correctly (6*1+1)
            # 16 is a class 3 pixel segmented as class 5 (6*2+4)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, y_preds, labels):
        """ Function finds the IoU for the input batch
        and add batch metrics to overall metrics """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def compute(self, matrix = None):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        # if a matrix is given as argument to the function, compute the metrices based on that matrix 
        if matrix:
            hist = matrix
        # divide number of pixels segmented correctly (area of overlap) 
        # by number of pixels that were segmented in this class and that should have been segmented in this class (hist.sum(axis=1) + hist.sum(axis=0))
        # minus 1 time the pixels segmented correctly in the denominator as they are in both sums
        # IoU = TP / (TP + FP + FN)
        # TP = np.diag(hist); FP = hist.sum(axis=0) - np.diag(hist); FN = hist.sum(axis=1) - np.diag(hist) ?
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) 
        # calculate mean of IoU per class
        mean_iu = np.nanmean(iu)
        # calculate accuracy
        accuracy = np.diag(hist).sum() / hist.sum().sum()
        # class_accuracy = (np.diag(hist) + (hist.sum().sum() - hist.sum(axis=1) - hist.sum(axis=0) + np.diag(hist))) / (hist.sum().sum())
        # calculate dice coefficient / f1 score
        f1 = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        meanf1 = np.nanmean(f1)
        # return {'hist' : hist, 'accuracy' : accuracy, 'classwise_accuracy' : class_accuracy, 'miou' : mean_iu, 'classwise_iou' : iu}
        return {'accuracy' : accuracy, 'miou' : mean_iu, 'classwise_iou' : iu, 'classwise_f1': f1, 'f1_mean': meanf1, 'matrix': hist}

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
        


###################################
# POLY LR DECAY SCHEDULER DEFINITION
###################################

class polynomial_lr_decay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    
    Reference:
        https://github.com/cmpark0126/pytorch-polynomial-lr-decay
    """
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


###################################
# FUNCTION TO PLOT TRAINING, VALIDATION CURVES
###################################


def plot_training_results(df, model_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('validationLoss', color='tab:blue')
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.suptitle(f'{model_name} Training, Validation Curves')
    plt.show()



###################################
# FUNCTION TO EVALUATE MODEL ON DATALOADER
###################################

def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())
            
    print(len(dataloader))
    print(total_loss)

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric



###################################
# FUNCTION TO TRAIN, VALIDATE MODEL ON DATALOADER
###################################

def train_validate_model(model, num_epochs, model_name, criterion, optimizer, 
                         device, dataloader_train, dataloader_valid, 
                         metric_class, num_classes, lr_scheduler = None,
                         output_path = '.', early_stop = -1):
    """Train model and validate
    # TODO
    Args:
        model
        num_epochs
        model_name
        criterion
        optimizer
        device
        dataloader_train
        dataloader_valid
        metric_class
        num_classes
        lr_scheduler 
        output_path
        early_stop: number of epochs after which the training 
            should be stopped if the validation loss did not increase; 
            -1: do not apply early stop
    """
    early_stop_threshold = early_stop
    
    # initialize placeholders for running values    
    results = []
    min_val_loss = np.Inf
    len_train_loader = len(dataloader_train)
    
    model_folder = os.path.join(output_path, model_name)
    lastmodel_path = f"{model_folder}/{model_name}_last.pt"
    print(lastmodel_path)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        if os.path.exists(lastmodel_path):
            print('model already exists. load last states..')
            checkpoint = torch.load(lastmodel_path)
            model.load_state_dict(checkpoint['model'].state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            if lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
            results = checkpoint['results']
            

    if results:
        epochs_trained = results[-1]['epoch']+1
        # get minimum validation loss from previous training
        min_val_loss = min(results, key=lambda x:x['validationLoss'])['validationLoss'] 
        best_epoch = min(results, key=lambda x:x['validationLoss'])['epoch'] 
        print(f"Best epoch: {best_epoch+1}")
        if epochs_trained >= num_epochs:
            print(f"Existing model already trained for at least {num_epochs} epochs")
            return  # terminate the training loop
    else:
        epochs_trained = 0
        best_epoch = -1
    
    # move model to device
    model.to(device)
    
    for epoch in range(epochs_trained, num_epochs):
        # epoch = epoch + epochs_trained
        
        print(f"Starting {epoch + 1} epoch ...")
        starttime = datetime.now()
        
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            
            # Forward pass
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            train_loss += loss.item()
              
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)

        endtime_train = datetime.now()
        validation_loss, validation_metric = evaluate_model(
                        model, dataloader_valid, criterion, metric_class, num_classes, device)
        
        endtime_val = datetime.now()
        
        duration_training = endtime_train - starttime
        
        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, validation_metrices: {validation_metric}, trainingDuration {duration_training}')
        
        # store results
        results.append({'epoch': epoch, 
                        'trainLoss': train_loss, 
                        'validationLoss': validation_loss, 
                        'metrices': validation_metric,
                        'duration_train': duration_training,
                       })
        
        torch.save({
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            # 'scheduler_state_dict': lr_scheduler.state_dict(),
            'min_val_loss': min_val_loss,
            'results': results,
            'epoch': epoch,
        }, f"{output_path}/{model_name}/{model_name}_last.pt")
        
        # if validation loss has decreased, save model and reset variable
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
            best_epoch = epoch
            torch.save({
                'model': model,
                # 'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                # 'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler,
                # 'scheduler_state_dict': lr_scheduler.state_dict(),
                'min_val_loss': min_val_loss,
                'results': results,
                'epoch': epoch,
            }, f"{output_path}/{model_name}/{model_name}_best.pt")
            print('best model saved')
        elif early_stop_threshold != -1:
            if epoch - best_epoch > early_stop_threshold:
                # stop training if validation_loss did not improve for early_stop_threshold epochs
                print(f"Early stopped training at epoch {epoch} because loss did not improve for {early_stop_threshold} epochs")
                break  # terminate the training loop
        



    # plot results
    results = pd.DataFrame(results)
    plot_training_results(results, model_name)
    return results


###################################
# FUNCTION TO VISUALIZE MODEL PREDICTIONS
###################################

# Constants for Standard color mapping
# reference : https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py


def train_id_to_color(classes):
    Label = namedtuple( "Label", [ "name", "train_id", "color"])
    if len(classes) == 6:
        drivables = [ 
            Label(classes[0], 0, (255, 255, 255)), 
            Label(classes[1], 1, (0, 0, 255)), 
            Label(classes[2], 2, (0, 255, 255)), 
            Label(classes[3], 3, (0, 255, 0)), 
            Label(classes[4], 4, (255, 255, 0)), 
            Label(classes[5], 5, (255, 0, 0))
        ]
    elif len(classes) == 10:
        drivables = [ 
            Label(classes[0], 0, (0, 0, 0)), 
            Label(classes[1], 1, (255, 0, 0)), 
            Label(classes[2], 2, (200, 90, 90)), 
            Label(classes[3], 3, (130, 130, 0)), 
            Label(classes[4], 4, (150, 150, 150)), 
            Label(classes[5], 5, (0, 255, 255)),
            Label(classes[6], 6, (0, 0, 255)), 
            Label(classes[7], 7, (255, 0, 255)), 
            Label(classes[8], 8, (250, 250, 0)), 
            Label(classes[9], 9, (0, 255, 0)) 
        ]
    else:
        return
    
    id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
    id_to_color = np.array(id_to_color)
    
    legend_elements = []
    for i, c in enumerate(classes):
        legend_elements.append(Patch(facecolor=id_to_color[i]/255, label=c))
        
    return id_to_color, legend_elements


# legend_elements = [
#     Patch(facecolor=train_id_to_color[0]/255, label=drivables[0].name),  
#     Patch(facecolor=train_id_to_color[1]/255, label=drivables[1].name),
#     Patch(facecolor=train_id_to_color[2]/255, label=drivables[2].name),
#     Patch(facecolor=train_id_to_color[3]/255, label=drivables[3].name),
#     Patch(facecolor=train_id_to_color[4]/255, label=drivables[4].name),
#     Patch(facecolor=train_id_to_color[5]/255, label=drivables[5].name),
#                   ]

diff_legend = [
    Patch(facecolor='green', label='true'), 
    Patch(facecolor='red', label='false'), 
]

def visualize_predictions(model : torch.nn.Module, 
                          dataSet : Dataset,  
                          axes, 
                          device :torch.device, 
                          numTestSamples : int,
                          # id_to_color : np.ndarray = train_id_to_color, 
                          seed : int = None, 
                          norm_dataset = 'own', 
                          # rgb = True, 
                          classes=None):
    """Function visualizes predictions of input model on samples from
    cityscapes dataset provided

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
        id_to_color (np.ndarray) : array to map class to colormap
    """
    model.to(device=device)
    model.eval()

    rgcmap = colors.ListedColormap(['green','red'])
    
    np.random.seed(seed)

    # predictions on random samples
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    # _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 4))
    
    id_to_color, legend_elements = train_id_to_color(classes)
    id_to_rg = np.array([[255, 0, 0], [0, 150, 0]])
    
    for i, sampleID in enumerate(testSamples):
        inputImage, gt = dataSet[sampleID]

        # input rgb image   
        inputImage = inputImage.to(device)
        if norm_dataset: 
            inv_norm = inverse_transform(norm_dataset)
            landscape = inv_norm(inputImage).permute(1, 2, 0).cpu().detach().numpy()
        else: 
            landscape = inputImage.permute(1, 2, 0).cpu().detach().numpy()
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title(dataSet.get_name(sampleID))

        # groundtruth label image
        label_class = gt.cpu().detach().numpy()
        axes[i, 1].imshow(id_to_color[label_class])
        axes[i, 1].set_title("Groundtruth Label")

        # predicted label image
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
        label_class_predicted = y_pred.cpu().detach().numpy()    
        axes[i, 2].imshow(id_to_color[label_class_predicted])
        axes[i, 2].legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor=(-0.7, 0.9))
        axes[i, 2].set_title("Predicted Label")

        # difference groundtruth and prediction
        diff = label_class == label_class_predicted
        axes[i, 3].imshow(id_to_rg[diff*1])#, cmap = rgcmap) # make int to map 0 and 1 to cmap, otherwise a 
        axes[i, 3].legend(handles=diff_legend)
        axes[i, 3].set_title("Difference")
        # print(diff*1)
        # issue: if the whole image is predicted wrong, it is visualized green (probably because imshow simply takes first color from cmap?)

    plt.show()
    
    

def compare_models_onOneImage(model1 : torch.nn.Module, 
                              model2 : torch.nn.Module, 
                              dataset: Dataset,
                              im_name : str,  
                              device :torch.device, 
                              normalization = True, 
                              classes=None,
                              norm_dataset='own',
                              model1_label = 'U-Net',
                              model2_label = 'SegFormer',
                              plot_title = None
                             ):
    """Function visualizes predictions of input model on samples from
    cityscapes dataset provided

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
        id_to_color (np.ndarray) : array to map class to colormap
    """
    _, axes = plt.subplots(2, 3, figsize=(4*5, 3 * 3))
    
    if plot_title:
        _.suptitle(plot_title, fontsize=20)
    
    model1.to(device=device)
    model1.eval()
    
    model2.to(device=device)
    model2.eval()
    
    #################
    # predictions on random samples
    # testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    imId = dataset.get_id_by_name(im_name)
    inputImage, gt = dataset[imId]
    # _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 4))
    
    id_to_color, legend_elements = train_id_to_color(classes)
    id_to_rg = np.array([[255, 0, 0], [0, 150, 0]])

    # input rgb image   
    inputImage = inputImage.to(device)
    if norm_dataset: 
        inv_norm = inverse_transform(norm_dataset)
        landscape = inv_norm(inputImage).permute(1, 2, 0).cpu().detach().numpy()
    else: 
        landscape = inputImage.permute(1, 2, 0).cpu().detach().numpy()
    axes[0, 0].imshow(landscape)
    axes[0, 0].set_title(im_name)

    # groundtruth label image
    label_class = gt.cpu().detach().numpy()
    axes[1, 0].imshow(id_to_color[label_class])
    axes[1, 0].set_title("Groundtruth Label")

    # predicted label image
    y_pred1 = torch.argmax(model1(inputImage.unsqueeze(0)), dim=1).squeeze(0)
    label_class_predicted1 = y_pred1.cpu().detach().numpy()    
    axes[0, 1].imshow(id_to_color[label_class_predicted1])
    axes[0, 1].set_title("Prediction "+model1_label)

    # difference groundtruth and prediction
    diff = label_class == label_class_predicted1
    axes[0, 2].imshow(id_to_rg[diff*1])#, cmap = rgcmap) # make int to map 0 and 1 to cmap, otherwise a 
    axes[0, 2].legend(handles=diff_legend)
    axes[0, 2].set_title("Difference")
    
    # predicted label image
    y_pred2 = torch.argmax(model2(inputImage.unsqueeze(0)), dim=1).squeeze(0)
    label_class_predicted2 = y_pred2.cpu().detach().numpy()    
    axes[1, 1].imshow(id_to_color[label_class_predicted2])
    axes[1, 1].legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor=(-0.6, 1.4))
    axes[1, 1].set_title("Prediction "+model2_label)

    # difference groundtruth and prediction
    diff = label_class == label_class_predicted2
    axes[1, 2].imshow(id_to_rg[diff*1])#, cmap = rgcmap) # make int to map 0 and 1 to cmap, otherwise a 
    axes[1, 2].legend(handles=diff_legend)
    axes[1, 2].set_title("Difference")

    plt.show()


#######################################
# Data processing utils
#######################################


# basic imports
import numpy as np
import os

import cv2

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


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
    label_seg [np.all(label==impervious,axis=-1)] = 0
    label_seg [np.all(label==building,axis=-1)] = 1
    label_seg [np.all(label==vegetation,axis=-1)] = 2
    label_seg [np.all(label==tree,axis=-1)] = 3
    label_seg [np.all(label==car,axis=-1)] = 4
    label_seg [np.all(label==clutter,axis=-1)] = 5

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg



class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    # CLASSES = ['impervious', 'building', 'vegetation', 'tree', 'car', 'clutter']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            normalization=False,
            patch_size=512,
            dataset=None
    ):
        self.im_ids = sorted(os.listdir(images_dir))
        # self.im_ids = list(filter(lambda x: x.endswith('11_RGB.tif'), self.im_ids))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.im_ids]
        self.mask_ids = sorted(os.listdir(masks_dir))
        # self.mask_ids = list(filter(lambda x: x.endswith('11_label.tif'), self.mask_ids))
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        self.CLASSES = classes
        
        self.dims = (patch_size, patch_size)
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.normalization = normalization
        if normalization:
            self.normalize = normalize_images(dataset)
            
    
    def __getitem__(self, i):
        
        # read data
        # print(self.images_fps[i])
        # print(self.masks_fps[i])
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.dims, interpolation=cv2.INTER_NEAREST)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # cv2 reads image as BGR, change to RGB
        mask = cv2.resize(mask, self.dims, interpolation=cv2.INTER_NEAREST)
        # (print(np.unique(mask)))
        if len(self.CLASSES) == 6:
            mask = rgb_to_2D_label(mask)
        mask = mask[:,:,0]
        
        # else:
        #     mask = mask[:,:,0]
        
        # print(self.images_fps[i])
        
        # # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        # if len(self.class_values) < len(self.CLASSES):
        #     mask = np.c_[np.zeros((np.shape(mask)[0], np.shape(mask)[1], 1)), mask] # add column to make everything not in selected classes background
        mask = torch.from_numpy(mask).long()
        
        # # apply augmentations
        if self.augmentation:
            # sample = self.augmentation(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
            image = Image.fromarray(image)
            image = self.augmentation(image)
        
        # apply preprocessing
        image = preprocess(image)
        if self.normalization:
            image = self.normalize(image)
            
        return image, mask
    
    def get_name(self, i):
        return self.im_ids[i]
    
    def get_id_by_name(self, im_name):
        for i, name in enumerate(self.im_ids):    
            if name == im_name:
                return i
        return 
        
    def __len__(self):
        return len(self.im_ids)


def load_datasets(data_dir, random_split = False, augmentation = None, normalize = True, classes='potsdam', patch_size=512, only_test=False, dataset='potsdam'):

    if classes == 'potsdam':
        CLASSES=['impervious', 'building', 'vegetation', 'tree', 'car', 'clutter']
    if classes == 'floodnet':
        CLASSES = ['background', 'building-flooded', 'building-non-flooded', 'road-flooded', 'road-non-flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']
    
    x_test_dir = os.path.join(data_dir, 'rgb_test')
    y_test_dir = os.path.join(data_dir, 'label_test')

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        # augmentation=get_validation_augmentation(), 
        augmentation=augmentation, 
        normalization=normalize,
        classes=CLASSES,
        patch_size=patch_size,
        dataset=dataset
    )
    
    if only_test:
        return test_dataset        
    
    # use train directory as input for training and validation data and split them randomly in two subsets
    if random_split: 
        x_train_dir = os.path.join(data_dir, 'rgb')
        y_train_dir = os.path.join(data_dir, 'label')
        
        training_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            augmentation=augmentation, 
            normalization=normalize,
            classes=CLASSES,
            patch_size=patch_size,
            dataset=dataset
        )

        generator = torch.Generator().manual_seed(42)
        train_dataset, valid_dataset = torch.utils.data.random_split(training_dataset, [math.floor(0.75*len(training_dataset)), math.ceil(0.25*len(training_dataset))], generator=generator)
        
    else:
        
        x_train_dir = os.path.join(data_dir, 'rgb')
        y_train_dir = os.path.join(data_dir, 'label')

        x_valid_dir = os.path.join(data_dir, 'rgb_valid')
        y_valid_dir = os.path.join(data_dir, 'label_valid')

        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            # augmentation=get_training_augmentation(), 
            augmentation=augmentation, 
            normalization=normalize,
            classes=CLASSES,
            patch_size=patch_size,
            dataset=dataset
        )

        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir, 
            # augmentation=get_validation_augmentation(), 
            augmentation=augmentation, 
            normalization=normalize,
            classes=CLASSES,
            patch_size=patch_size,
            dataset=dataset
        )
        

    return train_dataset, valid_dataset, test_dataset


def make_loader(train_set, val_set, test_set, train_batch=4, val_batch=2, train_worker=0, val_worker=0):

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=train_worker)
    valid_loader = DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=val_worker)
    test_loader = DataLoader(test_set, batch_size=val_batch, shuffle=False, num_workers=val_worker)
    
    return train_loader, valid_loader, test_loader