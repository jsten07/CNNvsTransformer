# basic imports
import os
from datetime import datetime
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# For dice loss function
import segmentation_models_pytorch as smp

# for interactive widgets
import IPython.display as Disp
from ipywidgets import widgets

###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])


# Constants for Standard color mapping
# reference : https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
    Label('impervious', 0, (255, 255, 255)), 
    Label('building', 1, (0, 0, 255)), 
    Label('vegetation', 2, (0, 255, 255)), 
    Label('tree', 3, (0, 255, 0)), 
    Label('car', 4, (255, 255, 0)), 
    Label('clutter', 5, (255, 0, 0))
]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)



###################################
# FUNCTION TO GET TORCH DATALOADER  #
###################################

def get_dataloaders(train_set, val_set, test_set, batch_size=8):
    train_dataloader = DataLoader(train_set, batch_size=batch_size,drop_last=True)
    val_dataloader   = DataLoader(val_set, batch_size=batch_size)
    test_dataloader  = DataLoader(test_set, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader   



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
        # print(hist[0:500])
        return hist

    def update(self, y_preds, labels):
        """ Function finds the IoU for the input batch
        and add batch metrics to overall metrics """
        # print(y_preds[:,:,0,0:500])
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def compute(self):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        # divide number of pixels segmented correctly (area of overlap) 
        # by number of pixels that were segmented in this class and that should have been segmented in this class (hist.sum(axis=1) + hist.sum(axis=0))
        # minus 1 time the pixels segmented correctly in the denominator as they are in both sums
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) 
        # calculate mean of IoU per class
        mean_iu = np.nanmean(iu)
        # calculate accuracy
        accuracy = np.diag(hist).sum() / hist.sum()
        class_accuracy = np.diag(hist) / hist.sum(axis = 1)
        # return {'hist' : hist, 'accuracy' : accuracy, 'classwise_accuracy' : class_accuracy, 'miou' : mean_iu, 'classwise_iou' : iu}
        return {'accuracy' : accuracy, 'classwise_accuracy' : class_accuracy, 'miou' : mean_iu, 'classwise_iou' : iu}

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

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric



###################################
# FUNCTION TO TRAIN, VALIDATE MODEL ON DATALOADER
###################################

def train_validate_model(model, num_epochs, model_name, criterion, optimizer, 
                         device, dataloader_train, dataloader_valid, 
                         metric_class, metric_name, num_classes, lr_scheduler = None,
                         output_path = '.'):
    early_stop_threshold = 5
    
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
            # print(inputs)
            # # print(len(inputs))
            # # print(max(inputs))
            # print(y_preds)
            # print(torch.max(y_preds))
            # # print(len(y_preds))
            # # print(max(y_preds))
            # print(labels)
            # print(inputs.shape)
            # print(y_preds.shape)
            # print(labels.shape)
            # # print(len(labels))
            # # print(max(labels))
            loss = criterion(y_preds, labels)
            # print(loss)
            train_loss += loss.item()
            # print(train_loss)
              
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)
        # train_loss_alt, train_metric = evaluate_model(
        #                 model, dataloader_train, criterion, metric_class, num_classes, device)
        # print(train_loss_alt)
        endtime_train = datetime.now()
        validation_loss, validation_metric = evaluate_model(
                        model, dataloader_valid, criterion, metric_class, num_classes, device)
        
        endtime_val = datetime.now()
        
        duration_training = endtime_train - starttime
        
        # print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, train_metrices: {train_metric}, validation_metrices: {validation_metric}')
        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, validation_metrices: {validation_metric}, ttainingDuration: {duration_training}')
        
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
        }, f"{output_path}{model_name}/{model_name}_last.pt")
        
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
            }, f"{output_path}{model_name}/{model_name}_best.pt")
            print('best model saved')
        elif epoch - best_epoch > early_stop_threshold:
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

def visualize_predictions(model : torch.nn.Module, dataSet : Dataset,  
        axes, device :torch.device, numTestSamples : int,
        id_to_color : np.ndarray = train_id_to_color):
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

    # predictions on random samples
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    # _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 4))
    
    for i, sampleID in enumerate(testSamples):
        inputImage, gt = dataSet[sampleID]

        # input rgb image   
        inputImage = inputImage.to(device)
        landscape = inverse_transform(inputImage).permute(1, 2, 0).cpu().detach().numpy()
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Image")

        # groundtruth label image
        label_class = gt.cpu().detach().numpy()
        axes[i, 1].imshow(id_to_color[label_class])
        axes[i, 1].set_title("Groundtruth Label")

        # predicted label image
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
        label_class_predicted = y_pred.cpu().detach().numpy()    
        axes[i, 2].imshow(id_to_color[label_class_predicted])
        axes[i, 2].set_title("Predicted Label")

    plt.show()

