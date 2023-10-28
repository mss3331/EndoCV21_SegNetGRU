from __future__ import print_function
from __future__ import division
import torch
from helpers.metrics import calculate_metrics_torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import time


def train_model(model, dataloaders, criterion, optimizer,device, num_epochs,input_size):
    since = time.time()

    results_dic= {"val_acc_history":[],"train_acc_history":[],
              "val_loss_history":[],"train_loss_history":[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_iou = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_true_maskes_torch = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels[:,1,:,:].data)

                # store all the true and pred masks
                if len(all_true_maskes_torch) == 0:
                    all_true_maskes_torch = labels.clone().detach()
                    all_pred_maskes_torch = outputs.clone().detach()
                else:
                    all_true_maskes_torch = torch.cat((all_true_maskes_torch, labels.clone().detach()))
                    all_pred_maskes_torch = torch.cat((all_pred_maskes_torch, outputs.clone().detach()))

            total_mask_pixels = labels.size(2) * labels.size(3)  # H * W
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * total_mask_pixels)

            iou = calculate_metrics_torch(true=all_true_maskes_torch, pred=all_pred_maskes_torch,
                                          reduction='mean',metrics='jaccard',cloned_detached=True)

            print('{} IoU: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(phase,iou, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and iou > best_iou:
                best_iou = iou
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            update_results(phase,results_dic,epoch_acc,epoch_loss)
            if phase == 'val':
                print('Best So far {} iou: {:.4f}'.format(phase, best_iou))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val iou:{:4f} and Acc: {:4f}'.format(best_iou, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint = {
        'accuracy': best_acc,
        'iou': best_iou,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'image_size':input_size
    }
    torch.save(checkpoint,'./checkpoints/SegNetGRU.pt')
    return model, results_dic

def update_results(phase,results_dic,epoch_acc,epoch_loss):
    results_dic[phase+"_acc_history"].append(epoch_acc)
    results_dic[phase+"_loss_history"].append(epoch_loss)


