#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021

@author: endocv2021@generalizationChallenge
"""

import checkpoints

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import skimage
from skimage import io
from  tifffile import imsave
from Detection import detect
import skimage.transform
from torchvision import models
from SegNet_GRU import SegNetGRU_Symmetric_columns_UltimateShare as SegNet_GRU_model


def create_predFolder(task_type):
    directoryName = 'EndoCV2021'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
        
    if not os.path.exists(os.path.join(directoryName, task_type)):
        os.mkdir(os.path.join(directoryName, task_type))
        
    return os.path.join(directoryName, task_type)

def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50', 'deeplabv3plus_mobilenet'], help='model name')
    
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    return parser


        
if __name__ == '__main__':
    '''
     You are not allowed to print the images or visualizing the test data according to the rule. 
     We expect all the users to abide by this rule and help us have a fair challenge "EndoCV2021-Generalizability challenge"
     
     FAQs:
         1) Most of my predictions do not have polyp.
            --> This can be the case as this is a generalisation challenge. The dataset is very different and can produce such results. In general, not all samples 
            have polyp.
        2) What format should I save the predictions.
            --> you can save it in the tif or jpg format. 
        3) Can I visualize the data or copy them in my local computer to see?
            --> No, you are not allowed to do this. This is against challenge rules. No test data can be copied or visualised to get insight. Please treat this as unseen image.!!!
        4) Can I use my own test code?
            --> Yes, but please make sure that you follow the rules. Any visulization or copy of test data is against the challenge rules. We make sure that the 
            competition is fair and results are replicative.
    '''
    # model, device = mymodel()
    checkpoint = torch.load('./checkpoints/highest_IOU_SegNetGRU_Symmetric_columns_UltimateShare.pt')
    model = SegNet_GRU_model(input_channels=3, num_classes=2)
    model.load_state_dict(checkpoint['state_dict'])

    task_type = 'segmentation'
    # set image folder here!
    directoryName = create_predFolder(task_type)
    
    # ----> three test folders [https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/wiki/EndoCV2021-Leaderboard-guide]
    subDirs = ['EndoCV_DATA1', 'EndoCV_DATA2', 'EndoCV_DATA3']
    
    for j in range(0, len(subDirs)):
        
        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        imgfolder='/project/def-sponsor00/endocv2021-test-noCopyAllowed-v1/' + subDirs[j]
        # imgfolder = './testfolders/' + subDirs[j]#************************************Delete this one before uploading!!!!!
        # set folder to save your checkpoints here!
        saveDir = os.path.join(directoryName , subDirs[j]+'_pred')
    
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        imgfiles = detect_imgs(imgfolder, ext='.jpg')
    
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        file = open(saveDir + '/'+"timeElaspsed" + subDirs[j] +'.txt', mode='w')
        timeappend = []
    
        for imagePath in imgfiles[:]:
            """plt.imshow(img1[:,:,(2,1,0)])
            Grab the name of the file. 
            """
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            # filename = (imagePath.split('\\')[-1]).split('.jpg')[0]
            print('filename is printing::=====>>', filename)

            #            
            img = skimage.io.imread(imagePath)
            size=img.shape
            start.record()
            
            resize_factor = 4
            train_img_size = (int(1080 / resize_factor), int(1350 / resize_factor))
          

            output = detect(model,imagePath, train_img_size)

            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            timeappend.append(start.elapsed_time(end))
            #
            
            preds = output
            pred = preds*255.0

            
            img_mask=skimage.transform.resize(pred, (size[0], size[1]), anti_aliasing=True) 
            imsave(saveDir +'/'+ filename +'_mask.jpg',(np.round(img_mask/255)*255).astype('uint8'))


            
            file.write('%s -----> %s \n' % 
               (filename, start.elapsed_time(end)))
            
    
            # TODO: write time in a text file
        
        file.write('%s -----> %s \n' % 
           ('average_t', np.mean(timeappend)))
