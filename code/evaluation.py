# -*-coding:utf-8-*-

import os

from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
from get_dataloader import get_dataloader
from modeling import deeplab
from dataloader import dataloaderIDA,dataloaderI,dataloaderIDAN,dataloaderDA,dataloladerN,dataloaderDAN
from modeling.PS_FCN.PS_FCN import PS_FCN
from modeling.smqFusion.smqFusion import smqFusion
import loss_functions
import API.utils
import numpy as np
import random
from evaluation import evaluation
from tensorboardX import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

parser = argparse.ArgumentParser(description='TransSfP')
parser.add_argument('-dataset_dir', default='/media/disk/dataset/TransSfP', help='path for TransSfP dataset')
parser.add_argument('-code_dir', default='/media/disk/code/TransSfP', help='path for TransSfP codes')
parser.add_argument('-batch_size', default='6', help='batch size')
parser.add_argument('-checkpoint', default='', help='path for checkpoint')

###################### DataLoader #############################

#-- 1、 config parameters


imgHeight = 512
imgWidth = 512
batch_size = 6
num_workers = 2
validation_split = 0.1
shuffle_dataset = True
pin_memory = False
prefetch_factor = 8

    #-- 2、create dataset
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({"height": imgHeight, "width": imgWidth }, interpolation='nearest'),  # Resize image
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),
    # Blur and Noise
    # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),x`x
    # iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    # iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),

    # Color, Contrast, etc.
    # iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    # iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    # iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    # iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
])
trainLoader,testLoader_tiny_white_cup,testLoader_tiny_white_cup_edges,testLoader_bird_back,testLoader_cat_back,testLoader_cat_front,testLoader_hemi_sphere_big,testLoader_hemi_sphere_small = get_dataloader(root_dir,augs_train,batch_size,num_workers,pin_memory)


###################### ModelBuilder #############################

#-- 1、 config parameters
backbone_model = 'resnet50'
sync_bn = False  # this is for Multi-GPU synchronize batch normalization
numClasses = 3
use_atten = False

#-- 2、create model


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smqFusion(backbone = backbone_model,num_classes=3,device=device,sync_bn=False)
#-- 3、Enable GPU for training

import os
# device = torch.device("cpu")
# model = model.to(device)
model = model.to(device)


###################### Setup Optimizer #############################


#-- 1、 config parameters
learningRate = 1e-6
weightDecay = 5e-4
momentum = 0.9
# lrSchedulerStep
lrScheduler = 'StepLR'
step_size = 7
gamma = 0.1
# lrSchedulerPlateau:
factor: 0.8
patience: 25
verbose: True
# loss func


#-- 2、create optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=float(learningRate),
                            momentum=float(momentum),
                            weight_decay=float(weightDecay))

#-- 3、create learningRate schduler
if not lrScheduler:
    pass
elif lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=float(gamma))
elif lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=float(factor),
                                                              patience=patience,
                                                              verbose=verbose)
elif lrScheduler == 'lr_poly':
    pass
else:
    raise ValueError(
        "Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(
            lrScheduler))

#-- 4、select koss fu
criterion = loss_functions.my_loss_cosine
writer = SummaryWriter()


###################### Train Model #############################
#-- 1、config parameters
MAX_EPOCH = 20
saveModelInterval = 1
CHECKPOINT_DIR = '/home/robotlab/smq/SurfaceNormals/CheckPoints'
total_iter_num = 0
START_EPOCH = 0
continue_train = True
preCheckPoint = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0018.pth')

#-- 2、load check point
if(continue_train):
    print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
    if not os.path.isfile(preCheckPoint):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(preCheckPoint))
    CHECKPOINT = torch.load(preCheckPoint, map_location='cpu')
    if 'model_state_dict' in CHECKPOINT:
        # Our weights file with various dicts
        model.load_state_dict(CHECKPOINT['model_state_dict'])
    elif 'state_dict' in CHECKPOINT:
        # Original Author's checkpoint
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
        model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
    else:
        # Our old checkpoint containing only model's state_dict()
        model.load_state_dict(CHECKPOINT)

    if continue_train and preCheckPoint:
        if 'optimizer_state_dict' in CHECKPOINT:
            optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
        else:
            print(
                colored(
                    'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                    '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))
    if continue_train and preCheckPoint:
        if 'model_state_dict' in CHECKPOINT:
            # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
            total_iter_num = CHECKPOINT['total_iter_num'] + 1
            START_EPOCH = CHECKPOINT['epoch'] + 1
            END_EPOCH = CHECKPOINT['epoch'] + MAX_EPOCH
        else:
            print(
                colored(
                    'Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                           Starting from epoch num 0', 'red'))
#-- 3、epoch cycle
import time
mean_list = []
median_list = []
for epoch in range(0,1):
    print('\n\nEpoch {}/{}'.format(epoch, MAX_EPOCH - 1))
    print('-' * 30)


    ###################### Validation Cycle #############################
    mean_all = 0
    median_all = 0
    acc_all_1 = 0
    acc_all_2 = 0
    acc_all_3 = 0
    count = 0
    print('\nValidation:')
    print('=' * 10)
    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_tiny_white_cup_edges,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'tiny_white_cup_edges',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/tiny-white-cup-edges')
    print('tiny-white-cup-edges:\n')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3

    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_tiny_white_cup,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'tiny_white_cup',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/tiny-white-cup')
    print('tiny-white-cup:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    count += 1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_bird_front,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'bird_front',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/bird-front')
    print('bird-front:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_bird_back,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'bird_back',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/bird-back')
    print('bird-back:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_cat_front,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'cat_front',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/cat-front')
    print('cat-front:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_cat_back,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'cat_back',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/cat-back')
    print('cat-back:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_hemi_sphere_big,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'hemi_sphere_big',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/hemi-sphere-big')
    print('hemi-sphere-big:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_hemi_sphere_small,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'hemi_sphere_small',writer=writer,resultPath='/home/robotlab/smq/SurfaceNormals/test/results/hemi-sphere-small')
    print('hemi-sphere-small:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    print('all mean: ',mean_all/count)
    print('all median: ',median_all/count)
    print('percentage 1: ',acc_all_1/count)
    print('percentage 2: ',acc_all_2/count)
    print('percentage 3: ',acc_all_3/count)

    mean_list.append(mean_all/count)
    median_list.append(median_all/count)

