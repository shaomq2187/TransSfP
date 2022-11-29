# -*-coding:utf-8-*-

import os

from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
from get_dataloader import get_dataloader
from modeling.TransSfPNet.TransSfPNet import TransSfPNet
import loss_functions
import API.utils
import numpy as np
import random
from tensorboardX import SummaryWriter
import argparse
def evaluation(model,testLoader,device,criterion,epoch,resultPath = None,name=None,writer = None):
    ###################### Validation Cycle #############################


    model.eval() # eval mode, freeze params
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    running_percentage_1 = 0
    running_percentage_2 = 0
    running_percentage_3 = 0
    mean_list = []
    for iter_num, sample_batched in enumerate(tqdm(testLoader)):
        # print('')
        params_t,normals_t, label_t,mask_t = sample_batched
        params_t = params_t.to(device)
        normals_t = normals_t.to(device)
        label_t = label_t.to(device)
        aolp = params_t[:,1,:,:]


        with torch.no_grad():
            normal_vectors,confidence_map = model(params_t,normals_t)

        normal_vectors_norm= nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        normal_vectors_norm = normal_vectors_norm
        # loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum',device=device)
        loss = criterion(normal_vectors_norm, label_t.double(),confidence_map,aolp,mask_tensor = mask_t,reduction='sum',device = device)

        # calcute metrics
        label_t = label_t.detach().cpu()
        normal_vectors_norm = normal_vectors_norm.detach().cpu()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, label_t.double())
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()
        mean_list.append(loss_deg_mean.item())
        running_loss += loss.item()
        running_percentage_1 += percentage_1.item()
        running_percentage_2 += percentage_2.item()
        running_percentage_3 += percentage_3.item()

        # save validation pictures

        label_t_rgb = label_t.numpy().squeeze(0).transpose(1, 2, 0)
        label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
        predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1, 2, 0)
        mask_t = mask_t.squeeze(1)
        predict_norm[mask_t.squeeze(0) == 0, :] = -1
        predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
        confidence_map_rgb = confidence_map.detach().cpu().numpy().squeeze(0).transpose(1,2,0)
        confidence_map_rgb = confidence_map_rgb*255
        confidence_map_rgb = confidence_map_rgb.astype(np.uint8)
        if not os.path.exists(resultPath):
            os.mkdir(resultPath)
        API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-label.png'),
                            label_t_rgb)
        API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-predict.png'),
                            predict_norm_rgb)
        API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-atten.png'),
                            confidence_map_rgb)

    assert testLoader.batch_size == 1, 'testLoader batch size is need to be 1 instead of : "%d"' % (testLoader.batch_size)

    numsamples = len(testLoader)
    running_loss = running_loss/numsamples
    running_mean = running_mean/numsamples
    running_median = running_median/numsamples
    running_percentage_1 = running_percentage_1/numsamples
    running_percentage_2 = running_percentage_2/numsamples
    running_percentage_3 = running_percentage_3/numsamples
    if(writer is not None):
        writer.add_scalar(name+'/'+'running_mean',running_mean,epoch)
        writer.add_scalar(name+'/'+'running_median',running_median,epoch)
        writer.add_scalar(name+'/'+'running_loss',running_loss,epoch)
        writer.add_scalar(name+'/'+'running_percentage_1',running_percentage_1,epoch)
        writer.add_scalar(name+'/'+'running_percentage_2',running_percentage_2,epoch)
        writer.add_scalar(name+'/'+'running_percentage_3',running_percentage_3,epoch)
    print('mean list:',mean_list)
    return running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='TransSfP')
    parser.add_argument('-dataset_dir', default='/media/disk/dataset/TransSfP', help='path for TransSfP dataset')
    parser.add_argument('-code_dir', default='/media/disk/code/TransSfP', help='path for TransSfP codes')
    parser.add_argument('-batch_size', default='5', help='batch size')

    parsed = parser.parse_args()
    root_dir = parsed.dataset_dir
    code_root_dir = parsed.code_dir
    batch_size = int(parsed.batch_size)


    setup_seed(20)

    ###################### DataLoader #############################

    #-- 1、 config parameters


    imgHeight = 512
    imgWidth = 512
    batch_size = 5
    num_workers = 2
    validation_split = 0.1
    shuffle_dataset = True
    pin_memory = False
    prefetch_factor = 8

    #-- 2、create dataloaders
    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize({"height": imgHeight, "width": imgWidth }, interpolation='nearest'),  # Resize image
    ])

    ######## get dataloaders ########
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
    model = TransSfPNet(backbone = backbone_model,num_classes=3,device=device,sync_bn=False)
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
    criterion = loss_functions.TransSfP_loss
    writer = SummaryWriter()


    ###################### Train Model #############################
    #-- 1、config parameters
    MAX_EPOCH = 20
    saveModelInterval = 1
    CHECKPOINT_DIR = code_root_dir + '/CheckPoints'
    total_iter_num = 0
    START_EPOCH = 0
    continue_train = False
    preCheckPoint = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0000.pth')

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    if not os.path.exists(code_root_dir + '/results'):
        os.mkdir(code_root_dir + '/results')

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
    for epoch in range(START_EPOCH,MAX_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, MAX_EPOCH - 1))
        print('-' * 30)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 10)
        model.train()  # set model mode to train mode

        running_loss = 0.0
        running_mean = 0
        running_median = 0
        for iter_num,batch  in enumerate(tqdm(trainLoader)):
            total_iter_num+=1
            params_t,normals_t, label_t,mask_t = batch
            params_t = params_t.to(device)
            aolp = params_t[:,1,:,:]
            normals_t = normals_t.to(device)
            label_t = label_t.to(device)
            # Forward + Backward Prop
            start = time.time()
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            with torch.autograd.set_detect_anomaly(True):
                normal_vectors,confidence_map = model(params_t,normals_t)
                normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
                normal_vectors_norm = normal_vectors_norm
                loss = criterion(normal_vectors_norm, label_t.double(),confidence_map,aolp,mask_tensor=mask_t,reduction='sum',device=device)
            loss /= batch_size
            loss.backward()
            optimizer.step()
            # print('time consume:',time.time()-start)

            # calcute metrics
            label_t = label_t.detach().cpu()
            normal_vectors_norm = normal_vectors_norm.detach().cpu()

            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
                normal_vectors_norm.detach().cpu(), label_t.double())
            running_mean += loss_deg_mean.item()
            running_median += loss_deg_median.item()
            running_loss += loss.item()

            #  output train set
            if(epoch % 10==0):
                label_t_rgb = label_t.numpy()[0,:,:,:].transpose(1, 2, 0)
                label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
                predict_norm = normal_vectors_norm.numpy()[0,:,:,:].transpose(1, 2, 0)
                mask_t = mask_t.squeeze(1)
                predict_norm[mask_t[0,:,:] == 0, :] = -1
                predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
                confidence_map = confidence_map[0,:,:,:]
                confidence_map_rgb = confidence_map.detach().cpu().numpy().transpose(1, 2, 0)

                confidence_map_rgb = confidence_map_rgb * 255
                confidence_map_rgb = confidence_map_rgb.astype(np.uint8)
                if not os.path.exists(code_root_dir + '/results/train'):
                    os.mkdir(code_root_dir + '/results/train')
                API.utils.png_saver(
                    os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-label.png'),
                    label_t_rgb)
                API.utils.png_saver(
                    os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-predict.png'),
                    predict_norm_rgb)
                API.utils.png_saver(
                    os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-atten.png'),
                    confidence_map_rgb)


            # print('loss_deg_mean:',loss_deg_mean)
            # print('loss_deg_median:',loss_deg_median)
        num_samples = (len(trainLoader))
        epoch_loss = running_loss/num_samples
        print("train running loss:",epoch_loss)
        print("train running mean:",running_mean/num_samples)
        print("train running median:",running_median/num_samples)


        # save the model check point every N epoch
        if epoch % saveModelInterval==0:
            filename = os.path.join(CHECKPOINT_DIR,'check-point-epoch-{:04d}.pth'.format(epoch))
            model_params = model.state_dict()
            torch.save(
                {
                    'model_state_dict': model_params,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'total_iter_num': total_iter_num,
                    'epoch_loss': epoch_loss,
                }, filename)



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
                testLoader= testLoader_tiny_white_cup_edges,device=device,criterion=criterion,epoch = epoch,name = 'tiny_white_cup_edges',writer=writer,resultPath= code_root_dir + '/results/tiny-white-cup-edges')
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
                testLoader= testLoader_tiny_white_cup,device=device,criterion=criterion,epoch = epoch,name = 'tiny_white_cup',writer=writer,resultPath= code_root_dir + '/results/tiny-white-cup')
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
                testLoader= testLoader_bird_back,device=device,criterion=criterion,epoch = epoch,name = 'bird_back',writer=writer,resultPath= code_root_dir + '/results/bird-back')
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
                testLoader= testLoader_cat_front,device=device,criterion=criterion,epoch = epoch,name = 'cat_front',writer=writer,resultPath= code_root_dir + '/results/cat-front')
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
                testLoader= testLoader_cat_back,device=device,criterion=criterion,epoch = epoch,name = 'cat_back',writer=writer,resultPath= code_root_dir + '/results/cat-back')
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
                testLoader= testLoader_hemi_sphere_big,device=device,criterion=criterion,epoch = epoch,name = 'hemi_sphere_big',writer=writer,resultPath= code_root_dir + '/results/hemi-sphere-big')
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
                testLoader= testLoader_hemi_sphere_small,device=device,criterion=criterion,epoch = epoch,name = 'hemi_sphere_small',writer=writer,resultPath= code_root_dir + '/results/hemi-sphere-small')
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
    print('mean_all_list:',mean_list)
    print('median_all_list:',median_list)



