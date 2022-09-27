"""
Evaluation
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d

def scene_flow_mask_np(pred, labels):

    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)
    num = pred.shape[1]

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.logical_or((error <= 0.1), (error/gtflow_len <= 0.1))
    acc1 = acc1.reshape((-1,1)) #####  n,1
    mask_acc = np.tile(acc1, [1, 3])

    return mask_acc

def outlier_mask_np(pred, labels):

    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)
    #num = pred.shape[1]

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    outlier = np.logical_or((error > 0.3), (error / gtflow_len > 0.1))
    outlier1 = outlier.reshape((-1,1)) #####  n,1
    mask_acc = np.tile(outlier1, [1, 3])

    return mask_acc

def scene_flow_EPE_np(pred, labels):

    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)
    num = pred.shape[1]

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05), (error/gtflow_len <= 0.05)), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1), (error/gtflow_len <= 0.1)), axis=1)
    outlier = np.sum(np.logical_or((error > 0.3), (error / gtflow_len > 0.1)), axis=1)

    acc1 = acc1/num
    acc1 = np.mean(acc1)
    acc2 = acc2/num
    acc2 = np.mean(acc2)
    outlier = outlier/num
    outlier = np.mean(outlier)


    EPE = np.sum(error, 1) / num
    EPE = np.mean(EPE)
    return EPE, acc1, acc2, outlier

def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessDataEval(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model = nn.DataParallel(model)
    model.cuda()

    NUM_POINT = 8192

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, path = data

        #move to cuda 
        pos1111 = pos1[0].numpy()
        pos1 = torch.tensor(pos1[0], dtype=torch.float32).cuda()
        pos2 = torch.tensor(pos2[0], dtype=torch.float32).cuda()
        #pos1 = pos1[0].cuda()
        #pos2 = pos2[0].cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.numpy()
        
        num_batches = min(pos1.shape[0] // NUM_POINT, pos2.shape[0] // NUM_POINT)

        model = model.eval()
        batch_data1 = []
        batch_data2 = []
        batch_label = []
        sampled_pc1 = np.zeros((num_batches * NUM_POINT, 3))
        sampled_pc2 = np.zeros((num_batches * NUM_POINT, 3))
        sampled_pred_pc2 = np.zeros((num_batches * NUM_POINT, 3))
        sampled_mask_acc_pc2 = np.zeros((num_batches * NUM_POINT, 3))
        for batch_idx in range(num_batches):

            start_idx = batch_idx*NUM_POINT
            end_idx = (batch_idx+1)*NUM_POINT
            
            batch_data1 = pos1[start_idx:end_idx, :3]
            batch_data2 = pos2[start_idx:end_idx, :3]###########1*n*3
            batch_label = flow[:,start_idx:end_idx, :3]
            batch_data1 = batch_data1.unsqueeze(0)
            batch_data2 = batch_data2.unsqueeze(0)
            with torch.no_grad(): 
                #print('batch_data1:',batch_data1)
                pred_flows, fps_pc1_idxs, _, _, _ = model(batch_data1, batch_data2, batch_data1, batch_data2)
                '''pred_flows1 = pred_flows[0].permute(0, 2, 1).cpu().numpy()[0]
                pred_pc2 = pos1[0].cpu().numpy() + pred_flows1
                np.save('point3/{}'.format(i),pred_pc2)'''
                #loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

                full_flow = pred_flows[0].permute(0, 2, 1)
                #print('-----full_flow-----',full_flow[0, :, :3])
                #epe3d = torch.norm(full_flow - flow, dim = 2).mean()
            full_flow = full_flow.cpu().numpy()
            batch_data1 = batch_data1[0].cpu().numpy() 
            batch_data2 = batch_data2[0].cpu().numpy()
            
            sampled_pc1[start_idx:end_idx, :3] = pos1111[start_idx:end_idx, :3]
            sampled_pc2[start_idx:end_idx, :3] = pos1111[start_idx:end_idx, :3] + flow[:,start_idx:end_idx, :3]
            sampled_pred_pc2[start_idx:end_idx, :3] = pos1111[start_idx:end_idx, :3] + full_flow[0, :, :3]

            sampled_mask_acc_pc2[start_idx:end_idx, :3]  = scene_flow_mask_np(full_flow, batch_label)
            #sampled_mask_acc_pc2[start_idx:end_idx, :3]  = outlier_mask_np(full_flow, batch_label)
        
        PRED = sampled_pred_pc2 - sampled_pc1
        LABEL = sampled_pc2 - sampled_pc1

        PRED = PRED.reshape(1, -1, 3)
        LABEL = LABEL.reshape(1, -1, 3)

        EPE3D, acc3d_1, acc3d_2, outlier = scene_flow_EPE_np(PRED, LABEL)
        #np.save('points/flow_{}'.format(i),PRED)
        #np.save('points/pc1_{}'.format(i),sampled_pc1)
        #np.save('points/pc2_{}'.format(i),sampled_pc2)
        fname = os.path.join('points/', str(i).zfill(3)+'.npz')
        print('--------------result-----------------------------------')
        print('iiiii: ', i)
        print ('EPE3D:' ,EPE3D)
        print('acc3d_1: ', acc3d_1)
        print('acc3d_2: ', acc3d_2)
        print('outlier: ', outlier)


        np.savez_compressed(fname, pc1 = sampled_pc1, pc2 = sampled_pc2, pc2_pred = sampled_pred_pc2, pc2_acc_mask = sampled_mask_acc_pc2)

    return 0


if __name__ == '__main__':
    main()




