# -*- coding: utf-8 -*
# https://github.com/deepinsight/insightface
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import mxnet as mx
import torch
from torchvision import transforms
from torch.utils import data
import torch.distributed as dist 
from torch.utils.data import Dataset
import subprocess
from datetime import datetime
import numbers
import os
import pickle
import csv
import math
import time
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torch.nn.utils import clip_grad_norm_

from backbones import get_model
from all_loss import * 
from metrics import *
from utils_insightface import CallBackLogging, get_lastest_model, get_allMetrics_allRace, load_bin, load_obj
from dataset import DataLoaderX, MXFaceDataset_Race

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):
    if args.lr_scheduler == 'cosine':
        T_total = args.max_epoch * nBatch
        T_cur = (epoch % args.max_epoch) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif args.lr_scheduler == 'multistep':
        lr = args.lr * (0.1 ** (epoch // args.lr_step))
    elif args.lr_scheduler == 'multistep_given': 
        if epoch < args.decay_step[0]:
           lr = args.lr
        elif epoch >= args.decay_step[0] and epoch < args.decay_step[1]:
           lr = args.lr * 0.1
        elif epoch >= args.decay_step[1] and epoch < args.decay_step[2]:
           lr = args.lr * 0.1 * 0.1
        else:
           lr = args.lr * 0.1 * 0.1 * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_train_loader(data_dir, imgidx, all_names_select, args):
    train_set = MXFaceDataset_Race(data_dir, local_rank=args.local_rank, imgidx = imgidx, all_names_select = all_names_select) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    trainloader = DataLoaderX(local_rank=args.local_rank, dataset=train_set, batch_size=args.train_batch_size, 
                            sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return trainloader
    
def train(trainloader, model, metric_fc, criterion, optimizer, epoch, callback_logging, args):
    start = time.time()
    epoch_loss = AverageMeter() 
    epoch_acc  = AverageMeter()
    batch_time = AverageMeter()
    epoch_loss_list = []
    epoch_acc_list = []

    model.train()
    end = time.time()
    lr_list = [] 
    for i, (input, target, originalLabel) in enumerate(trainloader):
        lr = adjust_learning_rate(optimizer, epoch=epoch, args = args, batch=i, nBatch=len(trainloader))
        lr_list = lr_list + [lr]
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        feature = model(input) 
        if args.metric == 'add_margin' or args.metric == 'arc_margin':
            output = metric_fc(feature, target)
        else:
            print('Specified metric not implemented')
            exit(0)

        loss,_ = criterion(output, target)

        epoch_loss.update(loss.item(), input.size(0))
        prec1 = accuracy(output, target)
        epoch_acc.update(prec1[0].item(), input.size(0)) 
    
        # compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm: 
            clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2) 
        optimizer.step()

        batch_time.update(time.time()-end)
        end = time.time()

        total_step = args.max_epoch * len(trainloader)
        global_step = epoch * len(trainloader) + i
        callback_logging(global_step, total_step, epoch_loss.avg, epoch_acc.avg, epoch, lr)

        epoch_loss_list.append(epoch_loss.val)
        epoch_acc_list.append(epoch_acc.val)
    return epoch_loss_list, epoch_acc_list,lr_list


def test(testloader, model, metric_fc, criterion, epoch, args):
    epoch_loss = AverageMeter() 
    epoch_acc  = AverageMeter()
    epoch_loss_list = []
    epoch_acc_list = []
    
    model.eval()
    with torch.no_grad():
        for i, (input, target, originalLabel) in enumerate(testloader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            feature = model(input)
            if args.metric == 'add_margin' or args.metric == 'arc_margin':
                output = metric_fc(feature, target)
            else:
                print('Specified metric not implemented')
                exit(0)
            
            loss,_ = criterion(output, target)

            prec1 = accuracy(output, target)
            epoch_loss.update(loss.item(), input.size(0))
            epoch_acc.update(prec1[0].item(), input.size(0)) 
            epoch_loss_list.append(epoch_loss.val)
            epoch_acc_list.append(epoch_acc.val)
        
    epoch_loss_avg = np.mean(epoch_loss_list)
    epoch_acc_avg = np.mean(epoch_acc_list)
    print('Test: epoch={}, test loss={:.4f}, test_acc={:.4f}'.format(epoch, epoch_loss_avg, epoch_acc_avg))

    return epoch_loss_avg, epoch_acc_avg


if __name__ == '__main__':
    import time
    import argparse
    from argsfile import args
    
    # multi-GPUs
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:{}".format(args.port),rank=0,world_size=1) 

    # GPU random seed
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # save directory
    ckpt_dir = 'checkpoints'
    if args.checkpoints_dir == 'checkpoints': 
        if args.metric == 'arc_margin':
            m = args.arc_m
        elif args.metric == 'add_margin':
            m = args.add_m
        args.checkpoints_dir = '../{}/{}_lr={}_lrS={}_metric={}_m={}'.format(
            ckpt_dir,  args.RFW_race, args.lr, args.lr_scheduler, args.metric, m)
    else: # providedtest_race
        pass
    print('checkpoints_dir', args.checkpoints_dir)
    
    device = torch.device("cuda")
    
    # train datasets
    race_meta = load_obj('BUPT-Equalizedface-{}'.format(args.RFW_race), 'meta')
    train_ids = race_meta['train_imgids']     
    all_names_select = race_meta['train_ids'] 
    trainloader = get_train_loader(args.data_dir, train_ids, all_names_select, args)

    # test datasets 
    data_name_list = ['Caucasian', 'African', 'Asian', 'Indian']
    Caucasian_bin = load_bin(os.path.join(args.test_bin_dir, 'Caucasian_test.bin'), [112,112])
    African_bin = load_bin(os.path.join(args.test_bin_dir, 'African_test.bin'), [112,112])
    Asian_bin = load_bin(os.path.join(args.test_bin_dir, 'Asian_test.bin'), [112,112])
    Indian_bin = load_bin(os.path.join(args.test_bin_dir, 'Indian_test.bin'), [112,112])
    test_bin_list = [Caucasian_bin, African_bin, Asian_bin, Indian_bin]

    # backbone --> metric_fc-->loss
    model = get_model(args.backbone, dropout=0.0, num_features=args.embedding_size, fp16 = args.fp16)
    
    # margin 
    if args.metric == 'add_margin': # cosface
        metric_fc = AddMarginProduct(args.embedding_size, args.num_classes, s=args.s, m=args.add_m) 
    elif args.metric == 'arc_margin': # arcface 
        metric_fc = ArcMarginProduct(args.embedding_size, args.num_classes, s=args.s, m=args.arc_m) 
    
    # loss
    criterion = CELoss()
    criterion = criterion.cuda()
    
    # insightface-changed learning rate
    if args.optimizer == 'sgd': 
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay)
    
    print(model)
    print(metric_fc)
    model.to(device)
    metric_fc.to(device)


    if args.resume or args.test_verification:
       lastest_model_name = get_lastest_model(args.checkpoints_dir)     
       previous_epoch = int(lastest_model_name.split('.')[0].split('_')[1])
       pretrained_model_path = os.path.join(args.checkpoints_dir, lastest_model_name)
       print('=====> loading from pretrained model path', pretrained_model_path)
       checkpoint = torch.load(pretrained_model_path)
       start_epoch = previous_epoch + 1 
       model.load_state_dict(checkpoint['model'])
       metric_fc.load_state_dict(checkpoint['metric_fc']) 
       optimizer.load_state_dict(checkpoint['optimizer'])
       print('current start_epoch={}'.format(start_epoch))
       for param_group in optimizer.param_groups:
           print('current learning rate', param_group['lr'])
       print('***************** Model being loaded')
    else:
        start_epoch = 0
        args.checkpoints_dir = os.path.join(args.working_dir, args.checkpoints_dir)
        if not os.path.exists(args.checkpoints_dir): # create ckpt dir if not exists
            os.makedirs(args.checkpoints_dir)
    
    # logging
    callback_logging = CallBackLogging(
        frequent=args.print_freq,
        total_step=args.num_image // args.train_batch_size * args.max_epoch,
        batch_size=args.train_batch_size,
        start_step = start_epoch * len(trainloader))

    # test only
    if args.test_verification:
        get_allMetrics_allRace(test_bin_list, data_name_list, model, args, epoch=start_epoch, save_name='allMetrics_test.csv')
        exit(0)
    
    for epoch in range(start_epoch, args.max_epoch):
        print('**** Training epoch={}'.format(epoch))
        epoch_loss_list, epoch_acc_list, lr_list = train(trainloader, model, metric_fc, criterion, optimizer, epoch, callback_logging, args)        
        
        # save model 
        if epoch % args.save_interval == 0 or epoch == args.max_epoch:
            state = {'epoch': epoch,
                    'model': model.state_dict(), 
                    'metric_fc':metric_fc.state_dict(),
                    'optimizer': optimizer.state_dict()}
            ckpt_save_name = os.path.join(args.checkpoints_dir, args.backbone + '_' + str(epoch) + '.pth') 
            torch.save(state, ckpt_save_name)  

        # save result for every epoch
        get_allMetrics_allRace(test_bin_list, data_name_list, model, args, epoch, save_name='{}.csv'.format(args.checkpoints_dir.split('/')[-1])) 
        
    torch.cuda.empty_cache()

    
