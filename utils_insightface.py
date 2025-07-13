# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import gc
import os
import csv
import scipy
import numpy as np
import pickle
import argparse
import sys
from scipy import misc
from numpy import linalg as line
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
import torch
import time 

from argsfile import args
from all_loss import * 
from metrics import * 
from backbones import get_model 


torch.manual_seed(args.manual_seed)               
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manual_seed)  
    torch.backends.cudnn.deterministic = True     
    torch.backends.cudnn.benchmark = False        

def load_obj(name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'rb') as f:
        return pickle.load(f)

import logging
logging.basicConfig(level=logging.INFO)
class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size

        self.init = False
        self.tic = 0
    def __call__(self,
                 global_step: int,
                 total_step: int,
                 loss:float,
                 acc:float,
                 epoch: int,
                 learning_rate: float):
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed = float('inf')
                    
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                time_str = time.asctime(time.localtime(time.time()))
                msg = "Time: {}\t Speed {:.2f} samples/sec   Loss {:.4f}  Acc {:.4f}  LearningRate {:.6f}   Epoch: {} Global Step/Total Step: {}/{} Required: {:.1f} hours".format(
                       time_str, speed, loss, acc, learning_rate, epoch, global_step, total_step, time_for_end)
                logging.info(msg)
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

def get_lastest_model(model_dir): 
    file_names = [path for path in os.listdir(model_dir) if 'pth' in path]
    sorted_files = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
    lastest_file = sorted_files[-1]
    return sorted_files[-1] 

def load_lastest_model(args):
    model = get_model(args.backbone, dropout=0.0, num_features=args.embedding_size, fp16 = False)
    device = torch.device("cuda")
    model.to(device)
    epoch_name = get_lastest_model(args.checkpoints_dir)
    pretrained_model_path = os.path.join(args.checkpoints_dir, epoch_name)
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model'])
    return model 

### 数据集已经按照 10-fold进行了划分,不需要再进行Shuffle
class LFold:
  def __init__(self, n_splits = 2, shuffle = False):
    self.n_splits = n_splits
    if self.n_splits>1:
      self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

  def split(self, indices):
    if self.n_splits>1:
      return self.k_fold.split(indices)
    else:
      return [(indices, indices)]

@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes') 
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data) 
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.imresize(img, image_size[0], image_size[1])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print('data_list[0].shape', data_list[0].shape)
    print('data_list[1].shape', data_list[1].shape)
    print('issame_list len', len(issame_list))
    return data_list, issame_list

def calculate_fnr_far_tar_acc(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    actual_issame = np.greater(actual_issame, 0.5) 
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    fnr = 0 if (fn+tp==0) else float(fn) / float(fn+tp) # false negative rate
    far = 0 if (fp+tn==0) else float(fp) / float(fp+tn) # false acceptance rate, false positive rate
    tar = 0 if (fn+tp==0) else float(tp) / float(fn+tp) # true acceptance rate, TAR = TP/(FN+TP), true positive rate
    acc = float(tp+tn)/dist.size
    return fnr, far, tar, acc

### 通过 10-fold cross validation 计算
def calculate_acc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False) # 作者已经将数据集的validation分好了

    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)


    veclist = np.concatenate((embeddings1,embeddings2),axis=0) 
    meana = np.mean(veclist,axis=0)
    embeddings1 -= meana
    embeddings2 -= meana
    dist = np.sum(embeddings1 * embeddings2, axis=1)
    dist = dist / line.norm(embeddings1,axis=1) / line.norm(embeddings2,axis=1)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds): 
            _, _, _, acc_train[threshold_idx] = calculate_fnr_far_tar_acc(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train) # 在每个fold中选一个best-threshold, 然后在validation-set上计算 ACCURACY
        _, _, _, accuracy[fold_idx] = calculate_fnr_far_tar_acc(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
        
    acc = np.mean(accuracy) 
    return acc 

### 用插值法计算给定 FAR 下的 TAR值
def calculate_far_tar(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    nrof_far_target = len(far_target)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False) 
    tar = np.zeros((nrof_far_target, nrof_folds))
    far = np.zeros((nrof_far_target, nrof_folds))

    veclist = np.concatenate((embeddings1,embeddings2),axis=0)
    mean_ = np.mean(veclist,axis=0)
    embeddings1 -= mean_
    embeddings2 -= mean_
    dist = np.sum(embeddings1 * embeddings2, axis=1)
    dist = dist / line.norm(embeddings1,axis=1) / line.norm(embeddings2,axis=1) 

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        tar_train = np.zeros(nrof_thresholds)
        far_train = np.zeros(nrof_thresholds)

        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx], tar_train[threshold_idx], _,  = calculate_fnr_far_tar_acc(threshold, dist[train_set], actual_issame[train_set])
            for i, far_tmp in enumerate(far_target):
                if np.max(far_train)>=far_tmp:
                    f = interpolate.interp1d(far_train, thresholds, kind='slinear') 
                    try:
                       threshold = f(far_tmp)
                    except:
                       threshold = 0.0
                else:
                    threshold = 0.0

                _, far[i, fold_idx], tar[i, fold_idx], _ = calculate_fnr_far_tar_acc(threshold, dist[test_set], actual_issame[test_set])

    far_mean = np.mean(far, axis=1) 
    tar_mean = np.mean(tar, axis=1) 
    return far_mean, tar_mean 


@torch.no_grad() 
def test(data_set, model, model_name, far_target, batch_size, nfolds=10, data_extra = None, label_shape = None):
    print('testing verification..')
    data_list = data_set[0]   
    issame_list = data_set[1] 
    embeddings_list = []
    if data_extra is not None:
        _data_extra = np.array(data_extra)
    time_consumed = 0.0
    
    if label_shape is None:
        _label = torch.from_numpy(np.ones((batch_size,)))
    else:
        _label = torch.from_numpy(np.ones(label_shape)) 
    
    for i in range( len(data_list) ): 
        data = data_list[i]
        embeddings = None
        ba = 0
        
        while ba<data.shape[0]:
            bb = min(ba+batch_size, data.shape[0])
            count = bb-ba
            _data = data[bb-batch_size:bb,...]
            time0 = datetime.datetime.now()
            _data = ((_data / 255) - 0.5) / 0.5 # 非常重要
            _data = _data.to(torch.device("cuda"))
            net_out = model(_data)
            _embeddings = net_out.data.cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed+=diff.total_seconds()
            
            if embeddings is None:
                embeddings = np.zeros( (data.shape[0], _embeddings.shape[1]) )
            embeddings[ba:bb,:] = _embeddings[(batch_size-count):,:]
            ba = bb

        embeddings_list.append(embeddings) 

    ### insightface official preprocess
    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    embeddings1 = embeddings[0::2] 
    embeddings2 = embeddings[1::2] 

    def evaluate(embeddings, actual_issame, far_target, nrof_folds=10):
        thresholds = np.arange(-1,1,0.001)
        embeddings1 = embeddings[0::2] 
        embeddings2 = embeddings[1::2] 
        
        acc = calculate_acc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds)
        far_mean, tar_mean = calculate_far_tar(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), far_target, nrof_folds=nrof_folds)# [5],[5]
        return acc, far_mean, tar_mean 
       
    acc, far_mean, tar_mean  = evaluate(embeddings, issame_list, far_target, nrof_folds=nfolds)
    print('acc={:.4f}, test dataset length={}'.format(acc, len(issame_list)))
    
    return acc, far_mean, tar_mean 

@torch.no_grad()
def get_allMetrics_allRace(data_list, data_name_list, model, args, epoch, save_name):
    model.eval()
    model_name = args.checkpoints_dir.split('/')[-1]   
    acc_all = {}  
    fars_all = {} 
    tars_all = {} 

    for name, data in zip(data_name_list, data_list):
        print('Now processing Dataset={}'.format(name))
        acc, far_mean, tar_mean  = test(data, model, model_name, args.far_target, args.test_batch_size, args.nfolds)
        print('Data={}, acc={:.4f}'.format(name, acc))

        acc_all[name] = acc
        fars_all[name] = far_mean
        tars_all[name] = tar_mean

    result = [acc_all['Caucasian'], acc_all['African'], acc_all['Asian'], acc_all['Indian'],
            tars_all['Caucasian'][0], tars_all['African'][0], tars_all['Asian'][0], tars_all['Indian'][0],
            tars_all['Caucasian'][1], tars_all['African'][1], tars_all['Asian'][1], tars_all['Indian'][1],
            tars_all['Caucasian'][2], tars_all['African'][2], tars_all['Asian'][2], tars_all['Indian'][2]]
    result = [round(x,4) for x in result]
    result = [model_name, epoch] + result
    columnName = ['model', 'epoch', 'Cau-ACC','Afr-ACC','Asi-ACC','Ind-ACC',
                  'Cau-TAR-0.1','Afr-TAR-0.1','Asi-TAR-0.1','Ind-TAR-0.1',
                  'Cau-TAR-0.01','Afr-TAR-0.01','Asi-TAR-0.01','Ind-TAR-0.01',
                  'Cau-TAR-0.001','Afr-TAR-0.001','Asi-TAR-0.001','Ind-TAR-0.001']

    if not os.path.exists(save_name):
        with open(save_name, 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(columnName)

    with open(save_name, 'a') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(result)
    

if __name__ == '__main__':
    import time
    stime = time.time()

    args.checkpoints_dir = '../checkpoints/Caucasian_lr=0.1_lrS=multistep_given_metric=arc_margin_m=0.5'
    model_name = args.checkpoints_dir.split('/')[-1]
    save_name = args.checkpoints_dir.split('/')[-1]

    model = get_model(args.backbone, dropout=0.0, num_features=args.embedding_size, fp16 = False)
    device = torch.device("cuda")
    model.to(device)
    epoch_name = get_lastest_model(args.checkpoints_dir)
    pretrained_model_path = os.path.join(args.checkpoints_dir, epoch_name)
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model'])

    epoch = int(epoch_name.split('.')[0].split('_')[1])

    data_name_list = ['Caucasian', 'African', 'Asian', 'Indian']
    Caucasian_bin = load_bin(os.path.join(args.test_bin_dir, 'Caucasian_test.bin'), [112,112])
    African_bin = load_bin(os.path.join(args.test_bin_dir, 'African_test.bin'), [112,112])
    Asian_bin = load_bin(os.path.join(args.test_bin_dir, 'Asian_test.bin'), [112,112])
    Indian_bin = load_bin(os.path.join(args.test_bin_dir, 'Indian_test.bin'), [112,112])
    test_bin_list = [Caucasian_bin, African_bin, Asian_bin, Indian_bin]

    get_allMetrics_allRace(test_bin_list, data_name_list, model, args, epoch, save_name='result.csv')

    etime = time.time()
    print('total time={}'.format((etime-stime)/60))

