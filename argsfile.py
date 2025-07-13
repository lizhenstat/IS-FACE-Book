import os
import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser(description='pytorch importance sampling for face verification')

# loss
parser.add_argument('--metric', type=str, default='arc_margin', help='metric type')
parser.add_argument('--exp_normalization', action='store_true', help='Exp-normalization trick(insightface torch)')
parser.add_argument('--clip_grad_norm', action='store_true', help='clip grad norm trick(insightface torch)')
# metric
parser.add_argument('--far_target', nargs='*', default=[0.1,0.01,0.001], help='FNR & FAR plot, FAR target value', type=float)
parser.add_argument('--arc_m', type=float, default=0.5, help='hyperparameter for m for arc_margin official implementation')
parser.add_argument('--add_m', type=float, default=0.35, help='hyperparameter for m for add_margin official implementation')
parser.add_argument('--s', type=float, default=64.0, help='feature normalized value') 
# network
parser.add_argument('--embedding_size', type=int, default=512, help='feature embedding size')
parser.add_argument('--backbone', type=str, default='r34', help='backbone network(r18,r50, etc)')
parser.add_argument('--working_dir', type=str, default='', help='root directory')
# dataset
parser.add_argument('--data_root_dir', type=str, help='root directory')
parser.add_argument('--test_dataset', type=str, default='bfw', help='test dataset choose from bfw, lfw, cfp, age, rfw')
# mxnet dataloader 
parser.add_argument('--local_rank', type=int, default=0, help='local rank, insightface arcface_torch')
parser.add_argument('--port', type=int, default=12584, help='address port') 
parser.add_argument('--RFW_race', type=str, default='Caucasian', help='select from Caucasian, African, Asian, Indian')
# optimizer
parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
parser.add_argument('--lr_step', type=int, default=10, help='step learning rate, step')
parser.add_argument('--decay_step', type=str, default='14-20-24', help='decay epoch for multistep learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95, help='step learning rate, decay')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum(RFW paper suggested)')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--lr_scheduler', type=str, default='multistep_given', help='learning rate scheduler')
# train
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--max_epoch', type=int, default=27, help='max training epochs')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size') 
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size') 
parser.add_argument('--num_workers', type=int, default=4, help='how many workers for loading data')
parser.add_argument('--input_shape', type=str, default='1-128-128', help='input image shape(default=(1,128,128))')
parser.add_argument('--save_interval', type=int, default=1, help='save models every n epochs')
parser.add_argument('--test_interval', type=int, default=1, help='test face verification every n epochs')
parser.add_argument('--print_freq', type=int, default=10, help='print every n epochs')
parser.add_argument('--manual_seed', type=int, default=0, help='randomness in GPU training')
parser.add_argument('--train_size', type=float, default=1.0, help='training on subset of webface_clean')
parser.add_argument('--random_state', type=int, default=0, help='choose image from whole training set')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory name, relative to working directory')
parser.add_argument('--resume', action='store_true', help='resume training from pretrained checkpoints')
parser.add_argument('--fp16', action='store_true', help='whether to use fp16 in model')
parser.add_argument('--test_verification', action='store_true', help='test pre-trained model on face verification task')
parser.add_argument('--data_dir', type=str, help='training data directory') 
parser.add_argument('--test_bin_dir', type=str, default='insightface_test_bin') 
# analysis
parser.add_argument('--save_name', type=str, help='save name')
parser.add_argument('--nfolds', type=int, default=10, help='cross validation folds when calculating metrics')
# python
args = parser.parse_args()
args.data_root_dir = '/home/sata/Dataset' 
num_classes_dict = race_names = {'Caucasian':7000,'African':7000,'Asian':7000,'Indian':6999}   
if args.RFW_race == 'Caucasian':
   args.num_image = 326484
if args.RFW_race == 'African':
   args.num_image = 324376
if args.RFW_race == 'Asian':
   args.num_image = 325493
if args.RFW_race == 'Indian':
   args.num_image = 275063
args.num_classes = num_classes_dict[args.RFW_race]
args.data_dir = os.path.join(args.data_root_dir,'Equalizedface')
args.test_bin_dir = os.path.join(args.data_root_dir,'insightface_test_bin')
args.decay_step = list(map(int, args.decay_step.split('-')))
args.input_shape = list(map(int, args.input_shape.split('-'))) 
for arg in vars(args):
    print(arg, getattr(args, arg)) 
