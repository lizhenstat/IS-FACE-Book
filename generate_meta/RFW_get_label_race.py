# -*- coding: utf-8 -*
# 构建并保存一个标签（label）到种族（race）的映射字典 label_race_dict，用于后续的人脸认证任务中按种族划分样本。
# BUPT-Equalizedface-label-race-dict.pkl

import os
import pickle
import numpy as np
def load_obj(name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

### 查看下 png 的文件夹的图片类别个数
RFW_dir = '/home/data/siriusShare/arcface-pytorch-master/Datasets/RFW/images/RFW_images_train'
Caucasian_dir = '/home/data/siriusShare/arcface-pytorch-master/Datasets/RFW/images/Caucasian'
African_dir = '/home/data/siriusShare/arcface-pytorch-master/Datasets/RFW/images/African'
Indian_dir = '/home/data/siriusShare/arcface-pytorch-master/Datasets/RFW/images/Indian'
Asian_dir = '/home/data/siriusShare/arcface-pytorch-master/Datasets/RFW/images/Asian'

len(os.listdir(RFW_dir))       # 28004 # 这里包括4个 tar.gz 文件
len(os.listdir(Caucasian_dir)) # 7000
len(os.listdir(African_dir))   # 7000 
len(os.listdir(Indian_dir))    # 7000
len(os.listdir(Asian_dir))     # 7000

### 检查缺少
name_label_path = '/home/sata/Dataset/BUPT/BUPT-Balancedface/rec_for_mxnet/train_balancedface.lst' # 1251416
RFW_images_dir = '/home/data/siriusShare/arcface-pytorch-master/Datasets/RFW/images'
race_names = ['African', 'Asian', 'Caucasian', 'Indian']

with open(name_label_path) as f: # strip last \n
     lines = [line.rstrip('\n') for line in f] # 1251416

# check total number of classes
all_classes = [d.split('/')[0] for d in lines] # classes 是用文件夹名称表示的
print(len(list(set(all_classes)))) # 27999


### class_name --> label
className_label_dict = {} # 27999
for d in lines:
    tmp = d.split('\t')
    className_label_dict[tmp[0].split('/')[0]] = int(tmp[1])
label_className_dict = dict((v,k) for k,v in className_label_dict.items()) # 27999


### class_name --> race
className_race_dict = {}
for race in race_names:  # race = 'African'
    race_names = os.listdir(os.path.join(RFW_images_dir, race)) ### 这里用的是png的文件夹
    for name in race_names:
        className_race_dict[name] = race

### label --> race
label_race_dict = {}
all_class_names = className_label_dict.keys()
for name in all_class_names:
    label_race_dict[className_label_dict[name]] = className_race_dict[name]


### python save and load object
save_obj(label_race_dict, 'BUPT-Equalizedface-label-race-dict', '../meta')
