# -*- coding: utf-8 -*
# 为四个种族提取对应训练样本的标签与图像索引，保存为如下四个文件：
# （1）BUPT-Equalizedface-Caucasian.pkl
# （2）BUPT-Equalizedface-African.pkl
# （3）BUPT-Equalizedface-Indian.pkl
# （4）BUPT-Equalizedface-Asian.pkl

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


for RFW_race in ['Caucasian','African','Asian','Indian']:   
    label_imgid_dict = load_obj('BUPT-Equalizedface-label-imgidList-dict','../meta')
    label_race_dict = load_obj('BUPT-Equalizedface-label-race-dict', '../meta')

    race_labels = [] 
    for k,v in label_race_dict.items():
        if v == RFW_race:
           race_labels.append(k)
    race_labels = sorted(race_labels) 

    race_imgid_dict = {}
    for k,v in label_imgid_dict.items(): 
        if k in race_labels  and len(v) > 0: 
           race_imgid_dict[k] = v
    numImages = np.sum([len(v) for k,v in race_imgid_dict.items()])
    print('Current processing race={}, numLabels={}, numImages={}'.format(RFW_race, len(race_labels), numImages))

    train_imgids = sorted(sum(race_imgid_dict.values(),[]))
    result = {'train_ids':race_labels, 'train_imgids':train_imgids}
    print('race={}, labels len={}, imgids len={}'.format(RFW_race, len(race_imgid_dict), len(train_imgids)))
    save_obj(result, 'BUPT-Equalizedface-{}'.format(RFW_race), '../meta')
