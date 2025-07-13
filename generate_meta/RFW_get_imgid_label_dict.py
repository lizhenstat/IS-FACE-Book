# -*- coding: utf-8 -*
# 从 MXNet 格式的 .rec 文件中提取每张图像的 ID 与对应标签，并生成两个字典：
# （1）图像 ID 到标签的映射（imgid → label），保存到 BUPT-Equalizedface-imgid-label-dict.pkl。
# （2）标签到图像 ID 列表的映射（label → imgid list），保存到 BUPT-Equalizedface-label-imgidList-dict.pkl。
import matplotlib.pyplot as plt

import os 
import time
import pickle

from torchvision import transforms
import mxnet as mx
import numpy as np
import torch
import numbers
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

### python save and load object
def save_obj(obj, name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'rb') as f:
        return pickle.load(f)

class MXFaceDataset_Race(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset_Race, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r') # support random access
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s) 
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]            # mxnet_webface:490623
        s = self.imgrec.read_idx(idx)       # return the record at the given index
        header, img = mx.recordio.unpack(s) # Unpack a MXImageRecord to string. # 随机样本:HEADER(flag=0, label=8099.0, id=410683, id2=0) 410683在train.idx文件中 应该是每个图片从哪里开始
        label = header.label                # header:meta data
        img_id = header.id                  # 标记当前image的index
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy() # mxnet_webface:[112,112,3] type=uint8
        return sample, label, img_id

    def __len__(self):
        return len(self.imgidx)

if __name__ == '__main__':    
    stime = time.time()
    data_dir = '/home/sata/Dataset/Equalizedface'
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:{}".format(12585),rank=0,world_size=1)
    train_set = MXFaceDataset_Race(data_dir, local_rank=0)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
    trainloader = DataLoader(dataset=train_set, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False)   # NumClass=27999 


    imgid_label_dict={}
    for i, (sample, label, img_id) in enumerate(trainloader):
        imgid_label_dict.update(dict(zip(img_id.cpu().numpy(), label.cpu().numpy())))
    etime = time.time()
    print('Total time ', (etime-stime)/60) # 11 min 
    save_obj(imgid_label_dict, 'BUPT-Equalizedface-imgid-label-dict', '../meta') # 不加预处理, 4 min数据过一遍，加预处理7分钟。

    # 将相同图片label的id合并
    label_imgid_dict = {}
    for imgid,label in imgid_label_dict.items():
        if label in label_imgid_dict.keys():
           label_imgid_dict[label].append(imgid)
        else: 
           label_imgid_dict[label]=[imgid]
    save_obj(label_imgid_dict, 'BUPT-Equalizedface-label-imgidList-dict', '../meta')

    ### mxnet数据集共1251416张图片，27999类
    ### header, _ = mx.recordio.unpack(s) 从rec文件中读数据header.label = (*,*) 经检查，第二个label都为1

    ### 数据 
    ### EqualizedFace 包含下面三个文件
    ### (1)property: 只有一行:27999, 112,112
    ### (2)train.rec: 很大 6G+, 里面存储的诗mxnet读数据的格式
    ### (3)train.idx: 共 1279416 行，最后一行是 1279415,7064611860(第二个数字是改image的起始字节数)
    ### train_balancedface.lst: 存储的img图片的路径和对应的label标签，共1251416行 (mxnet过一遍的个数)
    ### 1279416 - 1251416 = 280000
    ### jpg文件夹下的图片个数 1251430
    # train.idx 最后一行 1279415 7064611860

