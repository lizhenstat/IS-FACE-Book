 # -*- coding: utf-8 -*-
"""
Created on 2021-08-20

@author: sirius
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import numpy as np

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        
    def forward(self, logits, label):
        max_fc = torch.max(logits, dim=1, keepdim=True)[0]
        logits_exp = torch.exp(logits - max_fc) 
        logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
        logits_exp = torch.div(logits_exp,logits_sum_exp) 
        loss = logits_exp.gather(1, label.unsqueeze(1)) 
        loss = loss.clamp_min(1e-30) 
        loss = torch.log(loss) * (-1)
        celoss = torch.mean(loss, dim=0)[0]
        return celoss, loss




