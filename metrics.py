from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

### cosface
class AddMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(AddMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01) 

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) 
        index = torch.where(label != -1)[0]                            
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device) 
        m_hot.scatter_(1, label[index, None], self.m) 
        cosine[index] -= m_hot                        
        ret = cosine * self.s
        return ret

### arcface
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01) 

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        m_hot = torch.zeros(cosine.size()[0], cosine.size()[1], device=cosine.device)
        m_hot = m_hot.scatter(1, label.unsqueeze(1), self.m) 
        cosine = torch.acos(cosine) 
        cosine += m_hot   
        cosine = torch.mul(torch.cos(cosine),self.s)
        return cosine







