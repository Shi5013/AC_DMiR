# Cross-Task Attention Block

# Input:1,1,24,64,64

import torch
import torch.nn as nn
import torch.nn.functional as F
from Cross_attention import NLBlockND_cross

class CTAB(nn.Module):
    def __init__(self, in_channel=32):
        super(CTAB, self).__init__()

        self.model = NLBlockND_cross(in_channel)

    def forward(self,R_B,S_B):
        W_R = self.model(R_B,S_B)
        W_S = self.model(S_B,R_B)
        temp = torch.mul(W_R,W_S)
        W_Q = self.model(temp,temp)

        return W_Q
    
"""
# Test:
# 进行三次交叉运算后，输出的维度什么完全没有变化
model = CTAB(in_channel=1)
R = torch.randn(1,1,12,32,32)
S = torch.randn(1,1,12,32,32)

outputs = model(R,S)

print(outputs.size())

"""