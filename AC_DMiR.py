# 这个文件是对所有模块的组装

import torch
import torch.nn as nn
import torch.nn.functional as F
from bottleneck import *
from Cross_attention import *
from CrossTask_AttB import *
from Final_Decoder import *
from network_reg import *
from network_seg import *
from SPT import *

# 输入应该有三个：fixed,moving,label
class AC_DMiR(nn.Module):
    def __init__(self):
        super(AC_DMiR, self).__init__()
        # Weakly-Supervised Registration Learning
        self.Init_Reg = Reg_network() # return Deformable_Field,out_for_bottleneck
        self.spt = SpatialTransformer((96,256,256)) # return moved
        # self.spt = SpatialTransformer((96,256,256)) # return moved
        
        # Supervised egmentation Learning
        self.Init_Seg = Seg_net() # return Seg_output,out_for_bottleneck

        # self.bottleneck = BottleneckBlock(32,32,1)

        self.cross = CTAB() # return W_Q
        self.finaldecoder = FinalDecoder() # return x (1,1,96,256,256)

    def forward(self,x): # x是fixed和moving在dim=1的堆叠
        # 注意这里好多网络的输出是两个值，所以要把需要的拿出来
        moving = torch.unsqueeze(x[:, 1, :, :, :], 1) # 1,1,96,256,256
        
        output1 = self.Init_Reg(x)
        init_field = output1[0]
        reg_bottleneck = output1[1]

        # reg_bottleneck = self.bottleneck(reg_bottleneck)

        Init_moved = self.spt(moving,init_field)
        output2 = self.Init_Seg(Init_moved)
        mask = output2[0]
        seg_bottleneck = output2[1]

        # seg_bottleneck = self.bottleneck(seg_bottleneck)

        cross_task_attention = self.cross(reg_bottleneck,seg_bottleneck)
        final_field = self.finaldecoder(cross_task_attention)
        final_moved = self.spt(moving,final_field)
        return mask,final_moved,final_field,Init_moved,init_field

"""
# Test:
device = torch.device("cuda:0")
fixed = torch.randn(16,32,32)
moving = torch.randn(16,32,32)
fixed = fixed.to(device)
moving = moving.to(device)
fixed = fixed.unsqueeze(0).unsqueeze(0)
moving = moving.unsqueeze(0).unsqueeze(0)

x = torch.cat((fixed,moving),1)
print(fixed.size())
print(moving.size())
print("x size:",x.size())
x1 = torch.unsqueeze(x[:, 0, :, :, :], 1) # 1,1,96,256,256
x2 = torch.unsqueeze(x[:, 1, :, :, :], 1) # 1,1,96,256,256
print(x1.size())
print(x2.size())
# print(moving==x2) # True

model = AC_DMiR()
model.to(device)
output = model(x)
print("==========")
print(output[0].size())
print(output[1].size())
print(output[2].size())
print(output[3].size())
print(output[4].size())
# ==========
# torch.Size([1, 1, 96, 256, 256])
# torch.Size([1, 1, 96, 256, 256])
"""