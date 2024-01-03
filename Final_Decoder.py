# input:1,1,12,32,32
# output:1,3,96,256,256 deformation field
import torch
import torch.nn as nn
import torch.nn.functional as F
from Cross_attention import *

# 直接是一个Unet结构，不需要交叉注意力的模块
# 或者换一个思路，我是不是也带上这个交叉注意力模块？
class FinalDecoder(nn.Module):
    def __init__(self,in_channel=1):
        super(FinalDecoder, self).__init__()
        self.decoder1 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？先用转置卷积
            nn.ConvTranspose3d(in_channels=1, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, 
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？
            nn.ConvTranspose3d(in_channels=32, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？
            nn.ConvTranspose3d(in_channels=32, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.Final_Conv =nn.Conv3d(in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)

    def forward(self,x):
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.Final_Conv(x)

        return x

"""
# Test:
input = torch.randn(1,1,12,32,32)
model = FinalDecoder()
output = model(input)
print(output.size())
# output:
# torch.Size([1, 3, 96, 256, 256])
"""