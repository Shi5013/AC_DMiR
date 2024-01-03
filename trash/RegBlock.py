# 尝试了一些可能的结构，还是把整个配准放在一起，
# 因为用到一些中间结果，如果单独写encoder，中间结果拿不出来

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
input : fixed image,moving image
output: Initial Deformation Filed Φ

This module include three block:
1. Initial Registration Encoder
2. Registration Feature Bottleneck
3. Initial Registration Decoder

"""
# 1. Initial Registration Encoder 这个模块是不是要输出四个图像？
class Reg_Encoder(nn.Module):
    def __init__(self):
        super(Reg_Encoder, self).__init__()        
        # Encoder1:
        self.Encoder1 = nn.Sequential(
            # Encoder1-ConvBlock1：
            nn.Conv3d(in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Encoder1-ConvBlock2：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # Encoder2:
        self.Encoder2 = nn.Sequential(
            # Encoder2-ConvBlock1：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Encoder2-ConvBlock2:
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # Encoder3:
        self.Encoder3 = nn.Sequential(
            # Encoder3-ConvBlock1:
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Encoder3-ConvBlock2:
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Three MaxPolling
        self.MaxPooling1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2), padding=0)
        self.MaxPooling2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.MaxPooling3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0) 

    def forward(self, x):

        # 这一步操作是将x的两个通道的数据分别取出来，x是一个5D的数据:(N,C,T,H,W)
        fixed_0 = torch.unsqueeze(x[:, 0, :, :, :], 1) # 1,1,64,256,256
        moving_0 = torch.unsqueeze(x[:, 1, :, :, :], 1) # 1,1,64,256,256

        fixed_1      = self.Encoder1(fixed_0) # 1,1,64,256,256 -> 1,32,64,256,256
        fixed_1_pool = self.MaxPooling1(fixed_1) # 1,32,64,256,256 -> 1, 32, 32, 128, 128
        fixed_2      = self.Encoder2(fixed_1_pool) # 1, 32, 32, 128, 128->1, 32, 32, 128, 128
        fixed_2_pool = self.MaxPooling2(fixed_2) # 1,32,32,128,128 -> 1, 32, 16, 64, 64
        fixed_3      = self.Encoder3(fixed_2_pool) # 1, 32, 16, 64, 64->1, 32, 16, 64, 64
        fixed_3_pool = self.MaxPooling3(fixed_3) # 1, 32, 16, 64, 64 -> 1, 32, 8, 32, 32

        moving_1      = self.ConvBlock1(moving_0)
        moving_1_pool = self.MaxPooling1(moving_1)
        moving_2      = self.ConvBlock2(moving_1_pool)
        moving_2_pool = self.MaxPooling2(moving_2)
        moving_3      = self.ConvBlock3(moving_2_pool)
        moving_3_pool = self.MaxPooling3(moving_3)

        # 应该输出拼接卷积之后的结果，然后送去解码器
        # 先简单cat一下
        encoder1_output = torch.cat((fixed_1_pool,moving_1_pool),1) # channel:32->64
        encoder2_output = torch.cat((fixed_2_pool,moving_3_pool),1) # channel:32->64
        encoder3_output = torch.cat((fixed_3_pool,moving_3_pool),1) # channel:32->64

        return encoder1_output,encoder2_output,encoder3_output

# 2. Registration Feature Bottleneck
# (1,inchannel,32,64,64) -> (1,num_classes,32,64,64)
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, stride=1):
        super(BottleneckBlock, self).__init__()

        # First 3x3x3 convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second 3x3x3 convolution
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 1x1x1 convolution for residual connection
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.residual_bn = nn.BatchNorm3d(out_channels)

        # Final 3x3x3 convolution for segmentation task
        self.reg_seg_conv = nn.Conv3d(out_channels, num_classes, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        residual = x

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        residual = self.residual_conv(residual)
        residual = self.residual_bn(residual)
        out += residual
        out = self.relu(out)

        # Segmentation task final convolution
        bottleneck_output = self.reg_seg_conv(out)

        return bottleneck_output
    
class Reg_Decoder(nn.module):
    def __init__(self):
        self.decoder1 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？先用转置卷积
            nn.ConvTranspose3d(in_channels=32, 
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
            nn.ConvTranspose3d(in_channels=16, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):

        return defor_filed
