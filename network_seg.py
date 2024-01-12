import torch
import torch.nn as nn
import torch.nn.functional as F
from bottleneck import BottleneckBlock

# 直接是一个Unet结构，不需要交叉注意力的模块
# 或者换一个思路，我是不是也带上这个交叉注意力模块？
class Seg_net(nn.Module):
    def __init__(self):
        super(Seg_net, self).__init__()
        # Cross-Modal Attention 模块里面经过Cross Attention之后的操作
        self.relu_after_cross = nn.ReLU(inplace=True)
        """
        # 一个完整的block块：
        self.ConvBlock1 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )
        self.ConvBlock2 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333 
            # # 严格来说，这一个并不属于Block，先注释掉吧
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )
        self.ConvBlock3 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        """
        # 换成残差连接
        self.conforresidual = nn.Conv3d(1,32,3,1,1,bias=False)
        self.ConvBlock1 = ResidualBlock(32)
        self.ConvBlock2 = ResidualBlock(32)
        self.ConvBlock3 = ResidualBlock(32)

        # 最大池化kernelsize是3，3，3，虽然论文里面说的是2，2，2
        # 但是实际上好像有一定的不同，这里就暂时按照论文说的
        self.MaxPooling1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2), padding=0)
        self.MaxPooling2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.MaxPooling3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        self.Conv_size1_for_cat =nn.Conv3d(in_channels=64,
            out_channels=32,
            kernel_size=1,
            stride=(1, 1, 1),
            padding=0,# padding=1 还是会改变输入图像的大小的
            bias=False)

        self.bottleneck = BottleneckBlock(32,32,32)


        self.ConvBlock4 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock5 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=64,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.Final_Conv =nn.Conv3d(in_channels=64,
            out_channels=1,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)        

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
        
    def forward(self, Init_moved):

        # Seg_net输入的就只有一张图片，所以不用这么复杂，就只有一个Init_Moved
        Init_moved = self.conforresidual(Init_moved)
        Init_moved_1      = self.ConvBlock1(Init_moved) # 1,1,64,256,256 -> 1,32,64,256,256
        Init_moved_1_pool = self.MaxPooling1(Init_moved_1) # 1,32,64,256,256 -> 1, 32, 32, 128, 128
        Init_moved_2      = self.ConvBlock2(Init_moved_1_pool) # 1, 32, 32, 128, 128->1, 32, 32, 128, 128
        Init_moved_2_pool = self.MaxPooling2(Init_moved_2) # 1,32,32,128,128 -> 1, 32, 16, 64, 64
        Init_moved_3      = self.ConvBlock3(Init_moved_2_pool) # 1, 32, 16, 64, 64->1, 32, 16, 64, 64
        Init_moved_3_pool = self.MaxPooling3(Init_moved_3) # 1, 32, 16, 64, 64 -> 1, 32, 8, 32, 32
       
        # 经过bottleneck
        after_bottleneck = self.bottleneck(Init_moved_3_pool)

        # 下面进行解码过程，目的是为了得到分割的图像，也就是label
        seg_input = after_bottleneck # (1, 32, 8, 32, 32)
        seg_input = self.relu_after_cross(seg_input)
        # 这里要进行一个反卷积
        seg_decoder1 = self.decoder1(seg_input) # (1, 32, 8, 32, 32) -> (1, 32, 16, 64, 64)
        CB4_input = torch.cat((seg_decoder1,Init_moved_3),1)# (1, 32, 16, 64, 64)->(1, 64, 16, 64, 64)
        ConvBlock4 = self.ConvBlock4(CB4_input)# 输入64，输出32  1, 64, 16, 64, 64 -> 1, 32, 16, 64, 64
        # 第二次反卷积
        seg_decoder2 = self.decoder2(ConvBlock4) # 输入输出都是32，图像放大两倍 1, 32, 16, 64, 64 -> 1, 32, 32, 128, 128
        CB5_input = torch.cat((seg_decoder2,Init_moved_2),1) # 1, 32, 32, 128, 128 -> 1, 64, 32, 128, 128
        ConvBlock5 = self.ConvBlock5(CB5_input) # 1, 64, 32, 128, 128 -> 1, 16, 32, 128, 128
        # 第三次反卷积
        # 这里第三个的通道数需要修改，有两种改法，1.只修改decoder的，2.conv5和decoder，我先选择简单的，只修改decoder
        # reg_decoder3 = self.decoder3(ConvBlock5) # 1, 16, 32, 128, 128 -> 1, 16, 64, 256, 256
        seg_decoder3 = self.decoder3(ConvBlock5) # 1, 16, 32, 128, 128 -> 1, 32, 64, 256, 256
        
        final_conv_input = torch.cat((seg_decoder3,Init_moved_1),1) 
        # (1, 32, 64, 256, 256),(1,32,64,256,256) -> (1,64,64,256,256)
        
        Seg_output = self.Final_Conv(final_conv_input)# (1,64,64,256,256)->(1,1,64,256,256)
        # 二值化处理
        Seg_output = torch.sigmoid(Seg_output)
        # seg_mask = (Seg_output >= 0.5).float()
        seg_mask = Seg_output
        
        return seg_mask,after_bottleneck
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 将输入直接加到输出上
        out = self.relu(out)
        return out

"""
# Test:
Init_moved = torch.randn(96,224,224)
Init_moved = Init_moved.unsqueeze(0).unsqueeze(0)

model = Seg_net()

output = model(Init_moved)

print(output[0].size())
print(output[1].size())

# =============输出===========
torch.Size([1, 1, 96, 224, 224])
torch.Size([1, 32, 12, 28, 28])
"""
