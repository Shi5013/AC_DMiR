import torch
import torch.nn as nn
import torch.nn.functional as F

# 目前的这个配准里面没有交叉注意力
# 主体的通道数是16
class Reg_network(nn.Module):
    def __init__(self):
        super(Reg_network, self).__init__()
        
        # 一个完整的block块：
        self.ConvBlock1 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock2 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock3 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        # 最大池化kernelsize是3，3，3，虽然论文里面说的是2，2，2
        # 但是实际上好像有一定的不同，这里就暂时按照论文说的
        self.MaxPooling1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2), padding=0)
        self.MaxPooling2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.MaxPooling3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        self.Conv_size1_for_cat =nn.Conv3d(in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=(1, 1, 1),
            padding=0,# padding=1 还是会改变输入图像的大小的
            bias=False)

        self.ConvBlock4 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=32,
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
            nn.Conv3d(in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.Final_Conv =nn.Conv3d(in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)        

        self.decoder1 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？先用转置卷积
            nn.ConvTranspose3d(in_channels=32, 
                               out_channels=16, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, 
                               padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？
            nn.ConvTranspose3d(in_channels=32, 
                               out_channels=16, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？
            nn.ConvTranspose3d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.insteadofcross = nn.Conv3d(in_channels=32,
                                        out_channels=16,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.insteadbn = nn.BatchNorm3d(32)
        self.instead_relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # 这一步操作是将x的两个通道的数据分别取出来，x是一个5D的数据:(N,C,T,H,W)

        fixed_0 = torch.unsqueeze(x[:, 0, :, :, :], 1) # 1,1,64,256,256
        moving_0 = torch.unsqueeze(x[:, 1, :, :, :], 1) # 1,1,64,256,256

        fixed_1      = self.ConvBlock1(fixed_0) # 1,1,64,256,256 -> 1,32,64,256,256
        fixed_1_pool = self.MaxPooling1(fixed_1) # 1,32,64,256,256 -> 1, 32, 32, 128, 128
        fixed_2      = self.ConvBlock2(fixed_1_pool) # 1, 32, 32, 128, 128->1, 32, 32, 128, 128
        fixed_2_pool = self.MaxPooling2(fixed_2) # 1,32,32,128,128 -> 1, 32, 16, 64, 64
        fixed_3      = self.ConvBlock3(fixed_2_pool) # 1, 32, 16, 64, 64->1, 32, 16, 64, 64
        fixed_3_pool = self.MaxPooling3(fixed_3) # 1, 32, 16, 64, 64 -> 1, 32, 8, 32, 32

        moving_1      = self.ConvBlock1(moving_0)
        moving_1_pool = self.MaxPooling1(moving_1)
        moving_2      = self.ConvBlock2(moving_1_pool)
        moving_2_pool = self.MaxPooling2(moving_2)
        moving_3      = self.ConvBlock3(moving_2_pool)
        moving_3_pool = self.MaxPooling3(moving_3)

        
        # 这一块用来替代cross attention，减少内存开销
        instead1 = torch.cat((fixed_3_pool,moving_3_pool),1)
        instead2 = torch.cat((moving_3_pool,fixed_3_pool),1)
        a = self.insteadofcross(instead1)# (1,32,8,32,32)->(1,16,8,32,32)
        b = self.insteadofcross(instead2)# (1,32,8,32,32)->(1,16,8,32,32)
        instead = torch.cat((a,b),1) # (1,32,8,32,32)
        instead = self.insteadbn(instead) # (1,32,8,32,32)

        # 这个结果拿出来，去进行bottleneck
        out_for_bottleneck = instead# 这是一个32通道的，后面要弄成1通道


        cat1 = torch.cat((fixed_1,moving_1),1)# ((1,32,64,256,256),(1,32,64,256,256))->(1,64,64,256,256)
        cat1 = self.Conv_size1_for_cat(cat1) # (1,64,64,256,256)->(1,32,64,256,256)
        cat_for_deformable1 = cat1#111666

        cat2 = torch.cat((fixed_2,moving_2),1) #1, 32, 32, 128, 128 -> 1, 64, 32, 128, 128
        cat2 = self.Conv_size1_for_cat(cat2)  # 1, 64, 32, 128, 128 -> 1, 32, 32, 128, 128
        cat_for_deformable2 = cat2# 111666

        cat3 = torch.cat((fixed_3,moving_3),1) # 1, 32, 16, 64, 64 -> 1, 64, 16, 64, 64
        cat3 = self.Conv_size1_for_cat(cat3) # 1, 64, 16, 64, 64 -> 1, 32, 16, 64, 64
        cat_for_deformable3 = cat3 # 11116666

        reg_input = instead # (1, 32, 8, 32, 32)
        reg_input = self.instead_relu(reg_input)# 32
        
        # 第一次反卷积
        reg_decoder1 = self.decoder1(reg_input) # (1, 32, 8, 32, 32) -> (1, 32, 16, 64, 64)
        CB4_input = torch.cat((reg_decoder1,cat_for_deformable3),1)# (1, 32, 16, 64, 64)->(1, 64, 16, 64, 64)
        ConvBlock4 = self.ConvBlock4(CB4_input) # 32
        
        # 第二次反卷积
        reg_decoder2 = self.decoder2(ConvBlock4) # 输入输出都是32，图像放大两倍 1, 32, 16, 64, 64 -> 1, 32, 32, 128, 128
        CB5_input = torch.cat((reg_decoder2,cat_for_deformable2),1) # 1, 32, 32, 128, 128 -> 1, 64, 32, 128, 128
        ConvBlock5 = self.ConvBlock5(CB5_input) # 1, 64, 32, 128, 128 -> 1, 16, 32, 128, 128
        
        # 第三次反卷积
        # 这里第三个的通道数需要修改，有两种改法，1.只修改decoder的，2.conv5和decoder，我先选择简单的，只修改decoder
        reg_decoder3 = self.decoder3(ConvBlock5) # 1, 16, 32, 128, 128 -> 1, 32, 64, 256, 256
        
        final_conv_input = torch.cat((reg_decoder3,cat_for_deformable1),1) 
        # (1, 32, 64, 256, 256),(1,32,64,256,256) -> (1,64,64,256,256)
        
        Deformable_Field = self.Final_Conv(final_conv_input)# (1,64,64,256,256)->(1,3,64,256,256)
        
        return Deformable_Field,out_for_bottleneck
    
"""
# Test:
fixed = torch.randn(16,32,32)
moving = torch.randn(16,32,32)

fixed = fixed.unsqueeze(0).unsqueeze(0)
moving = moving.unsqueeze(0).unsqueeze(0)

inputimg = torch.cat((fixed,moving),1)

model = Reg_network()

output = model(inputimg)

print(output[0].size())
print(output[1].size())

# =================输出===================
# torch.Size([1, 3, 16, 32, 32])
# torch.Size([1, 32, 2, 4, 4])
"""
