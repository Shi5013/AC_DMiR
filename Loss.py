"""
之前编写的loss有些问题,训练过程中不会好像学习不到新的东西,loss不会发生变换。
现在根据论文的介绍,重新编写loss程序
如果依然有问题，就把每一块拆开来进行编写。

现在要计算五个地方的损失函数：
1.Initial Deformation Field
2.Initial moved
3.Initial moved seg
4.Final deformation field 
5.Final moved

"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def dice_loss(prediction_volume, true_volume):
    intersection = torch.sum(torch.mul(prediction_volume,true_volume))
    union = torch.sum(prediction_volume) + torch.sum(true_volume)
    dice_loss = 1 - (2.0 * intersection) / union

    return dice_loss

# 首先是Initial deformation field的梯度损失L_smooth
class Grad:
    """
    N-D gradient loss

    # (batch_size, channels, depth, height, width)
    y_pred = torch.randn(2, 1, 64, 64, 64, requires_grad=True)
    gradient_loss_calculator = Grad(penalty='l1')
    loss = gradient_loss_calculator.loss(None, y_pred)
    print("Gradient loss:", loss.item())

    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

def ssim_loss(moved,label):
    # 只接受np数组，所以要把tensor转化为numpy数据类型
    moved = moved.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    moved = moved.squeeze(0).squeeze(0)
    label = label.squeeze(0).squeeze(0)
    data_range = max(moved.max() - moved.min(), label.max() - label.min())
    ssi_index, _ = ssim(moved, label, full=True,data_range=data_range)
    ssi_index = 1.0-ssi_index
    ssi_index = torch.tensor(ssi_index,dtype = torch.float32)
    return ssi_index

def total_loss(moved,label):
    # edge = Edge_Loss(moved,label)
    edge = SobelLoss()
    edge_loss = edge(moved,label)
    ssim = ssim_loss(moved,label)
    l1 = F.l1_loss(moved,label)
    return edge_loss + ssim + l1

class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()

        sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],
                                [[-2,0,2],[-4,0,4],[-2,0,2]],
                                [[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32)
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]],
                                [[-2,-4,-2],[0,0,0],[2,4,2]],
                                [[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

        sobel_z = torch.tensor([[[1,2,1],[2,4,2],[1,2,1]],
                                [[0,0,0],[0,0,0],[0,0,0]],
                                [[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]], dtype=torch.float32)
        self.sobel_z = sobel_z.unsqueeze(0).unsqueeze(0)

    def forward(self, moved,label):

        device = moved.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        sobel_z = self.sobel_z.to(device)

        # moved和label都是五维的数组 1,1,64,256,256
        moved_grad_x = F.conv3d(moved, sobel_x, padding=1)
        moved_grad_y = F.conv3d(moved, sobel_y, padding=1)
        moved_grad_z = F.conv3d(moved, sobel_z, padding=1)

        label_grad_x = F.conv3d(label, sobel_x, padding=1)
        label_grad_y = F.conv3d(label, sobel_y, padding=1)
        label_grad_z = F.conv3d(label, sobel_z, padding=1)

        loss_x = F.l1_loss(moved_grad_x, label_grad_x)
        loss_y = F.l1_loss(moved_grad_y, label_grad_y)
        loss_z = F.l1_loss(moved_grad_z, label_grad_z)

        loss_x = torch.abs(loss_x)
        loss_y = torch.abs(loss_y)
        loss_z = torch.abs(loss_z)

        return (loss_x + loss_y + loss_z) / 3.0

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
"""
# Test
moved = torch.randn(1,1,96,256,256)
label = torch.randn(1,1,96,256,256)
# res = Edge_Loss(moved,label)
loss = SobelLoss()
res = loss(moved,label)
print("Sobel_Loss: ",res)

"""    




