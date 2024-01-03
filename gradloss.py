import torch
import torch.nn as nn
import torch.nn.functional as F
from Loss import *
# 梯度，表示平滑性，可以先不加试试看

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[[0, 0, 0],
                     [0, -1, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]]
        kernel_h = [[[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    [[0, -1, 0],
                     [0, 0, 0],
                     [0, 1, 0]],
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]]

        kernel_d = [[[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [-1, 0, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        kernel_d = torch.FloatTensor(kernel_d).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.weight_d = nn.Parameter(data=kernel_d, requires_grad=False)


    def get_gray(self, x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)
        x_v = F.conv3d(x, self.weight_v, padding=1,stride=1)
        x_h = F.conv3d(x, self.weight_v, padding=1,stride=1)
        x_d = F.conv3d(x, self.weight_v, padding=1,stride=1)

        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + torch.pow(x_d, 2) + 1e-6)
        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)

# 假设我们有一个预测图像和一个真实图像
# 这些图像应该是张量，形状为 (batch_size, channels, depth, height, width)
output = torch.randn(2, 3, 64, 64, 64)  # 预测图像
gt_img = torch.randn(2, 3, 64, 64, 64)  # 真实图像

# 实例化 GradLoss
grad_loss = GradLoss()

# 计算损失
loss_value = grad_loss(output, gt_img)

print(f'Gradient-based loss: {loss_value.item()}')

