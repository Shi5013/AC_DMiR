import torch
from torchsummary import summary
from AC_DMiR import *


# 创建模型实例
model = AC_DMiR()
# 指定输入维度
input_size = (1, 2, 96, 256, 256)  # 替换成你实际的输入维度

# 打印模型参数量
summary(model, input_size=input_size)
