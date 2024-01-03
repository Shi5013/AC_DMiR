import torch
from AC_DMiR import AC_DMiR  # 请确保导入了正确的模型类
from train import *

# 创建模型实例
model = AC_DMiR()
device = torch.device("cuda:0")
model = model.to(device)

# 加载保存的.pth文件
checkpoint_path = './models_save/model_epoch10.pth'  # 替换为你的.pth文件路径
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)

# 将模型设置为评估模式
model.eval()

# 在测试数据上进行推断
with torch.no_grad():
    output = model(input_data)
    mask = output[0]
    moved = output[1]

        # 在这里可以根据需要处理输出结果，比如保存或可视化

# 这里只是一个简单的示例，具体的数据加载和处理可能需要根据你的实际情况进行调整
