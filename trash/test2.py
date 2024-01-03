import torch

# 创建两个示例张量
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
tensor2 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)

# 逐元素相乘
result_tensor = torch.mul(tensor1, tensor2)

# 打印结果张量
print("Result Tensor:")
print(result_tensor)
