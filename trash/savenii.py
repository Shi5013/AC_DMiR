import nibabel as nib
import numpy as np
import torch

moved_save = torch.randn(1,1,96,256,256)
moved_save.to('cuda:0')


moved_save_cpu = moved_save.to('cpu').detach().numpy()
# 将 NumPy 数组的维度重新排列
moved_save_cpu = np.transpose(moved_save_cpu, (0, 3, 4, 2, 1))  # 从 (1, 1, 96, 256, 256) 变为 (1, 256, 256, 96, 1)
# 创建一个 NIfTI 图像对象
nifti_img = nib.Nifti1Image(moved_save_cpu[0, :, :, :, 0], affine=np.eye(4))  # 取第一个样本并去掉单维度
# 保存 NIfTI 图像到.nii.gz 文件
nib.save(nifti_img, 'output_image{}.nii.gz'.format(1))
