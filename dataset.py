import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class fixed_moving_seg(Dataset):
    def __init__(self, txt_file,label_file):
        # 开两个文件夹，一个是nii的文件，一个是label文件
        with open(txt_file, 'r') as file:
            self.file_paths = file.readlines()
        with open(label_file, 'r') as mask:
            self.label_file_paths = mask.readlines()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        fixed_path = self.file_paths[idx].strip() # 这个就是fixed
        fixed_mask_path = self.label_file_paths[idx].strip()
        
        # 随机选择另一个文件
        moving_idx = np.random.randint(len(self))
        moving_path = self.file_paths[moving_idx].strip()
        moving_mask_path = self.label_file_paths[moving_idx].strip()

        # 检查是否来自同一个文件夹
        while (not self.check_same_folder(fixed_path, moving_path)) and (moving_idx!=idx):
            # 如果不是同一个文件夹，则重新选择第二个文件
            moving_idx = np.random.randint(len(self))
            moving_path = self.file_paths[moving_idx].strip()
            moving_mask_path = self.label_file_paths[moving_idx].strip()
        # 使用nibabel加载NIfTI文件
        fixed_nifti = nib.load(fixed_path)
        moving_nifti = nib.load(moving_path)
        fixed_mask_nifti = nib.load(fixed_mask_path)
        moving_mask_nifti = nib.load(moving_mask_path)

        # 获取NIfTI数据的数组
        fixed = fixed_nifti.get_fdata()
        moving = moving_nifti.get_fdata()
        fixed_mask_data = fixed_mask_nifti.get_fdata()
        moving_mask_data = moving_mask_nifti.get_fdata()

        # 重新排列数组维度,将维度从 (256, 256, 96) 转换为 (96, 256, 256)
        fixed = np.transpose(fixed, (2, 0, 1))
        moving = np.transpose(moving, (2, 0, 1))
        fixed_mask_data = np.transpose(fixed_mask_data,(2,0,1))
        moving_mask_data = np.transpose(moving_mask_data,(2,0,1))

        fixed = torch.from_numpy(fixed).float()
        moving = torch.from_numpy(moving).float()
        fixed_mask_data = torch.from_numpy(fixed_mask_data).float()
        moving_mask_data = torch.from_numpy(moving_mask_data).float()

        fixed = torch.unsqueeze(fixed,dim=0).float()
        moving = torch.unsqueeze(moving,dim=0).float() #4D? 在daytaloader里面还有一个batch_size
        fixed_mask = fixed_mask_data.unsqueeze(0)
        moving_mask = moving_mask_data.unsqueeze(0)

        return {
            'fixed': fixed,
            'moving': moving,
            'fixed_mask' : fixed_mask,
            'moving_mask':moving_mask
        }

    def check_same_folder(self, path1, path2):
        folder1 = os.path.dirname(path1)
        folder2 = os.path.dirname(path2)
        return folder1 == folder2

"""  
# Test:
file_path = './file_list.txt'  # 替换为txt文件路径
label_path = './file_label_list.txt'
dataset1 = fixed_moving_seg(file_path,label_path)
data_loader = DataLoader(dataset=dataset1, batch_size=1, shuffle=True, num_workers=0)
for batch in data_loader:
    # 获取输入数据和标签
    fixed = batch['fixed']
    moving = batch['moving']
    fixed_mask = batch['fixed_mask']
    moving_mask = batch['moving_mask']
    print('fixed size:')
    print(fixed.size())
"""