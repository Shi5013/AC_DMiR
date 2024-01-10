import os
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

def resample_nii(input_path, output_path, target_shape):
    # 读取NIfTI文件
    nii_img = nib.load(input_path)
    
    # 获取NIfTI数据数组
    nii_data = nii_img.get_fdata()

    # 计算每个维度的缩放比例
    zoom_factors = (target_shape[0] / nii_data.shape[0],
                    target_shape[1] / nii_data.shape[1],
                    target_shape[2] / nii_data.shape[2])

    # 执行重采样，使用最近邻插值
    resampled_data = zoom(nii_data, zoom_factors, order=0, mode='nearest')

    # 创建新的NIfTI图像对象
    resampled_nii_img = nib.Nifti1Image(resampled_data, affine=nii_img.affine)

    # 保存重采样后的NIfTI文件
    nib.save(resampled_nii_img, output_path)

    print(f"Resampled NIfTI file saved at: {output_path}")

def resample_all_files(input_folder, output_folder, target_shape):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 列出输入文件夹中的所有NII文件
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 处理每个NII文件
    for nii_file in nii_files:
        input_path = os.path.join(input_folder, nii_file)
        output_path = os.path.join(output_folder, f"resampled_{nii_file}")
        resample_nii(input_path, output_path, target_shape)

# 定义输入文件夹和输出文件夹路径
for i in range(20):
    if i!=18:
        input_folder = '/media/user_gou/Elements/Shi/4D_Lung_nii_mask_and_data_20240109/{}_HM10395/lung_label'.format(100+i)
        output_folder = '/media/user_gou/Elements/Shi/Reg_Seg/new_data/{}_HM10395/lung_label'.format(100+i)
    else:
        continue
    
    target_shape = (256, 256, 96)

    resample_all_files(input_folder, output_folder, target_shape)
