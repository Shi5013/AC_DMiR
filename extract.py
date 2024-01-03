import os
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

def extract_label(input_path, output_path, target_label):
    # 读取NIfTI文件
    nii_img = nib.load(input_path)
    
    # 获取NIfTI数据数组
    nii_data = nii_img.get_fdata()

    # 仅提取目标标签的信息
    label_data = (nii_data == target_label).astype(np.float32)

    # 创建新的NIfTI图像对象
    label_nii_img = nib.Nifti1Image(label_data, affine=nii_img.affine)

    # 保存提取后的NIfTI文件
    nib.save(label_nii_img, output_path)

    print(f"Extracted label {target_label} NIfTI file saved at: {output_path}")

def extract_label_for_all_files(input_folder, output_folder, target_label):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 列出输入文件夹中的所有NII文件
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 处理每个NII文件
    for nii_file in nii_files:
        input_path = os.path.join(input_folder, nii_file)
        output_path = os.path.join(output_folder, f"extracted_label_{target_label}_{nii_file}")
        extract_label(input_path, output_path, target_label)

# 定义输入文件夹和输出文件夹路径以及目标标签
input_folder = './Data/Resampled_Liver_4DCT_label/'
output_folder = './Data/Extracted_Label_5/'
target_label = 5

# 执行提取操作
extract_label_for_all_files(input_folder, output_folder, target_label)
