import os
import nibabel as nib
import numpy as np

def normalize_nii(input_path, output_path):
    # 读取 NIfTI 文件
    nii_img = nib.load(input_path)

    # 获取 NIfTI 数据数组
    nii_data = nii_img.get_fdata()

    # 计算最大值和最小值
    max_value = nii_data.max()
    min_value = nii_data.min()

    # 归一化操作，将值映射到 -1 到 1 之间  0-1
    # normalized_data = 2 * (nii_data - min_value) / (max_value - min_value) - 1
    normalized_data = (nii_data - min_value) / (max_value - min_value)

    # 创建新的 NIfTI 图像对象
    normalized_nii_img = nib.Nifti1Image(normalized_data, affine=nii_img.affine)

    # 保存归一化后的 NIfTI 文件
    nib.save(normalized_nii_img, output_path)

    print(f"Normalized NIfTI file saved at: {output_path}")

def normalize_all_files(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 列出输入文件夹中的所有 NII 文件
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 处理每个 NII 文件
    for nii_file in nii_files:
        input_path = os.path.join(input_folder, nii_file)
        output_path = os.path.join(output_folder, f"norm_{nii_file}")
        normalize_nii(input_path, output_path)

for i in range(13):

    # 定义输入文件夹和输出文件夹路径
    input_folder = './processed_data/nii_resample/niidata{}'.format(i+1)
    output_folder = './processed_data/norm_nii/niidata{}'.format(i+1)

    # 执行归一化操作
    normalize_all_files(input_folder, output_folder)
