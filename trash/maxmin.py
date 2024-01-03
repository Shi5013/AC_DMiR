import nibabel as nib

# 读取 NIfTI 文件
nii_file_path = './processed_data/nii_resample/niidata1/resampled_Ex20%.nii'  # 替换为你的文件路径
nii_img = nib.load(nii_file_path)

# 获取 NIfTI 数据数组
nii_data = nii_img.get_fdata()

# 计算最大值和最小值
max_value = nii_data.max()
min_value = nii_data.min()

# 打印结果
print(f"Maximum Value: {max_value}")
print(f"Minimum Value: {min_value}")
