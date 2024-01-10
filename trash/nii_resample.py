import nibabel as nib
import os
import numpy as np
from scipy.ndimage import zoom

def resample_nifti(file_path, new_shape):
    try:
        # 加载NIfTI文件
        img = nib.load(file_path)
        data = img.get_fdata()

        # 计算重采样因子
        resample_factors = np.divide(new_shape, data.shape)

        # 执行重采样
        resampled_data = zoom(data, resample_factors, order=1, mode='nearest')

        # 创建新的NIfTI图像
        resampled_img = nib.Nifti1Image(resampled_data, img.affine)

        # 替换原文件
        nib.save(resampled_img, file_path)

        print(f"文件 {file_path} 已成功重采样并保存。")

    except Exception as e:
        print(f"Error: {e}")

# if __name__ == "__main__":
#     # 指定包含NIfTI文件的文件夹路径
#     for i in range(20):
#         folder_path = '/media/user_gou/Elements/Shi/Reg_Seg/data_20_patients/patient{}/mask'.format(i)
#         nifti_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
#         for nifti_file in nifti_files:
#             nifti_file_path = os.path.join(folder_path, nifti_file)
#             resample_nifti(nifti_file_path, new_shape=(256, 256, 48))

nifti_file_path = '/media/user_gou/Elements/Shi/10_CCE_4823203_Ex20%_label.nii.gz'
resample_nifti(nifti_file_path, new_shape=(256, 256, 96))