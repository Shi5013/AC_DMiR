import nibabel as nib
import os
import numpy as np

def combine_organs(file_paths):
    try:
        # 加载所有器官的分割结果
        organs_data = [nib.load(file_path).get_fdata() for file_path in file_paths]

        # 合并器官，将器官位置设为1，其余位置设为0
        combined_data = np.sum(organs_data, axis=0)
        combined_data[combined_data > 1] = 1

        # 创建新的NIfTI图像
        combined_img = nib.Nifti1Image(combined_data, nib.load(file_paths[0]).affine)

        # 替换原文件
        nib.save(combined_img, file_paths[0])

        print(f"文件 {file_paths[0]} 已成功合并器官并保存。")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # 指定包含NIfTI文件的文件夹路径
    for i in range(20):
        folder_path = '/media/user_gou/Elements/Shi/Reg_Seg/data_20_patients/patient{}/mask'.format(i)

        # 获取文件夹中所有NIfTI文件
        nifti_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

        # 遍历每个器官的文件并进行合并
        for organ_file in nifti_files:
            organ_file_path = os.path.join(folder_path, organ_file)
            
            # 调用合并函数
            combine_organs([organ_file_path])
