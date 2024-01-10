import os
import nibabel as nib
import numpy as np

def combine_labels(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有NIfTI文件
    nii_files = [file for file in os.listdir(input_folder) if file.endswith('.nii.gz')]

    # 遍历每个NIfTI文件
    for nii_file in nii_files:
        # 读取NIfTI文件
        nii_path = os.path.join(input_folder, nii_file)
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata()

        # 将标签10、11、12、13、14合并为一个标签
        combined_labels = np.isin(nii_data, [10, 11, 12, 13, 14])
        combined_labels = combined_labels.astype(np.uint8)

        # 创建新的NIfTI图像
        combined_nii_img = nib.Nifti1Image(combined_labels, nii_img.affine)

        # 生成新的文件名
        output_file = nii_file.replace('.nii.gz', '_lung.nii.gz')
        output_path = os.path.join(output_folder, output_file)

        # 保存新的NIfTI文件
        nib.save(combined_nii_img, output_path)

        print(f"Combined labels saved to {output_path}")

if __name__ == "__main__":
    input_folder = "/media/user_gou/Elements/Shi/4D_Lung_nii_mask_and_data_20240109/119_HM10395/organ_label"
    output_folder = "/media/user_gou/Elements/Shi/4D_Lung_nii_mask_and_data_20240109/119_HM10395/lung_label"

    combine_labels(input_folder, output_folder)