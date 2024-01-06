import nibabel as nib
import os

def crop_nifti_file(file_path):
    try:
        # 加载NIfTI文件
        img = nib.load(file_path)
        data = img.get_fdata()

        # 裁剪数据（在第三个维度减去两层）
        cropped_data = data[:, :, :48]

        # 保存裁剪后的数据回原位置（覆盖原文件）
        img_cropped = nib.Nifti1Image(cropped_data, img.affine)
        nib.save(img_cropped, file_path)

        print(f"文件 {file_path} 已成功裁剪并保存。")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    for i in range(20):
        folder_path = '/media/user_gou/Elements/Shi/Reg_Seg/data_20_patients/patient{}/mask'.format(i)
        nifti_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
        for nifti_file in nifti_files:
            nifti_file_path = os.path.join(folder_path, nifti_file)
            crop_nifti_file(nifti_file_path)
