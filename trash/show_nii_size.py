import nibabel as nib
import numpy as np

def get_nifti_size(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        size = data.shape
        return size
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # 指定NIfTI文件路径
    nifti_file_path = '/media/user_gou/Elements/Shi/Reg_Seg/data_20_patients/patient0/image/500.000000-P4P100S100I0 Gated 0.0-90726_image.nii.gz'

    # 获取NIfTI文件大小
    nifti_size = get_nifti_size(nifti_file_path)

    if nifti_size:
        print(f"NIfTI文件大小为: {nifti_size}")
    else:
        print("无法获取NIfTI文件大小。")
