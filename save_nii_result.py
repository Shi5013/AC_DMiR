import torch
import numpy as np
import nibabel as nib

def save_nii(x,path_name,field):
    x = x.to('cpu').detach().numpy()
    x = np.transpose(x, (0, 3, 4, 2, 1))
    if field == 0:
        nifti_img = nib.Nifti1Image(x[0, :, :, :, 0],affine=np.eye(4))
    else:
        nifti_img = nib.Nifti1Image(x[0, :, :, :, :],affine=np.eye(4))
    # nib.save(nifti_img,'./inference_results/{}.nii.gz'.format(name))
    nib.save(nifti_img,path_name)
    print(f"{path_name} saved")