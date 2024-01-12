import torch
import numpy as np
import nibabel as nib
from AC_DMiR import AC_DMiR
from train import *

def save_nii(x,name):
    x = x.to('cpu').detach().numpy()
    x = np.transpose(x, (0, 3, 4, 2, 1))
    nifti_img = nib.Nifti1Image(x[0, :, :, :, 0],affine=np.eye(4))
    nib.save(nifti_img,'./inference_results/{}.nii.gz'.format(name))
    print(f"{name} saved")

def read_data(file_path):
    file = nib.load(file_path)
    file = file.get_fdata()
    file = np.transpose(file, (2, 0, 1))
    file = torch.from_numpy(file).float()
    file = file.unsqueeze(0).unsqueeze(0)
    file = file.to(device)
    return file


# 创建模型实例
model2 = AC_DMiR()
device = torch.device("cuda:1")
model2 = model2.to(device)

# 加载数据
fixed_path = './new_data/100_HM10395/norm_img/norm_resampled_100_HM10395_07-02-2003-NA-p4-14571_0.0A-423.1_image.nii.gz'
moving_path = './new_data/100_HM10395/norm_img/norm_resampled_100_HM10395_07-02-2003-NA-p4-14571_10.0A-423.2_image.nii.gz'
fixed_file = read_data(fixed_path)
moving_file = read_data(moving_path)
input_data = torch.cat((fixed_file,moving_file),1)


# 加载保存的.pth文件
checkpoint_path = './models_save/model_epoch10.pth'  
checkpoint = torch.load(checkpoint_path, map_location=device)
model2.load_state_dict(checkpoint)

# 将模型设置为评估模式
model2.eval()

# 在测试数据上进行推断
with torch.no_grad():
    output = model2(input_data)
    # mask,final_moved,final_field,Init_moved,init_field

    mask = output[0]
    mask_save = (mask >= 0.5).float()
    final_moved = output[1]
    final_field = output[2]
    init_moved = output[3]
    init_field = output[4]

    save_nii(mask_save,"mask_save")
    save_nii(final_moved,"final_moved")
    save_nii(final_field,"final_field")
    save_nii(init_moved,"init_moved")
    save_nii(init_field,"init_field")

