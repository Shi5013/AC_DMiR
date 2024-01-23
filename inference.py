import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
from AC_DMiR import AC_DMiR
from Loss import ssim_loss
# from skimage.metrics import structural_similarity as ssim
# from train import *
# 不能添加这个，加上之后就会导致再次进入train的循环。

def save_nii(x,name,field):
    x = x.to('cpu').detach().numpy()
    x = np.transpose(x, (0, 3, 4, 2, 1))
    if field == 0:
        nifti_img = nib.Nifti1Image(x[0, :, :, :, 0],affine=np.eye(4))
    else:
        nifti_img = nib.Nifti1Image(x[0, :, :, :, :],affine=np.eye(4))
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
fixed_path = '/media/user_gou/Elements/Shi/recon/recon_image_resample/patient1/01_01_01_01.nii.gz'
moving_path = '/media/user_gou/Elements/Shi/recon/recon_image_resample/patient1/01_02_01_01.nii.gz'
ground_truth = '/media/user_gou/Elements/Shi/recon/Liver_4DCT_nii100slice/10_CCE_4823203_Ex20%/image.nii.gz'
fixed_file = read_data(fixed_path)
moving_file = read_data(moving_path)
ground_truth = read_data(ground_truth)
input_data = torch.cat((fixed_file,moving_file),1)


# 加载保存的.pth文件
checkpoint_path = './save_models/Reg_Seg_With_attention/4D_Liver_13_patients_recon/model_epoch15.pth'  
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

    # save_nii(mask_save,"mask_save",0)
    # save_nii(final_moved,"final_moved",0)
    # save_nii(final_field,"final_field",1)
    # save_nii(init_moved,"init_moved",0)
    # save_nii(init_field,"init_field",1)

# 计算指标
# 1. SSIM
ssim_value = ssim_loss(final_moved,ground_truth)
ssim_value_moving =  ssim_loss(moving_file,ground_truth)
ssim_value = ssim_value.to('cpu').detach().numpy()
ssim_value_moving = ssim_value_moving.to('cpu').detach().numpy()
# 直接把loss拿来用了，loss里面计算的1-ssim，这里再减一次，回头可以单独写一个函数。
print("ssim value: ",1 - ssim_value)
print("moving ssim value: ",1 - ssim_value_moving)
# 2. PSNR
mse = nn.MSELoss()
mse_value = mse(final_moved,ground_truth)
mse_value_moving = mse(moving_file,ground_truth)
mse_value = mse_value.to('cpu').detach().numpy()
mse_value_moving = mse_value_moving.to('cpu').detach().numpy()
psnr_value = 10 * np.log10(1 / mse_value)
psnr_value_moving = 10 * np.log10(1 / mse_value_moving)
print("panr value: ",psnr_value)
print("moving panr value : ",psnr_value_moving)
# 3. RMSE
rmse_value = np.sqrt(mse_value)
rmse_value_moving = np.sqrt(mse_value_moving)
print("rmse value: ",rmse_value)
print("moving rmse value: ",rmse_value_moving)