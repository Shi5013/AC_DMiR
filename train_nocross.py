import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import nibabel as nib
from SPT import *
from Loss import *
# from AC_DMiR import *
from without_cross import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter
from save_nii_result import *


parser = argparse.ArgumentParser(description='set hyperparemeters : lr,epoches,gpuid...')

parser.add_argument('-lr','--learning_rate',
                    dest='learning_rate',
                    default=2e-5,
                    help='learning rate,default=1e-3')
parser.add_argument('-g','--gpu_id',
                    dest='gpu_id',
                    default=1,
                    help='choose gpu,default=0')
parser.add_argument('-e','--epochs',
                    dest='epochs',
                    default=32,
                    help='epochs,default=300')
parser.add_argument('-s','--save_folder',
                    dest='save_folder',
                    default='./save_models/Reg_Seg_Without_attention/4D_Liver_13_patients_recon',
                    help='where models saves')
parser.add_argument('-fl','--file_list',
                    dest='file_list',
                    default='./file_label/4D_Liver_13patients/recon/Liver_4DCT_file.txt',
                    help='file list,txt file.include fixed and moving')
parser.add_argument('-ll','--label_list',
                    dest='label_list',
                    default='./file_label/4D_Liver_13patients/recon/Liver_4DCT_file_label_liver.txt',
                    help='label list,txt file')
parser.add_argument('-gt','--ground_truth_list',
                    dest='gt_list',
                    default='./file_label/4D_Liver_13patients/ground_truth/Liver_4DCT_ground_truth.txt',
                    help='label list,txt file')
parser.add_argument('-t','--tensorboard',
                    dest='tensorboard',
                    default='./save_logs/Reg_Seg_Without_attention/4D_Liver_13_patients_recon',
                    help='tensorboard file')


args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu_id))
model = AC_DMiR_without_cross_attention() # 输入是fixed和moving的堆叠
model = model.to(device) # model -> GPU

save_prefix = 'model_'
save_interval = 5  # 每隔10个 epoch 保存一次模型

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

spt = SpatialTransformer((96,256,256))

# loss的一些设定：
gradient_loss_calculator = Grad(penalty='l1') # 形变场的光滑损失
mse = nn.MSELoss()


start_time = time.time()

dataset = fixed_moving_seg(args.file_list,args.label_list,args.gt_list)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

writer = SummaryWriter(args.tensorboard)
count = 0

for epoch in range(args.epochs):
    # 设置模型为训练模式
    model.train()
    losss = 0
    moved_save = torch.zeros(1, 1, 96, 256, 256)
    mask_save = torch.zeros(1, 1, 96, 256, 256)
    field_save = torch.zeros(1, 3, 96, 256, 256)

    # 遍历数据加载器，获取每个批次的数据
    for batch in data_loader:
        # fixed
        fixed_file = batch['fixed']
        fixed_file = fixed_file.to(device)
        # moving 
        moving_file = batch['moving']
        moving_file = moving_file.to(device)
        # fixed mask
        fixed_seg_label = batch['fixed_mask']
        fixed_seg_label = fixed_seg_label.to(device)
        # moving mask
        moving_seg_label = batch['moving_mask']
        moving_seg_label = moving_seg_label.to(device)        
        # input
        input_data = torch.cat((fixed_file,moving_file),1)

        # 将梯度归零
        optimizer.zero_grad()

        # 前向传播
        output = model(input_data) # return mask,final_moved
        mask = output[0] # 这个是分割的结果
        final_moved = output[1] # 这个是最终的moved
        mask_save = (mask >= 0.5).float()
        moved_save = final_moved
        final_field = output[2]
        field_save = output[2]# 这个是final_field
        initial_moved = output[3]
        initial_field = output[4]
        # loss
        # 1 Initial_field的平滑性损失
        initial_smooth_loss = gradient_loss_calculator.loss(None, initial_field)

        # 2 Initial_field的一致性约束
        init_dsc_temp = spt(moving_seg_label,initial_field)
        init_dsc_loss = dice_loss(init_dsc_temp,fixed_seg_label)

        # 3 initial_moved的MSE损失
        init_mse_loss = mse(initial_moved,fixed_file)

        # 4 final_field的平滑性损失
        final_smooth_loss= gradient_loss_calculator.loss(None,final_field)

        # 5 final_field的一致性约束
        final_dsc_temp = spt(moving_seg_label,final_field)
        final_dsc_loss = dice_loss(final_dsc_temp,fixed_seg_label)

        # 6 final_moved的损失
        final_moved_loss = mse(final_moved,fixed_file) # mse
        loss_moved = total_loss(final_moved,fixed_file) # l1 + sobel + ssim

        # 7 分割网络结果的损失
        criterion = nn.BCELoss()
        loss_mask = criterion(mask,fixed_seg_label)
        dice_loss_mask = dice_loss(mask,fixed_seg_label)

        final_moved_loss = 3000*final_moved_loss
        init_mse_loss = 2000*init_mse_loss

        # loss = 2*loss_moved+0.3*loss_mask+init_dsc_loss+0.1*initial_smooth_loss+100*init_mse_loss+0.1*final_smooth_loss
        loss = final_moved_loss+init_mse_loss+dice_loss_mask
        writer.add_scalar('init_mse_loss',init_mse_loss,count)
        writer.add_scalar('final_moved_loss',final_moved_loss,count)
        writer.add_scalar('dice_loss',dice_loss_mask,count)
        count = count+1

        losss = losss + loss.item()

        loss.backward()

        optimizer.step()
    if (epoch + 1) % save_interval == 0:

        save_nii(fixed_file,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/fixed_file{}".format(epoch+1),0)
        save_nii(moving_file,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/moving_file{}".format(epoch+1),0)

        save_nii(mask_save,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/mask_save{}".format(epoch+1),0)
        save_nii(final_moved,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/final_moved{}".format(epoch+1),0)
        save_nii(final_field,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/final_field{}".format(epoch+1),1)
        save_nii(initial_moved,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/init_moved{}".format(epoch+1),0)
        save_nii(initial_field,"./save_results/Reg_Seg_Without_attention/4D_Liver_13_patients_recon/init_field{}".format(epoch+1),1)

        # moved_save_cpu = moved_save.to('cpu').detach().numpy()
        # mask_save_cpu = mask_save.to('cpu').detach().numpy()
        # field_save_cpu = field_save.to('cpu').detach().numpy()

        # mask_save_cpu = np.transpose(mask_save_cpu, (0, 3, 4, 2, 1))
        # moved_save_cpu = np.transpose(moved_save_cpu, (0, 3, 4, 2, 1))  # 从 (1, 1, 96, 256, 256) 变为 (1, 256, 256, 96, 1)
        # field_save_cpu = np.transpose(field_save_cpu, (0, 3, 4, 2, 1))  # 从 (1, 1, 96, 256, 256) 变为 (1, 256, 256, 96, 1)

        # nifti_img_mask = nib.Nifti1Image(mask_save_cpu[0, :, :, :, 0], affine=np.eye(4))
        # nifti_img_moved = nib.Nifti1Image(moved_save_cpu[0, :, :, :, 0], affine=np.eye(4))  # 取第一个样本并去掉单维度
        # nifti_img_field = nib.Nifti1Image(field_save_cpu[0, :, :, :, :], affine=np.eye(4))  # 取第一个样本并去掉单维度
        # # 保存 NIfTI 图像到.nii.gz 文件
        # nib.save(nifti_img_mask, './results_noatt/mask_image{}.nii.gz'.format(epoch+1))
        # nib.save(nifti_img_moved, './results_noatt/moved_image{}.nii.gz'.format(epoch+1))
        # nib.save(nifti_img_field, './results_noatt/field_image{}.nii.gz'.format(epoch+1))

        # 构建保存路径，包含有关模型和训练的信息
        save_path = f"{args.save_folder}{save_prefix}epoch{epoch+1}.pth"
    
        
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, Model saved as {save_path}")
    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {losss},Time:{epoch_time}")
    start_time = end_time
writer.close()