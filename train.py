import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import nibabel as nib
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from SPT import *
from Loss import *
from AC_DMiR import *
from dataset import *
from save_nii_result import *

parser = argparse.ArgumentParser(description='set hyperparemeters : lr,epoches,gpuid...')

parser.add_argument('-lr','--learning_rate',
                    dest='learning_rate',
                    default=2e-5,
                    help='learning rate,default=1e-4')
parser.add_argument('-g','--gpu_id',
                    dest='gpu_id',
                    default=0,
                    help='choose gpu,default=0')
parser.add_argument('-e','--epochs',
                    dest='epochs',
                    default=30,
                    help='epochs,default=300')
parser.add_argument('-s','--save_folder',
                    dest='save_folder',
                    default='./models_save/',
                    help='where models saves')
parser.add_argument('-fl','--file_list',
                    dest='file_list',
                    default='./file_label/new_list_norm.txt',
                    help='file list,txt file.include fixed and moving')
parser.add_argument('-ll','--label_list',
                    dest='label_list',
                    default='./file_label/new_list_label.txt',
                    help='label list,txt file')
parser.add_argument('-t','--tensorboard',
                    dest='tensorboard',
                    default='logs',
                    help='tensorboard file')


args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu_id))
model = AC_DMiR() # 输入是fixed和moving的堆叠
model = model.to(device) # model -> GPU

save_prefix = 'model_'
save_interval = 5  # 每隔10个 epoch 保存一次模型

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)  # 每隔5个epoch，将学习率乘以0.9

spt = SpatialTransformer((96,256,256))

# loss的一些设定：
gradient_loss_calculator = Grad(penalty='l1') # 形变场的光滑损失
mse = nn.MSELoss()
ncc_loss = NCC()


start_time = time.time()

dataset = fixed_moving_seg(args.file_list,args.label_list)
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
        init_mae_loss = F.l1_loss(initial_moved,fixed_file)
        
        # init_moved_ncc_loss = ncc_loss.loss(fixed_file,initial_moved)# ncc loss

        # 4 final_field的平滑性损失
        final_smooth_loss= gradient_loss_calculator.loss(None,final_field)

        # 5 final_field的一致性约束
        final_dsc_temp = spt(moving_seg_label,final_field)
        final_dsc_loss = dice_loss(final_dsc_temp,fixed_seg_label)

        # 6 final_moved的损失
        final_moved_loss = mse(final_moved,fixed_file) # mse
        loss_moved = total_loss(final_moved,fixed_file) # l1 + sobel + ssim
        final_moved_mae_loss = F.l1_loss(final_moved,fixed_file)

        # final_moved_ncc_loss = ncc_loss.loss(fixed_file,final_moved)# ncc loss

        # 7 分割网络结果的损失
        criterion = nn.BCELoss()
        loss_mask = criterion(mask,fixed_seg_label)
        dice_loss_mask = dice_loss(mask,fixed_seg_label)
       
        # loss = 0.5*init_mae_loss + final_moved_mae_loss + 0.01*dice_loss_mask
        # writer.add_scalar('init_mae_loss',init_mae_loss,count)
        # writer.add_scalar('final_moved_mae_loss',final_moved_mae_loss,count)
        # writer.add_scalar('dice_loss_mask',dice_loss_mask,count)

        final_moved_loss = 3000*final_moved_loss
        init_mse_loss = 2000*init_mse_loss
        loss = init_mse_loss + final_moved_loss + dice_loss_mask
        writer.add_scalar('init_mse_loss',init_mse_loss,count)
        writer.add_scalar('final_moved_loss',final_moved_loss,count)
        writer.add_scalar('dice_loss',dice_loss_mask,count)

        # loss = init_moved_ncc_loss + final_moved_ncc_loss + dice_loss_mask
        # writer.add_scalar('init_moved_ncc_loss',init_moved_ncc_loss,count)
        # writer.add_scalar('final_moved_ncc_loss',final_moved_ncc_loss,count)
        
        count = count+1
        losss = losss + loss.item()
        writer.add_scalar('loss',losss,epoch)

        loss.backward()
        # print(scheduler.get_lr())

        optimizer.step()
    scheduler.step()
    if (epoch + 1) % save_interval == 0:

        # 这里的保存回头可以写成一行，对于output的保存，不用重复写这么多
        save_nii(fixed_file,"./results/fixed_file{}".format(epoch),0)
        save_nii(moving_file,"./results/moving_file{}".format(epoch),0)
        save_nii(mask_save,"./results/mask_save{}".format(epoch),0)
        save_nii(final_moved,"./results/final_moved{}".format(epoch),0)
        save_nii(final_field,"./results/final_field{}".format(epoch),1)
        save_nii(initial_moved,"./results/init_moved{}".format(epoch),0)
        save_nii(initial_field,"./results/init_field{}".format(epoch),1)
        # 构建保存路径，包含有关模型和训练的信息
        save_path = f"{args.save_folder}{save_prefix}epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, Model saved as {save_path}")
    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {losss},Time:{epoch_time}")
    start_time = end_time
writer.close()