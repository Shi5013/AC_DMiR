# 仅仅是前面的配准
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
# from network_reg import *
from network_reg_with_cross_attention import Reg_network_with_attention
from dataset import *
from save_nii_result import *

parser = argparse.ArgumentParser(description='set hyperparemeters : lr,epoches,gpuid...')

parser.add_argument('-lr','--learning_rate',
                    dest='learning_rate',
                    default=1e-4,
                    help='learning rate,default=1e-4')
parser.add_argument('-g','--gpu_id',
                    dest='gpu_id',
                    default=0,
                    help='choose gpu,default=0')
parser.add_argument('-e','--epochs',
                    dest='epochs',
                    default=30,
                    help='epochs,default=30')
parser.add_argument('-s','--save_folder',
                    dest='save_folder',
                    default='./result_withseg/models/',
                    help='where models saves')
parser.add_argument('-fl','--file_list',
                    dest='file_list',
                    default='./file_label/4D_Liver_13patients/recon/Liver_4DCT_file.txt',
                    help='file list,txt file.include fixed and moving')
parser.add_argument('-ll','--label_list',
                    dest='label_list',
                    # default='./file_label/Liver_4DCT_file_label_lung.txt',
                    default='./file_label/4D_Liver_13patients/recon/Liver_4DCT_file_label_liver.txt',
                    help='label list,txt file')
parser.add_argument('-gt','--ground_truth_list',
                    dest='gt_list',
                    default='./file_label/4D_Liver_13patients/ground_truth/Liver_4DCT_ground_truth.txt',
                    help='label list,txt file')
parser.add_argument('-t','--tensorboard',
                    dest='tensorboard',
                    default='./result_withseg/logs/',
                    help='tensorboard file')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu_id))
model = Reg_network_with_attention() # 输入是fixed和moving的堆叠
model = model.to(device) # model -> GPU

save_prefix = 'model_'
save_interval = 5  # 每隔5个 epoch 保存一次模型

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)  # 每隔5个epoch，将学习率乘以0.9

spt = SpatialTransformer((96,256,256))
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

    # 遍历数据加载器，获取每个批次的数据
    for batch in data_loader:
        # fixed
        fixed_file = batch['fixed']
        fixed_file = fixed_file.to(device)
        # moving 
        moving_file = batch['moving']
        moving_file = moving_file.to(device)
        # ground_truth
        ground_truth = batch['ground_truth'] 
        ground_truth = ground_truth.to(device)
        # fixed seg
        fixed_mask = batch['fixed_mask']
        fixed_mask = fixed_mask.to(device)
        # moving seg
        moving_mask = batch['moving_mask']
        moving_mask = moving_mask.to(device)    

        # input
        input_data = torch.cat((fixed_file,moving_file),1)
        # 将梯度归零
        optimizer.zero_grad()
        # 前向传播
        output = model(input_data) # return mask,final_moved
        initial_field = output[0]
        initial_moved = spt(moving_file,initial_field)
        initial_moved_mask = spt(moving_mask,initial_field)

        #initial_moved的MSE损失
        # init_mse_loss = mse(initial_moved,ground_truth)
        init_mse_loss = mse(initial_moved,fixed_file)
        init_mask_mse_loss = mse(initial_moved_mask,fixed_mask)

        # init_mae_loss = F.l1_loss(initial_moved,fixed_file)
        # init_mae_loss = F.l1_loss(initial_moved,ground_truth)
        writer.add_scalar('init_mse_loss',init_mse_loss,count)
        writer.add_scalar('init_mask_mse_loss',init_mask_mse_loss,count)

        loss = init_mse_loss+init_mask_mse_loss
        count = count+1
        losss = losss + loss.item()
        loss.backward()

        optimizer.step()
    scheduler.step()
    if (epoch + 1) % save_interval == 0:

        # save_nii(fixed_file,"./result_withseg/results/fixed_file{}".format(epoch+1),0)
        # save_nii(moving_file,"./result_withseg/results/moving_file{}".format(epoch+1),0)
        save_nii(initial_moved,"./result_withseg/results/init_moved{}".format(epoch+1),0)
        save_nii(initial_field,"./result_withseg/results/init_field{}".format(epoch+1),1)
        # 构建保存路径，包含有关模型和训练的信息
        save_path = f"{args.save_folder}{save_prefix}epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}, Model saved as {save_path}")
    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {losss},Time:{epoch_time}")
    start_time = end_time
writer.close()