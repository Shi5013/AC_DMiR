import os
import time
import torch
import argparse
import numpy as np
import nibabel as nib
from Loss import *
from dataset import *
from network_seg import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='set hyperparemeters : lr,epoches,gpuid...')

parser.add_argument('-lr','--learning_rate',
                    dest='learning_rate',
                    default=2e-5,
                    help='learning rate,default=1e-3')
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

save_prefix = 'model_'
save_interval = 10  # 每隔10个 epoch 保存一次模型

model = Seg_net() # 输入是fixed和moving的堆叠
model = model.to(device) # model -> GPU

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

start_time = time.time()

dataset = fixed_moving_seg(args.file_list,args.label_list)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

writer = SummaryWriter(args.tensorboard)
count = 0
for epoch in range(args.epochs):
    # 设置模型为训练模式
    losss = 0
    model.train()
    mask_save = torch.zeros(1, 1, 96, 256, 256)


    # 遍历数据加载器，获取每个批次的数据
    for batch in data_loader:
        # fixed
        fixed_file = batch['fixed']
        fixed_file = fixed_file.to(device)

        # seg mask
        seg_label = batch['fixed_mask']
        seg_label = seg_label.to(device)

        # 将梯度归零
        optimizer.zero_grad()

        # 前向传播
        output = model(fixed_file) # return mask,final_moved
        mask = output[0]# sigmoid
        mask_save = (mask >= 0.5).float()# 前面输出sigmoid，在这里进行二值化

        # loss-dice_loss
        # loss_mask = F.l1_loss(mask,seg_label)
        #criterion = nn.BCELoss()
        #loss_mask = criterion(mask,seg_label)# 这是一个sigmoid和二值求这个？
        #不行，只要是两个二值化的求就回不去
        loss_mask = dice_loss(mask,seg_label)
        writer.add_scalar('loss_mask',loss_mask.item(),count)
        count = count + 1
        # print(loss_mask.item())

        losss = losss + loss_mask.item()

        loss_mask.backward()

        optimizer.step()
    if (epoch + 1) % save_interval == 0:

        mask_save_cpu = mask_save.to('cpu').detach().numpy()
        seg_label_cpu = seg_label.to('cpu').detach().numpy()

        # 将 NumPy 数组的维度重新排列
        mask_save_cpu = np.transpose(mask_save_cpu, (0, 3, 4, 2, 1))
        seg_label_cpu = np.transpose(seg_label_cpu, (0, 3, 4, 2, 1))

        # 创建一个 NIfTI 图像对象
        nifti_img_mask = nib.Nifti1Image(mask_save_cpu[0, :, :, :, 0], affine=np.eye(4))
        nifti_seg_label = nib.Nifti1Image(seg_label_cpu[0, :, :, :, 0], affine=np.eye(4))

        # 保存 NIfTI 图像到.nii.gz 文件
        nib.save(nifti_img_mask, 'mask_image{}.nii.gz'.format(epoch+1))
        nib.save(nifti_seg_label, 'label{}.nii.gz'.format(epoch+1))

        # 构建保存路径，包含有关模型和训练的信息
        save_path = f"{args.save_folder}{save_prefix}epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss_mask.item()}, Model saved as {save_path}")
    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {losss},Time:{epoch_time}")
    start_time = end_time
writer.close()