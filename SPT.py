import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import torch.nn.functional as nnf

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    # def __init__(self, size, gpuid, mode='bilinear'):
    def __init__(self, size, mode='bilinear'): # add gpuid for train and regis
        super().__init__()

        self.size = size
        self.mode = mode

        # 创建并注册 grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)  # 增加一个维度
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):  

        self.grid = self.grid.to(src.device)

        # new locations
        new_locs = self.grid + flow # 逐元素相加 torch.Size([1, 3, (size)])
        shape = flow.shape[2:] # 这是又返回原始图像的尺寸

        # need to normalize grid values to [-1, 1] for resampler
        # 标准化到[-1,1]之间
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        # 这尼玛的作者也不知道为啥，就转换一下维度。。
        # grid_sample 函数的要求是 (batch_size, height, width, N)，
        # 其中 N 表示坐标的维度。
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

# 这个SPT的核心就是nnf.grid_sample() , 难点就是张量维度的变换


"""
def read_data(file_path):
    file = nib.load(file_path)
    file = file.get_fdata()
    file = np.transpose(file, (2, 0, 1))
    file = torch.from_numpy(file).float()
    file = file.unsqueeze(0).unsqueeze(0)
    file = file.to(device)
    return file
def save_nii(x,name,field):
    x = x.to('cpu').detach().numpy()
    x = np.transpose(x, (0, 3, 4, 2, 1))
    if field == 0:
        nifti_img = nib.Nifti1Image(x[0, :, :, :, 0],affine=np.eye(4))
    else:
        nifti_img = nib.Nifti1Image(x[0, :, :, :, :],affine=np.eye(4))
    nib.save(nifti_img,'./{}.nii.gz'.format(name))
    print(f"{name} saved")


device = torch.device("cuda:0")

flow_field = torch.randn(1,3,96,256,256)
flow_field = flow_field.to(device)

seg_path = '/media/user_gou/Elements/Shi/recon/Liver_4DCT_label100slice_resample_liver/patient2/11_DMH_4165169_Ex20%_label_liver.nii.gz'
seg_file = read_data(seg_path)
spt = SpatialTransformer((96,256,256))
result = spt(seg_file,flow_field)
save_nii(result,"seg_spt_test",0)
"""

