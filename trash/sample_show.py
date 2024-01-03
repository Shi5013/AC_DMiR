import nibabel as nib
import numpy as np
# 总之数据不是全0
img = nib.load('./Data/Liver_4DCT_label/1_XJ_3958812_Ex20%_label.nii.gz')

data = img.get_data()

np.savetxt('sample.txt',data[:,:,10])

print(data[1])