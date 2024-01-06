import os

base_folder = '/media/user_gou/Elements/Shi/Reg_Seg/data_20_patients'

for i in range(20):
    patient_folder = os.path.join(base_folder, f'patient{i:02d}')
    os.makedirs(patient_folder)

    # 在每个patient文件夹下创建image和mask子文件夹
    image_folder = os.path.join(patient_folder, 'image')
    mask_folder = os.path.join(patient_folder, 'mask')
    os.makedirs(image_folder)
    os.makedirs(mask_folder)

print("文件夹和子文件夹创建完成。")
