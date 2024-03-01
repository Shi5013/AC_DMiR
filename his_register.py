# import os
# import SimpleITK as sitk

# def histogram_matching(fixed_image, moving_image):
#     """
#     使用直方图配准将移动图像匹配到固定图像的直方图。

#     参数:
#     fixed_image: 固定图像的文件路径
#     moving_image: 移动图像的文件路径

#     返回值:
#     registered_image: 配准后的移动图像
#     """

#     # 读取固定图像和移动图像
#     fixed_img = sitk.ReadImage(fixed_image, sitk.sitkFloat32)
#     moving_img = sitk.ReadImage(moving_image, sitk.sitkFloat32)

#     # 创建直方图匹配器
#     matcher = sitk.HistogramMatchingImageFilter()
#     matcher.SetNumberOfHistogramLevels(1024)
#     matcher.SetNumberOfMatchPoints(7)
#     matcher.ThresholdAtMeanIntensityOn()

#     # 执行直方图匹配
#     registered_img = matcher.Execute(moving_img, fixed_img)

#     # 获取移动图像文件名
#     moving_img_name = os.path.basename(moving_image)

#     # 保存配准后的移动图像到当前目录
#     sitk.WriteImage(registered_img, "registered_" + moving_img_name)

# def register_images(txt_file1, txt_file2):
#     """
#     从两个txt文件中读取nii文件路径并进行配准。

#     参数:
#     txt_file1: 包含固定图像路径的txt文件
#     txt_file2: 包含移动图像路径的txt文件

#     返回值:
#     None
#     """

#     # 读取txt文件中的nii文件路径
#     with open(txt_file1, 'r') as f:
#         fixed_images = f.readlines()
#     with open(txt_file2, 'r') as f:
#         moving_images = f.readlines()

#     # 去除路径中的空白符号
#     fixed_images = [line.strip() for line in fixed_images]
#     moving_images = [line.strip() for line in moving_images]

#     # 检查文件列表长度是否一致
#     if len(fixed_images) != len(moving_images):
#         print("Error: 文件列表长度不一致。")
#         return

#     # 逐对配准图像
#     for fixed_img_path, moving_img_path in zip(fixed_images, moving_images):
#         histogram_matching(fixed_img_path, moving_img_path)

#     print("配准完成。")

# # 示例用法
# if __name__ == "__main__":
#     txt_file1 = "./file_label/4D_Liver_13patients/ground_truth/Liver_4DCT_ground_truth.txt"
#     txt_file2 = "./file_label/4D_Liver_13patients/recon/Liver_4DCT_file.txt"
#     register_images(txt_file1, txt_file2)

import os

def save_file_paths_to_txt(folder_path, txt_file):
    # 获取文件夹中所有文件的绝对路径
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 将文件路径按照文件名的数字顺序排序
    file_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    # 将排序后的文件路径保存到txt文件中
    with open(txt_file, 'w') as f:
        for file_path in file_paths:
            f.write(file_path + '\n')

# 示例用法
if __name__ == "__main__":
    folder_path = "/media/user_gou/Elements/Shi/recon/his_regis"  # 替换为你的文件夹路径
    txt_file = "./file_label/4D_Liver_13patients/recon/Liver_4DCT_file_his_regis.txt"  # 保存文件路径的txt文件名

    save_file_paths_to_txt(folder_path, txt_file)
