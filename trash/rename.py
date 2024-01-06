import os

# 指定目标文件夹路径
folder_path = '/media/user_gou/Elements/Shi/Reg_Seg/data2/patient02/mask'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    # 获取文件的绝对路径
    old_file_path = os.path.join(folder_path, filename)

    # 提取文件名的后16位作为新文件名
    new_filename = filename[-16:]

    # 构建新文件的绝对路径
    new_file_path = os.path.join(folder_path, new_filename)

    # 更改文件名
    os.rename(old_file_path, new_file_path)

print("文件名修改完成。")
