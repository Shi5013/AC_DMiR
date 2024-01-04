import os

def delete_files_with_ending(folder_path):
    # 获取文件夹中所有文件的列表
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否以_为结尾，并且是一个文件而不是文件夹
        if file_name.endswith('_.nii.gz') and os.path.isfile(file_path):
            try:
                # 删除文件
                os.remove(file_path)
                print(f"文件 '{file_name}' 已删除.")
            except Exception as e:
                print(f"无法删除文件 '{file_name}': {e}")

# 替换 'your_folder_path' 为你的文件夹路径
folder_path = './recon_image'
delete_files_with_ending(folder_path)