import os

def list_files(directory, output_file):
    # 获取指定目录下的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 将每个文件的绝对路径写入文件
    with open(output_file, 'w') as file:
        for file_name in files:
            file_path = os.path.abspath(os.path.join(directory, file_name))
            file.write(file_path + '\n')

target_directory = "/media/user_gou/Elements/Shi/Reg_Seg/processed_data/Extracted_Label_5"

# 指定输出文件的路径
output_file_path = "file.txt"

# 调用函数将文件的绝对路径写入文件
list_files(target_directory, output_file_path)

