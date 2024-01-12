import os

def list_files(directory):
    # 获取指定目录下的所有文件
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def extract_number(filename):
    # 从文件名中提取数字部分
    return int(''.join(filter(str.isdigit, filename)))

def write_to_txt(file_list, output_file):
    # 将文件路径和文件名按照数字大小排序后写入txt文件
    with open(output_file, 'a') as f:
        for file_path in sorted(file_list, key=extract_number):
            f.write(file_path + '\n')

if __name__ == "__main__":
    # 指定文件夹路径

    for i in range(20):
        if i != 18:
            folder_path = '/media/user_gou/Elements/Shi/Reg_Seg/new_data/{}_HM10395/norm_img'.format(100+i)
            
            # 获取文件列表
            files = list_files(folder_path)
            
            # 指定输出文件路径
            output_file_path = '/media/user_gou/Elements/Shi/Reg_Seg/file_label/new_list_norm.txt'
            
            # 将文件路径写入txt文件，按数字大小排序
            write_to_txt(files, output_file_path)
        else:
            continue