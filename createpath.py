import os

def append_file_paths_to_txt(folder_path, output_txt_path):
    with open(output_txt_path, 'a') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                f.write(file_path + '\n')

if __name__ == "__main__":
    # 指定目标文件夹路径
    for i in range(20):
        folder_path_to_read = '/media/user_gou/Elements/Shi/Reg_Seg/data_20_patients/patient{}/mask'.format(i)
        output_txt_path = './file_label/sec_dataset_20patients_label.txt'
        append_file_paths_to_txt(folder_path_to_read, output_txt_path)

