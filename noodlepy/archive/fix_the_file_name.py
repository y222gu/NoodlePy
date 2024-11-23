import os
import glob

data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc", "test")

list_of_file_paths = glob.glob(os.path.join(data_folder, '**/*.txt'), recursive=True)

for file_path in list_of_file_paths:
    f_split = file_path.split('/')
    old_file_name = f_split[-1]
    old_file_name_splited = old_file_name.split('_')
    new_file_name = old_file_name_splited[1:]
    new_file_name = '_'.join(new_file_name)
    new_path = os.path.join(data_folder, new_file_name)
    print(new_path)
    if os.path.exists(new_path):
        print(f"File already exists: {new_path}")
        os.remove(new_path)
    os.rename(file_path, new_path)

    # if 'redone' in f_split[-1]:
    #     new_file_name = '_'.join(f_split[:-1])
    #     new_file_name = new_file_name + '.txt'
    #     if os.path.exists(new_file_name):
    #         print(f"File already exists: {new_file_name}")
    #         os.remove(new_file_name)
    #     os.rename(file_path, new_file_name)
