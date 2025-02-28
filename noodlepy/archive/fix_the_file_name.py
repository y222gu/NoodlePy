import os
import glob

data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc")

list_of_file_paths = glob.glob(os.path.join(data_folder, '**/*.txt'), recursive=True)

for file_path in list_of_file_paths:
    f_split = file_path.split('/')
    sub_folder = f_split[-2]
    old_file_name = f_split[-1]

    if 'redone' in old_file_name:
        new_file_name = old_file_name.split('_redone')[0]
        new_file_name = new_file_name + '.txt'
        new_file_path = os.path.join(data_folder, sub_folder, new_file_name)
        if os.path.exists(new_file_path):
            print(f"File already exists: {new_file_path}")
            os.remove(new_file_path)
        os.rename(file_path, new_file_path)
