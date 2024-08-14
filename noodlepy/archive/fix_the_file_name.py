import os
import glob

data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc" )

list_of_file_paths = glob.glob(os.path.join(data_folder, '**/*.txt'), recursive=True)

for file_path in list_of_file_paths:
    f_split = file_path.split('_')
    if 'redone' in f_split[-1]:
        new_file_name = '_'.join(f_split[:-1])
        new_file_name = new_file_name + '.txt'
        os.rename(file_path, new_file_name)
