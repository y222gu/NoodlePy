import os
import glob

data_folder = r'C:\Users\Yifei Gu\Documents\NoodlePy\noodlepy\data\202404_OvCa-project_calibrated\test\270OC\20240415_EVs_desalted1x_270OC'
list_of_file_paths = glob.glob(os.path.join(data_folder, '**/*.txt'), recursive=True)

for file_path in list_of_file_paths:
    f_split = file_path.split('_')
    f_split[7]+='OC'
    new_file_name = '_'.join(f_split)
    os.rename(file_path, new_file_name)
