import os
import shutil
import pandas as pd

def extract_txt_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    txt_file_count = 0
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                shutil.copy(file_path, output_folder)
                txt_file_count += 1
    print(f"Total .txt files found: {txt_file_count}")


def organize_files_by_parts(input_folder, output_folder, annotation_file_path):
    annotation_all = pd.read_excel(annotation_file_path)
    for file in os.listdir(input_folder):
        if file.endswith('.txt'):
            parts = file.split('_')
            date = parts[0]
            patient_id = int(parts[1])
            # find the corresponding patient in the metadata file
            if patient_id in annotation_all['OD Number'].values:
                patient_metadata_row = annotation_all[annotation_all['OD Number'] == patient_id]
                cancer_stage = patient_metadata_row['Staging'].values[0]
            else:
                cancer_stage = 'unknown'

            folder_name = f"{date}_patient_{patient_id}_stage_{cancer_stage}"
            folder_path = os.path.join(output_folder, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            shutil.copy(os.path.join(input_folder, file), os.path.join(folder_path, file))

    print("Files have been organized into folders.")

if __name__ == "__main__":
    input_folder = r'C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_cleaned'
    output_folder = r'C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_reformated'
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")
    # extract_txt_files(input_folder, output_folder)
    organize_files_by_parts(input_folder, output_folder, metadata_file)