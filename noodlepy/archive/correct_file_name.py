import os

def load_and_check_files(folder_path, output_folder_path):

    try:
        files = os.listdir(folder_path)
        identifiers = set()

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for file_name in files:
            parts = file_name.split('_')
            if len(parts) > 1:
                identifier = parts[0] + '_' + parts[1]
                identifiers.add(identifier)
                print(f"Found file: {file_name}, Identifier: {identifier}")
            else:
                print(f"Found file: {file_name}, but it does not have enough parts to form an identifier")
        print(f"Unique identifiers: {identifiers}")

        for identifier in identifiers:
            matching_files = [file for file in files if identifier in file]
            matching_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for i, file in enumerate(matching_files, start=1):
                name_parts = file.split('_')
                new_name = f"{'_'.join(name_parts[:-1])}_{i:02}.txt"
                os.rename(os.path.join(folder_path, file), os.path.join(output_folder_path, new_name))
                print(f"Renamed {file} to {new_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
folder_path = r'C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_reformated'
output_folder_path = r'C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_reformated_for_test'
load_and_check_files(folder_path, output_folder_path)