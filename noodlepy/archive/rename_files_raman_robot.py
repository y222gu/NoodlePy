import os

def load_and_check_files(folder_path, output_folder_path):

    try:
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith('.txt'):
                    # Store relative path from folder_path
                    rel_dir = os.path.relpath(root, folder_path)
                    rel_file = os.path.join(rel_dir, filename) if rel_dir != '.' else filename
                    files.append(rel_file)
        identifiers = set()

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for file_name in files:
            basename = os.path.basename(file_name)
            subfolder_name = os.path.dirname(file_name)
            parts = basename.split('_')
            # increase sample number by 2
            parts[5] = str(int(parts[5]) - 1)

            new_name = f"{'_'.join(parts[:])}"
            os.rename(os.path.join(folder_path, subfolder_name, basename), os.path.join(output_folder_path, subfolder_name, new_name))
            print(f"Renamed {basename} to {new_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
folder_path = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/2026_01_27/spectra/New folder'
output_folder_path = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/2026_01_27/spectra/New folder'
load_and_check_files(folder_path, output_folder_path)