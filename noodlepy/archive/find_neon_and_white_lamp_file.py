import os
import shutil

def get_unique_filename(dest_dir, filename):
    """
    Generate a unique filename by appending a number if a file with the same name already exists.
    """
    base_name, ext = os.path.splitext(filename)
    new_filename = filename
    counter = 1

    while os.path.exists(os.path.join(dest_dir, new_filename)):
        new_filename = f"{base_name}_{counter}{ext}"
        counter += 1

    return new_filename

def copy_matching_files(source_dir, dest_dir):
    # Define the patterns to search for in filenames
    patterns = [
        "spectralcal_60x_1mm_65mW_cw-853_grating3_air",
        "neon-cal_1mm"
    ]

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Walk through the directory tree
    for root, _, files in os.walk(source_dir):
        for filename in files:
            # Check if the filename contains any of the search patterns
            if any(pattern in filename for pattern in patterns):
                source_file = os.path.join(root, filename)
                unique_filename = get_unique_filename(dest_dir, filename)
                dest_file = os.path.join(dest_dir, unique_filename)
                print(f"Copying: {source_file} -> {dest_file}")
                shutil.copy2(source_file, dest_file)

if __name__ == "__main__":
    # Change these paths to your source and destination directories
    source_directory = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/hnc_raw_data_store"
    destination_directory = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/neon_lamp"

    copy_matching_files(source_directory, destination_directory)
