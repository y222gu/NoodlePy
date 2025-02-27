import os
import numpy as np
from sif_reader import SifReader

def convert_sif_to_txt(source_folder):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".sif"):
            sif_path = os.path.join(source_folder, filename)
            txt_path = os.path.join(source_folder, os.path.splitext(filename)[0] + ".txt")

            try:
                sif_data = SifReader(sif_path)
                image_data = sif_data.spectrum  # Get spectral data

                np.savetxt(txt_path, image_data, fmt="%.6f", delimiter="\t")

                print(f"Converted: {sif_path} -> {txt_path}")

            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    source_directory = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/neon_lamp"
    convert_sif_to_txt(source_directory)