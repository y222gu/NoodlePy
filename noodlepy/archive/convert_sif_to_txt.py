import os
import numpy as np
import atsif  # Requires Andor SDK

def convert_sif_to_txt(source_folder):
    """
    Converts all .sif files in the source_folder to .txt files.
    """
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".sif"):
            sif_path = os.path.join(source_folder, filename)
            txt_path = os.path.join(source_folder, os.path.splitext(filename)[0] + ".txt")

            try:
                # Initialize atsif library
                atsif.SetFileAccessMode(0)  # Ensure it's in read mode

                # Load the .sif file
                if atsif.ReadFromFile(sif_path) != atsif.ATSIF_SUCCESS:
                    raise Exception(f"Failed to read {sif_path}")

                # Get available data
                if atsif.GetNumberFrames(atsif.ATSIF_RAW, 0) != atsif.ATSIF_SUCCESS:
                    raise Exception(f"Failed to get frames from {sif_path}")

                # Get the image data
                data_size = atsif.GetFrameSize(atsif.ATSIF_RAW)
                data = np.zeros(data_size, dtype=np.float32)

                if atsif.GetFrame(atsif.ATSIF_RAW, 0, data) != atsif.ATSIF_SUCCESS:
                    raise Exception(f"Failed to extract data from {sif_path}")

                # Save data as .txt file
                np.savetxt(txt_path, data, fmt="%.6f", delimiter="\t")

                # Close file
                atsif.CloseFile()

                print(f"Converted: {sif_path} -> {txt_path}")

            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    # Change this to your actual folder path
    source_directory = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/neon_lamp"
    
    convert_sif_to_txt(source_directory)

