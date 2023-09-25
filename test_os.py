# In[1]:
# Imports
import os
import numpy as np
import torch
import itertools
import ramanspy as rp
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

def load_files(root_folder, file_format):
    """
    List and load all files with a specific file format in a folder and its subfolders.
    Args:
    - root_folder: The root folder to start the search from.
    - file_format: The file format you want to filter (e.g., '.txt').
    Returns:
    A list of loaded file contents that match the specified format.
    """
    matching_files = []
    # Search for all the matching files
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_format):
                # Collecting the file name
                file_path = os.path.join(root, file)
                matching_files.extend([file_path])

                try:
                    # load the spectral from the file
                    metadata,num_metadata_rows_to_skip = get_metadata(file_path)
                    data_df = pd.read_csv(file_path, sep='\s+',header=None, skiprows=num_metadata_rows_to_skip)
                    print(data_df)

                except FileNotFoundError:
                    print(f"The file '{file_path}' was not found when call load_files.")
                except Exception as e:
                    print(f"An error occurred when call load_files: {e}")

    return matching_files


def get_metadata(file_path):
    # Initialize variables
    metadata = {}
    empty_row = 0
    data_started = False

    try:
        # Open the file and read it line by line
        with open(file_path, 'r') as file:
            for line in file:
                if not data_started:
                    # Check if the line is empty, indicating the end of metadata
                    if line.strip() == "":
                        data_started = True
                        empty_row += 1
                    # Add the row into metadata if is not empty
                    else:
                        # Split the line into key and value based on the ":" delimiter
                        parts = line.strip().split(":")
                        key, value = parts[0].strip(), parts[1].strip()
                        metadata[key] = value # Add key-value pair to the metadata dictionary
                
                # Once the end of metadata is indicated, find where data starts, and them count the empty rows between metadata and data
                else:
                    if line.strip() == "":
                        empty_row += 1
                    else:
                        break

                num_metadata_rows_to_skip = len(metadata) +empty_row +1


        # Print metadata and data
        print("Metadata:")
        print(metadata)
        print("Empty rows between metadata and data:")
        print(empty_row)
        print("Number of rows to skip for metadata:")
        print(num_metadata_rows_to_skip)

    except FileNotFoundError:
        print(f"The file '{file_path}' was not found when call get_metadata.")
    except Exception as e:
        print(f"An error occurred when call get_metadata: {e}")
    return metadata,num_metadata_rows_to_skip


# Specify the root folder you're interested in.
root_folder = '/Users/yifeigu/Documents/Carney_Lab/Data'

# Call the function to list all files in the specified folder and one layer of subfolders.
# list_matching_files = list_files(root_folder, '.asc')
load_matching_files = load_files(root_folder, '.asc')


# In[3]:
# Preprocessing


def baseline_poly(x, y, order):
    coefs = poly.polyfit(x, y, order)
    ffit = poly.polyval(x, coefs)
    return y - ffit

    # From wavelenght (nm) to Raman shift (cm⁻1)

    # Baseline correction (ALS/Polyfit)

    # Smoothing (Savitzky Golay)

    # Normalization (SNV)

# %%
