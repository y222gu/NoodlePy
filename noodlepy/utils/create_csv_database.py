import os
import csv
import pandas as pd
from requests import head

# Path to the folder containing images
data_folder = "/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB"

# Path to the CSV file
csv_file_path = "/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB.csv"

# Initialize the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header (optional)
    writer.writerow(["patient_id","sample_type", "spectrum_id", "wavelength[nm]", "intensity"])

# Get a list of file names in the folder and sort them
files = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

# open the PatientLabel.txt file
pateint_table_file_name = "/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/PatientLabel.xlsx"
patient_stage_table = pd.read_excel(pateint_table_file_name, names=['patient_id', 'patient_stage'])

# Initialize the dataframe with id, sample_type, spectrum_id, spectra
df = pd.DataFrame(columns=['patient_id','sample_type', 'spectrum_id', 'wavelength[nm]', 'intensity'])

# Split the files into groups
for filename in files:
    f_split = filename.split('_')
    patient_id = f_split[0]
    sample_type = f_split[1]

    '''
    patient_stage = patient_stage_table[patient_stage_table['patient_id'] == int(patient_id)]
    patient_stage = patient_stage['patient_stage'].values[0]
    '''

    with open(os.path.join(data_folder, filename)) as f:
        data = pd.read_csv(f, sep=",", header=None)

    # Find the repeats in the first column
    repeats = data.iloc[:,0].value_counts()

    # Find the index of the first repeat
    first_value = repeats.idxmax()
    start_indexes = data[data.iloc[:,0] == first_value].index.tolist()

    # Segment the data
    for i in range(len(start_indexes)):
        spectrum_id = i + 1
        if i == len(start_indexes) - 1:
            wavelength = data.iloc[start_indexes[i]:, 0].values
            intensity = data.iloc[start_indexes[i]:, 1].values
        else:
            wavelength = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values
            intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values

        # Write the spectrum to the dataframe
        new_row = {'patient_id': patient_id, 'sample_type': sample_type,
                                      'spectrum_id': spectrum_id, 'wavelength[nm]': wavelength, 'intensity': intensity}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df['laser_wavelength[nm]']= 785
# write the dataframe to the csv file
df.to_csv(csv_file_path, index=False)


