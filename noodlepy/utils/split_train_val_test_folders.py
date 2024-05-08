import splitfolders

print("Splitting started.")

# Define the input folder containing all the files
input_folder = r"/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/Raman_DB/plasma/separate_by_stage"

# Define the output folder where train, validation, and test sets will be created
output_folder = r"/mnt/c/Users/Yifei/Documents/NoodlePy/noodlepy/data/Raman_DB/plasma/separate_by_stage"

# Define the split ratios (e.g., 80% train, 10% validation, 10% test)
# You can adjust these ratios according to your requirements
split_ratio = (0.7, 0.2, 0.1)  # Train, Validation, Test

# Perform the split
splitfolders.ratio(input_folder, output=output_folder, seed=88, ratio=split_ratio, group_prefix=None)

print("Splitting complete.")
