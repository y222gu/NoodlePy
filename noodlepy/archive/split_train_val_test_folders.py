import splitfolders

print("Splitting started.")

# Define the input folder containing all the files
input_folder = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB/cleaned/plasma"

# Define the output folder where train, validation, and test sets will be created
output_folder = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB/split_plasma"

# Define the split ratios (e.g., 80% train, 10% validation, 10% test)
# You can adjust these ratios according to your requirements
split_ratio = (0.8, 0, 0.2)  # Train, Validation, Test

# Perform the split
splitfolders.ratio(input_folder, output=output_folder, seed=88, ratio=split_ratio, group_prefix=None)

print("Splitting complete.")
