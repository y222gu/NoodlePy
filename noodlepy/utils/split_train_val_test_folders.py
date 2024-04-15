import splitfolders

print("Splitting started.")

# Define the input folder containing all the files
input_folder = r"C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\Raman_DB"

# Define the output folder where train, validation, and test sets will be created
output_folder = r"C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\Raman_DB"

# Define the split ratios (e.g., 80% train, 10% validation, 10% test)
# You can adjust these ratios according to your requirements
split_ratio = (0.8, 0.1, 0.1)  # Train, Validation, Test

# Perform the split
splitfolders.ratio(input_folder, output=output_folder, seed=88, ratio=split_ratio, group_prefix=None)

print("Splitting complete.")
