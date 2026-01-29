import os

folder1 = r"/Users/yifeigu/Downloads/2025_12_15/spectra"
folder2 = r"/Users/yifeigu/Downloads/2025_12_15/metadata"
folder3 = r"/Users/yifeigu/Downloads/2025_12_15"

# Parse the sample mapping file
sample_mapping = {}
with open(r"/Users/yifeigu/Downloads/2025_12_15/patient.txt", "r") as f:
    for line in f:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            sample_num = parts[0].split()[-1]
            patient_info = parts[1].split()
            patient_id = patient_info[0]
            staging = patient_info[1]
            sample_mapping[sample_num] = (patient_id, staging)

# for folder in [folder1, folder2]:
#     for filename in os.listdir(folder):
#         # Extract sample number from filename
#         for sample_num, (patient_id, staging) in sample_mapping.items():
#             old_prefix = f"patient_staging_sample_{sample_num}_"
#             if filename.startswith(old_prefix):
#                 new_prefix = f"patient_{patient_id}_staging_{staging}_sample_{sample_num}_date_20251215_"
#                 new_name = new_prefix + filename[len(old_prefix):]
#                 old_path = os.path.join(folder, filename)
#                 new_path = os.path.join(folder, new_name)
#                 os.rename(old_path, new_path)
#                 print(f"Renamed: {filename} → {new_name}")
#                 break

for filename in os.listdir(folder3):
    if filename.endswith(".png"):
        for sample_num, (patient_id, staging) in sample_mapping.items():
            if f"image_{sample_num}." in filename:
                new_name = filename.replace(f"image_{sample_num}.", f"patient_{patient_id}_image_{sample_num}.")
                old_path = os.path.join(folder3, filename)
                new_path = os.path.join(folder3, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_name}")
                break

# for folder in [folder1, folder2]:
#     for filename in os.listdir(folder):
#         if "point" in filename:
#             new_name = filename.replace("point", "_point")
#             old_path = os.path.join(folder, filename)
#             new_path = os.path.join(folder, new_name)
#             os.rename(old_path, new_path)
#             print(f"Renamed: {filename} → {new_name}")