import json
from pathlib import Path

# ---- paths ----
json_path = r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/outliers.json"          # JSON file you showed
data_dir = Path(r"/Users/yifeigu/Documents/Carney_Lab/Data/RamanRobot/2025_12_10_copy/spectra")   # directory containing spectrum .txt files

# ---- load JSON ----
with open(json_path, "r") as f:
    outliers = json.load(f)

# ---- collect outlier filenames ----
outlier_files = {item["spectrum_file"] for item in outliers}

# ---- delete files ----
deleted = 0
for file_name in outlier_files:
    file_path = data_dir / file_name
    if file_path.exists():
        file_path.unlink()
        deleted += 1
    else:
        print(f"File not found: {file_name}")

print(f"Deleted {deleted} outlier files.")
