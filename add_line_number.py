import os
import re

# Folder containing your files
folder = r"/Users/yifeigu/Downloads/2025_12_06/spectra"

point_to_line = {
    **{i: 1 for i in range(0, 10)},
    **{i: 2 for i in range(10, 20)},
    **{i: 3 for i in range(20, 30)},
    **{i: 4 for i in range(30, 40)},
}

# Regex to find rep number in the filename, e.g. "rep_26"
point_pattern = re.compile(r"point_(\d+)")

for filename in os.listdir(folder):
    # Skip directories
    if not os.path.isfile(os.path.join(folder, filename)):
        continue

    # Skip files that already have a line tag
    if "_line_" in filename:
        continue

    m = point_pattern.search(filename)
    if not m:
        continue  # no "rep_xx" in this filename

    point_num = int(m.group(1))

    if point_num not in point_to_line:
        print(f"rep {point_num} not in mapping, skipping {filename}")
        continue

    line_num = point_to_line[point_num]

    # Insert _line_X right after point_N
    new_filename = point_pattern.sub(f"point_{point_num}_line_{line_num}", filename)

    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_filename)

    os.rename(old_path, new_path)
    print(f"Renamed:\n  {filename}\n  -> {new_filename}")
