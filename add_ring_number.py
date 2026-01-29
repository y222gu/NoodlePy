import os
import re

# Folder containing your files
folder = r"/Users/yifeigu/Downloads/2025_12_06_old/metadata"

# rep -> ring mapping from your table
point_to_ring = {
    0: 1, 10: 1, 29: 1, 39: 1,
    1: 2, 11: 2, 28: 2, 38: 2,
    2: 3, 12: 3, 27: 3, 37: 3,
    3: 4, 13: 4, 26: 4, 36: 4,
    4: 5, 14: 5, 25: 5, 35: 5,
    5: 6, 15: 6, 24: 6, 34: 6,
    6: 7, 16: 7, 23: 7, 33: 7,
    7: 8, 17: 8, 22: 8, 32: 8,
    8: 9, 18: 9, 21: 9, 31: 9,
    9: 10,19: 10,20: 10,30: 10,
}

# Regex to find rep number in the filename, e.g. "rep_26"
point_pattern = re.compile(r"point_(\d+)")

for filename in os.listdir(folder):
    # Skip directories
    if not os.path.isfile(os.path.join(folder, filename)):
        continue

    # Skip files that already have a ring tag
    if "_ring_" in filename:
        continue

    m = point_pattern.search(filename)
    if not m:
        continue  # no "rep_xx" in this filename

    point_num = int(m.group(1))

    if point_num not in point_to_ring:
        print(f"rep {point_num} not in mapping, skipping {filename}")
        continue

    ring_num = point_to_ring[point_num]

    # Insert _ring_X right after rep_N
    new_filename = point_pattern.sub(f"point_{point_num}_ring_{ring_num}", filename)

    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_filename)

    os.rename(old_path, new_path)
    print(f"Renamed:\n  {filename}\n  -> {new_filename}")
