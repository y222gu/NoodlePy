# Specify the file path
file_path = "/Users/yifeigu/Documents/Carney_Lab/Data/Plasma/20230323 - cancer 398/20230323_398_plasma_60x_1mm_65mW_cw-853_grating3_quartz_24.asc"

# Initialize variables
metadata_lines = []
data_lines = []
data_started = False

try:
    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            if not data_started:
                # Check if the line is empty, indicating the end of metadata
                if line.strip() == "":
                    data_started = True
                else:
                    metadata_lines.append(line.strip())
            else:
                data_lines.append(line.strip())

    # Process metadata and data
    metadata = "\n".join(metadata_lines)
    data = "\n".join(data_lines)

    # Print metadata and data
    print("Metadata:")
    print(metadata)

except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
