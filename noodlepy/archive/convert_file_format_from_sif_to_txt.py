import os
import glob

def convert_all_asc_in_directory(directory):
    last_empty_line_index_list = []
    num_lines_to_write_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.asc'):
                input_file = os.path.join(root, file)
                output_file = os.path.splitext(input_file)[0] + '.txt'

                try:
                    with open(input_file, 'r') as asc_file:
                        data = asc_file.read()

                        data_lines = data.splitlines()
                        last_empty_line_index = -1
                        for i, line in enumerate(data_lines):
                            if line.strip() == '':
                                last_empty_line_index = i
                        print(f"Index of the last empty line: {last_empty_line_index}")
                        last_empty_line_index_list.append(last_empty_line_index)

                    if last_empty_line_index != -1:
                        data_to_write = '\n'.join(data_lines[last_empty_line_index + 1:])
                    else:
                        data_to_write = data

                    # Replace tabs or spaces with commas within each line
                    data_to_write = '\n'.join([','.join(line.split()) for line in data_to_write.splitlines()])

                    num_lines_to_write = len(data_to_write.splitlines())
                    print(f"Number of lines to write: {num_lines_to_write}")
                    num_lines_to_write_list.append(num_lines_to_write)

                    with open(output_file, 'w') as txt_file:
                        txt_file.write(data_to_write)
                    
                except Exception as e:
                    print(f"An error occurred: {e}")

    # find the unique last empty line indices
    last_empty_line_index_list = list(set(last_empty_line_index_list))
    print(f"Unique last empty line indices: {last_empty_line_index_list}")

    # find the unique number of lines to write
    num_lines_to_write_list = list(set(num_lines_to_write_list))
    print(f"Unique number of lines to write: {num_lines_to_write_list}")

if __name__ == "__main__":
    directory = r'C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data\Plasma_1'
    convert_all_asc_in_directory(directory)