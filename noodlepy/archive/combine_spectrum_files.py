import os
from pathlib import Path

def combine_spectrum_files(base_dir):
    """
    Combine all spectrum .txt files in each subfolder into a single file
    named as foldername_spectrum.txt, saved one level up from each subfolder.
    """
    base_path = Path(base_dir)
    
    # Check if base directory exists
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Iterate through each subfolder in the base directory
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            # Get all .txt files that match pattern
            spectrum_files = sorted(subfolder.glob("*.txt"))
            
            if spectrum_files:
                # Create output filename and save one level up (parent of subfolder)
                output_filename = f"{subfolder.name}_spectrum.txt"
                output_path = subfolder.parent / output_filename
                
                print(f"\nProcessing folder: {subfolder.name}")
                print(f"Found {len(spectrum_files)} spectrum files")
                print(f"Saving combined file to: {output_path}")
                
                # Combine all spectrum files
                with open(output_path, 'w') as outfile:
                    for i, spectrum_file in enumerate(spectrum_files, 1):
                        print(f"  Adding: {spectrum_file.name}")
                        
                        # Read and write the content
                        with open(spectrum_file, 'r') as infile:
                            content = infile.read()
                            outfile.write(content)
                
                print(f"✓ Created: {output_filename} at {output_path}")
            else:
                print(f"\nNo spectrum files found in {subfolder.name}")

if __name__ == "__main__":
    # Set your base directory here
    base_directory = r'/Users/yifeigu/Downloads/Air Lens Samples'
    
    print("Starting spectrum file combination...")
    print(f"Base directory: {base_directory}\n")
    
    combine_spectrum_files(base_directory)
    
    print("\n✓ Process completed!")