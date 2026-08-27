#!/usr/bin/env python3
"""
Script to rename spectra and metadata files with correct patient numbers and staging.
Reads sample_order.txt to map sample numbers to patient IDs and staging (cancer/control).

Usage:
    python rename_files.py [DATA_FOLDER] [--dry-run]

Examples:
    python rename_files.py                           # Use current directory, actually rename
    python rename_files.py --dry-run                 # Use current directory, preview only
    python rename_files.py /path/to/data             # Use specified folder, actually rename
    python rename_files.py /path/to/data --dry-run   # Use specified folder, preview only
"""

import argparse
import os
import re
from pathlib import Path


def parse_sample_order(filepath):
    """Parse sample_order.txt to get mapping of sample number to (patient_id, staging)."""
    sample_mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: sample_X: patient_id, staging
            match = re.match(r'sample_(\d+):\s*(\d+),\s*(\w+)', line)
            if match:
                sample_num = int(match.group(1))
                patient_id = match.group(2)
                staging = match.group(3)
                sample_mapping[sample_num] = (patient_id, staging)
    return sample_mapping


def extract_date_from_folder(folder_path):
    """
    Extract date from folder name.
    E.g., '2026_02_03_1' -> '20260203'
         '2026_02_02' -> '20260202'
    """
    folder_name = Path(folder_path).name
    # Match patterns like 2026_02_03 or 2026_02_03_1 (with optional suffix)
    match = re.match(r'^(\d{4})_(\d{2})_(\d{2})(?:_\d+)?$', folder_name)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return None


def rename_file(old_path, sample_mapping, folder_date=None, dry_run=True):
    """
    Rename a single file from old format to new format.

    Handles these formats:
    - date_DATE_patient_staging_sample_X_...
    - DATE_patient_staging_sample_X_...
    - patient_staging_sample_X_... (date extracted from folder name)

    Target format:
    date_20260202_patient_317_staging_control_sample_0_...
    """
    filename = os.path.basename(old_path)
    dirname = os.path.dirname(old_path)

    # Extract sample number from filename
    sample_match = re.search(r'sample_(\d+)', filename)
    if not sample_match:
        print(f"  Skipping (no sample number found): {filename}")
        return None

    sample_num = int(sample_match.group(1))

    # Look up patient info
    if sample_num not in sample_mapping:
        print(f"  Skipping (sample {sample_num} not in mapping): {filename}")
        return None

    patient_id, staging = sample_mapping[sample_num]

    # Skip files that are already properly renamed (have actual patient number)
    # Check if filename already has patient_<number>_staging_<word> pattern
    already_renamed = re.search(r'patient_\d+_staging_\w+_sample_', filename)
    if already_renamed:
        print(f"  Skipping (already renamed): {filename}")
        return None

    # Try to match different patterns
    new_filename = None

    # Pattern 1: date_DATE_patient_staging_sample_X_...
    # Date can be 8+ digits (e.g., 20260202 or 202602052 with suffix)
    pattern1 = r'^date_(\d+)_patient_staging_(sample_\d+_.*)$'
    match1 = re.match(pattern1, filename)
    if match1:
        date_str = match1.group(1)
        rest_of_filename = match1.group(2)
        new_filename = f"date_{date_str}_patient_{patient_id}_staging_{staging}_{rest_of_filename}"

    # Pattern 1b: datae_DATE_patient_staging_sample_X_... (typo: datae instead of date)
    if not new_filename:
        pattern1b = r'^datae_(\d+)_patient_staging_(sample_\d+_.*)$'
        match1b = re.match(pattern1b, filename)
        if match1b:
            date_str = match1b.group(1)
            rest_of_filename = match1b.group(2)
            new_filename = f"date_{date_str}_patient_{patient_id}_staging_{staging}_{rest_of_filename}"

    # Pattern 2: DATE_patient_staging_sample_X_... (no date_ prefix)
    # Date can be 8+ digits
    if not new_filename:
        pattern2 = r'^(\d+)_patient_staging_(sample_\d+_.*)$'
        match2 = re.match(pattern2, filename)
        if match2:
            date_str = match2.group(1)
            rest_of_filename = match2.group(2)
            new_filename = f"date_{date_str}_patient_{patient_id}_staging_{staging}_{rest_of_filename}"

    # Pattern 3: patient_staging_sample_X_... (no date in filename, use folder date)
    if not new_filename:
        pattern3 = r'^patient_staging_(sample_\d+_.*)$'
        match3 = re.match(pattern3, filename)
        if match3 and folder_date:
            rest_of_filename = match3.group(1)
            new_filename = f"date_{folder_date}_patient_{patient_id}_staging_{staging}_{rest_of_filename}"
        elif match3 and not folder_date:
            print(f"  Skipping (no date in filename and couldn't extract from folder): {filename}")
            return None

    if not new_filename:
        print(f"  Skipping (doesn't match expected pattern): {filename}")
        return None

    new_path = os.path.join(dirname, new_filename)

    if dry_run:
        print(f"  Would rename: {filename}")
        print(f"           to: {new_filename}")
    else:
        os.rename(old_path, new_path)
        print(f"  Renamed: {filename}")
        print(f"       to: {new_filename}")

    return new_path


def process_folder(folder_path, sample_mapping, folder_date=None, dry_run=True):
    """Process all .txt files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder does not exist: {folder_path}")
        return 0

    txt_files = list(folder.glob("*.txt"))
    print(f"\nProcessing {len(txt_files)} files in {folder_path}")

    renamed_count = 0
    for filepath in sorted(txt_files):
        result = rename_file(str(filepath), sample_mapping, folder_date=folder_date, dry_run=dry_run)
        if result:
            renamed_count += 1

    print(f"{'Would rename' if dry_run else 'Renamed'} {renamed_count} files")
    return renamed_count


def main():
    parser = argparse.ArgumentParser(
        description='Rename spectra and metadata files with correct patient numbers and staging.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python rename_files.py                           # Use current directory, actually rename
    python rename_files.py --dry-run                 # Use current directory, preview only
    python rename_files.py /path/to/data             # Use specified folder, actually rename
    python rename_files.py /path/to/data --dry-run   # Use specified folder, preview only
        """
    )
    parser.add_argument(
        'data_folder',
        nargs='?',
        default='.',
        help='Path to the data folder (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually renaming files'
    )

    args = parser.parse_args()

    # Resolve data folder path
    base_dir = Path(args.data_folder).resolve()
    sample_order_file = base_dir / "sample_order.txt"
    spectra_dir = base_dir / "spectra"
    metadata_dir = base_dir / "metadata"

    dry_run = args.dry_run

    print("=" * 60)
    print("File Renaming Script")
    print("=" * 60)
    print(f"Data folder: {base_dir}")
    print(f"Dry run: {dry_run}")
    if dry_run:
        print("\n*** DRY RUN MODE - No files will be renamed ***")
        print("*** Remove --dry-run flag to perform actual renaming ***\n")

    # Check if sample_order.txt exists
    if not sample_order_file.exists():
        print(f"\nError: sample_order.txt not found at {sample_order_file}")
        return 1

    # Parse sample order
    print(f"\nReading sample order from: {sample_order_file}")
    sample_mapping = parse_sample_order(sample_order_file)
    print(f"Found {len(sample_mapping)} sample mappings:")
    for sample_num in sorted(sample_mapping.keys()):
        patient_id, staging = sample_mapping[sample_num]
        print(f"  sample_{sample_num}: patient {patient_id}, {staging}")

    # Extract date from folder name (for files without date in filename)
    folder_date = extract_date_from_folder(base_dir)
    if folder_date:
        print(f"\nExtracted date from folder name: {folder_date}")
    else:
        print(f"\nNote: Could not extract date from folder name '{base_dir.name}'")
        print("Files without date in filename will be skipped.")

    total_renamed = 0

    # Process spectra folder
    if spectra_dir.exists():
        total_renamed += process_folder(spectra_dir, sample_mapping, folder_date=folder_date, dry_run=dry_run)
    else:
        print(f"\nSpectra folder not found: {spectra_dir}")

    # Process metadata folder
    if metadata_dir.exists():
        total_renamed += process_folder(metadata_dir, sample_mapping, folder_date=folder_date, dry_run=dry_run)
    else:
        print(f"\nMetadata folder not found: {metadata_dir}")

    print("\n" + "=" * 60)
    if dry_run:
        print(f"DRY RUN COMPLETE - Would rename {total_renamed} files total")
        print("Remove --dry-run flag to perform actual renaming")
    else:
        print(f"RENAMING COMPLETE - Renamed {total_renamed} files total")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
