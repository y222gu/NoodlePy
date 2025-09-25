import os
import pandas as pd
from typing import List, Optional


def write_median_spectrum(data_folder: str, filename: str) -> None:
    """
    Reads a comma-separated file containing repeated spectral measurements,
    computes the median intensity at each unique wavenumber, and overwrites
    the file with the median spectrum in-place.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the spectrum file.
    filename : str
        Name of the spectrum file (e.g., 'sample.csv' or 'sample.txt').
    """
    filepath = os.path.join(data_folder, filename)

    # Skip empty files
    if not os.path.isfile(filepath) or os.stat(filepath).st_size == 0:
        return

    # Load data: two columns, no header
    df = pd.read_csv(filepath, sep=",", header=None,
                     names=["wavenumber", "intensity"])

    # Compute median intensity at each unique wavenumber
    median_df = (
        df.groupby("wavenumber", as_index=False)
          ["intensity"]
          .median()
          .round(3)
    )

    # Overwrite the original file with median spectrum
    median_df.to_csv(
        filepath,
        sep=",",
        header=False,
        index=False,
        float_format="%.3f"
    )


def process_all_spectra(
    root_folder: str,
    extensions: Optional[List[str]] = None
) -> None:
    """
    Recursively traverses `root_folder` and applies `write_median_spectrum`
    to all files matching the given extensions.

    Parameters
    ----------
    root_folder : str
        Path to the top-level directory to search.
    extensions : list of str, optional
        List of file extensions to process (e.g., ['.txt', '.csv']).
        If None, defaults to ['.txt', '.csv'].
    """
    if extensions is None:
        extensions = ['.txt', '.csv']

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in extensions):
                try:
                    write_median_spectrum(dirpath, fname)
                except Exception as e:
                    # Log or handle files that fail to process
                    print(f"Failed to process {os.path.join(dirpath, fname)}: {e}")


if __name__ == "__main__":
    # Example usage
    root_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_test")
    process_all_spectra(root_folder)
