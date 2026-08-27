"""Allow running the spectrum viewer with: python -m spectrum_viewer [data_folder]"""

import sys
from .main import run_viewer

data_folder = sys.argv[1] if len(sys.argv) > 1 else None
run_viewer(data_folder)
