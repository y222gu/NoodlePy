# RamanPy

A metadata-centric, spatially aware analysis and visualization pipeline for Raman spectroscopy data collected from dried blood plasma droplets.

## Features

- **Spectrum Viewer** - Interactive PyQt5 application for visualizing and exploring Raman spectra
  - Dimensionality reduction (PCA, t-SNE, UMAP)
  - Clustering (K-Means, DBSCAN, Decision Tree)
  - Hyperspectral mapping with grid, ring, and scatter layouts
  - Metadata filtering with boolean expressions (AND/OR/NOT)
  - Color mapping by metadata, cluster labels, or spectral statistics
  - Classification with integrated ML models
- **Preprocessing** - Cropping, baseline correction (airPLS), cosmic ray removal, normalization, and smoothing
- **Augmentation** - Configurable spectral augmentation for training data
- **PyTorch Dataset** - ML-ready dataset with metadata integration and patient demographics
- **ML Models** - Standalone PyTorch model definitions (not yet integrated into the pipeline)

## Project Structure

```
RamanPy/
├── utils/                      # Core modules
│   ├── spectrum.py             #   Core Spectrum class
│   ├── spectrumpreprocessor.py #   Preprocessing pipeline
│   ├── spectrumaugmentor.py    #   Data augmentation
│   └── raman_robot_dataset.py  #   PyTorch dataset
├── rename_files.py             # Utility to rename spectra/metadata files
├── config/                     # YAML configuration files
├── spectrum_viewer/            # Interactive viewer application
│   ├── main.py                 #   App entry point
│   ├── data_manager.py         #   Data loading and management
│   ├── widgets/                #   UI tabs (preprocessing, clustering, etc.)
│   ├── plots/                  #   Plot widgets (scatter, spectrum, hyperspectral)
│   ├── clustering/             #   Clustering algorithms
│   └── utils/                  #   Dark theme, custom widgets, caching
├── ml/                         # Standalone model definitions (not integrated)
│   ├── backbone.py             #   CNN feature extractor
│   ├── simsiam.py              #   Self-supervised learning (PyTorch Lightning)
│   ├── raman_autoencoder.py    #   Autoencoder
│   ├── classifier.py           #   Classification head
│   ├── positionalcnn.py        #   Position-aware CNN
│   └── traintestmodel.py       #   Training/evaluation utilities (wandb)
└── analysis/                   # Standalone analysis & figure scripts
    ├── plot_selected_spectra.py
    ├── plot_preprocessed_spectra.py
    ├── plot_hyperspectral_map.py
    ├── plot_multi_sample_maps.py
    ├── plot_clustered_spectra.py
    └── create_hyperspectral_gif.py
```

## Installation

### Requirements

- Python 3.9+
- PyQt5
- PyTorch
- NumPy, SciPy, pandas, scikit-learn
- pybaselines
- UMAP (`umap-learn`)
- PyYAML

Install dependencies:

```bash
pip install numpy scipy pandas scikit-learn pybaselines PyQt5 torch umap-learn pyyaml matplotlib
```

## Usage

### Spectrum Viewer

```bash
# Launch with a file browser to select data
python -m spectrum_viewer

# Launch with a specific data folder
python -m spectrum_viewer /path/to/experiment/data
```

Or from Python:

```python
from spectrum_viewer import run_viewer

run_viewer("/path/to/data")
```

### Loading Spectra Programmatically

```python
from utils.spectrum import Spectrum
from utils.spectrumpreprocessor import SpectrumPreprocessor

# Load a spectrum from file
spectrum = Spectrum.load("path/to/spectrum.txt")

# Preprocess
preprocessor = SpectrumPreprocessor(
    cropping=True,
    baseline_correction=True,
    normalization=True,
    config_path="config/config_default.yml"
)
processed = preprocessor.preprocess(spectrum)
```

### PyTorch Dataset

```python
from utils.raman_robot_dataset import RamanRobotDataset

dataset = RamanRobotDataset(data_folder="/path/to/data")
```

## Data Format

Spectra are stored as `.txt` files with two columns: wavelength and intensity. Metadata is encoded in filenames and companion metadata files. See `CLAUDE.md` for the full data model specification.

## Configuration

Preprocessing and augmentation parameters are defined in YAML config files under `config/`. The default configuration is `config/config_default.yml`.
