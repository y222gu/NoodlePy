"""
Spectrum Viewer - A modular application for Raman spectroscopy data visualization.

This package provides:
- Interactive visualization of Raman spectra
- Dimensionality reduction (PCA, T-SNE, UMAP)
- Clustering (K-Means, DBSCAN, Decision Tree)
- Hyperspectral mapping
- Metadata filtering and exploration

Usage:
    from utils.spectrum_viewer import run_viewer, SpectraViewerApp

    # Run directly
    run_viewer("/path/to/data")

    # Or create app programmatically
    app = SpectraViewerApp(data_folder="/path/to/data")
"""

from .main import SpectraViewerApp, run_viewer
from .data_manager import ViewerDataManager

# Re-export useful components for external use
from .utils.dark_theme import apply_dark_theme, DEFAULT_BLUE
from .utils.embedding_cache import EmbeddingCache
from .clustering import (
    BaseClusterer,
    KMeansClusterer,
    DBSCANClusterer,
    DecisionTreeClusterer,
)

__all__ = [
    # Main application
    'SpectraViewerApp',
    'run_viewer',

    # Data management
    'ViewerDataManager',

    # Utilities
    'apply_dark_theme',
    'DEFAULT_BLUE',
    'EmbeddingCache',

    # Clustering
    'BaseClusterer',
    'KMeansClusterer',
    'DBSCANClusterer',
    'DecisionTreeClusterer',
]

__version__ = '3.0.0'
