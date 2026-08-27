"""Widget modules for the spectrum viewer."""

from .preprocessing_tab import PreprocessingTab
from .embedding_tab import EmbeddingTab
from .clustering_tab import ClusteringTab
from .color_mapping_tab import ColorMappingTab
from .hyperspectral_tab import HyperspectralTab
from .experiment_loader import ExperimentLoaderWidget
from .hidden_spectra_widget import HiddenSpectraWidget
from .classification_tab import ClassificationTab
from .filter_tab import FilterTab

__all__ = [
    'PreprocessingTab',
    'EmbeddingTab',
    'ClusteringTab',
    'ColorMappingTab',
    'HyperspectralTab',
    'ExperimentLoaderWidget',
    'HiddenSpectraWidget',
    'ClassificationTab',
    'FilterTab',
]
