"""Plot modules for the spectrum viewer."""

from .scatter_plot import ScatterPlotWidget
from .spectrum_plot import SpectrumPlotWidget
from .loadings_plot import LoadingsPlotWidget
from .hyperspectral_map import HyperspectralMapWidget

__all__ = [
    'ScatterPlotWidget',
    'SpectrumPlotWidget',
    'LoadingsPlotWidget',
    'HyperspectralMapWidget',
]
