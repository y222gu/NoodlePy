# Spectrum Viewer Refactoring - Implementation Progress

## Project Goal
Refactor `utils/spectrum_viewer.py` (2,627 lines) into a modular app package with:
- Hyperspectral mapping per sample
- Multi-method clustering (K-Means, DBSCAN, Decision Tree)
- Experiment folder loading
- PyTorch-compatible data manager

## Target Structure
```
utils/spectrum_viewer/
├── __init__.py
├── main.py                    # Main window
├── data_manager.py            # Wraps RamanRobotDataset
├── widgets/
│   ├── __init__.py
│   ├── preprocessing_tab.py
│   ├── embedding_tab.py
│   ├── clustering_tab.py
│   ├── color_mapping_tab.py
│   ├── hyperspectral_tab.py
│   └── experiment_loader.py
├── plots/
│   ├── __init__.py
│   ├── scatter_plot.py
│   ├── spectrum_plot.py
│   ├── loadings_plot.py
│   └── hyperspectral_map.py
├── utils/
│   ├── __init__.py
│   ├── dark_theme.py
│   ├── custom_widgets.py
│   └── embedding_cache.py
└── clustering/
    ├── __init__.py
    ├── base.py
    ├── dbscan_clusterer.py
    ├── kmeans_clusterer.py
    └── decision_tree_clusterer.py
```

## Implementation Checklist

### Phase 1: Package Structure
- [x] Create utils/spectrum_viewer/ directory
- [x] Create __init__.py files for all subpackages

### Phase 2: Utility Modules
- [x] Create utils/dark_theme.py (apply_dark_theme, style_dark_axes, style_dark_3d_axes)
- [x] Create utils/custom_widgets.py (ScrollableComboBox, SortableTableWidgetItem)
- [x] Create utils/embedding_cache.py (EmbeddingCache class)

### Phase 3: Core Components
- [x] Create data_manager.py (ViewerDataManager wraps RamanRobotDataset)

### Phase 4: Plots
- [x] Create plots/scatter_plot.py (ScatterPlotWidget)
- [x] Create plots/spectrum_plot.py (SpectrumPlotWidget)
- [x] Create plots/loadings_plot.py (LoadingsPlotWidget)
- [x] Create plots/hyperspectral_map.py (HyperspectralMapWidget, SampleMapCanvas)

### Phase 5: Clustering
- [x] Create clustering/base.py (BaseClusterer abstract class)
- [x] Create clustering/kmeans_clusterer.py (KMeansClusterer)
- [x] Create clustering/dbscan_clusterer.py (DBSCANClusterer with iterative outlier detection)
- [x] Create clustering/decision_tree_clusterer.py (DecisionTreeClusterer with feature importance)

### Phase 6: Widgets
- [x] Create widgets/preprocessing_tab.py (PreprocessingTab with filters)
- [x] Create widgets/embedding_tab.py (EmbeddingTab)
- [x] Create widgets/color_mapping_tab.py (ColorMappingTab)
- [x] Create widgets/clustering_tab.py (ClusteringTab with method switching)
- [x] Create widgets/hyperspectral_tab.py (HyperspectralTab with wavenumber slider)
- [x] Create widgets/experiment_loader.py (ExperimentLoaderWidget)

### Phase 7: Main Window
- [x] Create main.py with SpectraViewerApp class
- [x] Create package __init__.py with exports

### Phase 8: Testing & Cleanup
- [x] Test imports (basic Python imports) - ALL PASS
- [ ] Test with actual data folder (requires GUI)
- [ ] Verify all features work (requires GUI)
- [x] Keep old spectrum_viewer.py as backup (preserved)

## Files Created (23 total)

| File | Description | Status |
|------|-------------|--------|
| `__init__.py` | Package exports | Done |
| `main.py` | Main application entry point | Done |
| `data_manager.py` | ViewerDataManager class | Done |
| `utils/__init__.py` | Utils package exports | Done |
| `utils/dark_theme.py` | Dark theme styling | Done |
| `utils/custom_widgets.py` | Custom Qt widgets | Done |
| `utils/embedding_cache.py` | Embedding computation/caching | Done |
| `plots/__init__.py` | Plots package exports | Done |
| `plots/scatter_plot.py` | Scatter plot with lasso | Done |
| `plots/spectrum_plot.py` | Spectrum line plot | Done |
| `plots/loadings_plot.py` | PCA loadings plot | Done |
| `plots/hyperspectral_map.py` | Spatial intensity maps | Done |
| `clustering/__init__.py` | Clustering package exports | Done |
| `clustering/base.py` | Base clusterer interface | Done |
| `clustering/dbscan_clusterer.py` | DBSCAN implementation | Done |
| `clustering/kmeans_clusterer.py` | K-Means implementation | Done |
| `clustering/decision_tree_clusterer.py` | Decision Tree clustering | Done |
| `widgets/__init__.py` | Widgets package exports | Done |
| `widgets/preprocessing_tab.py` | Preprocessing controls | Done |
| `widgets/embedding_tab.py` | Dim reduction controls | Done |
| `widgets/color_mapping_tab.py` | Color mapping controls | Done |
| `widgets/clustering_tab.py` | Multi-method clustering | Done |
| `widgets/hyperspectral_tab.py` | Hyperspectral mapping | Done |
| `widgets/experiment_loader.py` | Experiment folder loader | Done |

## Key Design Decisions
1. ViewerDataManager wraps RamanRobotDataset for PyTorch compatibility
2. Clustering methods: K-Means, DBSCAN, Decision Tree with feature importance
3. Dark theme throughout (extracted to utils/dark_theme.py)
4. Hyperspectral maps grouped by sample metadata field
5. Signals/slots used for component communication
6. Original spectrum_viewer.py preserved as backup

## Usage

### Run the viewer
```python
from utils.spectrum_viewer import run_viewer
run_viewer("/path/to/data/folder")
```

### Or import components
```python
from utils.spectrum_viewer import SpectraViewerApp, ViewerDataManager
from utils.spectrum_viewer.clustering import KMeansClusterer
```

### Command line
```bash
python -m utils.spectrum_viewer.main --data /path/to/data
```

