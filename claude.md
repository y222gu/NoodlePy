# Claude Code Guidelines

## Raman Spectroscopy Data Analysis & Visualization Pipeline

---

## 1. Project Goal

Build a **scalable, unified analysis pipeline** for Raman spectroscopy data collected from dried blood plasma droplets, enabling:

* Robust metadata integration across experiments
* Spatially aware spectral visualization
* Flexible filtering and clustering
* Machine learning–ready datasets for cancer classification

The system must handle **heterogeneous folder structures, naming conventions, sampling patterns, and metadata availability**.

---

## 2. Data Model Overview

### 2.1 Patients (Global Metadata)

* Hundreds of patients: cancer + control
* Demographics stored in a *xlsx file**, including (but not limited to):

  * Patient ID
  * Cancer stage
  * Age
  * BMI
  * Gender
  * Race
* This metadata must be **joinable** with spectral data using patient identifiers.

---

### 2.2 Experiments

* Data organized by **experiment date folders**
* Each experiment folder may contain:

  * `captured_image_X`
  * `masked_image_X`
  * `sampled_masked_image_X`
  * `spectra/`
  * `metadata/`

⚠️ Folder contents are **not guaranteed to be consistent** across experiments.

---

### 2.3 Samples

* Each patient may have **multiple plasma droplets**
* Each droplet:

  * Is typically circular (but not guaranteed)
  * Is imaged on a quartz slide
  * Has **one or more associated images**
  * Has **multiple Raman point measurements**

---

### 2.4 Spectra Files

* Stored as `.txt` files
* Two columns:

  * Column 1: Wavelength
  * Column 2: Intensity
* Files may contain **multiple concatenated spectra**

  * All spectra in a file share the same wavelength range
* Spectral metadata is partially encoded in filenames:

  * Patient number
  * Sample number
  * Rep number
  * Ring number
  * Grid row / column
  * Point index
  * Experiment ID

---

### 2.5 Per-Spectrum Experimental Metadata

Metadata recorded per measurement location includes:

* Exposure time
* Laser power
* Timestamp
* (x, y) location **relative to the image center**
* Prusa focus position (coarse Z)
* Nanodrive focus position (fine Z)

---

## 3. Critical Caveats (Must Be Handled Explicitly)

1. **Image center ≠ droplet center**

   * All (x, y) positions are relative to the image center only
   * Droplet center must be inferred (e.g., from masks)
   * Distance of each sampling point to the edge of the sample mask must be calculated.

2. **File naming mismatches**

   * `sample_0` spectra corresponds to:

     * `captured_image_1`
     * `masked_image_1`
     * `sampled_masked_image_1`

3. **Inconsistent sampling patterns**

   * Pattern should be indicated by the experimental metadata in their file name:
     * Grid scans (row&cloumn)
     * Ring scans (point&ring)

4. **Variable completeness**

   * Different experiments may have:
     * Missing metadata
     * Different numbers of spectra
     * Different imaging availability

---

## 4. Dataset Architecture (PyTorch)

### 4.1 Dataset Responsibilities

The PyTorch dataset must:

* Load spectra **without breaking existing functionality**
* Preserve all current preprocessing and augmentation features
* Integrate metadata from:

  * Spectra filenames
  * Per-spectrum metadata files
  * Experiment-level context
  * Patient demographics xlsx

---

## 5. Spectra Viewer Enhancements

### 5.1 Metadata Unification

The viewer must support **simultaneous visualization** of:

* Raw/processed spectra
* Spatial location on droplet
* all metadata
* Cluster labels

All views must reference the **same underlying data objects**.

---

### 5.2 Hyperspectral Mapping

Add a new visualization tab:

#### Requirements

* One hyperspectral map **per droplet sample**
* Maps display intensity at a **selected wavelength**
* A **global wavelength slider** updates all maps simultaneously
* Maps must respect the **sampling pattern**:

  * Grid → 2D grid map
  * Ring → polar or radial layout
  * Irregular → scatter interpolation
* If multiple samples exist:

  * Display in vertically stacked panels
  * Enable vertical scrolling

---

### 5.3 Spatial Awareness

* Measurement locations must be overlaid on:

  * Droplet mask
  * Or inferred droplet boundary
* Distance-to-edge calculation must be:
  * Pattern-aware

---

## 6. Clustering & Learning Methods

Replace DBSCAN with selectable methods:

### Supported Methods

* K-Means
* Decision Tree–based clustering
* Fully Connected Neural Network (FCN)

### Visualization Integration

* Cluster labels must:

  * Color reduced-dimension scatter plots
  * Color spectral line plots
  * Update dynamically when method changes

---

## 7. UI & Visual Design

* **Dark theme only**

  * Dark background
  * Light text
  * High-contrast color maps
* UI must scale to:

  * Large patient cohorts
  * Dozens of samples per experiment
  * can change and load data folder

---


## 8. Filtering System Upgrade

Replace the current filter logic with a **boolean expression system**:

### Requirements

* Support:

  * AND
  * OR
  * NOT
* Filters may apply to:

  * Demographics
  * Experimental metadata
  * Spectral statistics
  * Cluster labels
* Filters must be:

  * Chainable
  * Persist across views
  * Reflected consistently in all plots

---

## 9. Constraints & Non-Goals

* Do **not** remove existing features
* Optimize and refactor only when functionality is preserved
* No assumptions about:

  * Folder completeness
  * Sampling regularity
  * Metadata availability

---

## 10. Summary for Claude Code

**Primary directive**:

> Build a metadata-centric, spatially aware Raman spectroscopy pipeline that unifies heterogeneous experimental data into a single, extensible analysis and visualization framework.

This system must remain robust to inconsistency, scalable to large cohorts, and visually interpretable for scientific exploration and machine learning.
