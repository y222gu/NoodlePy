import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from airPLS import airPLS
from enum import Enum
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px

class Normalization(str, Enum):
    """Enum class for different normalization methods."""
    minmax = "minmax"
    vector = "vector"
    zscore = "zscore"
    
    @classmethod
    def values(cls):
        """Returns a list of all normalization methods."""
        return list(cls)

    def apply(self, corrected_intensity: np.ndarray) -> np.ndarray:
        """Applies the selected normalization method."""
        if self == Normalization.minmax:
            return (corrected_intensity - np.min(corrected_intensity)) / (np.max(corrected_intensity) - np.min(corrected_intensity))
        elif self == Normalization.vector:
            return corrected_intensity / np.linalg.norm(corrected_intensity)
        elif self == Normalization.zscore:
            return (corrected_intensity - np.mean(corrected_intensity)) / np.std(corrected_intensity)
        else:
            raise ValueError(f"Invalid normalization method: {self}")

class RamanPreprocessor:
    """Class for preprocessing Raman or FTIR spectra, including cosmic ray removal, smoothing, baseline correction, and normalization."""

    def __init__(self, spectra: List[np.ndarray], labels_array, patient_codes_array, param_dict: Dict):
        """
        Initialize RamanPreprocessor with a list of spectra and parameters.

        Args:
            spectra (List[np.ndarray]): List of 2D arrays, each containing [wavenumber, intensity].
            param_dict (Dict): Dictionary containing preprocessing parameters.
        """
        self.spectra = spectra
        self.params = param_dict
        self.labels = labels_array
        self.patient_codes = patient_codes_array
        self.averaged_spectra = []
        self.averaged_labels = []
        self.averaged_patient_codes = []
        self.preprocessed_spectra = []
        self.nth = self.params.get("nth", 1)  # Default to 1 (no averaging)
        self.plot = self.params.get("plot_spectra", False)
        self.plot_cleaned_spectra = self.params.get("plot_cleaned_spectra", False)
        self.n_iterations = param_dict.get("n_outlier_iterations", 1)  # Number of PCA-DBSCAN filtering cycles
        self.remove_outliers = param_dict.get("remove_outliers", False)
        self.dbscan_eps = param_dict.get("dbscan_eps", 0.5)  # DBSCAN epsilon parameter
        
    def preprocess(self) -> List[np.ndarray]:
        """Preprocess all spectra in the batch."""
    
        processed_spectra = []
        processed_labels = []
        processed_codes = []

        total_spectra = len(self.spectra)
        num_groups = total_spectra // self.nth  # Number of groups to average

        for i in range(num_groups):
            start_idx = i * self.nth
            end_idx = start_idx + self.nth
 
            # Ensure we don't exceed array bounds
            if end_idx > total_spectra:
                break

            # Extract wavenumbers from the first spectrum (assuming they are identical for all)
            wavenumbers = self.spectra[start_idx][:, 0]

            # Average the intensities across the nth spectra
            intensities = np.array([self.spectra[j][:, 1] for j in range(start_idx, end_idx)])
            avg_intensity = np.mean(intensities, axis=0)

            self.averaged_spectra.append(np.column_stack((wavenumbers, avg_intensity)))
            self.averaged_labels.append(self.labels[start_idx]) 
            self.averaged_patient_codes.append(self.patient_codes[start_idx])

        if self.plot: colors = plt.get_cmap("jet")(np.linspace(0, 1, len(self.averaged_spectra)))
        
        for idx, spectrum in enumerate(self.averaged_spectra):
            try:
                wavenumber, processed_intensity = self.preprocess_single_spectrum(spectrum)                 
                processed_spectra.append(np.column_stack((wavenumber, processed_intensity)))
                processed_labels.append(self.averaged_labels[idx])  # Keep label if successful
                processed_codes.append(self.averaged_patient_codes[idx])
                # if self.plot: self.plot_spectrum(idx, wavenumber, processed_intensity, colors[idx])
            except ValueError as e:
                print(f"Error processing spectrum {idx}: {e}")
                continue

        # Convert to NumPy matrix for PCA
        data_matrix = np.array([spectrum[:, 1] for spectrum in processed_spectra])

        # Perform Iterative Outlier Removal
        if self.remove_outliers:
            data_matrix, processed_labels, processed_codes = self.iterative_outlier_removal(wavenumber, data_matrix, processed_labels, processed_codes)

        if self.plot: self.plot_spectra(wavenumber, data_matrix, processed_codes) 

        return data_matrix, np.array(processed_labels), np.array(processed_codes)

    def iterative_outlier_removal(self, wavenumber: np.ndarray, data_matrix: np.ndarray, labels: List[int], patient_codes: List[str]) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Runs iterative PCA + DBSCAN clustering to remove outliers, with persistence.

        Args:
            data_matrix (np.ndarray): Intensity matrix.
            labels (List[int]): Labels corresponding to each spectrum.
            patient_codes (List[str]): Patient IDs corresponding to each spectrum.

        Returns:
            Tuple[np.ndarray, List[int], List[str]]: Filtered data, labels, and patient codes.
        """

        removed_indices = set()

        for iteration in range(self.n_iterations):

            print(f"Running PCA + DBSCAN Outlier Removal - Iteration {iteration+1}/{self.n_iterations}")

            # Step 1: PCA Dimensionality Reduction
            pca_data = PCA().fit_transform(data_matrix)

            pcstart = 0
            pcend = 1
            # Choose which PCs to use for clustering based on the iteration
            if iteration == 0:
                pca_data = pca_data[:, :2]  # Use PC1 & PC2
            elif iteration == 1:
                pcstart = 3
                pcend = 5
                pca_data = pca_data[:, pcstart:pcend]  # Use PC2 & PC3
            else:
                pcstart = 3
                pcend = 5
                pca_data = pca_data[:, pcstart:pcend]  # Use PC2 & PC3

            # Step 2: Apply DBSCAN Clustering
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=5).fit(pca_data)
            cluster_labels = dbscan.labels_  # DBSCAN assigns -1 to noise

            # Step 3: Identify outliers (cluster -1 = noise)
            outlier_indices = {idx for idx, cluster in enumerate(cluster_labels) if cluster == -1}

            print("Size of pca_data:", pca_data.shape)
            # Save the indices of removed outliers
            df = pd.DataFrame({
                "PCA 1": pca_data[:, 0],
                "PCA 2": pca_data[:, 1],
                "Cluster": cluster_labels
            })

            # Plot the clusters
            fig = px.scatter(df, x="PCA 1", y="PCA 2", color=df["Cluster"].astype(str),
                            title="PCA of Raman Spectra with DBSCAN Clusters",
                            labels={"Cluster": "Cluster Label"})
            
            # Save the plot
            fig.write_html(f"pca_dbscan_clusters_run_{iteration}.html")
            print("PCA plot with DBSCAN clusters saved as pca_dbscan_clusters.html. Open it manually.")

            # Remove previously saved indices
            outlier_indices.update(removed_indices)

            # Step 4: Filter Data
            mask = np.array([i not in outlier_indices for i in range(len(data_matrix))])
            data_matrix = data_matrix[mask]
            labels = [label for i, label in enumerate(labels) if mask[i]]
            patient_codes = [code for i, code in enumerate(patient_codes) if mask[i]]

            print(f"Removed {len(outlier_indices)} spectra in iteration {iteration+1}")

            if self.plot_cleaned_spectra: self.plot_spectra(wavenumber, data_matrix, patient_codes) 

        return data_matrix, labels, patient_codes
    
    def preprocess_single_spectrum(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess a single spectrum: Converts wavenumber and intensity to float if necessary."""
        
        wavenumber_range = tuple(self.params.get("wavenumber_range", (400.0, 1800.0)))
        
        # Trim to the specified wavenumber range
        mask = (spectrum[:, 0] >= wavenumber_range[0]) & (spectrum[:, 0] <= wavenumber_range[1])
        cropped_spectrum = spectrum[mask, :]  # Keep only rows within the range

        wavenumber = cropped_spectrum[:, 0]
        intensity = cropped_spectrum[:, 1]

        # **Step 1: Remove Cosmic Rays**
        intensity = self.remove_cosmic_rays(intensity, threshold=self.params.get("cosmic_ray_threshold", 10))

        # **Step 2: Smooth Spectrum using Savitzky-Golay filter**
        smoothed_intensity = savgol_filter(intensity, window_length=self.params["smoothing_window"], polyorder=self.params["smoothing_order"])

        # **Step 3: Baseline Correction using airPLS**
        baseline = airPLS(smoothed_intensity, lambda_=self.params["baseline_lambda"])
        corrected_intensity = smoothed_intensity - baseline

        # **Step 4: Normalize Spectrum**
        normalization_method = self.params["normalization"]
        normalized_intensity = normalization_method.apply(corrected_intensity)

        return wavenumber, normalized_intensity

    def remove_cosmic_rays(self, intensity: np.ndarray, threshold: float = 10) -> np.ndarray:
        """
        Removes cosmic ray spikes using median filtering and thresholding.

        Args:
            intensity (np.ndarray): 1D intensity array.
            threshold (float): Multiplier of local standard deviation for detecting outliers.

        Returns:
            np.ndarray: Corrected intensity array with cosmic rays removed.
        """
        median_filtered = medfilt(intensity, kernel_size=5)  # Apply median filter to get local baseline
        residual = intensity - median_filtered  # Difference between raw and filtered data
        std_dev = np.std(residual)  # Compute standard deviation of residuals

        cosmic_ray_mask = np.abs(residual) > (threshold * std_dev)  # Identify spikes
        corrected_intensity = np.copy(intensity)  # Make a copy to modify

        # Replace cosmic ray spikes with median-filtered values
        corrected_intensity[cosmic_ray_mask] = median_filtered[cosmic_ray_mask]

        return corrected_intensity

    def plot_spectrum(self, idx: int, wavenumber: np.ndarray, intensity: np.ndarray, color) -> None:
        """
        Plots a single processed spectrum **on the same figure**.

        Args:
            idx (int): Spectrum index for labeling.
            wavenumber (np.ndarray): Processed wavenumber array.
            intensity (np.ndarray): Processed intensity array.
            color: Color for plotting.
        """
        plt.plot(wavenumber, intensity, label=f"Spectra {idx}", color=color, alpha=0.8)
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Processed Intensity (a.u.)")
        plt.title(f"Raman Spectra for Group {idx}; Patient: {self.averaged_patient_codes[idx]}")
        # plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.grid(True)
        plt.show()  # Show all spectra on the same figure

    def plot_spectra(self, wavenumber: np.ndarray, data: List[np.ndarray], patient_codes: np.ndarray) -> None:
        """
        Plots all processed spectra overlaid on the same figure.

        Args:
            spectra (List[np.ndarray]): List of processed spectra.
            labels (np.ndarray): Array of labels for each spectrum.
        """
        colors = plt.get_cmap("jet")(np.linspace(0, 1, len(patient_codes)))
        for idx, intensity in enumerate(data):
            plt.plot(wavenumber, intensity, label=f"Group {patient_codes[idx]}", color=colors[idx], alpha=0.8)
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Processed Intensity (a.u.)")
        plt.title("All spectra overlaid")
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.grid(True)
        plt.show()

