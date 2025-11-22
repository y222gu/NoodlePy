"""
Raman Spectrum Analysis Script

This script analyzes Raman spectra from a folder of text files and performs:
1. Spectrum plotting with smoothing
2. Entropy vs position analysis
3. Focus score calculations using multiple metrics
4. Reference spectrum comparison

Author: Cleaned up version
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import scipy


class RamanSpectrumAnalyzer:
    """A class to analyze Raman spectra and calculate focus metrics."""
    
    def __init__(self, folder_path: str, reference_path: str, 
                 reference_smooth_window: int = 100, 
                 spectrum_smooth_window: int = 5):
        """
        Initialize the analyzer with data folder and reference spectrum paths.
        
        Args:
            folder_path: Path to folder containing spectrum files
            reference_path: Path to reference spectrum file
            reference_smooth_window: Window size for smoothing reference spectrum
            spectrum_smooth_window: Window size for smoothing all other spectra
        """
        self.folder_path = folder_path
        self.reference_path = reference_path
        self.reference_smooth_window = reference_smooth_window
        self.spectrum_smooth_window = spectrum_smooth_window
        self.files = []
        self.sorted_files = []
        self.reference_data = None
        
        print(f"Initialized analyzer with:")
        print(f"  Reference smoothing window: {self.reference_smooth_window}")
        print(f"  Spectrum smoothing window: {self.spectrum_smooth_window}")
        
    def load_files(self) -> None:
        """Load and sort all .txt files from the folder."""
        self.files = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')]
        self.sorted_files = sorted(self.files, key=self._extract_position)
        print(f"Found {len(self.sorted_files)} spectrum files")
        
    def _extract_position(self, filename: str) -> float:
        """Extract the position number from filename."""
        match = re.search(r'position_(\d+\.?\d*)', filename)
        return float(match.group(1)) if match else float('inf')
    
    def _extract_entropy(self, filename: str) -> Optional[float]:
        """Extract the entropy value from filename."""
        match = re.search(r'entropy_(\d+\.?\d*)', filename)
        return float(match.group(1)) if match else None
    
    def _read_spectrum_file(self, file_path: str) -> Tuple[List[float], List[float]]:
        """Read wavelength and intensity data from a spectrum file."""
        wavelengths, intensities = [], []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        wavelengths.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                    except ValueError:
                        continue
        
        return wavelengths, intensities
    
    def _smooth_spectrum(self, intensity, window_length=9, polyorder=2):
        # print('Smoothing spectrum using Savitzky-Golay filter with window length:', window_length, 'and polynomial order:', polyorder, '...')
        smoothed_intensity = scipy.signal.savgol_filter(intensity, window_length, polyorder)
        return smoothed_intensity
    
    def _normalize_spectrum(self, intensities: np.ndarray) -> np.ndarray:
        """Min-max normalize spectrum to [0, 1]. Handles constant arrays safely."""
        arr = np.asarray(intensities, dtype=float)
        if arr.size == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val == 0:
            # constant signal -> return zeros (or ones if you prefer)
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    def load_reference_spectrum(self) -> None:
        """Load and process the reference spectrum using configured window size."""
        try:
            ref_wavelengths, ref_intensities = self._read_spectrum_file(self.reference_path)
            
            if not ref_intensities:
                print("Warning: Reference spectrum is empty")
                return
                
            # Smooth and normalize reference using configured window size
            smoothed_ref = self._smooth_spectrum(ref_intensities, self.reference_smooth_window)
            normalized_ref = self._normalize_spectrum(smoothed_ref)
            # normalized_ref = smoothed_ref
            
            self.reference_data = {
                'wavelengths': ref_wavelengths,
                'intensities': normalized_ref,
                'raw_wavelengths': ref_wavelengths,
                'raw_intensities': ref_intensities
            }
            
        except FileNotFoundError:
            print(f"Warning: Reference file not found at {self.reference_path}")
        except Exception as e:
            print(f"Error loading reference spectrum: {e}")
    
    def plot_entropy_vs_position(self, save_path: str = 'entropy_vs_position.png') -> None:
        """Plot entropy values against position."""
        positions, entropies = [], []
        
        for filename in self.sorted_files:
            position = self._extract_position(filename)
            entropy = self._extract_entropy(filename)
            if entropy is not None:
                positions.append(position)
                entropies.append(entropy)
        
        if not positions:
            print("No entropy data found in filenames")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(positions, entropies, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title('Entropy vs Position', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved entropy plot to {save_path}")
        print("\nEntropy vs Position:")
        for pos, ent in zip(positions, entropies):
            print(f"  Position {pos}: Entropy {ent}")
    
    def plot_reference_spectrum(self, save_path: str = 'reference_smoothed_spectrum.png') -> None:
        """Plot the smoothed reference spectrum."""
        if not self.reference_data:
            print("Reference spectrum not loaded")
            return
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.reference_data['wavelengths'], self.reference_data['intensities'], 
                color='k', linewidth=1.8, label='Reference (smoothed)')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('Smoothed Reference Spectrum')
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(0, 1)  # enforce y-limits from 0 to 1
        ax.legend()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved reference spectrum plot to {save_path}")
    
    def calculate_focus_scores(self) -> dict:
        """Calculate focus scores for all spectra using configured smoothing window."""
        positions = []
        focus_scores = {
            'convolution': [],
            'correlation': [],
            'sharpness': []
        }
        all_spectra = []
        
        if not self.reference_data:
            print("Warning: No reference spectrum loaded for focus score calculation")
            return {}
        
        ref_intensities = self.reference_data['intensities']
        
        for filename in self.sorted_files:
            file_path = os.path.join(self.folder_path, filename)
            position = self._extract_position(filename)
            positions.append(position)
            
            wavelengths, intensities = self._read_spectrum_file(file_path)
            
            if not intensities:
                # Handle empty files
                focus_scores['convolution'].append(0.0)
                focus_scores['correlation'].append(0.0)
                focus_scores['sharpness'].append(0.0)
                all_spectra.append({'wavelengths': np.array([]), 'intensities': np.array([])})
                continue
            
            # Smooth and normalize spectrum using configured window size
            smoothed_intensities = self._smooth_spectrum(intensities, self.spectrum_smooth_window)
            normalized_intensities = self._normalize_spectrum(smoothed_intensities)

            all_spectra.append({
                'wavelengths': np.array(wavelengths),
                'intensities': normalized_intensities,
                'position': position
            })
            
            # Calculate focus scores
            min_len = min(len(ref_intensities), len(normalized_intensities))
            
            # 1. Convolution score (alignment/match)
            if min_len > 0:
                conv_score = np.max(np.correlate(normalized_intensities[:min_len],
                                               ref_intensities[:min_len], mode='valid'))
            else:
                conv_score = 0.0
            focus_scores['convolution'].append(conv_score)
            
            # 2. Correlation score
            if min_len > 1:
                corr_score = np.corrcoef(normalized_intensities[:min_len],
                                       ref_intensities[:min_len])[0, 1]
                corr_score = 0.0 if np.isnan(corr_score) else corr_score
            else:
                corr_score = 0.0
            focus_scores['correlation'].append(corr_score)
            
            # 3. Sharpness score (gradient variance)
            if len(normalized_intensities) > 1:
                intensity_gradient = np.gradient(normalized_intensities)
                sharpness = np.var(intensity_gradient)
            else:
                sharpness = 0.0
            focus_scores['sharpness'].append(sharpness)
        
        print(f"Calculated focus scores using spectrum smoothing window: {self.spectrum_smooth_window}")
        
        return {
            'positions': positions,
            'scores': focus_scores,
            'spectra': all_spectra
        }
    
    def plot_focus_scores(self, focus_data: dict, reference_position: float = 6.28, 
                         save_path: str = 'focus_scores.png') -> None:
        """Plot all focus score metrics."""
        if not focus_data:
            print("No focus data to plot")
            return
        
        positions = focus_data['positions']
        scores = focus_data['scores']
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        # Convolution score
        axes[0].plot(positions, scores['convolution'], 'ro-', linewidth=2, markersize=8)
        axes[0].set_ylabel('Convolution Score', fontsize=11)
        axes[0].set_title('Focus Score Metrics vs Position', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=reference_position, color='g', linestyle='--', 
                       label=f'Reference ({reference_position})', alpha=0.7)
        axes[0].legend()
        
        # Correlation score
        axes[1].plot(positions, scores['correlation'], 'go-', linewidth=2, markersize=8)
        axes[1].set_ylabel('Correlation Score', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=reference_position, color='g', linestyle='--', 
                       label=f'Reference ({reference_position})', alpha=0.7)
        axes[1].legend()
        
        # Sharpness score
        axes[2].plot(positions, scores['sharpness'], 'bo-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Position', fontsize=11)
        axes[2].set_ylabel('Sharpness Score', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(x=reference_position, color='g', linestyle='--', 
                       label=f'Reference ({reference_position})', alpha=0.7)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved focus scores plot to {save_path}")
    
    def plot_all_smoothed_spectra(self, focus_data: dict, save_path: str = 'all_smoothed_spectra.png') -> None:
        """Plot all smoothed spectra in separate subplots."""
        if not focus_data or not focus_data['spectra']:
            print("No spectrum data to plot")
            return
        
        valid_spectra = [s for s in focus_data['spectra'] if len(s['intensities']) > 0]
        num_plots = len(valid_spectra)
        
        if num_plots == 0:
            print("No valid spectra to plot")
            return
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]
        
        cmap = plt.get_cmap('viridis')
        
        for idx, spectrum in enumerate(valid_spectra):
            ax = axes[idx]
            color = cmap(idx / max(1, num_plots - 1))
            ax.plot(spectrum['wavelengths'], spectrum['intensities'], 
                   color=color, alpha=0.7, linewidth=1)
            ax.text(0.02, 0.95, f'Position: {spectrum["position"]:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_yticks([])
            ax.set_ylim(0, 1)  # enforce consistent y-limits for all subplots
            if idx < num_plots - 1:
                ax.set_xticks([])
        
        axes[-1].set_xlabel('Wavelength')
        fig.text(0.04, 0.5, 'Normalized Intensity', va='center', rotation='vertical')
        plt.suptitle('All Smoothed Spectra (normalized)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved all spectra plot to {save_path}")
    

def main():
    """Main execution function with configuration parameters."""
    
    # ====================================================================
    # CONFIGURATION SECTION
    # ====================================================================
    
    # File paths
    FOLDER_PATH = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/3_bad'
    REFERENCE_PATH = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/focus_test/rough/rough_prusa_focus_position_5.13_entropy_8.330878485585052.txt'
    
    # Smoothing window sizes (adjust these as needed)
    REFERENCE_SMOOTH_WINDOW = 3  # Window size for smoothing the reference spectrum
    SPECTRUM_SMOOTH_WINDOW = 3   # Window size for smoothing all test spectra
    
    # Other parameters
    REFERENCE_POSITION = 6.0      # Position of reference spectrum (for plotting reference line)
    
    # ====================================================================
    
    # Initialize analyzer with configured window sizes
    print("="*70)
    print("RAMAN SPECTRUM ANALYSIS")
    print("="*70)
    analyzer = RamanSpectrumAnalyzer(
        folder_path=FOLDER_PATH, 
        reference_path=REFERENCE_PATH,
        reference_smooth_window=REFERENCE_SMOOTH_WINDOW,
        spectrum_smooth_window=SPECTRUM_SMOOTH_WINDOW
    )
    
    # Load files
    print("\nLoading spectrum files...")
    analyzer.load_files()
    
    # Load reference spectrum
    print("Loading reference spectrum...")
    analyzer.load_reference_spectrum()
    
    # Generate all plots and analyses
    print("\nGenerating plots and analyses...")
    print("-" * 40)
    
    # 2. Plot entropy vs position
    print("Plotting entropy analysis...")
    analyzer.plot_entropy_vs_position()
    
    # 3. Plot reference spectrum
    print("Plotting reference spectrum...")
    analyzer.plot_reference_spectrum()
    
    # 4. Calculate focus scores
    print("Calculating focus scores...")
    focus_data = analyzer.calculate_focus_scores()

    # 4b. Plot all smoothed spectra
    print("Plotting all smoothed spectra...")
    analyzer.plot_all_smoothed_spectra(focus_data)

    # 5. Plot focus scores
    print("Plotting focus scores...")
    analyzer.plot_focus_scores(focus_data, reference_position=REFERENCE_POSITION)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("Check the generated PNG files for results:")
    print("  - spectrum_plot.png")
    print("  - entropy_vs_position.png") 
    print("  - reference_smoothed_spectrum.png")
    print("  - focus_scores.png")
    print("  - all_smoothed_spectra.png")
    print("="*70)


if __name__ == "__main__":
    main()