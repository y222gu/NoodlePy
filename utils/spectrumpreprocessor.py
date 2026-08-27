import copy
from utils.spectrum import Spectrum
import os
import yaml

class SpectrumPreprocessor:
    def __init__(self, 
                cropping: bool = False,
                baseline_correction: bool = False,
                remove_cosmic_rays: bool = False,
                normalization: bool = False,
                smoothing: bool = False,
                config_path: str = None):
        
        if config_path is None:
            current_directory = os.getcwd()
            config_path = os.path.join(current_directory, "noodlepy", "config", "config_default.yml")
        
        with open(config_path, "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)

        self.config = config['preprocessing']
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.remove_cosmic_rays = remove_cosmic_rays
        self.normalization = normalization
        self.smoothing = smoothing

    def preprocess(self, spectrum: Spectrum) -> Spectrum:
        preprocessed_spectrum = copy.deepcopy(spectrum)

        #REQ: The sequence of the preprocessing steps matters

        # Cropping
        if self.cropping:
            preprocessed_spectrum.crop_spectrum(**self.config['cropping']) # Range temporary chosen by Victor

        if self.remove_cosmic_rays:
            preprocessed_spectrum.remove_cosmic_rays(**self.config['cosmic_rays_removal'])

        # Baseline correction from config
        if self.baseline_correction:
            preprocessed_spectrum.airPLS(**self.config['baseline_correction'])

        # Normalization from config
        if self.normalization:
            preprocessed_spectrum.normalize_spectrum(self.config['normalization_type'])

        # Smoothing from config
        if self.smoothing:
            preprocessed_spectrum.savgol_filter(**self.config['smoothing'])

        return preprocessed_spectrum