import copy
from noodlepy.utils.class_Spectrum import Spectrum
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
            config_path = os.path.join(current_directory, 'NoodlePy', "noodlepy", "config", "config_default.yml")
        
        with open(config_path, "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)

        self.config = config['preprocessing']
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.remove_cosmic_rays = remove_cosmic_rays
        self.normalization = normalization
        self.smoothing = smoothing

    def pre_process(self, spectrum: Spectrum) -> Spectrum:
        pre_processed_spectrum = copy.deepcopy(spectrum)

        #REQ: The sequence of the preprocessing steps matters

        # Cropping
        if self.cropping:
            pre_processed_spectrum.crop_spectrum(**self.config['cropping']) # Range temporary chosen by Victor

        # Baseline correction from config
        if self.baseline_correction:
            pre_processed_spectrum.airPLS(**self.config['baseline_correction'])

        if self.remove_cosmic_rays:
            pre_processed_spectrum.remove_cosmic_rays(**self.config['cosmic_rays_removal'])

        # Normalization from config
        if self.normalization:
            pre_processed_spectrum.normalize_spectrum(self.config['normalization_type'])

        # Smoothing from config
        if self.smoothing:
            pre_processed_spectrum.savgol_filter(**self.config['smoothing'])

        return pre_processed_spectrum