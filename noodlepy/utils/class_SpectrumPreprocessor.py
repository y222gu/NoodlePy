import copy
from noodlepy.utils.class_Spectrum import Spectrum
import os
import yaml

class SpectrumPreprocessor:
    def __init__(self, 
                cropping: bool = False,
                baseline_correction: bool = False,
                despike: bool = False,
                normalization: bool = False,
                smoothing: bool = False,
                config_path: str = None):
        
        if config_path is None:
            current_directory = os.getcwd()
            config_path = os.path.join(current_directory, "noodlepy", "config", "config.yml")
        
        with open(config_path, "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)

        self.config = config['preprocessing']
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.despike = despike
        self.normalization = normalization
        self.smoothing = smoothing

    def pre_process(self, spectrum: Spectrum) -> Spectrum:
        pre_processed_spectrum = copy.deepcopy(spectrum)

        #REQ: The sequence of the preprocessing steps matters

        # Cropping
        if self.cropping:
            pre_processed_spectrum.crop_spectrum(self.config['cropping']['start'], self.config['cropping']['end']) # Range temporary chosen by Victor

        # Baseline correction from config
        if self.baseline_correction:
            pre_processed_spectrum.airPLS(self.config['baseline_correction'])

        # Despike from config
        if self.despike:
            pre_processed_spectrum.despike(self.config['despike']['kernel_size'], self.config['despike']['threshold'])

        # Normalization from config
        if self.normalization:
            pre_processed_spectrum.normalize_spectrum(self.config['normalization_type'])

        # Smoothing from config
        if self.smoothing:
            pre_processed_spectrum.savgol_filter(self.config['smoothing']['window_length'], self.config['smoothing']['polyorder'])

        return pre_processed_spectrum