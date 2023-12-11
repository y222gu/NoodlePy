import yaml
import argparse
import numpy as np
import os

# Import noodlespy modules
import noodlespy.utils.raman_create_augment as create_augment

# notes: python -m noodlespy.ml.generate_raman_spectra_from_uniform_distribution

def parse_args():
    # Commands
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(
        "--config_path",
        dest="config_path",
        type=str,
        default="noodlespy/config/config_test.yml",
        help="Path to save output spectra",
    )

    p.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output/",
        help="Path to save output spectra",
    )

    p.add_argument(
        "--n_spectra",
        dest="n_spectra",
        type=int,
        default=100,
        help="Number of spectra to be generated",
    )

    args = p.parse_args()
    return args


def main():

    # Parser
    args = parse_args()
    config_path = args.config_path
    output_path = args.output_path
    n_spectra = args.n_spectra

    # Initiate constant variables
    with open(config_path, "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # load metabolomics names and concentration
    metabolite_name_list, metabolite_ratios = create_augment.load_metabolites(sample_type = config["sample_type"], file_path = config["metabolomics_file_path"])
    
    metabolite_spectrum_dict = create_augment.load_spectra(DFT_file_name = config["DFT_file_path"], metabolite_name_list = metabolite_name_list, laser_wavelength = config["laser_wavelength"])
    # crop all expectra to the same range
    metabolite_spectrum_dict = create_augment.crop_spectra(metabolite_spectrum_dict, start_wavenumber=config["raman_shift_range_pars"]["start_wavenumber"], end_wavenumber = config["raman_shift_range_pars"]["end_wavenumber"])
    # normalize all spectra individually
    metabolite_spectrum_dict = create_augment.normalize_spectra(metabolite_spectrum_dict, config["normalization_option_for_individual_spectrum"])
    
    for n in range(n_spectra):

        # TODO: generate a randon set of ratios
        # metabolite_ratios = np.random.rand(n_wavelengths)
    
        # Create the spectrum of a mixture
        mixture_spectrum_dict = create_augment.mix_spectra(
            metabolite_spectrum_dict,
            metabolite_ratios,
            mixture_name = config["sample_type"]
        )

        # TODO: set output file
        # output_file = os.join(output_path, f"raman_patient_{n}.pkl")

        # TODO: Save pickle
        # np.save()
