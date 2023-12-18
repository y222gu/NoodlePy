import yaml
import argparse
import numpy as np
import os
import pickle
import json

# Import noodlespy modules
import noodlepy.utils.raman_create_augment as create_augment

# notes: python -m noodlepy.ml.generate_raman_spectra_from_uniform_distribution

def parse_args():
    # Commands
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(
        "--config_path",
        dest="config_path",
        type=str,
        default="noodlepy/config/config_test.yml",
        help="Path to save output spectra",
    )

    p.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="noodlepy/data/simulated_patients/",
        help="Path to save output spectra",
    )

    p.add_argument(
        "--n_spectra",
        dest="n_spectra",
        type=int,
        default=5,
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
    # igonore the concentration and generate random ratios later
    metabolite_name_list, *_ = create_augment.load_metabolites(sample_type = config["sample_type"], file_path = config["metabolomics_file_path"])
    
    metabolite_spectrum_dict = create_augment.load_spectra(DFT_file_name = config["DFT_file_path"], metabolite_name_list = metabolite_name_list, laser_wavelength = config["laser_wavelength"])
    # crop all expectra to the same range
    metabolite_spectrum_dict = create_augment.crop_spectra(metabolite_spectrum_dict, start_wavenumber=config["raman_shift_range_pars"]["start_wavenumber"], end_wavenumber = config["raman_shift_range_pars"]["end_wavenumber"])
    # normalize all spectra individually
    metabolite_spectrum_dict = create_augment.normalize_spectra(metabolite_spectrum_dict, config["normalization_option_for_individual_spectrum"])
    
    for n in range(n_spectra):

        # generate a randon set of ratios
        # get the number of metabolites in the dictionary
        metabolite_ratios = np.random.uniform(0, 1, size=len(metabolite_spectrum_dict.keys()))
        print(f'Generating random metabolites ratios for patient {n}: {metabolite_ratios}')

        # Create the spectrum of a mixture
        mixture_spectrum_dict = create_augment.mix_spectra(
            metabolite_spectrum_dict,
            metabolite_ratios,
            mixture_name = config["sample_type"]
        )

        # set output file
        # connect words to make a file name
        output_file = os.path.join(output_path, f"sim_patient_{n}_pristine_spectrum.pickle")

        # Save as a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(mixture_spectrum_dict, f)


        print(f"Saved mixture spectrum to {output_file}")

if __name__ == "__main__":
    main()