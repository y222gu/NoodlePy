# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import yaml
import pandas as pd
import pickle
import seaborn as sns
import subprocess

def create_spectrum(
    metabolite_spectrum_dict: list,
    concentrations: np.array,
    config: dict,
    gaussian_pars: dict,
) -> tuple:
    """
    Create a pristine spectrum base on the leaterature data.

    Parameters:
    metabolite_spectrum_dict (dict): List of component spectra.
    concentrations: (np.array): Concentrations of the components.
    config (dict): Experiment conditions.
    gaussian_pars (dict): The parameters for the gaussian distribution.

    Returns:
    tuple[np.array, np.array]: The generated spectrum and corresponding wavenumbers.
    """
    raman_shift_pars = config["raman_shift_pars"]
    raman_shift_range = np.linspace(*raman_shift_pars)
    spectrum = np.zeros(len(raman_shift_range))

    for i, component_i in enumerate(metabolite_spectrum_dict):
        peak_locations = component_i["peak_locations"]
        peak_shapes = [gaussian_pars[peak] for peak in component_i["peak_shapes"]]
        peak_intensities = [
            gaussian_pars[peak] for peak in component_i["peak_intensities"]
        ]

        for peak_i in range(len(peak_locations)):
            spectrum += (
                concentrations[i]
                * stats.norm.pdf(
                    raman_shift_range, peak_locations[peak_i], peak_shapes[peak_i]
                )
                * peak_intensities[peak_i]
            )
    spectrum /= np.max(spectrum)
    spectrum *= config[
        "spectrum_amplifying_factor"
    ]  # Normalize the mixture spectrum
    return spectrum, raman_shift_range

def load_metabolites(file_path: str, sample_type: str) -> tuple[list, np.array]:
    """
    Load metabolomics names and concentrations from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.
    sample_type (str): The type of sample.

    Returns:
    tuple[list, np.array]: The list of metabolomics and the corresponding concentrations.
    """
    # Load the XLSX file into a DataFrame
    df = pd.read_excel(file_path)
    # Extract the column as a NumPy array
    if sample_type == "Saliva":
        metabolite_name_list = df["label"]
        metabolite_ratios = df["Saliva"].to_numpy()
    elif sample_type == "Plasma":
        metabolite_name_list = df["label"]
        metabolite_ratios = df["Plasma"].to_numpy()
    elif sample_type == "Serum":
        metabolite_name_list = df["label"]
        metabolite_ratios = df["Serum"].to_numpy()
    else:
        raise ValueError("sample_type must be either Saliva, Plasma or Serum")

    metabolite_name_list = metabolite_name_list[~np.isnan(metabolite_ratios)]
    metabolite_ratios = metabolite_ratios[~np.isnan(metabolite_ratios)]
    metabolite_name_list = [x for _, x in sorted(zip(metabolite_ratios, metabolite_name_list), reverse=True)]
    metabolite_ratios = sorted(metabolite_ratios, reverse=True)

    return metabolite_name_list, metabolite_ratios

def load_spectra(DFT_file_name: str, metabolite_name_list: list, laser_wavelength: str) -> dict:
    """
    Load the spectra database from a pickle file.

    Parameters:
    file_name (str): The path to the pickle file.

    Returns:
    dict: The spectra database.
    """
    print(f'Start to load spectra from the pickle file...')

    # Load data from the pickle file
    with open(DFT_file_name, 'rb') as file:
        database = pickle.load(file)

    # take an array out of the list
    metabolite_name_list = np.array(metabolite_name_list)
 
   # Extract values for the specified molecules
    spectrum_dict = {}
    for metabolite_name in metabolite_name_list:
        metabolite_data = database.get(metabolite_name, None)
        if metabolite_data:
            spectrum = extract_spectrum(metabolite_data, laser_wavelength
    )
            spectrum_dict[metabolite_name] = spectrum
        else:
            print(f'Molecule "{metabolite_name}" not found in the pickle file. Skipping...')

    print(f'Loding spectra from the pickle file is done...')

    return spectrum_dict


    # Define a function to extract values
def extract_spectrum(metabolite_data, laser_wavelength: int):
    # Define the intensity generated at specific laser wavelength to be used
    intensity_to_use = "intensity_" + str(laser_wavelength)
    raman_shift = np.array(metabolite_data['freq'])
    intensity = metabolite_data[intensity_to_use]
    spectrum = {
        'raman_shift': raman_shift,
        'intensity': intensity
    }
    return spectrum

def crop_spectra(
    spectrum_dict: dict,
    start_wavenumber: int,
    end_wavenumber: int
) -> dict:
    """
    Crop a spectrum based on a specified wavelength range.

    Args:
        raman_shift_range (np.array): Array of wavenumber values starting with.
        spectrum (np.array): Array of corresponding spectrum values.
        start_wavenumber (np.array): The starting wavenumber for cropping.
        end_wavenumber (np.array): The ending wavenumber for cropping.

    Returns:
        cropped_raman_shift_range (np.array): Cropped wavenumber values.
        cropped_spectrum (np.array): Cropped spectrum values.
    """
    print(f'Start to align all spectra to the specified range...')
    print(f'Checking if the raman shift range of molecules are matching')

    cropped_spectrum_dict={}

    for name in spectrum_dict:
        raman_shift = spectrum_dict[name]["raman_shift"]

        # test if the start_wavenumber exist in the raman shift

        if start_wavenumber in raman_shift and end_wavenumber in raman_shift:
            # Find the indices corresponding to the start and end wavelengths
            start_index = np.where(raman_shift==start_wavenumber)[0][0]
            end_index = np.where(raman_shift==end_wavenumber)[0][0]

            # Crop the spectrum based on the specified wavelength range
            cropped_raman_shift = spectrum_dict[name]["raman_shift"][start_index:end_index]
            cropped_intensity = spectrum_dict[name]["intensity"][start_index:end_index]
            cropped_spectrum = {
                "raman_shift":cropped_raman_shift, 
                "intensity":cropped_intensity
            }
            cropped_spectrum_dict[name] = cropped_spectrum

        else:
            print(f'Molecule {name} Raman shift range could not be aligned to the specified range. Removing from the spectrum dict...')
    
    print(f'All spectra are aligned and cropped between Wavenumber {start_wavenumber} to {end_wavenumber}')

    return cropped_spectrum_dict

def normalize_spectra(
    spectrum_dict: dict,
    normalization_option: str,
) -> dict:
    """
    Normalize a spectrum.

    Args:
        spectrum (np.array): Array of corresponding spectrum values.
        normalization_option (str): The normalization method.

    Returns:
        normalized_spectrum (np.array): Normalized spectrum values.
    """
    print(f'Normalizing spectra by: ', normalization_option, '...')

    for name in spectrum_dict:
       
        # intensity
        intensity = spectrum_dict[name]["intensity"]

        # normalize by the area under the curve:
        if normalization_option == 'area':
            intensity /= np.trapz(intensity, spectrum_dict[name]["raman_shift"])
        # normalize by the maximum intensity:
        elif normalization_option == 'max':
            intensity /= np.max(intensity)
        else:
            raise ValueError(f'Mormalization method is not defined')
        
        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict

def quantumn_efficiency(spectrum_dict: dict, quantumn_efficiency: float) -> dict:
    """
    Add quantumn efficiency to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    quantumn_efficiency (float): The factor determining the amount of quantumn efficiency.

    Returns:
    np.array: The spectrum with added quantumn efficiency.
    """
    for name in spectrum_dict:
        # intensity
        intensity = spectrum_dict[name]["intensity"]* quantumn_efficiency

        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict
    

def mix_spectra(
    spectrum_dict: dict,
    metabolite_ratios: np.array,
    mixture_name: str,
) -> dict:
    """
    Create a mixture spectrum from a dictionary of spectra.

    Parameters:
    component_spectrum_list (list of dict): List of component spectra.
    concentrations: (np.array): Concentrations of the components.
    config (dict): Experiment conditions.

    Returns:
    tuple[np.array, np.array]: The generated spectrum and corresponding wavenumbers.
    """
    print(f'Start to mix spectra...')
    mixture_spectrum_dict = {}

    for i, name in enumerate(spectrum_dict):
        if i == 0:
            raman_shift = spectrum_dict[name]["raman_shift"]

            intensity = spectrum_dict[name]["intensity"]
            mixture_intensity = (metabolite_ratios[i] * intensity)

            print(f'Molecule "{name}" is added to the mixture...')

        else:
            # double check if the raman_shift_range is matching
            if np.array_equal(spectrum_dict[name]["raman_shift"], raman_shift):
                raman_shift = spectrum_dict[name]["raman_shift"]

                intensity = spectrum_dict[name]["intensity"]
                mixture_intensity += (metabolite_ratios[i] * intensity)

                print(f'Molecule "{name}" is added to the mixture...')
            
            else:
                raise ValueError(f'Raman shift range of "{name}" does not match with added metabolites')
            
    print(f'Mixing is done...')        

    mixture_spectrum_dict[mixture_name] = {"raman_shift": raman_shift, "intensity": mixture_intensity, "metabolite_ratios": metabolite_ratios}

    return mixture_spectrum_dict


def add_noise(spectrum_dict: dict, noise_pars: dict) -> dict:
    """
    Add shot noise to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    shot_noise_factor (float): The factor determining the amount of shot noise.

    Returns:
    np.array: The spectrum with added shot noise.
    """
    rng = np.random.default_rng() # this is using the default PCG64 generator

    for name in spectrum_dict:

        # intensity
        intensity = spectrum_dict[name]["intensity"]
        noise_type = noise_pars["noise_type"]

        if noise_type == "poisson":
            intensity += 0.001*rng.poisson(noise_pars["lam"], len(intensity))

        elif noise_type == "gaussian":
            intensity += rng.normal(noise_pars["mean"], noise_pars["std"], len(intensity))

        elif noise_type == "uniform":
            intensity += rng.uniform(noise_pars["low"],noise_pars["high"],len(intensity))

        elif noise_type == "exponential":
            intensity += rng.exponential(noise_pars["mean"], len(intensity))

        elif noise_type == "lognormal":
            intensity += rng.lognormal(noise_pars["mean"],noise_pars["sigma"],len(intensity))
        else:
            raise ValueError("noise_type must be one among 'poisson', 'gaussian', 'uniform', 'expoenetial', and 'lognormal'")
        
        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict


def add_cosmic_rays(spectrum_dict: dict, cosmic_ray_pars: dict) -> dict:
    """
    Add cosmic rays to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    cosmic_ray_pars (dict): Parameters for cosmic rays.

    Returns:
    np.array: The spectrum with added cosmic rays.
    """

    for name in spectrum_dict:
        # intensity
        intensity = spectrum_dict[name]["intensity"]
        raman_shift = spectrum_dict[name]["raman_shift"]

        number_spikes = cosmic_ray_pars["spike_num"]
        spike_amplitude = cosmic_ray_pars["spike_amplitude"]
        spikes = np.random.randint(0, len(raman_shift), number_spikes)

        for spike in spikes:
            intensity[spike] = intensity[spike] + spike_amplitude * np.random.random()
        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict


def create_baseline(
    spectrum_dict: dict, baseline_pars: dict
) -> dict:
    """
    Add a baseline to a spectrum.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    config (dict): Experiment conditions.

    Returns:
    np.array: The spectrum with the added baseline.
    """
    baseline_dict = {}
    for name in spectrum_dict:
        # Raman shift
        raman_shift = spectrum_dict[name]["raman_shift"]

        baseline_type = baseline_pars["baseline_type"]

        if baseline_type == "poly":
            poly_orders = baseline_pars["poly_orders"]
            poly_coefficients = baseline_pars["poly_coefficients"]
            poly_displacement = baseline_pars["poly_displacement"]

            baseline_intensity = 0
            for i in range(poly_orders + 1):
                baseline_intensity += poly_coefficients[i] * (raman_shift - poly_displacement[i]) ** i

        elif baseline_type == "sine":  # Sine wave baseline
            sine_amplitude = baseline_pars["sine_amplitude"]
            sine_frequency = baseline_pars["sine_frequency"]
            sine_phase = baseline_pars["sine_phase"]
            baseline_intensity = sine_amplitude * np.sin(sine_frequency * raman_shift + sine_phase)

        else:
            baseline_intensity = np.zeros_like(raman_shift)  # No baseline

        # Add the baseline to the spectrum
        # intensity = intensity + baseline * config["baseline_amplifying_factor"]
        # spectrum["intensity"] = intensity
        baseline_dict[name] = {"raman_shift": raman_shift ,"intensity": baseline_intensity}
    return baseline_dict

def add_baseline(spectrum_dict: dict, baseline_dict: dict) -> dict:
    """
    Add a baseline to a spectrum.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    baseline (np.array): The baseline.

    Returns:
    np.array: The spectrum with the added baseline.
    """
    for name in spectrum_dict:
        # intensity
        intensity = spectrum_dict[name]["intensity"]
        baseline_intensity = baseline_dict[name]["intensity"]

        intensity = intensity + baseline_intensity
        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict

def shift_spectrum(spectrum_dict: dict, instrument_shift: float) -> dict:
    """
    Shift the spectrum's wavenumbers.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    shift (float): The amount to shift the wavenumbers.

    Returns:
    np.array: The shifted wavenumbers.
    """
    for name in spectrum_dict:
            
        spectrum_dict[name]["raman_shift"] = spectrum_dict[name]["raman_shift"] + instrument_shift

    return spectrum_dict

# define a function to amplify signal with nan as the default input for the baseline


def amplify(spectrum_dict: dict, spectrum_amplifying_factor: float) -> dict:
    
    """
    Amplify a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    amplifying_factor (float): The factor to amplify the spectrum.

    Returns:
    np.array: The amplified spectrum.
    """
    for name in spectrum_dict:

        spectrum_dict[name]["intensity"] = spectrum_dict[name]["intensity"] * spectrum_amplifying_factor
    
    return spectrum_dict


def convolute_kernel(
    spectrum_dict: dict, kernel_std: float
) -> dict:
    """
    Convolute the spectrum with a kernel.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    kernel_std (float): The standard deviation of the kernel.

    Returns:
    np.array: The convoluted spectrum.
    """
    for name in spectrum_dict:
        # raman shift
        raman_shift = spectrum_dict[name]["raman_shift"]
        # intensity
        intensity = spectrum_dict[name]["intensity"]

        # Create the kernel
        kernel = signal.windows.gaussian(len(raman_shift), kernel_std)
        # Calculate the convolution
        intensity = signal.convolve(kernel, intensity, mode="same") * sum(kernel)
        # update the raman_shift in the spectrum
        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict


def pre_process(spectrum: dict, config: dict) -> np.array:
    """
    Preprocess a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    config (dict): Experiment conditions.

    Returns:
    np.array: The preprocessed spectrum.
    """
    # raman shift
    raman_shift = spectrum["raman_shift"]
    # intensity
    intensity = spectrum["intensity"]

    # Define the pipeline for preprocessing
    pipe = ramanspy.preprocessing.protocols.Pipeline(
        [
            ramanspy.preprocessing.despike.WhitakerHayes(),
            ramanspy.preprocessing.denoise.SavGol(window_length=12, polyorder=3),
            ramanspy.preprocessing.baseline.ASPLS(),
            ramanspy.preprocessing.normalise.MinMax(pixelwise=True),
        ]
    )
    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pipe.apply(
        ramanspy.Spectrum(intensity, raman_shift)
    )
    return preprocessed_spectrum

def plot_spectrum(
    spectrum_dict: dict, filename: str
):
    """
    Plot and save a spectrum to a file.

    Parameters:
    spectrum (np.array): The spectrum to be plotted.
    spectrum_range (np.array): The corresponding wavenumbers.
    color (str): The color for the plot.
    filename (str): The filename for the saved plot.
    """

    for name in spectrum_dict:
        # raman shift
        raman_shift = spectrum_dict[name]["raman_shift"]
        intensity = spectrum_dict[name]["intensity"]

        # line plot
        sns.lineplot(x=raman_shift, y=intensity)
        # define the size of the plot
        plt.gcf().set_size_inches(15, 5)
        # show subplot number on top of each subplot
        # ax.text(0, 1.15, "C", fontsize=16, transform=ax["C"].transAxes)
        # set the title
        plt.title(filename)
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel("Intensity (a.u.)")
        # set y axis range
        plt.ylim(0, 1)
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
    return

def update(frame,figure_list):
    plt.clf()
    # This function will be called for each frame in the animation
    current_figure = figure_list(frame)
    plt.draw()


def wavelengthToWavenumber(wl:np.array)->np.array:
    """
    Convert wavelength to wavenumber
    
    Parameters:
    wl (np.array): The array of wavelengths.
    
    Returns:
    np.array: The array of wavenumbers.
    """
    # takes a 1-D array of wavelengths(nm) and returns a 1-D array of
    # wavenumber (cm^-1) to perform element-wise multiplication rather than matrix multiplication,
    # use the .* operator. (e.g. .^3 or ./10 will put each element to the third power or divide by 10, respectively). 
    # Copied from Noodle Matlab code

    # first convent from nm to cm
    wlCM = wl*(1e-7)
    # then invert to cm^-1
    wn = wlCM^(-1)

    return wn

def main():
    # Initiate constant variables
    with open("noodlespy/config/config_test.yml", "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # load metabolomics names and concentrations
    metabolite_name_list, metabolite_ratios = load_metabolites(sample_type = config["sample_type"], file_path = config["metabolomics_file_path"])
    # load metabolomic spectra from DFT database
    metabolite_spectrum_dict = load_spectra(DFT_file_name = config["DFT_file_path"], metabolite_name_list = metabolite_name_list, laser_wavelength = config["laser_wavelength"])
    # crop all expectra to the same range
    metabolite_spectrum_dict = crop_spectra(metabolite_spectrum_dict, start_wavenumber=config["raman_shift_range_pars"]["start_wavenumber"], end_wavenumber = config["raman_shift_range_pars"]["end_wavenumber"])
    # normalize all spectra individually
    metabolite_spectrum_dict = normalize_spectra(metabolite_spectrum_dict, config["normalization_option_for_individual_spectrum"])
    # Create the spectrum of a mixture
    mixture_spectrum_dict = mix_spectra(
        metabolite_spectrum_dict,
        metabolite_ratios,
        mixture_name = config["sample_type"]
    )
    plot_spectrum(mixture_spectrum_dict,'step_01_mix_spectra.png')

    # Normalize the mixture spectrum
    mixture_spectrum_dict = normalize_spectra(mixture_spectrum_dict, normalization_option = config["normalization_option_for_mixture_spectrum"])
    plot_spectrum(mixture_spectrum_dict,'step_02_normalize_spectra.png')

    # light source-> photons:
    # TODO: laser power, NA, n, wavelength, exposure time
    # Amplify the mixture spectrum
    mixture_spectrum_dict = amplify(mixture_spectrum_dict, spectrum_amplifying_factor = config["spectrum_amplifying_factor"])
    plot_spectrum(mixture_spectrum_dict,'step_03_amplified_spectra.png')
   
    # create a baseline with different options:
    baseline_dict = create_baseline(mixture_spectrum_dict, baseline_pars = config["baseline_pars"])
    # amplify baseline, not linearly with the spectrum signal
    baseline_dict = amplify(baseline_dict, spectrum_amplifying_factor = config["baseline_amplifying_factor"]) 
   
   # add baseline to the spectrum
    mixture_spectrum_dict = add_baseline(mixture_spectrum_dict, baseline_dict = baseline_dict)
    plot_spectrum(mixture_spectrum_dict,'step_04_add_baseline.png')
   
   # smear a guassian curve on the spectrum to simulate the optical abberration
    mixture_spectrum_dict = convolute_kernel(mixture_spectrum_dict, kernel_std = config["abbrration_kernel_std"])
    plot_spectrum(mixture_spectrum_dict,'step_05_add_abbrration.png')
   
    # add photon shot noise (Possion distribution)
    mixture_spectrum_dict = add_noise(mixture_spectrum_dict, noise_pars = config["photon_shot_noise_pars"])
    plot_spectrum(mixture_spectrum_dict,'step_06_add_photon_shot_noise.png')
    
    # add spikes of cosmic rays:
    mixture_spectrum_dict = add_cosmic_rays(mixture_spectrum_dict, cosmic_ray_pars = config["cosmic_ray_pars"])
    plot_spectrum(mixture_spectrum_dict,'step_07_add_cosmic_ray.png')
    
    # Photons -> Electrons:
    # quantumn efficiency of the detector
    mixture_spectrum_dict = quantumn_efficiency(mixture_spectrum_dict, quantumn_efficiency = config["quantumn_efficiency"])
    plot_spectrum(mixture_spectrum_dict,'step_08_quantumn_eff.png')
    
    # Electron -> Voltage:
    # Dark current shot noise (Possion distribution)
    mixture_spectrum_dict = add_noise(mixture_spectrum_dict, noise_pars = config["dark_current_shot_noise_pars"])
    plot_spectrum(mixture_spectrum_dict,'step_09_add_dark_current_shot_noise.png')
       
    # add photo response non-uniformity (caused by the defects on the semiconductor materials, Gaussian distribution)
    mixture_spectrum_dict = add_noise(mixture_spectrum_dict, noise_pars = config["photo_response_non_uniformity_pars"])
    plot_spectrum(mixture_spectrum_dict,'step_10_add_photon_response_non_uniformity.png')
    # TODO: noise from binning?

    # Voltage -> Counts:
    # add dark signal fixed-pattern noise (Log-nomral distribution)
    mixture_spectrum_dict = add_noise(mixture_spectrum_dict, noise_pars = config["dark_signal_FPN_noise_pars"])
    plot_spectrum(mixture_spectrum_dict,'step_11_dark_signal_FPN_noisea.png')

    # adding instrument shifts (shifting the whole spectrum)
    mixture_spectrum_dict = shift_spectrum(mixture_spectrum_dict, instrument_shift = config["instrument_shift"])
    plot_spectrum(mixture_spectrum_dict,'step_12_instrument_shift.png')

    subprocess.call(["python", "./make_movies.py"])

if __name__ == "__main__":
    main()