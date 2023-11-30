import itertools
import numpy as np
import yaml

# Random sampling with Numpy
# https://numpy.org/doc/stable/reference/random/index.html

# Sampling a random uniform distribution
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html

# Sampling from a normal distribution
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

# Log normal
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html


# TODO: Check if shot noise should be a strictly positive distribution.  Maybe Poisson
# TODO: Make different sources of noise explicit
# TODO: Define the range of the parameters
# TODO: Define the distribution of the parameters
# TODO: Decide the structure of the configuration file
# TODO: Read the pickle file and generate spectra database
# TODO: Test the distribution of the parameters
# TODO: Automate the experiment

# Load the original configuration file
with open('test_config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Get all combinations of parameters
parameters = [config['exp_conditions']['spectrum_range_pars'],
              config['exp_conditions']['spectrum_amplifying_factor'],
              config['exp_conditions']['baseline_amplifying_factor'],
              config['exp_conditions']['shot_noise_factor'],
              config['exp_conditions']['instrumment_shift'],
              config['exp_conditions']['abbreviation_pars']['kernel_std'],
              config['exp_conditions']['cosmic_ray_pars']['spike_num'],
              config['exp_conditions']['cosmic_ray_pars']['spike_amplitude'],
]

combinations = list(itertools.product(*parameters))

# Generate a new configuration file for each combination
for i, combination in enumerate(combinations):
    config['exp_conditions']['spectrum_range_pars'] = combination[0]
    config['exp_conditions']['spectrum_amplifying_factor'] = combination[1]
    config['exp_conditions']['baseline_amplifying_factor'] = combination[2]
    config['exp_conditions']['shot_noise_factor'] = combination[3]
    config['exp_conditions']['instrumment_shift'] = combination[4]
    config['exp_conditions']['abbreviation_pars']['kernel_std'] = combination[5]
    config['exp_conditions']['cosmic_ray_pars']['spike_num'] = combination[6]
    config['exp_conditions']['cosmic_ray_pars']['spike_amplitude'] = combination[7]

    with open(f'new_config_{i}.yml', 'w') as file:
        yaml.safe_dump(config, file)