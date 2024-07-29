import os
import numpy as np
import pandas as pd

# calculate the SNR of the spectra using FSD
def calculate_snr(data):
    # calculate the SNR of the spectra using FSD
    snr = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        snr[i] = np.mean(data[i, 1000:2000]) / np.std(data[i, 1000:2000])
    return snr

