import os
import random
import numpy as np
import torch
from noodlepy.archive.raw_data_hnc_dataset import HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba

def plot_all_patient_spectra(dataset):
    """
    Plot all spectra in an HNC_Dataset using the dataset's __getitem__
    for the y-values (intensity) and the raw Spectrum in dataset.db[idx]
    for patient metadata and the x-axis (raman_shift_cm).

    Parameters
    ----------
    dataset : HNC_Dataset
        A fully-built dataset whose __getitem__ returns a tuple of two
        preprocessed intensity tensors.
    """

    # ------------------------------------------
    # 1. Gather spectra by patient via __getitem__
    # ------------------------------------------
    spectra_by_patient = {}   # pid -> list of (x, y) arrays
    for idx in range(len(dataset)):
        # a) get the y-values (first augmented spectrum) through __getitem__
        y_tensor , _, metadata, raman_shift,  = dataset.__getitem__(idx)         # tuple of two tensors
        y = y_tensor.squeeze().cpu().numpy()


        pid = metadata.get("patient_id", None)  # patient identifier
        if pid is None:
            raise AttributeError("Missing patient identifier on Spectrum.")

        x = np.asarray(raman_shift)  # x-values (Raman shift)

        # store
        spectra_by_patient.setdefault(pid, []).append((x, y))

    # Group spectra by patient and staging
    patient_staging = {}  # pid -> staging
    for idx in range(len(dataset)):
        _, _, metadata, _ = dataset.__getitem__(idx)
        pid = metadata.get("patient_id", None)
        staging = str(metadata.get("staging", "unknown"))
        patient_staging[pid] = staging

    # Take the average and std of all spectra for each patient
    patient_avg = {}
    patient_std = {}
    for pid, xy_pairs in spectra_by_patient.items():
        y_stack = np.stack([y for _, y in xy_pairs], axis=0)
        avg_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        x = xy_pairs[0][0]
        patient_avg[pid] = (x, avg_y)
        patient_std[pid] = std_y

    # Separate patients by staging
    stage0_pids = [pid for pid, stage in patient_staging.items() if stage == "0"]
    stage1_pids = [pid for pid, stage in patient_staging.items() if stage != "0"]

    # Color palettes
    blues = plt.cm.Blues(np.linspace(0.5, 1, max(2, len(stage0_pids))))
    reds = plt.cm.Reds(np.linspace(0.5, 1, max(2, len(stage1_pids))))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot stage 0 patients (blue shades)
    for i, pid in enumerate(stage0_pids):
        x, avg_y = patient_avg[pid]
        std_y = patient_std[pid]
        color = blues[i % len(blues)]
        label = f"Patient {pid} (stage 0)"
        ax0.plot(x, avg_y, color=color, label=label)
        ax0.fill_between(x, avg_y - std_y, avg_y + std_y, color=color, alpha=0.2)

    # Plot stage 1 patients (red shades)
    for i, pid in enumerate(stage1_pids):
        x, avg_y = patient_avg[pid]
        std_y = patient_std[pid]
        color = reds[i % len(reds)]
        label = f"Patient {pid} (stage 1)"
        ax1.plot(x, avg_y, color=color, label=label)
        ax1.fill_between(x, avg_y - std_y, avg_y + std_y, color=color, alpha=0.2)

    # Set axis labels and titles in white
    ax1.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=14, color='white')
    ax0.set_ylabel("Normalized Intensity (a.u.)", fontsize=14, color='white')
    ax1.set_ylabel("Normalized Intensity (a.u.)", fontsize=14, color='white')
    ax0.set_title("Example Control Spectra", fontsize=16, color='white')
    ax1.set_title("Example Cancer Spectra", fontsize=16, color='white')

    # Set tick and spine colors to white
    ax0.tick_params(axis='both', colors='white')
    ax1.tick_params(axis='both', colors='white')
    for ax in [ax0, ax1]:
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    # Fully transparent figure/axes for black-slide presentations
    ax0.set_facecolor("none")
    ax1.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    plt.tight_layout()
    fig.savefig("all_patient_spectra.png", dpi=300, transparent=True)

# -------------------------------------------------------------------------


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "cosmic_ray_removed","train")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    seed = 4
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization= True,
                                        smoothing=True)
    
    dataset = HNC_Dataset(data_folder, metadata_file, preprocessor, augmentor = None)

    # Plot all patient spectra  
    plot_all_patient_spectra(dataset)