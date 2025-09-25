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

    # Separate patients by staging
    stage0_pids = [pid for pid, stage in patient_staging.items() if stage == "0"]
    stage1_pids = [pid for pid, stage in patient_staging.items() if stage != "0"]

    # Color palettes
    blues = plt.cm.Blues(np.linspace(0.5, 1, max(2, len(stage0_pids))))
    reds = plt.cm.Reds(np.linspace(0.5, 1, max(2, len(stage1_pids))))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    # Plot all spectr of stage 0 patients
    stage_0_spectra = []
    for i, pid in enumerate(stage0_pids):
        for j, (x, y) in enumerate(spectra_by_patient[pid]):
            ax0.plot(x, y, color=blues[2], alpha=0.5, linewidth=0.5)
            stage_0_spectra.append(y)
            if j == 1:  # Only plot the first 50 spectra for stage 0
                break

    # ax0.plot(x, np.mean(stage_0_spectra, axis=0), color=blues[0], linewidth=1, label="Average Control Spectrum")
    # ax0.fill_between(x, np.mean(stage_0_spectra, axis=0) - np.std(stage_0_spectra, axis=0),
    #                 np.mean(stage_0_spectra, axis=0) + np.std(stage_0_spectra, axis=0),
    #                 color=blues[0], alpha=0.2, label="Std Control Spectrum")

    stage_1_spectra = []
    for i, pid in enumerate(stage1_pids):
        for j, (x, y) in enumerate(spectra_by_patient[pid]):
            ax1.plot(x, y, color=reds[2], alpha=0.5, linewidth=0.5)
            stage_1_spectra.append(y)
        
            if j == 1:  # Only plot the first 50 spectra for stage 1
                break

    # ax1.plot(x, np.mean(stage_1_spectra, axis=0), color=reds[0], linewidth=1, label="Average Cancer Spectrum")
    # ax1.fill_between(x, np.mean(stage_1_spectra, axis=0) - np.std(stage_1_spectra, axis=0),
    #                 np.mean(stage_1_spectra, axis=0) + np.std(stage_1_spectra, axis=0),
    #                 color=reds[0], alpha=0.2, label="Std Cancer Spectrum")
    # Add legends
    ax0.legend(loc='upper right', fontsize=16, frameon=False)
    ax1.legend(loc='upper right', fontsize=16, frameon=False)
    leg0 = ax0.get_legend()
    for text in leg0.get_texts():
        text.set_color('white')

    leg1 = ax1.get_legend()
    for text in leg1.get_texts():
        text.set_color('white')
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
    fig.savefig("without_preprocessing_one_cancer_patient_one_control_spectra_ptrptocessed_without_average.png", dpi=300, transparent=True)

    # avg_diff_y = np.mean(stage_1_spectra, axis=0) - np.mean(stage_0_spectra, axis=0)
    # # std_diff_y = np.std(stage_1_spectra, axis=0) - np.std(stage_0_spectra, axis=0)
    # greens = plt.cm.Greens(np.linspace(0.5, 1, max(2, 1)))
    # fig, ax2 = plt.subplots(figsize=(8, 6))
    # ax2.plot(x, avg_diff_y, color=greens[0], label=f"Mean(Cancer - Control)")
    # # ax2.fill_between(x, avg_diff_y - std_diff_y, avg_diff_y + std_diff_y, color='w', alpha=0.2, label="Std")
    # ax2.axhline(0, color='white', linestyle='--', linewidth=0.5)
    # ax2.set_facecolor("none")
    
    # # Set all ticks, labels, and axis to white
    # ax2.tick_params(axis='both', colors='white')
    # ax2.xaxis.label.set_color('white')
    # ax2.yaxis.label.set_color('white')
    # ax2.title.set_color('white')
    # for spine in ax2.spines.values():
    #     spine.set_edgecolor('white')
    # leg = ax2.legend(frameon=False, fontsize=14, loc="best", facecolor='none')
    # for text in leg.get_texts():
    #     text.set_color("white")
        
    # ax2.set_title("Difference between mean of cancer and control spectra", fontsize=18)
    # ax2.set_ylabel("Delta Normalized Intensity (a.u.)", fontsize=16)
    # ax2.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=16)
    # # ax2.set_ylim(-0.35, 0.35)  # Set y-limits for better visibility
    # fig.patch.set_alpha(0.0)  # Make the figure background transparent
    # plt.tight_layout()
    # plt.savefig(f"Average_Differece_spectra_cancer_control.png", dpi=300, transparent=True)





    y_diff = []
    mid = len(stage_0_spectra) // 2
    for i in range(len(stage_0_spectra)):
        for j in range(len(stage_1_spectra)):
            # subtract each mean spectrum between stage 0 and stage 1
            example_patient_stage0 = stage_0_spectra[i]
            example_patient_stage1 = stage_1_spectra[j]
            diff = example_patient_stage1 - example_patient_stage0
            y_diff.append(diff)

    avg_diff_y = np.mean(y_diff, axis=0)
    std_diff_y = np.std(y_diff, axis=0)


    greens = plt.cm.Greens(np.linspace(0.5, 1, max(2, 1)))
    fig, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(x, avg_diff_y, color=greens[0], label=f"Mean(Cancer - Control)")
    ax3.fill_between(x, avg_diff_y - std_diff_y, avg_diff_y + std_diff_y, color='w', alpha=0.2, label="Std")
    # ax3.axhline(0, color='white', linestyle='--', linewidth=0.5)
    ax3.set_facecolor("none")
    
    # Set all ticks, labels, and axis to white
    ax3.tick_params(axis='both', colors='white')
    ax3.xaxis.label.set_color('white')
    ax3.yaxis.label.set_color('white')
    ax3.title.set_color('white')
    for spine in ax3.spines.values():
        spine.set_edgecolor('white')
    leg = ax3.legend(frameon=False, fontsize=14, loc="best", facecolor='none')
    for text in leg.get_texts():
        text.set_color("white")
        
    ax3.set_title("Difference between mean of cancer and control spectra", fontsize=18)
    ax3.set_ylabel("Delta Normalized Intensity (a.u.)", fontsize=16)
    ax3.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=16)
    # ax2.set_ylim(-0.35, 0.35)  # Set y-limits for better visibility
    fig.patch.set_alpha(0.0)  # Make the figure background transparent
    plt.tight_layout()
    plt.savefig(f"without_preprocessing_first_pair_Differece_spectra_cancer_control.png", dpi=300, transparent=True)




# -------------------------------------------------------------------------


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_train")
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
                                        baseline_correction=False,
                                        remove_cosmic_rays= False,
                                        normalization= False,
                                        smoothing=False)
    
    dataset = HNC_Dataset(data_folder, metadata_file, preprocessor, augmentor = None)

    # Plot all patient spectra  
    plot_all_patient_spectra(dataset)