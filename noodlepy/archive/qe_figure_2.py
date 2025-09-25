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

    len_stage0 = len(stage0_pids)
    len_stage1 = len(stage1_pids)

    for stage_1_pid in range(len_stage1):
        for stage_0_pid in range(len_stage0):

            # first patient from stage 0 list and first patient from stage 1 list
            example_patient_stage0 = stage0_pids[stage_0_pid]
            example_patient_stage1 = stage1_pids[stage_1_pid]

            print(f"Example patient stage 0: {example_patient_stage0}")
            print(f"Example patient stage 1: {example_patient_stage1}")

            # get all example spectra for stage 0 and stage 1
            example_spectra_stage0 = spectra_by_patient[example_patient_stage0]
            example_spectra_stage1 = spectra_by_patient[example_patient_stage1]

            # subtract each example spectrum between stage 0 and stage 1
            example_spectra_diff = []
            for (x0, y0), (x1, y1) in zip(example_spectra_stage0, example_spectra_stage1):
                if not np.array_equal(x0, x1):
                    raise ValueError("Raman shifts do not match between stage 0 and stage 1.")
                diff_y = y1 - y0
                example_spectra_diff.append((x0, diff_y))
            
            # Calculate the average and std of the differences
            diff_y_stack = np.stack([y for _, y in example_spectra_diff], axis=0)
            avg_diff_y = np.mean(diff_y_stack, axis=0)
            std_diff_y = np.std(diff_y_stack, axis=0)
            x_diff = example_spectra_diff[0][0]  # x-values are the same for all

            # difference between average of example patient stage 0 and stage 1
            avg_y_stage0 = patient_avg[example_patient_stage0][1]
            avg_y_stage1 = patient_avg[example_patient_stage1][1]
            diff_avg_y = avg_y_stage1 - avg_y_stage0


            # Color palettes
            blues = plt.cm.Blues(np.linspace(0.5, 1, max(2, len(stage0_pids))))
            reds = plt.cm.Reds(np.linspace(0.5, 1, max(2, len(stage1_pids))))
            greens = plt.cm.Greens(np.linspace(0.5, 1, max(2, len(stage1_pids))))

            # Plot stage 0 patients (blue shades)
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
            pid = example_patient_stage0
            x, avg_y = patient_avg[pid]
            std_y = patient_std[pid]
            label = f"Mean Spectrum of Subject#{pid}"
            
            # Plot all spectra for stage 0 patient
            # for i, (x_i, y_i) in enumerate(spectra_by_patient[pid]):
            #     ax0.plot(x_i, y_i, color=blues[0], linewidth=0.5)
            # ax0.plot(x_i, y_i, color=blues[0], linewidth=0.5, label=label)
            ax0.plot(x, avg_y, color=blues[0], label=label)
            ax0.fill_between(x, avg_y - std_y, avg_y + std_y, color=blues[0], alpha=0.2)
            
            pid = example_patient_stage1 
            x, avg_y = patient_avg[pid]
            std_y = patient_std[pid]
            label = f"Mean Spectrum of Subject#{pid}"
            
            # Plot all spectra for stage 1 patient
            # for i, (x_i, y_i) in enumerate(spectra_by_patient[pid]):
            #     ax1.plot(x_i, y_i, color=reds[0], linewidth=0.5)
            # ax1.plot(x_i, y_i, color=reds[0], linewidth=0.5, label=label)

            ax1.plot(x, avg_y, color=reds[0], label=label)
            ax1.fill_between(x, avg_y - std_y, avg_y + std_y, color=reds[0], alpha=0.2)
            # Axis labels and legend with bigger font sizes
            ax1.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=14)
            ax0.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
            ax1.set_ylabel("Normalized Intensity (a.u.)", fontsize=14)
            ax0.set_title("Example Control Spectra", fontsize=16)
            ax1.set_title("Example Cancer Spectra", fontsize=16)
            leg0 = ax0.legend(frameon=False, fontsize=12, loc="upper right")
            for text in leg0.get_texts():
                text.set_color("white")
            leg1 = ax1.legend(frameon=False, fontsize=12, loc="upper right")
            for text in leg1.get_texts():
                text.set_color("white")
            # Fully transparent figure/axes for black-slide presentations
            ax0.set_facecolor("none")
            ax1.set_facecolor("none")
            # Set ticks, labels, and axis to white
            for ax in (ax0, ax1):
                ax.tick_params(axis='both', colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
            
            fig.patch.set_alpha(0.0)
            plt.tight_layout()
            plt.savefig(f"Average_spectra_patient{example_patient_stage1}_stage1_patient{example_patient_stage0}_stage0.png", dpi=300, transparent=True)

            # Plot the average spectra for stage 0 and stage 1
            fig, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(x_diff, avg_diff_y, color=greens[0], label=f"Cancer Subject#{example_patient_stage1} - Control Subject#{example_patient_stage0}")
            # ax2.fill_between(x_diff, avg_diff_y - std_diff_y, avg_diff_y + std_diff_y, color='w', alpha=0.2)
            ax2.axhline(0, color='white', linestyle='--', linewidth=0.5)
            ax2.set_facecolor("none")
            
            # Set all ticks, labels, and axis to white
            ax2.tick_params(axis='both', colors='white')
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
            ax2.title.set_color('white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('white')
            leg = ax2.legend(frameon=False, fontsize=14, loc="upper right", facecolor='none')
            for text in leg.get_texts():
                text.set_color("white")
                
            ax2.set_title("Difference between mean of cancer and control spectra", fontsize=18)
            ax2.set_ylabel("Delta Normalized Intensity (a.u.)", fontsize=16)
            ax2.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=16)
            ax2.set_ylim(-0.3, 0.3)  # Set y-limits for better visibility
            fig.patch.set_alpha(0.0)  # Make the figure background transparent
            plt.tight_layout()
            plt.savefig(f"Differece_spectra_patient{example_patient_stage1}_stage1_patient{example_patient_stage0}_stage0.png", dpi=300, transparent=True)

        # If you need a transparent PNG for slides, uncomment:
        # fig.savefig("all_patient_spectra.png", dpi=300, transparent=True)
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "cosmic_ray_removed","test")
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