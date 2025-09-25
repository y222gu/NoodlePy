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

    # Take the average and std of all spectra for each patient and separate by staging
    patient_avg_stage0 = {}
    patient_std_stage0 = {}
    patient_avg_stage1 = {}
    patient_std_stage1 = {}
    for pid, xy_pairs in spectra_by_patient.items():
        y_stack = np.stack([y for _, y in xy_pairs], axis=0)
        avg_y = np.mean(y_stack, axis=0)
        std_y = np.std(y_stack, axis=0)
        x = xy_pairs[0][0]
        if patient_staging.get(pid, "unknown") == "0":
            patient_avg_stage0[pid] = (x, avg_y)
            patient_std_stage0[pid] = std_y
        else:
            patient_avg_stage1[pid] = (x, avg_y)
            patient_std_stage1[pid] = std_y

    diff = []
    for stage_1_pid in patient_avg_stage1.keys():
        for stage_0_pid in patient_avg_stage0.keys():

            # subtract each mean spectrum between stage 0 and stage 1
            example_patient_stage0 = patient_avg_stage0[stage_0_pid]
            example_patient_stage1 = patient_avg_stage1[stage_1_pid]

            x_diff = example_patient_stage1[0]
            y_diff = example_patient_stage1[1] - example_patient_stage0[1]
            diff.append(y_diff)

    avg_diff_y = np.mean(diff, axis=0)
    std_diff_y = np.std(diff, axis=0)

    greens = plt.cm.Greens(np.linspace(0.5, 1, max(2, 1)))
    fig, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(x_diff, avg_diff_y, color=greens[0], label=f"Mean(Cancer - Control)")
    ax2.fill_between(x_diff, avg_diff_y - std_diff_y, avg_diff_y + std_diff_y, color='w', alpha=0.2, label="Std")
    ax2.axhline(0, color='white', linestyle='--', linewidth=0.5)
    ax2.set_facecolor("none")
    
    # Set all ticks, labels, and axis to white
    ax2.tick_params(axis='both', colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')
    leg = ax2.legend(frameon=False, fontsize=14, loc="best", facecolor='none')
    for text in leg.get_texts():
        text.set_color("white")
        
    ax2.set_title("Difference between mean of cancer and control spectra", fontsize=18)
    ax2.set_ylabel("Delta Normalized Intensity (a.u.)", fontsize=16)
    ax2.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=16)
    ax2.set_ylim(-0.35, 0.35)  # Set y-limits for better visibility
    fig.patch.set_alpha(0.0)  # Make the figure background transparent
    plt.tight_layout()
    plt.savefig(f"Differece_spectra_cancer_control.png", dpi=300, transparent=True)

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