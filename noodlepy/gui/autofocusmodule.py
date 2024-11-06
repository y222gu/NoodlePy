import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import numpy as np
from ctypes import *
import time
from noodlepy.gui.wasatchmodule import WasatchManager
from tkinter import StringVar

class NanoDrive:
    def __init__(self):
        # Initialize DLL and handle
        self.dll_path = os.path.join(os.getcwd(),"noodlepy",'dlls','Madlib.dll')
        self.axis = c_uint(3) # Move along Z-axis
        self.mcldll = CDLL(self.dll_path)
        self.mcldll.MCL_ReleaseHandle.restype = None
        self.mcldll.MCL_SingleReadN.restype = c_double
        self.handle = self.mcldll.MCL_InitHandle()
        print("MCL Handle = ", self.handle)

    def initialize_position(self):
        pos = c_double(0)
        error = self.mcldll.MCL_SingleWriteN(pos, self.axis, self.handle)
        if error != 0:
            raise RuntimeError(f"Error initializing position: {error}")
        time.sleep(0.025)  # Wait for nanopositioner to settle
        position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
        print("Initial Position = ", position)

    def get_position(self):
        position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
        return position

    def move_to_position(self, z_pos):
        pos = c_double(z_pos)
        error = self.mcldll.MCL_SingleWriteN(pos, self.axis, self.handle)
        if error != 0:
            print("Error =", error)
        time.sleep(0.025)  # Wait for nanopositioner to settle

    def close(self):
        self.mcldll.MCL_ReleaseHandle(self.handle)
        plt.ioff()
        plt.show()
        print("NanoDrive shutdown")

class AutoFocusModule(ttk.Frame):
    def __init__(self, parent, wasatch_manager):
        super().__init__(parent)
        self.nano_drive = NanoDrive()
        self.wasatch_manager = wasatch_manager
        self.create_widgets()

    def create_widgets(self):

        self.nanodrive_frame = ttk.Labelframe(self, text='Auto Focus', padding=5)
        self.nanodrive_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)

        self.autofocus_fig, self.entropy_ax, self.intensity_ax = self.initialize_autofocus_figure()
        self.autofocus_canvas = FigureCanvasTkAgg(self.autofocus_fig, master=self.nanodrive_frame)
        self.autofocus_canvas.draw()
        self.autofocus_canvas.get_tk_widget().grid(row=0, column=0, columnspan=5, rowspan=2, sticky="nsew", padx=5, pady=5)

        # create initial values for autofocus parameters
        self.autofocus_step_size_var = StringVar()
        self.autofocus_step_size_var.set(10)
        self.autofocus_num_rep_var = StringVar()
        self.autofocus_num_rep_var.set(5)
        self.autofocus_range_low_var = StringVar()
        self.autofocus_range_low_var.set(0)
        self.autofocus_range_high_var = StringVar()
        self.autofocus_range_high_var.set(100)
        
        # Create the entries for the autofocus parameters
        self.autofocus_step_size_label = ttk.Label(self.nanodrive_frame, text="Number of Steps")
        self.autofocus_step_size_label.grid(row=2, column=0, pady=5, padx=5)
        self.autofocus_step_size_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_step_size_var)
        self.autofocus_step_size_entry.grid(row=2, column=1, padx=5, pady=5)
        self.autofocus_step_size_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_step_size_entry.bind("<Return>", self.update_settings)

        self.autofocus_num_rep_label = ttk.Label(self.nanodrive_frame, text="Number of Reps")
        self.autofocus_num_rep_label.grid(row=2, column=2, pady=5, padx=5)
        self.autofocus_num_rep_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_num_rep_var)
        self.autofocus_num_rep_entry.grid(row=2, column=3, padx=5, pady=5)
        self.autofocus_num_rep_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_num_rep_entry.bind("<Return>", self.update_settings)
    
        self.autofocus_range_label = ttk.Label(self.nanodrive_frame, text="Z Min [um]")
        self.autofocus_range_label.grid(row=3, column=0,pady=5, padx=5)
        self.autofocus_range_low_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_range_low_var)
        self.autofocus_range_low_entry.grid(row=3, column=1, padx=5, pady=5)
        self.autofocus_range_low_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_range_low_entry.bind("<Return>", self.update_settings)

        self.autofocus_range_label = ttk.Label(self.nanodrive_frame, text="Z Max [um]")
        self.autofocus_range_label.grid(row=3, column=2,pady=5, padx=5)
        self.autofocus_range_low_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.autofocus_range_high_var)
        self.autofocus_range_low_entry.grid(row=3, column=3, padx=5, pady=5)
        self.autofocus_range_low_entry.bind("<FocusOut>", self.update_settings)
        self.autofocus_range_low_entry.bind("<Return>", self.update_settings)

        self.autofocus_button = ttk.Button(self.nanodrive_frame, text="Focus", command=self.autofocus, bootstyle="info", width=5)
        self.autofocus_button.grid(row=2, column=4, rowspan=2, padx=5, pady=5, sticky="news")

    def initialize_autofocus_figure(self):
                # Create the figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))

        entropy_ax = axes[0]
        intensity_ax = axes[1]

        entropy_ax.set_xlabel('Z-axis position',fontsize=6, color='white')
        entropy_ax.set_ylabel('Entropy', fontsize=6, color='white')
        entropy_ax.set_title('Entropy', fontsize=8, color='white')

        intensity_ax.set_xlabel('Z-axis position', fontsize=6, color='white')
        intensity_ax.set_ylabel('Intensity', fontsize=6, color='white')
        intensity_ax.set_title('Spectrum', fontsize=8, color='white')

        for ax in entropy_ax, intensity_ax:
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.set_facecolor("none")
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.grid(color='white', linestyle='-', linewidth=0.2)

        plt.tight_layout(pad=0.5)
        fig.patch.set_alpha(0)
        return fig, entropy_ax, intensity_ax

    def autofocus(self):
        # Rough Autofocusing
        min = float(self.autofocus_range_low_var.get())
        max = float(self.autofocus_range_high_var.get())
        step_size = int(self.autofocus_step_size_var.get())
        num_rep = int(self.autofocus_num_rep_var.get())

        z_axis_range = np.linspace(min, max, step_size)
        entropy_list, x1_data, y1_data, int_array = [], [], [], []

        for i, z_pos in enumerate(z_axis_range):
            self.nano_drive.move_to_position(z_pos)
            position, wavelengths, intensities = self.measure_spectra(num_rep)
            ent = self.calculate_entropy(intensities)
            entropy_list.append(ent)
            x1_data.append(position)
            y1_data.append(ent)

            # Plot entropy and spectrum
            self.plot_data(x1_data, y1_data, wavelengths, np.mean(intensities, 0))

        fineMin, fineMax = self.refine_focus_range(entropy_list, z_axis_range)
        self.fine_autofocus(fineMin, fineMax, num_rep)

    def fine_autofocus(self, fineMin, fineMax, num_rep=5):
        z_axis_range = np.linspace(fineMin, fineMax, 10)
        entropy_list, x2_data, y2_data = [], [], []

        for i, z_pos in enumerate(z_axis_range):
            self.nano_drive.move_to_position(z_pos)
            position, wavelengths, intensities = self.measure_spectra(num_rep)
            ent = self.calculate_entropy(intensities)
            entropy_list.append(ent)
            x2_data.append(position)
            y2_data.append(ent)

            # Plot fine focus data
            self.plot_data(x2_data, y2_data, wavelengths, np.mean(intensities, 0))

        focusedPos = z_axis_range[np.argmin(entropy_list)]
        self.nano_drive.move_to_position(focusedPos)
        print("Focused Position =", focusedPos)
        print("Autofocus complete")

    def measure_spectra(self, num_rep):
        intensities, wavelengths = [], []
        wavelengths = self.wasatch_manager.settings.wavelengths
        for _ in range(num_rep):
            spectrum = self.wasatch_manager.get_spectrum()
            intensities.append(spectrum)

        lengths = [len(seq) for seq in intensities]
        print("Lengths of sequences in intensities:", lengths)

        intensities = np.array(intensities).reshape((num_rep, len(wavelengths)))
        position = self.nano_drive.get_position()
        return position, wavelengths, intensities

    def calculate_entropy(self, intensities):
        median_intensities = np.median(intensities, 0)
        normalized_int = median_intensities / np.sum(median_intensities)
        entropy = -np.sum(normalized_int * np.log2(normalized_int))
        return entropy

    def refine_focus_range(self, entropy_list, z_axis_range):
        k = np.argmin(entropy_list)
        fineMin = z_axis_range[np.clip(k - 1, 0, len(z_axis_range) - 1)]
        fineMax = z_axis_range[np.clip(k + 1, 0, len(z_axis_range) - 1)]
        print("The fine range is:", fineMin, fineMax)
        return fineMin, fineMax

    def update_settings(self, event):
        self.autofocus_step_size_var.set(float(self.autofocus_step_size_var.get()))
        self.autofocus_num_rep_var.set(int(self.autofocus_num_rep_var.get()))
        self.autofocus_range_low_var.set(float(self.autofocus_range_low_var.get()))
        self.autofocus_range_high_var.set(float(self.autofocus_range_high_var.get()))

    def plot_data(self, x_data, y_data, wavelengths, intensities):
        self.entropy_ax.clear()
        self.intensity_ax.clear()
        self.entropy_ax.set_xlabel('Z-axis position', fontsize=6, color='white')
        self.entropy_ax.set_ylabel('Entropy', fontsize=6, color='white')
        self.entropy_ax.set_title('Entropy', fontsize=8, color='white')
        self.intensity_ax.set_xlabel('Wavelength (nm)', fontsize=6, color='white')
        self.intensity_ax.set_ylabel('Intensity', fontsize=6, color='white')
        self.intensity_ax.set_title('Spectrum', fontsize=8, color='white')
        self.entropy_ax.grid(color='white', linestyle='-', linewidth=0.2)
        self.intensity_ax.grid(color='white', linestyle='-', linewidth=0.2)
        self.entropy_ax.set_facecolor("none")
        self.intensity_ax.set_facecolor("none")

        self.entropy_ax.tick_params(axis='both', which='major', labelsize=6)
        self.entropy_ax.tick_params(axis='both', which='minor', labelsize=6)
        self.entropy_ax.tick_params(axis='x', colors='white')
        self.entropy_ax.tick_params(axis='y', colors='white')
        self.intensity_ax.tick_params(axis='both', which='major', labelsize=6)
        self.intensity_ax.tick_params(axis='both', which='minor', labelsize=6)
        self.intensity_ax.tick_params(axis='x', colors='white')
        self.intensity_ax.tick_params(axis='y', colors='white')


        self.entropy_ax.plot(x_data, y_data, color='#5bc0de')
        self.entropy_ax.scatter([x_data[-1]], [y_data[-1]], color='red')
        self.intensity_ax.plot(wavelengths, intensities, color='#5bc0de')
        self.intensity_ax.lines[0].set_linewidth(0.5)



        self.autofocus_canvas.draw()
        self.autofocus_canvas.flush_events()  # Process any pending events for real-time updates
        self.update_idletasks()



if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    wasatch_manager = WasatchManager(100, 50)
    app = AutoFocusModule(root, wasatch_manager)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
