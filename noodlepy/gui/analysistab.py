import os
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from noodlepy.utils.spectrum import Spectrum
from tkinter import filedialog, Text, END, RIGHT, LEFT, Y, BOTH, VERTICAL
from ttkbootstrap.constants import *
import ttkbootstrap as ttk
import yaml
import copy
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor

class AnalysisGUI(ttk.Frame):
    def __init__(self, parent, file_types='*.txt'):
        super().__init__(parent)
        self.file_types = file_types if file_types else ["*"]
        self.create_widgets()

    def create_widgets(self):
        # Create frame for loading data files
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        data_frame = ttk.Labelframe(main_frame, text='Load Data', padding=5)
        data_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.folder_path_entry = ttk.Entry(data_frame)
        self.folder_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.browse_button = ttk.Button(data_frame, text="Browse", command=self.browse_folder)
        self.browse_button.grid(row=0, column=1, padx=5, pady=5)

        self.select_all_button = ttk.Button(data_frame, text="Select All", command=self.select_all)
        self.select_all_button.grid(row=0, column=2, padx=5, pady=5)

        self.scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL)
        self.file_listbox = ttk.Treeview(data_frame, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.scrollbar.grid(row=1, column=3, sticky="ns")

        self.fig_raw, self.ax_raw = plt.subplots(figsize=(4, 2), tight_layout=True)
        self.ax_raw.set_xlabel('Raman Shift (cm^-1)', fontsize=4)
        self.ax_raw.set_ylabel('Intensity',fontsize=4)
        self.ax_raw.set_title('Spectra',fontsize=6)
        self.ax_raw.tick_params(axis='both', which='major', labelsize=4)
        self.ax_raw.tick_params(axis='both', which='minor', labelsize=4)
        self.ax_raw.margins(x=0)
        self.ax_raw.margins(y=0)
        self.ax_raw.figure.tight_layout(pad=0.1)
        self.ax_raw.spines['top'].set_visible(False)
        self.ax_raw.spines['right'].set_visible(False)

        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=main_frame)
        self.canvas_raw.get_tk_widget().grid(row=0, column=4, columnspan=2, sticky="ew", padx=5, pady=5)

        calibration_frame = ttk.Labelframe(main_frame, text='Calibration', padding=5)
        calibration_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.calibration_file1_entry = ttk.Entry(calibration_frame)
        self.calibration_file1_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.load_calibration_button = ttk.Button(calibration_frame, text="Load Neon File",width=15)
        self.load_calibration_button.grid(row=0, column=1, padx=5, pady=5)

        self.calibration_file2_entry = ttk.Entry(calibration_frame)
        self.calibration_file2_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.load_calibration_button = ttk.Button(calibration_frame, text="Load White Lamp File", width=15)
        self.load_calibration_button.grid(row=1, column=1, padx=5, pady=5)

        self.calibration_option_1 = ttk.Checkbutton(calibration_frame, text="X axis", variable=tk.BooleanVar(value=0))
        self.calibration_option_1.grid(row=2, column=0, padx=5, pady=5)

        self.calibration_option_2 = ttk.Checkbutton(calibration_frame, text="Y axis", variable=tk.BooleanVar())
        self.calibration_option_2.grid(row=3, column=0, padx=5, pady=5)

        self.calibrate_button = ttk.Button(calibration_frame, text="Calibrate", width=15)
        self.calibrate_button.grid(row=2, column=1, rowspan=2, padx=5, pady=5)


        preprocess_frame = ttk.Labelframe(main_frame, text='Preprocess', padding=5)
        preprocess_frame.grid(row=3, column=1, columnspan=3, sticky="ew", padx=5, pady=5)    

        self.config_path_entry = ttk.Entry(preprocess_frame)
        self.config_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # load button
        self.load_button = ttk.Button(preprocess_frame, text="Load Config", command=self.load_config, width=15)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # create a checkbox to select the calibration option
        self.cropping_var = tk.BooleanVar(value=False)
        self.cropping = ttk.Checkbutton(preprocess_frame, text="Cropping", variable=self.cropping_var, command=self.plot_processed_spectra)
        self.cropping.grid(row=1, column=0, sticky="ew", padx=5, pady=5)


        self.baseline_correction_var = tk.BooleanVar(value=False)
        self.baseline_correction = ttk.Checkbutton(preprocess_frame, text="Baseline Correction", variable=self.baseline_correction_var, command=self.plot_processed_spectra)
        self.baseline_correction.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.smoothing_var = tk.BooleanVar(value=False)
        self.smoothing = ttk.Checkbutton(preprocess_frame, text="Smoothing", variable=self.smoothing_var, command=self.plot_processed_spectra)
        self.smoothing.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.normalization_var = tk.BooleanVar(value=False)
        self.normalization = ttk.Checkbutton(preprocess_frame, text="Normalization", variable=self.normalization_var, command=self.plot_processed_spectra)
        self.normalization.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.despiking_var = tk.BooleanVar(value=False)
        self.despiking = ttk.Checkbutton(preprocess_frame, text="Despiking", variable=self.despiking_var, command=self.plot_processed_spectra)
        self.despiking.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

        self.fig_processed, self.ax_processed = plt.subplots(figsize=(4, 2), tight_layout=True)
        self.ax_processed.set_xlabel('Raman Shift (cm^-1)', fontsize=4)
        self.ax_processed.set_ylabel('Intensity',fontsize=4)

        self.ax_processed.tick_params(axis='both', which='major', labelsize=4)
        self.ax_processed.tick_params(axis='both', which='minor', labelsize=4)
        self.ax_processed.margins(x=0)
        self.ax_processed.margins(y=0)
        self.ax_processed.figure.tight_layout(pad=0.1)
        self.ax_processed.spines['top'].set_visible(False)
        self.ax_processed.spines['right'].set_visible(False)
        self.canvas_processed = FigureCanvasTkAgg(self.fig_processed, master=main_frame)
        self.canvas_processed.get_tk_widget().grid(row=1, column=4, columnspan=2, sticky="nsew", padx=5, pady=5)

        # bind the selected file to the plot
        self.file_listbox.bind('<<TreeviewSelect>>', self.update_plots)

    def update_plots(self, event):
        self.plot_raw_spectra()
        self.plot_processed_spectra()

    def browse_folder(self):
        # set the starting folder path to the current folder path
        current_folder = os.getcwd()+ '/noodlepy' +'/data'
        folder_path = filedialog.askdirectory(initialdir=current_folder)
        if folder_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, folder_path)
            self.load_files()

    def load_files(self):
        self.folder_path = self.folder_path_entry.get()
        if self.folder_path:
            # get all files with the specified file types in the folder and subfolders
            self.file_listbox.delete(*self.file_listbox.get_children())
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    if any(file.endswith(ft) for ft in self.file_types):
                        file_path = os.path.join(root, file)
                        # Use the full path as the item identifier (iid) and display only the file name
                        self.file_listbox.insert('', 'end', iid=file_path, text=file)

    def select_all(self):
        for item in self.file_listbox.get_children():
            self.file_listbox.selection_add(item)

    def plot_raw_spectra(self):
        # plot the selected spectra in the preview window
        selected_files = self.file_listbox.selection()
        # plot the all selected spectra in one plot
        list_of_spectra = []
        self.ax_raw.clear()

        for file in selected_files:
            # get the file path
            file_path = file
            spectra = Spectrum.load_from_file(file_path)
            list_of_spectra.append(spectra)

            for spectrum in spectra:
                raman_shift_cm = spectrum.raman_shift_cm
                intensity = spectrum.intensity
                self.ax_raw.plot(raman_shift_cm, intensity, linewidth=0.5)

        self.ax_raw.set_xlabel('Raman Shift (cm^-1)', fontsize=4)
        self.ax_raw.set_ylabel('Intensity',fontsize=4)
        self.ax_raw.set_title('Raw Spectra',fontsize=6)
        # change font size of the x y labels
        self.ax_raw.tick_params(axis='both', which='major', labelsize=4)
        self.ax_raw.tick_params(axis='both', which='minor', labelsize=4)

        # Remove excessive whitespace around the plot
        self.ax_raw.margins(x=0)
        self.ax_raw.margins(y=0)
        
        # Adjust subplot parameters to give some padding around the plot
        self.ax_raw.figure.tight_layout(pad=0.1)
        
        # Remove top and right spines for a cleaner look
        self.ax_raw.spines['top'].set_visible(False)
        self.ax_raw.spines['right'].set_visible(False)

        self.canvas_raw.draw()

        self.list_of_spectra_being_previewed = list_of_spectra

    def plot_processed_spectra(self):
        cropping = self.cropping_var.get()
        baseline_correction = self.baseline_correction_var.get()
        remove_cosmic_rays = self.despiking_var.get()
        normalization = self.normalization_var.get()
        smoothing = self.smoothing_var.get()

        preprocessor = SpectrumPreprocessor(
            cropping= cropping,
            baseline_correction=baseline_correction,
            remove_cosmic_rays= remove_cosmic_rays,
            normalization= normalization,
            smoothing= smoothing,
            config_path=self.config_path
        )

        # plot the selected spectra in the preview window
        selected_files = self.file_listbox.selection()
        # plot the all selected spectra in one plot
        list_of_spectra = []
        self.ax_processed.clear()

        for file in selected_files:
            # get the file path
            file_path = file
            spectra = Spectrum.load_from_file(file_path)
            list_of_spectra.append(spectra)

            for spectrum in spectra:
                preprocessed_spectrum = copy.deepcopy(spectrum)
                preprocessed_spectrum = preprocessor.preprocess(preprocessed_spectrum)
                raman_shift_cm = preprocessed_spectrum.raman_shift_cm
                intensity = preprocessed_spectrum.intensity
                self.ax_processed.plot(raman_shift_cm, intensity, linewidth=0.5)

        self.ax_processed.set_xlabel('Raman Shift (cm^-1)', fontsize=4)
        self.ax_processed.set_ylabel('Intensity',fontsize=4)
        self.ax_processed.set_title('Processed Spectra',fontsize=6)
        # change font size of the x y labels
        self.ax_processed.tick_params(axis='both', which='major', labelsize=4)
        self.ax_processed.tick_params(axis='both', which='minor', labelsize=4)

        # Remove excessive whitespace around the plot
        self.ax_processed.margins(x=0)
        self.ax_processed.margins(y=0)
        
        # Adjust subplot parameters to give some padding around the plot
        self.ax_processed.figure.tight_layout(pad=0.1)
        
        # Remove top and right spines for a cleaner look
        self.ax_processed.spines['top'].set_visible(False)
        self.ax_processed.spines['right'].set_visible(False)

        self.canvas_processed.draw()


    def load_config(self):
        # set the starting folder path to the current folder path
        current_folder = os.getcwd()+ '/noodlepy' +'/config'
        # select the default config file
        self.config_path = filedialog.askopenfilename(initialdir=current_folder)

        if self.config_path:
            self.config_path_entry.delete(0, tk.END)
            self.config_path_entry.insert(0, self.config_path)
            # load the config.yml file


if __name__ == '__main__':
    root = ttk.Window()
    # set the ttkbootstrap theme to superhero
    root.style.theme_use('superhero')
    app = AnalysisGUI(root)
    app.pack()
    root.mainloop()