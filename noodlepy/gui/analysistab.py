import os
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from noodlepy.utils.spectrum import Spectrum
from tkinter import filedialog, Text, END, RIGHT, LEFT, Y, BOTH, VERTICAL
from ttkbootstrap.constants import *
import ttkbootstrap as ttk

class AnalysisGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        # Create frame for loading data files
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        file_browser = FileBrowser(main_frame, file_types='*.txt')
        file_browser.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        calibration_frame = CalibrationModule(main_frame)
        calibration_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
  
        preprocess_frame = PreprocessModule(main_frame)
        preprocess_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)


class FileBrowser(ttk.Frame):
    def __init__(self, parent, file_types=None):
        super().__init__(parent)
        self.file_types = file_types if file_types else ["*"]
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Labelframe(self, text='Load Data', padding=5)
        main_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.folder_path_entry = ttk.Entry(main_frame)
        self.folder_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.browse_button = ttk.Button(main_frame, text="Browse", command=self.browse_folder)
        self.browse_button.grid(row=0, column=1, padx=5, pady=5)

        self.select_all_button = ttk.Button(main_frame, text="Select All", command=self.select_all)
        self.select_all_button.grid(row=0, column=2, padx=5, pady=5)

        self.scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        self.file_listbox = ttk.Treeview(main_frame, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.scrollbar.grid(row=1, column=3, sticky="ns")

        self.fig, self.ax = plt.subplots(figsize=(4, 2), tight_layout=True)
        self.ax.set_xlabel('Wavelength (nm)', fontsize=4)
        self.ax.set_ylabel('Intensity',fontsize=4)
        self.ax.set_title('Spectra',fontsize=6)
        self.ax.tick_params(axis='both', which='major', labelsize=4)
        self.ax.tick_params(axis='both', which='minor', labelsize=4)
        self.ax.margins(x=0)
        self.ax.margins(y=0)
        self.ax.figure.tight_layout(pad=0.1)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=4, columnspan=2, rowspan=2, sticky="nsew", padx=5, pady=5)

        # bind the selected file to the plot
        self.file_listbox.bind('<<TreeviewSelect>>', self.plot_spectra)

        # # Configure the grid to expand properly
        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=0)
        # self.columnconfigure(2, weight=0)
        # self.rowconfigure(1, weight=1)

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

    def plot_spectra(self, event):
        # plot the selected spectra in the preview window
        selected_files = self.file_listbox.selection()
        # plot the all selected spectra in one plot
        list_of_spectra = []
        self.ax.clear()

        for file in selected_files:
            # get the file path
            file_path = file
            spectra = Spectrum.load_from_file(file_path)
            list_of_spectra.append(spectra)

            for spectrum in spectra:
                wavelength_nm = spectrum.wavelength_nm
                intensity = spectrum.intensity
                self.ax.plot(wavelength_nm, intensity)

        self.ax.set_xlabel('Wavelength (nm)', fontsize=4)
        self.ax.set_ylabel('Intensity',fontsize=4)
        self.ax.set_title('Spectra',fontsize=6)
        # change font size of the x y labels
        self.ax.tick_params(axis='both', which='major', labelsize=4)
        self.ax.tick_params(axis='both', which='minor', labelsize=4)

        # Remove excessive whitespace around the plot
        self.ax.margins(x=0)
        self.ax.margins(y=0)
        
        # Adjust subplot parameters to give some padding around the plot
        self.ax.figure.tight_layout(pad=0.1)
        
        # Remove top and right spines for a cleaner look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.canvas.draw()

        self.list_of_spectra_being_previewed = list_of_spectra

class CalibrationModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Labelframe(self, text='Load Data', padding=5)
        main_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.calibration_file1_entry = ttk.Entry(main_frame)
        self.calibration_file1_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.load_calibration_button = ttk.Button(main_frame, text="Load Neon File")
        self.load_calibration_button.grid(row=0, column=1, padx=5, pady=5)

        self.calibration_file2_entry = ttk.Entry(main_frame)
        self.calibration_file2_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.load_calibration_button = ttk.Button(main_frame, text="Load White Lamp File")
        self.load_calibration_button.grid(row=1, column=1, padx=5, pady=5, )

        self.calibration_option_1 = ttk.Checkbutton(main_frame, text="X axis", variable=tk.BooleanVar(value=0))
        self.calibration_option_1.grid(row=2, column=0, padx=5, pady=5)

        self.calibration_option_2 = ttk.Checkbutton(main_frame, text="Y axis", variable=tk.BooleanVar())
        self.calibration_option_2.grid(row=3, column=0, padx=5, pady=5)

        self.calibrate_button = ttk.Button(main_frame, text="Calibrate")
        self.calibrate_button.grid(row=2, column=1, rowspan=2, padx=5, pady=5)

class PreprocessModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Preprocess', padding=5)
        main_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)    

        self.folder_path_entry = ttk.Entry(main_frame)
        self.folder_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # load button
        self.load_button = ttk.Button(main_frame, text="Load Config", command=self.load_config)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # create a checkbox to select the calibration option
        self.baseline_correction = ttk.Checkbutton(main_frame, text="Baseline Correction", variable=tk.BooleanVar())
        self.baseline_correction.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.smoothing = ttk.Checkbutton(main_frame, text="Smoothing", variable=tk.BooleanVar())
        self.smoothing.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.normalization = ttk.Checkbutton(main_frame, text="Normalization", variable=tk.BooleanVar())
        self.normalization.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.despiking = ttk.Checkbutton(main_frame, text="Despiking", variable=tk.BooleanVar())
        self.despiking.grid(row=4, column=0, sticky="ew", padx=5, pady=5)


    def load_config(self):
        # set the starting folder path to the current folder path
        current_folder = os.getcwd()+ '/noodlepy' +'/config'
        folder_path = filedialog.askdirectory(initialdir=current_folder)
        if folder_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, folder_path)
            self.load_files()


if __name__ == '__main__':
    root = ttk.Window()
    # set the ttkbootstrap theme to superhero
    root.style.theme_use('superhero')
    app = AnalysisGUI(root)
    app.pack()
    root.mainloop()