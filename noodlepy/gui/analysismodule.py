import os
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from noodlepy.utils.spectrum import Spectrum
from tkinter import filedialog, Text, END, RIGHT, LEFT, Y, BOTH, VERTICAL
from ttkbootstrap.constants import *



class FileBrowser(ttk.Frame):
    def __init__(self, parent, title, file_types=None):
        super().__init__(parent)
        self.parent = parent
        self.title = title
        self.file_types = file_types if file_types else ["*"]
        self.create_widgets()

    def create_widgets(self):

        self.folder_path_entry = ttk.Entry(self)
        self.folder_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.browse_button = ttk.Button(self, text="Browse", command=self.browse_folder)
        self.browse_button.grid(row=0, column=1, padx=5, pady=5)

        self.select_all_button = ttk.Button(self, text="Select All", command=self.select_all)
        self.select_all_button.grid(row=0, column=2, padx=5, pady=5)

        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.file_listbox = ttk.Treeview(self, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.file_listbox.yview)
        
        self.file_listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.scrollbar.grid(row=1, column=3, sticky="ns")

        self.fig, self.ax = plt.subplots(figsize=(4, 2), tight_layout=True)
        # add x and y labels to the plot
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title('Spectra')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=4, columnspan=2, rowspan=2, sticky="nsew", padx=5, pady=5)

        # # add import button
        # self.import_button = ttk.Button(self, text="Import Selected Spectra")
        # # place the import button above the preview frame in the mdidle
        # self.import_button.grid(row=0, column=4, columnspan=2, padx=5, pady=5)

        # bind the selected file to the plot
        self.file_listbox.bind('<<TreeviewSelect>>', self.plot_spectra)

        # Configure the grid to expand properly
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(1, weight=1)

        # place the frame in the parent
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

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

        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title('Spectra')
        self.canvas.draw()

        self.list_of_spectra_being_previewed = list_of_spectra

class Preprocess(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        self.folder_path_entry = ttk.Entry(self)
        self.folder_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # load button
        self.load_button = ttk.Button(self, text="Load Config", command=self.load_config)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # Configure the grid to expand properly
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(1, weight=1)

        # place the frame in the parent
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)


    def load_config(self):
        # set the starting folder path to the current folder path
        current_folder = os.getcwd()+ '/noodlepy' +'/config'
        folder_path = filedialog.askdirectory(initialdir=current_folder)
        if folder_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, folder_path)
            self.load_files()

class Stage():
    def __init__(self, parent):
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        self.stage_frame = ttk.Frame(self.parent)
        self.stage_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)

        self.stage_label = ttk.Label(self.stage_frame, text="Stage")
        self.stage_label.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.stage_entry = ttk.Entry(self.stage_frame)
        self.stage_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.stage_button = ttk.Button(self.stage_frame, text="Load Config")
        self.stage_button.grid(row=0, column=2, padx=5, pady=5)

        self.stage_frame.columnconfigure(0, weight=1)
        self.stage_frame.columnconfigure(1, weight=0)
        self.stage_frame.columnconfigure(2, weight=0)

        self.stage_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)