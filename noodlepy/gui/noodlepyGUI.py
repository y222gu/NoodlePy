from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, Text, END, RIGHT, LEFT, Y, BOTH, VERTICAL
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import pandas as pd
from noodlepy.utils.spectrum import Spectrum
from noodlepy.gui.analysismodule import FileBrowser
from noodlepy.gui.analysismodule import Preprocess

PATH = Path(__file__).parent / 'assets'


class NoodlepyGUI(ttk.Window):

    def __init__(self):
        super().__init__(themename="superhero")
        mainframe = ttk.Frame(self)
        mainframe.pack(fill=BOTH, expand=YES)

        for i in range(3):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(0, weight=1)

        # column 1
        col1 = ttk.Frame(self, padding=10)
        col1.grid(row=0, column=0, sticky=NSEW)

        # Create Notebook for tabs
        notebook = ttk.Notebook(col1)
        notebook.pack(side=TOP, fill=BOTH, expand=YES)

        aquistion_tab = ttk.Frame(notebook)
        notebook.add(aquistion_tab, text='Aquistion')
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text='Analysis')

        # Create frame for loading data files
        data_files_frame = ttk.Labelframe(analysis_tab, text='Load Data', padding=5)
        data_files_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)
        FileBrowser(data_files_frame, 'Data Files', file_types='*.txt')

        calibration_frame = ttk.Labelframe(analysis_tab, text='Calibration', padding=5)
        calibration_frame.pack(side=TOP, fill=BOTH, expand=YES, padx=5, pady=5)

                # create three entry fields for the user to browse and select calibration files
        self.calibration_file1_entry = ttk.Entry(calibration_frame)
        self.calibration_file1_entry.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.load_calibration_button = ttk.Button(calibration_frame, text="Load Neon File")
        self.load_calibration_button.grid(row=2, column=1, padx=5, pady=5)

        self.calibration_file2_entry = ttk.Entry(calibration_frame)
        self.calibration_file2_entry.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.load_calibration_button = ttk.Button(calibration_frame, text="Load White Lamp File")
        self.load_calibration_button.grid(row=3, column=1, padx=5, pady=5, )

        # create a checkbox to select the calibration option
        self.calibration_option_1 = ttk.Checkbutton(calibration_frame, text="X axis", variable=tk.BooleanVar())
        self.calibration_option_1.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.calibration_option_2 = ttk.Checkbutton(calibration_frame, text="Y axis", variable=tk.BooleanVar())
        self.calibration_option_2.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        # create a button to calibrate the spectra
        self.calibrate_button = ttk.Button(calibration_frame, text="Calibrate")
        self.calibrate_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
  
        preprocess_frame = ttk.Labelframe(analysis_tab, text='Preprocess', padding=5)
        preprocess_frame.pack(side=TOP, fill=BOTH, expand=YES, padx=5, pady=5)
        Preprocess(preprocess_frame)

        # Column 2
        col2 = ttk.Frame(self, padding=10)
        col2.grid(row=0, column=1, columnspan=2,sticky=NSEW)


    def callback(self):
        """Demo callback"""
        Messagebox.ok(
            title='Button callback', 
            message="You pressed a button."
        )

if __name__ == '__main__':

    app = NoodlepyGUI()
    app.mainloop()
