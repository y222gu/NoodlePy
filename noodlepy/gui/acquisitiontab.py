import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
import tkinter as tk
from noodlepy.gui.liveviewmodule import LiveViewModule
from noodlepy.gui.wasatchmodule import Wasatchmodule
from noodlepy.gui.stagecontrolmodule import StageControlModule
from noodlepy.gui.autofocusmodule import AutoFocusModule
class AcquisitionGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):

        # Create the stage control frame and pack it to the left
        stage_control_frame = StageControlModule(self)
        stage_control_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")

        middle_frame = ttk.Frame(self)
        middle_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        
        # Create the autofocus frame and pack it to the left of the remaining space
        autofocus_frame = AutoFocusModule(middle_frame)
        autofocus_frame.grid(row=0, column=0, sticky="nsew")

        # # Create the live spectrum frame and pack it below the autofocus frame
        live_spectrum_frame = Wasatchmodule(middle_frame)
        live_spectrum_frame.grid(row=1, column=0, sticky="nsew")

        # # Create the Acquisition protocol frame and pack it below the live spectrum frame
        acquisition_protocol_frame = ttk.Labelframe(middle_frame, text='Acquisition Protocol', padding=5)
        acquisition_protocol_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        template_label = ttk.Label(acquisition_protocol_frame, text="Template")
        template_label.grid(row=0, column=0, pady=5, padx=5)
        template_entry = ttk.Entry(acquisition_protocol_frame, width=5)
        template_entry.grid(row=0, column=1, padx=5, pady=5)



        # # Create the live view frame and pack it to the rightmost space
        live_view_frame = LiveViewModule(self)
        live_view_frame.grid(row=0, column=2, rowspan=2, sticky="nsew")


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = AcquisitionGUI(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
