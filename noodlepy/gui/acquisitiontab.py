import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
import tkinter as tk
from noodlepy.gui.liveviewmodule import LiveViewModule
from noodlepy.gui.wasatchmodule import Wasatchmodule
from noodlepy.gui.stagecontrolmodule import StageControlModule
from noodlepy.gui.autofocusmodule import AutoFocusModule
from noodlepy.gui.protocolmodule import ProtocolModule
from threading import Thread

class AcquisitionGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):

        stage_control_frame = StageControlModule(self)
        stage_control_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")

        middle_frame = ttk.Frame(self)
        middle_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        
        autofocus_frame = AutoFocusModule(middle_frame)
        autofocus_frame.grid(row=0, column=0, sticky="nsew")

        live_spectrum_frame = Wasatchmodule(middle_frame)
        live_spectrum_frame.grid(row=1, column=0, sticky="nsew")

        acquisition_protocol_frame = ProtocolModule(middle_frame)
        acquisition_protocol_frame.grid(row=2, column=0, sticky="nsew")

        live_view_frame = LiveViewModule(self)
        live_view_frame.grid(row=0, column=2, rowspan=2, sticky="nsew")

if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = AcquisitionGUI(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
