import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class ProtocolModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        acquisition_protocol_frame = ttk.Labelframe(self, text='Acquisition Protocol', padding=5)
        acquisition_protocol_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        template_label = ttk.Label(acquisition_protocol_frame, text="Template")
        template_label.grid(row=0, column=0, pady=5, padx=5)
        template_entry = ttk.Entry(acquisition_protocol_frame, width=5)
        template_entry.grid(row=0, column=1, padx=5, pady=5)

    