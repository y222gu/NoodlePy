import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class AutoFocusModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Auto Focus', padding=5)
        main_frame.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)

        # Create the figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        self.ax1, self.ax2= axes.flatten()

        self.ax1.set_xlabel('Z-axis position',fontsize=6, color='white')
        self.ax1.set_ylabel('Entropy', fontsize=6, color='white')
        self.ax1.set_title('Entropy', fontsize=8, color='white')

        self.ax2.set_xlabel('Z-axis position', fontsize=6, color='white')
        self.ax2.set_ylabel('Intensity', fontsize=6, color='white')
        self.ax2.set_title('Focus', fontsize=8, color='white')

        for ax in axes.flatten():
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

        plt.tight_layout(pad=0.5)
        fig.patch.set_alpha(0)

        # Create a canvas widget for the figure
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Create the entries for the autofocus parameters
        num_steps_label = ttk.Label(main_frame, text="Number of Steps")
        num_steps_label.grid(row=2, column=0, pady=5, padx=5)
        num_steps_entry = ttk.Entry(main_frame, width=5)
        num_steps_entry.grid(row=2, column=1, padx=5, pady=5)

        num_rep_label = ttk.Label(main_frame, text="Number of Reps")
        num_rep_label.grid(row=2, column=2, pady=5, padx=5)
        num_rep_entry = ttk.Entry(main_frame, width=5)
        num_rep_entry.grid(row=2, column=3, padx=5, pady=5)
        
        exposure_time_label = ttk.Label(main_frame, text="Exposure Time [ms]")
        exposure_time_label.grid(row=3, column=0,pady=5, padx=5)
        exposure_time_entry = ttk.Entry(main_frame, width=5)
        exposure_time_entry.grid(row=3, column=1, padx=5, pady=5)

        range_label = ttk.Label(main_frame, text="Z-axis Range [um]")
        range_label.grid(row=3, column=2,pady=5, padx=5)
        range_low_entry = ttk.Entry(main_frame, width=5)
        range_low_entry.grid(row=3, column=3, padx=5, pady=5)

        button = ttk.Button(main_frame, text="Focus", command=self.autofocus, bootstyle="info", width=5)
        button.grid(row=2, column=4, rowspan=2, padx=5, pady=5, sticky="news")

    def autofocus(self):
        for i in range(10):
            print(f"Running autofocus iteration {i+1}")
            entropy = 1 / (i + 1)
            wavelength = [400, 500, 600, 700]
            intensity = [0.1, 0.2, 0.3, 0.4]

            self.ax1.plot(i, entropy, 'ro')
            self.ax2.plot(wavelength, intensity, 'bo')
            self.canvas.draw()

    def stop_autofocus(self):
        print("Stopping autofocus")
    
    def save_results(self):
        print("Saving autofocus results")


