import itertools
import numpy as np
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import queue
import logging
import threading
import os
import scipy.signal
from noodlepy.gui.publisher_subscriber import Publisher
from noodlepy.gui.wasatch_manager import WasatchManager
from noodlepy.gui.nanodrive import NanoDrive
import time
from PIL import Image, ImageTk

class WasatchAutofocusModule(Publisher, ttk.Frame):
    def __init__(self, parent):
        Publisher.__init__(self, ["update_nanodrive_position", 'task_completed'])
        ttk.Frame.__init__(self, parent)
        self.name = "WasatchAutofocusModule"

        self.initial_integ_time_ms = 1000
        self.initial_laser_power_mW = 450
        self.image_height = 300
        self.image_width = 600
        self.laser_power_spinbox_increment = 20

        self.LASER_POWER_SPINBOX_LOWER_LIMIT = 0 # fixed 
        self.LASER_POWER_SPINBOX_UPPER_LIMIT = 450 # fixed

        self.integ_time_spinbox_lower_limit = 100
        self.integ_time_spinbox_upper_limit = 100000
        self.integ_time_spinbox_increment = 5

        self.nanodrive_movement_stabilization_time = 1.5 # seconds

        self.keep_refreshing_live_spectrum = False
        self.units = "wavelength"
        self.nano_drive = NanoDrive()
        self.create_live_spectrum_widgets()  
        self.create_widgets_for_autofocus()

        self.spectrum_queue = queue.Queue()
        self.wasatch_manager = WasatchManager(self.initial_integ_time_ms, self.initial_laser_power_mW)
        if self.wasatch_manager.connect():
            self.start_spectrum_thread()
            self.update_live_spectrum()
        else:
            logging.error('Failed to connect to Wasatch spectrometer. Check connection')


    def create_live_spectrum_widgets(self):
        play_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","play_icon.png")).resize((55, 55))
        pause_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","pause_icon.png")).resize((55, 55))
        self.play_icon = ImageTk.PhotoImage(play_icon)
        self.pause_icon = ImageTk.PhotoImage(pause_icon)

        self.spectrum_frame = ttk.Labelframe(self, text="Live Spectrum", padding=5)
        self.spectrum_frame.grid(row=0, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)

        self.laser_power_label = ttk.Label(self.spectrum_frame, text="Power (mW)", width=5)
        self.laser_power_label.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.laser_power_var = StringVar()
        self.laser_power_var.set(self.initial_laser_power_mW)
        self.laser_power_spinbox = ttk.Spinbox(self.spectrum_frame, textvariable=self.laser_power_var, from_=self.LASER_POWER_SPINBOX_LOWER_LIMIT, to=self.LASER_POWER_SPINBOX_UPPER_LIMIT, increment=self.laser_power_spinbox_increment, width=5, justify='center')
        self.laser_power_spinbox.grid(row=2, column=0, sticky='ew', pady=5, padx=5)
        self.laser_power_spinbox.bind("<FocusOut>", self.update_wasatch_settings)
        self.laser_power_spinbox.bind("<Return>", self.update_wasatch_settings)

        self.integ_time_label = ttk.Label(self.spectrum_frame, text="Expo. Time (ms)", width=5)
        self.integ_time_label.grid(row=1, column=1, sticky='nsew', pady=5, padx=5)
        self.integ_time_var = StringVar()
        self.integ_time_var.set(self.initial_integ_time_ms)
        self.integ_time_spinbox = ttk.Spinbox(self.spectrum_frame, textvariable=self.integ_time_var, from_=self.integ_time_spinbox_lower_limit, to=self.integ_time_spinbox_upper_limit, increment=self.integ_time_spinbox_increment, width=5, justify='center')
        self.integ_time_spinbox.grid(row=2, column=1, sticky='ew', pady=5, padx=5)
        self.integ_time_spinbox.bind("<FocusOut>", self.update_wasatch_settings)
        self.integ_time_spinbox.bind("<Return>", self.update_wasatch_settings)

        self.laser_button = ttk.Button(self.spectrum_frame, text="Laser On", command=self.turn_laser_on, bootstyle ='info-outline', width=5)
        self.laser_button.grid(row=1, column=2, rowspan=2, sticky='nsew', pady=5, padx=5)

        self.play_button = ttk.Button(self.spectrum_frame, image=self.play_icon, command=self.start_to_play_live_spectrum, bootstyle ='dark', width=5)
        self.play_button.grid(row=1, column=3, rowspan =2, sticky='ew', pady=5, padx=5)

        self.capture_button = ttk.Button(self.spectrum_frame, text="Capture", bootstyle ='info-outline', command=self.capture_live_spectrum)
        self.capture_button.grid(row=1, column=4, sticky='ew', pady=5, padx=5)

        self.ref_polystyrene_button = ttk.Button(self.spectrum_frame, text="Ref. Polystyrene", command=self.debug_with_polystyrene, bootstyle ='info-outline')
        self.ref_polystyrene_button.grid(row=2, column=4, sticky='ew', pady=5, padx=5)

        self.live_spectrum_canvas = ttk.Canvas(self.spectrum_frame, width=self.image_width, height=self.image_height)
        self.live_spectrum_canvas.grid(row=0, column=0, columnspan=5, sticky='nsew', pady=5, padx=5)

        self.unit_toggle_var = ttk.BooleanVar(value=False)
        self.unit_toggle_btn = ttk.Checkbutton(self.spectrum_frame, variable=self.unit_toggle_var, text="To cm⁻¹", bootstyle="info-round-toggle", command=self.update_live_spectrum_units)
        self.unit_toggle_btn.grid(row=0, column=3, sticky='en', pady=5, padx=5)

        self.live_spectrum_fig, self.live_spectrum_ax = plt.subplots(figsize=(8, 4))
        self.live_spectrum_line, = self.live_spectrum_ax.plot([], [])
        self.live_spectrum_line.set_linewidth(0.8)
        self.live_spectrum_line.set_color('#5bc0de')

        if self.units == "wavelength":
            self.live_spectrum_ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
        elif self.units == "wavenumber":
            self.live_spectrum_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')
        else:
            print("Invalid units")

        self.live_spectrum_ax.set_ylabel("Intensity (counts)", fontsize=6, color='white')
        self.live_spectrum_ax.set_title("Raman Spectrum", fontsize=8, color='white')
        self.live_spectrum_ax.tick_params(axis='both', which='major', labelsize=6)
        self.live_spectrum_ax.tick_params(axis='both', which='minor', labelsize=6)
        self.live_spectrum_ax.tick_params(axis='x', colors='white')
        self.live_spectrum_ax.tick_params(axis='y', colors='white')
        self.live_spectrum_ax.spines['bottom'].set_color('white')
        self.live_spectrum_ax.spines['left'].set_color('white')
        self.live_spectrum_ax.spines['right'].set_visible(False)
        self.live_spectrum_ax.spines['top'].set_visible(False)
        self.live_spectrum_ax.set_facecolor("none")
        self.live_spectrum_ax.xaxis.label.set_color('white')
        self.live_spectrum_ax.yaxis.label.set_color('white')
        plt.tight_layout(pad=0.5)
        self.live_spectrum_fig.patch.set_alpha(0)
        self.live_spectrum_canvas = FigureCanvasTkAgg(self.live_spectrum_fig, master=self.live_spectrum_canvas)
        self.live_spectrum_canvas.draw()
        self.live_spectrum_canvas.get_tk_widget().grid(row=0, column=0, columnspan=4, sticky='nsew')
        self.background = self.live_spectrum_canvas.copy_from_bbox(self.live_spectrum_ax.bbox)

    def create_widgets_for_autofocus(self):

        self.nanodrive_frame = ttk.Labelframe(self, text='Auto Focus', padding=5)
        self.nanodrive_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)

        self.wasatch_autofocus_fig, self.entropy_ax, self.intensity_ax = self.initialize_wasatch_autofocus_figure()
        self.wasatch_autofocus_canvas = FigureCanvasTkAgg(self.wasatch_autofocus_fig, master=self.nanodrive_frame)
        self.wasatch_autofocus_canvas.draw()
        self.wasatch_autofocus_canvas.get_tk_widget().grid(row=0, column=0, columnspan=5, rowspan=2, sticky="nsew", padx=5, pady=5)

        # create initial values for autofocus parameters
        self.wasatch_autofocus_step_num_var = StringVar()
        self.wasatch_autofocus_step_num_var.set(10)
        self.wasatch_autofocus_num_rep_var = StringVar()
        self.wasatch_autofocus_num_rep_var.set(3)
        self.wasatch_autofocus_range_low_var = StringVar()
        self.wasatch_autofocus_range_low_var.set(0)
        self.wasatch_autofocus_range_high_var = StringVar()
        self.wasatch_autofocus_range_high_var.set(80)
        
        # Create the entries for the autofocus parameters
        self.wasatch_autofocus_step_num_label = ttk.Label(self.nanodrive_frame, text="Number of Steps")
        self.wasatch_autofocus_step_num_label.grid(row=2, column=0, pady=5, padx=5)
        self.wasatch_autofocus_step_num_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.wasatch_autofocus_step_num_var)
        self.wasatch_autofocus_step_num_entry.grid(row=2, column=1, padx=5, pady=5)
        self.wasatch_autofocus_step_num_entry.bind("<FocusOut>", self.update_wasatch_settings)
        self.wasatch_autofocus_step_num_entry.bind("<Return>", self.update_wasatch_settings)

        self.wasatch_autofocus_rep_num_label = ttk.Label(self.nanodrive_frame, text="Number of Reps")
        self.wasatch_autofocus_rep_num_label.grid(row=2, column=2, pady=5, padx=5)
        self.wasatch_autofocus_rep_num_entry = ttk.Entry(self.nanodrive_frame, width=5, textvariable=self.wasatch_autofocus_num_rep_var)
        self.wasatch_autofocus_rep_num_entry.grid(row=2, column=3, padx=5, pady=5)
        self.wasatch_autofocus_rep_num_entry.bind("<FocusOut>", self.update_wasatch_settings)
        self.wasatch_autofocus_rep_num_entry.bind("<Return>", self.update_wasatch_settings)
    
        self.wasatch_autofocus_range_label = ttk.Label(self.nanodrive_frame, text="Z Min [um]")
        self.wasatch_autofocus_range_label.grid(row=3, column=0,pady=5, padx=5)
        self.wasatch_autofocus_range_low_spinbox = ttk.Spinbox(self.nanodrive_frame, from_=self.nano_drive.min_position_um, to=self.nano_drive.max_position_um, increment=1, textvariable=self.wasatch_autofocus_range_low_var, width=5)
        self.wasatch_autofocus_range_low_spinbox.grid(row=3, column=1, padx=5, pady=5)
        self.wasatch_autofocus_range_low_spinbox.bind("<FocusOut>", self.update_wasatch_settings)
        self.wasatch_autofocus_range_low_spinbox.bind("<Return>", self.update_wasatch_settings)

        self.wasatch_autofocus_range_label = ttk.Label(self.nanodrive_frame, text="Z Max [um]")
        self.wasatch_autofocus_range_label.grid(row=3, column=2,pady=5, padx=5)
        self.wasatch_autofocus_range_high_spinbox = ttk.Spinbox(self.nanodrive_frame, from_=self.nano_drive.min_position_um, to=self.nano_drive.max_position_um, increment=1, textvariable=self.wasatch_autofocus_range_high_var, width=5)
        self.wasatch_autofocus_range_high_spinbox.grid(row=3, column=3, padx=5, pady=5)
        self.wasatch_autofocus_range_high_spinbox.bind("<FocusOut>", self.update_wasatch_settings)
        self.wasatch_autofocus_range_high_spinbox.bind("<Return>", self.update_wasatch_settings)

        self.wasatch_autofocus_button = ttk.Button(self.nanodrive_frame, text="Focus", command=self.start_autofocus, bootstyle="info", width=5)
        self.wasatch_autofocus_button.grid(row=2, column=4, rowspan=2, padx=5, pady=5, sticky="news")

    def start_to_play_live_spectrum(self):
        self.keep_refreshing_live_spectrum = True
        self.play_button.config(image=self.pause_icon, command=self.stop_to_play_live_spectrum, bootstyle ='info')
        self.update_live_spectrum()

    def stop_to_play_live_spectrum(self):
        self.keep_refreshing_live_spectrum = False
        self.play_button.config(image=self.play_icon, command=self.start_to_play_live_spectrum, bootstyle ='dark')

    def initialize_wasatch_autofocus_figure(self):
                # Create the figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        entropy_ax = axes[0]
        intensity_ax = axes[1]

        entropy_ax.set_xlabel('Z-axis position',fontsize=6, color='white')
        entropy_ax.set_ylabel('Entropy', fontsize=6, color='white')
        entropy_ax.set_title('Entropy', fontsize=8, color='white')

        intensity_ax.set_xlabel('Z-axis position', fontsize=6, color='white')
        intensity_ax.set_ylabel('Intensity', fontsize=6, color='white')
        intensity_ax.set_title('Average Spectrum at Current Position', fontsize=8, color='white')

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
    
    def update_live_spectrum_units(self):
        if self.unit_toggle_var.get():
            self.units = "wavenumber"
            self.live_spectrum_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')
            self.unit_toggle_btn.configure(text="To nm")
        else:
            self.units = "wavelength"
            self.live_spectrum_ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
            self.unit_toggle_btn.configure(text="To cm⁻¹")

        self.live_spectrum_ax.relim()
        self.live_spectrum_ax.autoscale_view()
        self.live_spectrum_canvas.draw()

    def update_live_spectrum(self):
        """Update the spectrum plot with the latest data from the queue."""

        try:
            while True:
                self.latest_spectrum = self.spectrum_queue.get_nowait()
                
                if self.units == "wavelength":
                    x_data = self.wasatch_manager.settings.wavelengths
                else:
                    x_data = self.wasatch_manager.settings.wavenumbers
                
                self.live_spectrum_line.set_data(x_data, self.latest_spectrum)
                self.live_spectrum_ax.relim()
                self.live_spectrum_ax.autoscale_view()

                # Update the plot
                self.live_spectrum_canvas.draw_idle()

                # print("threading.current_thread().name: ", threading.current_thread().name)

        except queue.Empty:
            pass
        if self.keep_refreshing_live_spectrum:
            self.after(self.wasatch_manager.integ_time_ms, self.update_live_spectrum)
        
    def start_spectrum_thread(self):
        """Start a thread to collect spectrum data."""
        if not hasattr(self, 'spectrum_thread') or not self.spectrum_thread.is_alive():
            self.spectrum_thread = Thread(target=self.collect_spectrum)
            self.spectrum_thread.daemon = True
            self.spectrum_thread.start()
            print("Started spectrum thread")

    def collect_spectrum(self):
        # print("Collecting spectrum:", threading.current_thread().name)
        while True:
            spectrum = self.wasatch_manager.get_spectrum()
            if spectrum is not None:
                self.spectrum_queue.put(spectrum)
                # time.sleep(self.wasatch_manager.integ_time_ms / 1000.0)  # wait based on integration time

    def update_wasatch_settings(self, event=None):
        try:
            self.wasatch_manager.set_integration_time(int(self.integ_time_var.get()))
            self.wasatch_manager.set_laser_power(int(self.laser_power_var.get()))
        except ValueError:
            pass

    def capture_live_spectrum(self):
        ## check which thread is capture running in
        spectrum = self.latest_spectrum
        # print("Capturing spectrum in thread: ", threading.current_thread().name)
        if self.units == "wavelength":
            x_axis = self.wasatch_manager.settings.wavelengths
        else:
            x_axis = self.wasatch_manager.settings.wavenumbers
            
        new_window = ttk.Toplevel()
        new_window.title("Captured Raman Spectrum")

        captured_spectrum_fig = plt.figure(figsize=(4, 2))
        captured_spectrum_fig, captured_spectrum_ax = plt.subplots(figsize=(4, 2))
        captured_spectrum_line, =  captured_spectrum_ax.plot(x_axis, spectrum)
        captured_spectrum_line.set_linewidth(0.8)
        captured_spectrum_line.set_color('#5bc0de')

        if self.units == "wavelength":
            captured_spectrum_ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
        elif self.units == "wavenumber":
            captured_spectrum_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')

        captured_spectrum_ax.set_ylabel("Intensity (counts)", fontsize=6, color='white')
        captured_spectrum_ax.set_title("Raman Spectrum", fontsize=8, color='white')
        captured_spectrum_ax.tick_params(axis='both', which='major', labelsize=6)
        captured_spectrum_ax.tick_params(axis='both', which='minor', labelsize=6)
        captured_spectrum_ax.tick_params(axis='x', colors='white')
        captured_spectrum_ax.tick_params(axis='y', colors='white')
        captured_spectrum_ax.spines['bottom'].set_color('white')
        captured_spectrum_ax.spines['left'].set_color('white')
        captured_spectrum_ax.spines['right'].set_visible(False)
        captured_spectrum_ax.spines['top'].set_visible(False)
        captured_spectrum_ax.set_facecolor("none")
        captured_spectrum_ax.xaxis.label.set_color('white')
        captured_spectrum_ax.yaxis.label.set_color('white')
        plt.tight_layout(pad=0.5)
        captured_spectrum_fig.patch.set_alpha(0)
        captured_spectrum_canvas = FigureCanvasTkAgg(captured_spectrum_fig, master=new_window)
        captured_spectrum_canvas.draw()
        captured_spectrum_canvas.get_tk_widget().pack(side=ttk.TOP, fill=ttk.BOTH, expand=1)

        # check if the capture_spectrum fig already exists
        filelist = [f for f in os.listdir() if f.endswith(".png")]
        if "Captured_spectrum_1.png" in filelist:
            file_number = 2
            while f"Captured_spectrum_{file_number}.png" in filelist:
                file_number += 1
        else:
            file_number = 1

        # save the captured spectrum as a png file
        captured_spectrum_fig.savefig(f"Captured_spectrum_{file_number}.png", dpi=300)

        with open(f"Captured_spectrum_{file_number}.txt", "w") as outfile:
            for i in range(len(x_axis)):
                outfile.write(f"{x_axis[i]:0.2f}, {spectrum[i]}\n")
        plt.close(captured_spectrum_fig)
        

    def debug_with_polystyrene(self):
        expected_peak = 1006.22
        expected_counts = 1500 
        peak_tolerance_cm = 5

        print("Take a sample spectrum")
        # get the latest spectrum
        measurement = self.latest_spectrum

        print('Comparing peaks to polystyrene')
        peak_pixels = scipy.signal.find_peaks(measurement, height=700)[0]

        peak_pixel = None
        for pixel in peak_pixels:
            peak_cm = self.wasatch_manager.settings.wavenumbers[pixel]
            if abs(expected_peak - peak_cm) <= peak_tolerance_cm:
                print(f"Found expected {expected_peak}cm⁻¹ at pixel {pixel} ({peak_cm:0.2f}cm⁻¹)")
                peak_pixel = pixel
                break

        if peak_pixel is None:
            print(f"Failed to find {expected_peak}cm⁻¹ peak in sample")
            return False

        counts = measurement[peak_pixel]
        if counts < expected_counts:
            print(f"Failed. {expected_peak}cm⁻¹ peak counts too low ({counts} < {expected_counts}): adjust working distance")
            return False
        
        print(f"Success! {expected_peak}cm⁻¹ peak found with {counts} counts.")
        return True
    
    def turn_laser_on(self):
        if self.wasatch_manager is None:
            laser_power_mW = int(self.laser_power_var.get())
            integ_time_ms = int(self.integ_time_var.get())
            self.wasatch_manager = WasatchManager(integ_time_ms, laser_power_mW)
            if not self.wasatch_manager.connect():
                print("Failed to connect to Wasatch spectrometer. Check connection and try again.")
                return
        self.wasatch_manager.turn_laser_on()
        self.laser_button.config(text="Laser Off", command=self.turn_laser_off, bootstyle ='info')

    def turn_laser_off(self):
        self.wasatch_manager.turn_laser_off()
        self.laser_button.config(text="Laser On", command=self.turn_laser_on, bootstyle ='info-outline')

    def run_in_thread(self, func):
        thread = Thread(target=func)
        thread.daemon = True
        thread.start()

    def start_autofocus(self):
        # Stop the live spectrum and set up parameters
        self.stop_to_play_live_spectrum()
        self.autofocus_worker_thread = threading.Thread(target=self.perform_rough_autofocus_step)
        self.autofocus_worker_thread.start()

    def perform_rough_autofocus_step(self):
        print("Starting rough autofocus for Wasatch")
        self.min = float(self.wasatch_autofocus_range_low_var.get())
        self.max = float(self.wasatch_autofocus_range_high_var.get())
        self.num_step = int(float(self.wasatch_autofocus_step_num_var.get()))
        self.num_rep = int(float(self.wasatch_autofocus_num_rep_var.get()))
        self.z_axis_range =  np.round(np.linspace(self.min, self.max, self.num_step))
        self.x1_data, self.y1_data = [], []
        self.current_step = 0

        while self.current_step < len(self.z_axis_range):
            # Move nano drive and get current position
            z_pos = self.z_axis_range[self.current_step]
            self.nano_drive.move_to(z_pos)

            while True:
                position = self.nano_drive.get_current_position()
                if position == z_pos:
                    break
            
            time.sleep(self.nanodrive_movement_stabilization_time)
            self.dispatch("update_nanodrive_position", position)

            # Measure spectra in the background thread (self.num_rep reps)
            wavelength, wavelengths, intensities = self.measure_spectra(self.num_rep)
            avg_intensity = np.mean(intensities, 0)

            ent = self.calculate_entropy(avg_intensity)
            self.x1_data.append(position)
            self.y1_data.append(ent)

            # Update plot in the main thread
            self.after(0, self.plot_autofocus_data, self.x1_data, self.y1_data, wavelength, avg_intensity)

            # Proceed to the next step
            self.current_step += 1

        # Once done with rough autofocus, proceed to fine autofocus
        fineMin, fineMax = self.refine_focus_range(self.y1_data, self.z_axis_range)
        fine_focus = self.start_fine_autofocus(fineMin, fineMax, self.num_rep)
        if fine_focus:
            print("Autofocus complete")
            return True

    def start_fine_autofocus(self, fineMin, fineMax, num_rep):
        print("Starting fine autofocus for Wasatch")
        self.z_axis_range =  np.round(np.linspace(fineMin, fineMax, 20))
        self.x2_data, self.y2_data = [], []
        self.current_step = 0
        self.num_rep = num_rep

        while self.current_step < len(self.z_axis_range):
            z_pos = self.z_axis_range[self.current_step]
            self.nano_drive.move_to(z_pos)

            while True:
                position = self.nano_drive.get_current_position()
                if position == z_pos:
                    break

            self.dispatch("update_nanodrive_position", position)
            time.sleep(self.nanodrive_movement_stabilization_time)

            # Measure spectra in the background thread
            wavelength, wavelengths, intensities = self.measure_spectra(self.num_rep)
            avg_intensity = np.mean(intensities, 0)

            ent = self.calculate_entropy(avg_intensity)
            self.x2_data.append(position)
            self.y2_data.append(ent)

            # Update plot in the main thread
            self.after(0, self.plot_autofocus_data, self.x2_data, self.y2_data, wavelength, avg_intensity)

            # Proceed to the next fine autofocus step
            self.current_step += 1

        # After fine autofocus, finalize focus position
        focusedPos = self.z_axis_range[np.argmin(self.y2_data)]
        self.nano_drive.move_to(focusedPos)
        time.sleep(self.nanodrive_movement_stabilization_time)
        self.dispatch("update_nanodrive_position", self.nano_drive.get_current_position())
        print("Focused Position =", focusedPos)
        print("Fine focusing complete")
        return True

    def handle_autofocus_wasatch_during_aquisition(self):
        def on_focus_complete(result):
            if result:
                self.dispatch('task_completed')
            else:
                self.dispatch('abort_aquisition')

        # Call the thread-running function, passing the callback
        self.stop_to_play_live_spectrum()
        self.run_in_thread_with_callback(self.perform_rough_autofocus_step, on_focus_complete)


    def run_in_thread_with_callback(self, func, callback, *args):
        def wrapper():
            # Execute the target function and store the result
            result = func(*args)
            # Pass the result to the callback function
            callback(result)

        # Run the wrapper function in a separate thread
        thread = Thread(target=wrapper, daemon=True)
        thread.start()

    def measure_spectra(self, num_rep):
        print("Measuring spectra:", threading.current_thread().name)
        intensities = []
        wavelengths = []
        wavelength = self.wasatch_manager.settings.wavelengths

        for i in range(int(num_rep)):
            spectrum = self.spectrum_queue.get()

            intensities.append(spectrum)
            wavelengths.append(wavelength)

        return wavelength, wavelengths, intensities

    def calculate_entropy(self, intensities):
        normalized_int = intensities / np.sum(intensities)
        entropy = -np.sum(normalized_int * np.log2(normalized_int))
        return entropy

    def refine_focus_range(self, entropy_list, z_axis_range):
        k = np.argmin(entropy_list)
        fineMin = z_axis_range[max(k - 1, 0)]
        fineMax = z_axis_range[min(k + 1, len(z_axis_range) - 1)]
        print("The fine range is:", fineMin, fineMax)
        return fineMin, fineMax

    def plot_autofocus_data(self, x_data, y_data, wavelengths, intensities):
        print("Plotting autofocus data:", threading.current_thread().name)
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

        self.wasatch_autofocus_canvas.draw_idle()
        self.wasatch_autofocus_canvas.flush_events()  # Process any pending events for real-time updates
        self.update_idletasks()

    def update_nanodrive_settings(self, event):
        self.wasatch_autofocus_step_num_var.set(float(self.wasatch_autofocus_step_num_var.get()))
        self.wasatch_autofocus_num_rep_var.set(int(self.wasatch_autofocus_num_rep_var.get()))
        self.wasatch_autofocus_range_low_var.set(float(self.wasatch_autofocus_range_low_var.get()))
        self.wasatch_autofocus_range_high_var.set(float(self.wasatch_autofocus_range_high_var.get()))

    def handle_get_nanodrive_position(self):
        self.dispatch("update_nanodrive_position", np.round(self.nano_drive.get_current_position()))
        return None

    def handle_move_nanodrive_by_request(self, distance):
        self.nano_drive.move_by(distance)
        # print("Moving nanodrive up by {distance} um")
        self.handle_get_nanodrive_position()
        return None

    def handle_move_nanodrive_to_request(self, position):
        self.nano_drive.move_to(position) # starting from the lowest position
        self.handle_get_nanodrive_position()
        return None

    def handle_measure_spectra_and_save_to_specific_folder(self, num_rep, folder_path, filename):
        wavelength, wavelengths, intensities = self.measure_spectra(num_rep)

        # flatten the intensities
        wavelengths_flatten = list(itertools.chain.from_iterable(wavelengths))
        intensities_flatten = list(itertools.chain.from_iterable(intensities))

        # Save the spectra to a file
        with open(os.path.join(folder_path, filename + ".txt"), "w") as outfile:
            for i in range(len(wavelengths_flatten)):
                outfile.write(f"{wavelengths_flatten[i]:0.2f}, {intensities_flatten[i]}\n")
        
        self.dispatch("task_completed")
        return None

    def handle_get_nanodrive_min_max(self):
        return self.nano_drive.min_position_um, self.nano_drive.max_position_um

if __name__ == "__main__":
    root = ttk.Window(themename="noodlepy")
    root.title("Wasatch Raman Spectrometer")
    root.geometry("800x400")
    WasatchAutofocusModule(root).pack(fill='both', expand=True)
    root.mainloop()
