import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import StringVar
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import queue
import time

from wasatch.WasatchBus import WasatchBus
from wasatch.WasatchDevice import WasatchDevice
from wasatch.RealUSBDevice import RealUSBDevice

class WasatchController():
    def __init__(self, integ_time_ms, laser_power_mW):
        self.integ_time_ms = integ_time_ms
        self.laser_power_mW = laser_power_mW
        self.connect()

    def connect(self):
        bus = WasatchBus(use_sim=False)
        if not bus.device_ids:
            print("No Wasatch USB spectrometers found.")
            return False

        device_id = bus.device_ids[0]
        print(f"Connecting to {device_id}")
        device_id.device_type = RealUSBDevice(device_id)
        device = WasatchDevice(device_id)
        ok = device.connect()
        if not ok:
            print("Can't connect to %s", device_id)
            return False

        self.settings = device.settings
        self.fid = device.hardware
        if self.settings.wavelengths is None:
            print("Script requires Raman spectrometer")
            return False

        print("connected to %s %s with %d pixels (%.2f, %.2fnm) (%.2f, %.2fcm¹)" % (
            self.settings.eeprom.model,
            self.settings.eeprom.serial_number,
            self.settings.pixels(),
            self.settings.wavelengths[0],
            self.settings.wavelengths[-1],
            self.settings.wavenumbers[0],
            self.settings.wavenumbers[-1]))
        
        self.fid.set_laser_power_high_resolution(True)
        return True

    def turn_laser_on(self):
        self.fid.set_laser_enable(True)
        print("Turning laser on. Please wait for laser to stabilize.")
    
    def turn_laser_off(self):
        self.fid.set_laser_enable(False)
        print("Turning laser off.")

    def set_integration_time(self, time_ms):
        self.fid.set_integration_time_ms(time_ms)
        print(f"Integration time set to {time_ms}ms")

    def set_laser_power(self, power_mW):
        self.fid.set_laser_power_mW(power_mW)
        print(f"Laser power set to {power_mW}mW")

    def get_spectrum(self):
        response = self.fid.get_line()
        if response and response.data:
            spectrum = response.data.spectrum
            return np.asarray(spectrum)

    def debug_with_polystyrene(self):
        expected_peak = 1006.22
        expected_counts = 1500 
        peak_tolerance_cm = 5

        print("Take a sample spectrum")
        measurement = self.get_spectrum()

        print('Comparing peaks to polystyrene')
        peak_pixels = scipy.signal.find_peaks(measurement, height=700)[0]

        peak_pixel = None
        for pixel in peak_pixels:
            peak_cm = self.settings.wavenumbers[pixel]
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

class Wasatchmodule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.initial_integ_time_ms = 100
        self.initial_laser_power_mW = 400
        self.units = "wavelength"
        self.spectrum_queue = queue.Queue()
        self.create_widgets()
        self.wasatchcontroller = WasatchController(self.initial_integ_time_ms, self.initial_laser_power_mW)
        if self.wasatchcontroller.connect():
            self.start_spectrum_thread()
            self.update_spectrum()
        else:
            print('Failed to connect to Wasatch spectrometer. Check connection')

    def create_widgets(self):
        self.spectrum_frame = ttk.Labelframe(self, text="Live Spectrum", padding=5)
        self.spectrum_frame.grid(row=0, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)

        self.laser_power_label = ttk.Label(self.spectrum_frame, text="Power (mW)", width=5)
        self.laser_power_label.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.laser_power_var = StringVar()
        self.laser_power_var.set(self.initial_laser_power_mW)
        self.laser_power_spinbox = ttk.Spinbox(self.spectrum_frame, textvariable=self.laser_power_var, from_=0, to=450, increment=20, width=5, justify='center')
        self.laser_power_spinbox.grid(row=2, column=0, sticky='ew', pady=5, padx=5)
        self.laser_power_spinbox.bind("<FocusOut>", self.update_settings)
        self.laser_power_spinbox.bind("<Return>", self.update_settings)

        self.integ_time_label = ttk.Label(self.spectrum_frame, text="Expo. Time (ms)", width=5)
        self.integ_time_label.grid(row=1, column=1, sticky='nsew', pady=5, padx=5)
        self.integ_time_var = StringVar()
        self.integ_time_var.set(self.initial_integ_time_ms)
        self.integ_time_spinbox = ttk.Spinbox(self.spectrum_frame, textvariable=self.integ_time_var, from_=1, to=10000, increment=5, width=5, justify='center')
        self.integ_time_spinbox.grid(row=2, column=1, sticky='ew', pady=5, padx=5)
        self.integ_time_spinbox.bind("<FocusOut>", self.update_settings)
        self.integ_time_spinbox.bind("<Return>", self.update_settings)

        self.start_button = ttk.Button(self.spectrum_frame, text="Laser On", command=self.start, bootstyle ='info-outline', width=5)
        self.start_button.grid(row=1, column=2, rowspan=2, sticky='nsew', pady=5, padx=5)

        self.capture_button = ttk.Button(self.spectrum_frame, text="Capture", bootstyle ='info-outline', command=self.capture)
        self.capture_button.grid(row=2, column=3, sticky='ew', pady=5, padx=5)

        self.stop_button = ttk.Button(self.spectrum_frame, text="Ref. Polystyrene", command=self.reference, bootstyle ='info-outline')
        self.stop_button.grid(row=1, column=3, sticky='ew', pady=5, padx=5)

        self.spectrum_canvas = ttk.Canvas(self.spectrum_frame)
        self.spectrum_canvas.grid(row=0, column=0, columnspan=4, sticky='nsew', pady=5, padx=5)

        self.toggle_var = ttk.BooleanVar(value=False)
        self.toggle_btn = ttk.Checkbutton(self.spectrum_frame, variable=self.toggle_var, text="To cm⁻¹", bootstyle="info-round-toggle", command=self.update_units)
        self.toggle_btn.grid(row=0, column=3, sticky='en', pady=5, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.line, = self.ax.plot([], [])
        self.line.set_linewidth(0.8)
        self.line.set_color('#5bc0de')

        if self.units == "wavelength":
            self.ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
        elif self.units == "wavenumber":
            self.ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')
        else:
            print("Invalid units")

        self.ax.set_ylabel("Intensity (counts)", fontsize=6, color='white')
        self.ax.set_title("Raman Spectrum", fontsize=8, color='white')
        self.ax.tick_params(axis='both', which='major', labelsize=6)
        self.ax.tick_params(axis='both', which='minor', labelsize=6)
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.set_facecolor("none")
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        plt.tight_layout(pad=0.5)
        self.fig.patch.set_alpha(0)
        self.spectrum_plot = FigureCanvasTkAgg(self.fig, master=self.spectrum_canvas)
        self.spectrum_plot.get_tk_widget().grid(row=0, column=0, columnspan=4, sticky='nsew')

    def update_settings(self, event=None):
        self.wasatchcontroller.set_laser_power(int(self.laser_power_var.get()))
        self.wasatchcontroller.set_integration_time(int(self.integ_time_var.get()))

    def update_units(self):
        if self.toggle_var.get():
            self.units = "wavenumber"
            self.ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')
            self.toggle_btn.configure(text="To nm")
        else:
            self.units = "wavelength"
            self.ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
            self.toggle_btn.configure(text="To cm⁻¹")

        self.ax.relim()
        self.ax.autoscale_view()
        self.spectrum_plot.draw()

    def update_spectrum(self):
        """Update the spectrum plot with the latest data from the queue."""
        try:
            while True:
                spectrum_data = self.spectrum_queue.get_nowait()
                
                if self.units == "wavelength":
                    x_data = self.wasatchcontroller.settings.wavelengths
                else:
                    x_data = self.wasatchcontroller.settings.wavenumbers
                
                self.line.set_data(x_data, spectrum_data)
                self.ax.relim()
                self.ax.autoscale_view()

                # Update the plot
                self.spectrum_plot.draw_idle()

        except queue.Empty:
            pass
        self.after(1, self.update_spectrum)

    def start(self):
        if self.wasatchcontroller is None:
            laser_power_mW = int(self.laser_power_var.get())
            integ_time_ms = int(self.integ_time_var.get())
            self.wasatchcontroller = WasatchController(integ_time_ms, laser_power_mW)
            if not self.wasatchcontroller.connect():
                print("Failed to connect to Wasatch spectrometer. Check connection and try again.")
                return
        self.wasatchcontroller.turn_laser_on()
        self.start_button.config(text="Laser Off", command=self.stop, bootstyle ='info')

    def stop(self):
        self.wasatchcontroller.turn_laser_off()
        self.start_button.config(text="Laser On", command=self.start, bootstyle ='info-outline')

    def start_spectrum_thread(self):
        """Start a thread to collect spectrum data."""
        if not hasattr(self, 'spectrum_thread') or not self.spectrum_thread.is_alive():
            self.spectrum_thread = Thread(target=self.collect_spectrum)
            self.spectrum_thread.daemon = True
            self.spectrum_thread.start()
            print("Started spectrum thread")

    def collect_spectrum(self):
        """Collect spectrum data and put it into the queue."""
        while True:
            spectrum = self.wasatchcontroller.get_spectrum()
            if spectrum is not None:
                self.spectrum_queue.put(spectrum)
            time.sleep(0.1)

    def capture(self):
        spectrum = self.wasatchcontroller.get_spectrum()
        if self.units == "wavelength":
            x_axis = self.wasatchcontroller.settings.wavelengths
        else:
            x_axis = self.wasatchcontroller.settings.wavenumbers
            
        new_window = ttk.Toplevel()
        new_window.title("Captured Raman Spectrum")

        captured_fig = plt.figure()
        captured_fig, captured_ax = plt.subplots(figsize=(4, 2))
        captured_line, =  captured_ax.plot(x_axis, spectrum)
        captured_line.set_linewidth(0.8)
        captured_line.set_color('#5bc0de')

        if self.units == "wavelength":
            captured_ax.set_xlabel("Wavelength (nm)", fontsize=6, color='white')
        elif self.units == "wavenumber":
            captured_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=6, color='white')

        captured_ax.set_ylabel("Intensity (counts)", fontsize=6, color='white')
        captured_ax.set_title("Raman Spectrum", fontsize=8, color='white')
        captured_ax.tick_params(axis='both', which='major', labelsize=6)
        captured_ax.tick_params(axis='both', which='minor', labelsize=6)
        captured_ax.tick_params(axis='x', colors='white')
        captured_ax.tick_params(axis='y', colors='white')
        captured_ax.spines['bottom'].set_color('white')
        captured_ax.spines['left'].set_color('white')
        captured_ax.spines['right'].set_visible(False)
        captured_ax.spines['top'].set_visible(False)
        captured_ax.set_facecolor("none")
        captured_ax.xaxis.label.set_color('white')
        captured_ax.yaxis.label.set_color('white')
        plt.tight_layout(pad=0.5)
        captured_fig.patch.set_alpha(0)
        canvas = FigureCanvasTkAgg(captured_fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=ttk.TOP, fill=ttk.BOTH, expand=1)
        captured_fig.savefig("Captured_spectrum.png", dpi=300)

        with open("Captured_spectrum.txt", "w") as outfile:
            for i in range(len(x_axis)):
                outfile.write(f"{x_axis[i]:0.2f}, {spectrum[i]}\n")

    def reference(self):
        self.wasatchcontroller.debug_with_polystyrene()

if __name__ == "__main__":
    root = ttk.Window(themename="noodlepy")
    root.title("Wasatch Raman Spectrometer")
    root.geometry("800x400")
    Wasatchmodule(root).pack(fill='both', expand=True)
    root.mainloop()
