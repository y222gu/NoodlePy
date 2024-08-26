import tkinter as tk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import typing
import threading
import queue
import ttkbootstrap as ttk
import numpy as np
from noodlepy.gui.edgedetector import EdgeDetector
import os

try:
    from noodlepy.utils.windows_setup import configure_path
    print("Configuring path...")
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK


class LiveViewCanvas(tk.Canvas):
    def __init__(self, parent, image_queue, width, height):
        self.image_queue = image_queue
        self._image_width = width
        self._image_height = height
        self._image = None
        self.tk_image = None
        tk.Canvas.__init__(self, parent, width=width, height=height)
        self.grid(row=0, column=0, sticky='nsew')
        self._get_image()

    def _resize(self, event=None):
        if self._image:
            resized_image = self._image.resize((self._image_width, self._image_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            self.create_image(0, 0, image=self.tk_image, anchor='nw')

    def _get_image(self):
        try:
            self._image = self.image_queue.get_nowait()
            self._resize()
        except queue.Empty:
            pass
        self.after(10, self._get_image)


class ImageAcquisitionThread(threading.Thread):
    def __init__(self, camera):
        super(ImageAcquisitionThread, self).__init__()
        camera.exposure_time_us = 40000
        self._camera = camera
        self._previous_timestamp = 0
        self._is_color = False
        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        image8bit = np.asarray(scaled_image, np.uint8)
        processed_image = image8bit.squeeze()
        return Image.fromarray(processed_image)

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    pil_image = self._get_image(frame)
                    self._image_queue.put_nowait(pil_image)
            except queue.Full:
                pass
            except Exception as error:
                print(f"Encountered error: {error}, image acquisition will stop.")
                break
        print("Image acquisition has stopped")


class LiveViewModule(tk.Frame):
    def __init__(self, parent, width=288, height=216):
        super().__init__(parent)
        self.width = width
        self.height = height

        self.camera_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","camera_icon.png"))
        self.camera_icon = self.camera_icon.resize((30, 30))
        self.camera_icon = ImageTk.PhotoImage(self.camera_icon)

        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()
        if not camera_list:
            raise Exception("No cameras found")
        self.camera = self.sdk.open_camera(camera_list[0])
        self.image_acquisition_thread = ImageAcquisitionThread(self.camera)

        print("Setting camera parameters...")
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.arm(2)
        self.camera.issue_software_trigger()

        print("Starting image acquisition thread...")
        self.image_acquisition_thread.start()
        self.create_widgets()

    def create_widgets(self):
        live_frame = ttk.Labelframe(self, text="Live View", width=self.width, padding=5)
        live_frame.grid(row=0, column=0, sticky='nsew', pady=5, padx=5)
        # self.camera_widget = LiveViewCanvas(parent=live_frame, image_queue=self.image_acquisition_thread.get_output_queue(), width=self.width, height=self.height)
        self.camera_widget = tk.Canvas(live_frame, width=self.width, height=self.height)
        self.camera_widget.grid(row=0, column=0, sticky='nsew')
        capture_button = ttk.Button(live_frame, image = self.camera_icon, command=self.capture_frame, padding=5, style='success')
        capture_button.grid(row=1, column=0, columnspan=2, sticky='ew')

        capture_frame = ttk.Labelframe(self, text="Captured Frame", width=self.width, padding=5)
        capture_frame.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.initial_image = Image.fromarray(np.zeros((self.height, self.width), dtype=np.uint8))
        self.image_to_display = ImageTk.PhotoImage(self.initial_image)
        self.captured_image_label = ttk.Label(capture_frame, text="No Frame Captured", image= self.image_to_display, compound='center', foreground='white')
        self.captured_image_label.grid(row=0, column=0, sticky='nsew')
        self.captured_image_label.image = self.image_to_display

        self.edge_detection_point_button = ttk.Button(capture_frame, text="Point Detection", command=lambda: self.edge_detection('point'), state=DISABLED, style='success')
        self.edge_detection_point_button.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)

        # three tabs for selecting the sampling method
        self.sampling_method_var = tk.StringVar()
        self.sampling_method_var.set("Random")
        sampling_method_frame = ttk.Labelframe(capture_frame, text="Sampling Method", padding=5)
        sampling_method_frame.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)
        sampling_method_frame.columnconfigure(0, weight=1)
        sampling_method_frame.columnconfigure(1, weight=1)
        sampling_method_frame.columnconfigure(2, weight=1)
        random_radio = ttk.Radiobutton(sampling_method_frame, text="Random", variable=self.sampling_method_var, value="Random", command=self.on_sampling_method_selected)
        random_radio.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)
        grid_radio = ttk.Radiobutton(sampling_method_frame, text="Grid", variable=self.sampling_method_var, value="Grid", command=self.on_sampling_method_selected)
        grid_radio.grid(row=0, column=1, sticky='nesw', padx=5, pady=5)
        rings_radio = ttk.Radiobutton(sampling_method_frame, text="Rings", variable=self.sampling_method_var, value="Rings", command=self.on_sampling_method_selected)
        rings_radio.grid(row=0, column=2, sticky='nesw', padx=5, pady=5)

        self.create_sampling_profile_buttons= ttk.Button(sampling_method_frame, text="Create", command=self.create_sampling_points, state=DISABLED, style='success')
        self.create_sampling_profile_buttons.grid(row=0, column=3, sticky='ew', pady=5, padx=5)
        
        # entry for the number of sampling points
        self.number_of_sampling_points_label = ttk.Label(sampling_method_frame, text="# of Points")
        self.number_of_sampling_points_label.grid(row=1, column=0, sticky='nesw', padx=5)
        self.number_of_sampling_points_entry = ttk.Entry(sampling_method_frame, width=5)
        self.number_of_sampling_points_entry.grid(row=2, column=0, sticky='ew', padx=5)
        self.number_of_sampling_points_entry.insert(0, "10")

        # entry for the number of rings
        self.rings_number_label = ttk.Label(sampling_method_frame, text="# of Rings")
        self.rings_number_label.grid(row=1, column=1, sticky='nesw', padx=5)
        self.rings_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.rings_number_entry.grid(row=2, column=1, sticky='ew', padx=5)
        self.rings_number_entry.insert(0, "3")

        # entry for the number of rows and columns for the grid
        self.row_number_label = ttk.Label(sampling_method_frame, text="Rows")
        self.row_number_label.grid(row=1, column=2, sticky='nesw', padx=5)
        self.row_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.row_number_entry.grid(row=2, column=2, sticky='ew', padx=5)
        self.row_number_entry.insert(0, "3")

        self.column_number_label = ttk.Label(sampling_method_frame, text="Columns")
        self.column_number_label.grid(row=1, column=3, sticky='nesw', padx=5)
        self.column_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.column_number_entry.grid(row=2, column=3, sticky='ew', padx=5)
        self.column_number_entry.insert(0, "3")


    def edge_detection(self, detection_type):
        self.edgedetector = EdgeDetector(self.captured_image)

        if detection_type == "auto":
            masked_image = self.edgedetector.auto_mask_generate()
        elif detection_type == "point":
            masked_image =self.edgedetector.point_prompt_mask_generate()
        elif detection_type == "box":   
            masked_image = self.edgedetector.box_prompt_mask_generate()
        print("Edge detection button clicked")

        self.masked_image = masked_image
        self.image_to_display = ImageTk.PhotoImage(self.masked_image)
        self.captured_image_label.configure(image=self.image_to_display)
        self.captured_image_label.image = self.image_to_display


    def on_sampling_method_selected(self, event):
        selected_method = self.sampling_method_var.get()
        if selected_method == "Random":
            self.row_number_entry.configure(state=DISABLED)
            self.column_number_entry.configure(state=DISABLED)
            self.number_of_sampling_points_entry.configure(state=NORMAL)
            self.rings_number_entry.configure(state=DISABLED)
            num_points = self.number_of_sampling_points_entry.get()
            self.edgedetector.generate_sampling_points(shape='random', num_points= num_points)

        elif selected_method == "Grid":
            self.create_sampling_points_grid()
        elif selected_method == "Rings":
            self.create_sampling_points_rings()

    def create_sampling_points(self, shape, num_points):
        self.edgedetector.generate_sampling_points(shape, num_points)
        print("Sampling points generated")


    def capture_frame(self):
        try:
            self.captured_image = self.image_acquisition_thread.get_output_queue().get(timeout = 2)
            print("Frame captured")
            resized_image = self.captured_image.resize((self.width, self.height), Image.LANCZOS)
            self.image_to_display = ImageTk.PhotoImage(resized_image)
            self.captured_image_label.configure(image=self.image_to_display)
            self.captured_image_label.configure(text="")
            self.captured_image_label.image = self.image_to_display
            self.edge_detection_auto_button.configure(state=NORMAL)
            self.edge_detection_point_button.configure(state=NORMAL)
            self.edge_detection_box_button.configure(state=NORMAL)

        except queue.Empty:
            print("No frame available to capture")
        except Exception as e:
            print(f"Failed to capture frame: {e}")

    def on_closing(self):
        print("Stopping image acquisition thread...")
        self.image_acquisition_thread.stop()
        self.image_acquisition_thread.join()
        self.camera.dispose()
        self.sdk.dispose()
        self.master.destroy()


if __name__ == "__main__":
    root = ttk.Window()
    root.style.theme_use('superhero')
    live_view_frame = LiveViewModule(root)
    live_view_frame.grid(row=0, column=0, sticky='nsew')
    root.protocol("WM_DELETE_WINDOW", live_view_frame.on_closing)
    root.mainloop()
