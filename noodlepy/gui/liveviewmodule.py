import tkinter as tk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw
import typing
import threading
import queue
import ttkbootstrap as ttk
import numpy as np
from noodlepy.gui.edgedetector import EdgeDetector
import os
from threading import Thread

try:
    from noodlepy.utils.windows_setup import configure_path
    print("Configuring path...")
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK


class LiveCanvas(tk.Canvas):
    '''Canvas widget for displaying live images from a camera. 
    Resizes the image to fit the canvas and draws a crossline in the middle of the image.
    Converts the image to a PhotoImage object for display.
    
    Args:
        parent: Parent widget
        image_queue: Queue object for storing images
        width: Width of the canvas
        height: Height of the canvas
        flip: Boolean for flipping the image
    '''
    def __init__(self, parent, image_queue, width, height, refresh_rate, flip=False):
        self.image_queue = image_queue
        self._image_width = width
        self._image_height = height
        self._image = None
        self.tk_image = None
        self.flip = flip
        self.refresh_rate = refresh_rate #ms
        tk.Canvas.__init__(self, parent, width=width, height=height)
        self.grid(row=0, column=0, sticky='nsew')
        self._get_image()

    def _get_image(self):
        try:
            self._image = self.image_queue.get_nowait()
            # the image is mirrored, so we flip it
            if self.flip:
                self._image = self._image.transpose(Image.FLIP_LEFT_RIGHT)

            self._resize()
            self._draw_crossline()
            self._display_image()
        except queue.Empty:
            pass
        self.after(self.refresh_rate, self._get_image)

    def _resize(self, event=None):
        if self._image:
            self.tk_image = self._image.resize((self._image_width, self._image_height), Image.LANCZOS)

    def _draw_crossline(self):
        # Draw middle dash lines on the image
        draw = ImageDraw.Draw(self.tk_image)
        draw.line((0, self.tk_image.height / 2, self.tk_image.width, self.tk_image.height / 2), fill='black', width=1)
        draw.line((self.tk_image.width / 2, 0, self.tk_image.width / 2, self.tk_image.height), fill='black', width=1)
        del draw

    def _display_image(self):
        self.tk_image = ImageTk.PhotoImage(self.tk_image)
        self.create_image(0, 0, image=self.tk_image, anchor='nw')

class ImageAcquisitionThread(threading.Thread):
    '''Thread for acquiring images from a camera in PIL format and putting them into a queue.
    
    Args:
        camera: TLCamera object
    '''
    def __init__(self, camera):
        super().__init__()
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
        # Process the image frame to convert it to a PIL image
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
                    self._image_queue.put(pil_image)

            except queue.Full:
                pass
            except Exception as error:
                print(f"Encountered error: {error}, image acquisition will stop.")
                break
        
        print("Image acquisition has stopped")

class CameraController:
    """Manages camera initialization and acquisition threading."""
    def __init__(self):
        # finding all available cameras
        self._sdk = TLCameraSDK()
        self._camera_list = self._sdk.discover_available_cameras()
        self._camera_id = None
        self._camera = None
        self._camera_id = camera_id
        self._sdk = TLCameraSDK()
        self._camera = self._sdk.open_camera(camera_id)
        self._camera_thread = ImageAcquisitionThread(self._camera)
        self._camera_thread.start()

    def get_image_queue(self):
        return self._camera_thread.get_output_queue()

    def stop_acquisition(self):
        self._camera_thread.stop()
        self._camera_thread.join()
        self._camera.dispose()
        self._sdk.dispose()

    def __del__(self):
        self.stop_acquisition()


class LiveViewModule(tk.Frame):
    def __init__(self, parent, width=288, height=216):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.camera_icon = self.load_icon(os.path.join(os.getcwd(), "noodlepy","assets","camera_icon.png"))
        self.exchange_icon = self.load_icon(os.path.join(os.getcwd(), "noodlepy","assets","exchange_icon.png"))

        self.sdk = TLCameraSDK()
        camera_list = self.sdk.discover_available_cameras()
        if not camera_list:
            raise Exception("No cameras found")
        
        # make sure both cameras are connected
        if '14938' not in camera_list or '14628' not in camera_list:
            raise Exception("Make sure both cameras are connected")
        
        self.widefield_camera = self.sdk.open_camera('14938')
        self.smallfield_camera = self.sdk.open_camera('14628')
        self.widefield_camera_thread = ImageAcquisitionThread(self.widefield_camera)
        self.smallfield_camera_thread = ImageAcquisitionThread(self.smallfield_camera)

        print("Setting camera parameters...")
        self.widefield_camera.frames_per_trigger_zero_for_unlimited = 0
        self.widefield_camera.arm(2)
        self.widefield_camera.issue_software_trigger()

        self.smallfield_camera.frames_per_trigger_zero_for_unlimited = 0
        self.smallfield_camera.arm(2)
        self.smallfield_camera.issue_software_trigger()

        print("Starting image acquisition thread...")
        self.widefield_camera_thread.start()
        self.smallfield_camera_thread.start()
        self.active_camera_thread = self.widefield_camera_thread
        self.create_widgets()


    def create_widgets(self):
        self.live_frame = ttk.Labelframe(self, text="Wide FOV", width=self.width, padding=5)
        self.live_frame.grid(row=0, column=0, sticky='nsew', pady=5, padx=5)

        self.camera_widget = LiveCanvas(parent=self.live_frame, image_queue=self.active_camera_thread.get_output_queue(), width=self.width, height=self.height, refresh_rate=10, flip=False)
        self.camera_widget.grid(row=0, column=0, columnspan=2, sticky='nsew')
        self.switch_view_button = ttk.Button(self.live_frame, image = self.exchange_icon, command= lambda: self.switch_view('TO_SMALL'), style='info')
        self.switch_view_button.grid(row=1, column=0, columnspan=1, sticky='nsew', pady=5, padx=5)
        capture_button = ttk.Button(self.live_frame, image=self.camera_icon, command= self.capture_frame, style='info')
        capture_button.grid(row=1, column=1, columnspan=1, sticky='nsew', pady=5, padx=5)

        capture_frame = ttk.Labelframe(self, text="Captured Frame", width=self.width, padding=5)
        capture_frame.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)
        self.initial_image = Image.fromarray(np.zeros((self.height, self.width), dtype=np.uint8))
        self.image_to_display = ImageTk.PhotoImage(self.initial_image)
        self.captured_image_label = ttk.Label(capture_frame, text="No Frame Captured", image= self.image_to_display, compound='center', foreground='white')
        self.captured_image_label.grid(row=0, column=0, sticky='nsew')
        self.captured_image_label.image = self.image_to_display

        # three tabs for selecting the sampling method
        self.edge_detection_point_button = ttk.Button(capture_frame, text="Point Detection", command=lambda: self.edge_detection('point'), state=DISABLED, style='info')
        self.edge_detection_point_button.grid(row=1, column=0, sticky='nsew', pady=5, padx=5)
        self.sampling_method_var = tk.StringVar()
        self.sampling_method_var.set("Random")
        sampling_method_frame = ttk.Labelframe(capture_frame, text="Sampling Method", padding=5)
        sampling_method_frame.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)
        sampling_method_frame.columnconfigure(0, weight=1)
        sampling_method_frame.columnconfigure(1, weight=1)
        sampling_method_frame.columnconfigure(2, weight=1)
        self.random_radio = ttk.Radiobutton(sampling_method_frame, text="Random", variable=self.sampling_method_var, value="Random", command=lambda: self.on_sampling_method_selected('Random'), style='info', state=DISABLED)
        self.random_radio.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)
        self.rings_radio = ttk.Radiobutton(sampling_method_frame, text="Rings", variable=self.sampling_method_var, value="Rings", command=lambda: self.on_sampling_method_selected('Rings'), style='info', state=DISABLED)
        self.rings_radio.grid(row=0, column=1, sticky='nesw', padx=5, pady=5)
        self.grid_radio = ttk.Radiobutton(sampling_method_frame, text="Grid", variable=self.sampling_method_var, value="Grid", command=lambda: self.on_sampling_method_selected('Grid'), style='info', state=DISABLED)
        self.grid_radio.grid(row=0, column=2, sticky='nesw', padx=5, pady=5)

        self.create_sampling_profile_buttons= ttk.Button(sampling_method_frame, text="Create", command=self.on_create_button_clicked, state=DISABLED, style='info')
        self.create_sampling_profile_buttons.grid(row=3, column=2, rowspan=2, sticky='nsew', pady=5, padx=5)
        
        # entry for the number of sampling points
        self.number_of_sampling_points_label = ttk.Label(sampling_method_frame, text="# of Points")
        self.number_of_sampling_points_label.grid(row=1, column=0, sticky='nesw', padx=5)
        self.number_of_sampling_points_entry = ttk.Entry(sampling_method_frame, width=5)
        self.number_of_sampling_points_entry.grid(row=2, column=0, sticky='ew', padx=5)
        self.number_of_sampling_points_entry.insert(0, "80")
        self.number_of_sampling_points_entry.configure(state=DISABLED)

        # entry for the number of rings
        self.rings_number_label = ttk.Label(sampling_method_frame, text="# of Rings")
        self.rings_number_label.grid(row=1, column=1, sticky='nesw', padx=5)
        self.rings_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.rings_number_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.rings_number_entry.insert(0, "3")
        self.rings_number_entry.configure(state=DISABLED)

        # entry for the interval between rings
        self.interval_label = ttk.Label(sampling_method_frame, text="Interval")
        self.interval_label.grid(row=1, column=2, sticky='nesw', padx=5)
        self.interval_entry = ttk.Entry(sampling_method_frame, width=5)
        self.interval_entry.grid(row=2, column=2, sticky='ew', padx=5, pady=5)
        self.interval_entry.insert(0, "30")
        self.interval_entry.configure(state=DISABLED)

        # entry for the number of rows and columns for the grid
        self.row_number_label = ttk.Label(sampling_method_frame, text="Rows")
        self.row_number_label.grid(row=3, column=0, sticky='nesw', padx=5)
        self.row_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.row_number_entry.grid(row=4, column=0, sticky='ew', padx=5, pady=5)
        self.row_number_entry.insert(0, "5")
        self.row_number_entry.configure(state=DISABLED)

        self.column_number_label = ttk.Label(sampling_method_frame, text="Columns")
        self.column_number_label.grid(row=3, column=1, sticky='nesw', padx=5)
        self.column_number_entry = ttk.Entry(sampling_method_frame, width=5)
        self.column_number_entry.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        self.column_number_entry.insert(0, "5")
        self.column_number_entry.configure(state=DISABLED)

    def load_icon(self, icon_path):
        icon = Image.open(icon_path)
        icon = icon.resize((30, 30))
        icon = ImageTk.PhotoImage(icon)
        return icon

    def run_in_thread(self, func, *args):
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()

    def edge_detection(self, detection_type):
        self.run_in_thread(self._edge_detection, detection_type)

    def _edge_detection(self, detection_type):
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

        self.create_sampling_profile_buttons.configure(state=NORMAL)
        self.random_radio.configure(state=NORMAL)
        self.rings_radio.configure(state=NORMAL)
        self.grid_radio.configure(state=NORMAL)
        self.sampling_method_var.set("Random")
        self.number_of_sampling_points_entry.configure(state=NORMAL)


    def on_sampling_method_selected(self, event):
        selected_method = self.sampling_method_var.get()
        if selected_method == "Random":
            self.row_number_entry.configure(state=DISABLED)
            self.column_number_entry.configure(state=DISABLED)
            self.number_of_sampling_points_entry.configure(state=NORMAL)
            self.rings_number_entry.configure(state=DISABLED)
            self.interval_entry.configure(state=DISABLED)

        elif selected_method == "Grid":
            self.row_number_entry.configure(state=NORMAL)
            self.column_number_entry.configure(state=NORMAL)
            self.number_of_sampling_points_entry.configure(state=DISABLED)
            self.rings_number_entry.configure(state=DISABLED)
            self.interval_entry.configure(state=DISABLED)

        elif selected_method == "Rings":
            self.row_number_entry.configure(state=DISABLED)
            self.column_number_entry.configure(state=DISABLED)
            self.number_of_sampling_points_entry.configure(state=NORMAL)
            self.rings_number_entry.configure(state=NORMAL)
            self.interval_entry.configure(state=NORMAL)


    def on_create_button_clicked(self):
        self.run_in_thread(self._on_create_button_clicked)

    def _on_create_button_clicked(self):
        selected_method = self.sampling_method_var.get()
        if selected_method == "Random":
            num_points = int(self.number_of_sampling_points_entry.get())
            sampled_mask_image, x, y = self.edgedetector.generate_sampling_points(shape='random', num_points=num_points)
        elif selected_method == "Grid":
            row_number = int(self.row_number_entry.get())
            col_number = int(self.column_number_entry.get())
            sampled_mask_image, x, y = self.edgedetector.generate_sampling_points(shape='grid', row_number=row_number, col_number=col_number)
        elif selected_method == "Rings":
            num_points = int(self.number_of_sampling_points_entry.get())
            num_rings = int(self.rings_number_entry.get())
            interval = int(self.interval_entry.get())
            sampled_mask_image, x, y = self.edgedetector.generate_sampling_points(shape='rings', num_points=num_points, num_rings=num_rings, interval=interval)

        self.sampled_mask_image = sampled_mask_image
        self.image_to_display = ImageTk.PhotoImage(self.sampled_mask_image)
        self.captured_image_label.configure(image=self.image_to_display)
        self.captured_image_label.image = self.image_to_display
        print("Sampling points generated")

    def capture_frame(self):
        self.run_in_thread(self._capture_frame) 


    def _capture_frame(self):
        try:
                self.captured_image = self.active_camera_thread.get_output_queue().get(timeout = 2)
                print("Frame captured")
                resized_image = self.captured_image.resize((self.width, self.height), Image.LANCZOS)
                self.image_to_display = ImageTk.PhotoImage(resized_image)
                self.captured_image_label.configure(image=self.image_to_display)
                self.captured_image_label.configure(text="")
                self.captured_image_label.image = self.image_to_display
                self.edge_detection_point_button.configure(state=NORMAL)

        except queue.Empty:
            print("No frame available to capture")
        except Exception as e:
            print(f"Failed to capture frame: {e}")

    def switch_view(self, view):
        if view == 'TO_WIDE':
            self.active_camera_thread = self.widefield_camera_thread
            self.camera_widget.image_queue = self.active_camera_thread.get_output_queue()
            self.camera_widget.flip = False
            self.switch_view_button.configure(text="Go To Small View")
            self.switch_view_button.configure(command= lambda: self.switch_view('TO_SMALL'))
            self.live_frame.configure(text="Wide FOV")
        elif view == 'TO_SMALL':
            self.active_camera_thread = self.smallfield_camera_thread
            self.camera_widget.image_queue = self.active_camera_thread.get_output_queue()
            self.camera_widget.flip = True
            self.switch_view_button.configure(text="Go To Wide View")
            self.switch_view_button.configure(command= lambda: self.switch_view('TO_WIDE'))
            self.live_frame.configure(text="Small FOV")


    def on_closing(self):
        print("Stopping image acquisition thread...")
        self.widefield_camera_thread.stop()
        self.smallfield_camera_thread.stop()
        self.widefield_camera_thread.join()
        self.smallfield_camera_thread.join()
        self.widefield_camera.dispose()
        self.smallfield_camera.dispose()

        self.sdk.dispose()
        self.master.destroy()


if __name__ == "__main__":
    root = ttk.Window()
    root.style.theme_use('superhero')
    live_view_frame = LiveViewModule(root)
    live_view_frame.grid(row=0, column=0, sticky='nsew')
    root.protocol("WM_DELETE_WINDOW", live_view_frame.on_closing)
    root.mainloop()
