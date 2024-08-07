import tkinter as tk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import typing
import threading
import queue
import ttkbootstrap as ttk
import numpy as np
from noodlepy.gui.sampledetectionmodule import EdgeDetector
import os

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
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
        # type: (typing.Any, queue.Queue) -> LiveViewCanvas
        self.image_queue = image_queue
        self._image_width = width
        self._image_height = height
        self._image = None
        self.tk_image = None
        tk.Canvas.__init__(self, parent)
        self.grid(row=0, column=0, sticky='nsew')
        # set the size of the canvas to match the incoming image size
        # self.bind("<Configure>", self._resize)
        self._get_image()

    def _resize(self, event=None):
        if self._image:
            # new_width = self.winfo_width()
            # new_height = self.winfo_height()
            # resized_image = self._image.resize((new_width, new_height), Image.LANCZOS)
            resized_image = self._image.resize((self._image_width, self._image_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            self.create_image(0, 0, image=self.tk_image, anchor='nw')

    def _get_image(self):
        try:
            self._image = self.image_queue.get_nowait()
            self._resize()
            # self._image = ImageTk.PhotoImage(master=self, image=self._image)

        except queue.Empty:
            pass
        self.after(10, self._get_image)

class ImageAcquisitionThread(threading.Thread):
    def __init__(self, camera):
        # type: (TLCamera) -> ImageAcquisitionThread
        super(ImageAcquisitionThread, self).__init__()
        camera.exposure_time_us = 40000
        self._camera = camera
        self._previous_timestamp = 0
        self._is_color = False
        self._bit_depth = camera.bit_depth
        
        # print("bit_depth is: ", self._bit_depth)  # 10
        self._camera.image_poll_timeout_ms = 0  # Do not want to block for long periods of time
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        # type: (Frame) -> Image
        # no coloring, just scale down image to 8 bpp and place into PIL Image object
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
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        print("Image acquisition has stopped")


class LiveViewModule(tk.Frame):
    def __init__(self, parent, width=288, height=216):
        super().__init__(parent)
        self.width = width
        self.height = height
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
        main_frame = ttk.Labelframe(self, text="Live View")
        main_frame.grid(row=0, column=0, columnspan=3, sticky='nsew')

        self.camera_widget = LiveViewCanvas(parent=main_frame, image_queue=self.image_acquisition_thread.get_output_queue(), width=self.width, height=self.height)
        self.camera_widget.grid(row=0, column=0,columnspan=3, sticky='nsew')
        
        capture_button = ttk.Button(main_frame, text="Capture Frame", command=self.capture_frame, padding=5, style='success')
        capture_button.grid(row=1, column=1, sticky='nsew')

        # initialize captured image to grey image in numpy array and convert to PIL image
        self.initial_image = Image.fromarray(np.zeros((self.height, self.width), dtype=np.uint8))
        self.image_to_display = ImageTk.PhotoImage(self.initial_image)
        self.captured_image_label = ttk.Label(main_frame, text="No Frame Captured", image = self.image_to_display, padding=5, compound='center', foreground='white')
        self.captured_image_label.grid(row=3, column=0, columnspan=3, sticky='nsew')
        self.captured_image_label.image = self.image_to_display

        self.edge_detection_auto_button = ttk.Button(main_frame, text="Auto Detection", command=lambda: self.edge_detection('auto'), padding=5, state=DISABLED, style='success')
        self.edge_detection_auto_button.grid(row=2, column=0, sticky='ew')

        self.edge_detection_point_button = ttk.Button(main_frame, text="Point Detection", command=lambda: self.edge_detection('point'), padding=5, state=DISABLED, style='success')
        self.edge_detection_point_button.grid(row=2, column=1, sticky='ew')

        self.edge_detection_box_button = ttk.Button(main_frame, text="Box Detection", command=lambda: self.edge_detection('box'), padding=5, state=DISABLED, style='success')
        self.edge_detection_box_button.grid(row=2, column=2, sticky='ew')


    def edge_detection(self, detection_type):
        edgedetector = EdgeDetector(self.captured_image)

        if detection_type == "auto":
            masked_image = edgedetector.auto_mask_generate()
        elif detection_type == "point":
            masked_image =edgedetector.point_prompt_mask_generate()
        elif detection_type == "box":   
            masked_image = edgedetector.box_prompt_mask_generate()
        print("Edge detection button clicked")

        self.masked_image = masked_image
        self.image_to_display = ImageTk.PhotoImage(self.masked_image)
        self.captured_image_label.configure(image=self.image_to_display)
        self.captured_image_label.image = self.image_to_display

    def capture_frame(self):
        try:
            self.captured_image = self.image_acquisition_thread.get_output_queue().get(timeout = 2)
            # image_path = os.path.join(os.getcwd(), "output_plots", "captured_frame.png")
            # self.captured_image.save("captured_frame.png")
            # Wait for 2 seconds for a frame
            print("Frame captured")
            # resize the image
            resized_image = self.captured_image.resize((self.width, self.height), Image.LANCZOS)
            # update the label with the captured image
            self.image_to_display = ImageTk.PhotoImage(resized_image)
            self.captured_image_label.configure(image=self.image_to_display)
            # remove the text from the label
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
    live_view_frame.pack(fill='both', expand=True)
    root.protocol("WM_DELETE_WINDOW", live_view_frame.on_closing)
    root.mainloop()

