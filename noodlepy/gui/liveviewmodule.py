import tkinter as tk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import typing
import threading
import queue
import ttkbootstrap as ttk
import numpy as np

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
    def __init__(self, parent, image_queue):
        # type: (typing.Any, queue.Queue) -> LiveViewCanvas
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0
        self._image = None
        self.tk_image = None
        tk.Canvas.__init__(self, parent)
        self.grid(row=0, column=0, sticky='nsew')
        self.bind("<Configure>", self._resize)
        self._get_image()

    def _resize(self, event=None):
        if self._image:
            new_width = self.winfo_width()
            new_height = self.winfo_height()
            resized_image = self._image.resize((new_width, new_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            self.create_image(0, 0, image=self.tk_image, anchor='nw')

    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            self._image = image.convert("RGB")  # Ensure image is in RGB mode
            self._resize()  # Trigger resize to adjust the initial image
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
    def __init__(self, parent):
        super().__init__(parent)
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
        main_frame.grid(row=0, column=0, sticky='nsew')

        self.camera_widget = LiveViewCanvas(parent=main_frame, image_queue=self.image_acquisition_thread.get_output_queue())
        
        capture_button = ttk.Button(main_frame, text="Capture Frame", command=self.capture_frame)
        capture_button.grid(row=1, column=0, sticky='nsew')

        self.captured_image = Image.new("RGB", (640, 480), "black")
        self.captured_image_label = ttk.Label(main_frame, image=ImageTk.PhotoImage(self.captured_image))
        self.captured_image_label.grid(row=3, column=0, sticky='nsew')

        self.edge_detection_button = ttk.Button(main_frame, text="Edge Detection", command=self.edge_detection, padding=5, state=DISABLED)
        self.edge_detection_button.grid(row=2, column=0, sticky='nsew')

    def edge_detection(self):
        print("Edge detection button clicked")




    def capture_frame(self):
        try:
            self.captured_image = self.image_acquisition_thread.get_output_queue().get(timeout = 2)
            self.captured_image.save("captured_frame.png")
            self.captured_image = ImageTk.PhotoImage(self.captured_image)
            # Wait for 2 seconds for a frame
            print("Frame captured")
            # update the label with the captured image
            self.captured_image_label.configure(image=self.captured_image)
            self.edge_detection_button.configure(state=NORMAL)

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
    root.protocol("WM_DELETE_WINDOW", live_view_frame.on_closing)
    root.mainloop()
