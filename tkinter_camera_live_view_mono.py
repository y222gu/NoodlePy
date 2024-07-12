"""
Camera Live View - TkInter

This example shows how one could create a live image viewer using TkInter.
It also uses the third party library 'pillow', which is a fork of PIL.

This example detects if a camera is a color camera and will process the
images using the tl_mono_to_color_processor module.

This example uses threading to enqueue images coming off the camera in one thread, and
dequeue them in the UI thread for quick displaying.

"""

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    print("Configuring path...")
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

import tkinter as tk
from PIL import Image, ImageTk
import typing
import threading
import queue
import numpy as np
import cv2
import pandas as pd

""" LiveViewCanvas

This is a Tkinter Canvas object that can be reused in custom programs. The Canvas expects a parent Tkinter object and 
an image queue. The image queue is a queue.Queue that it will pull images from, and is expected to hold PIL Image 
objects that will be displayed to the canvas. It automatically adjusts its size based on the incoming image dimensions.

"""
class LiveViewCanvas(tk.Canvas):

    def __init__(self, parent, image_queue):
        # type: (typing.Any, queue.Queue) -> LiveViewCanvas
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0
        tk.Canvas.__init__(self, parent)
        self.pack()
        self._get_image()

    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            self._image = ImageTk.PhotoImage(master=self, image=image)
            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                # resize the canvas to match the new image size
                self._image_width = self._image.width()
                self._image_height = self._image.height()
                self.config(width=self._image_width, height=self._image_height)
            self.create_image(0, 0, image=self._image, anchor='nw')
        except queue.Empty:
            pass
        self.after(10, self._get_image)

def find_spots(image):
    
        """
        Detect spots in the image and return their top center coordinates.

        :param image: Input image
        :param threshold: Threshold value for spot detection
        :return: List of coordinates of the top center of each spot
        """
        df = pd.DataFrame(image)
        df.to_csv("file.csv")

        im = Image.fromarray(image)
        im.save("img0.png","PNG")

        # Convert to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binary thresholding
        # _, thresh = cv2.threshold(image, 165, 200, cv2.THRESH_BINARY)

        # Find circles
        minDist = 200
        circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,minDist,
                                   param1=35,param2=30,minRadius=150,maxRadius=200)
        circles = np.uint16(np.around(circles))
        print("number of circles: ", len(circles[0,:]))
        if circles is None:
            return image
        for i in circles[0,:]:
            print("i is: ", i)
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
        
        im = Image.fromarray(image)
        im.save("img0+circle.png","PNG")

        return image

""" ImageAcquisitionThread

This class derives from threading.Thread and is given a TLCamera instance during initialization. When started, the 
thread continuously acquires frames from the camera and converts them to PIL Image objects. These are placed in a 
queue.Queue object that can be retrieved using get_output_queue(). The thread doesn't do any arming or triggering, 
so users will still need to setup and control the camera from a different thread. Be sure to call stop() when it is 
time for the thread to stop.

"""
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
        # type: (type(None)) -> queue.Queue
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        # type: (Frame) -> Image
        # no coloring, just scale down image to 8 bpp and place into PIL Image object
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)

        image8bit = np.asarray(scaled_image, np.uint8)
        processed_image = find_spots(image8bit.squeeze())
   
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


""" Main

When run as a script, a simple Tkinter app is created with just a LiveViewCanvas widget. 

"""
if __name__ == "__main__":
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        with sdk.open_camera(camera_list[0]) as camera:

            # create generic Tk App with just a LiveViewCanvas widget
            print("Generating app...")
            root = tk.Tk()
            # root.title(camera.name)
            root.title(" CS165MU/M Camera Live View")
            image_acquisition_thread = ImageAcquisitionThread(camera)
            camera_widget = LiveViewCanvas(parent=root, image_queue=image_acquisition_thread.get_output_queue())

            print("Setting camera parameters...")
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.arm(2)
            camera.issue_software_trigger()

            print("Starting image acquisition thread...")
            image_acquisition_thread.start()

            print("App starting")
            
            root.mainloop()

            print("Waiting for image acquisition thread to finish...")
            image_acquisition_thread.stop()
            image_acquisition_thread.join()

            print("Closing resources...")

    print("App terminated. Goodbye!")
