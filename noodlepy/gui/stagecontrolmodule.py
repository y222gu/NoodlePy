import os
import time
import serial
import serial.tools.list_ports
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import StringVar
import tkinter as tk
import numpy as np
from threading import Thread, Lock
import tkinter.messagebox as messagebox
from ctypes import CDLL, c_uint, c_double # for the MCL stage control

class StageControlModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        
        self.initialize_MCL_nanopositioner()

        self.img_small_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","small_step.png")).resize((20, 20))
        self.img_medium_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","medium_step.png")).resize((20, 20))
        self.img_large_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","large_step.png")).resize((20, 20))
        self.circle_black = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","circle-black.png")).resize((20, 20))
        self.ser = None
        self.create_widgets()
        self.connect_device()
        self.small_step_size_um = 0.06 #firmware seems to limit the smallest step size to 0.06 (60 um)
        self.medium_step_size_um = 0.6
        self.large_step_size_um = 6
        self.small_step_size_z_um = 0.01
        self.medium_step_size_z_um = 0.1
        self.large_step_size_z_um = 1


    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Stage Control', padding=5)
        main_frame.grid(row=0, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)

        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=6, sticky="nsew")

        # Reference buttons frame
        reference_frame = ttk.Labelframe(top_frame, text="Reference", padding=5)
        reference_frame.grid(row=0, column=0, sticky='nsew', columnspan=3, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref X", command=lambda: self.ref("X"), bootstyle="info-outline", width=5).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref Y", command=lambda: self.ref("Y"), bootstyle="info-outline", width=5).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref Z", command=lambda: self.ref("Z"), bootstyle="info-outline", width=5).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref All", command=lambda: self.ref("ALL"), bootstyle="info", width=6).grid(row=1, column=0, columnspan=3, sticky='nesw',padx=5, pady=5)

        # Frame for register
        register_frame = ttk.Labelframe(top_frame, text="Register", padding=5)
        register_frame.grid(row=0, column=3, columnspan=3, sticky='nsew', padx=5, pady=5)
        self.register_first_smaple_button = ttk.Button(register_frame, text="Record XY as first sample", command = self.register_first_smaple, bootstyle ="info", width=22)
        self.register_first_smaple_button.grid(row=0, column=0, padx=5, pady=5)
        self.register_lowest_point_button = ttk.Button(register_frame, text="Record Z as the lowest Z", command=self.register_lowest_point, bootstyle = "info", width=22)
        self.register_lowest_point_button.grid(row=1, column=0, padx=5, pady=5)

        self.remove_first_sample_button = ttk.Button(register_frame, text="x", command= self.remove_first_sample_registration,bootstyle="warning", state=DISABLED, width=2)
        self.remove_first_sample_button.grid(row=0, column=1, padx=5, pady=5)
        self.remove_lowest_point_button = ttk.Button(register_frame, text="x", command= self.remove_lowest_point_registration,bootstyle="warning", state=DISABLED, width=2)
        self.remove_lowest_point_button.grid(row=1, column=1, padx=5, pady=5)

        self.go_to_first_sample_button = ttk.Button(register_frame, text="Go", command=self.go_to_first_sample, bootstyle="info", width=3, state=DISABLED)
        self.go_to_first_sample_button.grid(row=0, column=2, padx=5, pady=5)
        self.go_to_lowest_point_button = ttk.Button(register_frame, text="Go", command=self.go_to_lowest_point, bootstyle="info", width=3, state=DISABLED)
        self.go_to_lowest_point_button.grid(row=1, column=2, padx=5, pady=5)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text='Move' , padding=5)
        movement_frame.grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        ttk.Label(movement_frame, text="X").grid(row=0, column=1)
        ttk.Label(movement_frame, text="Y").grid(row=0, column=2)
        ttk.Label(movement_frame, text="Z").grid(row=0, column=3)
        ttk.Label(movement_frame, text="Current").grid(row=1, column=0, padx=5, pady=5)
        self.p1_x_entry = ttk.Label(movement_frame, text="Nan")
        self.p1_y_entry = ttk.Label(movement_frame, text="Nan")
        self.p1_z_entry = ttk.Label(movement_frame, text="Nan")
        self.p1_x_entry.grid(row=1, column=1, padx=5, pady=5)
        self.p1_y_entry.grid(row=1, column=2, padx=5, pady=5)
        self.p1_z_entry.grid(row=1, column=3, padx=5, pady=5)
        ttk.Label(movement_frame, text="Move To").grid(row=2, column=0, padx=5, pady=5)
        self.p2_x_entry = ttk.Entry(movement_frame, width=5)
        self.p2_y_entry = ttk.Entry(movement_frame, width=5)
        self.p2_z_entry = ttk.Entry(movement_frame, width=5)
        self.p2_x_entry.grid(row=2, column=1, padx=5, pady=5)
        self.p2_y_entry.grid(row=2, column=2, padx=5, pady=5)
        self.p2_z_entry.grid(row=2, column=3, padx=5, pady=5)
        self.p2_x_entry.insert(0, "76")
        self.p2_y_entry.insert(0, "126.5")
        self.p2_z_entry.insert(0, "50")
        self.go_button = ttk.Button(movement_frame, text="Go", command=lambda: self.send_gcode("GO"),bootstyle="info")
        self.go_button.grid(row=2, column=4, padx=10, pady=5)
        self.go_button.config(width=6)
        self.go_button = ttk.Button(movement_frame, text="Go to Laser", command=lambda: self.send_gcode("GL"),bootstyle="info")
        self.go_button.grid(row=2, column=5, padx=10, pady=5)
        self.go_button.config(width=10)

        # Speed slider
        self.slider_value = StringVar()
        self.speed_slider = ttk.Scale(
            movement_frame,
            from_=1000,
            to=3000,
            orient=HORIZONTAL,
            bootstyle="info",
            variable=self.slider_value
        )
        self.speed_slider.grid(row=3, column=2, columnspan=2, pady=15)
        self.speed_slider.config(length=200)
        self.speed_slider.set(10)
        self.speed = self.speed_slider.get()
        ttk.Label(movement_frame, text="Speed").grid(row=3, column=0, padx=5, pady=5)
        ttk.Label(movement_frame, text="1000 mm/s").grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(movement_frame, text="3000 mm/s").grid(row=3, column=4, padx=5, pady=5)
        self.speed_slider.bind("<ButtonRelease-1>", lambda e: self.update_speed())

        # Frame for direction buttons
        direction_frame = ttk.Frame(movement_frame, padding=3)
        direction_frame.grid(row=4, column=0, columnspan=10, pady=3, padx=3)
        ttk.Label(direction_frame, text="Y").grid(row=0, column=4, pady=3)
        ttk.Label(direction_frame, text="X").grid(row=4, column=0, padx=3)
        ttk.Label(direction_frame, text="Z (up)").grid(row=0, column=10, pady=3)
        ttk.Label(direction_frame, text="Z (down)").grid(row=8, column=10, pady=3)
        ttk.Label(direction_frame, text="NP Z (up)").grid(row=0, column=11, pady=3)
        ttk.Label(direction_frame, text="NP Z (down)").grid(row=8, column=11, pady=3)
        self.positionLabel = ttk.Label(direction_frame, text=f"Pos: F{self.position} um")
        self.positionLabel.grid(row=9, column=11, pady=3)
        ttk.Label(direction_frame, text="").grid(row=2, column=9,columnspan=2, padx=50)
        ttk.Label(direction_frame, text="5").grid(row=3, column=1, pady=3)
        ttk.Label(direction_frame, text="0.5").grid(row=3, column=2, pady=3)
        ttk.Label(direction_frame, text="0.05").grid(row=3, column=3,  pady=3)

        self.img_up_low = ImageTk.PhotoImage(self.img_small_step.rotate(90))
        self.img_up_medium = ImageTk.PhotoImage(self.img_medium_step.rotate(-90))
        self.img_up_high = ImageTk.PhotoImage(self.img_large_step.rotate(90))
        self.img_down_low = ImageTk.PhotoImage(self.img_small_step.rotate(-90))
        self.img_down_medium = ImageTk.PhotoImage(self.img_medium_step.rotate(90))
        self.img_down_high = ImageTk.PhotoImage(self.img_large_step.rotate(-90))
        self.img_left_low = ImageTk.PhotoImage(self.img_small_step.rotate(180))
        self.img_left_medium = ImageTk.PhotoImage(self.img_medium_step)
        self.img_left_high = ImageTk.PhotoImage(self.img_large_step.rotate(180))
        self.img_right_low = ImageTk.PhotoImage(self.img_small_step)
        self.img_right_medium = ImageTk.PhotoImage(self.img_medium_step.rotate(180))
        self.img_right_high = ImageTk.PhotoImage(self.img_large_step)

        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.send_gcode("BACK SMALL"), bootstyle="light").grid(row=3, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.send_gcode("BACK MEDIUM"), bootstyle="secondary").grid(row=2, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.send_gcode("BACK LARGE"), bootstyle="dark").grid(row=1, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_left_low, command=lambda: self.send_gcode("LEFT SMALL"), bootstyle="light").grid(row=4, column=3, padx=3)
        ttk.Button(direction_frame, image=self.img_left_medium, command=lambda: self.send_gcode("LEFT MEDIUM"), bootstyle="secondary").grid(row=4, column=2, padx=3)
        ttk.Button(direction_frame, image=self.img_left_high, command=lambda: self.send_gcode("LEFT LARGE"), bootstyle="dark").grid(row=4, column=1, padx=3)
        ttk.Button(direction_frame, image=self.img_right_low, command=lambda: self.send_gcode("RIGHT SMALL"),bootstyle="light").grid(row=4, column=5, padx=3)
        ttk.Button(direction_frame, image=self.img_right_medium, command=lambda: self.send_gcode("RIGHT MEDIUM"), bootstyle="secondary").grid(row=4, column=6, padx=3)
        ttk.Button(direction_frame, image=self.img_right_high, command=lambda: self.send_gcode("RIGHT LARGE"), bootstyle="dark").grid(row=4, column=7, padx=3, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.send_gcode("FRONT SMALL"), bootstyle="light").grid(row=5, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.send_gcode("FRONT MEDIUM"), bootstyle="secondary").grid(row=6, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.send_gcode("FRONT LARGE"), bootstyle="dark").grid(row=7, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.send_gcode("UP SMALL"), bootstyle="light").grid(row=3, column=10, columnspan=2)
        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.move_relative(-1), bootstyle="light").grid(row=3, column=11, columnspan=2)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.move_relative(-5), bootstyle="secondary").grid(row=2, column=11, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.send_gcode("UP MEDIUM"), bootstyle="secondary").grid(row=2, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.send_gcode("UP LARGE"), bootstyle="dark").grid(row=1, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.send_gcode("DOWN SMALL"), bootstyle="light").grid(row=5, column=10, columnspan=2)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.move_relative(1), bootstyle="light").grid(row=5, column=11, columnspan=2)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.send_gcode("DOWN MEDIUM"), bootstyle="secondary").grid(row=6, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.move_relative(5), bootstyle="secondary").grid(row=6, column=11, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.send_gcode("DOWN LARGE"), bootstyle="dark").grid(row=7, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_right_low, command=lambda: self.move_absolute()).grid(row=4, column=11, columnspan=2)

        sample_spot_register_frame = ttk.Labelframe(main_frame, text="Sample Grid", padding=5)
        sample_spot_register_frame.grid(row=2, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)
        self.x_interval_label = ttk.Label(sample_spot_register_frame, text="X interval:")
        self.x_interval_label.grid(row=0, column=0, padx=5, pady=5)

        self.y_interval_label = ttk.Label(sample_spot_register_frame, text="Y interval:")
        self.y_interval_label.grid(row=1, column=0, padx=5, pady=5)
        self.x_number_label = ttk.Label(sample_spot_register_frame, text="# in X:")
        self.x_number_label.grid(row=0, column=2, padx=5, pady=5)
        self.y_number_label = ttk.Label(sample_spot_register_frame, text="# in Y:")
        self.y_number_label.grid(row=1, column=2, padx=5, pady=5)
        self.x_interval_entry = ttk.Entry(sample_spot_register_frame, width=5)
        self.x_interval_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_interval_entry.insert(0, "4.5")
        self.y_interval_entry = ttk.Entry(sample_spot_register_frame, width=5)
        self.y_interval_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_interval_entry.insert(0, "4.5")
        self.x_number_entry = ttk.Entry(sample_spot_register_frame, width=5)
        self.x_number_entry.grid(row=0, column=3, padx=5, pady=5)
        self.x_number_entry.insert(0, "5")
        self.y_number_entry = ttk.Entry(sample_spot_register_frame, width=5)
        self.y_number_entry.grid(row=1, column=3, padx=5, pady=5)
        self.y_number_entry.insert(0, "5")
        self.test_sample_spot_button = ttk.Button(sample_spot_register_frame, text="Test Sample Grid", command=self.test_sample_grid, bootstyle="info", state=DISABLED)
        self.test_sample_spot_button.grid(row=0, column=4, rowspan=2, sticky='nsew', padx=5, pady=5)

    def initialize_MCL_nanopositioner(self):
        
        # Load the DLL for the MCL stage control
        self.mcldll = CDLL("C:/Users/yifei/Documents/NoodlePy/noodlepy/dlls/Madlib.dll")
        self.mcldll.MCL_ReleaseHandle.restype = None
        self.mcldll.MCL_SingleReadN.restype = c_double
        
        # Initialize variables
        self.handle = self.mcldll.MCL_InitHandle()
        if self.handle == 0:
            raise RuntimeError("Failed to initialize MCL handle  (is it plugged in?)")
        print("MCL Handle = ", self.handle)

        self.axis = c_uint(3)
        self.position = c_double(0)
          
        # Move to a new position
        error = self.mcldll.MCL_SingleWriteN(self.position, self.axis, self.handle)
        print("Error = ", error)
        
        # Wait for nanopositioner to settle
        time.sleep(0.025)
        
        # Read the new position
        self.position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
        print("Position = ", self.position)

    def move_relative(self, delta_z: float):
        # Get the current position
        current_position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
     
        # Calculate the new position
        new_position = current_position + delta_z
        new_position = max(0, min(new_position, 100)) # Ensure new_position stays within bounds [0, 100]
        new_position_c_double = c_double(new_position)
        
        # Move to the new position
        error = self.mcldll.MCL_SingleWriteN(new_position_c_double, self.axis, self.handle)
        if error != 0:
            raise RuntimeError(f"MCL Error: {error}")
               
        time.sleep(0.025) # Wait for nanopositioner to settle
        
        # Read the new position
        final_position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
        print(f"Moved from {current_position:.4f} um to {final_position:.4f} um")
        self.positionLabel.config(text=f"Pos: F{final_position: .2f} um")

    def move_absolute(self, abs_z: float = 50):
        # Get the current position
        current_position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)
     
        new_position = max(0, min(abs_z, 100)) # Ensure new_position stays within bounds [0, 100]
        new_position_c_double = c_double(new_position)
        
        # Move to the new position
        error = self.mcldll.MCL_SingleWriteN(new_position_c_double, self.axis, self.handle)
        if error != 0:
            raise RuntimeError(f"MCL Error: {error}")
       
        time.sleep(0.025) # Wait for nanopositioner to settle
        
        # Read the new position
        final_position = self.mcldll.MCL_SingleReadN(self.axis, self.handle)

        # Return the final position as a string
        print(f"Moved from {current_position:.4f} um to {final_position:.4f} um")
        self.positionLabel.config(text=f"Pos: F{final_position: .2f} um")

    def run_in_thread(self, func, *args):
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()

    # def go_to_first_sample(self):
    #     self.run_in_thread(self._go_to_first_sample)

    def go_to_first_sample(self):
        if self.position_first_smaple:
            # rise to a safe height
            self.go_to_xyz(z=30)
            # move to the XY position of the first sample
            self.go_to_xyz(x=float(self.position_first_smaple[0]), y=float(self.position_first_smaple[1]))
            self.go_to_xyz(z=float(self.position_first_smaple[2]))
        else:
            print("Please register the first sample first")

    # def go_to_lowest_point(self):
    #     self.run_in_thread(self._go_to_lowest_point)

    def go_to_lowest_point(self):
        if self.lowest_point:
            self.go_to_xyz(z=float(self.lowest_point))
        else:
            print("Please register the lowest point first")

    # def send_gcode(self, option):
    #     self.run_in_thread(self._send_gcode, option)

    def send_gcode(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "UP SMALL":
            self.go_by_xyz(z=self.small_step_size_z_um)
            print("Moving up small")
        elif option == "UP MEDIUM":
            self.go_by_xyz(z=self.medium_step_size_z_um)
            print("Moving up medium")
        elif option == "UP LARGE":
            self.go_by_xyz(z=self.large_step_size_z_um)
            print("Moving up large")

        elif option == "DOWN SMALL":
            self.go_by_xyz(z=-self.small_step_size_z_um)
            print("Moving down small")
        elif option == "DOWN MEDIUM":
            self.go_by_xyz(z=-self.medium_step_size_z_um)
            print("Moving down medium")
        elif option == "DOWN LARGE":
            self.go_by_xyz(z=-self.large_step_size_z_um)
            print("Moving down large")

        elif option == "LEFT SMALL":
            self.go_by_xyz(x=-self.small_step_size_um)
            print("Moving left small")
        elif option == "LEFT MEDIUM":
            self.go_by_xyz(x=-self.medium_step_size_um)
            print("Moving left medium")
        elif option == "LEFT LARGE":
            self.go_by_xyz(x=-self.large_step_size_um)
            print("Moving left large")

        elif option == "RIGHT SMALL":
            self.go_by_xyz(x=self.small_step_size_um)
            print("Moving right small")
        elif option == "RIGHT MEDIUM":
            self.go_by_xyz(x=self.medium_step_size_um)
            print("Moving right medium")
        elif option == "RIGHT LARGE":
            self.go_by_xyz(x=self.large_step_size_um)
            print("Moving right large")

        elif option == "BACK SMALL":
            self.go_by_xyz(y=self.small_step_size_um)
            print("Moving back small")
        elif option == "BACK MEDIUM":
            self.go_by_xyz(y=self.medium_step_size_um)
            print("Moving back medium")
        elif option == "BACK LARGE":
            self.go_by_xyz(y=self.large_step_size_um)
            print("Moving back large")

        elif option == "FRONT SMALL":
            self.go_by_xyz(y=-self.small_step_size_um)
            print("Moving front small")
        elif option == "FRONT MEDIUM":
            self.go_by_xyz(y=-self.medium_step_size_um)
            print("Moving front medium")
        elif option == "FRONT LARGE":
            self.go_by_xyz(y=-self.large_step_size_um)
            print("Moving front large")
        elif option == "GO":
            p2_x = self.p2_x_entry.get()
            p2_y = self.p2_y_entry.get()
            p2_z = self.p2_z_entry.get()

            if p2_x and p2_y and p2_z:
                self.go_to_xyz(x=p2_x, y=p2_y, z=p2_z)
            else:
                print("Please enter all the coordinates")
                return
        elif option == "GL":
            # manual calibration from 08/31/2024
            x_cal = 41.38
            y_cal = -3.66
            z_cal = -3
            p2_x = str(float(self.p1_x_entry.cget("text"))+x_cal)
            p2_y = str(float(self.p1_y_entry.cget("text"))+y_cal)
            p2_z = str(float(self.p1_z_entry.cget("text"))+z_cal)

            if p2_x and p2_y and p2_z:
                label_text = self.p1_z_entry.cget("text") # Get the text from the label

                try:
                    current_z = float(label_text)
                    safe_z = 30
                    self.go_to_xyz(z=str(current_z+safe_z)) # move to a safe height
                    self.go_to_xyz(x=p2_x, y=p2_y) # travel x-y
                    self.go_to_xyz(z=p2_z) #descend to the correct height
                except ValueError:
                    # Handle the case where the text is not a valid float, e.g., "Nan"
                    messagebox.showerror("Conversion Error", f"Cannot convert '{label_text}' to float.")

                
            else:
                print("Please enter all the coordinates")
                return
        else:
            print("Invalid option")
    

    def ref(self, option):
        self.run_in_thread(self._ref, option)


    def _ref(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "X":
            self.ser.write(str.encode("G28 X\r\n"))
            self.get_current_position('X')
        elif option == "Y":
            self.ser.write(str.encode("G28 Y\r\n"))
            self.get_current_position('Y')
        elif option == "Z":
            self.ser.write(str.encode("G28 Z\r\n"))
            self.get_current_position('Z')
        elif option == "ALL":
            self.ser.write(str.encode("G28 X Y Z\r\n"))
            self.ser.write(str.encode("G90\r\n"))
            self.ser.write(str.encode("G0 X0 Y0 Z0 F3000\r\n"))
            self.get_current_position('XYZ')
        else:
            print("Invalid option")
            return
        
    def test_sample_grid(self):
        self.run_in_thread(self._test_sample_grid)

    def _test_sample_grid(self):

        if not self.x_interval_entry.get() or not self.y_interval_entry.get() or not self.x_number_entry.get() or not self.y_number_entry.get():
            print("Please enter all the parameters")
            return
        
        x_interval = self.x_interval_entry.get()
        y_interval = self.y_interval_entry.get()
        x_number = self.x_number_entry.get()
        y_number = self.y_number_entry.get()

        if self.position_first_smaple is None:
            print("Please register the first sample first")
            return

        if x_interval and y_interval and x_number and y_number:
            first_x = float(self.position_first_smaple[0])
            first_y = float(self.position_first_smaple[1])
            first_z = float(self.position_first_smaple[2])

            x = np.linspace(first_x, first_x + float(x_interval) * (int(x_number) - 1), int(x_number))
            y = np.linspace(first_y - float(y_interval) * (int(y_number) - 1), first_y, int(y_number))
            xx, yy = np.meshgrid(x, y)
            # make y descending order
            yy = np.flip(yy, axis=0)
            xx = xx.flatten(order='F')
            yy = yy.flatten(order='F')
            # make 
            zz = np.ones(xx.size) * first_z

            print(f"A grid containing {xx.size} points will be tested")
        else:
            print("Please enter all the parameters")
        for i in range(xx.size):
            self.go_to_xyz(x=xx[i], y=yy[i], z=zz[i])
            print(f"Moving to {xx[i]}, {yy[i]}, {zz[i]}")
            time.sleep(1)
        print("Test completed")


    def go_to_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G90\r\n"))
        gcode = "G0"
        if x is not None:
            gcode += f" X{x}"
        if y is not None:
            gcode += f" Y{y}"
        if z is not None:
            gcode += f" Z{z}"

        gcode += f" F{self.speed}\r\n"
        self.ser.write(str.encode(gcode))
        self.get_current_position('XYZ')

    def go_by_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G91\r\n"))
        gcode = "G1"
        if x is not None:
            gcode += f" X{x}"
        if y is not None:
            gcode += f" Y{y}"
        if z is not None:
            gcode += f" Z{z}"

        gcode += f" F{self.speed}\r\n"
        self.ser.write(str.encode(gcode))
        self.get_current_position('XYZ')


    def find_printer_com_ports():
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.description == "Original Prusa i3 MK3 (COM3)":
                return port.name

    def wait_for_process_complete(self, process_name, critiria):
        """Waits for a signal from the printer that homing is complete."""
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            if critiria in line:
                print(f"Received during {process_name}: {line}")
                return line
            elif line:
                print(f"Received during {process_name}: {line}")

    def update_speed(self):
        self.speed = self.speed_slider.get()
        print(f"Speed of stage is updated to: {self.speed} mm/s")

    def connect_device(self):
        self.port = StageControlModule.find_printer_com_ports()
        self.ser = serial.Serial(self.port, 115200)
        
        printer_status = StageControlModule.is_printer_on(self.ser)
        if printer_status:
            print("Connected to the printer " + self.ser.name)
        else:
            self.ser.close()
            print("Couldn't connect to the printer")

    def is_printer_on(ser):
        try:
            ser.flushInput()
            ser.flushOutput()
            ser.write(b'M105\n')
            time.sleep(3)
            response = ser.read_all().decode('utf-8')
            print(response)

            if 'start\necho:' in response:
                return True
            else:
                return False
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return False

    def get_current_position(self, dim = 'XYZ'):
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write(b'M114\n')
        line = self.wait_for_process_complete(process_name = "get_current_position", critiria='X')
        if dim == 'XYZ':
            X = line.split(' ')[0].split(':')[1]
            Y = line.split(' ')[1].split(':')[1]
            Z = line.split(' ')[2].split(':')[1]
            self.p1_x_entry.config(text=X)
            self.p1_y_entry.config(text=Y)
            self.p1_z_entry.config(text=Z)   
            return X, Y, Z
        elif dim == 'X':
            X = line.split(' ')[0].split(':')[1]
            self.p1_x_entry.config(text=X)
            return X
        elif dim == 'Y':
            Y = line.split(' ')[1].split(':')[1]
            self.p1_y_entry.config(text=Y)
            return Y
        elif dim == 'Z':
            Z = line.split(' ')[2].split(':')[1]
            self.p1_z_entry.config(text=Z)
            return Z

    def register_first_smaple(self):
        self.position_first_smaple = self.get_current_position('XYZ')
        self.register_first_smaple_button.configure(text="First Sample Registered")
        self.register_first_smaple_button.configure(state=DISABLED)
        self.remove_first_sample_button.configure(state=NORMAL)
        self.go_to_first_sample_button.configure(state=NORMAL)
        self.test_sample_spot_button.configure(state=NORMAL)
        print("First sample registered")

    def register_lowest_point(self):
        current_position = self.get_current_position('XYZ')
        self.lowest_point = current_position[2]
        self.register_lowest_point_button.configure(text="Lowest Z Point recorded")
        self.register_lowest_point_button.configure(state=DISABLED)
        self.remove_lowest_point_button.configure(state=NORMAL)
        self.go_to_lowest_point_button.configure(state=NORMAL)
        print("Lowest Z registered")

    def remove_first_sample_registration(self):
        self.position_first_smaple = None
        print("Registration of the first sample removed")
        self.remove_first_sample_button.configure(state=DISABLED)
        self.register_first_smaple_button.configure(text="Record XY as first sample")
        self.register_first_smaple_button.configure(state=NORMAL)
        self.register_first_smaple_button.configure(bootstyle="info")
        self.go_to_first_sample_button.configure(state=DISABLED)
        self.test_sample_spot_button.configure(state=DISABLED)

    def remove_lowest_point_registration(self):
        self.lowest_point = None
        print("Registration of the lowest Z removed")
        self.remove_lowest_point_button.configure(state=DISABLED)
        self.register_lowest_point_button.configure(text ='Record Z as the lowest Z', state = NORMAL, bootstyle = 'info')
        self.go_to_lowest_point_button.configure(state=DISABLED)

    def update_label(self, value):
        self.slider_value.set(f"{float(value):.2f} mm/s")

if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = StageControlModule(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()