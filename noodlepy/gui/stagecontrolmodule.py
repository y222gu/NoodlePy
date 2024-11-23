import os
import time
import serial
import serial.tools.list_ports
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk
from tkinter import StringVar
import tkinter as tk
import numpy as np
from threading import Thread, Lock
from noodlepy.gui.publisher_subscriber import Subscriber, Publisher
class StageControlModule(ttk.Frame, Publisher, Subscriber):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        Publisher.__init__(self, ['move_nanodrive_by', 'move_nanodrive_to'])
        Subscriber.__init__(self)
        self.name = 'StageControlModule_obserableobserver'

        self.img_small_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","small_step.png")).resize((20, 20))
        self.img_medium_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","medium_step.png")).resize((20, 20))
        self.img_large_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","large_step.png")).resize((20, 20))
        self.circle_black = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","circle-black.png")).resize((20, 20))
        self.nanodrive_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","nanodrive_arrow_1.png")).resize((20, 20))
       
        self.position_first_smaple = [154.02, 148.62, 30]
        self.calibration_from_widefield_to_objective_x = -65.1 # calibrated on 11/21/2024
        self.calibration_from_widefield_to_objective_y = -5.94 # calibrated on 11/21/2024
        self.small_step_size_mm = 0.06 #firmware seems to limit the smallest step size to 0.06 (60 um)
        self.medium_step_size_mm = 0.6
        self.large_step_size_mm = 6
        self.small_step_size_z_mm = 0.01
        self.medium_step_size_z_mm = 0.1
        self.large_step_size_z_mm = 1
        self.nanodrive_step_size_um = 1
        self.current_nanodrive_position = None
        self.prusa_x_referenced = False
        self.prusa_y_referenced = False
        self.prusa_z_referenced = False
        self.capture_frame_center_widefield_x= None
        self.capture_frame_center_widefield_y = None
        self.capture_frame_center_objective_x= None
        self.capture_frame_center_objective_y = None
        self.ser = None
        self.create_widgets()
        self.connect_prusa_device()



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
        ttk.Button(reference_frame, text="Ref NP", command=lambda: self.ref("nanodrive"), bootstyle="info-outline", width=7).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref All", command=lambda: self.ref("ALL"), bootstyle="info", width=6).grid(row=1, column=0, columnspan=4, sticky='nesw',padx=5, pady=5)

        # Frame for register
        register_frame = ttk.Labelframe(top_frame, text="Register", padding=5)
        register_frame.grid(row=0, column=3, columnspan=3, sticky='nsew', padx=5, pady=5)
        # self.register_first_smaple_button = ttk.Button(register_frame, text="Record XY as first sample", command = self.register_first_smaple, bootstyle ="info", width=22)
        # self.register_first_smaple_button.grid(row=0, column=0, padx=5, pady=5)
        self.register_lowest_point_button = ttk.Button(register_frame, text="Record Z as the lowest Z", command=self.register_lowest_point, bootstyle = "info", width=22)
        self.register_lowest_point_button.grid(row=1, column=0, padx=5, pady=5)

        # self.remove_first_sample_button = ttk.Button(register_frame, text="x", command= self.remove_first_sample_registration,bootstyle="warning", state=DISABLED, width=2)
        # self.remove_first_sample_button.grid(row=0, column=1, padx=5, pady=5)
        self.remove_lowest_point_button = ttk.Button(register_frame, text="x", command= self.remove_lowest_point_registration,bootstyle="warning", state=DISABLED, width=2)
        self.remove_lowest_point_button.grid(row=1, column=1, padx=5, pady=5)

        self.go_to_first_sample_button = ttk.Button(register_frame, text="Go", command=self.prusa_go_to_first_sample_xy, bootstyle="info", width=3, state=DISABLED)
        self.go_to_first_sample_button.grid(row=0, column=2, padx=5, pady=5)
        self.go_to_lowest_point_button = ttk.Button(register_frame, text="Go", command=self.prusa_go_to_lowest_z, bootstyle="info", width=3, state=DISABLED)
        self.go_to_lowest_point_button.grid(row=1, column=2, padx=5, pady=5)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text='Move' , padding=5)
        movement_frame.grid(row=1, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        ttk.Label(movement_frame, text="X [mm]").grid(row=0, column=1)
        ttk.Label(movement_frame, text="Y [mm]").grid(row=0, column=2)
        ttk.Label(movement_frame, text="Z [mm]").grid(row=0, column=3)
        ttk.Label(movement_frame, text="ND Z [um]").grid(row=0, column=4)
        ttk.Label(movement_frame, text="Current").grid(row=1, column=0, padx=5, pady=5)
        self.current_x_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_y_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_z_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_nanodrive_z_entry = ttk.Label(movement_frame, text= 'N/A')
        self.current_x_entry.grid(row=1, column=1, padx=5, pady=5)
        self.current_y_entry.grid(row=1, column=2, padx=5, pady=5)
        self.current_z_entry.grid(row=1, column=3, padx=5, pady=5)
        self.current_nanodrive_z_entry.grid(row=1, column=4, padx=5, pady=5)

        ttk.Label(movement_frame, text="Move To").grid(row=2, column=0, padx=5, pady=5)
        self.target_x_entry = ttk.Entry(movement_frame, width=8)
        self.target_y_entry = ttk.Entry(movement_frame, width=8)
        self.target_z_entry = ttk.Entry(movement_frame, width=8)
        self.target_nanodrive_entry = ttk.Spinbox(movement_frame, from_=-100, to=0, width=8)
        self.target_x_entry.grid(row=2, column=1, padx=5, pady=5)
        self.target_y_entry.grid(row=2, column=2, padx=5, pady=5)
        self.target_z_entry.grid(row=2, column=3, padx=5, pady=5)
        self.target_nanodrive_entry.grid(row=2, column=4, padx=5, pady=5)
        self.target_x_entry.insert(0, self.position_first_smaple[0])
        self.target_y_entry.insert(0, self.position_first_smaple[1])
        self.target_z_entry.insert(0, self.position_first_smaple[2])
        self.target_nanodrive_entry.insert(0, "0")
        self.go_button = ttk.Button(movement_frame, text="Go", command=self.move_both_prusa_nanodrive, bootstyle="info")
        self.go_button.grid(row=2, column=5, padx=10, pady=5)
        self.go_button.config(width=6)

        # Speed slider
        self.slider_value = StringVar()
        self.slider_value.set(200)
        self.speed_slider = ttk.Scale(
            movement_frame,
            from_=10,
            to=1000,
            orient=HORIZONTAL,
            bootstyle="info",
            variable=self.slider_value
        )
        self.speed_slider.grid(row=3, column=2, columnspan=2, pady=15)
        self.speed_slider.config(length=200)
        self.speed = self.speed_slider.get()
        ttk.Label(movement_frame, text="Speed").grid(row=3, column=0, padx=5, pady=5)
        ttk.Label(movement_frame, text="1000 mm/s").grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(movement_frame, text="5000 mm/s").grid(row=3, column=4, padx=5, pady=5)
        self.speed_slider.bind("<ButtonRelease-1>", lambda e: self.update_prusa_speed())

        # Frame for direction buttons
        direction_frame = ttk.Frame(movement_frame, padding=3)
        direction_frame.grid(row=4, column=0, columnspan=10, pady=3, padx=3)
        ttk.Label(direction_frame, text="Y").grid(row=0, column=4, pady=3)
        ttk.Label(direction_frame, text="X").grid(row=4, column=0, padx=3)
        ttk.Label(direction_frame, text="Z (up)").grid(row=0, column=10, pady=3)
        ttk.Label(direction_frame, text="Z (down)").grid(row=8, column=10, pady=3)
        ttk.Label(direction_frame, text="ND Z (up)").grid(row=0, column=11, pady=3)
        ttk.Label(direction_frame, text="ND Z (down)").grid(row=8, column=11, pady=3)
        ttk.Label(direction_frame, text="").grid(row=2, column=9,columnspan=2, padx=50)
        ttk.Label(direction_frame, text="6 mm").grid(row=1, column=3, pady=3)
        ttk.Label(direction_frame, text="0.6 mm").grid(row=2, column=3, pady=3)
        ttk.Label(direction_frame, text="0.06 mm").grid(row=3, column=3,  pady=3)

        ttk.Label(direction_frame, text="1 mm").grid(row=1, column=9, pady=3)
        ttk.Label(direction_frame, text="0.1 mm").grid(row=2, column=9, pady=3)
        ttk.Label(direction_frame, text="0.01 mm").grid(row=3, column=9, pady=3)

        ttk.Label(direction_frame, text="1 um").grid(row=4, column=11, pady=3)

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
        self.img_up_nano = ImageTk.PhotoImage(self.nanodrive_icon.rotate(180))
        self.img_down_nano = ImageTk.PhotoImage(self.nanodrive_icon)


        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.move_prusa("BACK SMALL"), bootstyle="light").grid(row=3, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.move_prusa("BACK MEDIUM"), bootstyle="secondary").grid(row=2, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.move_prusa("BACK LARGE"), bootstyle="dark").grid(row=1, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_left_low, command=lambda: self.move_prusa("LEFT SMALL"), bootstyle="light").grid(row=4, column=3, padx=3)
        ttk.Button(direction_frame, image=self.img_left_medium, command=lambda: self.move_prusa("LEFT MEDIUM"), bootstyle="secondary").grid(row=4, column=2, padx=3)
        ttk.Button(direction_frame, image=self.img_left_high, command=lambda: self.move_prusa("LEFT LARGE"), bootstyle="dark").grid(row=4, column=1, padx=3)
        ttk.Button(direction_frame, image=self.img_right_low, command=lambda: self.move_prusa("RIGHT SMALL"),bootstyle="light").grid(row=4, column=5, padx=3)
        ttk.Button(direction_frame, image=self.img_right_medium, command=lambda: self.move_prusa("RIGHT MEDIUM"), bootstyle="secondary").grid(row=4, column=6, padx=3)
        ttk.Button(direction_frame, image=self.img_right_high, command=lambda: self.move_prusa("RIGHT LARGE"), bootstyle="dark").grid(row=4, column=7, padx=3, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.move_prusa("FRONT SMALL"), bootstyle="light").grid(row=5, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.move_prusa("FRONT MEDIUM"), bootstyle="secondary").grid(row=6, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.move_prusa("FRONT LARGE"), bootstyle="dark").grid(row=7, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.move_prusa("UP SMALL"), bootstyle="light").grid(row=3, column=10)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.move_prusa("UP MEDIUM"), bootstyle="secondary").grid(row=2, column=10, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.move_prusa("UP LARGE"), bootstyle="dark").grid(row=1, column=10, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.move_prusa("DOWN SMALL"), bootstyle="light").grid(row=5, column=10)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.move_prusa("DOWN MEDIUM"), bootstyle="secondary").grid(row=6, column=10, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.move_prusa("DOWN LARGE"), bootstyle="dark").grid(row=7, column=10, pady=3)
        ttk.Button(direction_frame, image=self.img_up_nano, command=lambda: self.move_nanodrive("UP"), bootstyle="light").grid(row=3, column=11)
        ttk.Button(direction_frame, image=self.img_down_nano, command=lambda: self.move_nanodrive("DOWN"), bootstyle="light").grid(row=5, column=11)

    def run_in_thread(self, func, *args):
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()

    def prusa_go_to_first_sample_xy(self):
        # if self.position_first_smaple:
        #     # rise to a safe height
        #     self.prusa_go_to_xyz(z=30)
        #     # move to the XY position of the first sample
        #     self.prusa_go_to_xyz(x=float(self.position_first_smaple[0]), y=float(self.position_first_smaple[1]))
        #     self.prusa_go_to_xyz(z=float(self.position_first_smaple[2]))
        # else:
        #     print("Please register the first sample first")
        self.prusa_go_to_xyz(x=float(self.position_first_smaple[0]), y=float(self.position_first_smaple[1]), z=float(self.position_first_smaple[2]))

    def prusa_go_to_lowest_z(self):
        if self.lowest_point:
            self.prusa_go_to_xyz(z=float(self.lowest_point))
        else:
            print("Please register the lowest point first")

    def move_both_prusa_nanodrive(self):
        print("Moving both the stage and the nanodrive")
        self.move_prusa("GO")
        self.move_nanodrive("GO")

    def move_nanodrive(self, direction):
        if direction == "UP":
            self.dispatch('move_nanodrive_by', -self.nanodrive_step_size_um)
        elif direction == "DOWN":
            self.dispatch('move_nanodrive_by', self.nanodrive_step_size_um)
        elif direction == "GO":
            self.dispatch('move_nanodrive_to', self.target_nanodrive_entry.get())

    def move_prusa(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "UP SMALL":
            self.prusa_go_by_xyz(z=self.small_step_size_z_mm)
            print("Moving up small")
        elif option == "UP MEDIUM":
            self.prusa_go_by_xyz(z=self.medium_step_size_z_mm)
            print("Moving up medium")
        elif option == "UP LARGE":
            self.prusa_go_by_xyz(z=self.large_step_size_z_mm)
            print("Moving up large")

        elif option == "DOWN SMALL":
            self.prusa_go_by_xyz(z=-self.small_step_size_z_mm)
            print("Moving down small")
        elif option == "DOWN MEDIUM":
            self.prusa_go_by_xyz(z=-self.medium_step_size_z_mm)
            print("Moving down medium")
        elif option == "DOWN LARGE":
            self.prusa_go_by_xyz(z=-self.large_step_size_z_mm)
            print("Moving down large")

        elif option == "LEFT SMALL":
            self.prusa_go_by_xyz(x=-self.small_step_size_mm)
            print("Moving left small")
        elif option == "LEFT MEDIUM":
            self.prusa_go_by_xyz(x=-self.medium_step_size_mm)
            print("Moving left medium")
        elif option == "LEFT LARGE":
            self.prusa_go_by_xyz(x=-self.large_step_size_mm)
            print("Moving left large")

        elif option == "RIGHT SMALL":
            self.prusa_go_by_xyz(x=self.small_step_size_mm)
            print("Moving right small")
        elif option == "RIGHT MEDIUM":
            self.prusa_go_by_xyz(x=self.medium_step_size_mm)
            print("Moving right medium")
        elif option == "RIGHT LARGE":
            self.prusa_go_by_xyz(x=self.large_step_size_mm)
            print("Moving right large")

        elif option == "BACK SMALL":
            self.prusa_go_by_xyz(y=self.small_step_size_mm)
            print("Moving back small")
        elif option == "BACK MEDIUM":
            self.prusa_go_by_xyz(y=self.medium_step_size_mm)
            print("Moving back medium")
        elif option == "BACK LARGE":
            self.prusa_go_by_xyz(y=self.large_step_size_mm)
            print("Moving back large")

        elif option == "FRONT SMALL":
            self.prusa_go_by_xyz(y=-self.small_step_size_mm)
            print("Moving front small")
        elif option == "FRONT MEDIUM":
            self.prusa_go_by_xyz(y=-self.medium_step_size_mm)
            print("Moving front medium")
        elif option == "FRONT LARGE":
            self.prusa_go_by_xyz(y=-self.large_step_size_mm)
            print("Moving front large")
        elif option == "GO":
            p2_x = self.target_x_entry.get()
            p2_y = self.target_y_entry.get()
            p2_z = self.target_z_entry.get()

            if p2_x and p2_y and p2_z:
                # raise to a safe height
                self.prusa_go_to_xyz(z=30)
                self.prusa_go_to_xyz(x=p2_x, y=p2_y, z=p2_z)
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
            self.prusa_x_referenced = True
        elif option == "Y":
            self.ser.write(str.encode("G28 Y\r\n"))
            self.prusa_y_referenced = True
        elif option == "Z":
            self.ser.write(str.encode("G28 Z\r\n"))
            self.prusa_z_referenced = True
        elif option == "nanodrive":
            self.dispatch('move_nanodrive_to', 0)
        elif option == "ALL":
            # move to a safe height
            self.prusa_go_by_xyz(z=5)

            self.ser.write(str.encode("G28 X Y Z\r\n"))
            self.ser.write(str.encode("G90\r\n"))
            self.prusa_x_referenced = True
            self.prusa_y_referenced = True
            self.prusa_z_referenced = True
            self.dispatch('move_nanodrive_to', 0)
        else:
            print("Invalid option")
            return

    def prusa_go_to_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G90\r\n"))
        gcode = "G0"
        if x is not None:
            if self.prusa_x_referenced:
                gcode += f" X{x}"
        if y is not None:
            if self.prusa_y_referenced:
                gcode += f" Y{y}"
        if z is not None:
            if self.prusa_z_referenced:
                gcode += f" Z{z}"

        if x is not None or y is not None or z is not None:
            gcode += f" F{self.speed}\r\n"
            self.ser.write(str.encode(gcode))
            self.update_prusa_position()

    def prusa_go_by_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G91\r\n"))
        gcode = "G1"
        if x is not None:
            if self.prusa_x_referenced:
                gcode += f" X{x}"
        if y is not None:
            if self.prusa_y_referenced:
                gcode += f" Y{y}"
        if z is not None:
            if self.prusa_z_referenced:
                gcode += f" Z{z}"

        if x is not None or y is not None or z is not None:
            gcode += f" F{self.speed}\r\n"
            self.ser.write(str.encode(gcode))
            self.update_prusa_position()

    def find_prusa_com_ports():
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.description == "Original Prusa i3 MK3 (COM3)":
                return port.name

    def wait_for_prusa_process_complete(self, process_name, critiria):
        """Waits for a signal from the printer that homing is complete."""
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            if critiria in line:
                # print(f"Received during {process_name}: {line}")
                return line
            # elif line:
                # print(f"Received during {process_name}: {line}")
                

    def update_prusa_speed(self):
        self.speed = self.speed_slider.get()
        print(f"Speed of stage is updated to: {self.speed} mm/s")

    def connect_prusa_device(self):
        self.port = StageControlModule.find_prusa_com_ports()
        self.ser = serial.Serial(self.port, 115200)
        
        printer_status = StageControlModule.is_prusa_on(self.ser)
        if printer_status:
            print("Connected to the PRUSA " + self.ser.name)
        else:
            self.ser.close()
            print("Couldn't connect to the PRUSA")

    def is_prusa_on(ser):
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

    def get_current_prusa_position(self, dim = 'XYZ'):
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write(b'M114\n')
        line = self.wait_for_prusa_process_complete(process_name = "get_current_position", critiria='X')
        print(line)
        if dim == 'XYZ':
            X = line.split(' ')[0].split(':')[1]
            Y = line.split(' ')[1].split(':')[1]
            Z = line.split(' ')[2].split(':')[1]
            return X, Y, Z
        elif dim == 'X':
            X = line.split(' ')[0].split(':')[1]
            return X
        elif dim == 'Y':
            Y = line.split(' ')[1].split(':')[1]
            return Y
        elif dim == 'Z':
            Z = line.split(' ')[2].split(':')[1]
            return Z

    # def register_first_smaple(self):
    #     self.position_first_smaple = self.get_current_prusa_position('XYZ')
    #     self.register_first_smaple_button.configure(text="First Sample Registered")
    #     self.register_first_smaple_button.configure(state=DISABLED)
    #     self.remove_first_sample_button.configure(state=NORMAL)
    #     self.go_to_first_sample_button.configure(state=NORMAL)
    #     self.test_sample_spot_button.configure(state=NORMAL)
    #     print("First sample registered")

    def register_lowest_point(self):
        current_position = self.get_current_prusa_position('XYZ')
        self.lowest_point = current_position[2]
        self.register_lowest_point_button.configure(text="Lowest Z Point recorded")
        self.register_lowest_point_button.configure(state=DISABLED)
        self.remove_lowest_point_button.configure(state=NORMAL)
        self.go_to_lowest_point_button.configure(state=NORMAL)
        print("Lowest Z registered")

    # def remove_first_sample_registration(self):
    #     self.position_first_smaple = None
    #     print("Registration of the first sample removed")
    #     self.remove_first_sample_button.configure(state=DISABLED)
    #     self.register_first_smaple_button.configure(text="Record XY as first sample")
    #     self.register_first_smaple_button.configure(state=NORMAL)
    #     self.register_first_smaple_button.configure(bootstyle="info")
    #     self.go_to_first_sample_button.configure(state=DISABLED)
    #     self.test_sample_spot_button.configure(state=DISABLED)

    def remove_lowest_point_registration(self):
        self.lowest_point = None
        print("Registration of the lowest Z removed")
        self.remove_lowest_point_button.configure(state=DISABLED)
        self.register_lowest_point_button.configure(text ='Record Z as the lowest Z', state = NORMAL, bootstyle = 'info')
        self.go_to_lowest_point_button.configure(state=DISABLED)

    def handle_switch_view(self, view):
        if view == "TO_OBJECTIVE":
            # send g code to move the stage to the right
            print("Switching to objective view")
            self.prusa_go_by_xyz(x=self.calibration_from_widefield_to_objective_x, y=self.calibration_from_widefield_to_objective_y)

        elif view == "TO_WIDE":
            # send g code to move the stage to the left
            print("Switching to wide view")
            self.prusa_go_by_xyz(x=-self.calibration_from_widefield_to_objective_x, y=-self.calibration_from_widefield_to_objective_y)

        else:
            print("Error Happened", view)

    # def roll_over_coarse_fine_z_position(self, coarse_mm, fine_um):
    #             # Check if fine_nm exceeds 1000 nm and adjust the coarse stage accordingly
    #     total_nm = coarse_mm * 1000 + fine_um  # Convert coarse to nm and add fine adjustment
    #     adjusted_coarse_um = int(total_nm // 1000)  # New coarse value in µm
    #     remaining_nm = total_nm % 1000  # Fine position within the µm range
    #     return f"{adjusted_coarse_um} mm + {remaining_nm:.0f} um"

    def handle_update_nanodrive_position(self, nanodrive_position):
        self.current_nanodrive_position = nanodrive_position * (-1)
        self.current_nanodrive_z_entry.config(text=nanodrive_position)

    def handle_test_sampling_points(self, view_to_inspect_in, relative_distance_for_sampling_points):
        # Display a confirmation dialog
        response = Messagebox.show_question(
            title="Confirm Operation",
            message="Is the objective at a safe height?",
            alert=True,
            buttons=["Yes", "No"],
            bootstyle="danger"
        )
        if response == "Yes":  # If the user clicks "Yes"
            print("Operation started!")
            self.run_in_thread(self._handle_test_sampling_points, view_to_inspect_in, relative_distance_for_sampling_points)
        else:  # If the user clicks "No"
            print("Operation canceled.")


    def _handle_test_sampling_points(self, view_to_inspect_in, relative_distance_for_sampling_points):
        # convert the unit of the sampling points position from um to mm
        if view_to_inspect_in == "OBJECTIVE":
            if self.capture_frame_center_objective_x and self.capture_frame_center_objective_y:
                # printer's y axis is flipped
                filped_sampling_position_y = relative_distance_for_sampling_points[1] * -1

                sampling_position_x = relative_distance_for_sampling_points[0] / 1000 + self.capture_frame_center_objective_x
                sampling_position_y = filped_sampling_position_y / 1000 + self.capture_frame_center_objective_y
        elif view_to_inspect_in == "WIDEFIELD":
            if self.capture_frame_center_widefield_x and self.capture_frame_center_widefield_y:
                print('when handling test_sampling_points, the center used for x: ', self.capture_frame_center_widefield_x)
                print('when handling test_sampling_points, the center used for y: ', self.capture_frame_center_widefield_y)

                # printer's y axis is flipped
                filped_sampling_position_y = relative_distance_for_sampling_points[1] * -1

                sampling_position_x = relative_distance_for_sampling_points[0] / 1000 + self.capture_frame_center_widefield_x
                sampling_position_y = filped_sampling_position_y / 1000 + self.capture_frame_center_widefield_y

                print('the sampling_position_x: ', sampling_position_x)
                print('the sampling_position_y: ', sampling_position_y)
        else:
            raise ValueError("The center of the frame is not captured yet")

        # round up the sampling position to 2 decimal places
        sampling_position_x = np.round(sampling_position_x, 2)
        sampling_position_y = np.round(sampling_position_y, 2)

        # move to the test sample spot one by one
        for i in range(len(sampling_position_x)):
            print(f'Moving to the test sample spot {sampling_position_x[i]}, {sampling_position_y[i]}')
            self.prusa_go_to_xyz(x=sampling_position_x[i], y=sampling_position_y[i])
            time.sleep(1)

    def handle_updated_captured_frame_center(self, field_of_view):
        print("Captured the center of the frame")
        if field_of_view == 'WIDEFIELD':
            self.capture_frame_center_widefield_x = float(self.get_current_prusa_position('XYZ')[0])
            self.capture_frame_center_widefield_y = float(self.get_current_prusa_position('XYZ')[1])

            print('captured center position is in type: ',type(self.capture_frame_center_widefield_x))

            print(f"Captured the center of the frame in the physical space: {self.capture_frame_center_widefield_x}, {self.capture_frame_center_widefield_y}")

            self.capture_frame_center_objective_x = self.capture_frame_center_widefield_x + self.calibration_from_widefield_to_objective_x
            self.capture_frame_center_objective_y = self.capture_frame_center_widefield_y + self.calibration_from_widefield_to_objective_y
        elif field_of_view == 'OBJECTIVE':
            self.capture_frame_center_objective_x = float(self.get_current_prusa_position('XYZ')[0])
            self.capture_frame_center_objective_y = float(self.get_current_prusa_position('XYZ')[1])

            self.capture_frame_center_widefield_x = self.capture_frame_center_objective_x - self.calibration_from_widefield_to_objective_x
            self.capture_frame_center_widefield_y = self.capture_frame_center_objective_y - self.calibration_from_widefield_to_objective_y


    def update_prusa_position(self):
        current_prusa_position = self.get_current_prusa_position('XYZ')

        if self.prusa_x_referenced:
            current_prusa_position_x = current_prusa_position[0]
            self.current_x_entry.config(text=current_prusa_position_x)
        
        if self.prusa_y_referenced:
            current_prusa_position_y = current_prusa_position[1]
            self.current_y_entry.config(text=current_prusa_position_y)

        if self.prusa_z_referenced:
            current_prusa_position_z = current_prusa_position[2]
            self.current_z_entry.config(text=current_prusa_position_z)


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = StageControlModule(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()