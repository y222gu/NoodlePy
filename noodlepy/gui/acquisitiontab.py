import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import StringVar
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from noodlepy.gui.liveviewmodule import LiveViewModule
import os
import time
import serial
import cv2
from matplotlib import pyplot as plt
import serial.tools.list_ports
from noodlepy.gui.sampledetectionmodule import SampleDetectionModule
class AcquisitionGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
                
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew")

        stage_control_frame = StageControlerModule(main_frame)
        stage_control_frame.grid(row = 0, column= 0, sticky="nsew", padx=5, pady=5)

        autofocus_frame = AutoFocusModule(main_frame)
        autofocus_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        live_view_frame = LiveViewModule(main_frame)
        live_view_frame.grid(row=0, column=2, columnspan=3, sticky="nsew", padx=5, pady=5)

        # sample_detection_frame = SampleDetectionModule(main_frame)
        # sample_detection_frame.grid(row=1, column=2, columnspan=2, sticky="nsew", padx=5, pady=5)

class StageControlerModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.img_small_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","small_step.png"))
        self.img_medium_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","medium_step.png"))
        self.img_large_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","large_step.png"))
        self.img_small_step = self.img_small_step.resize((20, 20))
        self.img_medium_step = self.img_medium_step.resize((20, 20))
        self.img_large_step = self.img_large_step.resize((20, 20))
        self.ser = None
        self.create_widgets()

    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Stage Control', padding=5)
        main_frame.grid(row=0, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        self.connect_button = ttk.Button(top_frame, text="Connect", command=self.connect_device, bootstyle="secondary")
        self.connect_button.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.connect_button.config(width=10)

        # Reference buttons frame
        reference_frame = ttk.Labelframe(top_frame, text="Reference", padding=5)
        reference_frame.grid(row=0, column=1, columnspan=4, padx=5, pady=5)

        ttk.Button(reference_frame, text="Ref X", command=lambda: self.ref("X"), bootstyle="secondary", width=6).grid(row=0, column=0, padx=5)
        ttk.Button(reference_frame, text="Ref Y", command=lambda: self.ref("Y"), bootstyle="secondary", width=6).grid(row=0, column=1, padx=5)
        ttk.Button(reference_frame, text="Ref Z", command=lambda: self.ref("Z"), bootstyle="secondary", width=6).grid(row=0, column=2, padx=5)
        ttk.Button(reference_frame, text="Ref All", command=lambda: self.ref("ALL"), bootstyle="warning", width=6).grid(row=0, column=3, padx=5)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text="Move", padding=5)
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

        self.go_button = ttk.Button(movement_frame, text="Go", command=lambda: self.send_gcode("GO"),bootstyle="success")
        self.go_button.grid(row=2, column=4, padx=10, pady=5)
        self.go_button.config(width=6)

        # Speed slider
        self.slider_value = StringVar()
        self.speed_slider = ttk.Scale(
            movement_frame,
            from_=1000,
            to=3000,
            orient=HORIZONTAL,
            bootstyle="success",
            variable=self.slider_value
        )
        self.speed_slider.grid(row=3, column=2, columnspan=2, pady=20)
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

        # X, Y axis Direction buttons
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

        # label for step size
        ttk.Label(direction_frame, text="10").grid(row=3, column=1, pady=3)
        ttk.Label(direction_frame, text="1").grid(row=3, column=2, pady=3)
        ttk.Label(direction_frame, text="0.1").grid(row=3, column=3,  pady=3)

        # Add an empty label for spacing between x, y and z buttons
        ttk.Label(direction_frame, text="").grid(row=2, column=9,columnspan=2, padx=50)
        
        # Z-axis control labels
        ttk.Label(direction_frame, text="Z (up)").grid(row=0, column=10, pady=3)
        ttk.Label(direction_frame, text="Z (down)").grid(row=8, column=10, pady=3)

        # Z-axis control buttons
        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.send_gcode("UP SMALL"), bootstyle="light").grid(row=3, column=10, columnspan=2)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.send_gcode("UP MEDIUM"), bootstyle="secondary").grid(row=2, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.send_gcode("UP LARGE"), bootstyle="dark").grid(row=1, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.send_gcode("DOWN SMALL"), bootstyle="light").grid(row=5, column=10, columnspan=2)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.send_gcode("DOWN MEDIUM"), bootstyle="secondary").grid(row=6, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.send_gcode("DOWN LARGE"), bootstyle="dark").grid(row=7, column=10, columnspan=2, pady=3)

        # Frame for register
        register_frame = ttk.Labelframe(main_frame, text="Register", padding=5)
        register_frame.grid(row=2, column=0, columnspan=5, sticky="ew", padx=5, pady=5)
        self.register_first_smaple_button = ttk.Button(register_frame, text="Register Current Position as First Sample", command = self.register_first_smaple, bootstyle ="success", width=36)
        self.register_first_smaple_button.grid(row=0, column=1, padx=5, pady=5)
        self.register_lowest_point_button = ttk.Button(register_frame, text="Register Current Z as the Lowest Point", command=self.register_lowest_point, bootstyle = "success", width=36)
        self.register_lowest_point_button.grid(row=1, column=1, padx=5, pady=5)

        # Remove button
        self.remove_first_sample_button = ttk.Button(register_frame, text="x", command= self.remove_first_sample_registration,bootstyle="danger", state=DISABLED)
        self.remove_first_sample_button.grid(row=0, column=2, padx=5, pady=5)
        self.remove_lowest_point_button = ttk.Button(register_frame, text="x", command= self.remove_lowest_point_registration,bootstyle="danger", state=DISABLED)
        self.remove_lowest_point_button.grid(row=1, column=2, padx=5, pady=5)

        self.go_to_first_sample_button = ttk.Button(register_frame, text="Go", command=self.go_to_first_sample, bootstyle="success", width=6)
        self.go_to_first_sample_button.grid(row=0, column=3, padx=5, pady=5)
        self.go_to_lowest_point_button = ttk.Button(register_frame, text="Go", command=self.go_to_lowest_point, bootstyle="success", width=6)
        self.go_to_lowest_point_button.grid(row=1, column=3, padx=5, pady=5)

    def find_printer_com_ports():
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.description == "Original Prusa i3 MK3 (COM3)":
                return port.name

    def ref(self, option):
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

    def wait_for_process_complete(self, process_name, critiria):
        """Waits for a signal from the printer that homing is complete."""
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            if critiria in line:
                print(f"Received during {process_name}: {line}")
                return line
            elif line:
                print(f"Received during {process_name}: {line}")

    def send_gcode(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "UP SMALL":
            gecode = f"G0 Z0.1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gecode))
            print("Moving up small")
        elif option == "UP MEDIUM":
            gcode = f"G0 Z1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
            print("Moving up medium")
        elif option == "UP LARGE":
            gcode = f"G0 Z10 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
            print("Moving up large")

        elif option == "DOWN SMALL":
            gcode = f"G1 Z-0.1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "DOWN MEDIUM":
            gcode = f"G1 Z-1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "DOWN LARGE":
            gcode = f"G1 Z-10 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))

        elif option == "LEFT SMALL":
            gcode = f"G1 X-0.1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "LEFT MEDIUM":
            gcode = f"G1 X-1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "LEFT LARGE":
            gcode = f"G1 X-10 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))

        elif option == "RIGHT SMALL":
            gcode = f"G1 X0.1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "RIGHT MEDIUM":
            gcode = f"G1 X1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "RIGHT LARGE":
            gcode = f"G1 X10 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))

        elif option == "BACK SMALL":
            gcode = f"G1 Y0.1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "BACK MEDIUM":
            gcode = f"G1 Y1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "BACK LARGE":
            gcode = f"G1 Y10 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))

        elif option == "FRONT SMALL":
            gcode = f"G1 Y-0.1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "FRONT MEDIUM":
            gcode = f"G1 Y-1 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "FRONT LARGE":
            gcode = f"G1 Y-10 F{self.speed}\r\n"
            self.ser.write(str.encode("G91\r\n"))
            self.ser.write(str.encode(gcode))
        elif option == "GO":
            p2_x = self.p2_x_entry.get()
            p2_y = self.p2_y_entry.get()
            p2_z = self.p2_z_entry.get()

            if p2_x and p2_y and p2_z:
                gcode = f"G0 X{p2_x} Y{p2_y} Z{p2_z} F{self.speed}\r\n"
                self.ser.write(str.encode("G90\r\n"))
                self.ser.write(str.encode(gcode))
            else:
                print("Please enter all the coordinates")
                return
        else:
            print("Invalid option")
        
        self.get_current_position('XYZ')
        return

    def update_speed(self):
        self.speed = self.speed_slider.get()
        print(f"Speed of stage is updated to: {self.speed} mm/s")

    def connect_device(self):
        self.port = StageControlerModule.find_printer_com_ports()
        self.ser = serial.Serial(self.port, 115200)
        
        printer_status = StageControlerModule.is_printer_on(self.ser)
        # Check if the device is online
        if printer_status:
            self.connect_button.configure(text="Disconnect", command=self.disconnect_device)
            self.connect_button.configure(bootstyle="success")
            print("Connected to the printer " + self.ser.name)
        else:
            self.ser.close()
            print("Couldn't connect to the printer")


    def disconnect_device(self):
        self.connect_button.configure(text="Connect", command=self.connect_device)
        self.connect_button.configure(bootstyle="secondary")
        self.port = None
        self.ser = None
        print("Disconnected to the printer")

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
        # record the current position as the first sample
        self.position_first_smaple = self.get_current_position('XYZ')
        self.register_first_smaple_button.configure(text="First Sample Registered")
        self.register_first_smaple_button.configure(bootstyle="success")
        self.register_first_smaple_button.configure(state=DISABLED)
        self.remove_first_sample_button.configure(state=NORMAL)
        print("First sample registered")

    def register_lowest_point(self):
        current_position = self.get_current_position('XYZ')
        self.lowest_point = current_position[2]
        self.register_lowest_point_button.configure(text="Lowest Z Point registered")
        self.register_lowest_point_button.configure(bootstyle="success")
        self.register_lowest_point_button.configure(state=DISABLED)
        self.remove_lowest_point_button.configure(state=NORMAL)
        print("Lowest Z registered")

    def remove_first_sample_registration(self):
        self.position_first_smaple = None
        print("Registration of the first sample removed")
        self.remove_first_sample_button.configure(state=DISABLED)
        self.register_first_smaple_button.configure(text="Register Current Position as First Sample")
        self.register_first_smaple_button.configure(state=NORMAL)
        self.register_first_smaple_button.configure(bootstyle="success")

    def remove_lowest_point_registration(self):
        self.lowest_point = None
        print("Registration of the lowest Z removed")
        self.remove_lowest_point_button.configure(state=DISABLED)
        self.register_lowest_point_button.configure(text ='Register Current Z as the Lowest Point', state = NORMAL, bootstyle = 'success')

    def update_label(self, value):
        self.slider_value.set(f"{float(value):.2f} mm/s")

    def go_to_first_sample(self):
        if self.position_first_smaple:
            # rise to a safe height
            self.ser.write(str.encode("G90\r\n"))
            self.ser.write(str.encode(f"G0 Z30 \r\n"))
            # move to the position of the first sample
            self.ser.write(str.encode("G90\r\n"))
            gcode = f"G0 X{self.position_first_smaple[0]} Y{self.position_first_smaple[1]} F{self.speed}\r\n"
            self.ser.write(str.encode(gcode))
            self.ser.write(str.encode("G90\r\n"))
            gcode = f"G0 Z{self.position_first_smaple[2]}\r\n"
            self.ser.write(str.encode(gcode))

            self.get_current_position('XYZ')
        else:
            print("Please register the first sample first")

    def go_to_lowest_point(self):
        if self.lowest_point:
            self.ser.write(str.encode("G90\r\n"))
            self.ser.write(str.encode(f"G0 Z{self.lowest_point} \r\n"))
            self.get_current_position('XYZ')
        else:
            print("Please register the lowest point first")

class AutoFocusModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Auto Focus', padding=5)
        main_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Create the figure and axes
        fig, axes = plt.subplots(2, 1, figsize=(2, 2))
        self.ax1, self.ax2= axes.flatten()

        self.ax1.set_xlabel('Z-axis position',fontsize=5)
        self.ax1.set_ylabel('Entropy', fontsize=5)
        self.ax1.set_title('Entropy minimization', fontsize=5)
        self.ax1.tick_params(axis='both', which='major', labelsize=4)
        self.ax1.tick_params(axis='both', which='minor', labelsize=4)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.grid(True)

        self.ax2.set_xlabel('Z-axis position', fontsize=5)
        self.ax2.set_ylabel('Intensity', fontsize=5)
        self.ax2.set_title('Focus', fontsize=5)
        self.ax2.tick_params(axis='both', which='major', labelsize=4)
        self.ax2.tick_params(axis='both', which='minor', labelsize=4)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.grid(True)

        plt.tight_layout(pad=0.5)

        # Create a canvas widget for the figure
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Create the entries for the autofocus parameters
        num_steps_label = ttk.Label(main_frame, text="Number of Steps")
        num_steps_label.grid(row=2, column=0,pady=5)
        num_steps_entry = ttk.Entry(main_frame, width=5)
        num_steps_entry.grid(row=2, column=1)

        range_label = ttk.Label(main_frame, text="Z-axis Range [um]")
        range_label.grid(row=3, column=0,pady=5)
        range_entry = ttk.Entry(main_frame, width=5)
        range_entry.grid(row=3, column=1)

        exposure_time_label = ttk.Label(main_frame, text="Exposure Time [ms]")
        exposure_time_label.grid(row=4, column=0,pady=5)
        exposure_time_entry = ttk.Entry(main_frame, width=5)
        exposure_time_entry.grid(row=4, column=1)

        num_rep_label = ttk.Label(main_frame, text="Number of Reps")
        num_rep_label.grid(row=5, column=0, pady=5)
        num_rep_entry = ttk.Entry(main_frame, width=5)
        num_rep_entry.grid(row=5, column=1)

        # Create the button to run autofocus
        button = ttk.Button(main_frame, text="Coarse Focus", command=self.autofocus, bootstyle="success", width=15)
        button.grid(row=2, column=2, padx=5, pady=5)

        # Create the button to stop autofocus
        button = ttk.Button(main_frame, text="Fine Focus", command=self.autofocus, bootstyle="success", width=15)
        button.grid(row=3, column=2, padx=5, pady=5)

        # Create the button to save the autofocus results
        button = ttk.Button(main_frame, text="Run Both", command=self.autofocus, bootstyle="warning", width=15)
        button.grid(row=4, column=2, columnspan=3, padx=5, pady=5)

        button = ttk.Button(main_frame, text="Stop", command=self.stop_autofocus, bootstyle="danger", width=15)
        button.grid(row=5, column=2, columnspan=3, padx=5, pady=5)

    def autofocus(self):
        for i in range(10):
            # Implement the autofocus algorithm here
            print(f"Running autofocus iteration {i+1}")
            # For example, you might use serial communication:
            # ser.write((f"RUN AUTOFOCUS {i}\n").encode())
            # Update the plots
            entropy = 1 / (i + 1)
            wavelength = [400, 500, 600, 700]
            intensity = [0.1, 0.2, 0.3, 0.4]

            self.ax1.plot(i, entropy, 'ro')
            self.ax2.plot(wavelength, intensity, 'bo')
            self.canvas.draw()

    def stop_autofocus(self):
        # Implement the code to stop autofocus here
        print("Stopping autofocus")
        # For example, you might use serial communication:
        # ser.write("STOP AUTOFOCUS\n".encode())
    
    def save_results(self):
        # Implement the code to save the autofocus results here
        print("Saving autofocus results")
        # For example, you might save the plots as images:
        # fig.savefig("autofocus_results.png")


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('superhero')
    app = AcquisitionGUI(root)
    app.pack()
    root.mainloop()
