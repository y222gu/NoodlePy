import os
import time
import serial
import serial.tools.list_ports
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk
from tkinter import StringVar, messagebox
import tkinter as tk
import numpy as np
from threading import Thread, Lock
from noodlepy.gui.publisher_subscriber import Subscriber, Publisher
from noodlepy.gui.calibrationmodule import CalibrationModule

class StageControlModule(ttk.Frame, Publisher, Subscriber):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        Publisher.__init__(self, ['move_nanodrive_by', 
                                  'move_nanodrive_to', 
                                  'calculate_focus_score', 
                                  'get_nanodrive_position', 
                                  'activate_camera_autofocus_button', 
                                  'activate_switch_view_button', 
                                  'activate_protocol_module_state',
                                  'get_nanodrive_min_max',
                                  'task_completed',
                                  'abort_aquisition'])
        Subscriber.__init__(self)
        self.name = 'StageControlModule_obserableobserver'
        self.parent = parent

        self.img_small_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","small_step.png")).resize((20, 20))
        self.img_medium_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","medium_step.png")).resize((20, 20))
        self.img_large_step = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","large_step.png")).resize((20, 20))
        self.circle_black = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","circle-black.png")).resize((20, 20))
        self.nanodrive_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","nanodrive_arrow_1.png")).resize((20, 20))
        self.home_icon = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","home_icon.png")).resize((40, 40))

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
        self.img_home_icon = ImageTk.PhotoImage(self.home_icon)
       
        self.safety_height = 20 # calibrated on 1/4/2025
        self.home_position = [118.58, 181.94, self.safety_height]
        self.calibration_from_widefield_to_objective_x = -61.12 # calibrated on 2/06/2025
        self.calibration_from_widefield_to_objective_y = -3.72 # calibrated on 2/06/2025
        self.calibration_from_widefield_to_objective_z = - 1.44 #  in mm calibrated on 1/4/2025
        self.initial_nanodrive_position = 50
        self.small_step_size_xy_mm = 0.06 #firmware seems to limit the smallest step size to 0.06 (60 um)
        self.medium_step_size_xy_mm = 0.5
        self.large_step_size_xy_mm = 4.5
        self.small_step_size_z_mm = 0.02
        self.medium_step_size_z_mm = 0.1
        self.large_step_size_z_mm = 1
        self.nanodrive_step_size_um = 1

        self.prusa_fine_focus_step_number = 20 # in objective view
        self.prusa_rough_focus_step_number = 20 # in widefield view
        self.nanodrive_movement_stabilization_time = 0.5
        self.prusa_movement_stabilization_time = 1
        self.prusa_rough_fine_focus_overlap_step_number = 4 # the actual number of steps to overlap between rough and fine focus is double this number
        self.focus_upper_limit = None
        self.focus_lower_limit = None
        self.focus_score_at_current_z_position = None
        self.current_nanodrive_position = None #um
        self.prusa_x_referenced = False
        self.prusa_y_referenced = False
        self.prusa_z_referenced = False
        self.capture_frame_center_widefield_x= None
        self.capture_frame_center_widefield_y = None
        self.capture_frame_center_objective_x= None
        self.capture_frame_center_objective_y = None
        self.ser = None
        self.lower_limit_p1 = None
        self.upper_limit_p1 = None
        self.lower_limit_p2 = None
        self.upper_limit_p2 = None
        self.lower_limit_p3 = None
        self.upper_limit_p3 = None
        
        # parameters for the sample drop panel
        self.interval_between_drop_x = 4.5 # mm
        self.interval_between_drop_y = 4.5 # mm 
        self.offset_home_to_p1_x = 8.28 # mm
        self.offset_home_to_p1_y = -8.64 # mm
        self.sample_drop_panel_circle_radius = 15 # pixels
        self.sample_drop_panel_spacing = 40 # pixels
        self.sample_drop_panel_grid_pattern = [5, 5, 5, 5]  # Column-wise circle counts for symmetry
        self.sample_drop_panel_max_rows = max(self.sample_drop_panel_grid_pattern)
        self.sample_drop_panel_total_width = len(self.sample_drop_panel_grid_pattern) * self.sample_drop_panel_spacing
        self.sample_drop_panel_total_height = self.sample_drop_panel_max_rows * self.sample_drop_panel_spacing
        self.sample_drop_panel_current_sample_row_column = None
        self.sample_drop_panel_circle_pursa_coordinates = {}  # Store physical coordinates of circles
        self.sample_drop_panel_calculate_circle_pursa_coordinates()
        self.create_widgets()
        self.connect_prusa_device()

        # parameters for the protocol module
        self.selected_circle_positions = []

    def create_widgets(self):

        main_frame = ttk.Labelframe(self, text='Stage Control', padding=5)
        main_frame.grid(row=0, column=0, columnspan=7, sticky="nsew", padx=5, pady=5)

        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=7, sticky="nsew")

        # Reference buttons frame
        reference_frame = ttk.Labelframe(top_frame, text="Reference", padding=5)
        reference_frame.grid(row=0, column=0,  columnspan=4, rowspan=2, padx=5, sticky='nsew', pady=5)
        ttk.Button(reference_frame, text="Ref X", command=lambda: self.ref("X"), bootstyle="info-outline", width=8).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref Y", command=lambda: self.ref("Y"), bootstyle="info-outline", width=8).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref Z", command=lambda: self.ref("Z"), bootstyle="info-outline", width=8).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref NP", command=lambda: self.ref("nanodrive"), bootstyle="info-outline", width=12).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(reference_frame, text="Ref All", command=lambda: self.ref("ALL"), bootstyle="info").grid(row=1, column=0, columnspan=4, sticky='nesw',padx=5, pady=5)

        # Home frame
        self.home_button = ttk.Button(top_frame, image=self.img_home_icon, command=self.prusa_go_to_home_position, bootstyle="info", state=DISABLED)
        self.home_button.grid(row=0, column=4, columnspan=2, sticky='ew',padx=5, pady=5)

        self.calibrate_home_position_button = ttk.Button(top_frame, text="Calibrate Home Position", command = self.update_home_position, bootstyle ="info-outline", state=DISABLED)
        self.calibrate_home_position_button.grid(row=1, column=4, columnspan=2, padx=5, pady=5)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text='Move' , padding=5)
        movement_frame.grid(row=1, column=0, columnspan=6, sticky="ew", padx=5, pady=5)
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
        self.target_nanodrive_entry = ttk.Spinbox(movement_frame, from_=0, to=100, width=8)
        self.target_x_entry.grid(row=2, column=1, padx=5, pady=5)
        self.target_y_entry.grid(row=2, column=2, padx=5, pady=5)
        self.target_z_entry.grid(row=2, column=3, padx=5, pady=5)
        self.target_nanodrive_entry.grid(row=2, column=4, padx=5, pady=5)
        self.target_x_entry.insert(0, self.home_position[0])
        self.target_y_entry.insert(0, self.home_position[1])
        self.target_z_entry.insert(0, self.home_position[2])
        self.target_nanodrive_entry.insert(0, "0")
        self.go_button = ttk.Button(movement_frame, text="Go", command=self.move_both_prusa_nanodrive, bootstyle="info", state=DISABLED)
        self.go_button.grid(row=2, column=5, padx=10, pady=5)
        self.go_button.config(width=6)
        self.update_position_button = ttk.Button(movement_frame, text="Update", command=self.update_prusa_position, bootstyle="info", state=DISABLED)
        self.update_position_button.grid(row=1, column=5, padx=10, pady=5)
        self.update_position_button.config(width=6)

        # Speed slider
        self.slider_value = StringVar()
        self.slider_value.set(3000)
        self.speed_slider = ttk.Scale(
            movement_frame,
            from_=10,
            to=3000,
            orient=HORIZONTAL,
            bootstyle="info",
            variable=self.slider_value
        )
        self.speed_slider.grid(row=3, column=2, columnspan=2, pady=15)
        self.speed_slider.config(length=200)
        self.speed = self.speed_slider.get()
        ttk.Label(movement_frame, text="Speed").grid(row=3, column=0, padx=5, pady=5)
        ttk.Label(movement_frame, text="10 mm/s").grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(movement_frame, text="3000 mm/s").grid(row=3, column=4, padx=5, pady=5)
        self.speed_slider.bind("<ButtonRelease-1>", lambda e: self.update_prusa_speed())

        # Frame for direction buttons
        direction_frame = ttk.Frame(movement_frame, padding=3)
        direction_frame.grid(row=5, column=0, columnspan=10, pady=3, padx=3)
        ttk.Label(direction_frame, text="Y").grid(row=0, column=4, pady=3)
        ttk.Label(direction_frame, text="X").grid(row=4, column=0, padx=3)
        ttk.Label(direction_frame, text="Z (up)").grid(row=0, column=10, pady=3)
        ttk.Label(direction_frame, text="Z (down)").grid(row=8, column=10, pady=3)
        ttk.Label(direction_frame, text="ND Z (up)").grid(row=0, column=11, pady=3)
        ttk.Label(direction_frame, text="ND Z (down)").grid(row=8, column=11, pady=3)
        ttk.Label(direction_frame, text="").grid(row=2, column=9,columnspan=2, padx=50)
        ttk.Label(direction_frame, text=f"{self.large_step_size_xy_mm} mm").grid(row=1, column=3, pady=3)
        ttk.Label(direction_frame, text=f"{self.medium_step_size_xy_mm} mm").grid(row=2, column=3, pady=3)
        ttk.Label(direction_frame, text=f"{self.small_step_size_xy_mm} mm").grid(row=3, column=3,  pady=3)

        ttk.Label(direction_frame, text=f"{self.large_step_size_z_mm} mm").grid(row=1, column=9, pady=3)
        ttk.Label(direction_frame, text=f"{self.medium_step_size_z_mm} mm").grid(row=2, column=9, pady=3)
        ttk.Label(direction_frame, text=f"{self.small_step_size_z_mm} mm").grid(row=3, column=9, pady=3)
        ttk.Label(direction_frame, text=f"{self.nanodrive_step_size_um} um").grid(row=4, column=11, pady=3)


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

        # Frame for drop position
        sample_drop_position_frame = ttk.Labelframe(main_frame, text="Sample Drop Position", padding=5)
        sample_drop_position_frame.grid(row=3, column=0, rowspan=5, sticky='nsew', padx=5, pady=5)
        self.sample_drop_panel_canvas = tk.Canvas(sample_drop_position_frame, width=self.sample_drop_panel_total_width, height=self.sample_drop_panel_total_height, state=DISABLED)
        self.sample_drop_panel_canvas.grid(row=0, column=0, rowspan=5, padx=5, pady=5)
        self.sample_drop_panel_tooltip = tk.Label(sample_drop_position_frame, text="", relief=tk.SOLID, bd=1, state=DISABLED)
        self.sample_drop_panel_create_circle_grid()

        # registering the focus range
        register_focus_range_frame = ttk.Labelframe(main_frame, text="Register Focus Range", padding=5)
        register_focus_range_frame.grid(row=3, column=1, rowspan=5, padx=5, pady=5)

        self.xy_position_label = ttk.Label(register_focus_range_frame, text="(X, Y) [mm]")
        self.xy_position_label.grid(row=0, column=1, padx=5, pady=5)
        self.focus_lower_limit_label = ttk.Label(register_focus_range_frame, text="Z Range[mm]")
        self.focus_lower_limit_label.grid(row=0, column=2, padx=5, pady=5)

        self.point_1_button = ttk.Button(register_focus_range_frame, text="Point 1", command=lambda: self.register_focus_range("P1"), bootstyle="info_outline", state=DISABLED)
        self.point_1_button.grid(row=1, column=0, padx=5, pady=5)
        self.point_1_position_label = ttk.Label(register_focus_range_frame, text="N/A", width=5)
        self.point_1_position_label.grid(row=1, column=1, padx=5, pady=5)
        self.point_1_focus_range_label = ttk.Label(register_focus_range_frame, text="N/A", width=5)
        self.point_1_focus_range_label.grid(row=1, column=2, padx=5, pady=5)

        self.point_2_button = ttk.Button(register_focus_range_frame, text="Point 2", command=lambda: self.register_focus_range("P2"), bootstyle="info_outline", state=DISABLED)
        self.point_2_button.grid(row=2, column=0, padx=5, pady=5)
        self.point_2_position_label = ttk.Label(register_focus_range_frame, text="N/A", width=5)
        self.point_2_position_label.grid(row=2, column=1, padx=5, pady=5)
        self.point_2_focus_range_label = ttk.Label(register_focus_range_frame, text="N/A", width=5)
        self.point_2_focus_range_label.grid(row=2, column=2, padx=5, pady=5)

        self.point_3_button = ttk.Button(register_focus_range_frame, text="Point 3",  command=lambda: self.register_focus_range("P3"), bootstyle="info_outline", state=DISABLED)
        self.point_3_button.grid(row=3, column=0, padx=5, pady=5)
        self.point_3_position_label = ttk.Label(register_focus_range_frame, text="N/A", width=5)
        self.point_3_position_label.grid(row=3, column=1, padx=5, pady=5)
        self.point_3_focus_range_label = ttk.Label(register_focus_range_frame, text="N/A", width=5)
        self.point_3_focus_range_label.grid(row=3, column=2, padx=5, pady=5)

        self.interpolate_button = ttk.Button(register_focus_range_frame, text="Interpolate", command=self.interpolate_prusa_focus_range, bootstyle="info_outline", state=DISABLED)
        self.interpolate_button.grid(row=4, column=2, columnspan=2, sticky='nesw', padx=5, pady=5)
    
    def run_in_thread(self, func, *args):
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()

    def run_in_thread_with_callback(self, func, callback, *args):
        def wrapper():
            # Execute the target function and store the result
            result = func(*args)
            # Pass the result to the callback function
            callback(result)

        # Run the wrapper function in a separate thread
        thread = Thread(target=wrapper, daemon=True)
        thread.start()

    def prusa_go_to_home_position(self):
        self.prusa_go_to_xyz(z=self.safety_height) # raise to a safe height first
        self.prusa_go_to_xyz(x=float(self.home_position[0]), y=float(self.home_position[1]))
        # self.prusa_go_to_xyz(z=float(self.home_position[2]))

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

    def get_nanodrive_position(self):
        self.current_nanodrive_position = None
        self.dispatch('get_nanodrive_position')
        # wait until nanodrive position is updated
        while self.current_nanodrive_position is None:
            time.sleep(0.1)
        return self.current_nanodrive_position

    def move_prusa(self, option):
        if self.ser is None:
            print("Please connect to the printer first")
            return
        if option == "UP SMALL":
            self.prusa_go_by_xyz(z=self.small_step_size_z_mm)
        elif option == "UP MEDIUM":
            self.prusa_go_by_xyz(z=self.medium_step_size_z_mm)
        elif option == "UP LARGE":
            self.prusa_go_by_xyz(z=self.large_step_size_z_mm)

        elif option == "DOWN SMALL":
            self.prusa_go_by_xyz(z=-self.small_step_size_z_mm)
        elif option == "DOWN MEDIUM":
            self.prusa_go_by_xyz(z=-self.medium_step_size_z_mm)
        elif option == "DOWN LARGE":
            self.prusa_go_by_xyz(z=-self.large_step_size_z_mm)

        elif option == "LEFT SMALL":
            self.prusa_go_by_xyz(x=-self.small_step_size_xy_mm)
        elif option == "LEFT MEDIUM":
            self.prusa_go_by_xyz(x=-self.medium_step_size_xy_mm)
        elif option == "LEFT LARGE":
            self.prusa_go_by_xyz(x=-self.large_step_size_xy_mm)

        elif option == "RIGHT SMALL":
            self.prusa_go_by_xyz(x=self.small_step_size_xy_mm)
        elif option == "RIGHT MEDIUM":
            self.prusa_go_by_xyz(x=self.medium_step_size_xy_mm)
        elif option == "RIGHT LARGE":
            self.prusa_go_by_xyz(x=self.large_step_size_xy_mm)

        elif option == "BACK SMALL":
            self.prusa_go_by_xyz(y=self.small_step_size_xy_mm)
        elif option == "BACK MEDIUM":
            self.prusa_go_by_xyz(y=self.medium_step_size_xy_mm)
        elif option == "BACK LARGE":
            self.prusa_go_by_xyz(y=self.large_step_size_xy_mm)

        elif option == "FRONT SMALL":
            self.prusa_go_by_xyz(y=-self.small_step_size_xy_mm)
        elif option == "FRONT MEDIUM":
            self.prusa_go_by_xyz(y=-self.medium_step_size_xy_mm)
        elif option == "FRONT LARGE":
            self.prusa_go_by_xyz(y=-self.large_step_size_xy_mm)
        elif option == "GO":
            p2_x = self.target_x_entry.get()
            p2_y = self.target_y_entry.get()
            p2_z = self.target_z_entry.get()

            if p2_x and p2_y and p2_z:
                # # raise to a safe height
                # self.prusa_go_to_xyz(z=30)
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
            self.ser.write(str.encode("G28 X F600\r\n"))
            self.prusa_x_referenced = True
        elif option == "Y":
            self.ser.write(str.encode("G28 Y F600\r\n"))
            self.prusa_y_referenced = True
        elif option == "Z":
            self.ser.write(str.encode("G28 Z F600\r\n"))
            self.prusa_z_referenced = True
        elif option == "nanodrive":
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            self.nanodrive_min, self.nanodrive_max = self.dispatch('get_nanodrive_min_max')
            self.target_nanodrive_entry.config(from_=self.nanodrive_min, to=self.nanodrive_max)
        elif option == "ALL":
            # Disable software endstops to allow Z-axis movement before homing
            self.ser.write(b'M211 S0\n')
            time.sleep(1)

            # Move the Z-axis up by 10mm to avoid collisions
            self.ser.write(b'G91\n')         # Set to relative positioning
            self.ser.write(b'G1 Z20 F600\n')  # Move Z up 10mm at 600mm/min
            time.sleep(2)

            # Re-enable software endstops
            self.ser.write(b'M211 S1\n')

            time.sleep(1)

            self.ser.write(str.encode("G28 X Y Z F600\r\n"))

            self.prusa_x_referenced = True
            self.prusa_y_referenced = True
            self.prusa_z_referenced = True

            self.prusa_go_to_xyz(z=self.safety_height)
            self.update_prusa_position()
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            results = self.dispatch('get_nanodrive_min_max')
            self.nanodrive_min, self.nanodrive_max = results[0]
            self.target_nanodrive_entry.config(from_=self.nanodrive_min, to=self.nanodrive_max)

            self.home_button.configure(state=NORMAL)
            self.calibrate_home_position_button.configure(state=NORMAL)
            self.point_1_button.configure(state=NORMAL)
            self.point_2_button.configure(state=NORMAL)
            self.point_3_button.configure(state=NORMAL)
            self.sample_drop_panel_canvas.configure(state=NORMAL)
            self.sample_drop_panel_activate_circles()
            self.sample_drop_panel_tooltip.configure(state=NORMAL)
            self.go_button.configure(state=NORMAL)
            self.update_position_button.configure(state=NORMAL)
            self.dispatch('activate_switch_view_button')
            self.dispatch('activate_protocol_module_state', 'stage_reference_is_setup')

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
            # # # pause for 1 second to allow the printer to move
            # time.sleep(2)
            self.update_prusa_position()

    def prusa_go_by_xyz(self, x=None, y=None, z=None):
        self.ser.write(str.encode("G91\r\n"))
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

    def find_prusa_com_ports(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.description == "Original Prusa i3 MK3 (COM3)":
                return port.name

    def wait_for_prusa_process_complete(self, process_name, critiria):
        """Waits for a signal from the printer that homing is complete."""
        while True:
            line = self.ser.readline().decode('utf-8').strip()
            print(line)
            if critiria in line:
                return line
            # elif line:
                # print(f"Received during {process_name}: {line}")
                

    def update_prusa_speed(self):
        self.speed = self.speed_slider.get()
        print(f"Speed of stage is updated to: {self.speed} mm/s")

    def connect_prusa_device(self):
        self.port = self.find_prusa_com_ports()
        self.ser = serial.Serial(self.port, 115200)
        # time.sleep(3)

        printer_status = self.is_prusa_on()
        if printer_status:
            print("Connected to the PRUSA " + self.ser.name)
        else:
            self.ser.close()
            print("Couldn't connect to the PRUSA")

    def is_prusa_on(self):
        try:
            self.ser.flushInput()
            self.ser.flushOutput()
            self.ser.write(b'M105\n')
            print('write M105')
            time.sleep(3)
            response = self.ser.read_all().decode('utf-8')
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
        if dim == 'XYZ':
            X = np.round(float(line.split(' ')[0].split(':')[1]),2)
            Y = np.round(float(line.split(' ')[1].split(':')[1]),2)
            Z = np.round(float(line.split(' ')[2].split(':')[1]),2)
            return X, Y, Z
        elif dim == 'X':
            X = np.round(float(line.split(' ')[0].split(':')[1]),2)
            return X
        elif dim == 'Y':
            Y = np.round(float(line.split(' ')[1].split(':')[1]),2)
            return Y
        elif dim == 'Z':
            Z = np.round(float(line.split(' ')[2].split(':')[1]),2)
            return Z

    def update_home_position(self):
        self.home_position = self.get_current_prusa_position('XYZ')

        # recalculate the circle positions
        self.sample_drop_panel_calculate_circle_pursa_coordinates()
        print("Home position updated")

    def handle_switch_view(self, view):
        if view == "TO_OBJECTIVE":
            # send g code to move the stage to the right
            print("Switching to objective view")
            self.prusa_go_by_xyz(x=self.calibration_from_widefield_to_objective_x, y=self.calibration_from_widefield_to_objective_y)
            self.prusa_go_by_xyz(z=self.calibration_from_widefield_to_objective_z)

        elif view == "TO_WIDE":
            # send g code to move the stage to the left
            print("Switching to wide view")
            self.prusa_go_by_xyz(z=-self.calibration_from_widefield_to_objective_z)
            self.prusa_go_by_xyz(x=-self.calibration_from_widefield_to_objective_x, y=-self.calibration_from_widefield_to_objective_y)
        else:
            print("Error Happened", view)

        return True

    # def roll_over_coarse_fine_z_position(self, coarse_mm, fine_um):
    #             # Check if fine_nm exceeds 1000 nm and adjust the coarse stage accordingly
    #     total_nm = coarse_mm * 1000 + fine_um  # Convert coarse to nm and add fine adjustment
    #     adjusted_coarse_um = int(total_nm // 1000)  # New coarse value in µm
    #     remaining_nm = total_nm % 1000  # Fine position within the µm range
    #     return f"{adjusted_coarse_um} mm + {remaining_nm:.0f} um"

    def handle_update_nanodrive_position(self, nanodrive_position):
        self.current_nanodrive_position = nanodrive_position
        self.current_nanodrive_z_entry.config(text=nanodrive_position)
        return None

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
        return None

    def _handle_test_sampling_points(self, view_to_inspect_in, relative_distance_for_sampling_points):
        # calculate the absolute position of the sampling points
        sampling_position_x, sampling_position_y = self.calculate_sampling_point_position_in_absolute_coordinate(view_to_inspect_in, relative_distance_for_sampling_points)

        # move to the test sample spot one by one
        for i in range(len(sampling_position_x)):
            print(f'Moving to the test sample spot {sampling_position_x[i]}, {sampling_position_y[i]}')
            self.prusa_go_to_xyz(x=sampling_position_x[i], y=sampling_position_y[i])
            time.sleep(3)
        
        # Move back to the center of the captured frame
        if view_to_inspect_in == "OBJECTIVE":
            self.prusa_go_to_xyz(x=self.capture_frame_center_objective_x, y=self.capture_frame_center_objective_y)
        elif view_to_inspect_in == "WIDEFIELD":
            self.prusa_go_to_xyz(x=self.capture_frame_center_widefield_x, y=self.capture_frame_center_widefield_y)
        else:
            raise ValueError("The center of the frame is not captured yet")
        
    def handle_move_to_a_single_sampling_point_during_acquisition(self, view_to_inspect_in, relative_distance_for_sampling_point):
        sampling_position_x, sampling_position_y = self.calculate_sampling_point_position_in_absolute_coordinate(view_to_inspect_in, relative_distance_for_sampling_point)
        self.prusa_go_to_xyz(x=sampling_position_x, y=sampling_position_y)
        
        while True:
            current_x = float(self.get_current_prusa_position('XYZ')[0])
            current_y = float(self.get_current_prusa_position('XYZ')[1])
            if current_x == sampling_position_x and current_y == sampling_position_y:
                break

        self.dispatch('task_completed')
        return True

    def calculate_sampling_point_position_in_absolute_coordinate(self, view_to_inspect_in, relative_distance_for_sampling_points):
        if view_to_inspect_in == "OBJECTIVE":
            if self.capture_frame_center_objective_x and self.capture_frame_center_objective_y:
                # printer's y axis is flipped
                filped_sampling_position_y = relative_distance_for_sampling_points[1] * -1

                sampling_position_x = relative_distance_for_sampling_points[0] / 1000 + self.capture_frame_center_objective_x
                sampling_position_y = filped_sampling_position_y / 1000 + self.capture_frame_center_objective_y
        elif view_to_inspect_in == "WIDEFIELD":
            if self.capture_frame_center_widefield_x and self.capture_frame_center_widefield_y:
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

        return sampling_position_x, sampling_position_y

    def handle_updated_captured_frame_center(self, field_of_view):
        print("Captured the center of the frame")
        if field_of_view == 'WIDEFIELD':
            self.capture_frame_center_widefield_x = float(self.get_current_prusa_position('XYZ')[0])
            self.capture_frame_center_widefield_y = float(self.get_current_prusa_position('XYZ')[1])

            print(f"Captured the center of the frame in the physical space: {self.capture_frame_center_widefield_x}, {self.capture_frame_center_widefield_y}")

            self.capture_frame_center_objective_x = self.capture_frame_center_widefield_x + self.calibration_from_widefield_to_objective_x
            self.capture_frame_center_objective_y = self.capture_frame_center_widefield_y + self.calibration_from_widefield_to_objective_y
        elif field_of_view == 'OBJECTIVE':
            self.capture_frame_center_objective_x = float(self.get_current_prusa_position('XYZ')[0])
            self.capture_frame_center_objective_y = float(self.get_current_prusa_position('XYZ')[1])

            self.capture_frame_center_widefield_x = self.capture_frame_center_objective_x - self.calibration_from_widefield_to_objective_x
            self.capture_frame_center_widefield_y = self.capture_frame_center_objective_y - self.calibration_from_widefield_to_objective_y
        return None


    def handle_move_stage_to_target_sample_drop_in_widefield_view_during_aquisition(self, sample_drop_row_column):
        # get current position of the stage
        current_x = float(self.get_current_prusa_position('XYZ')[0])
        current_y = float(self.get_current_prusa_position('XYZ')[1])
        lower_limit_current, upper_limit_current = self.calculate_focus_range_at_position(current_x, current_y)
        lower_limit_target, upper_limit_target = self.calculate_focus_range_at_position(sample_drop_row_column[0], sample_drop_row_column[1])

        # get the bigger upper limit to avoid collision
        upper_limit = max(upper_limit_current, upper_limit_target)

        self.prusa_go_to_xyz(z=upper_limit)
        print('moving to the upper limit')
        # check if the stage is at the target position
        while True:
            current_z_position = np.round(float(self.get_current_prusa_position('Z')),2)
            print('stuck when trying to move to the upper limit')
            print(f"current z position: {current_z_position}")
            print(f"upper limit: {upper_limit}")
            if current_z_position == upper_limit:
                break

        # re-position nanodrive to the initial position
        self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
        print('moving nanodrive to the initial position')

        # move to the target sample drop position
        self.sample_drop_panel_move_stage(sample_drop_row_column[0], sample_drop_row_column[1])

        self.dispatch('task_completed')
        return None

    def handle_reposition_stage_and_nanodrive_in_objective_view_during_acquisition(self):
        # move the stage to the best focus position in widefield view
        self.prusa_go_to_xyz(z=self.best_prusa_focus_position_widefield)

        # check if the stage is at the target position
        while True:
            current_position = self.get_current_prusa_position('Z')
            if current_position == self.best_prusa_focus_position_widefield:
                break

        # re-position nanodrive to the initial position
        self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
        self.dispatch('task_completed')
        return True

    def handle_focus_widefield_camera(self):
        self.run_in_thread(self._handle_focus_widefield_camera)

    def _handle_focus_widefield_camera(self):
        print("Focusing the widefield camera")
        # move nanodrive to the middle position
        self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)

        # wait until the nanodrive position is at the target position
        while True:
            current_nanodrive_position = self.get_nanodrive_position()
            print(f"current nanodrive position: {current_nanodrive_position}")
            if current_nanodrive_position == self.initial_nanodrive_position:
                break
        
        # get the upper limit and lower limit for prusa
        current_x = float(self.get_current_prusa_position('XYZ')[0])
        current_y = float(self.get_current_prusa_position('XYZ')[1])
        lower_limit, upper_limit = self.calculate_focus_range_at_position(current_x, current_y)
    
        if upper_limit and lower_limit:

            self.best_prusa_focus_position_widefield, self.best_prusa_focus_position_widefield_refine_min, self.best_prusa_focus_position_widefield_refine_max = self.focus_prusa(lower_limit, upper_limit, self.prusa_rough_focus_step_number)
            print(f"Widefield pursa focus [{lower_limit}, {upper_limit}] with {self.prusa_rough_focus_step_number} steps")
            print(f"Best focus position for widefield camera: {self.best_prusa_focus_position_widefield}")
            print(f"Refined focus range: {self.best_prusa_focus_position_widefield_refine_min} - {self.best_prusa_focus_position_widefield_refine_max}")
            return True
        else:
            print("Please interpolate the focus range first")
            return False
        
    def focus_prusa(self, lower_limit, upper_limit, step_number):
        focus_score = []
        self.focus_score_at_current_z_position = None

        z_range = np.linspace(float(upper_limit), float(lower_limit), int(step_number))
        # round up the z range to 2 decimal places
        z_range = np.round(z_range, 2)

        print(f"Pursa z focus within [{lower_limit}, {upper_limit}]mm with {step_number} steps")
        print(f"z positions: {z_range}")

        for z in z_range:
            self.prusa_go_to_xyz(z=z)

            # wait until the stage position is at the target position
            while True:
                current_z_position = float(self.get_current_prusa_position('Z'))
                if current_z_position == z:
                    break

            # wait until the robot is stable
            time.sleep(self.prusa_movement_stabilization_time)

            self.dispatch('calculate_focus_score')

            # wait until the score is different from the previous one
            while True:
                if self.focus_score_at_current_z_position:
                    if len(focus_score) > 0:
                        if self.focus_score_at_current_z_position != focus_score[-1]:
                            break
                    else:
                        break

            focus_score.append(self.focus_score_at_current_z_position)

        best_prusa_focus_position, best_prusa_focus_refine_min, best_prusa_focus_refine_max= self.refine_focus_range(focus_score, z_range, self.prusa_rough_fine_focus_overlap_step_number)
        print(f"Best focus position for prusa: {best_prusa_focus_position}")
        print(f"Refined focus range for prusa: {best_prusa_focus_refine_min} - {best_prusa_focus_refine_max}")

        self.prusa_go_to_xyz(z=best_prusa_focus_position)
        time.sleep(self.prusa_movement_stabilization_time)
        return best_prusa_focus_position, best_prusa_focus_refine_min, best_prusa_focus_refine_max
    
    def focus_nanodrive(self, lower_limit, upper_limit, step_number):
        # focus with nanodrive
        nanodrive_z_range = np.linspace(lower_limit, upper_limit, step_number)
        print(f"z range: {nanodrive_z_range}")
        # round up the z range to 2 decimal places
        nanodrive_z_range = np.round(nanodrive_z_range)
        print(f"z range after rounding: {nanodrive_z_range}")
        focus_score = []
        self.focus_score_at_current_z_position = None

        for z in nanodrive_z_range:
            self.dispatch('move_nanodrive_to', z)

            # wait until the nanodrive position is at the target position
            while True:
                current_nanodrive_position = self.get_nanodrive_position()
                if current_nanodrive_position == z:
                    break

            # wait until the robot is stable
            time.sleep(self.nanodrive_movement_stabilization_time)

            self.dispatch('calculate_focus_score')

            # wait until the score is different from the previous one
            while True:
                if self.focus_score_at_current_z_position:
                    #check if the focus score is different from the previous one except the first one
                    if len(focus_score) > 0:
                        if self.focus_score_at_current_z_position != focus_score[-1]:
                            break
                    else:
                        break

            focus_score.append(self.focus_score_at_current_z_position)

        # find the best focus position
        best_nanodrive_focus_position_objective = nanodrive_z_range[np.argmax(focus_score)]
        self.dispatch('move_nanodrive_to', best_nanodrive_focus_position_objective)

        print(f'focus score at each z position: {focus_score}')
        print(f"The best focus position is {best_nanodrive_focus_position_objective}, with the focus score of {max(focus_score)}")
        return best_nanodrive_focus_position_objective


    def refine_focus_range(self, focus_score, z_axis_range, overlap_step_number):
        k = np.argmax(focus_score)
        fineMin = z_axis_range[max(k - overlap_step_number, 0)]
        fineMax = z_axis_range[min(k + overlap_step_number, len(z_axis_range) - 1)]
        print("Refined focus range for Prusa:", fineMin, fineMax)
        best_focus_position = z_axis_range[k]
        return best_focus_position, fineMin, fineMax

    def handle_focus_widefield_camera_during_acquisition(self):
        def on_focus_complete(result):
            if result:
                self.dispatch('task_completed')
            else:
                self.dispatch('abort_aquisition')
        self.run_in_thread_with_callback(self._handle_focus_widefield_camera, on_focus_complete)

    def handle_focus_objective_camera(self):
        self.run_in_thread(self._handle_focus_objective_camera)

    def _handle_focus_objective_camera(self):
        print("Focusing the objective camera")
        if self.best_prusa_focus_position_widefield_refine_max and self.best_prusa_focus_position_widefield_refine_min:

            # re-position nanodrive before refining the prusa focus position
            self.dispatch('move_nanodrive_to', self.initial_nanodrive_position)
            while True:
                current_nanodrive_position = self.get_nanodrive_position()
                if current_nanodrive_position == self.initial_nanodrive_position:
                    break

            # calculate the refined prusa focus range in the objective view
            prusa_focus_position_objective_refine_max = round(self.best_prusa_focus_position_widefield_refine_max + self.calibration_from_widefield_to_objective_z, 2)
            prusa_focus_position_objective_refine_min = round(self.best_prusa_focus_position_widefield_refine_min + self.calibration_from_widefield_to_objective_z, 2)
            print('Converting focus range from widefield camera for the objective camera')
            print(f"focus plane offset widefield to objective: {self.calibration_from_widefield_to_objective_z}")
            print(f"Converted range for objective camera: {prusa_focus_position_objective_refine_min} - {prusa_focus_position_objective_refine_max}")

            # fine the best focus position in the objective view with the refined range
            self.best_prusa_focus_position_objective, _, _ = self.focus_prusa(prusa_focus_position_objective_refine_min, prusa_focus_position_objective_refine_max, self.prusa_fine_focus_step_number)
            # update the best focus position in the widefield view
            self.best_prusa_focus_position_widefield = round(self.best_prusa_focus_position_objective - self.calibration_from_widefield_to_objective_z, 2)
            return True
        else:
            print("Please focus the widefield camera first")
            return False

    def handle_focus_objective_camera_during_acquisition(self):
        def on_focus_complete(result):
            if result:
                self.dispatch('task_completed')
            else:
                self.dispatch('abort_aquisition')

        # Call the thread-running function, passing the callback
        self.run_in_thread_with_callback(self._handle_focus_objective_camera, on_focus_complete)


    def handle_update_focus_score(self, focus_score):
        self.focus_score_at_current_z_position = focus_score
        return None

    def update_prusa_position(self):
        current_prusa_position = self.get_current_prusa_position('XYZ')

        if self.prusa_x_referenced:
            current_prusa_position_x = str(current_prusa_position[0])
            self.current_x_entry.config(text=current_prusa_position_x)
        
        if self.prusa_y_referenced:
            current_prusa_position_y = str(current_prusa_position[1])
            self.current_y_entry.config(text=current_prusa_position_y)

        if self.prusa_z_referenced:
            current_prusa_position_z = str(current_prusa_position[2])
            self.current_z_entry.config(text=current_prusa_position_z)

    def register_focus_range(self, point_label):
        lower, upper = self.get_user_input_focus_range()
        position = self.get_current_prusa_position('XYZ')

        if point_label == "P1":
            if lower is not None and upper is not None and position is not None:
                self.lower_limit_p1, self.upper_limit_p1 = float(lower), float(upper)
                self.focus_p1 = [float(position[0]), float(position[1]), float(position[2])]
                # Update the labels
                self.point_1_focus_range_label.config(text=f"{self.lower_limit_p1}-{self.upper_limit_p1}")
                self.point_1_position_label.config(text=f"{self.focus_p1}")
                self.point_1_button.configure(bootstyle='info')
            else:
                self.point_1_focus_range_label.config(text="N/A")
                self.point_1_position_label.config(text="N/A")
                self.point_1_button.configure(bootstyle='info_outline')

        elif point_label == "P2":
            if lower is not None and upper is not None and position is not None:
                self.lower_limit_p2, self.upper_limit_p2 = float(lower), float(upper)
                self.focus_p2 = [float(position[0]), float(position[1]), float(position[2])]
                # Update the labels
                self.point_2_focus_range_label.config(text=f"{self.lower_limit_p2}-{self.upper_limit_p2}")
                self.point_2_position_label.config(text=f"{self.focus_p2}")
                self.point_2_button.configure(bootstyle='info')
            else:
                self.point_2_focus_range_label.config(text="N/A")
                self.point_2_position_label.config(text="N/A")
                self.point_2_button.configure(bootstyle='info_outline')

        elif point_label == "P3":
            if lower is not None and upper is not None and position is not None:
                self.lower_limit_p3, self.upper_limit_p3 = float(lower), float(upper)
                self.focus_p3 = [float(position[0]), float(position[1]), float(position[2])]
                # Update the labels
                self.point_3_focus_range_label.config(text=f"{self.lower_limit_p3}-{self.upper_limit_p3}")
                self.point_3_position_label.config(text=f"{self.focus_p3}")
                self.point_3_button.configure(bootstyle='info')
            else:
                self.point_3_focus_range_label.config(text="N/A")
                self.point_3_position_label.config(text="N/A")
                self.point_3_button.configure(bootstyle='info_outline')

        # Check if all the points are registered
        if self.lower_limit_p1 and self.upper_limit_p1 and self.lower_limit_p2 and self.upper_limit_p2 and self.lower_limit_p3 and self.upper_limit_p3:
            self.interpolate_button.configure(state=NORMAL)
        else:
            self.interpolate_button.configure(state=DISABLED)

    def get_user_input_focus_range(self):
        # Create a dialog window
        dialog = tk.Toplevel(self.parent)
        dialog.title("Focus Range")
        dialog.grab_set()  # Make this dialog modal (prevents interacting with main window)

        # Set a fixed position for the dialog window
        dialog.update_idletasks()  # Update "requested size" from geometry manager

        # Use absolute coordinates to center the dialog on the parent window
        x = self.parent.winfo_rootx() + (self.parent.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.parent.winfo_rooty() + (self.parent.winfo_height() // 2) - (dialog.winfo_height() // 2)
        
        dialog.geometry(f"+{x}+{y}")

        # Variables to store user input
        user_input = {"upper_limit": None, "lower_limit": None}

        # Create labels and entries for upper and lower limits
        ttk.Label(dialog, text="Z Upper Limit [mm]:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        upper_limit_entry = tk.Entry(dialog)
        upper_limit_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(dialog, text="Z Lower Limit [mm]:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        lower_limit_entry = tk.Entry(dialog)
        lower_limit_entry.grid(row=1, column=1, padx=5, pady=5)

        # Define what happens when user clicks OK
        def on_ok():
            try:
                # Fetch and convert user input
                user_input["upper_limit"] = float(upper_limit_entry.get())
                user_input["lower_limit"] = float(lower_limit_entry.get())
                # Close the dialog window
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numeric values.")

        # Define what happens when user clicks Cancel
        def on_cancel():
            user_input["upper_limit"] = None
            user_input["lower_limit"] = None
            dialog.destroy()

        # OK and Cancel buttons
        ok_button = ttk.Button(dialog, text="OK", command=on_ok, bootstyle='info')
        ok_button.grid(row=2, column=0, padx=5, pady=5, sticky="e")

        cancel_button = ttk.Button(dialog, text="Cancel", command=on_cancel, bootstyle='info')
        cancel_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # You can force the focus on the entry widget to let user type immediately
        upper_limit_entry.focus_set()

        # Wait for the dialog to close
        dialog.wait_window()

        # Return the user input values after the dialog is closed
        return user_input["lower_limit"], user_input["upper_limit"]

    def interpolate_prusa_focus_range(self):
        # Check if all the points are registered
        if self.lower_limit_p1 and self.upper_limit_p1 and \
        self.lower_limit_p2 and self.upper_limit_p2 and \
        self.lower_limit_p3 and self.upper_limit_p3 and \
        self.focus_p1 and self.focus_p2 and self.focus_p3:
        
            # Extract the lower limit points
            lower_points = [
                [self.focus_p1[0], self.focus_p1[1], self.lower_limit_p1],
                [self.focus_p2[0], self.focus_p2[1], self.lower_limit_p2],
                [self.focus_p3[0], self.focus_p3[1], self.lower_limit_p3]
            ]

            upper_points = [
                [self.focus_p1[0], self.focus_p1[1], self.upper_limit_p1],
                [self.focus_p2[0], self.focus_p2[1], self.upper_limit_p2],
                [self.focus_p3[0], self.focus_p3[1], self.upper_limit_p3]
            ]

            # Function to calculate plane coefficients
            def fit_plane(points):
                import numpy as np
                x1, y1, z1 = points[0]
                x2, y2, z2 = points[1]
                x3, y3, z3 = points[2]

                # Solve for the plane coefficients
                A = [
                    [x1, y1, 1],
                    [x2, y2, 1],
                    [x3, y3, 1]
                ]
                B = [z1, z2, z3]
                coefficients = np.linalg.solve(A, B)
                return coefficients  # Returns a, b, c for the plane z = ax + by + c

            # Fit both planes
            lower_plane_coeffs = fit_plane(lower_points)
            upper_plane_coeffs = fit_plane(upper_points)

            # Define functions for the planes
            def lower_plane(x, y):
                a, b, c = lower_plane_coeffs
                return a * x + b * y + c

            def upper_plane(x, y):
                a, b, c = upper_plane_coeffs
                return a * x + b * y + c
            
            # Store as attributes
            self.lower_plane_func = lower_plane
            self.upper_plane_func = upper_plane

            print("Focus range lower and upper limits registered and interpolated across the XY plane successfully.")

            self.dispatch('activate_camera_autofocus_button')
            self.dispatch('activate_protocol_module_state','focus_range_is_setup')

        else:
            print("Please register the focus range first")

    def calculate_focus_range_at_position(self, x, y):
        if self.lower_plane_func and self.upper_plane_func:
            # Calculate the focus range at the given XY position
            lower_limit = np.round(self.lower_plane_func(x, y),2)
            upper_limit = np.round(self.upper_plane_func(x, y),2)
            return lower_limit, upper_limit
        else:
            print("Please register the focus range first")

    def sample_drop_panel_create_circle_grid(self): 
        skip_positions = [(0, 0), (4, 0), (0, 3), (4, 3)]  # Positions to skip

        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            vertical_offset = (max(self.sample_drop_panel_grid_pattern) - num_circles) * self.sample_drop_panel_spacing // 2
            for row_index in range(num_circles):
                true_row_index = row_index + vertical_offset // self.sample_drop_panel_spacing
                
                # Skip the specified corner positions
                if (true_row_index, col_index) in skip_positions:
                    continue
                
                x = col_index * self.sample_drop_panel_spacing + self.sample_drop_panel_spacing // 2
                y = vertical_offset + row_index * self.sample_drop_panel_spacing + self.sample_drop_panel_spacing // 2
                
                circle = self.sample_drop_panel_canvas.create_oval(
                    x - self.sample_drop_panel_circle_radius, y - self.sample_drop_panel_circle_radius,
                    x + self.sample_drop_panel_circle_radius, y + self.sample_drop_panel_circle_radius,
                    fill='grey', outline='black', tags=f"circle_{true_row_index}_{col_index}"
                )
                
                # Bind click event to each circle
                self.sample_drop_panel_canvas.tag_bind(circle, '<Button-1>',
                    lambda event, row=true_row_index, col=col_index: self.sample_drop_panel_move_stage(row, col))
                # Bind mouse over and leave events for hover effect and tooltip
                self.sample_drop_panel_canvas.tag_bind(circle, '<Enter>',
                    lambda event, row=true_row_index, col=col_index: self.sample_drop_panel_on_hover(event, row, col))
                self.sample_drop_panel_canvas.tag_bind(circle, '<Leave>', self.sample_drop_panel_on_leave)

    def sample_drop_panel_calculate_circle_pursa_coordinates(self):
        home_x, home_y = self.home_position[0], self.home_position[1]

        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            for row_index in range(num_circles):
                pursa_coordinate_x = np.round(home_x + self.offset_home_to_p1_x + col_index * self.interval_between_drop_x, 2)
                pursa_coordinate_y = np.round(home_y + self.offset_home_to_p1_y - row_index * self.interval_between_drop_y, 2)
                self.sample_drop_panel_circle_pursa_coordinates[(row_index, col_index)] = (pursa_coordinate_x, pursa_coordinate_y)

    def sample_drop_panel_move_stage(self, row, col):
        if not self.prusa_x_referenced or not self.prusa_y_referenced or not self.prusa_z_referenced:
            print("Stage is not referenced. Action disabled.")
            return  # Disable action if stage is not referenced

        # Reset the previous circle color
        if self.sample_drop_panel_current_sample_row_column:
            prev_row, prev_col = self.sample_drop_panel_current_sample_row_column
            self.sample_drop_panel_canvas.itemconfig(f"circle_{prev_row}_{prev_col}", fill='skyblue')

        # Highlight the selected circle
        self.sample_drop_panel_canvas.itemconfig(f"circle_{row}_{col}", fill='yellow')
        self.sample_drop_panel_current_sample_row_column = (row, col)
        print(f"Moving XY stage to position: Row {row}, Column {col}")
        self.sample_drop_panel_current_sample_pursa_coordinates = self.sample_drop_panel_circle_pursa_coordinates[(row, col)]
        print(f"Pursa coordinates: {self.sample_drop_panel_current_sample_pursa_coordinates}")
        self.prusa_go_to_xyz(x=self.sample_drop_panel_current_sample_pursa_coordinates[0], y=self.sample_drop_panel_current_sample_pursa_coordinates[1])

    def sample_drop_panel_on_hover(self, event, row, col):
        # Prevent changing color if hovering over the current position
        if self.sample_drop_panel_current_sample_row_column != (row, col):
            self.sample_drop_panel_canvas.itemconfig(f"circle_{row}_{col}", fill='lightgreen')
        self.sample_drop_panel_tooltip.config(text="Move to this drop")
        self.sample_drop_panel_tooltip.place(x=event.x_root - self.master.winfo_rootx() + 10, y=event.y_root - self.master.winfo_rooty() + 10)

    def sample_drop_panel_on_leave(self, event):
        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            for row_index in range(num_circles):
                if self.sample_drop_panel_current_sample_row_column != (row_index, col_index):
                    self.sample_drop_panel_canvas.itemconfig(f"circle_{row_index}_{col_index}", fill='skyblue')
        self.sample_drop_panel_tooltip.place_forget()

    def sample_drop_panel_activate_circles(self):
        if self.prusa_x_referenced and self.prusa_y_referenced and self.prusa_z_referenced:
            for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
                for row_index in range(num_circles):
                    self.sample_drop_panel_canvas.itemconfig(f"circle_{row_index}_{col_index}", fill='skyblue')
        else:
            print("Stage is not referenced. Action disabled.")

    def handle_select_sample_for_protocol(self):
        # Check if the window is already open
        if hasattr(self, 'protocol_select_sample_window') and self.protocol_select_sample_window.winfo_exists():
            print("The sample selection window is already open.")
            self.protocol_select_sample_window.focus_force()
            return

        # Create the selection window
        self.protocol_select_sample_window = tk.Toplevel(self.parent)
        self.protocol_select_sample_window.title("Select Sample")
        self.protocol_select_sample_window.geometry("700x650")
        self.protocol_select_sample_window.resizable(False, False)

        # Create a canvas to display the sample
        self.protocol_select_sample_frame = ttk.Frame(self.protocol_select_sample_window)
        self.protocol_select_sample_frame.grid(row=0, column=0, padx=10, pady=5)

        self.protocol_select_sample_canvas = tk.Canvas(self.protocol_select_sample_frame, width=500, height=650)
        self.protocol_select_sample_canvas.grid(row=0, column=0, columnspan=2, rowspan=11, padx=20, pady=5)

        # Create the circles
        self.protocol_select_sample_create_circle_grid()

        # Add a confirm button
        select_all_button = ttk.Button(self.protocol_select_sample_frame, text="Select All", command=self.protocol_select_all_samples, bootstyle='info_outline')
        select_all_button.grid(row=4, column=2, pady=10, padx=30, sticky='ew')

        cancel_all_button = ttk.Button(self.protocol_select_sample_frame, text="Cancel All", command=self.protocol_cancel_all_samples, bootstyle='danger-outline')
        cancel_all_button.grid(row=5, column=2, pady=10, padx=30, sticky='ew')

        confirm_button = ttk.Button(self.protocol_select_sample_frame, text="Confirm", command=self.confirm_sample_selection, bootstyle='info')
        confirm_button.grid(row=6, column=2, pady=10, padx=30, sticky='ew')

        # Focus the attention to the new window
        self.protocol_select_sample_window.focus_force()

    def protocol_select_sample_create_circle_grid(self):
        enlarged_spacing = self.sample_drop_panel_spacing * 3  # Enlarge the spacing for better visualization
        enlarged_circle_radius = self.sample_drop_panel_circle_radius * 3 # Enlarge the circle radius for better visualization

        skip_positions = [(0, 0), (4, 0), (0, 3), (4, 3)]

        # Ensure `selected_circles` and `selected_circle_positions` exist and are persistent
        if not hasattr(self, 'selected_circles'):
            self.selected_circles = set()  # Track selected circles by their IDs
        if not hasattr(self, 'selected_circle_positions'):
            self.selected_circle_positions = []  # Track (row, column) positions of selected circles

        self.circles = []  # Store references to all circles

        for col_index, num_circles in enumerate(self.sample_drop_panel_grid_pattern):
            vertical_offset = (max(self.sample_drop_panel_grid_pattern) - num_circles) * enlarged_spacing // 2
            for row_index in range(num_circles):
                true_row_index = row_index + vertical_offset // enlarged_spacing

                # Skip the specified corner positions
                if (true_row_index, col_index) in skip_positions:
                    continue

                x = col_index * enlarged_spacing + enlarged_spacing // 2
                y = vertical_offset + row_index * enlarged_spacing + enlarged_spacing // 2

                # Determine the initial color of the circle based on its selection status
                if (true_row_index, col_index) in self.selected_circle_positions:
                    fill_color = 'skyblue'  # Selected
                else:
                    fill_color = 'grey'  # Not selected

                circle = self.protocol_select_sample_canvas.create_oval(
                    x - enlarged_circle_radius, y - enlarged_circle_radius,
                    x + enlarged_circle_radius, y + enlarged_circle_radius,
                    fill=fill_color, outline='black', tags=f"circle_{true_row_index}_{col_index}"
                )
                self.circles.append((circle, x, y, true_row_index, col_index))

                # Bind click event for toggling selection
                self.protocol_select_sample_canvas.tag_bind(circle, '<Button-1>',
                            lambda event, c=circle, r=true_row_index, col=col_index: self.protocol_select_sample_toggle_selection(c, r, col))

        # Bind mouse events for dragging selection
        self.protocol_select_sample_canvas.bind('<Button-1>', self.protocol_select_sample_on_drag_start)
        self.protocol_select_sample_canvas.bind('<B1-Motion>', self.protocol_select_sample_on_drag_motion)
        self.protocol_select_sample_canvas.bind('<ButtonRelease-1>', self.protocol_select_sample_on_drag_end)

    def protocol_select_sample_toggle_selection(self, circle, row, col):
        """
        Toggle selection state of the clicked circle.
        """
        if circle in self.selected_circles:  # Deselect if already selected
            self.selected_circles.remove(circle)
            self.selected_circle_positions.remove((row, col))  # Remove position from the list
            self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')
        else:  # Select if not already selected
            self.selected_circles.add(circle)
            self.selected_circle_positions.append((row, col))  # Add position to the list
            self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')

        # Print updated selected circle positions
        print("Currently selected circles (row, col):", self.selected_circle_positions)

    def protocol_select_sample_on_drag_start(self, event):
        # Store the starting position of the drag
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        # Initialize a rectangle for visual feedback during the drag
        self.drag_rectangle = self.protocol_select_sample_canvas.create_rectangle(
            self.drag_start_x, self.drag_start_y, self.drag_start_x, self.drag_start_y, outline='skyblue', tag='drag_rectangle'
        )

    def protocol_select_sample_on_drag_motion(self, event):
        # Update the rectangle to match the current drag area
        self.protocol_select_sample_canvas.coords(
            self.drag_rectangle, self.drag_start_x, self.drag_start_y, event.x, event.y
        )
        # Compute the selection rectangle bounds
        x1, y1 = min(self.drag_start_x, event.x), min(self.drag_start_y, event.y)
        x2, y2 = max(self.drag_start_x, event.x), max(self.drag_start_y, event.y)

        # Check which circles are within the rectangle (highlight only for visual feedback)
        for circle, cx, cy, row, col in self.circles:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if circle not in self.selected_circles:  # Highlight if not already selected
                    self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='skyblue')
            elif circle not in self.selected_circles:  # Revert unselected circles to grey
                self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')

    def protocol_select_sample_on_drag_end(self, event):
        # Remove the drag rectangle after the drag operation ends
        self.protocol_select_sample_canvas.delete(self.drag_rectangle)

        # Compute the selection rectangle bounds
        x1, y1 = min(self.drag_start_x, event.x), min(self.drag_start_y, event.y)
        x2, y2 = max(self.drag_start_x, event.x), max(self.drag_start_y, event.y)

        # Toggle selection for circles within the rectangle
        for circle, cx, cy, row, col in self.circles:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if circle in self.selected_circles:  # If already selected, deselect
                    self.selected_circles.remove(circle)
                    self.selected_circle_positions.remove((row, col))  # Remove position from the list
                    self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')
                else:  # Otherwise, select it
                    self.selected_circles.add(circle)
                    self.selected_circle_positions.append((row, col))  # Add position to the list
                    self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')

        # Print updated selected circle positions
        print("Currently selected circles (row, col):", self.selected_circle_positions)

    def protocol_select_all_samples(self):
        for circle, cx, cy, row, col in self.circles:
            if circle not in self.selected_circles:
                self.selected_circles.add(circle)
                self.selected_circle_positions.append((row, col))
                self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')
        print("All circles selected:")
        self.update_circle_colors()

    def protocol_cancel_all_samples(self):
        for circle, cx, cy, row, col in self.circles:
            if circle in self.selected_circles:
                self.selected_circles.remove(circle)
                self.selected_circle_positions.remove((row, col))
                self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')
        print("All circles deselected.")
        self.update_circle_colors()

    def update_circle_colors(self):
        for circle, cx, cy, row, col in self.circles:
            if (row, col) in self.selected_circle_positions:
                self.protocol_select_sample_canvas.itemconfig(circle, fill='skyblue', outline='black')
            else:
                self.protocol_select_sample_canvas.itemconfig(circle, fill='grey', outline='black')


    def confirm_sample_selection(self):
        print("Saving selected circle positions:", self.selected_circle_positions)
        self.protocol_select_sample_window.destroy()

if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = StageControlModule(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()