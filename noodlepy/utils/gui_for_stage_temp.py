import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tkinter import StringVar

class StageControlGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")

        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=5, pady=5, fill=BOTH, expand=YES)

        # Top frame for connection and home buttons
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(side=TOP, fill=X, expand=YES)

        # Connection status button
        self.connect_button = ttk.Button(top_frame, text="Connect", command=self.connect_device, bootstyle="secondary")
        self.connect_button.pack(side=LEFT, fill=Y, padx=5, pady=5)
        self.connect_button.config(width=8)

        # Reference buttons frame
        reference_frame = ttk.Labelframe(top_frame, text="Reference", padding=5)
        reference_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)

        ttk.Button(reference_frame, text="Ref X", command=lambda: self.send_gcode("Ref X"), bootstyle="secondary").grid(row=0, column=0, padx=5)
        ttk.Button(reference_frame, text="Ref Y", command=lambda: self.send_gcode("Ref Y"), bootstyle="secondary").grid(row=0, column=1, padx=5)
        ttk.Button(reference_frame, text="Ref Z", command=lambda: self.send_gcode("Ref Z"), bootstyle="secondary").grid(row=0, column=2, padx=5)
        ttk.Button(reference_frame, text="Ref All", command=lambda: self.send_gcode("Ref ALL"), bootstyle="warning").grid(row=0, column=3, padx=5)

        # set all button sizes to 6
        for child in reference_frame.winfo_children():
            child.config(width=6)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text="Move", padding=5)
        movement_frame.pack(side=TOP, fill=BOTH, expand=YES, padx=5, pady=5)

        ttk.Label(movement_frame, text="X").grid(row=0, column=1)
        ttk.Label(movement_frame, text="Y").grid(row=0, column=2)
        ttk.Label(movement_frame, text="Z").grid(row=0, column=3)

        # Labels for current position
        ttk.Label(movement_frame, text="Current").grid(row=1, column=0, padx=5, pady=5)
        self.p1_x_entry = ttk.Label(movement_frame, text="0")
        self.p1_y_entry = ttk.Label(movement_frame, text="0")
        self.p1_z_entry = ttk.Label(movement_frame, text="0")
        self.p1_x_entry.grid(row=1, column=1, padx=5, pady=5)
        self.p1_y_entry.grid(row=1, column=2, padx=5, pady=5)
        self.p1_z_entry.grid(row=1, column=3, padx=5, pady=5)

        # Labels and entries for a position to move to
        ttk.Label(movement_frame, text="Move To").grid(row=2, column=0, padx=5, pady=5)
        self.p2_x_entry = ttk.Entry(movement_frame, width=5)
        self.p2_y_entry = ttk.Entry(movement_frame, width=5)
        self.p2_z_entry = ttk.Entry(movement_frame, width=5)
        self.p2_x_entry.grid(row=2, column=1, padx=5, pady=5)
        self.p2_y_entry.grid(row=2, column=2, padx=5, pady=5)
        self.p2_z_entry.grid(row=2, column=3, padx=5, pady=5)

        # Go button
        self.go_button = ttk.Button(movement_frame, text="Go", command=self.move_to_coordinates, bootstyle="success")
        self.go_button.grid(row=2, column=4, padx=10, pady=5)
        self.go_button.config(width=6)

        # Speed slider
        self.slider_value = StringVar()
        ttk.Label(movement_frame, text="Speed").grid(row=3, column=0, padx=5, pady=5)
        ttk.Label(movement_frame, text="1 mm/s").grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(movement_frame, text="20 mm/s").grid(row=3, column=4, padx=5, pady=5)
        self.speed_slider = ttk.Scale(movement_frame, from_=1, to=100, orient=HORIZONTAL, bootstyle="success")
        self.speed_slider.grid(row=3, column=2, columnspan=2, pady=20)
        self.speed_slider.config(length=200)
        self.speed_slider.set(50)
        self.speed_slider.config()
        ttk.Label(movement_frame, textvariable=self.slider_value).grid(row=4, column=1, columnspan=3, pady=5)
        self.update_label(self.speed_slider.get())
        self.speed_slider.bind("<ButtonRelease-1>", lambda e: self.update_label(self.speed_slider.get()))

        # Load images for the direction buttons
        self.img_low_speed = Image.open("/Users/yifeigu/Downloads/chevron-right_116844.png")
        self.img_medium_speed = Image.open("/Users/yifeigu/Downloads/double-left-chevron_icon-icons.com_73595.png")
        self.img_high_speed = Image.open("/Users/yifeigu/Downloads/chevron_triple_right_icon_137766.png")

        # set the size of the images
        self.img_low_speed = self.img_low_speed.resize((20, 20))
        self.img_medium_speed = self.img_medium_speed.resize((20, 20))
        self.img_high_speed = self.img_high_speed.resize((20, 20))
        # rotate the images
        self.img_up_low = ImageTk.PhotoImage(self.img_low_speed.rotate(90))
        self.img_up_medium = ImageTk.PhotoImage(self.img_medium_speed.rotate(-90))
        self.img_up_high = ImageTk.PhotoImage(self.img_high_speed.rotate(90))
        self.img_down_low = ImageTk.PhotoImage(self.img_low_speed.rotate(-90))
        self.img_down_medium = ImageTk.PhotoImage(self.img_medium_speed.rotate(90))
        self.img_down_high = ImageTk.PhotoImage(self.img_high_speed.rotate(-90))
        self.img_left_low = ImageTk.PhotoImage(self.img_low_speed.rotate(180))
        self.img_left_medium = ImageTk.PhotoImage(self.img_medium_speed)
        self.img_left_high = ImageTk.PhotoImage(self.img_high_speed.rotate(180))
        self.img_right_low = ImageTk.PhotoImage(self.img_low_speed)
        self.img_right_medium = ImageTk.PhotoImage(self.img_medium_speed.rotate(180))
        self.img_right_high = ImageTk.PhotoImage(self.img_high_speed)

        # Frame for direction buttons
        direction_frame = ttk.Frame(movement_frame, padding=3)
        direction_frame.grid(row=4, column=0, columnspan=10, pady=3, padx=3)

        # X, Y axis Direction labels
        ttk.Label(direction_frame, text="Y").grid(row=0, column=4, pady=3)
        ttk.Label(direction_frame, text="X").grid(row=4, column=0, padx=3)

        # X, Y axis Direction buttons
        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.send_gcode("UP SLOW"), bootstyle="light").grid(row=3, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.send_gcode("UP MEDIUM"), bootstyle="secondary").grid(row=2, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.send_gcode("UP FAST"), bootstyle="dark").grid(row=1, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_left_low, command=lambda: self.send_gcode("LEFT SLOW"), bootstyle="light").grid(row=4, column=3, padx=3)
        ttk.Button(direction_frame, image=self.img_left_medium, command=lambda: self.send_gcode("LEFT MEDIUM"), bootstyle="secondary").grid(row=4, column=2, padx=3)
        ttk.Button(direction_frame, image=self.img_left_high, command=lambda: self.send_gcode("LEFT FAST"), bootstyle="dark").grid(row=4, column=1, padx=3)
        ttk.Button(direction_frame, image=self.img_right_low, command=lambda: self.send_gcode("RIGHT SLOW"),bootstyle="light").grid(row=4, column=5, padx=3)
        ttk.Button(direction_frame, image=self.img_right_medium, command=lambda: self.send_gcode("RIGHT MEDIUM"), bootstyle="secondary").grid(row=4, column=6, padx=3)
        ttk.Button(direction_frame, image=self.img_right_high, command=lambda: self.send_gcode("RIGHT FAST"), bootstyle="dark").grid(row=4, column=7, padx=3, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.send_gcode("DOWN SLOW"), bootstyle="light").grid(row=5, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.send_gcode("DOWN MEDIUM"), bootstyle="secondary").grid(row=6, column=4, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.send_gcode("DOWN FAST"), bootstyle="dark").grid(row=7, column=4, pady=3)

        # label for step size with bold font

        ttk.Label(direction_frame, text="10").grid(row=3, column=1, pady=3)
        ttk.Label(direction_frame, text="1").grid(row=3, column=2, pady=3)
        ttk.Label(direction_frame, text="0.1").grid(row=3, column=3,  pady=3)

        # Add an empty label for spacing between x, y and z buttons
        ttk.Label(direction_frame, text="").grid(row=2, column=9,columnspan=2, padx=50)
        
        # Z-axis control labels
        ttk.Label(direction_frame, text="Z (up)").grid(row=0, column=10, pady=3)
        ttk.Label(direction_frame, text="Z (down)").grid(row=8, column=10, pady=3)

        # Z-axis control buttons
        ttk.Button(direction_frame, image=self.img_up_low, command=lambda: self.send_gcode("MOVE Z UP"), bootstyle="light").grid(row=3, column=10, columnspan=2)
        ttk.Button(direction_frame, image=self.img_up_medium, command=lambda: self.send_gcode("MEDIUM STEP Z UP"), bootstyle="secondary").grid(row=2, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_up_high, command=lambda: self.send_gcode("BIG STEP Z UP"), bootstyle="dark").grid(row=1, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_low, command=lambda: self.send_gcode("MOVE Z DOWN"), bootstyle="light").grid(row=5, column=10, columnspan=2)
        ttk.Button(direction_frame, image=self.img_down_medium, command=lambda: self.send_gcode("MEDIUM STEP Z DOWN"), bootstyle="secondary").grid(row=6, column=10, columnspan=2, pady=3)
        ttk.Button(direction_frame, image=self.img_down_high, command=lambda: self.send_gcode("BIG STEP Z DOWN"), bootstyle="dark").grid(row=7, column=10, columnspan=2, pady=3)

        # Frame for register
        register_frame = ttk.Labelframe(main_frame, text="Register", padding=5)
        register_frame.pack(side=TOP, fill=BOTH, expand=YES, padx=5, pady=5)

        # Register labels and entries
        self.register_first_smaple_button = ttk.Button(register_frame, text="Register Current Position as First Sample", command = self.register_first_smaple, bootstyle ="success")
        self.register_first_smaple_button.grid(row=0, column=1, padx=5, pady=5)
        self.register_lowest_point_button = ttk.Button(register_frame, text="Register Current Z as the Lowest Point", command=self.register_lowest_point, bootstyle = "success")
        self.register_lowest_point_button.grid(row=1, column=1, padx=5, pady=5)

        # configure the size of the buttons
        for child in register_frame.winfo_children():
            child.config(width=30)

        # Remove button
        self.remove_first_sample_button = ttk.Button(register_frame, text="Remove", command= self.remove_first_sample_registration,bootstyle="danger", state=DISABLED)
        self.remove_first_sample_button.grid(row=0, column=2, padx=5, pady=5)
        self.remove_lowest_point_button = ttk.Button(register_frame, text="Remove", command= self.remove_lowest_point_registration,bootstyle="danger", state=DISABLED)
        self.remove_lowest_point_button.grid(row=1, column=2, padx=5, pady=5)

    def move_to_coordinates(self):
        p2_x = self.p2_x_entry.get()
        p2_y = self.p2_y_entry.get()
        p2_z = self.p2_z_entry.get()

        if p2_x and p2_y and p2_z:
            gcode = f"G0 X{p2_x} Y{p2_y} Z{p2_z}"
            self.send_gcode(gcode)

    def send_gcode(self, gcode):
        # Implement the code to send G-code to your device here.
        print(f"Sending G-code: {gcode}")
        # For example, you might use serial communication:
        # ser.write((gcode + '\n').encode())

    def update_speed(self):
        speed_x = self.speed_x_entry.get()
        speed_y = self.speed_y_entry.get()
        speed_z = self.speed_z_entry.get()

        if speed_x and speed_y and speed_z:
            gcode = f"SET SPEED X{speed_x} Y{speed_y} Z{speed_z}"
            self.send_gcode(gcode)

    def connect_device(self):
        self.connect_button.configure(text="Disconnect", command=self.disconnect_device)
        self.connect_button.configure(bootstyle="success")

    def disconnect_device(self):
        self.connect_button.configure(text="Connect", command=self.connect_device)
        # change the color of the connect button to primary
        self.connect_button.configure(bootstyle="secondary")

    def register_first_smaple(self):
        # change the text of the button
        self.register_first_smaple_button.configure(text="First Sample Registered")
        # change the color of the button
        self.register_first_smaple_button.configure(bootstyle="success")
        self.register_first_smaple_button.configure(state=DISABLED)
        # enable the remove button
        self.remove_first_sample_button.configure(state=NORMAL)
        print("First sample registered")

    def register_lowest_point(self):
        self.register_lowest_point_button.configure(text="Lowest Z Point registered")
        self.register_lowest_point_button.configure(bootstyle="success")
        self.register_lowest_point_button.configure(state=DISABLED)
        self.remove_lowest_point_button.configure(state=NORMAL)
        print("Lowest Z registered")


    def remove_first_sample_registration(self):
        self.remove_first_sample_button.configure(state=DISABLED)
        self.register_first_smaple_button.configure(text="Register Current Position as First Sample")
        self.register_first_smaple_button.configure(state=NORMAL)
        self.register_first_smaple_button.configure(bootstyle="success")
        print("Registration of the first sample removed")

    def remove_lowest_point_registration(self):
        print("Registration of the lowest Z removed")
        self.remove_lowest_point_button.configure(state=DISABLED)
        self.register_lowest_point_button.configure(text ='Register Current Z as the Lowest Point', state = NORMAL, bootstyle = 'success')

    # Function to update the StringVar with the slider's value
    def update_label(self, value):
        self.slider_value.set(f"{float(value):.2f} mm/s")

if __name__ == "__main__":
    app = StageControlGUI()
    app.mainloop()
