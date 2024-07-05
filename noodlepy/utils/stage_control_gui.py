import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class StageControlGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="litera")

        self.title("Stage Control GUI")

        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=5, pady=5, fill=BOTH, expand=YES)

        # Top frame for connection and home buttons
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(side=TOP, fill=X, expand=YES)

        # Connection status button
        self.connect_button = ttk.Button(top_frame, text="Connect", command=self.connect_device, bootstyle="primary")
        self.connect_button.pack(side=LEFT, fill=Y, padx=5, pady=5)
        self.connect_button.config(width=8)


        # Home buttons frame
        home_frame = ttk.Labelframe(top_frame, text="Home", padding=5)
        home_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)
        
        ttk.Button(home_frame, text="Home X", command=lambda: self.send_gcode("HOME X"), bootstyle="warning").grid(row=0, column=0, padx=5)
        ttk.Button(home_frame, text="Home Y", command=lambda: self.send_gcode("HOME Y"), bootstyle="warning").grid(row=0, column=1, padx=5)
        ttk.Button(home_frame, text="Home Z", command=lambda: self.send_gcode("HOME Z"), bootstyle="warning").grid(row=0, column=2, padx=5)
        ttk.Button(home_frame, text="Home All", command=lambda: self.send_gcode("HOME ALL"), bootstyle="danger").grid(row=0, column=3, padx=5)

        # set all button sizes to 6
        for child in home_frame.winfo_children():
            child.config(width=6)

        # Frame for movement
        movement_frame = ttk.Labelframe(main_frame, text="Move", padding=5)
        movement_frame.pack(side=BOTTOM, fill=BOTH, expand=YES, padx=5, pady=5)

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

        # Labels and entries for speed
        ttk.Label(movement_frame, text="Speed mm/s").grid(row=3, column=0, padx=5, pady=5)
        self.speed_x_entry = ttk.Entry(movement_frame, width=5)
        self.speed_y_entry = ttk.Entry(movement_frame, width=5)
        self.speed_z_entry = ttk.Entry(movement_frame, width=5)
        self.speed_x_entry.grid(row=3, column=1, padx=5, pady=5)
        self.speed_y_entry.grid(row=3, column=2, padx=5, pady=5)
        self.speed_z_entry.grid(row=3, column=3, padx=5, pady=5)

        # Go button
        go_button = ttk.Button(movement_frame, text="Go", command=self.move_to_coordinates, bootstyle="success")
        go_button.grid(row=2, column=4, padx=10, pady=5)
        go_button.config(width=6)

        # update button
        update_button = ttk.Button(movement_frame, text="Update", command=self.update_speed, bootstyle="primary")
        update_button.grid(row=3, column=4, padx=10, pady=5)
        update_button.config(width=6)

        # Frame for direction buttons
        direction_frame = ttk.Frame(movement_frame, padding=5)
        direction_frame.grid(row=4, column=0, columnspan=9, pady=5, padx=5)

        # X, Y axis Direction buttons
        ttk.Button(direction_frame, text="↑", command=lambda: self.send_gcode("MOVE UP"), bootstyle="primary").grid(row=1, column=2)
        ttk.Button(direction_frame, text="↑", command=lambda: self.send_gcode("BIG STEP UP"), bootstyle="primary").grid(row=0, column=2, pady=5)
        ttk.Button(direction_frame, text="←", command=lambda: self.send_gcode("MOVE LEFT"), bootstyle="primary").grid(row=2, column=1)
        ttk.Button(direction_frame, text="→", command=lambda: self.send_gcode("MOVE RIGHT"), bootstyle="primary").grid(row=2, column=3)
        ttk.Button(direction_frame, text="↓", command=lambda: self.send_gcode("MOVE DOWN"), bootstyle="primary").grid(row=4, column=2)
        ttk.Button(direction_frame, text="↓", command=lambda: self.send_gcode("BIG STEP DOWN"), bootstyle="primary").grid(row=3, column=2, pady=5)
        ttk.Button(direction_frame, text="←", command=lambda: self.send_gcode("BIG STEP LEFT"), bootstyle="primary").grid(row=2, column=0, padx=5)
        ttk.Button(direction_frame, text="→", command=lambda: self.send_gcode("BIG STEP RIGHT"), bootstyle="primary").grid(row=2, column=4, padx=5)
        
        # Z-axis control buttons
        ttk.Button(direction_frame, text="↑", command=lambda: self.send_gcode("MOVE Z UP"), bootstyle="primary").grid(row=1, column=8, columnspan=2)
        ttk.Button(direction_frame, text="↑", command=lambda: self.send_gcode("BIG STEP Z UP"), bootstyle="primary").grid(row=0, column=8, columnspan=2, pady=5)
        ttk.Button(direction_frame, text="↓", command=lambda: self.send_gcode("MOVE Z DOWN"), bootstyle="primary").grid(row=3, column=8, columnspan=2)
        ttk.Button(direction_frame, text="↓", command=lambda: self.send_gcode("BIG STEP Z DOWN"), bootstyle="primary").grid(row=4, column=8, columnspan=2, pady=5)

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
        # Implement the code to connect to your device here.
        # For example, you might use serial communication:
        # ser = serial.Serial('COM3', 115200, timeout=1)
        # self.connection_status_label.configure(text="Connected")
        # change the text of the connection status label
        # self.connection_status_label.configure(text="Connected")
        # change the connect button to disconnect   
        self.connect_button.configure(text="Disconnect", command=self.disconnect_device)
        # change the color of the connect button to success
        self.connect_button.configure(bootstyle="success")
        
    def disconnect_device(self):
        # Implement the code to disconnect from your device here.
        # For example, you might use serial communication:
        # ser.close()
        # change the text of the connection status label
        # self.connection_status_label.configure(text="Not connected")
        # change the connect button to connect
        self.connect_button.configure(text="Connect", command=self.connect_device)
        # change the color of the connect button to primary
        self.connect_button.configure(bootstyle="primary")


if __name__ == "__main__":
    app = StageControlGUI()
    app.mainloop()
