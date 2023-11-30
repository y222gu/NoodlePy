import tkinter as tk
from tkinter import ttk, filedialog
import configparser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ConfigInputApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Configuration File Analysis")

        # Create and set a custom style for the UI
        self.style = ttk.Style()

        # Configure the overall style
        self.style.configure(
            "TFrame",
            background="#F0F0F0"
        )

        # Configure the style for labels
        self.style.configure(
            "TLabel",
            font=('Helvetica', 12),
            background="#F0F0F0"
        )

        # Configure the style for buttons
        self.style.configure(
            "TButton",
            font=('Helvetica', 12, 'bold'),
            foreground="#FFFFFF",
            background="#4CAF50",
            padding=(10, 10, 10, 10)
        )

        # Configure the style for entry widgets
        self.style.configure(
            "TEntry",
            font=('Helvetica', 12),
            padding=(10, 10, 10, 10)
        )

        # Create a frame with a light gray background
        frame = ttk.Frame(self.master, style="TFrame")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Allow resizing of the frame
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # Create a text entry widget
        self.config_entry = ttk.Entry(frame, width=50, style="TEntry")
        self.config_entry.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky="ew")

        # Create a button to browse for the configuration file
        self.browse_button = ttk.Button(frame, text="Browse", command=self.browse_file, style="TButton")
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Create a button to submit the configuration file
        self.submit_button = ttk.Button(frame, text="Submit", command=self.submit_config, style="TButton")
        self.submit_button.grid(row=1, column=0, columnspan=3, pady=10)

        # Dictionary to store configuration parameters
        self.config_params = {}

        # Create a figure for the plot
        self.figure, self.ax = plt.subplots(figsize=(6, 4), dpi=100)

        # Create a canvas to embed the plot in the tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    def browse_file(self):
        # Open a file dialog to select the configuration file
        file_path = filedialog.askopenfilename(filetypes=[("Config files", "*.config"), ("All files", "*.*")])

        # Update the entry widget with the selected file path
        self.config_entry.delete(0, tk.END)
        self.config_entry.insert(0, file_path)

    def submit_config(self):
        # Get the configuration file path from the entry widget
        config_path = self.config_entry.get()

        # Read the configuration file and store parameters in the dictionary
        self.config_params = self.read_config_file(config_path)

        # Plot the data based on the configuration parameters
        self.plot_data()

    def read_config_file(self, file_path):
        # Read the configuration file using configparser
        config = configparser.ConfigParser()
        config.read(file_path)

        # Extract parameters from the configuration file
        params = {}
        for section in config.sections():
            params[section] = dict(config.items(section))

        return params

    def plot_data(self):
        # Clear the previous plot
        self.ax.clear()

        # Dummy data for illustration (replace with your actual data)
        x = [1, 2, 3, 4, 5]
        y = [float(self.config_params['data']['param1']),
             float(self.config_params['data']['param2']),
             float(self.config_params['data']['param3']),
             float(self.config_params['data']['param4']),
             float(self.config_params['data']['param5'])]

        # Plot the data
        self.ax.plot(x, y, marker='o', linestyle='-', color='b', label='Data')

        # Set labels and title
        self.ax.set_xlabel('X-axis Label')
        self.ax.set_ylabel('Y-axis Label')
        self.ax.set_title('Data Plot')

        # Add legend
        self.ax.legend()

        # Redraw the canvas
        self.canvas.draw()

# Create the main application window
root = tk.Tk()

# Create an instance of the ConfigInputApp class
app = ConfigInputApp(root)

# Run the application
root.mainloop()
