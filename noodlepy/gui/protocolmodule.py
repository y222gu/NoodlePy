import threading
import time
from tkinter import Tk, Button, Label, Entry, StringVar, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from noodlepy.gui.publisher_subscriber import Publisher, Subscriber
import os
from concurrent.futures import ThreadPoolExecutor
import queue

# Protocol Manager
class ProtocolModule(Publisher, ttk.Frame):
    def __init__(self, parent):
        Publisher.__init__(self, ["update_nanodrive_position",
                                  "select_sample_for_protocol",
                                  "check_current_camera_view",
                                  "move_stage_to_target_sample_drop_during_aquisition",
                                  "focus_widefield_camera",
                                  "focus_objective_camera",
                                  "capture_current_image_and_detect_sample_drop_and_create_sampling_points",
                                  "call_switch_view_button_in_live_camera_module",
                                  "move_to_a_single_sampling_point",
                                  "focus_wasatch",
                                  "measure_spectra_and_save_to_specific_folder",
                                  "reposition_stage_and_nanodrive_in_objective_view"])
        ttk.Frame.__init__(self, parent)

        self.name = "ProtocolModule"
        self.parent = parent
        self.focus_range_is_setup = False
        self.stage_reference_is_setup = False

        self.state = "IDLE"   # The initial state
        self.task_queue = queue.Queue()  # Queue for tasks
        self.current_task = None  # Keep track of the current task
        self.abort_flag = False  # Flag to abort the current task

        self.protocol_canvas_circle_spacing = 80  # Enlarge the spacing for better visualization
        self.protocol_canvas_circle_radius = 30 # Enlarge the circle radius for better visualization
        self.protocol_canvas_grid_pattern = [5, 5, 5, 5]  # Column-wise circle counts for symmetry
        # calculate the canvas size based on the circle spacing and radius
        self.protocol_canvas_width = len(self.protocol_canvas_grid_pattern) * self.protocol_canvas_circle_spacing
        self.protocol_canvas_height = max(self.protocol_canvas_grid_pattern) * self.protocol_canvas_circle_spacing
        self.selected_circles = set()  # Track selected circles
        self.selected_circle_positions = []  # Track selected circle positions
        self.sampling_points_relative_distance_to_camera_center = None

        self.create_widgets()
        
        # Periodically check the state machine
        self.check_state_machine()

    def check_state_machine(self):
        if self.state == "IDLE":
            if not self.task_queue.empty() and not self.abort_flag:
                self.current_task = self.task_queue.get()
                self.state = "BUSY"
                self.execute_task()

        elif self.state == "BUSY":
            # Wait for task completion; the task will update the state itself.
            pass

        elif self.state == "TASK_COMPLETED":
            self.state = "IDLE"  # Otherwise, continue to the next task

        elif self.state == "ABORT":
            print("Acquisition aborted due to an error or condition.")
            self.abort_flag = True
            self.task_queue.queue.clear()  # Clear remaining tasks to stop the process
            self.state = "IDLE"  # Reset state to IDLE

        # Continue checking the state machine after a short delay (50 ms)
        self.after(50, self.check_state_machine)

    def execute_task(self):
        event, args = self.current_task
        self.dispatch(event, *args)

    def enqueue_task(self, event, *args):
        self.task_queue.put((event, args))

    def clear_task_queue(self):
        """Clear the task queue and reset state."""
        with self.task_queue.mutex:
            self.task_queue.queue.clear()
        self.current_task = None
        self.abort_flag = False
        print("All tasks cleared.")

    def abort_execution(self):
        """Abort the execution of the current task and clear the queue."""
        self.abort_flag = True
        print("Abort flag set. Tasks will be stopped.")

    def start_aquisition(self):
        print("Checking setup status")
        if not self.focus_range_is_setup:
            print("Focus range not setup")
            return
        if not self.stage_reference_is_setup:
            print("Stage not referenced")
            return
        
        print("Executing protocol")
        self.current_task_generator = self.task_sequence_generator(self.selected_circle_positions)
        self.enqueue_next_task()

    def enqueue_next_task(self):
        try:
            next_task = next(self.current_task_generator)
            self.enqueue_task(*next_task)
        except StopIteration:
            print("All tasks for the current sample drop are completed.")
            self.current_task_generator = None

    def task_sequence_generator(self, selected_circle_positions):
        for sample_drop_row_column in selected_circle_positions:
            print('Measuring the sample drop at row', sample_drop_row_column[0], 'column', sample_drop_row_column[1])
            yield ('check_current_camera_view', "WIDEFIELD")
            yield ("move_stage_to_target_sample_drop_during_aquisition", sample_drop_row_column)

            if self.autofocus_widefield_var.get() == "True":
                yield ("focus_widefield_camera",)

            yield ("capture_current_image_and_detect_sample_drop_and_create_sampling_points",)
            yield ("call_switch_view_button_in_live_camera_module", "TO_OBJECTIVE")
            yield ("check_current_camera_view", "OBJECTIVE")

            print("Sampling points relative to camera center:", self.sampling_points_relative_distance_to_camera_center)
            for i, point in enumerate(self.sampling_points_relative_distance_to_camera_center):
                print(f"Measuring #{i} point at position {point}")
                yield ("move_to_a_single_sampling_point", 'OBJECTIVE', point)

                if self.autofocus_objective_var.get() == "True":
                    yield ("focus_objective_camera",)

                if self.autofocus_wasatch_var.get() == "True":
                    yield ("focus_wasatch",)

                subfolder = f"sample_drop_row{sample_drop_row_column[0]}_column{sample_drop_row_column[1]}"
                folder = os.path.join(self.folder_path.get(), subfolder)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                filename = f"{self.file_base_name.get()}_sample_drop_row{sample_drop_row_column[0]}_column{sample_drop_row_column[1]}_point_x{point[0]}_y{point[1]}"
                yield ("measure_spectra_and_save_to_specific_folder", self.number_of_samples.get(), folder, filename)
                print(f"Data saved to {folder}/{filename}")

            print("All points for the current sample drop are completed.")
            yield ("reposition_stage_and_nanodrive_in_objective_view",)
            yield ("call_switch_view_button_in_live_camera_module", "TO_WIDE")
            yield ("check_current_camera_view", "WIDEFIELD")

    def handle_task_completed(self):
        self.state = "TASK_COMPLETED"
        self.enqueue_next_task()

    def handle_abort_aquisition(self):
        self.state = "ABORT"

    def handle_update_sampling_points_to_protocol_module(self, sampling_points_relative_distance_to_camera_center):
        print('Handle_update_sampling_points_to_protocol_module is running')
        self.sampling_points_relative_distance_to_camera_center = sampling_points_relative_distance_to_camera_center

    def create_widgets(self):
        protocol_frame = ttk.Labelframe(self, text="Aquisition", padding=5)
        protocol_frame.grid(row=0, column=0, columnspan=3, sticky="nsew",padx=5, pady=5)

        # file name entry for saving the data
        self.save_data = ttk.Frame(protocol_frame, padding=5)
        self.save_data.grid(row=0, column=0,columnspan=3, padx=5, pady=5, sticky="nsew")

        # file path entry for saving the data
        default_path = os.path.join(os.path.expanduser("~"), "Documents", "RamanData")
        self.folder_path = StringVar()
        self.folder_path.set(default_path)
        self.folder_path_label = ttk.Label(self.save_data, text="Folder Path:").grid(row=0, column=0, padx=5, pady=5)
        self.folder_path_entry = ttk.Entry(self.save_data, textvariable=self.folder_path, width = 62).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")     
        self.folder_path_button = ttk.Button(self.save_data, text="Browse", command=self.browse_folder, bootstyle ='info').grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.file_base_name = StringVar()
        self.file_base_name.set("date_patientOD_sampleType")
        self.file_base_name_label = ttk.Label(self.save_data, text="File Name:").grid(row=1, column=0, padx=5, pady=5)
        self.file_base_name_entry = ttk.Entry(self.save_data, textvariable=self.file_base_name, width = 62).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # set protocol button
        self.select_sample_drop_button = ttk.Button(protocol_frame, text="Set Protocol", command=self.create_protocol_window, bootstyle = 'info')
        self.select_sample_drop_button.grid(row=1, column=0,  padx=5, pady=5, sticky="nsew")

        # button to start the protocol
        self.start_protocol_button = ttk.Button(protocol_frame, text="Start Aquisition", command=self.start_aquisition, bootstyle = 'info')
        self.start_protocol_button.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="nsew")

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        self.folder_path.set(folder_path)

    def handle_update_setup_status(self, status):
        if status == "focus_range_is_setup":
            self.focus_range_is_setup = True
        elif status == "stage_reference_is_setup":
            self.stage_reference_is_setup = True

        if self.focus_range_is_setup and self.stage_reference_is_setup:
            self.start_protocol_button.configure(state=NORMAL)

        return None

    def create_protocol_window(self):
        # Check if the window is already open
        if hasattr(self, 'set_protocol_window') and self.set_protocol_window.winfo_exists():
            print("The sample selection window is already open.")
            self.set_protocol_window.focus_force()
            return

        # For sample drop selection
        self.set_protocol_window = ttk.Toplevel(self.parent)
        self.set_protocol_window.title("Set Protocol")
        self.set_protocol_window.resizable(False, False)
        self.set_protocol_window.focus_force()
        self.set_protocol = ttk.Frame(self.set_protocol_window)
        self.set_protocol.grid(row=0, column=0, padx=10, pady=5)

        # canvas for selecting the sample drops
        self.select_sample_drop_frame = ttk.Labelframe(self.set_protocol, text="Select Sample Drops", padding=5)
        self.select_sample_drop_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
        self.protocol_select_sample_canvas = ttk.Canvas(self.select_sample_drop_frame, width=self.protocol_canvas_width, height=self.protocol_canvas_height)
        self.protocol_select_sample_canvas.grid(row=0, column=0,columnspan=2, padx=40, pady=5, sticky='nsew')
        self.protocol_select_sample_create_circle_grid()
        
        select_all_button = ttk.Button(self.select_sample_drop_frame, text="Select All", command=self.protocol_select_all_samples, bootstyle='info_outline')
        select_all_button.grid(row=1, column=0, pady=10, padx=10, sticky='ew')
        
        cancel_all_button = ttk.Button(self.select_sample_drop_frame, text="Cancel All", command=self.protocol_cancel_all_samples, bootstyle='danger-outline')
        cancel_all_button.grid(row=1, column=1, pady=10, padx=10, sticky='ew')
        
        # Configure grid to center align the canvas and buttons
        self.select_sample_drop_frame.grid_columnconfigure(0, weight=1)
        self.select_sample_drop_frame.grid_columnconfigure(1, weight=1)
        self.select_sample_drop_frame.grid_rowconfigure(0, weight=1)
        self.select_sample_drop_frame.grid_rowconfigure(1, weight=1)

        # entry for the number of reps
        self.number_of_samples = StringVar()
        self.number_of_samples.set("1")
        self.number_of_samples_label = ttk.Label(self.set_protocol, text="Number of repetitions per point per sample drop:").grid(row=1, column=0, padx=5, pady=5)
        self.number_of_samples_entry = ttk.Entry(self.set_protocol, textvariable=self.number_of_samples, width = 5).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # check box for autofocusing in widefield, objective, and wasatch
        self.autofocus_checkbuttons_frame = ttk.Frame(self.set_protocol)
        self.autofocus_checkbuttons_frame.grid(row=2, column=0, columnspan=3, pady=10, padx=10, sticky='ew')
        self.autofocus_widefield_var = StringVar(value="True")
        self.autofocus_widefield = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Autofocus Widefield", variable=self.autofocus_widefield_var, onvalue="True", offvalue="False", bootstyle='info')
        self.autofocus_widefield.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.autofocus_objective_var = StringVar(value="False")
        self.autofocus_objective = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Autofocus Objective", variable=self.autofocus_objective_var, onvalue="True", offvalue="False", bootstyle='info')
        self.autofocus_objective.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.autofocus_wasatch_var = StringVar(value="True")
        self.autofocus_wasatch = ttk.Checkbutton(self.autofocus_checkbuttons_frame, text="Autofocus Wasatch", variable=self.autofocus_wasatch_var, onvalue="True", offvalue="False", bootstyle='info')
        self.autofocus_wasatch.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # button to confirm the selection
        self.ok_button = ttk.Button(self.set_protocol, text="OK", command=self.confirm_sample_selection, bootstyle='info')
        self.ok_button.grid(row=3, column=0, columnspan=3, pady=10, padx=10, sticky='ew')

        self.set_protocol_window.update_idletasks()
        self.set_protocol_window.geometry(f"{self.set_protocol_window.winfo_reqwidth()}x{self.set_protocol_window.winfo_reqheight()}")


    def protocol_select_sample_create_circle_grid(self):
        skip_positions = [(0, 0), (4, 0), (0, 3), (4, 3)]

        # Ensure `selected_circles` and `selected_circle_positions` exist and are persistent
        if not hasattr(self, 'selected_circles'):
            self.selected_circles = set()  # Track selected circles by their IDs
        if not hasattr(self, 'selected_circle_positions'):
            self.selected_circle_positions = []  # Track (row, column) positions of selected circles

        self.circles = []  # Store references to all circles

        for col_index, num_circles in enumerate(self.protocol_canvas_grid_pattern):
            vertical_offset = (max(self.protocol_canvas_grid_pattern) - num_circles) * self.protocol_canvas_circle_spacing // 2
            for row_index in range(num_circles):
                true_row_index = row_index + vertical_offset // self.protocol_canvas_circle_spacing

                # Skip the specified corner positions
                if (true_row_index, col_index) in skip_positions:
                    continue

                x = col_index * self.protocol_canvas_circle_spacing + self.protocol_canvas_circle_spacing // 2
                y = vertical_offset + row_index * self.protocol_canvas_circle_spacing + self.protocol_canvas_circle_spacing // 2

                # Determine the initial color of the circle based on its selection status
                if (true_row_index, col_index) in self.selected_circle_positions:
                    fill_color = 'skyblue'  # Selected
                else:
                    fill_color = 'grey'  # Not selected

                circle = self.protocol_select_sample_canvas.create_oval(
                    x - self.protocol_canvas_circle_radius, y - self.protocol_canvas_circle_radius,
                    x + self.protocol_canvas_circle_radius, y + self.protocol_canvas_circle_radius,
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
        self.set_protocol_window.destroy()


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    protocol_frame = ProtocolModule(root)
    protocol_frame.grid(row=0, column=0, sticky="nsew")
    root.mainloop()