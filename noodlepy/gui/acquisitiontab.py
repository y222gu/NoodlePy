import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
import tkinter as tk
from noodlepy.gui.liveviewmodule import LiveViewModule
from noodlepy.gui.stagecontrolmodule import StageControlModule
from noodlepy.gui.protocolmodule import ProtocolModule
from noodlepy.gui.wasatch_autofocus_module import WasatchAutofocusModule
from noodlepy.gui.droppositionmodule import DropPositionModule

class AcquisitionGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):

        first_column = ttk.Frame(self)
        first_column.grid(row=0, column=0, sticky="nsew")

        middle_column = ttk.Frame(self)
        middle_column.grid(row=0, column=1, rowspan=3, sticky="nsew")
        
        last_column = ttk.Frame(self)
        last_column.grid(row=0, column=2, sticky="nsew")

        live_view_frame = LiveViewModule(first_column)
        live_view_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")

        stage_control_frame = StageControlModule(middle_column)
        stage_control_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        live_view_frame.add_subscriber('switch_view', stage_control_frame, stage_control_frame.handle_switch_view)
        live_view_frame.add_subscriber('test_sampling_points', stage_control_frame, callback = stage_control_frame.handle_test_sampling_points)
        live_view_frame.add_subscriber('update_captured_frame_center', stage_control_frame, callback = stage_control_frame.handle_updated_captured_frame_center)

        wasatch_autofocus_frame = WasatchAutofocusModule(last_column)
        wasatch_autofocus_frame.grid(row=0, column=0, sticky="nsew")
        wasatch_autofocus_frame.add_subscriber('update_nanodrive_position', stage_control_frame, callback = stage_control_frame.handle_update_nanodrive_position)
        stage_control_frame.add_subscriber('move_nanodrive_by', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_move_nanodrive_by_request)
        stage_control_frame.add_subscriber('move_nanodrive_to', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_move_nanodrive_to_request)

        # sample_grid_frame = DropPositionModule(last_column)
        # sample_grid_frame.grid(row=1, column=0, sticky="nsew")
        # sample_grid_frame.add_subscriber('test_sample_grid', stage_control_frame, callback = stage_control_frame.handle_test_sample_grid)


        # acquisition_protocol_frame = ProtocolModule(last_column)
        # acquisition_protocol_frame.grid(row=1, column=0, sticky="nsew")


if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = AcquisitionGUI(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
