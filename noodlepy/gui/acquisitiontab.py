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
        middle_column.grid(row=0, column=1, rowspan=4, sticky="nsew")
        
        last_column = ttk.Frame(self)
        last_column.grid(row=0, column=2, sticky="nsew")

        live_view_frame = LiveViewModule(first_column)
        live_view_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")

        stage_control_frame = StageControlModule(middle_column)
        stage_control_frame.grid(row=0, column=0, rowspan=3, columnspan=1, sticky="nsew")
        live_view_frame.add_subscriber('switch_view', stage_control_frame, stage_control_frame.handle_switch_view)
        live_view_frame.add_subscriber('test_all_sampling_points', stage_control_frame, callback = stage_control_frame.handle_test_sampling_points)
        live_view_frame.add_subscriber('update_captured_frame_center', stage_control_frame, callback = stage_control_frame.handle_updated_captured_frame_center)
        live_view_frame.add_subscriber('focus_widefield_camera', stage_control_frame, callback = stage_control_frame.handle_focus_widefield_camera)
        live_view_frame.add_subscriber('focus_objective_camera', stage_control_frame, callback = stage_control_frame.handle_focus_objective_camera)
        live_view_frame.add_subscriber('update_focus_score', stage_control_frame, callback = stage_control_frame.handle_update_focus_score)
        stage_control_frame.add_subscriber('calculate_focus_score', live_view_frame, callback = live_view_frame.handle_calculate_focus_score)
        stage_control_frame.add_subscriber('activate_camera_autofocus_button', live_view_frame, callback = live_view_frame.handle_activate_camera_autofocus_button)
        stage_control_frame.add_subscriber('activate_switch_view_button', live_view_frame, callback = live_view_frame.handle_activate_switch_view_button)

        wasatch_autofocus_frame = WasatchAutofocusModule(last_column)
        wasatch_autofocus_frame.grid(row=0, column=0, sticky="nsew")
        wasatch_autofocus_frame.add_subscriber('update_nanodrive_position', stage_control_frame, callback = stage_control_frame.handle_update_nanodrive_position)
        stage_control_frame.add_subscriber('move_nanodrive_by', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_move_nanodrive_by_request)
        stage_control_frame.add_subscriber('move_nanodrive_to', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_move_nanodrive_to_request)
        stage_control_frame.add_subscriber('get_nanodrive_position', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_get_nanodrive_position)
        stage_control_frame.add_subscriber('get_nanodrive_min_max', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_get_nanodrive_min_max)
        
        acquisition_protocol_frame = ProtocolModule(middle_column)
        acquisition_protocol_frame.grid(row=3, column=0, rowspan=1, columnspan=1, sticky="nsew")
        stage_control_frame.add_subscriber('activate_protocol_module_state', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_update_setup_status)
        stage_control_frame.add_subscriber('abort_aquisition', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_abort_aquisition)
        acquisition_protocol_frame.add_subscriber('move_stage_to_target_sample_drop_during_aquisition', stage_control_frame, callback = stage_control_frame.handle_move_stage_to_target_sample_drop_in_widefield_view_during_aquisition)
        acquisition_protocol_frame.add_subscriber('focus_widefield_camera', stage_control_frame, callback = stage_control_frame.handle_focus_widefield_camera_during_acquisition)
        acquisition_protocol_frame.add_subscriber('check_current_camera_view', live_view_frame, callback = live_view_frame.handle_check_current_camera_view)
        acquisition_protocol_frame.add_subscriber('capture_current_image_and_detect_sample_drop_and_create_sampling_points', live_view_frame, callback = live_view_frame.handling_create_sampling_points_during_aquisition)
        acquisition_protocol_frame.add_subscriber('call_switch_view_button_in_live_camera_module', live_view_frame, live_view_frame.handle_switch_view_during_aquisition)
        acquisition_protocol_frame.add_subscriber('move_to_a_single_sampling_point', stage_control_frame, callback = stage_control_frame.handle_move_to_a_single_sampling_point_during_acquisition)
        acquisition_protocol_frame.add_subscriber('focus_objective_camera', stage_control_frame, callback = stage_control_frame.handle_focus_objective_camera_during_acquisition)
        acquisition_protocol_frame.add_subscriber('focus_wasatch', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_autofocus_wasatch_during_aquisition)
        acquisition_protocol_frame.add_subscriber('measure_spectra_and_save_to_specific_folder', wasatch_autofocus_frame, callback = wasatch_autofocus_frame.handle_measure_spectra_and_save_to_specific_folder)
        acquisition_protocol_frame.add_subscriber('reposition_stage_and_nanodrive_in_objective_view', stage_control_frame, callback = stage_control_frame.handle_reposition_stage_and_nanodrive_in_objective_view_during_acquisition)
        stage_control_frame.add_subscriber('task_completed', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_task_completed)
        live_view_frame.add_subscriber('task_completed', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_task_completed)
        wasatch_autofocus_frame.add_subscriber('task_completed', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_task_completed)
        live_view_frame.add_subscriber('abort_aquisition', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_abort_aquisition)
        live_view_frame.add_subscriber('update_sampling_points_to_protocol_module', acquisition_protocol_frame, callback = acquisition_protocol_frame.handle_update_sampling_points_to_protocol_module)



if __name__ == '__main__':
    root = ttk.Window()
    root.style.theme_use('noodlepy')
    app = AcquisitionGUI(root)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
