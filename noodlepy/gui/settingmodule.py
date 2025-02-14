import yaml
import tkinter as tk
from tkinter import ttk


class ConfigManager:
    _instance = None

    def __new__(cls, config_file="config.yml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file):
        self.config_file = config_file
        with open(config_file, "r") as file:
            self.config_data = yaml.safe_load(file)

    def get(self, module_name, param_name):
        return self.config_data.get(module_name, {}).get(param_name)

    def update(self, module_name, param_name, value):
        if module_name in self.config_data:
            self.config_data[module_name][param_name] = value

    def save_config(self):
        with open(self.config_file, "w") as file:
            yaml.dump(self.config_data, file)