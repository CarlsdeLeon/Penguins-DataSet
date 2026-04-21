import numpy as np
import tkinter as tk
from typing import Any
from .base_filter import ImageFilter

class RGBFilter(ImageFilter):
    
    def __init__(self):
        super().__init__()
        self._params = {'R': 0, 'G': 0, 'B': 0}
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return image
            
        result = image.astype(np.int16)
        result[:, :, 2] += self._params['R']
        result[:, :, 1] += self._params['G']
        result[:, :, 0] += self._params['B']
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def get_control_widget(self, parent, callback) -> Any:
        frame = tk.LabelFrame(parent, text="Ajuste RGB", padx=5, pady=5)
        
        sliders = {}
        colores = [("R", 'R'), ("G", 'G'), ("B", 'B')]
        
        for nombre, key in colores:
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(row, text=f"{nombre}:").pack(side=tk.LEFT)
            slider = tk.Scale(row, from_=-100, to=100, orient=tk.HORIZONTAL,
                             command=lambda v, k=key: self._update_param(k, int(v), callback))
            slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
            sliders[key] = slider
        
        self._control_widgets = sliders
        return frame
    
    def _update_param(self, key: str, value: int, callback):
        self._params[key] = value
        callback()
    
    def reset(self):
        for key in self._params:
            self._params[key] = 0
        if hasattr(self, '_control_widgets'):
            for key, slider in self._control_widgets.items():
                slider.set(0)