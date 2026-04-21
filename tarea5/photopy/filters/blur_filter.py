import cv2
import numpy as np
import tkinter as tk
from typing import Any
from .base_filter import ImageFilter

class BlurFilter(ImageFilter):
    
    def __init__(self):
        super().__init__()
        self._params = {'kernel_size': 1}
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return image
            
        k = self._params['kernel_size']
        if k <= 1:
            return image
            
        k = k if k % 2 else k + 1
        return cv2.GaussianBlur(image, (k, k), 0)
    
    def get_control_widget(self, parent, callback) -> Any:
        frame = tk.LabelFrame(parent, text="Desenfoque", padx=5, pady=5)
        
        slider = tk.Scale(frame, from_=1, to=25, orient=tk.HORIZONTAL,
                         command=lambda v: self._update_param(int(v), callback))
        slider.pack(fill=tk.X)
        slider.set(1)
        
        self._control_slider = slider
        return frame
    
    def _update_param(self, value: int, callback):
        self._params['kernel_size'] = value
        callback()
    
    def reset(self):
        self._params['kernel_size'] = 1
        if hasattr(self, '_control_slider'):
            self._control_slider.set(1)