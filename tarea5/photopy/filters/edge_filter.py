import cv2
import numpy as np
import tkinter as tk
from typing import Any
from .base_filter import ImageFilter

class EdgeFilter(ImageFilter):
    
    def __init__(self):
        super().__init__()
        self._params = {
            'intensity': 0,
            'axis_x': True,
            'axis_y': True
        }
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return image
            
        intensity = self._params['intensity']
        if intensity == 0:
            return image
            
        axis_x = self._params['axis_x']
        axis_y = self._params['axis_y']
        
        if not (axis_x or axis_y):
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) if axis_x else 0
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) if axis_y else 0
        
        edges = np.abs(sobel_x) + np.abs(sobel_y)
        
        factor = intensity / 50.0
        edges = np.clip(edges * factor, 0, 255).astype(np.uint8)
        
        edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        result = cv2.addWeighted(image, 0.7, edges_3c, 0.3, 0)
        return result
    
    def get_control_widget(self, parent, callback) -> Any:
        frame = tk.LabelFrame(parent, text="Detección de Bordes", padx=5, pady=5)
        
        intensity_frame = tk.Frame(frame)
        intensity_frame.pack(fill=tk.X, pady=2)
        tk.Label(intensity_frame, text="Intensidad:").pack(side=tk.LEFT)
        slider = tk.Scale(intensity_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                         command=lambda v: self._update_param('intensity', int(v), callback))
        slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        axis_frame = tk.Frame(frame)
        axis_frame.pack(pady=5)
        
        var_x = tk.BooleanVar(value=self._params['axis_x'])
        var_y = tk.BooleanVar(value=self._params['axis_y'])
        
        chk_x = tk.Checkbutton(axis_frame, text="Eje X", variable=var_x,
                              command=lambda: self._update_param('axis_x', var_x.get(), callback))
        chk_x.pack(side=tk.LEFT, padx=5)
        
        chk_y = tk.Checkbutton(axis_frame, text="Eje Y", variable=var_y,
                              command=lambda: self._update_param('axis_y', var_y.get(), callback))
        chk_y.pack(side=tk.LEFT, padx=5)
        
        self._control_widgets = {'slider': slider, 'var_x': var_x, 'var_y': var_y}
        return frame
    
    def _update_param(self, key: str, value, callback):
        self._params[key] = value
        callback()
    
    def reset(self):
        self._params = {'intensity': 0, 'axis_x': True, 'axis_y': True}
        if hasattr(self, '_control_widgets'):
            self._control_widgets['slider'].set(0)
            self._control_widgets['var_x'].set(True)
            self._control_widgets['var_y'].set(True)