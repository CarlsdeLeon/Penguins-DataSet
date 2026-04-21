import tkinter as tk
from typing import Callable, Dict

class RGBControls:
    
    def __init__(self, parent, callback: Callable):
        self.callback = callback
        self.sliders: Dict[str, tk.Scale] = {}
        self.setup_ui(parent)
    
    def setup_ui(self, parent):
        self.frame = tk.LabelFrame(parent, text="Ajuste RGB", padx=5, pady=5)
        
        colores = [("R", "R"), ("G", "G"), ("B", "B")]
        
        for nombre, key in colores:
            row = tk.Frame(self.frame)
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(row, text=f"{nombre}:").pack(side=tk.LEFT)
            slider = tk.Scale(row, from_=-100, to=100, orient=tk.HORIZONTAL,
                             command=lambda v, k=key: self._on_change(k, int(v)))
            slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.sliders[key] = slider
    
    def _on_change(self, key: str, value: int):
        if self.callback:
            self.callback(key, value)
    
    def get_value(self, key: str) -> int:        return self.sliders.get(key, tk.Scale()).get()
    
    def set_value(self, key: str, value: int):
        if key in self.sliders:
            self.sliders[key].set(value)
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)