import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from filters.rgb_filter import RGBFilter
from filters.blur_filter import BlurFilter
from filters.edge_filter import EdgeFilter

class MainView:
    
    def __init__(self, controller):
        self.controller = controller
        self.controller.set_view(self)
        
        self.root = tk.Tk()
        self.root.title("Mini Photoshop")
        self.root.geometry("1200x700")
        
        self.panel = None
        self.filters = []
        self.current_image_tk = None
        self.setup_ui()
        
    def setup_ui(self):
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = tk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        image_frame = tk.LabelFrame(left_frame, text="Imagen", padx=5, pady=5, font=("Arial", 10, "bold"))
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.panel = tk.Label(image_frame, bg='#2c3e50')
        self.panel.pack(expand=True)
        
        selection_frame = tk.LabelFrame(left_frame, text="Selección de Área", padx=10, pady=10, font=("Arial", 10, "bold"))
        selection_frame.pack(fill=tk.X)
        
        self._add_selection_controls(selection_frame)
        
        right_frame = tk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))
        
        btn_cargar = tk.Button(right_frame, text="Cargar Imagen",
                              command=self.cargar_imagen, width=20, height=2,
                              bg="#3498db", fg="white")
        btn_cargar.pack(pady=5)
        
        filters_frame = tk.LabelFrame(right_frame, text="Filtros", padx=10, pady=5, font=("Arial", 10, "bold"))
        filters_frame.pack(fill=tk.X, pady=5)
        
        self._add_filters(filters_frame)
        
        rotation_frame = tk.LabelFrame(right_frame, text="Rotacion", padx=10, pady=5, font=("Arial", 10, "bold"))
        rotation_frame.pack(fill=tk.X, pady=5)
        
        self._add_rotation_controls(rotation_frame)
        
        transform_frame = tk.LabelFrame(right_frame, text="Voltear", padx=10, pady=5, font=("Arial", 10, "bold"))
        transform_frame.pack(fill=tk.X, pady=5)
        
        self._add_transform_buttons(transform_frame)
        
        btn_reset = tk.Button(right_frame, text="Resetear Todo",
                             command=self.reset_all, width=20, height=2,
                             bg="#e74c3c", fg="white")
        btn_reset.pack(pady=10)
        
        info_label = tk.Label(right_frame, text="Consejo: Usa los sliders RGB\npara ajustar el color del área\nseleccionada", 
                             fg="#7f8c8d", justify=tk.LEFT)
        info_label.pack(pady=10)
        
    def _add_filters(self, parent):
        self.rgb_filter = RGBFilter()
        self.blur_filter = BlurFilter()
        self.edge_filter = EdgeFilter()
        
        filter_manager = self.controller._filter_manager
        filter_manager.add_filter(self.rgb_filter)
        filter_manager.add_filter(self.blur_filter)
        filter_manager.add_filter(self.edge_filter)
        
        rgb_frame = self.rgb_filter.get_control_widget(parent, self.apply_filters)
        rgb_frame.pack(fill=tk.X, pady=2)
        
        blur_frame = self.blur_filter.get_control_widget(parent, self.apply_filters)
        blur_frame.pack(fill=tk.X, pady=2)
        
        edge_frame = self.edge_filter.get_control_widget(parent, self.apply_filters)
        edge_frame.pack(fill=tk.X, pady=2)
    
    def _add_rotation_controls(self, parent):
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        
        btn_rotate_left = tk.Button(btn_frame, text="← 15°", 
                                   command=lambda: self.rotate(-15), width=8)
        btn_rotate_left.pack(side=tk.LEFT, padx=5)
        
        btn_rotate_right = tk.Button(btn_frame, text="15° →", 
                                    command=lambda: self.rotate(15), width=8)
        btn_rotate_right.pack(side=tk.LEFT, padx=5)
        
        self.angle_slider = tk.Scale(parent, from_=-180, to=180, orient=tk.HORIZONTAL,
                                     command=self.set_angle, length=200)
        self.angle_slider.pack(fill=tk.X, pady=5)
        self.angle_slider.set(0)
        
        btn_reset_rotation = tk.Button(parent, text="Resetear Rotacion", 
                                      command=self.reset_rotation, width=18)
        btn_reset_rotation.pack(pady=2)
    
    def _add_transform_buttons(self, parent):
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        
        btn_flip_h = tk.Button(btn_frame, text="Horizontal",
                              command=self.controller.flip_horizontal, width=10)
        btn_flip_h.pack(side=tk.LEFT, padx=5)
        
        btn_flip_v = tk.Button(btn_frame, text="Vertical",
                              command=self.controller.flip_vertical, width=10)
        btn_flip_v.pack(side=tk.LEFT, padx=5)
    
    def _add_selection_controls(self, parent):
        shape_frame = tk.Frame(parent)
        shape_frame.pack(pady=5)
        
        self.shape_var = tk.StringVar(value="ninguno")
        
        rb_none = tk.Radiobutton(shape_frame, text="Ninguno", variable=self.shape_var, 
                                 value="ninguno", command=self._on_shape_change)
        rb_none.pack(side=tk.LEFT, padx=10)
        
        rb_cuadrado = tk.Radiobutton(shape_frame, text="Cuadrado", variable=self.shape_var, 
                                     value="cuadrado", command=self._on_shape_change)
        rb_cuadrado.pack(side=tk.LEFT, padx=10)
        
        rb_circulo = tk.Radiobutton(shape_frame, text="Círculo", variable=self.shape_var, 
                                    value="circulo", command=self._on_shape_change)
        rb_circulo.pack(side=tk.LEFT, padx=10)
        
        controls_container = tk.Frame(parent)
        controls_container.pack(fill=tk.X, pady=5)
        
        pos_frame = tk.Frame(controls_container)
        pos_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Label(pos_frame, text="Posicion X:", font=("Arial", 9)).pack()
        self.x_slider = tk.Scale(pos_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 command=self._on_position_change, length=150)
        self.x_slider.pack(fill=tk.X)
        self.x_slider.set(50)
        
        tk.Label(pos_frame, text="Posicion Y:", font=("Arial", 9)).pack()
        self.y_slider = tk.Scale(pos_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 command=self._on_position_change, length=150)
        self.y_slider.pack(fill=tk.X)
        self.y_slider.set(50)
        
        size_frame = tk.Frame(controls_container)
        size_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Label(size_frame, text="Tamaño (%)", font=("Arial", 9)).pack()
        self.size_slider = tk.Scale(size_frame, from_=5, to=50, orient=tk.HORIZONTAL,
                                    command=self._on_size_change, length=150)
        self.size_slider.pack(fill=tk.X)
        self.size_slider.set(20)
        
        color_frame = tk.Frame(controls_container)
        color_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Label(color_frame, text="Ajuste de Color", font=("Arial", 9)).pack()
        
        rgb_frame = tk.Frame(color_frame)
        rgb_frame.pack()
        
        r_frame = tk.Frame(rgb_frame)
        r_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(r_frame, text="R", fg="red", font=("Arial", 10, "bold")).pack()
        self.r_slider = tk.Scale(r_frame, from_=-100, to=100, orient=tk.VERTICAL,
                                 command=self._on_color_change, length=100)
        self.r_slider.pack()
        self.r_slider.set(0)
        
        g_frame = tk.Frame(rgb_frame)
        g_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(g_frame, text="G", fg="green", font=("Arial", 10, "bold")).pack()
        self.g_slider = tk.Scale(g_frame, from_=-100, to=100, orient=tk.VERTICAL,
                                 command=self._on_color_change, length=100)
        self.g_slider.pack()
        self.g_slider.set(0)
        
        b_frame = tk.Frame(rgb_frame)
        b_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(b_frame, text="B", fg="blue", font=("Arial", 10, "bold")).pack()
        self.b_slider = tk.Scale(b_frame, from_=-100, to=100, orient=tk.VERTICAL,
                                 command=self._on_color_change, length=100)
        self.b_slider.pack()
        self.b_slider.set(0)
    
    def _on_shape_change(self):
        shape = self.shape_var.get()
        if shape == "ninguno":
            self.controller.disable_selection()
        else:
            self.controller.set_selection_mode(shape)
    
    def _on_position_change(self, value):
        x = self.x_slider.get()
        y = self.y_slider.get()
        self.controller.set_selection_position(x, y)
    
    def _on_size_change(self, value):
        size = self.size_slider.get()
        self.controller.set_selection_size(size)
    
    def _on_color_change(self, value):
        r = self.r_slider.get()
        g = self.g_slider.get()
        b = self.b_slider.get()
        self.controller.set_selection_color(r, g, b)
    
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Todos los archivos", "*.*")
        ])
        
        if ruta:
            self.controller.load_image(ruta)
            self.reset_all()
    
    def reset_all(self):
        self.rgb_filter.reset()
        self.blur_filter.reset()
        self.edge_filter.reset()
        self.angle_slider.set(0)
        self.shape_var.set("ninguno")
        self.x_slider.set(50)
        self.y_slider.set(50)
        self.size_slider.set(20)
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(0)
        self.controller.reset_filters()
    
    def reset_rotation(self):
        self.angle_slider.set(0)
        self.controller.reset_rotation()
    
    def rotate(self, delta):
        new_angle = self.controller.get_rotation_angle() + delta
        self.angle_slider.set(new_angle)
        self.controller.rotate(delta)
    
    def set_angle(self, value):
        angle = float(value)
        self.controller.set_rotation_angle(angle)
    
    def apply_filters(self):
        self.controller.apply_filters()
        
    def on_image_changed(self):
        self.root.after(100, self.update_display)
        
    def update_display(self):
        img = self.controller.get_current_image()
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            width, height = img_pil.size
            max_width = 500
            max_height = 400
            
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.current_image_tk = ImageTk.PhotoImage(img_pil)
            self.panel.config(image=self.current_image_tk)
            self.panel.image = self.current_image_tk
            
    def run(self):
        self.root.mainloop()