import cv2
import numpy as np
from typing import Optional, Tuple

class ImageModel:
    
    def __init__(self):
        self._original_image: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify_observers(self):
        for observer in self._observers:
            observer.on_image_changed()
    
    def load_image(self, path: str) -> bool:
        try:
            self._original_image = cv2.imread(path)
            if self._original_image is None:
                return False
            self._current_image = self._original_image.copy()
            self.notify_observers()
            return True
        except Exception:
            return False
    
    def update_current_image(self, image: np.ndarray):
        self._current_image = image.copy()
        self.notify_observers()
    
    def get_original_image(self) -> Optional[np.ndarray]:
        return self._original_image
    
    def get_current_image(self) -> Optional[np.ndarray]:
        return self._current_image
    
    def has_image(self) -> bool:
        return self._original_image is not None
    
    def get_image_size(self) -> Tuple[int, int]:
        if self._current_image is not None:
            height, width = self._current_image.shape[:2]
            return width, height
        return 0, 0