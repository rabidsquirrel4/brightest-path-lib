import numpy as np

class ImageStats:

    def __init__(self, image: np.ndarray):
        self.min_intensity = np.min(image)
        self.max_intensity = np.max(image)
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0
        self.x_max = len(image[0]) - 1
        self.y_max = len(image) - 1
        self.z_max = 0
        if len(image.shape) == 3:
            self.z_max = image.shape[0] - 1
            self.x_max = image.shape[1] - 1
            self.y_max = image.shape[2] - 1
    
    @property
    def min_intensity(self) -> float:
        return self._min_intensity

    @min_intensity.setter
    def min_intensity(self, value: float):
        self._min_intensity = value 
    
    @property
    def max_intensity(self) -> float:
        return self._max_intensity
    
    @max_intensity.setter
    def max_intensity(self, value: float):
        self._max_intensity = value
    
    @property
    def x_min(self) -> float:
        return self._x_min
    
    @x_min.setter
    def x_min(self, value: float):
        self._x_min = value
    
    @property
    def y_min(self) -> float:
        return self._y_min
    
    @y_min.setter
    def y_min(self, value: float):
        self._y_min = value
    
    @property
    def z_min(self) -> float:
        return self._z_min
    
    @z_min.setter
    def z_min(self, value: float):
        self._z_min = value
    
    @property
    def x_max(self) -> float:
        return self._x_max
    
    @x_max.setter
    def x_max(self, value: float):
        self._x_max = value
    
    @property
    def y_max(self) -> float:
        return self._y_max
    
    @y_max.setter
    def y_max(self, value: float):
        self._y_max = value
    
    @property
    def z_max(self) -> float:
        return self._z_max
    
    @z_max.setter
    def z_max(self, value: float):
        self._z_max = value