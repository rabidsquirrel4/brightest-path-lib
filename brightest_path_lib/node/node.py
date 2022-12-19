import numpy as np

class Node:
    def __init__(
        self,
        point: np.ndarray,
        g_score: float,
        h_score: float,
        predecessor: 'Node' = None
    ):
        self.point = point
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = self.g_score + self.h_score
        self.predecessor = predecessor
    
    @property
    def point(self):
        return self._point
    
    @point.setter
    def point(self, value: np.ndarray):
        self._point = value

    @property
    def g_score(self):
        return self._g_score
    
    @g_score.setter
    def g_score(self, value: float):
        self._g_score = value
    
    @property
    def h_score(self):
        return self._h_score
    
    @h_score.setter
    def h_score(self, value: float):
        self._h_score = value
    
    @property
    def f_score(self):
        return self._f_score
    
    @f_score.setter
    def f_score(self, value: float):
        self._f_score = value
    
    @property
    def predecessor(self):
        return self._predecessor
    
    @predecessor.setter
    def predecessor(self, value: float):
        self._predecessor = value