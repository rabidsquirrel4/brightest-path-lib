import numpy as np

class Node:
    """Class holding information about a node

    Parameters
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node (can be a pixel or a voxel)
    g_score : float
        the distance from a starting node to the current node
    h_score : float
        the estimated distance from the current node to a goal_node
    predecessor : Node
        the current node's immediate predecessor, from which we
        travelled to the current node
    
    Attributes
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node
    direction : float
        the direction that the algorithm moved to get to this node. An angle 
        in degrees with respect to the postitive x axis.
    g_score : float
        the actual cost from a starting node to the current node
    h_score : float
        the estimated cost from the current node to a goal_node
    f_score : float
        the sum of the node's g_score and h_score
    predecessor : Node
        the current node's immediate predecessor, from which we
        travelled to the current node
    
    """
    def __init__(
        self,
        point: np.ndarray,
        direction: float,
        g_score: float,
        h_score: float,
        predecessor: 'Node' = None
    ):
        self.point = point
        self.direction: float = direction
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = self.g_score + self.h_score
        self.predecessor = predecessor
    
    @property
    def point(self):
        return self._point
    
    @point.setter
    def point(self, value: np.ndarray):
        if value is None:
            raise TypeError
        if len(value) == 0:
            raise ValueError
        self._point = value

    @property
    def direction(self):
        """The direction that the algorithm moved to get to this node."""
        return self._direction
    
    @direction.setter
    def direction(self, value: float):
        if value is None:
            raise TypeError
        self._direction = value

    @property
    def g_score(self):
        return self._g_score
    
    @g_score.setter
    def g_score(self, value: float):
        if value is None:
            raise TypeError
        self._g_score = value
    
    @property
    def h_score(self):
        return self._h_score
    
    @h_score.setter
    def h_score(self, value: float):
        if value is None:
            raise TypeError
        self._h_score = value
    
    @property
    def f_score(self):
        return self._f_score
    
    @f_score.setter
    def f_score(self, value: float):
        if value is None:
            raise TypeError
        self._f_score = value
    
    @property
    def predecessor(self):
        return self._predecessor
    
    @predecessor.setter
    def predecessor(self, value: float):
        self._predecessor = value