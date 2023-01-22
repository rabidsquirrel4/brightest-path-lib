import numpy as np

class BidirectionalNode:
    """Class holding information about a node

    Parameters
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node (can be a pixel or a voxel)
    g_score_from_start : float
        the distance from a starting node to the current node
    g_score_from_goal : float
        the distance from a goal node to the current node
    h_score_from_start : float
        the estimated distance from the current node to a goal node
    h_score_from_goal : float
        the estimated distance from the current node to a start node
    predecessor_from_start : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the start node
    predecessor_from_goal : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the goal node
    
    Attributes
    ----------
    point : numpy ndarray
        the 2D/3D coordinates of the node (can be a pixel or a voxel)
    g_score_from_start : float
        the distance from a starting node to the current node
    g_score_from_goal : float
        the distance from a goal node to the current node
    h_score_from_start : float
        the estimated distance from the current node to a goal node
    h_score_from_goal : float
        the estimated distance from the current node to a start node
    f_score_from_start : float
        the sum of g_score_from_start and h_score_from_start
    f_score_from_goal : float
        the sum of g_score_from_goal and h_score_from_goal
    predecessor_from_start : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the start node
    predecessor_from_goal : BidirectionalNode
        the current node's immediate predecessor, from which we
        travelled to the current node
        The predecessor's first ancestor is the goal node
    
    """
    def __init__(
        self,
        point: np.ndarray,
        g_score_from_start: float,
        g_score_from_goal: float,
        h_score_from_start: float,
        h_score_from_goal: float,
        predecessor_from_start: 'BidirectionalNode' = None,
        predecessor_from_goal: 'BidirectionalNode' = None
    ):
        self.point = point
        self.g_score_from_start = g_score_from_start
        self.g_score_from_goal = g_score_from_goal
        self.h_score_from_start = h_score_from_start
        self.h_score_from_goal = h_score_from_goal
        self.f_score_from_start = self.g_score_from_start + self.h_score_from_start
        self.f_score_from_goal = self.g_score_from_goal + self.h_score_from_goal
        self.predecessor_from_start = predecessor_from_start
        self.predecessor_from_goal = predecessor_from_goal
    
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
    def g_score_from_start(self):
        return self._g_score_from_start
    
    @g_score_from_start.setter
    def g_score_from_start(self, value: float):
        if value is None:
            raise TypeError
        self._g_score_from_start = value
    
    @property
    def g_score_from_goal(self):
        return self._g_score_from_goal
    
    @g_score_from_goal.setter
    def g_score_from_goal(self, value: float):
        if value is None:
            raise TypeError
        self._g_score_from_goal = value
    
    @property
    def h_score_from_start(self):
        return self._h_score_from_start
    
    @h_score_from_start.setter
    def h_score_from_start(self, value: float):
        if value is None:
            raise TypeError
        self._h_score_from_start = value
    
    @property
    def h_score_from_goal(self):
        return self._h_score_from_goal
    
    @h_score_from_start.setter
    def h_score_from_goal(self, value: float):
        if value is None:
            raise TypeError
        self._h_score_from_goal = value
    
    @property
    def f_score_from_start(self):
        return self._f_score_from_start
    
    @f_score_from_start.setter
    def f_score_from_start(self, value: float):
        if value is None:
            raise TypeError
        self._f_score_from_start = value
    
    @property
    def f_score_from_goal(self):
        return self._f_score_from_goal
    
    @f_score_from_start.setter
    def f_score_from_goal(self, value: float):
        if value is None:
            raise TypeError
        self._f_score_from_goal = value
    
    @property
    def predecessor_from_start(self):
        return self._predecessor_from_start
    
    @predecessor_from_start.setter
    def predecessor_from_start(self, value: float):
        if value is None:
            raise TypeError
        self._predecessor_from_start = value
    
    @property
    def predecessor_from_goal(self):
        return self._predecessor_from_goal
    
    @predecessor_from_goal.setter
    def predecessor_from_goal(self, value: float):
        if value is None:
            raise TypeError
        self._predecessor_from_goal = value