from brightest_path_lib.heuristic import Heuristic
import math
import numpy as np

class Euclidean(Heuristic):
    """heuristic cost estimation using Euclidean distance from current point to goal point

    Parameters
    ----------
    scale : numpy ndarray
        the scale of the image's axes. For example [1.0 1.0] for a 2D image.
    
    Attributes
    ----------
    scale_x : float
        the scale of the image's X-axis
    scale_y : float
        the scale of the image's Y-axis
    scale_z : float
        the scale of the image's Z-axis

    """

    def __init__(self, scale: np.ndarray):
        self.scale_x = scale[0]
        self.scale_y = scale[1]
        self.scale_z = 1.0
        if len(scale) == 3:
            self.scale_z = scale[2]

    def estimate_cost_to_goal(self, current_point: np.ndarray, goal_point: np.ndarray) -> float:
        """calculates the estimated cost from current point to the goal
    
        Parameters
        ----------
        current_point : numpy ndarray
            the coordinates of the current point
                - for 2D points, x and y coordinates
                - for 3D points, x, y and z coordinates
        goal_point : numpy ndarray
            the coordinates of the current point
                - for 2D points, x and y coordinates
                - for 3D points, x, y and z coordinates
        
        Returns
        -------
        float
            the estimated cost to goal in the form of Euclidean distance
        
        Notes
        -----
        If the image is zoomed in or out, then the scale of one of more
        axes will be more or less than 1.0. For example, if the image is zoomed
        in to twice its size then the scale of X and Y axes will be 2.0.
        
        By including the scale in the calculation of distance to the goal we
        can get an accurate cost.
        """
        current_x, current_y, current_z = current_point[0], current_point[1], 0
        goal_x, goal_y, goal_z = goal_point[0], goal_point[1], 0

        if len(current_point) == 3 and len(goal_point) == 3:
            current_z = current_point[2]
            goal_z = goal_point[2]

        x_diff = (goal_x - current_x) * self.scale_x
        y_diff = (goal_y - current_y) * self.scale_y
        z_diff = (goal_z - current_z) * self.scale_z
        
        return math.sqrt((x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff))