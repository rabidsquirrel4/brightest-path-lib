from brightest_path_lib.cost import Cost
from typing import List, Tuple
import numpy as np
import math

class ConsistencyTurns(Cost):
    """Uses the reciprocal of pixel/voxel intensity to compute the cost of moving
    to a neighboring point

    Parameters
    ----------
    min_intensity : float
        The minimum intensity a pixel/voxel can have in a given image
    max_intensity : float
        The maximum intensity a pixel/voxel can have in a given image
    target_rgb : np.ndarray
        The rbg value of comparison when calculating consistency of pixel color.
        (ex. the rgb values for the end pixel or start pixel or an 
        average of rgb values of the end pixel and start pixel)

    Attributes
    ----------
    RECIPROCAL_MIN : float
        To cope with zero intensities, RECIPROCAL_MIN is added to the intensities
        in the range before reciprocal calculation
    RECIPROCAL_MAX : float
        We set the maximum intensity <= RECIPROCAL_MAX so that the intensity
        is between RECIPROCAL MIN and RECIPROCAL_MAX

    """
    # TODO: make weights for intensity and consistency optional parameters
    def __init__(self, min_intensity: float, max_intensity: float, 
                 target_rgb: np.ndarray) -> None:
        super().__init__()
        if min_intensity is None or max_intensity is None:
            raise TypeError
        if min_intensity > max_intensity:
            raise ValueError
        shape = target_rgb.shape
        if (shape != (4,) and shape != (3,) and shape != (4) and shape != (3)):
            raise ValueError("RGB or RGBA of target pixel not correct " +
                             f"shape.\nExpected ndarray of shape (4,), (3,), " + 
                             f"found: {shape}")
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.target_rgb = target_rgb
        self.RECIPROCAL_MIN = float(1E-6)
        self.RECIPROCAL_MAX = 255.0
        self.INT_THRESHOLD = 0.5
        self.RGB_DIST_THRESHOLD = 0.1
        self.RGB_WEIGHT = 255.0
        self.DIRECTION_WEIGHT = 10000000.
        self._min_step_cost = 1.0 / self.RECIPROCAL_MAX

    @staticmethod
    def _rgb_distance(rgb_arr1: np.ndarray, rgb_arr2: np.ndarray):
        r_dist = rgb_arr1[0] - rgb_arr2[0]
        g_dist = rgb_arr1[1] - rgb_arr2[1]
        b_dist = rgb_arr1[2] - rgb_arr2[2]
        return math.sqrt(r_dist*r_dist + g_dist*g_dist + b_dist*b_dist)
    
    # TODO:
    def _direction_cost():
        pass
    
    # TODO:
    def _intensity_cost():
        pass

    # TODO: x, y, intensity, theta(pervious direction moved)
    def cost_of_moving_to(self, curr_pt: Tuple, # (y, x)
                          new_pt: Tuple,
                          intensity_at_new_point: float,
                          intensity_at_curr_point: float,
                          rgba_at_new_point: np.ndarray,
                          rgba_at_curr_point: np.ndarray,
                          curr_dir: float,
                          new_dir: float,
                          block_distance: float) -> float:
        """calculates the cost of moving to a point

        Parameters
        ----------
        intensity_at_new_point : float
            The intensity of the new point under consideration
        rgba_at_new_point : 
            
        Returns
        -------
        float
            the cost of moving to the new point
        
        Notes
        -----
        - To cope with zero intensities, RECIPROCAL_MIN is added to the intensities in the range before reciprocal calculation
        - We set the maximum intensity <= RECIPROCAL_MAX so that the intensity is between RECIPROCAL MIN and RECIPROCAL_MAX
        
        """
        if intensity_at_new_point > self.max_intensity:
            raise ValueError

        rgb_cost = self._rgb_distance(rgba_at_curr_point, rgba_at_new_point)
        
        dir_cost = block_distance
        
        intensity_at_new_point = self.RECIPROCAL_MAX * (intensity_at_new_point - self.min_intensity) / (self.max_intensity - self.min_intensity)
        if intensity_at_new_point < self.RECIPROCAL_MIN:
            intensity_at_new_point = self.RECIPROCAL_MIN
        int_cost = 1.0 / intensity_at_new_point
        
        weighted_sum_cost = self.RGB_WEIGHT * rgb_cost + int_cost 
        if rgb_cost < self.RGB_DIST_THRESHOLD:
            weighted_sum_cost += self.DIRECTION_WEIGHT * dir_cost 
        # TODO: remove Testing code below:
        if new_pt[1] == 1569:
            wght_dir_cost = self.DIRECTION_WEIGHT * dir_cost
        return max(weighted_sum_cost, self.minimum_step_cost())

    def minimum_step_cost(self) -> float:
        """calculates the minimum step cost
        
        Returns
        -------
        float
            the minimum step cost
        """
        return self._min_step_cost
