from brightest_path_lib.cost import Cost
import numpy as np
import math

class ReciprocalConsistency(Cost):
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
        self.THRESHOLD = 0.1
        self._min_step_cost = 1.0 / self.RECIPROCAL_MAX

    @staticmethod
    def _rgb_distance(rgb_arr1: np.ndarray, rgb_arr2: np.ndarray):
        r_dist = rgb_arr1[0] - rgb_arr2[0]
        g_dist = rgb_arr1[1] - rgb_arr2[1]
        b_dist = rgb_arr1[2] - rgb_arr2[2]
        return math.sqrt(r_dist*r_dist + g_dist*g_dist + b_dist*b_dist)

    def cost_of_moving_to(self, intensity_at_new_point: float, 
                          rgba_at_new_point: np.ndarray, 
                          movement_distance: float) -> float:
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

        intensity_at_new_point = self.RECIPROCAL_MAX * (intensity_at_new_point - self.min_intensity) / (self.max_intensity - self.min_intensity)

        #
        if intensity_at_new_point > self.THRESHOLD * self.RECIPROCAL_MAX:
            rgb_dist = 255.0 * self._rgb_distance(self.target_rgb, rgba_at_new_point)
            if rgb_dist > self.RECIPROCAL_MIN:
                return rgb_dist
            else:
                return self.RECIPROCAL_MIN
        else:    
            if intensity_at_new_point < self.RECIPROCAL_MIN:
                intensity_at_new_point = self.RECIPROCAL_MIN
            
            return 1.0 / intensity_at_new_point
    
    def minimum_step_cost(self) -> float:
        """calculates the minimum step cost
        
        Returns
        -------
        float
            the minimum step cost
        """
        return self._min_step_cost
