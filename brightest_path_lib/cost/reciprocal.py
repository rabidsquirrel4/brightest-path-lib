from cost import Cost

RECIPROCAL_MIN = 1E-6
RECIPROCAL_MAX = 255.0

class Reciprocal(Cost):
    """Uses the reciprocal of pixel/voxel intensity to compute the cost of moving to a neighboring point

    Parameters
    ----------
    min_intensity : float
        The minimum intensity a pixel/voxel can have in a given image
    max_intensity : float
        The maximum intensity a pixel/voxel can have in a given image

    """

    def __init__(self, min_intensity: float, max_intensity: float) -> None:
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        

    def cost_of_moving_to(self, intensity_at_new_point: float) -> float:
        """Returns the cost of moving to a point

        Parameters
        ----------
        intensity_at_new_point : float
            The intensity of the new point under consideration
        
        Returns
        -------
        float
            the cost of moving to the new point
        
        Notes
        -----
        - To cope with zero intensities, RECIPROCAL_MIN is added to the intensities in the range before reciprocal calculation
        - We set the maximum intensity <= RECIPROCAL_MAX so that the intensity is between RECIPROCAL MIN and RECIPROCAL_MAX
        
        """
        intensity_at_new_point = 255.0 * (intensity_at_new_point - self.min_intensity) / (self.max_intensity - self.min_intensity)

        if intensity_at_new_point <= 0:
            intensity_at_new_point = RECIPROCAL_MIN
        elif intensity_at_new_point > RECIPROCAL_MAX:
            intensity_at_new_point = RECIPROCAL_MAX
        
        return 1.0 / intensity_at_new_point
    
    def minimum_step_cost(self) -> float:
        """Returns the minimum step cost
        
        Returns
        -------
        float
            the minimum step cost
        """
        return 1.0 / RECIPROCAL_MAX

