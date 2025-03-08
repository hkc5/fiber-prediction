import numpy as np
from skimage.feature import hog
from PIL import Image

class HogFeatureExtractor:
    def __init__(self, grid_quotient, orientations):
        """
        HOG Feature Extractor that encapsulates grid quotient and orientations.
        
        Parameters:
        - grid_quotient (int or list of int): Defines the cell size for HOG in terms of image size ratio.
        - orientations (int): Number of orientation bins for HOG.
        """
        self.grid_quotient = grid_quotient if isinstance(grid_quotient, (list, np.ndarray)) else [grid_quotient]
        self.orientations = orientations

    def __call__(self, image):
        """
        Extract HOG features from an image with specified grid quotient and orientations.
        
        Parameters:
        - image (PIL.Image): The image to process.
        
        Returns:
        - np.array: Extracted HOG feature vector.
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image_array = image
        else:
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be either a numpy array or PIL.Image")
            image_array = np.array(image)
        
        hog_features = [
            hog(
                image_array,
                orientations=self.orientations,
                pixels_per_cell=(image_array.shape[0] // gq, image_array.shape[1] // gq),
                cells_per_block=(1, 1),
                feature_vector=True,
            )
            for gq in self.grid_quotient
        ]
        
        return np.concatenate(hog_features)
