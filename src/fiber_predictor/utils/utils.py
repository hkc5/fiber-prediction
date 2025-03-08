import numpy as np
import cv2
import os
import pandas as pd
from typing import List, Tuple, Union, Optional

def get_img(image_directory: str, square: bool = True) -> Optional[np.ndarray]:
    """
    Load an image from a directory.
    
    Args:
        image_directory (str): Path to the image
        square (bool): Whether to make the image square
    
    Returns:
        Optional[np.ndarray]: Loaded image or None if loading fails
    """
    try:
        img = cv2.imread(image_directory)
        if img is None:
            return None
        
        if square:
            # Make image square by taking the minimum dimension
            min_dim = min(img.shape[:2])
            h, w = img.shape[:2]
            start_row = (h - min_dim) // 2
            start_col = (w - min_dim) // 2
            img = img[start_row:start_row+min_dim, 
                      start_col:start_col+min_dim]
        
        return img
    except Exception as e:
        print(f"Error loading image {image_directory}: {e}")
        return None

def get_feature_vector(image: np.ndarray, grid_q: Union[int, List[int]] = 10, no_orientation: int = 10) -> np.ndarray:
    """
    Extract HOG features from an image.
    
    Args:
        image (np.ndarray): Input image
        grid_q (int or list): Grid size or list of grid sizes
        no_orientation (int): Number of orientation bins
    
    Returns:
        np.ndarray: Extracted HOG features
    """
    # Ensure image is grayscale for HOG
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Handle multiple grid sizes
    if isinstance(grid_q, list):
        hog_features = []
        for gq in grid_q:
            hf = get_feature_vector(image, grid_q=gq, no_orientation=no_orientation)
            hog_features.extend(hf)
        return np.array(hog_features)
    
    # Ensure image dimensions are compatible with HOG parameters
    h, w = image.shape
    
    # Adjust grid size to ensure compatibility
    grid_q = max(1, min(grid_q, min(h, w) // 8))
    
    # Calculate pixels per cell
    cell_size = max(1, min(h, w) // grid_q)
    
    # Compute HOG features using skimage for more robust feature extraction
    from skimage.feature import hog
    
    try:
        features = hog(
            image, 
            orientations=no_orientation, 
            pixels_per_cell=(cell_size, cell_size),
            cells_per_block=(1, 1),
            block_norm='L2-Hys'
        )
        return features
    except Exception as e:
        print(f"HOG feature extraction error: {e}")
        return np.array([])

def slice_image(image: np.ndarray, voxel_size: int) -> List[np.ndarray]:
    """
    Slice an image into smaller voxels.
    
    Args:
        image (np.ndarray): Input image
        voxel_size (int): Size of each voxel
    
    Returns:
        List[np.ndarray]: List of image slices
    """
    image_height, image_width = image.shape[:2]
    
    slices = []
    for y in range(0, image_height - voxel_size + 1, voxel_size):
        for x in range(0, image_width - voxel_size + 1, voxel_size):
            slice_img = image[y:y+voxel_size, x:x+voxel_size]
            if slice_img.shape[0] == voxel_size and slice_img.shape[1] == voxel_size:
                slices.append(slice_img)
    
    return slices

def augment_data(img: np.ndarray, label: int,
                 feature_matrix: List, labels: List, 
                 GRIDQ=np.array([1, 2, 3, 4, 5]), ORIENTATION: int = 10) -> Tuple[List, List]:
    """
    Augment image data through rotations and feature extraction.
    
    Args:
        img (np.ndarray): Input image
        label (int): Image label
        img_dir (str): Directory to save augmented images
        img_directories (str): Additional directories
        feature_matrix (List): Existing feature matrix
        labels (List): Existing labels
        GRIDQ (np.ndarray): Grid sizes for feature extraction
        ORIENTATION (int): Number of orientation bins
    
    Returns:
        Tuple of augmented feature matrix and labels
    """
    # Rotation angles
    rotation_angles = [0, 90, 180, 270]
    
    for angle in rotation_angles:
        # Use OpenCV for rotation
        rotation_matrix = cv2.getRotationMatrix2D(
            (img.shape[1]/2, img.shape[0]/2), 
            angle, 
            1.0
        )
        rotated_img = cv2.warpAffine(
            img, 
            rotation_matrix, 
            (img.shape[1], img.shape[0])
        )
        
        # Extract features for different grid sizes
        for grid_q in GRIDQ:
            feature_vec = get_feature_vector(
                rotated_img, 
                grid_q=grid_q, 
                no_orientation=ORIENTATION
            )
            
            # Only append non-empty feature vectors
            if len(feature_vec) > 0:
                feature_matrix.append(feature_vec)
                labels.append(label)
    
    return feature_matrix, labels

def get_data(file_dir: str, csv_dir: str, augment: bool = False, 
             GRIDQ=np.array([1, 2, 3, 4, 5]), ORIENTATION: int = 10) -> Tuple[List, List]:
    """
    Load image data from directory.
    
    Args:
        file_dir (str): Directory containing images
        csv_dir (str): Directory containing CSV metadata
        i_start (int): Starting index
        augment (bool): Whether to augment data
        GRIDQ (np.ndarray): Grid sizes for feature extraction
        ORIENTATION (int): Number of orientation bins
    
    Returns:
        Tuple of feature matrix and labels
    """
    # Ensure CSV file exists
    csv_path = os.path.join(csv_dir, 'mock_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV using pandas
    bio_csv = pd.read_csv(csv_path)
    
    feature_matrix = []
    labels = []
    
    for _, row in bio_csv.iterrows():
        # Construct full image path
        img_path = os.path.join(file_dir, row['image_path'])
        
        # Load image
        img = get_img(img_path)
        if img is None:
            continue
        
        label = row['label']
        
        # Extract features
        feature_vec = get_feature_vector(
            img, 
            grid_q=GRIDQ[0], 
            no_orientation=ORIENTATION
        )
        
        # Only append non-empty feature vectors
        if len(feature_vec) > 0:
            feature_matrix.append(feature_vec)
            labels.append(label)
        
        # Optional data augmentation
        if augment:
            feature_matrix, labels = augment_data(
                img, label,
                feature_matrix, labels, 
                GRIDQ=GRIDQ, ORIENTATION=ORIENTATION
            )
    
    return feature_matrix, labels
