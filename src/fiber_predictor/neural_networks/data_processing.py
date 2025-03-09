"""Data processing utilities for CNN-based fiber orientation prediction."""

import os
import numpy as np
import pandas as pd
import torch
import random
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset

class AngleDataset(Dataset):
    """Dataset class for fiber orientation images with on-the-fly augmentation.
    
    Attributes:
        image_paths: List of paths to image files
        angles: List of corresponding angles
        augment: Whether to apply random augmentations
        config: Dictionary containing configuration parameters
    """
    
    def __init__(self, image_paths, angles, config, augment=False):
        self.image_paths = image_paths
        self.angles = angles
        self.augment = augment
        self.config = config
    
    def __len__(self):
        return len(self.image_paths)
    
    def apply_augmentation(self, image, angle):
        """Apply random augmentation to image and adjust angle accordingly."""
        # Randomly choose an augmentation
        aug_type = random.choice(['none', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v'])
        
        if aug_type == 'none':
            return image, angle
        
        elif aug_type.startswith('rot'):
            # Extract rotation degree from aug_type
            deg = int(aug_type[3:])
            # Rotate image
            image = image.rotate(deg)
            # Adjust angle
            angle = (angle + deg) % 180
            
        elif aug_type == 'flip_h':
            # Horizontal flip
            image = ImageOps.mirror(image)
            angle = (180 - angle) % 180
            
        elif aug_type == 'flip_v':
            # Vertical flip
            image = ImageOps.flip(image)
            angle = (180 - angle) % 180
        
        return image, angle
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        angle = self.angles[idx]
        
        # Apply random augmentation if enabled
        if self.augment:
            image, angle = self.apply_augmentation(image, angle)
        
        # Resize to standard input size
        image = image.resize(self.config['image_size'])
        
        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(image)
        angle_tensor = torch.tensor([angle], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1)
        
        return image_tensor, angle_tensor

def load_dataset(image_dir, labels_csv, config, balance=False, augment=False):
    """Load and preprocess image dataset.
    
    Args:
        image_dir: Directory containing images
        labels_csv: Path to CSV file with image labels
        config: Dictionary containing configuration parameters
        balance: Whether to balance the dataset by angle bins
        augment: Whether to apply data augmentation
    
    Returns:
        Dictionary containing:
            - images: Tensor of preprocessed images
            - labels: Tensor of angle labels
            - n_samples: Number of samples
    """
    # Read labels
    labels_df = pd.read_csv(labels_csv)
    
    # Get image paths and angles
    image_paths = [os.path.join(image_dir, f) for f in labels_df['filename']]
    angles = labels_df['angle'].values
    
    if balance:
        # Create angle bins (0-20°, 20-40°, etc.)
        angle_bins = np.arange(0, 181, 20)
        bin_labels = pd.cut(angles, bins=angle_bins)
        
        # Count samples in each bin and find non-empty bins
        bin_counts = bin_labels.value_counts()
        non_empty_bins = bin_counts[bin_counts > 0].index
        
        if len(non_empty_bins) > 0:
            # Find the size of the smallest non-empty bin
            min_bin_size = bin_counts.min()
            
            # Downsample each non-empty bin to min_bin_size
            balanced_paths = []
            balanced_angles = []
            
            for bin_label in non_empty_bins:
                mask = bin_labels == bin_label
                bin_paths = np.array(image_paths)[mask]
                bin_angles = angles[mask]
                
                # Randomly select min_bin_size samples from this bin
                resampled_indices = np.random.choice(
                    len(bin_paths),
                    size=min_bin_size,
                    replace=False  # No replacement to avoid duplicates
                )
                resampled_paths = bin_paths[resampled_indices]
                resampled_angles = bin_angles[resampled_indices]
                
                balanced_paths.extend(resampled_paths)
                balanced_angles.extend(resampled_angles)
                
            image_paths = balanced_paths
            angles = balanced_angles
        else:
            print("Warning: No samples found in any angle bin. Skipping balancing.")
        
    # Create dataset with on-the-fly augmentation if requested
    dataset = AngleDataset(
        image_paths,
        angles,
        config,
        augment=augment
    )
    
    # Load all data into memory
    all_images = []
    all_angles = []
    
    for i in range(len(dataset)):
        image, angle = dataset[i]
        all_images.append(image)
        all_angles.append(angle)
    
    # Stack into tensors
    images_tensor = torch.stack(all_images)
    angles_tensor = torch.cat(all_angles)
    
    return {
        'images': images_tensor,
        'labels': angles_tensor,
        'n_samples': len(dataset)
    }
