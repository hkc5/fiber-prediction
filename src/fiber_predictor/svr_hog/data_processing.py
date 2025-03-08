import os
import numpy as np
import pandas as pd

from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader

from fiber_predictor.svr_hog.feature_extraction import HogFeatureExtractor


class HogDataset(Dataset):
    def __init__(self, labels_path, file_dir, grid_quotient, orientations, augment_whole_dataset=False):
        """
        PyTorch Dataset for loading images and extracting HOG features.

        Parameters:
        - labels_path (str): Path to the CSV file containing filenames and angles.
        - file_dir (str): Directory containing the images.
        - grid_quotient (int or list of int): Defines the cell size for HOG in terms of image size ratio.
        - orientations (int): Number of orientation bins for HOG.
        - augment_whole_dataset (bool): Whether to augment the entire dataset at once.
        """
        self.file_dir = file_dir
        self.hog_extractor = HogFeatureExtractor(grid_quotient, orientations)

        # Load labels
        labels_df = pd.read_csv(labels_path)

        # Initialize lists for filenames and labels
        self.filenames = labels_df['filename'].tolist()
        self.labels = labels_df['angle'].tolist()

        # Perform full augmentation in memory if required
        if augment_whole_dataset:
            self._augment_entire_dataset()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        angle = self.labels[idx]

        # Load and preprocess the image
        img_path = os.path.join(self.file_dir, img_filename)
        img = self.load_image(img_path)

        # Convert to tensor for compatibility with PyTorch Dataloader
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32)

        # Extract HOG features using the HogFeatureExtractor instance
        features = self.hog_extractor(img)

        # Convert features to tensor for DataLoader compatibility
        features_tensor = torch.tensor(features, dtype=torch.float32)

        return img_tensor, features_tensor, angle

    def load_image(self, img_path):
        """Load an image and convert it to grayscale."""
        return Image.open(img_path).convert('L')

    def _augment_entire_dataset(self):
        """
        Augment the entire dataset in memory without saving to disk.
        """
        augmented_images = []
        augmented_labels = []

        # Iterate over all images and apply augmentations
        for idx in range(len(self.filenames)):
            img_filename = self.filenames[idx]
            angle = self.labels[idx]
            img_path = os.path.join(self.file_dir, img_filename)
            img = self.load_image(img_path)

            # Apply augmentations (rotations, flips)
            # 90-degree rotations
            for i in range(1, 4):
                rotated_img = img.rotate(90 * i)
                augmented_angle = (angle + 90 * i) % 180
                augmented_images.append(rotated_img)
                augmented_labels.append(augmented_angle)

            # Horizontal flip (mirror)
            mirrored_img = ImageOps.mirror(img)
            mirrored_angle = (180 - angle) % 180
            augmented_images.append(mirrored_img)
            augmented_labels.append(mirrored_angle)

            # Vertical flip
            flipped_img = ImageOps.flip(img)
            flipped_angle = (180 - angle) % 180
            augmented_images.append(flipped_img)
            augmented_labels.append(flipped_angle)

            # Horizontal + Vertical flip
            flipped_mirrored_img = ImageOps.mirror(flipped_img)
            flipped_mirrored_angle = angle
            augmented_images.append(flipped_mirrored_img)
            augmented_labels.append(flipped_mirrored_angle)

        # Append augmented data to existing lists
        for idx, aug_img in enumerate(augmented_images):
            # Generate unique filename for each augmented image
            new_filename = f"augmented_{idx}.png"
            # Add the augmented image and label to the dataset
            self.filenames.append(new_filename)
            self.labels.append(augmented_labels[idx])
            # Save augmented image to an in-memory format (or optionally to disk if needed)
            aug_img.save(os.path.join(self.file_dir, new_filename))

# Abstracted balance_dataset function
def balance_dataset(dataset, bins):
    """
    Balance the dataset across specified angle bins.

    Parameters:
    - dataset (HogDataset): Instance of the HogDataset class.
    - bins (list or np.array): The bins used to balance the dataset.
    """
    # Categorize each data point into a bin
    bin_indices = defaultdict(list)
    for idx, angle in enumerate(dataset.labels):
        # Find the bin the angle belongs to
        for i in range(len(bins) - 1):
            if bins[i] <= angle < bins[i + 1]:
                bin_indices[i].append(idx)
                break

    # Find the minimum number of samples across all bins
    min_count = min(len(indices) for indices in bin_indices.values())

    # Resample the dataset to balance across all bins
    balanced_filenames = []
    balanced_labels = []
    for indices in bin_indices.values():
        sampled_indices = np.random.choice(indices, min_count, replace=False)
        for idx in sampled_indices:
            balanced_filenames.append(dataset.filenames[idx])
            balanced_labels.append(dataset.labels[idx])

    # Replace filenames and labels with the balanced version
    dataset.filenames = balanced_filenames
    dataset.labels = balanced_labels

def load_dataset(file_dir, csv_dir, config, augment=False, balance=False):
    grid_quotient = config['grid_quotient']
    orientations = config['orientations']
    
    dataset = HogDataset(
        labels_path=csv_dir,
        file_dir=file_dir,
        grid_quotient=grid_quotient,
        orientations=orientations,
        augment_whole_dataset=augment
    )
    
    if balance:
        angle_bins = config['angle_bins']
        balance_dataset(dataset, angle_bins)
    
    feature_matrix = []
    labels = []
    for idx in range(len(dataset)):
        _, features_tensor, angle = dataset[idx]
        feature_matrix.append(features_tensor.numpy())
        labels.append(angle)
    
    feature_matrix = np.array(feature_matrix)
    labels = np.array(labels)
    
    return {
        'features': feature_matrix,
        'labels': labels,
        'n_samples': len(labels)
    }
