#!/usr/bin/env python3
"""
Script to train machine learning models for fiber orientation prediction.
"""
import argparse
import logging
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.utils import get_data
from neural_networks.model import FiberOrientationModel

def setup_logging():
    """Configure logging for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def prepare_data(data_dir, csv_dir, test_size=0.2, random_state=42):
    """
    Prepare data for training.
    
    Args:
        data_dir (str): Directory containing image data
        csv_dir (str): Directory containing CSV metadata
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: Training and testing data and labels
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing data from {data_dir}")
    
    # Get feature matrix and labels
    feature_matrix, labels = get_data(data_dir, csv_dir, augment=True)
    
    # Convert to numpy arrays
    X = np.array(feature_matrix)
    y = np.array(labels)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_neural_network(X_train, X_test, y_train, y_test, epochs=50, learning_rate=0.001):
    """
    Train a neural network model.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Testing labels
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        nn.Module: Trained model
    """
    logger = logging.getLogger(__name__)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Initialize model, loss, and optimizer
    model = FiberOrientationModel(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], '
                        f'Train Loss: {loss.item():.4f}, '
                        f'Test Loss: {test_loss.item():.4f}')
    
    return model

def save_model(model, scaler, output_path='data/models/neural_networks/fiber_orientation_model.pth'):
    """
    Save the trained model and scaler.
    
    Args:
        model (nn.Module): Trained neural network model
        scaler (StandardScaler): Feature scaler
        output_path (str): Path to save the model
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model state dict and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, output_path)
    
    logger.info(f"Model saved to {output_path}")

def main():
    """
    Main training script entry point.
    """
    # Set up logging
    logger = setup_logging()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Fiber Orientation Prediction Model')
    parser.add_argument('--data-dir', required=True, help='Directory containing training images')
    parser.add_argument('--csv-dir', required=True, help='Directory containing CSV metadata')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            args.data_dir, args.csv_dir
        )
        
        # Train model
        model = train_neural_network(
            X_train, X_test, y_train, y_test, 
            epochs=args.epochs, 
            learning_rate=args.lr
        )
        
        # Save model
        save_model(model, scaler)
        
        logger.info("Training completed successfully!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
