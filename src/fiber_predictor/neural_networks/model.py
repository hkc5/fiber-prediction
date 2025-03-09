"""CNN model for fiber orientation prediction."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        conv_channels=[32, 64, 128, 256],
        pool_kernel=2, 
        activation=nn.ReLU,
        dropout=0.3,
        fc_dims=[128, 64],
        output_dim=1
    ):
        """
        A parameterized CNN class.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            conv_channels (list): List of output channels for each convolutional layer.
            pool_kernel (int or tuple): Kernel size for pooling layers.
            activation (nn.Module): Activation function class to use (default is nn.ReLU).
            dropout (float): Dropout probability for fully connected layers.
            fc_dims (list): List specifying the number of units in each hidden FC layer.
            output_dim (int): Number of outputs (e.g., 1 for a single regression output).
        """
        super(CNN, self).__init__()

        # -----------------------------
        # 1. Define the Convolutional Layers
        # -----------------------------
        conv_layers = []
        prev_channels = in_channels
        
        for out_channels in conv_channels:
            conv_layers.append(nn.Conv2d(prev_channels, out_channels,
                                         kernel_size=3,
                                         padding=1))
            conv_layers.append(activation(inplace=True))
            conv_layers.append(nn.MaxPool2d(pool_kernel))
            prev_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)

        # -----------------------------
        # 2. Global Average Pooling
        # -----------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # -----------------------------
        # 3. Define the Fully Connected Layers
        # -----------------------------
        fc_layers = [nn.Flatten()]
        
        # First FC layer takes input from the last conv layer’s channel dimension.
        in_features = conv_channels[-1]

        for dim in fc_dims:
            fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(in_features, dim))
            fc_layers.append(activation(inplace=True))
            in_features = dim
        
        fc_layers.append(nn.Linear(in_features, output_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        x = self.conv_layers(x)
        x = self.gap(x)
        x = self.fc_layers(x)
        # Assume x is between 0 and 1 and scale to 0-180
        x = 180 * torch.sigmoid(x)
        return x