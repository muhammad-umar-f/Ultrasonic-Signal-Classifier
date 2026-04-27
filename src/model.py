"""
1D CNN model for ultrasonic signal classification.

Architecture:
- Configurable number of convolutional layers
- Configurable filter sizes and kernel sizes
- Dropout for regularization
- Binary classification head
"""

import torch
import torch.nn as nn


class Conv1DClassifier(nn.Module):
    """1D CNN classifier for ultrasonic signals."""
    
    def __init__(
        self,
        input_length: int = 256,
        num_filters: int = 32,
        kernel_size: int = 5,
        depth: int = 3,
        dropout_rate: float = 0.5,
        num_classes: int = 2
    ):
        """
        Initialize the 1D CNN classifier.
        
        Args:
            input_length: Length of input signals
            num_filters: Number of filters in first convolutional layer
            kernel_size: Kernel size for convolutional layers
            depth: Number of convolutional blocks
            dropout_rate: Dropout rate
            num_classes: Number of output classes (2 for binary classification)
        """
        super(Conv1DClassifier, self).__init__()
        
        self.input_length = input_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = 1
        out_channels = num_filters
        
        for i in range(depth):
            # Convolutional layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                )
            )
            
            # Batch normalization
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            
            # Max pooling
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)  # Double filters, capped at 256
        
        # Calculate flattened size
        self.flat_size = self._calculate_flat_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def _calculate_flat_size(self) -> int:
        """Calculate the flattened size after convolution and pooling."""
        # Create a dummy input to pass through the layers
        x = torch.randn(1, 1, self.input_length)
        
        with torch.no_grad():
            for i in range(self.depth):
                x = self.conv_layers[i](x)
                x = self.bn_layers[i](x)
                x = self.relu(x)
                x = self.pool_layers[i](x)
            
            flat_size = x.view(1, -1).shape[1]
        
        return flat_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional blocks
        for i in range(self.depth):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.relu(x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
    
    def get_config(self) -> dict:
        """Get model configuration as a dictionary."""
        return {
            "input_length": self.input_length,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "depth": self.depth,
            "dropout_rate": self.dropout_rate,
        }
