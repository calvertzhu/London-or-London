import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Baseline CNN model for London-or-London classification.
    Simple architecture with 2 conv layers + max pooling + dense layers.
    """
    def __init__(self):
        super(BaselineCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling
        # Input: 224x224x3
        # After conv1 + pool: 112x112x32
        # After conv2 + pool: 56x56x64
        # Flattened: 56 * 56 * 64 = 200,704
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(56 * 56 * 64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Normalize pixel values (0-255 to 0-1)
        x = x / 255.0
        
        # First conv block
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second conv block
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Note: We don't apply sigmoid here because we'll use BCEWithLogitsLoss
        # which includes sigmoid internally
        return x 