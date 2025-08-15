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
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling
        # Input: 224x224x3
        # After conv1 + pool: 112x112x32
        # After conv2 + pool: 56x56x64
        # Flattened: 56 * 56 * 64 = 200,704
        
        # Global Pooling to tiny head 
        self.gap     = nn.AdaptiveAvgPool2d(1)  # (B,64,1,1)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(64, 64)
        self.drop    = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU(inplace= True)
    
    def forward(self, x):

        # First conv block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Second conv block
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.gap(x)         # (B,64,1,1)
        x = self.flatten(x)     # (B,64)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)         # (B,1) logit
        # Note: We don't apply sigmoid here because we'll use BCEWithLogitsLoss
        # which includes sigmoid internally

        return x