import torch
import torch.nn as nn
from models.resnet_cbam_mlp import ResNet50_CBAM_MLP

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
model = ResNet50_CBAM_MLP().to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Dummy input
x = torch.randn(4, 3, 224, 224).to(device)  # batch of 4 RGB images
y = model(x)
print(f"Output shape: {y.shape}")