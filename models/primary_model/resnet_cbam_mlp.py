import torch
import torch.nn as nn
import torchvision.models as models


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Adds both channel and spatial attention to focus on the most relevant features.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        x = x * self.sigmoid_spatial(self.conv_spatial(spatial))
        return x


class ResNet50_CBAM_MLP(nn.Module):
    """
    Modified ResNet50 model with:
    - Pretrained ResNet50 backbone (excluding final classifier)
    - CBAM attention after last conv layer
    - Custom multi-layer MLP head with BatchNorm, Dropout, ReLU
    """
    def __init__(self):
        super(ResNet50_CBAM_MLP, self).__init__()
        # Load ResNet50 up to before its final avgpool & fc
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # up to last conv layer (2048 channels)

        self.cbam = CBAM(2048)

        # Adaptive pooling to flatten to (batch, 2048, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.Flatten(),               # (batch, 2048)
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)           # single output for binary classification
            # sigmoid is NOT here because we'll use BCEWithLogitsLoss which includes sigmoid
        )

    def forward(self, x):
        x = self.backbone(x)    # (batch, 2048, H, W)
        x = self.cbam(x)         # apply CBAM attention
        x = self.pool(x)         # (batch, 2048, 1, 1)
        x = self.mlp_head(x)     # (batch, 1)
        return x
