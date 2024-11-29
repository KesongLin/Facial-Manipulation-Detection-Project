import torch
import torch.nn as nn
from typing import List


class Conv3DBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple = (3, 3, 3),
            stride: tuple = (1, 1, 1)
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNN3D(nn.Module):
    def __init__(
            self,
            channels: List[int],
            embedding_dim: int = 512
    ):
        super().__init__()

        # Input shape: [B, 3, T, H, W]
        self.blocks = nn.ModuleList([
            Conv3DBlock(3 if i == 0 else channels[i - 1], channels[i])
            for i in range(len(channels))
        ])

        # Calculate the output size after conv blocks
        self.gap = nn.AdaptiveAvgPool3d((16, 1, 1))  # Force temporal dim to 16

        # Project to embedding space
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Process through 3D CNN blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling with fixed temporal dimension
        x = self.gap(x)  # Shape: [B, C, 16, 1, 1]

        # Reshape: [B, 16, C]
        x = x.squeeze(-1).squeeze(-1).transpose(1, 2)

        # Project each temporal feature to embedding space
        x = self.projection(x)  # Shape: [B, 16, embedding_dim]

        return x