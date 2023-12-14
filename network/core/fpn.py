from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsnet import FSNet


class FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FSNet()

        out_channels = self.backbone.channels * 4
        self.reduceLayer = reduceBlock(out_channels * 4, 32)
        self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)

        self.cbam2 = CBAM(out_channels, kernel_size=9)
        self.cbam3 = CBAM(out_channels, kernel_size=7)
        self.cbam4 = CBAM(out_channels, kernel_size=5)
        self.cbam5 = CBAM(out_channels, kernel_size=3)

    def upsample(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4, c5 = self.backbone(x)
        c2 = self.cbam2(c2)
        c3 = self.cbam3(c3)
        c4 = self.cbam4(c4)
        c5 = self.cbam5(c5)

        h, w = c2.shape[2:]
        c3 = self.upsample(c3, size=(h, w))
        c4 = self.upsample(c4, size=(h, w))
        c5 = self.upsample(c5, size=(h, w))

        c1 = self.upc1(self.reduceLayer(torch.cat([c2, c3, c4, c5], dim=1)))
        return c1


class reduceBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class CBAM(nn.Module):
    def __init__(self, inplane: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(inplane)
        self.sp = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sp(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)],
            dim=1,
        )
        x = self.conv1(x)
        return self.sigmoid(x)
