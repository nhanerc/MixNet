from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSNet(nn.Module):
    def __init__(
        self, channels: int = 64, numofblocks: int = 4, layers: List[int] = [1, 2, 3, 4]
    ):
        super().__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.layers = layers
        self.blocks = nn.ModuleList()
        self.steps = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        for layer in layers:
            self.steps.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                )
            )
            next_channels = self.channels * layer
            for i in range(layer):
                tmp = [block(channels, next_channels)]
                for j in range(self.numofblocks - 1):
                    tmp.append(block(next_channels, next_channels))
                self.blocks.append(nn.Sequential(*tmp))
            channels = next_channels

    def forward(self, x):
        x = self.stem(x)  # 64 > H/4, W/4
        x1 = self.steps[0](x)

        x1 = self.blocks[0](x1)
        x2 = self.steps[1](x1)

        x1 = self.blocks[1](x1)
        x2 = self.blocks[2](x2)
        x3 = self.steps[2](x2)
        x1, x2 = shuffle_layer([x1, x2])

        x1 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)
        x3 = self.blocks[5](x3)
        x4 = self.steps[3](x3)
        x1, x2, x3 = shuffle_layer([x1, x2, x3])

        x1 = self.blocks[6](x1)
        x2 = self.blocks[7](x2)
        x3 = self.blocks[8](x3)
        x4 = self.blocks[9](x4)

        return x1, x2, x3, x4


class block(nn.Module):
    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resid:
            residual = self.resid(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        return x


def shuffle_layer(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    n = len(feats)
    feats_split = []
    for feat in feats:
        feats_split.append(feat.chunk(n, dim=1))

    for i in range(n):
        h, w = feats_split[i][i].shape[2:]
        feats_shuffle = []
        for j in range(n):
            x = feats_split[j][i]
            if i > j:
                x = F.avg_pool2d(x, kernel_size=(2 * (i - j)))
            elif i < j:
                x = F.interpolate(x, (h, w))
            feats_shuffle.append(x)
        feats[i] = torch.cat(feats_shuffle, dim=1)
    return feats
