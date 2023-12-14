import torch
import torch.nn as nn
from .core import FPN, CTBlock


class Config:
    num_points: int = 40
    scale: int = 1
    approx_factor: float = 0.004
    dis_threshold: float = 0.3
    cls_threshold: float = 0.8


class TextNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.args = Config()

        # Feature Pyramid Network (including FSNet)
        self.fpn = FPN()

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

        # Postprocessing + CTBlock
        self.BPN = CTBlock(self.args, seg_channel=32 + 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.training is False
        # assert x.shape[0] == 1, "Only support batch size 1"

        fpn_feats = self.fpn(x)  # (B, 32, H, W)
        preds, others = self.seg_head(fpn_feats).split([2, 2], dim=1)  # (B, 4, H, W)
        return torch.sigmoid(preds).permute(0, 2, 3, 1)  # (B, H, W, 2)
        preds = torch.sigmoid(preds)
        feats = torch.cat([fpn_feats, preds, others], dim=1)
        py_preds, *_ = self.BPN(feats, seg_preds=preds)
        return py_preds[-1]
