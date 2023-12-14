from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

from .transformer import Transformer
from .utils import get_sample_point


class CTBlock(nn.Module):
    def __init__(self, args, seg_channel: int) -> None:
        super().__init__()
        self.args = args
        self.seg_channel = seg_channel
        self.clip_dis = 100
        self.midline_preds = nn.ModuleList()
        self.contour_preds = nn.ModuleList()
        self.iter = 3  # 3
        for i in range(self.iter):
            self.midline_preds.append(
                Transformer(
                    seg_channel,
                    128,
                    num_heads=8,
                    dim_feedforward=1024,
                    drop_rate=0.0,
                    if_resi=True,
                    block_nums=3,
                    pred_num=2,
                    batch_first=False,
                )
            )
            self.contour_preds.append(
                Transformer(
                    seg_channel,
                    128,
                    num_heads=8,
                    dim_feedforward=1024,
                    drop_rate=0.0,
                    if_resi=True,
                    block_nums=3,
                    pred_num=2,
                    batch_first=False,
                )
            )
        if not self.training:
            self.iter = 1

    def get_boundary_proposal(self, seg_preds: torch.Tensor):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, :].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > self.args.dis_threshold
            ret, labels = cv2.connectedComponents(
                dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U
            )
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = cls_preds[bid][text_mask].mean()
                if confidence < self.args.cls_threshold:
                    continue

                confidences.append(confidence)
                inds.append(bid)

                poly = get_sample_point(
                    text_mask,
                    self.args.num_points,
                    self.args.approx_factor,
                    scales=None,
                )
                init_polys.append(poly)

        return (
            torch.from_numpy(np.array(init_polys)).float(),
            torch.from_numpy(np.array(inds)),
            confidences,
        )

    def forward(self, features: torch.Tensor, seg_preds: Optional[torch.Tensor] = None):
        assert self.training is False
        device = features.device

        init_polys, inds, confidences = self.get_boundary_proposal(seg_preds=seg_preds)

        if init_polys.shape[0] == 0:
            return [init_polys, init_polys], inds, confidences, None

        init_polys = init_polys.to(device)
        inds = inds.to(device)

        mid_pt_num = init_polys.shape[1] // 2
        contours = [init_polys]
        midlines = []
        for i in range(self.iter):
            node_feat = get_node_feature(features, contours[i], inds)
            midline = (
                contours[i][:, :mid_pt_num]
                + torch.clamp(
                    self.midline_preds[i](node_feat).permute(0, 2, 1),
                    -self.clip_dis,
                    self.clip_dis,
                )[:, :mid_pt_num]
            )
            midlines.append(midline)

            mid_feat = get_node_feature(features, midline, inds)
            node_feat = torch.cat((node_feat, mid_feat), dim=2)
            new_contour = (
                contours[i]
                + torch.clamp(
                    self.contour_preds[i](node_feat).permute(0, 2, 1),
                    -self.clip_dis,
                    self.clip_dis,
                )[:, : self.args.num_points]
            )
            contours.append(new_contour)

        return contours, inds, confidences, midlines


def get_node_feature(features: torch.Tensor, img_poly, ind):
    device = features.device
    b, c, hi, wi = features.shape
    ho, wo = img_poly.shape[:2]
    img_poly = img_poly.clone().float()
    img_poly[..., 0] = img_poly[..., 0] / ((wi - 1) / 2.0) - 1
    img_poly[..., 1] = img_poly[..., 1] / ((hi - 1) / 2.0) - 1

    gcn_feature = torch.zeros([ho, c, wo]).to(device)
    for i, feat in enumerate(features.split(1, dim=0)):
        grid = img_poly[ind == i].unsqueeze(0).to(device)
        # print(feat.shape)
    for i in range(b):
        poly = img_poly[ind == i].unsqueeze(0).to(device)
        gcn_feature[ind == i] = F.grid_sample(
            features[i : i + 1], poly, align_corners=True
        )[0].permute(1, 0, 2)
    return gcn_feature
