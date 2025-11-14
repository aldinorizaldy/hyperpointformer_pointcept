"""
Point Transformer V1 for Semantic Segmentation

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS
from .utils import CrossViTPointFusion

from .point_transformer_seg import TransitionDown, TransitionUp, Bottleneck

class EncoderBlock(nn.Module):
    def __init__(self, block, in_planes, planes, blocks, share_planes=8, stride=1, nsample=16):
        super().__init__()
        self.transition = TransitionDown(in_planes, planes * block.expansion, stride, nsample)
        self.blocks = nn.Sequential(
            *[block(planes * block.expansion, planes * block.expansion, share_planes, nsample=nsample)
              for _ in range(blocks)]
        )

    def forward(self, pxo, fps_idx=None):
        p, x, o, idx = self.transition(pxo, fps_idx)
        pxo = [p, x, o]
        p, x, o = self.blocks(pxo)
        return p, x, o, idx


class HyperPointFormer(nn.Module):
    def __init__(self, block, blocks, lidar_in_channels=6, hs_in_channels=8, num_classes=13):
        super().__init__()
        self.lidar_in_channels = lidar_in_channels
        self.hs_in_channels = hs_in_channels
        planes = [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        # Lidar branch
        in_planes_L = self.lidar_in_channels
        self.enc1_L = self._make_enc(block, in_planes_L, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        in_planes_L = planes[0] * block.expansion
        self.enc2_L = self._make_enc(block, in_planes_L, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        in_planes_L = planes[1] * block.expansion
        self.enc3_L = self._make_enc(block, in_planes_L, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        in_planes_L = planes[2] * block.expansion
        self.enc4_L = self._make_enc(block, in_planes_L, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        in_planes_L = planes[3] * block.expansion
        self.enc5_L = self._make_enc(block, in_planes_L, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        in_planes_L = planes[4] * block.expansion

        # HS branch
        in_planes_HS = self.hs_in_channels
        self.enc1_HS = self._make_enc(block, in_planes_HS, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        in_planes_HS = planes[0] * block.expansion
        self.enc2_HS = self._make_enc(block, in_planes_HS, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        in_planes_HS = planes[1] * block.expansion
        self.enc3_HS = self._make_enc(block, in_planes_HS, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        in_planes_HS = planes[2] * block.expansion
        self.enc4_HS = self._make_enc(block, in_planes_HS, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        in_planes_HS = planes[3] * block.expansion
        self.enc5_HS = self._make_enc(block, in_planes_HS, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        in_planes_HS = planes[4] * block.expansion
        self.fusion1 = CrossViTPointFusion(dim=planes[0])
        self.fusion2 = CrossViTPointFusion(dim=planes[1])
        self.fusion3 = CrossViTPointFusion(dim=planes[2])
        self.fusion4 = CrossViTPointFusion(dim=planes[3])
        self.fusion5 = CrossViTPointFusion(dim=planes[4])
        self.dec5 = self._make_dec(
            block, planes[4]*block.expansion, planes[4], 1, share_planes, nsample=nsample[4], is_head=True
        )  # transform p5
        self.dec4 = self._make_dec(
            block, planes[3]*block.expansion, planes[3], 1, share_planes, nsample=nsample[3]
        )  # fusion p5 and p4
        self.dec3 = self._make_dec(
            block, planes[2]*block.expansion, planes[2], 1, share_planes, nsample=nsample[2]
        )  # fusion p4 and p3
        self.dec2 = self._make_dec(
            block, planes[1]*block.expansion, planes[1], 1, share_planes, nsample=nsample[1]
        )  # fusion p3 and p2
        self.dec1 = self._make_dec(
            block, planes[0]*block.expansion, planes[0], 1, share_planes, nsample=nsample[0]
        )  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, block, in_planes, planes, blocks, share_planes=8, stride=1, nsample=16):
        return EncoderBlock(block, in_planes, planes, blocks, share_planes, stride, nsample)

    def _make_dec(
        self, block, in_planes, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        layers = []
        layers.append(TransitionUp(in_planes, None if is_head else planes * block.expansion))
        current_in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(current_in_planes, current_in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"] # (B,N,3)
        x0 = data_dict["feat"] # (B,N,C)
        o0 = data_dict["offset"].int()

        # Split features
        feat_L, feat_HS = torch.split(x0, [self.lidar_in_channels, self.hs_in_channels], dim=1)

        # LiDAR encoder
        p1_L, x1_L, o1_L, idx1 = self.enc1_L([p0, feat_L, o0])
        p2_L, x2_L, o2_L, idx2 = self.enc2_L([p1_L, x1_L, o1_L])
        p3_L, x3_L, o3_L, idx3 = self.enc3_L([p2_L, x2_L, o2_L])
        p4_L, x4_L, o4_L, idx4 = self.enc4_L([p3_L, x3_L, o3_L])
        p5_L, x5_L, o5_L, idx5 = self.enc5_L([p4_L, x4_L, o4_L])

        # HSI encoder
        p1_HS , x1_HS, o1_HS, _ = self.enc1_HS([p0, feat_HS, o0], fps_idx=idx1)
        p2_HS , x2_HS, o2_HS, _ = self.enc2_HS([p1_HS, x1_HS, o1_HS], fps_idx=idx2)
        p3_HS , x3_HS, o3_HS, _ = self.enc3_HS([p2_HS, x2_HS, o2_HS], fps_idx=idx3)
        p4_HS , x4_HS, o4_HS, _ = self.enc4_HS([p3_HS, x3_HS, o3_HS], fps_idx=idx4)
        p5_HS , x5_HS, o5_HS, _ = self.enc5_HS([p4_HS, x4_HS, o4_HS], fps_idx=idx5)

        # Cross Attenton Fusion
        x1_fused = self.fusion1(x1_L, x1_HS, o1_L)
        x2_fused = self.fusion2(x2_L, x2_HS, o2_L)
        x3_fused = self.fusion3(x3_L, x3_HS, o3_L)
        x4_fused = self.fusion4(x4_L, x4_HS, o4_L)
        x5_fused = self.fusion5(x5_L, x5_HS, o5_L)
        
        print("x4_fused", x4_fused.shape)
        print("TransitionUp expects", self.dec4[0].linear2.in_features)

        x5 = self.dec5[1:]([p5_L, self.dec5[0]([p5_L, x5_fused, o5_L]), o5_L])[1]
        x4 = self.dec4[1:]([p4_L, self.dec4[0]([p4_L, x4_fused, o4_L], [p5_L, x5, o5_L]), o4_L])[1]
        x3 = self.dec3[1:]([p3_L, self.dec3[0]([p3_L, x3_fused, o3_L], [p4_L, x4, o4_L]), o3_L])[1]
        x2 = self.dec2[1:]([p2_L, self.dec2[0]([p2_L, x2_fused, o2_L], [p3_L, x3, o3_L]), o2_L])[1]
        x1 = self.dec1[1:]([p1_L, self.dec1[0]([p1_L, x1_fused, o1_L], [p2_L, x2, o2_L]), o1_L])[1]
        x = self.cls(x1)
        return x



@MODELS.register_module("HyperPointFormer")
class HyperPointFormer(HyperPointFormer):
    def __init__(self, **kwargs):
        super(HyperPointFormer, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )
