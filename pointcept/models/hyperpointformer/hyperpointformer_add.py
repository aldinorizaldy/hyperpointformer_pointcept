"""
HyperPointFormer, 
adapted from Point Transformer V1 for Semantic Segmentation
"""

import torch
import torch.nn as nn
import einops
import pointops

from pointcept.models.builder import MODELS
from .utils import LayerNorm1d

class EncoderStage(nn.Module):
    def __init__(self, down: nn.Module, blocks: list):
        """
        down: TransitionDown module
        blocks: list of Bottleneck / PointTransformerLayer blocks
        """
        super().__init__()
        self.down = down
        self.blocks = nn.ModuleList(blocks)

    def forward(self, pxo, idx=None, n_o=None):
        # pxo = [p, x, o]
        p, x, o = pxo

        # Run TransitionDown, pass external idx/n_o if provided
        p, x, o = self.down([p, x, o], idx=idx, n_o=n_o)

        # Pass through residual/transformer blocks
        for block in self.blocks:
            p, x, o = block([p, x, o])

        return [p, x, o]


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(
            x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True
        )
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo, idx=None, n_o=None):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            if idx is None or n_o is None:
                n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
                for i in range(1, o.shape[0]):
                    count += (o[i].item() - o[i - 1].item()) // self.stride
                    n_o.append(count)
                n_o = torch.cuda.IntTensor(n_o)
                idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1
            )
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

def compute_new_offsets(o, stride):
    n_o, count = [o[0].item() // stride], o[0].item() // stride
    for i in range(1, o.shape[0]):
        count += (o[i].item() - o[i - 1].item()) // stride
        n_o.append(count)
    return torch.cuda.IntTensor(n_o)


class HyperPointFormer(nn.Module):
    def __init__(self, block, blocks, lidar_in_channels=6, hs_in_channels = 6, num_classes=13):
        super().__init__()
        self.lidar_in_channels = lidar_in_channels
        self.hs_in_channels = hs_in_channels
        self.lidar_in_planes, self.hs_in_planes, planes = lidar_in_channels, hs_in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        
        self.enc1 = EncoderStage(
            down=TransitionDown(self.lidar_in_planes, planes[0], stride=stride[0], nsample=nsample[0]),
            blocks=[Bottleneck(planes[0], planes[0], share_planes, nsample=nsample[0]) for _ in range(blocks[0])]
        )

        self.enc1_HS = EncoderStage(
            down=TransitionDown(self.hs_in_planes, planes[0], stride=stride[0], nsample=nsample[0]),
            blocks=[Bottleneck(planes[0], planes[0], share_planes, nsample=nsample[0]) for _ in range(blocks[0])]
        )

        self.enc2 = EncoderStage(
            down=TransitionDown(planes[0], planes[1], stride=stride[1], nsample=nsample[1]),
            blocks=[Bottleneck(planes[1], planes[1], share_planes, nsample=nsample[1]) for _ in range(blocks[1])]
        )

        self.enc2_HS = EncoderStage(
            down=TransitionDown(planes[0], planes[1], stride=stride[1], nsample=nsample[1]),
            blocks=[Bottleneck(planes[1], planes[1], share_planes, nsample=nsample[1]) for _ in range(blocks[1])]
        )

        self.enc3 = EncoderStage(
            down=TransitionDown(planes[1], planes[2], stride=stride[2], nsample=nsample[2]),
            blocks=[Bottleneck(planes[2], planes[2], share_planes, nsample=nsample[2]) for _ in range(blocks[2])]
        )

        self.enc3_HS = EncoderStage(
            down=TransitionDown(planes[1], planes[2], stride=stride[2], nsample=nsample[2]),
            blocks=[Bottleneck(planes[2], planes[2], share_planes, nsample=nsample[2]) for _ in range(blocks[2])]
        )
        
        self.enc4 = EncoderStage(
            down=TransitionDown(planes[2], planes[3], stride=stride[3], nsample=nsample[3]),
            blocks=[Bottleneck(planes[3], planes[3], share_planes, nsample=nsample[3]) for _ in range(blocks[3])]
        )

        self.enc4_HS = EncoderStage(
            down=TransitionDown(planes[2], planes[3], stride=stride[3], nsample=nsample[3]),
            blocks=[Bottleneck(planes[3], planes[3], share_planes, nsample=nsample[3]) for _ in range(blocks[3])]
        )

        self.enc5 = EncoderStage(
            down=TransitionDown(planes[3], planes[4], stride=stride[4], nsample=nsample[4]),
            blocks=[Bottleneck(planes[4], planes[4], share_planes, nsample=nsample[4]) for _ in range(blocks[4])]
        )

        self.enc5_HS = EncoderStage(
            down=TransitionDown(planes[3], planes[4], stride=stride[4], nsample=nsample[4]),
            blocks=[Bottleneck(planes[4], planes[4], share_planes, nsample=nsample[4]) for _ in range(blocks[4])]
        )

        self.lidar_in_planes = planes[4]
        self.hs_in_planes = planes[4]

        self.dec5 = self._make_dec(
            block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True
        )  # transform p5
        self.dec4 = self._make_dec(
            block, planes[3], 1, share_planes, nsample=nsample[3]
        )  # fusion p5 and p4
        self.dec3 = self._make_dec(
            block, planes[2], 1, share_planes, nsample=nsample[2]
        )  # fusion p4 and p3
        self.dec2 = self._make_dec(
            block, planes[1], 1, share_planes, nsample=nsample[1]
        )  # fusion p3 and p2
        self.dec1 = self._make_dec(
            block, planes[0], 1, share_planes, nsample=nsample[0]
        )  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_dec(
        self, block, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        layers = [
            TransitionUp(self.lidar_in_planes, None if is_head else planes * block.expansion)
        ]
        self.lidar_in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.lidar_in_planes, self.lidar_in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        
        # split features
        x0, x0_HS  = torch.split(x0, [self.lidar_in_channels, self.hs_in_channels], dim=1)

        n_o1 = compute_new_offsets(o0, stride=self.enc1.down.stride)
        idx1 = pointops.farthest_point_sampling(p0, o0, n_o1)
        p1, x1, o1 = self.enc1([p0, x0, o0], idx=idx1, n_o=n_o1)
        _ , x1_HS, _ = self.enc1_HS([p0, x0_HS, o0], idx=idx1, n_o=n_o1)
        x1_fused = x1 + x1_HS

        n_o2 = compute_new_offsets(o1, stride=self.enc2.down.stride)
        idx2 = pointops.farthest_point_sampling(p1, o1, n_o2)
        p2, x2, o2 = self.enc2([p1, x1, o1], idx=idx2, n_o=n_o2)
        _ , x2_HS, _ = self.enc2_HS([p1, x1_HS, o1], idx=idx2, n_o=n_o2)
        x2_fused = x2 + x2_HS

        n_o3 = compute_new_offsets(o2, stride=self.enc3.down.stride)
        idx3 = pointops.farthest_point_sampling(p2, o2, n_o3)
        p3, x3, o3 = self.enc3([p2, x2, o2], idx=idx3, n_o=n_o3)
        _ , x3_HS, _ = self.enc3_HS([p2, x2_HS, o2], idx=idx3, n_o=n_o3)
        x3_fused = x3 + x3_HS

        n_o4 = compute_new_offsets(o3, stride=self.enc4.down.stride)
        idx4 = pointops.farthest_point_sampling(p3, o3, n_o4)
        p4, x4, o4 = self.enc4([p3, x3, o3], idx=idx4, n_o=n_o4)
        _ , x4_HS, _ = self.enc4_HS([p3, x3_HS, o3], idx=idx4, n_o=n_o4)
        x4_fused = x4 + x4_HS

        n_o5 = compute_new_offsets(o4, stride=self.enc5.down.stride)
        idx5 = pointops.farthest_point_sampling(p4, o4, n_o5)
        p5, x5, o5 = self.enc5([p4, x4, o4], idx=idx5, n_o=n_o5)
        _ , x5_HS, _ = self.enc5_HS([p4, x4_HS, o4], idx=idx5, n_o=n_o5)
        x5_fused = x5 + x5_HS

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5_fused, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4_fused, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3_fused, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2_fused, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1_fused, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x



@MODELS.register_module("HyperPointFormer_Add")
class HyperPointFormer(HyperPointFormer):
    def __init__(self, **kwargs):
        super(HyperPointFormer, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )
