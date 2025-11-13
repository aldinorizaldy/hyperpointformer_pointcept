import torch
import torch.nn as nn

from pointcept.models.builder import MODELS
# Import PointTransformer building blocks (adjust path if your file has a different name)
from .point_transformer_seg import TransitionDown, TransitionUp, Bottleneck
# Import the CrossAttentionOffset you saved in utils.py
from .utils import CrossAttentionOffset


class HyperPointFormer(nn.Module):
    """
    Multimodal Point Transformer segmentation model.
    Dual encoders (geometry + spectral) with cross-attention fusion at every scale.
    Decoder path follows PointTransformerSeg (Seg50 config by default).
    """

    def __init__(
        self,
        block=Bottleneck,
        blocks=[1, 2, 3, 5, 2],  # Seg50-like by default
        in_channels_geom=6,  # e.g., xyz + normXYZ
        in_channels_spect=12,  # e.g., RGB + I + HSI(8)
        num_classes=13,
        share_planes=8,
        stride=[1, 4, 4, 4, 4],
        nsample=[8, 16, 16, 16, 16],
        cross_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.in_channels_geom = in_channels_geom
        self.in_channels_spect = in_channels_spect

        # channel plan (same as PointTransformerSeg)
        self.planes = [32, 64, 128, 256, 512]
        self.blocks_cfg = blocks

        # Build dual encoders
        self.enc_geom = nn.ModuleList()
        self.enc_spect = nn.ModuleList()
        self.cross_geom = nn.ModuleList()
        self.cross_spect = nn.ModuleList()

        in_g = in_channels_geom
        in_s = in_channels_spect

        for i, plane in enumerate(self.planes):
            # TransitionDown consumes the current in_planes and outputs plane * expansion
            enc_g = self._make_enc(block, plane, blocks[i], share_planes, stride[i], nsample[i], in_g)
            enc_s = self._make_enc(block, plane, blocks[i], share_planes, stride[i], nsample[i], in_s)
            self.enc_geom.append(enc_g)
            self.enc_spect.append(enc_s)

            # Cross-attention for this scale (use plane as dim)
            self.cross_geom.append(CrossAttentionOffset(dim=plane, num_heads=cross_heads, dropout=dropout))
            self.cross_spect.append(CrossAttentionOffset(dim=plane, num_heads=cross_heads, dropout=dropout))

            # after first scale, subsequent in_planes become plane * expansion
            in_g = plane * block.expansion
            in_s = plane * block.expansion

        # Decoders (mirrors PointTransformerSeg)
        self.dec5 = self._make_dec(block, self.planes[4], self.blocks_cfg[4], share_planes, nsample[4], is_head=True)
        self.dec4 = self._make_dec(block, self.planes[3], self.blocks_cfg[3], share_planes, nsample[3])
        self.dec3 = self._make_dec(block, self.planes[2], self.blocks_cfg[2], share_planes, nsample[2])
        self.dec2 = self._make_dec(block, self.planes[1], self.blocks_cfg[1], share_planes, nsample[1])
        self.dec1 = self._make_dec(block, self.planes[0], self.blocks_cfg[0], share_planes, nsample[0])

        # Classification head
        self.cls = nn.Sequential(
            nn.Linear(self.planes[0], self.planes[0]),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.planes[0], num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16, in_planes=None):
        """
        Construct a single encoder stage (TransitionDown + multiple blocks).
        in_planes argument is used so we can create encoders for different input dims.
        """
        assert in_planes is not None, "_make_enc requires in_planes"
        layers = [TransitionDown(in_planes, planes * block.expansion, stride, nsample)]
        for _ in range(blocks):
            layers.append(block(planes * block.expansion, planes * block.expansion, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        """
        Construct a decoder stage: TransitionUp + multiple blocks (Bottleneck).
        Behavior mirrors PointTransformerSeg._make_dec
        """
        layers = [TransitionUp(planes * block.expansion, None if is_head else planes * block.expansion)]
        for _ in range(blocks):
            layers.append(block(planes * block.expansion, planes * block.expansion, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        """
        data_dict: {
            "coord": [N, 3],
            "feat":  [N, C_total],  # geometry features first then spectral: [geom | spect] is expected
            "offset": [B]           # cumulative points per sample
        }
        """
        p0 = data_dict["coord"]
        feat0 = data_dict["feat"]
        offset0 = data_dict["offset"].int()

        # Split features into geometry and spectral (user's convention)
        # Expectation: feat layout = [geom(6) | spect(12)] per point
        geom_feat = feat0[:, : self.in_channels_geom].contiguous()
        spect_feat = feat0[:, self.in_channels_geom : (self.in_channels_geom + self.in_channels_spect)].contiguous()

        # Lists to store per-level (p, x, o)
        pg_list = []
        xg_list = []
        og_list = []

        ps_list = []
        xs_list = []
        os_list = []

        fused_list = []  # store fused (p, x_sum, o) for decoder

        # Initial inputs
        p_g, x_g, o_g = p0, geom_feat, offset0
        p_s, x_s, o_s = p0, spect_feat, offset0

        # Encoder: run both encoders in parallel and fuse with cross-attention per level
        for i in range(len(self.planes)):
            # encode geometry branch
            p_g, x_g, o_g = self.enc_geom[i]([p_g, x_g, o_g])
            # encode spectral branch
            p_s, x_s, o_s = self.enc_spect[i]([p_s, x_s, o_s])

            # Note: p_g and p_s should match spatially (both are sampled same way if TransitionDown uses same stride).
            # Use the offset tensor from one branch (o_g) for batching info
            # apply cross-attention within each sample using offsets
            xg_f = self.cross_geom[i](x_g, x_s, o_g)  # geometry attends to spectral
            xs_f = self.cross_spect[i](x_s, x_g, o_g)  # spectral attends to geometry

            x_sum = xg_f + xs_f  # fused features at this scale

            # Save per-level outputs for skip connections
            pg_list.append(p_g)
            xg_list.append(x_g)
            og_list.append(o_g)

            ps_list.append(p_s)
            xs_list.append(x_s)
            os_list.append(o_s)

            fused_list.append((p_g, x_sum, o_g))

        # Bottleneck / decoder entry
        # Use last fused feature as decoder input (p5, x5, o5)
        p5, x5_fused, o5 = fused_list[-1]

        # dec5: special head stage in original PTv1
        # TransitionUp (dec5[0]) expects pxo or pxo + pxo2 depending on is_head; original flow:
        # x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x5_tmp = self.dec5[0]([p5, x5_fused, o5])  # TransitionUp (head) -> returns x (not a list)
        x5 = self.dec5[1:]([p5, x5_tmp, o5])[1]   # pass through Bottleneck(s), take x

        # dec4 ... dec1: each merges higher-level decoded features with the skip fused features
        # iterate decoder levels in descending order using fused_list
        # fused_list indices: [level0 (p1), level1 (p2), ..., level4 (p5)]
        # dec order: dec4 consumes p4 (fused_list[3]) and p5 (higher-level)
        x_high = x5
        p_high = p5
        o_high = o5

        # dec4
        p4, x4_fused, o4 = fused_list[-2]
        x4_tmp = self.dec4[0]([p4, x4_fused, o4], [p_high, x_high, o_high])
        x4 = self.dec4[1:]([p4, x4_tmp, o4])[1]

        # dec3
        p3, x3_fused, o3 = fused_list[-3]
        x3_tmp = self.dec3[0]([p3, x3_fused, o3], [p4, x4, o4])
        x3 = self.dec3[1:]([p3, x3_tmp, o3])[1]

        # dec2
        p2, x2_fused, o2 = fused_list[-4]
        x2_tmp = self.dec2[0]([p2, x2_fused, o2], [p3, x3, o3])
        x2 = self.dec2[1:]([p2, x2_tmp, o2])[1]

        # dec1
        p1, x1_fused, o1 = fused_list[-5]
        x1_tmp = self.dec1[0]([p1, x1_fused, o1], [p2, x2, o2])
        x1 = self.dec1[1:]([p1, x1_tmp, o1])[1]

        # Final classification head on x1 (point-level features of original resolution)
        x_out = self.cls(x1)
        return x_out


@MODELS.register_module("HyperPointFormer")
class HyperPointFormer(HyperPointFormer):
    def __init__(self, **kwargs):
        super(HyperPointFormer, self).__init__(block=Bottleneck, blocks=[1, 2, 3, 5, 2], **kwargs)
