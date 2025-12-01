import torch
import torch.nn as nn
import torch.nn.functional as F

torch.nn.LayerNorm

class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )

class BatchMeanPool:
    """Compute per-sample mean pooling given offsets (cumulative counts)."""
    @staticmethod
    def forward(x, offset):
        # x: (N, C), offset: (B,) cumulative counts (torch.int)
        B = offset.shape[0]
        outs = []
        s = 0
        for i in range(B):
            e = int(offset[i].item())
            if i == 0:
                cnt = e
            else:
                cnt = e - int(offset[i-1].item())
            xi = x[s:s+cnt]
            outs.append(xi.mean(dim=0, keepdim=True))
            s += cnt
        return torch.cat(outs, dim=0)  # (B, C)

class CrossViTPointFusion(nn.Module):
    """
    CrossViT-inspired fusion for point clouds.
    - Builds per-sample CLS tokens by mean-pooling features (per-branch).
    - Query = cls_branch_A, Keys/Values = tokens_branch_B
    - Returns updated per-point features for branch A: original_x + fuse(projected_cls_to_points)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.0, attn_drop=0.0, return_cls=False):
        """
        dim: embedding dim per-point (assumed same for both branches)
        num_heads: attention heads
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)   # will be applied to CLS (B,1,C)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)   # applied to all tokens (N, C)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # map updated CLS back to per-point shape (broadcast fusion)
        self.reproj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # small gating scalar
        self.gamma = nn.Parameter(torch.zeros(1))

        self.return_cls = return_cls

        # ensure dimension matches
        self.out_proj = nn.Linear(dim, dim)

    def _reshape_for_heads(self, x, B, L=None):
        # x: (B, L, C) or (total_tokens, C) => reshape to (B, H, L, C/H)
        # If input is (total_tokens, C) we expect L varies per sample; we avoid reshaping there.
        # This helper expects batched inputs with uniform L.
        assert L is not None
        x = x.reshape(B, L, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        return x  # (B, H, L, C/H)

    def forward(self, x_a, x_b, offset):
        """
        x_a: (N, C) features of branch A (geometry) aggregated for a batch of B samples
        x_b: (N, C) features of branch B (spectral) same ordering and offsets
        offset: (B,) cumulative counts per sample (torch.int)
        Returns:
            x_a_out: (N, C) fused features for branch A
            optionally cls tokens if return_cls True
        """
        device = x_a.device
        B = offset.shape[0]

        # 1) compute per-sample CLS tokens by mean pooling
        cls_a = BatchMeanPool.forward(x_a, offset)  # (B, C)
        cls_b = BatchMeanPool.forward(x_b, offset)  # (B, C)

        # 2) build Q from cls_a  -> (B, 1, C)
        q = self.wq(cls_a).unsqueeze(1)  # (B, 1, C)
        # reshape to heads
        qh = q.reshape(B, 1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # (B, H, 1, C/H)

        # 3) K, V from x_b (all tokens). We must process per-sample variable-length tokens.
        # We'll compute attention per-sample in a loop (still O(N·C), OK).
        out_cls_list = []
        s = 0
        prev_e = 0
        # precompute linear projections for all tokens to avoid repeated work
        kb_all = self.wk(x_b)   # (N, C)
        vb_all = self.wv(x_b)   # (N, C)

        for i in range(B):
            e = int(offset[i].item())
            if i == 0:
                cnt = e
                s_i = 0
            else:
                cnt = e - int(offset[i-1].item())
                s_i = int(offset[i-1].item())
            # slice per-sample tokens
            kb = kb_all[s_i:s_i+cnt]  # (cnt, C)
            vb = vb_all[s_i:s_i+cnt]  # (cnt, C)
            # reshape to heads: (1, H, cnt, C/H)
            kb_h = kb.reshape(1, cnt, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
            vb_h = vb.reshape(1, cnt, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)

            # qh for this sample: (1, H, 1, C/H)
            qh_i = qh[i:i+1]  # (1, H, 1, C/H)

            # attention: (1,H,1,c) @ (1,H,c,cnt) -> (1,H,1,cnt)
            attn = (qh_i @ kb_h.transpose(-2, -1)) * self.scale  # (1,H,1,cnt)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            # weighted sum: (1,H,1,cnt) @ (1,H,cnt,c) -> (1,H,1,c)
            out_h = (attn @ vb_h)  # (1,H,1,C/H)
            out_h = out_h.transpose(1, 2).reshape(1, 1, self.dim)  # (1,1,C)
            out_cls_list.append(out_h.squeeze(0))  # (1, C)
            s += cnt

        # concat per-sample updated CLS: (B, 1, C)
        out_cls = torch.cat(out_cls_list, dim=0)  # (B, 1, C)
        out_cls = out_cls.squeeze(1)  # (B, C)
        out_cls = self.proj(out_cls)
        out_cls = self.proj_drop(out_cls)  # (B, C)

        # 4) map updated cls back to point space for branch A:
        # Broadcast the updated cls to all tokens in same sample and fuse
        s = 0
        x_a_out_list = []
        for i in range(B):
            e = int(offset[i].item())
            if i == 0:
                cnt = e
                s_i = 0
            else:
                cnt = e - int(offset[i-1].item())
                s_i = int(offset[i-1].item())
            cls_i = out_cls[i:i+1]  # (1, C)
            # broadcast and project
            cls_proj = self.reproj(cls_i)  # (1, C)
            cls_broadcast = cls_proj.repeat(cnt, 1)  # (cnt, C)
            xa_slice = x_a[s_i:s_i+cnt]  # (cnt, C)
            fused = xa_slice + self.gamma * cls_broadcast
            x_a_out_list.append(fused)
            s += cnt

        x_a_out = torch.cat(x_a_out_list, dim=0)  # (N, C)

        x_a_out = self.out_proj(x_a_out)

        if self.return_cls:
            return x_a_out, out_cls  # (N,C), (B,C)
        return x_a_out
