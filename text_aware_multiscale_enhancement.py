"""Next-generation text-aware multiscale enhancement for remote sensing RIS.

Key ideas
---------
- SpectralPyramidGather keeps low/high frequency evidence from each level on a shared grid.
- LanguagePyramidSummarizer builds coarse + fine language descriptors with mask awareness.
- SpatialLanguageInjector aligns spatial tokens with language using dual-path queries (content + geometry).
- RadianceBlendMixer stabilizes illumination shifts via learnable radiance priors and depthwise mixing.
- CrossScaleRedistributor re-injects refined global context back to each level with adaptive gates.
- NovaEnhancer orchestrates the pipeline, producing scale-aligned outputs ready for decoding.
- CAMRecorder + GradientCAMProjector provide Grad-CAM-style evidence maps for debugging and reporting.

The block implementations are refreshed with SOTA-inspired stability tricks (cosine attention,
language-aware gates, residual preservation) aimed at improving segmentation quality without
adding heavy compute.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class AdaptiveLogitGate(nn.Module):
    """Smooth gate with learnable temperature and offset for stable fusion.

    The temperature is constrained to a reasonable range via ``softplus`` to
    avoid saturating gradients and to keep gating inexpensive. Both the
    temperature and bias are broadcast across spatial dimensions.
    """

    def __init__(self, channels: int, init_temp: float = 1.5, min_temp: float = 0.5, max_temp: float = 10.0):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(channels) * init_temp)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.min_temp = min_temp
        self.max_temp = max_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        temp = torch.clamp(F.softplus(self.temp), self.min_temp, self.max_temp).view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return torch.sigmoid((x + bias) / temp)


class LanguagePyramidSummarizer(nn.Module):
    """Construct coarse + fine language descriptors using pooling over valid tokens.

    The summarizer is intentionally lightweight and can be reused across blocks
    to avoid recomputation. It gracefully handles inputs of shape ``(B, L, C)``
    or ``(B, C, L)``.
    """

    def __init__(self, lang_dim: int = 768, proj_dim: Optional[int] = None):
        super().__init__()
        proj_dim = proj_dim or lang_dim
        self.coarse = nn.Linear(lang_dim, proj_dim)
        self.fine = nn.Linear(lang_dim, proj_dim)
        self.coarse_query = nn.Linear(lang_dim, 1)
        self.fine_query = nn.Linear(lang_dim, 1)
        self._last_stats: Optional[dict] = None

    def forward(self, lang_tokens: torch.Tensor, lang_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lang_mask.dim() == 3 and lang_mask.shape[1] == 1:
            lang_mask = lang_mask.transpose(1, 2)
        if lang_mask.dim() == 2:
            lang_mask = lang_mask.unsqueeze(-1)
        if lang_tokens.dim() == 3 and lang_tokens.shape[1] != lang_mask.shape[1] and lang_tokens.shape[2] == lang_mask.shape[1]:
            lang_tokens = lang_tokens.transpose(1, 2)
        L = lang_tokens.shape[1]
        # Ensure mask is float for downstream reductions (PyTorch 1.10 compatibility)
        lang_mask = lang_mask[..., :L].float()

        # Collapse mask to (B, L) to avoid accidental rank-3 broadcasting
        mask = lang_mask.float().squeeze(-1)
        safe_mask = torch.clamp(mask, min=1e-6)

        coarse_logits = self.coarse_query(lang_tokens).squeeze(-1)
        coarse_logits = coarse_logits.masked_fill(mask == 0, -1e4)
        coarse_attn = torch.softmax(coarse_logits, dim=1)
        coarse = (lang_tokens * coarse_attn.unsqueeze(-1)).sum(dim=1)

        fine_logits = self.fine_query(lang_tokens).squeeze(-1)
        token_norm = torch.norm(lang_tokens, dim=-1)
        fine_logits = fine_logits + token_norm
        fine_logits = fine_logits.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(fine_logits, dim=1)

        sorted_weights, sorted_idx = torch.sort(weights, dim=1, descending=True)
        ranks = torch.arange(weights.shape[1], device=weights.device).unsqueeze(0).expand_as(sorted_weights)
        topk = torch.clamp((safe_mask.sum(dim=1, keepdim=True) / 2).long(), min=1)
        topk_mask = (ranks < topk).float()

        topk_tokens = torch.gather(
            lang_tokens,
            1,
            sorted_idx.unsqueeze(-1).expand(-1, -1, lang_tokens.shape[-1]),
        )
        # topk_mask: (B, L) -> (B, L, 1) for broadcast against gathered tokens
        fine = (topk_tokens * topk_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(topk_mask.sum(dim=1, keepdim=True), min=1.0)

        with torch.no_grad():
            valid_ratio = float(mask.mean().item())
            topk_ratio = float((topk_mask.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).mean().item())
            self._last_stats = {
                "lang_valid_ratio": valid_ratio,
                "lang_topk_ratio": topk_ratio,
            }

        return self.coarse(coarse), self.fine(fine)

    def debug_state(self) -> Optional[dict]:
        return self._last_stats


class SpatialLanguageInjector(nn.Module):
    """Language-guided spatial alignment with dual-path queries (content + geometry)."""

    def __init__(self, dim: int, lang_dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_content = nn.Linear(dim, dim)
        self.q_geom = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(lang_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_lang = nn.LayerNorm(lang_dim)
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.out_gate = nn.Parameter(torch.ones(1))
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, tokens: torch.Tensor, lang: torch.Tensor, lang_mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        if lang_mask.dim() == 3 and lang_mask.shape[1] == 1:
            lang_mask = lang_mask.transpose(1, 2)
        if lang_mask.dim() == 2:
            lang_mask = lang_mask.unsqueeze(-1)
        if lang.dim() == 3 and lang.shape[1] != lang_mask.shape[1] and lang.shape[2] == lang_mask.shape[1]:
            lang = lang.transpose(1, 2)
        L = lang.shape[1]
        lang_mask = lang_mask[..., :L].float()

        tokens = self.norm_tokens(tokens)
        lang = self.norm_lang(lang)

        q = self.q_content(tokens) + self.q_geom(pos)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(lang).view(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Cosine-style attention stabilizes training when token magnitudes differ across modalities.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.scale * self.attn_scale)
        mask = lang_mask.transpose(1, 2).unsqueeze(1)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        gate = torch.sigmoid(self.out_gate * lang_mask.mean(dim=1, keepdim=True))
        return self.out_proj(out) * gate + tokens * (1 - gate * self.res_scale)


class RadianceBlendMixer(nn.Module):
    """Feed-forward mixer with radiance priors and depthwise spatial mixing."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        kernel_size: int = 5,
        lang_dim: int = 768,
        lang_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        lang_proj_dim = lang_proj_dim or dim

        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.use_depthwise = kernel_size > 1
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.radiance_mean = nn.Parameter(torch.zeros(hidden_dim))
        self.radiance_std = nn.Parameter(torch.ones(hidden_dim))

        self.lang_proj_dim = lang_proj_dim
        self.lang_expect_dim = lang_proj_dim
        self.lang_adapter = None
        if lang_dim != lang_proj_dim:
            self.lang_adapter = nn.Linear(lang_dim, lang_proj_dim)
        self.lang_mean_proj = nn.Linear(lang_proj_dim, hidden_dim)
        self.lang_std_proj = nn.Linear(lang_proj_dim, hidden_dim)
        self.lang_gate = nn.Linear(lang_proj_dim, dim)
        self.mix_scale = nn.Parameter(torch.zeros(1))
        self.res_scale = nn.Parameter(torch.ones(1))

    def _collapse_lang(self, lang: torch.Tensor) -> torch.Tensor:
        if lang.dim() == 2:
            return lang
        B = lang.shape[0]
        lang = lang.view(B, -1, lang.shape[-1])
        return lang.mean(dim=1)

    def _resize_to_hidden(self, lang_vec: torch.Tensor) -> torch.Tensor:
        if lang_vec.shape[1] == self.hidden_dim:
            return lang_vec
        pooled = F.adaptive_avg_pool1d(lang_vec.unsqueeze(1), self.hidden_dim)
        return pooled.squeeze(1)

    def _ensure_lang_dim(self, lang: torch.Tensor) -> torch.Tensor:
        if lang.shape[-1] == self.lang_expect_dim:
            return lang
        if self.lang_adapter is not None:
            return self.lang_adapter(lang)
        pooled = F.adaptive_avg_pool1d(lang.unsqueeze(1), self.lang_expect_dim)
        return pooled.squeeze(1)

    def forward(self, tokens: torch.Tensor, H: int, W: int, lang_coarse: torch.Tensor, lang_fine: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens)
        x = self.activation(self.fc1(x))

        B, N, C = x.shape
        spatial = x.transpose(1, 2).reshape(B, C, H, W)
        if self.use_depthwise:
            spatial = self.dwconv(spatial)

        lang_coarse = self._ensure_lang_dim(lang_coarse)
        lang_fine = self._ensure_lang_dim(lang_fine)
        lang_mean = self.lang_mean_proj(lang_coarse)
        lang_std = self.lang_std_proj(lang_fine)
        lang_gate = torch.sigmoid(self.lang_gate(lang_coarse))

        lang_mean = self._resize_to_hidden(self._collapse_lang(lang_mean))
        lang_std = self._resize_to_hidden(self._collapse_lang(lang_std))

        lang_mean = lang_mean.view(B, self.hidden_dim, 1, 1)
        lang_std = lang_std.view(B, self.hidden_dim, 1, 1)
        mean = self.radiance_mean.view(1, self.hidden_dim, 1, 1) + lang_mean
        std = torch.clamp(self.radiance_std.view(1, self.hidden_dim, 1, 1) + lang_std, min=1e-3)
        spatial = (spatial - mean) / std

        x = spatial.flatten(2).transpose(1, 2)
        x = self.dropout(self.fc2(x))
        gate = lang_gate.unsqueeze(1).expand_as(x)
        mixed = x * (1 + self.mix_scale * gate)
        return tokens + self.res_scale * mixed


class CrossScaleRedistributor(nn.Module):
    """Fuse refined global map into each scale with language-conditioned channel gating."""

    def __init__(self, local_channels: int, global_channels: Optional[int] = None, lang_dim: int = 192):
        super().__init__()
        global_channels = global_channels or local_channels
        self.local_proj = nn.Conv2d(local_channels, local_channels, kernel_size=1)
        self.global_proj = nn.Conv2d(global_channels, local_channels, kernel_size=1)
        self.mix = nn.Conv2d(local_channels * 2, local_channels, kernel_size=3, padding=1, groups=local_channels)
        self.gate = AdaptiveLogitGate(local_channels)
        self.norm = nn.BatchNorm2d(local_channels)

        # language -> channel gate (novel in your enhancer)
        self.lang_gate = nn.Sequential(
            nn.Linear(lang_dim, max(lang_dim // 2, 32)),
            nn.GELU(),
            nn.Linear(max(lang_dim // 2, 32), local_channels),
            nn.Sigmoid(),
        )

        self.res_scale = nn.Parameter(torch.zeros(1))
        self.sim_scale = nn.Parameter(torch.ones(1))

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor, lang_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = local_feat.shape
        local = self.local_proj(local_feat)
        global_up = F.interpolate(self.global_proj(global_feat), size=(H, W), mode="bilinear", align_corners=False)

        gate_map = self.gate(self.mix(torch.cat([local, global_up], dim=1)))
        similarity = torch.mean(F.normalize(local, dim=1) * F.normalize(global_up, dim=1), dim=1, keepdim=True)
        gate_map = gate_map + self.sim_scale * similarity

        fused = local * gate_map + global_up * (1 - gate_map)

        if lang_vec is not None:
            ch_gate = self.lang_gate(lang_vec).view(B, C, 1, 1)
            fused = fused * (0.5 + ch_gate)  # keep stable scale

        return self.norm(fused + self.res_scale * local_feat)



class SpectralPyramidGather(nn.Module):
    """Aggregate multi-scale maps onto a shared grid with language-conditioned low/high blending."""

    def __init__(self, stride: int = 4, lang_dim: int = 192):
        super().__init__()
        self.stride = stride

        # language -> (low/high) gate shared across levels (simple + robust)
        self.lang_gate = nn.Sequential(
            nn.Linear(lang_dim, max(lang_dim // 2, 32)),
            nn.GELU(),
            nn.Linear(max(lang_dim // 2, 32), 2),
        )
        self.blend = nn.Parameter(torch.tensor(0.5))
        self._last_stats: Optional[dict] = None

    def forward(self, inputs: Sequence[torch.Tensor], lang_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int, int]:
        B, _, H0, W0 = inputs[0].shape
        min_h = min(feat.shape[-2] for feat in inputs)
        min_w = min(feat.shape[-1] for feat in inputs)
        target_h = max(1, (H0 + self.stride - 1) // self.stride)
        target_w = max(1, (W0 + self.stride - 1) // self.stride)
        max_upsample = 2
        target_h = min(target_h, min_h * max_upsample)
        target_w = min(target_w, min_w * max_upsample)

        if lang_vec is not None:
            # lang_vec: (B, D)
            g = torch.softmax(self.lang_gate(lang_vec), dim=-1)  # (B,2)
            w_low = g[:, 0].view(B, 1, 1, 1)
            w_high = g[:, 1].view(B, 1, 1, 1)
        else:
            w_low = w_high = None

        gathered = []
        for feat in inputs:
            feat_h, feat_w = feat.shape[-2:]
            if target_h > feat_h or target_w > feat_w:
                low = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
                high = torch.zeros_like(low)
            else:
                low = F.adaptive_avg_pool2d(feat, (target_h, target_w))
                high = feat - F.interpolate(low, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                high = F.adaptive_max_pool2d(high, (target_h, target_w))

            if w_low is not None:
                low = low * w_low
                high = high * w_high

            gathered.append(torch.cat([low, high], dim=1))

        fused = torch.cat(gathered, dim=1)
        weight = torch.clamp(self.blend, 0.0, 1.0)
        with torch.no_grad():
            self._last_stats = {
                "gather_weight": float(weight.item()),
                "gather_target_hw": (int(target_h), int(target_w)),
                "w_low_mean": float(w_low.mean().item()) if w_low is not None else None,
                "w_high_mean": float(w_high.mean().item()) if w_high is not None else None,
            }
        return fused, target_h, target_w

    def debug_state(self) -> Optional[dict]:
        return self._last_stats



class GradientCAMProjector(nn.Module):
    """Produces Grad-CAM style maps from captured activations and gradients."""

    def __init__(self, in_channels: int, out_channels: int = 1, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Conv2d(1, out_channels, kernel_size=1)
        self.eps = eps

    def forward(self, feat: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feat, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = self.proj(cam)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + self.eps)
        return cam


class CAMRecorder:
    """Utility to capture activations and gradients for CAM generation."""

    def __init__(self):
        self.feat: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None

    def save_feat(self, feat: torch.Tensor) -> torch.Tensor:
        self.feat = feat
        return feat

    def save_grad(self, grad: torch.Tensor) -> torch.Tensor:
        self.grad = grad
        return grad

    def clear(self):
        self.feat = None
        self.grad = None

    def ready(self) -> bool:
        return self.feat is not None and self.grad is not None


class NovaBlock(nn.Module):
    """Alternates language-driven spatial alignment with radiance-blend mixing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        lang_dim: int = 768,
    ):
        super().__init__()
        self.pos_embed = nn.Linear(2, dim)
        self.lang_summary = LanguagePyramidSummarizer(lang_dim=lang_dim, proj_dim=dim)
        self.attn = SpatialLanguageInjector(dim, lang_dim=lang_dim, num_heads=num_heads, dropout=dropout)
        self.mixer = RadianceBlendMixer(dim, dropout=dropout, lang_dim=lang_dim)
        self.residual_attn = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.residual_mix = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_mix = nn.LayerNorm(dim)
        self._cached_pos: Optional[Tuple[int, int, torch.Tensor]] = None

    def forward(
        self,
        tokens: torch.Tensor,
        H: int,
        W: int,
        lang: torch.Tensor,
        lang_mask: torch.Tensor,
        lang_summary: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, N, _ = tokens.shape

        if self._cached_pos is None or self._cached_pos[0] != H or self._cached_pos[1] != W or self._cached_pos[2].device != tokens.device:
            y = torch.linspace(-1, 1, steps=H, device=tokens.device)
            x = torch.linspace(-1, 1, steps=W, device=tokens.device)
            grid_y, grid_x = torch.meshgrid(y, x)
            pos = torch.stack([grid_y, grid_x], dim=-1).view(1, N, 2)
            self._cached_pos = (H, W, pos)
        else:
            pos = self._cached_pos[2]

        pos = self.pos_embed(pos).expand(B, -1, -1)

        if lang_summary is None:
            lang_coarse, lang_fine = self.lang_summary(lang, lang_mask)
        else:
            lang_coarse, lang_fine = lang_summary

        attn_out = self.attn(self.norm_attn(tokens), lang, lang_mask, pos)
        tokens = tokens + self.residual_attn(attn_out)

        mix_out = self.mixer(self.norm_mix(tokens), H, W, lang_coarse, lang_fine)
        tokens = tokens + self.residual_mix(mix_out)
        return tokens


class NovaEnhancer(nn.Module):
    """Language-aware multiscale enhancer tailored for remote sensing RIS."""

    def __init__(
        self,
        dim: int,
        num_blocks: int = 3,
        channels: Optional[Sequence[int]] = None,
        downsample: int = 4,
        num_heads: int = 8,
        drop_path: float = 0.05,
        dropout: float = 0.0,
        enable_cam: bool = True,
        lang_dim: int = 768,
    ):
        super().__init__()
        self.dim = dim
        self.enable_cam = enable_cam
        if channels is None:
            base = max(1, dim // 15)
            channels = [base, base * 2, base * 4]
            remaining = dim - sum(channels)
            channels.append(max(base * 8, remaining))
        self.channels = list(channels)

        self.gather = SpectralPyramidGather(stride=downsample, lang_dim=dim)

        fused_channels = sum(c * 2 for c in self.channels)
        self.reduce = nn.Conv2d(fused_channels, dim, kernel_size=1)

        self.lang_summary = LanguagePyramidSummarizer(lang_dim=lang_dim, proj_dim=dim)
        self.blocks = nn.ModuleList(
            [
                NovaBlock(
                    dim,
                    num_heads=num_heads,
                    drop_path=drop_path,
                    dropout=dropout,
                    lang_dim=lang_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.post_norm = nn.LayerNorm(dim)
        self.post_bn = nn.BatchNorm2d(dim)

        self.global_projs = nn.ModuleList([nn.Conv2d(dim, ch, kernel_size=1) for ch in self.channels])
        self.redistributors = nn.ModuleList([CrossScaleRedistributor(local_ch, local_ch, lang_dim=dim) for local_ch in self.channels])


        self.cam_recorders = [CAMRecorder() for _ in self.channels]
        self.cam_heads = nn.ModuleList([GradientCAMProjector(ch) for ch in self.channels])
        self._last_stats: Optional[dict] = None

    def forward(self, inputs: Sequence[torch.Tensor], l: torch.Tensor, l_mask: torch.Tensor) -> List[torch.Tensor]:
        lang_summary = self.lang_summary(l, l_mask)
        lang_vec = 0.5 * (lang_summary[0] + lang_summary[1])  # (B, dim)
        fused, Ht, Wt = self.gather(inputs, lang_vec=lang_vec)

        fused = self.reduce(fused)

        lang_summary = self.lang_summary(l, l_mask)

        tokens = fused.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, Ht, Wt, l, l_mask, lang_summary)
        tokens = self.post_norm(tokens)

        refined = tokens.transpose(1, 2).reshape(-1, self.dim, Ht, Wt)
        refined = self.post_bn(refined)

        outputs: List[torch.Tensor] = []
        for feat, proj, redist, recorder in zip(inputs, self.global_projs, self.redistributors, self.cam_recorders):
            global_map = proj(refined)
            if self.enable_cam:
                recorder.clear()
                captured = recorder.save_feat(global_map)
                captured.register_hook(recorder.save_grad)
            outputs.append(redist(feat, global_map, lang_vec=lang_vec))

        with torch.no_grad():
            self._last_stats = {
                "enhancer_levels": len(outputs),
                "enhancer_lang_vec_mean": float(lang_vec.mean().item()),
                "enhancer_lang_vec_std": float(lang_vec.std().item()),
                "enhancer_refined_shape": tuple(refined.shape),
            }
        return outputs

    def debug_state(self) -> dict:
        state = dict(self._last_stats or {})
        gather_stats = self.gather.debug_state()
        if gather_stats:
            state.update({f"gather_{k}": v for k, v in gather_stats.items()})
        summary_stats = self.lang_summary.debug_state()
        if summary_stats:
            state.update(summary_stats)
        return state

    def get_last_cam(self) -> Optional[List[torch.Tensor]]:
        if not self.enable_cam:
            return None
        cams: List[torch.Tensor] = []
        for recorder, head in zip(self.cam_recorders, self.cam_heads):
            if not recorder.ready():
                return None
            cams.append(head(recorder.feat, recorder.grad))
        return cams


__all__ = [
    "NovaEnhancer",
    "NovaBlock",
    "AdaptiveLogitGate",
    "LanguagePyramidSummarizer",
    "SpatialLanguageInjector",
    "RadianceBlendMixer",
    "CrossScaleRedistributor",
    "SpectralPyramidGather",
    "GradientCAMProjector",
    "CAMRecorder",
]


if __name__ == "__main__":
    # Simple shape sanity check with Grad-CAM path
    enhancer = NovaEnhancer(dim=192, channels=[64, 96, 128, 192], num_blocks=2, enable_cam=True)
    l = torch.randn(2, 768, 20, requires_grad=True)
    l_mask = torch.ones(2, 20, 1)
    x1 = torch.randn(2, 64, 120, 120, requires_grad=True)
    x2 = torch.randn(2, 96, 60, 60, requires_grad=True)
    x3 = torch.randn(2, 128, 30, 30, requires_grad=True)
    x4 = torch.randn(2, 192, 15, 15, requires_grad=True)
    outputs = enhancer((x1, x2, x3, x4), l, l_mask)
    for i, out in enumerate(outputs, 1):
        print(f"Level {i}: {tuple(out.shape)}")

    loss = sum(out.mean() for out in outputs)
    loss.backward()

    cams = enhancer.get_last_cam()
    if cams is not None:
        for i, cam in enumerate(cams, 1):
            print(f"CAM {i}: {tuple(cam.shape)}")
    else:
        print("CAMs not ready")
