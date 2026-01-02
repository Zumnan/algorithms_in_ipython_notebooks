"""Lightweight mask predictor for remote-sensing referring segmentation.

Upgraded innovations:
- Language-conditioned multiscale fusion (scale selection driven by text).
- Mask-aware FiLM pooling (ignores padded tokens).
- Uncertainty-guided refinement pass (refine hard pixels without extra labels).
"""

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SeparableConvBN(nn.Module):
    """Depthwise separable convolution + BN + activation."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: bool = True, p_dropout: float = 0.0):
        super().__init__()
        padding = k // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(p_dropout) if p_dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.act(x)


class CheapFrequencyGate(nn.Module):
    """Boundary emphasis with Laplacian/blur and a learnable gate."""

    def __init__(self, channels: int):
        super().__init__()
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.groups = channels

        blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer("blur", blur.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = F.conv2d(x, self.weight, padding=1, groups=self.groups)
        low = F.conv2d(x, self.blur, padding=1, groups=self.groups)
        gate = self.gate(torch.cat([high, low], dim=1))
        return gate * high + (1 - gate) * low


class SobelEdgeGate(nn.Module):
    """Lightweight Sobel filter to reinforce boundary-sensitive channels."""

    def __init__(self, channels: int):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.groups = channels
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.groups)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.groups)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag * self.gate(x)


def _masked_mean(lang: torch.Tensor, lang_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    lang: (B, L, D) or (B, D, L) or (B, D)
    lang_mask: (B, L, 1) or (B, L) or (B, 1, L) (values 0/1)
    returns: (B, D)
    """
    if lang is None:
        raise ValueError("lang is None")
    if lang.dim() == 2:
        return lang

    # normalize to (B, L, D)
    if lang_mask is not None:
        if lang_mask.dim() == 3 and lang_mask.shape[1] == 1:
            lang_mask = lang_mask.transpose(1, 2)
        if lang_mask.dim() == 2:
            lang_mask = lang_mask.unsqueeze(-1)
        L = lang_mask.shape[1]
        if lang.dim() == 3 and lang.shape[1] != L and lang.shape[2] == L:
            lang = lang.transpose(1, 2)
    elif lang.dim() == 3 and lang.shape[1] > lang.shape[2]:
        # fall back: assume (B, D, L)
        lang = lang.transpose(1, 2)

    if lang_mask is None:
        return lang.mean(dim=1)

    if lang_mask.dim() == 3 and lang_mask.shape[1] == 1:
        lang_mask = lang_mask.transpose(1, 2)  # (B, L, 1)
    if lang_mask.dim() == 2:
        lang_mask = lang_mask.unsqueeze(-1)  # (B, L, 1)

    lang_mask = lang_mask.float()
    denom = torch.clamp(lang_mask.sum(dim=1), min=1.0)  # (B,1)
    return (lang * lang_mask).sum(dim=1) / denom


class LevelWeighter(nn.Module):
    """Language-conditioned softmax fusion over feature levels."""

    def __init__(self, num_levels: int, hidden_dim: int, lang_dim: int = 768, temperature: float = 1.0):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Static learnable logits (keeps behavior stable even when lang is noisy)
        self.static_logits = nn.Parameter(torch.zeros(num_levels))

        # Language-conditioned logits (novel in your stack)
        self.lang_mlp = nn.Sequential(
            nn.Linear(lang_dim, max(lang_dim // 4, 64)),
            nn.GELU(),
            nn.Linear(max(lang_dim // 4, 64), num_levels),
        )

        self.mix = SeparableConvBN(hidden_dim, hidden_dim, p_dropout=0.05)
        self._last_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        lang: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # feats: list of [B, hidden, H, W]
        logits = self.static_logits.view(1, -1)  # (1, L)

        if lang is not None:
            pooled = _masked_mean(lang if lang.dim() != 3 else lang, lang_mask)  # (B, D)
            logits = logits + self.lang_mlp(pooled)  # (B, L)

        weights = F.softmax(logits / self.temperature, dim=-1)  # (B, L)
        self._last_weights = weights.detach()
        fused = 0.0
        for i, f in enumerate(feats):
            fused = fused + weights[:, i].view(-1, 1, 1, 1) * f
        return self.mix(fused)

    def debug_state(self) -> dict:
        if self._last_weights is None:
            return {"level_weights": None}
        mean_w = self._last_weights.mean(dim=0).cpu().tolist()
        return {"level_weights_mean": [round(v, 4) for v in mean_w]}


class LightweightFiLM(nn.Module):
    """Mask-aware FiLM modulation (ignores padded tokens)."""

    def __init__(self, channels: int, lang_dim: int):
        super().__init__()
        self.gamma = nn.Linear(lang_dim, channels)
        self.beta = nn.Linear(lang_dim, channels)
        self.gate = nn.Sequential(nn.Linear(lang_dim, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, lang: Optional[torch.Tensor], lang_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lang is None:
            return x
        pooled = _masked_mean(lang, lang_mask)  # (B, D)
        gate = self.gate(pooled).unsqueeze(-1).unsqueeze(-1)
        gamma = self.gamma(pooled).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(pooled).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gate * gamma) + gate * beta


class ResidualRefiner(nn.Module):
    """Two separable conv blocks with SE gating."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = SeparableConvBN(channels, channels, p_dropout=0.05)
        self.conv2 = SeparableConvBN(channels, channels, p_dropout=0.05)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.gate(x)
        return x + residual


class UncertaintyRefiner(nn.Module):
    """Refine features guided by uncertainty map from first-pass logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = SeparableConvBN(channels + 1, channels, k=3, p_dropout=0.05)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        # uncertainty: (B,1,H,W)
        if uncertainty.shape[-2:] != feat.shape[-2:]:
            uncertainty = F.interpolate(uncertainty, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([feat, uncertainty], dim=1))
        g = self.gate(x)
        return feat + self.scale * (g * x)


class RemoteMaskPredictor(nn.Module):
    """Decoder with language-conditioned fusion + uncertainty-guided refinement."""

    def __init__(
        self,
        in_channels: Iterable[int],
        out_channels: int = 2,
        hidden_dim: int = 96,
        lang_dim: int = 768,
    ):
        super().__init__()
        in_channels = list(in_channels)
        if len(in_channels) != 4:
            raise ValueError("Expected four feature levels (c1..c4) for fusion.")

        self.proj = nn.ModuleList([nn.Conv2d(ch, hidden_dim, 1, bias=False) for ch in in_channels])
        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in in_channels])

        self.freq = nn.ModuleList([CheapFrequencyGate(hidden_dim) for _ in in_channels])
        self.align = nn.ModuleList([SeparableConvBN(hidden_dim, hidden_dim, p_dropout=0.05) for _ in in_channels])

        self.weighter = LevelWeighter(num_levels=4, hidden_dim=hidden_dim, lang_dim=lang_dim, temperature=0.85)
        self.film = LightweightFiLM(hidden_dim, lang_dim)

        self.refine = ResidualRefiner(hidden_dim)
        self.boundary = CheapFrequencyGate(hidden_dim)
        self.sobel = SobelEdgeGate(hidden_dim)
        self.sobel_scale = nn.Parameter(torch.tensor(0.05))

        # Two-pass logits (pass-1 -> uncertainty -> pass-2)
        self.head0 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.uncert_refine = UncertaintyRefiner(hidden_dim)
        self.head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self._debug: dict = {}

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        lang: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("features must be a tuple of four tensors")

        c1, c2, c3, c4 = features
        base_h, base_w = c1.shape[-2:]

        aligned = []
        for idx, feat in enumerate((c1, c2, c3, c4)):
            x = self.proj[idx](feat)
            x = self.bn[idx](x)
            x = F.relu(x, inplace=True)
            x = self.freq[idx](x)
            if x.shape[-2:] != (base_h, base_w):
                x = F.interpolate(x, size=(base_h, base_w), mode="bilinear", align_corners=False)
            x = self.align[idx](x)
            aligned.append(x)

        fused = self.weighter(aligned, lang=lang, lang_mask=lang_mask)
        fused = self.film(fused, lang=lang, lang_mask=lang_mask)

        # boundary reinforcement (cheap)
        fused = fused + 0.15 * self.boundary(fused)
        fused = self.refine(fused)
        fused = fused + self.sobel_scale * self.sobel(fused)

        logits0 = self.head0(fused)
        prob = F.softmax(logits0, dim=1)
        uncertainty = 1.0 - prob.max(dim=1, keepdim=True)[0]  # (B,1,H,W)

        fused2 = self.uncert_refine(fused, uncertainty)
        logits = self.head(fused2)
        with torch.no_grad():
            self._debug = {
                "fused_mean": float(fused.mean().item()),
                "fused_std": float(fused.std().item()),
                "uncertainty_mean": float(uncertainty.mean().item()),
                "uncertainty_max": float(uncertainty.max().item()),
            }
        return logits

    def debug_state(self) -> dict:
        state = dict(self._debug)
        state.update(self.weighter.debug_state())
        return state


if __name__ == "__main__":
    torch.manual_seed(0)
    c1 = torch.randn(2, 96, 120, 120)
    c2 = torch.randn(2, 192, 60, 60)
    c3 = torch.randn(2, 384, 30, 30)
    c4 = torch.randn(2, 768, 15, 15)
    lang = torch.randn(2, 20, 768)
    lang_mask = torch.ones(2, 20, 1)

    model = RemoteMaskPredictor(in_channels=[96, 192, 384, 768], out_channels=2, hidden_dim=96, lang_dim=768)
    with torch.no_grad():
        out = model((c1, c2, c3, c4), lang=lang, lang_mask=lang_mask)
    print("Logits:", tuple(out.shape))
