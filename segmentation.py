"""Factory functions for LAVT-style segmentation models (Nova stack).

This project now uses a MultiModal Swin backbone whose forward signature is:

    backbone(x_img, l_feats, l_mask3, t_feats=None, t_mask3=None, p_feats=None, p_mask3=None)
        -> (c1, c2, c3, c4)

Key behaviors preserved:
- Decoder wrapper can receive language tokens + mask (head not blind to text).
- Factory names and model class names unchanged (lavt / lavt_one).
- Robust local imports for PyCharm (no shell assumptions).

Notes on fusion heads
---------------------
The backbone supports per-stage fusion head counts via `num_heads_fusion=[h1,h2,h3,h4]`.
We reuse `args.mha` (format: "a-b-c-d") to set these values, defaulting to 1 head per stage.
"""

import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Ensure local import roots (PyCharm-friendly)
module_dir = os.path.join(os.path.dirname(__file__), "lib")
if module_dir not in sys.path:
    sys.path.append(module_dir)
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lib.mask_predictor import RemoteMaskPredictor
from lib.backbone import MultiModalSwinTransformer
from lib._utils import LAVT, LAVTOne, LoRAConfig
from lib.text_aware_multiscale_enhancement import NovaEnhancer

__all__ = ["lavt", "lavt_one"]


# ----------------------------- Decoder wrapper -----------------------------


class MaskDecoderWrapper(nn.Module):
    """Adapter for RemoteMaskPredictor with LAVT-style feature order.

    - Predictor expects (c1,c2,c3,c4)
    - LAVT calls classifier as (c4,c3,c2,c1) and may forward (lang, lang_mask)
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        hidden_dim: int,
        lang_dim: int,
        enhancer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.predictor = RemoteMaskPredictor(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            lang_dim=lang_dim,
        )
        self.enhancer = enhancer

    def forward(
        self,
        x_c4: torch.Tensor,
        x_c3: torch.Tensor,
        x_c2: torch.Tensor,
        x_c1: torch.Tensor,
        lang: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        feats = (x_c1, x_c2, x_c3, x_c4)
        if self.enhancer is not None and lang is not None and lang_mask is not None:
            feats = tuple(self.enhancer(feats, lang, lang_mask))
        return self.predictor(feats, lang=lang, lang_mask=lang_mask)


# ----------------------------- Swin config helpers -----------------------------


def _select_swin_config(args) -> Tuple[int, List[int], List[int]]:
    """Return (embed_dim, depths, num_heads) for Swin variants."""
    if args.swin_type == "tiny":
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == "small":
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == "base":
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == "large":
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        raise ValueError(f"Unsupported swin_type: {getattr(args, 'swin_type', None)}")
    return embed_dim, depths, num_heads


def _infer_window_size(pretrained: str, args) -> int:
    """Infer Swin window size from args/window12 or pretrained filename token."""
    if getattr(args, "window12", False):
        return 12
    if isinstance(pretrained, str) and ("window12" in pretrained):
        return 12
    return 7


def _parse_mha(mha: str) -> List[int]:
    """
    Parse args.mha ("a-b-c-d") into fusion head counts [a,b,c,d].
    Falls back to [1,1,1,1] on empty/invalid values.
    """
    default = [1, 1, 1, 1]
    if not isinstance(mha, str) or not mha.strip():
        return default

    parts = [p.strip() for p in mha.strip().split("-") if p.strip()]
    if len(parts) != 4:
        return default

    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
            out.append(max(1, v))
        except Exception:
            return default
    return out


# ----------------------------- Builders -----------------------------


def _build_backbone(args, pretrained: str):
    """
    Build the new MultiModalSwinTransformer backbone.

    Expected backbone signature:
        forward(x_img, l_feats, l_mask3) -> tuple(c1,c2,c3,c4)

    Important args used:
    - args.swin_type
    - args.use_checkpoint
    - args.fusion_drop
    - args.mha (optional, "a-b-c-d" fusion heads)
    - args.lang_dim (language hidden size, default 768)
    """
    embed_dim, depths, num_heads = _select_swin_config(args)
    window_size = _infer_window_size(pretrained, args)

    # Fusion heads: default 1 per stage unless args.mha is provided.
    num_heads_fusion = _parse_mha(getattr(args, "mha", ""))

    backbone = MultiModalSwinTransformer(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=bool(getattr(args, "use_checkpoint", False)),
        num_heads_fusion=num_heads_fusion,
        fusion_drop=float(getattr(args, "fusion_drop", 0.0)),
        language_dim=int(getattr(args, "lang_dim", 768)),
    )

    # Init weights (pretrained optional)
    if pretrained:
        print("Initializing Multi-modal Swin Transformer weights from " + str(pretrained))
        backbone.init_weights(pretrained=pretrained)
    else:
        print("Randomly initialize Multi-modal Swin Transformer weights.")
        backbone.init_weights(pretrained=None)

    return backbone


def _build_decoder(embed_dim: int, args) -> MaskDecoderWrapper:
    in_channels: List[int] = [int(embed_dim * (2**i)) for i in range(4)]
    out_channels = int(getattr(args, "num_classes", 2))
    lang_dim = int(getattr(args, "lang_dim", 768))

    hidden_dim = getattr(args, "decoder_dim", None)
    if hidden_dim is None:
        hidden_dim = embed_dim
    hidden_dim = int(hidden_dim)

    enhancer = None
    enhancer_blocks = int(getattr(args, "enhancer_blocks", 0) or 0)
    if enhancer_blocks > 0:
        enhancer = NovaEnhancer(
            dim=hidden_dim,
            num_blocks=enhancer_blocks,
            channels=in_channels,
            downsample=int(getattr(args, "enhancer_downsample", 4)),
            num_heads=int(getattr(args, "enhancer_heads", 8)),
            drop_path=float(getattr(args, "enhancer_drop_path", 0.05)),
            dropout=float(getattr(args, "enhancer_dropout", 0.0)),
            enable_cam=bool(getattr(args, "enable_cam", False)),
            lang_dim=lang_dim,
        )

    return MaskDecoderWrapper(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        lang_dim=lang_dim,
        enhancer=enhancer,
    )


def _maybe_lora_config(args) -> Optional["LoRAConfig"]:
    if getattr(args, "lora_rank", None) is None:
        return None
    return LoRAConfig(
        rank=int(args.lora_rank),
        alpha=float(getattr(args, "lora_alpha", 32.0)),
        dropout=float(getattr(args, "lora_dropout", 0.1)),
        full_match=bool(getattr(args, "lora_full_match", False)),
    )


def _segm_lavt(pretrained: str, args) -> nn.Module:
    if args is None:
        raise ValueError("args must be provided (use args.py parser or a compatible args object).")

    backbone = _build_backbone(args, pretrained)
    embed_dim, _, _ = _select_swin_config(args)
    classifier = _build_decoder(embed_dim=embed_dim, args=args)
    model = LAVTOne(backbone, classifier, args, lora_config=_maybe_lora_config(args))
    return model


def _segm_lavt_one(pretrained: str, args) -> nn.Module:
    return _segm_lavt(pretrained, args)


# ----------------------------- Public factory functions -----------------------------


def lavt(pretrained: str = "", args=None) -> nn.Module:
    return _segm_lavt(pretrained, args)


def lavt_one(pretrained: str = "", args=None) -> nn.Module:
    return _segm_lavt_one(pretrained, args)


# ----------------------------- Local sanity check -----------------------------

if __name__ == "__main__":
    class DummyArgs:
        # backbone
        swin_type = "tiny"
        use_checkpoint = False
        fusion_drop = 0.0
        mha = ""          # can be "1-1-1-1" or "2-2-2-2" etc
        window12 = False  # if True forces window size 12

        # text
        ck_bert = "/10T/students/doctor/2025/zum/models/bert-base-uncased/"
        lang_dim = 768

        # decoder / task
        num_classes = 2
        decoder_dim = 96

        # LoRA (off by default)
        lora_rank = None
        lora_alpha = 32.0
        lora_dropout = 0.1
        lora_full_match = False

        # optional flags used by _utils
        triple_text_encode = False
        train_text_encoder = False
        force_single_stream_backbone = True

    args = DummyArgs()
    model = lavt(pretrained="", args=args)

    # Minimal forward sanity:
    # - text ids: (B,L)
    # - masks: (B,L) or (B,1,L) or (B,L,1)
    torch.manual_seed(0)
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_text = torch.ones(1, 16, dtype=torch.long)
    dummy_lmask = torch.ones(1, 16, dtype=torch.long)
    dummy_tmask = torch.ones(1, 16, dtype=torch.long)
    dummy_pmask = torch.ones(1, 16, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(dummy_img, dummy_text, dummy_lmask, dummy_tmask, dummy_pmask)
    print("Segmentation logits:", tuple(out.shape))
