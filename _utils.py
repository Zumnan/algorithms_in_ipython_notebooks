"""Utility modules for LAVT-style multimodal segmentation.

Core behavior
-------------
- TriMask Shared Encoding: run BERT once, derive (l/t/p) features via masks.
- Decoder language forwarding: pass lang tokens + mask to classifier when supported.
- Robust train() behavior: frozen text encoder stays eval to reduce drift/overfit.

Backbone compatibility
----------------------
This project now supports a Swin backbone whose forward signature is:

    backbone(x_img, l_feats, l_mask3) -> (c1, c2, c3, c4)

where:
- l_feats: (B, D, L) language features (channels-first over tokens)
- l_mask3: (B, L, 1) float mask (1 for valid tokens, 0 for padding)

For backward compatibility with older backbones, we also support:

    backbone(x_img, l_feats, l_mask3, t_feats, t_mask3, p_feats, p_mask3)

We auto-detect which signature works at runtime.

Mask robustness
---------------
- Accepts masks shaped (B, L), (B, 1, L), or (B, L, 1).
- Ensures masks match token length L (crop/pad).
- Ensures each sample has at least one valid token (forces token[0]=1 if needed).
- Ensures t_mask / p_mask are subsets of l_mask.
- If a per-stream mask becomes empty for a sample after sanitization, it falls back to l_mask
  for that sample (prevents all-masked attention paths).
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BertModel

logger = logging.getLogger(__name__)


# ----------------------------- LoRA (unchanged API) -----------------------------

@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: Sequence[str] = ("query", "key", "value", "dense", "out_proj")
    full_match: bool = False

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.

    IMPORTANT:
    - New LoRA parameters MUST be created on the same device/dtype as the base
      Linear weights to avoid CPU/GPU mismatch during forward().
    """

    def __init__(self, linear: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        config.validate()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Clone base weights/bias on the SAME device/dtype as the original module.
        w = linear.weight.detach()
        self.weight = nn.Parameter(w.clone(), requires_grad=False)

        if linear.bias is not None:
            b = linear.bias.detach()
            self.bias = nn.Parameter(b.clone(), requires_grad=False)
        else:
            self.bias = None

        # Create LoRA params on the same device/dtype as base weight.
        self.lora_A = nn.Parameter(self.weight.new_zeros((self.rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.rank)))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Use a device-aware buffer.
        self.register_buffer("_merged", self.weight.new_zeros((1,), dtype=torch.bool), persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self._merged.item():
            return F.linear(x, self.weight, self.bias)

        base = F.linear(x, self.weight, self.bias)
        lora_intermediate = F.linear(self.dropout(x), self.lora_A)
        lora_update = F.linear(lora_intermediate, self.lora_B)
        return base + self.scaling * lora_update

    @torch.no_grad()
    def merge_weights(self) -> None:
        if self._merged.item():
            return
        update = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.add_(update)
        self._merged.fill_(True)

    @torch.no_grad()
    def unmerge_weights(self) -> None:
        if not self._merged.item():
            return
        update = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.sub_(update)
        self._merged.fill_(False)


def _should_wrap(name: str, config: LoRAConfig) -> bool:
    if config.full_match:
        return name in config.target_modules
    return any(target in name for target in config.target_modules)


def apply_lora_adapters(module: nn.Module, config: LoRAConfig, *, _prefix: str = "") -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name

        if isinstance(child, nn.Linear) and _should_wrap(full_name, config):
            dev = child.weight.device
            dtype = child.weight.dtype

            wrapped = LoRALinear(child, config)
            # Ensure wrapped module matches original device/dtype.
            try:
                wrapped.to(device=dev, dtype=dtype)
            except Exception:
                try:
                    wrapped.to(dev)
                except Exception:
                    pass

            setattr(module, name, wrapped)
            replaced += 1
            logger.debug("Wrapped %s with LoRALinear", full_name)
        else:
            replaced += apply_lora_adapters(child, config, _prefix=full_name)

    if _prefix == "" and replaced == 0:
        logger.warning("No modules matched LoRA targets: %s", ", ".join(config.target_modules))
    return replaced


def freeze_non_lora_params(module: nn.Module) -> Tuple[int, int]:
    trainable, frozen = 0, 0
    for name, param in module.named_parameters():
        if any(key in name for key in ("lora_A", "lora_B")):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return trainable, frozen


# ----------------------------- Checkpoint load helper -----------------------------

def load_weights(model: nn.Module, load_path: str, strict: bool = False) -> nn.Module:
    checkpoint = torch.load(load_path, map_location="cpu")
    state_dict: Dict[str, Tensor] = checkpoint.get("model", checkpoint)
    current = model.state_dict()

    loaded, skipped, shape_mismatch = [], [], []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append(key)
            continue
        if value.shape != current[key].shape:
            shape_mismatch.append(key)
            continue
        current[key] = value
        loaded.append(key)

    model.load_state_dict(current, strict=strict)
    torch.cuda.empty_cache()

    logger.info(
        "Loaded %d tensors from %s (skipped %d, shape mismatches %d)",
        len(loaded), load_path, len(skipped), len(shape_mismatch)
    )
    if skipped:
        logger.debug("Skipped keys: %s", ", ".join(skipped))
    if shape_mismatch:
        logger.warning("Shape mismatches: %s", ", ".join(shape_mismatch))
    return model


# ----------------------------- Mask utilities -----------------------------

def _normalize_attn_mask(mask: Tensor, L: int, *, device: torch.device) -> Tensor:
    """
    Normalize token mask to shape (B, L) with values in {0,1} and dtype long.

    Accepts:
    - (B, L)
    - (B, 1, L)
    - (B, L, 1)

    Behavior:
    - crops/pads to L
    - ensures each sample has at least one valid token (forces mask[:,0]=1 if empty)
    """
    if mask is None:
        raise ValueError("mask is None")

    # Squeeze common singleton dimensions.
    if mask.dim() == 3 and mask.shape[1] == 1 and mask.shape[2] == L:
        mask = mask.squeeze(1)  # (B, L)
    elif mask.dim() == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(2)  # (B, L)
    elif mask.dim() == 3 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    if mask.dim() != 2:
        raise ValueError(f"Expected mask with 2 dims after squeeze, got {tuple(mask.shape)}")

    # Crop/pad to L
    if mask.size(1) > L:
        mask = mask[:, :L]
    elif mask.size(1) < L:
        pad = L - mask.size(1)
        mask = F.pad(mask, (0, pad), value=0)

    mask = mask.to(device=device)
    if mask.dtype != torch.long:
        mask = mask.long()
    mask = (mask > 0).long()

    # Ensure at least one token is valid per sample.
    sums = mask.sum(dim=1)
    if (sums == 0).any():
        mask = mask.clone()
        mask[sums == 0, 0] = 1
    return mask


def _sanitize_submask(base_mask: Tensor, sub_mask: Tensor, *, device: torch.device) -> Tensor:
    """
    Make sub_mask a robust subset of base_mask.
    - sub_mask := sub_mask & base_mask
    - if sub_mask becomes empty for a sample, fall back to base_mask for that sample
    """
    L = base_mask.size(1)
    sub = _normalize_attn_mask(sub_mask, L, device=device)
    sub = sub * base_mask

    sums = sub.sum(dim=1)
    if (sums == 0).any():
        sub = sub.clone()
        sub[sums == 0] = base_mask[sums == 0]
    return sub


def _mask_to_3d(mask: Tensor) -> Tensor:
    """Convert (B,L) / (B,1,L) / (B,L,1) to (B,L,1) float mask."""
    if mask.dim() == 3 and mask.shape[1] == 1:
        mask = mask.transpose(1, 2)  # (B, L, 1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # (B, L, 1)
    if mask.dim() != 3 or mask.shape[-1] != 1:
        raise ValueError(f"Expected mask to become (B,L,1), got {tuple(mask.shape)}")
    return mask.float()


def _masked_token_view(hidden: Tensor, mask_2d: Tensor, proj: nn.Module, null_token: Tensor) -> Tuple[Tensor, Tensor]:
    """
    hidden: (B, L, D)
    mask_2d: (B, L) in {0,1}
    returns:
      feats: (B, D, L)
      mask3: (B, L, 1) float
    """
    mask3 = _mask_to_3d(mask_2d)  # (B,L,1)
    h = hidden
    h_proj = proj(h)
    h = h + mask3 * h_proj
    h = h * mask3 + (1.0 - mask3) * null_token
    feats = h.permute(0, 2, 1).contiguous()  # (B,D,L)
    return feats, mask3


# ----------------------------- Classifier/backbone callers -----------------------------

def _call_classifier(classifier: nn.Module, x_c4, x_c3, x_c2, x_c1, lang=None, lang_mask=None):
    """
    Call classifier with optional language forwarding.
    - If classifier.forward accepts (lang/lang_mask), we pass them.
    - Otherwise we call the legacy signature.
    """
    try:
        sig = inspect.signature(classifier.forward)
        params = sig.parameters
        if ("lang" in params) or ("lang_mask" in params):
            return classifier(x_c4, x_c3, x_c2, x_c1, lang=lang, lang_mask=lang_mask)
    except Exception:
        pass
    return classifier(x_c4, x_c3, x_c2, x_c1)


def _call_backbone(
    backbone: nn.Module,
    x_img: Tensor,
    l_feats: Tensor,
    l_mask3: Tensor,
    t_feats: Optional[Tensor] = None,
    t_mask3: Optional[Tensor] = None,
    p_feats: Optional[Tensor] = None,
    p_mask3: Optional[Tensor] = None,
):
    """
    Backbone forward compatibility:
    - Preferred (new): backbone(x, l_feats, l_mask3)
    - Legacy (old):    backbone(x, l_feats, l_mask3, t_feats, t_mask3, p_feats, p_mask3)
    """
    if t_feats is not None and t_mask3 is not None and p_feats is not None and p_mask3 is not None:
        try:
            sig = inspect.signature(backbone.forward)
            params = sig.parameters
            if "t" in params or "t_mask" in params or "p" in params or "p_mask" in params:
                return backbone(x_img, l_feats, l_mask3, t_feats, t_mask3, p_feats, p_mask3)
        except Exception:
            pass
    try:
        return backbone(x_img, l_feats, l_mask3)
    except TypeError:
        if t_feats is None or t_mask3 is None or p_feats is None or p_mask3 is None:
            raise
        return backbone(x_img, l_feats, l_mask3, t_feats, t_mask3, p_feats, p_mask3)


# ----------------------------- Simple decode wrappers -----------------------------

class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.lang_dim = _infer_backbone_lang_dim(backbone)
        self.lang_proj: Optional[nn.Conv1d] = None

    def forward(self, x: Tensor, l_feats: Tensor, l_mask: Tensor) -> Tensor:  # type: ignore[override]
        input_shape = x.shape[-2:]

        # Accept l_mask as (B,L) or (B,L,1) and normalize for backbone.
        if l_mask.dim() == 3 and l_mask.shape[-1] == 1:
            l_mask3 = l_mask.float()
        else:
            l_mask3 = _mask_to_3d(l_mask)

        if self.lang_dim is not None and l_feats.size(1) != self.lang_dim:
            if self.lang_proj is None:
                self.lang_proj = nn.Conv1d(l_feats.size(1), self.lang_dim, kernel_size=1).to(
                    device=l_feats.device, dtype=l_feats.dtype
                )
            l_feats = self.lang_proj(l_feats)

        x_c1, x_c2, x_c3, x_c4 = _call_backbone(self.backbone, x, l_feats, l_mask3)
        lang_tokens = l_feats.permute(0, 2, 1).contiguous()
        x = _call_classifier(self.classifier, x_c4, x_c3, x_c2, x_c1, lang=lang_tokens, lang_mask=l_mask3)
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)


class LAVT(_LAVTSimpleDecode):
    """Alias kept for compatibility."""


def _infer_backbone_lang_dim(backbone: nn.Module) -> Optional[int]:
    for module in backbone.modules():
        if isinstance(module, nn.Conv1d):
            in_ch = int(module.in_channels)
            if in_ch >= 256:
                return in_ch
    return None


# ----------------------------- Full model (image + text) -----------------------------

def _assert_local_bert_dir(path: str) -> None:
    """
    Provide a clearer error than the default HF loader when running offline.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("args.ck_bert must be a non-empty string path to a local BERT directory.")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"args.ck_bert directory not found: {path}\n"
            "Tip: point ck_bert to your local bert-base-uncased snapshot folder."
        )

    # Minimal sanity (do not over-constrain; some snapshots use safetensors).
    expected_any = [
        os.path.join(path, "config.json"),
        os.path.join(path, "pytorch_model.bin"),
        os.path.join(path, "model.safetensors"),
        os.path.join(path, "vocab.txt"),
    ]
    if not any(os.path.exists(p) for p in expected_any):
        raise FileNotFoundError(
            f"args.ck_bert directory exists but looks incomplete: {path}\n"
            "Expected at least one of: config.json, pytorch_model.bin, model.safetensors, vocab.txt"
        )


class _LAVTOneSimpleDecode(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        args,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

        _assert_local_bert_dir(args.ck_bert)
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

        self.train_text_encoder = lora_config is not None or getattr(args, "train_text_encoder", False)

        if lora_config is not None:
            replaced = apply_lora_adapters(self.text_encoder, lora_config)
            trainable, frozen = freeze_non_lora_params(self.text_encoder)
            self.lora_replacements = replaced
            self.lora_trainable = trainable
            if replaced == 0:
                logger.warning("LoRA config provided but no layers were wrapped.")
            else:
                logger.info("Applied LoRA to text encoder (wrapped %d layers).", replaced)
                logger.info("LoRA trainable params: %d (frozen base params: %d)", trainable, frozen)

        if not self.train_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            logger.info("Text encoder frozen; gradients will not flow into BERT.")

        lang_dim = getattr(args, "lang_dim", 768)
        self.l_proj = nn.Linear(lang_dim, lang_dim)
        self.t_proj = nn.Linear(lang_dim, lang_dim)
        self.p_proj = nn.Linear(lang_dim, lang_dim)

        # Used when masking out padding tokens in feature construction.
        self.null_token = nn.Parameter(torch.zeros(1, 1, lang_dim))

        self.text_encoder.train(self.train_text_encoder)

        # If enabled, runs BERT three times with different masks (slow, but supported).
        self.triple_text_encode = bool(getattr(args, "triple_text_encode", False))

        # Optional strictness: force only l-stream into backbone even if legacy backbone exists.
        if hasattr(args, "force_single_stream_backbone"):
            self.force_single_stream_backbone = bool(getattr(args, "force_single_stream_backbone"))
        else:
            try:
                sig = inspect.signature(self.backbone.forward)
                params = sig.parameters
                self.force_single_stream_backbone = not (
                    "t" in params or "t_mask" in params or "p" in params or "p_mask" in params
                )
            except Exception:
                self.force_single_stream_backbone = True
        self._last_debug: Optional[Dict[str, float]] = None
        self.last_hidden: Optional[Tensor] = None
        self.last_attn_mask: Optional[Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.train(mode if self.train_text_encoder else False)
        return self

    def _bert_forward(self, text: Tensor, attention_mask: Tensor) -> Tensor:
        if self.train_text_encoder:
            return self.text_encoder(text, attention_mask=attention_mask)[0]
        with torch.no_grad():
            return self.text_encoder(text, attention_mask=attention_mask)[0]

    def forward(
        self,
        x: Tensor,
        text: Tensor,
        l_mask: Tensor,
        t_mask: Tensor,
        p_mask: Tensor,
    ) -> Tensor:  # type: ignore[override]
        input_shape = x.shape[-2:]

        if text.dim() != 2:
            raise ValueError(f"text must be (B,L) token ids, got {tuple(text.shape)}")

        B, L = text.shape[0], text.shape[1]
        dev = text.device

        # Normalize and sanitize masks.
        l_mask2 = _normalize_attn_mask(l_mask, L, device=dev)
        t_mask2 = _sanitize_submask(l_mask2, t_mask, device=dev)
        p_mask2 = _sanitize_submask(l_mask2, p_mask, device=dev)
        self.last_attn_mask = l_mask2
        with torch.no_grad():
            self._last_debug = {
                "l_mask_ratio": float(l_mask2.float().mean().item()),
                "t_mask_ratio": float(t_mask2.float().mean().item()),
                "p_mask_ratio": float(p_mask2.float().mean().item()),
            }

        # Encode language (shared hidden unless triple_text_encode=True).
        if not self.triple_text_encode:
            hidden = self._bert_forward(text, attention_mask=l_mask2)  # (B,L,D)

            l_feats, l_mask3 = _masked_token_view(hidden, l_mask2, self.l_proj, self.null_token)
            t_feats, t_mask3 = _masked_token_view(hidden, t_mask2, self.t_proj, self.null_token)
            p_feats, p_mask3 = _masked_token_view(hidden, p_mask2, self.p_proj, self.null_token)
        else:
            hidden_l = self._bert_forward(text, attention_mask=l_mask2)
            hidden_t = self._bert_forward(text, attention_mask=t_mask2)
            hidden_p = self._bert_forward(text, attention_mask=p_mask2)

            l_feats, l_mask3 = _masked_token_view(hidden_l, l_mask2, self.l_proj, self.null_token)
            t_feats, t_mask3 = _masked_token_view(hidden_t, t_mask2, self.t_proj, self.null_token)
            p_feats, p_mask3 = _masked_token_view(hidden_p, p_mask2, self.p_proj, self.null_token)
            hidden = hidden_l

        # --- backbone forward ---
        # New backbone consumes only l-stream; legacy backbone can consume l/t/p.
        if self.force_single_stream_backbone:
            x_c1, x_c2, x_c3, x_c4 = _call_backbone(self.backbone, x, l_feats, l_mask3)
        else:
            x_c1, x_c2, x_c3, x_c4 = _call_backbone(self.backbone, x, l_feats, l_mask3, t_feats, t_mask3, p_feats, p_mask3)

        # Decoder language forwarding:
        # - lang tokens: (B,L,D)
        # - lang mask:   (B,L,1) float
        lang_tokens = hidden
        lang_mask3 = _mask_to_3d(l_mask2)
        self.last_hidden = hidden

        x_logits = _call_classifier(self.classifier, x_c4, x_c3, x_c2, x_c1, lang=lang_tokens, lang_mask=lang_mask3)
        return F.interpolate(x_logits, size=input_shape, mode="bilinear", align_corners=True)

    def debug_state(self) -> Dict[str, float]:
        state: Dict[str, float] = {}
        if self._last_debug:
            state.update(self._last_debug)
        state.update(
            {
                "train_text_encoder": float(bool(self.train_text_encoder)),
                "lora_wrapped_layers": float(getattr(self, "lora_replacements", 0) or 0),
            }
        )
        return state


class LAVTOne(_LAVTOneSimpleDecode):
    """Alias kept for compatibility."""


__all__ = [
    "LAVT",
    "LAVTOne",
    "LoRAConfig",
    "LoRALinear",
    "apply_lora_adapters",
    "freeze_non_lora_params",
    "load_weights",
]
