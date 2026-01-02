# Copyright (c) Open-MMLab. All rights reserved.
# MultiModal Swin backbone (no mmseg requirement, robust imports)

from collections import OrderedDict
import sys
import os
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

# -------------------------------------------------------------------------
# Minimal, mmcv-free load_checkpoint that works in any env
# -------------------------------------------------------------------------
def _simple_load_checkpoint(model, filename, strict=False, logger=None):
    """Lightweight replacement for mmcv.load_checkpoint.

    - Only supports local .pth files.
    - Tries to unwrap common dict formats: {'state_dict': ...}, {'model': ...}, etc.
    - Strips common prefixes ('module.', 'backbone.', 'encoder.') from keys.
    - Loads non-strict by default and logs missing/unexpected keys.
    """
    if logger is None:
        def _log(msg): print(msg)
    else:
        def _log(msg): logger.info(msg)

    if not filename or not os.path.isfile(filename):
        _log(f"[simple_load_checkpoint] WARNING: checkpoint not found: {filename!r}")
        return

    _log(f"[simple_load_checkpoint] Loading checkpoint from: {filename}")
    ckpt = torch.load(filename, map_location="cpu")

    # Unwrap common containers
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "backbone"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                _log(f"[simple_load_checkpoint] Using nested dict under key='{key}'")
                break
        if state_dict is None:
            # Fall back: assume the dict itself is a state_dict
            state_dict = ckpt
    else:
        # Rare case: raw state_dict
        state_dict = ckpt

    # Clean keys: strip common prefixes
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
        new_state[nk] = v

    if strict:
        # Strict mode: let PyTorch raise if things don't match
        model.load_state_dict(new_state, strict=True)
        _log("[simple_load_checkpoint] Loaded with strict=True")
        return

    # Non-strict: only keep matching tensors to avoid noisy missing/unexpected logs.
    filtered = OrderedDict()
    model_state = model.state_dict()
    skipped = 0
    mismatched = 0
    for k, v in new_state.items():
        if k not in model_state:
            skipped += 1
            continue
        if model_state[k].shape != v.shape:
            mismatched += 1
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    _log(
        "[simple_load_checkpoint] Loaded with strict=False "
        f"(loaded={len(filtered)}, skipped={skipped}, mismatched={mismatched})"
    )
    if missing:
        _log(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        _log(f"  Unexpected keys (first 10): {unexpected[:10]}")

# -------------------------------------------------------------------------
# load_checkpoint: prefer project-local mmcv_custom if it works, else fallback
# -------------------------------------------------------------------------
try:
    # when imported as `from lib.backbone import ...`
    from lib.mmcv_custom.checkpoint import load_checkpoint as _load_ckpt
except Exception:
    try:
        # when imported within the package as `from .mmcv_custom...`
        from .mmcv_custom.checkpoint import load_checkpoint as _load_ckpt
    except Exception:
        # final fallback: mmcv_custom/mmcv not available or broken -> use simple loader
        _load_ckpt = _simple_load_checkpoint

load_checkpoint = _load_ckpt

# --- timm imports with backward compatibility ---
try:
    # Newer timm (>= ~0.8): preferred, non-deprecated path
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    # Older timm (the one you have on Python 3.7/3.9 sometimes)
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from timm.models.vision_transformer import Mlp, Block
# -----------------------------------------------

# -----------------------------------------------


# ---- tiny local logger (avoid importing mmseg/mmengine entirely) ----
def get_root_logger():
    logger = logging.getLogger("RRSIS.backbone")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ----------------------------- MLP -----------------------------
class Mlp(nn.Module):
    """Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# --------------------- Swin window helpers ---------------------
def window_partition(x, window_size):
    """x: (B, H, W, C) -> (num_windows*B, window, window, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window, window, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ----------------------- Window Attention -----------------------
class WindowAttention(nn.Module):
    """Window-based multi-head self attention (W-MSA) with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C); mask: (num_windows, N, N) or None"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        ).permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------- Swin Transformer Block ----------------------
class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, w, w, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, N, C

        # attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # unpad
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --------------------------- Patch Merging ---------------------------
class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        # pad if odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------- Patch Embed ----------------------------
class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)  # B, C, Wh, Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


# ---------------------- MultiModal Swin Transformer ----------------------
class MultiModalSwinTransformer(nn.Module):
    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        num_heads_fusion=[1, 1, 1, 1],
        fusion_drop=0.0,
        language_dim=768,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.language_dim = language_dim
        # Learnable gate for target/position fusion into language stream.
        self.tp_gate = nn.Parameter(torch.tensor(0.0))

        # patch embed
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MMBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_heads_fusion=num_heads_fusion[i_layer],
                language_dim=self.language_dim,
                fusion_drop=fusion_drop,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # per-stage norm
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 1 and getattr(self, "absolute_pos_embed", None) is not None:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str | None): path/uri to pre-trained weights.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            # strict only when the checkpoint is from an "upernet" composite
            load_checkpoint(self, pretrained, strict=("upernet" in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, l, l_mask, t=None, t_mask=None, p=None, p_mask=None):
        """Forward.
        Args:
            x: image tensor (B, 3, H, W)
            l: language feats (B, C_l, N_l)
            l_mask: (B, N_l, 1)
            t/p: optional target/position feats (B, C_l, N_l)
            t_mask/p_mask: optional masks (B, N_l, 1)
        """
        if t is not None or p is not None:
            t_mask = l_mask if t_mask is None else t_mask
            p_mask = l_mask if p_mask is None else p_mask
            t = t if t is not None else 0.0
            p = p if p is not None else 0.0
            t = t * t_mask.transpose(1, 2)
            p = p * p_mask.transpose(1, 2)
            gate = torch.sigmoid(self.tp_gate)
            l = l + gate * 0.5 * (t + p)
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if getattr(self, "absolute_pos_embed", None) is not None:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, l, l_mask, t, t_mask, p, p_mask)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)  # (B, H*W, C)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


# ----------------------- Multimodal Basic Layer -----------------------
class MMBasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        num_heads_fusion=1,
        fusion_drop=0.0,
        language_dim=768,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # fusion before downsampling
        self.fusion = PWAM(
            dim,          # dim for both visual seq and combined
            dim,          # v_in
            language_dim, # l_in (language hidden size)
            dim,          # key
            dim,          # value
            num_heads=num_heads_fusion,
            dropout=fusion_drop,
        )
        self.opab = OPAB(
            dim,
            dim,
            language_dim,
            dim,
            dim,
            num_heads=num_heads_fusion,
            dropout=fusion_drop,
        )
        self.tp_gate = nn.Parameter(torch.tensor(0.0))

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh(),
        )

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

        # initialize gate to 0 for a gentle start
        nn.init.zeros_(self.res_gate[0].weight)
        nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, x, H, W, l, l_mask, t=None, t_mask=None, p=None, p_mask=None):
        """x: (B, H*W, C)"""
        # compute attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, w, w, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        # PWAM fusion (x is (B, H*W, C); l is (B, C_l, N_l); l_mask is (B, N_l, 1))
        x_residual = self.fusion(x, l, l_mask)
        if t is not None and p is not None:
            t_mask = l_mask if t_mask is None else t_mask
            p_mask = l_mask if p_mask is None else p_mask
            opab_residual = self.opab(x, H, W, t, t_mask, p, p_mask)
            gate = torch.sigmoid(self.tp_gate)
            x_residual = x_residual + gate * opab_residual
        x = x + (self.res_gate(x_residual) * x_residual)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww
        else:
            return x_residual, H, W, x, H, W


# ----------------------------- PWAM -----------------------------
class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super().__init__()
        # input x: (B, H*W, dim)
        self.vis_project = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.image_lang_att = SpatialImageLanguageAttention(
            v_in_channels,
            l_in_channels,
            key_channels,
            value_channels,
            out_channels=value_channels,
            num_heads=num_heads,
        )

        self.project_mm = nn.Sequential(
            nn.Conv1d(value_channels, value_channels, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, l, l_mask):
        # x: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        lang = self.image_lang_att(x, l, l_mask)    # (B, H*W, dim)
        lang = lang.permute(0, 2, 1)                # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)                    # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)                    # (B, H*W, dim)
        return mm


class OPAB(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(OPAB, self).__init__()

        self.dim = dim

        # Ground Object Branch
        self.go_cross_attn = SpatialImageLanguageAttention(
            v_in_channels,
            l_in_channels,
            key_channels,
            value_channels,
            out_channels=value_channels,
            num_heads=num_heads,
        )
        self.go_tanh_gating = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh(),
        )

        self.go_embedding = nn.Linear(dim, dim, bias=False)

        # Spatial Position Branch
        self.sp_cross_attn = SpatialImageLanguageAttention(
            v_in_channels,
            l_in_channels,
            key_channels,
            value_channels,
            out_channels=value_channels,
            num_heads=num_heads,
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, t, t_mask, p, p_mask):
        B = x.shape[0]

        # Ground object branch
        go = self.go_cross_attn(x, t, t_mask)
        go = self.go_tanh_gating(go) * go

        # Spatial position branch
        sp = self.sp_cross_attn(x, p, p_mask)
        sp = sp.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
        avg_sp = torch.mean(sp, dim=1, keepdim=True)
        max_sp, _ = torch.max(sp, dim=1, keepdim=True)
        sp = torch.cat([avg_sp, max_sp], dim=1)
        sp = self.sigmoid(self.conv1(sp))
        sp = sp.permute(0, 2, 3, 1)
        sp = sp.reshape(B, H * W, 1)
        out = go * sp

        return out


# ------------------ Spatial Image-Language Attention ------------------
class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super().__init__()
        # x: (B, H*W, v_in_channels)
        # l: (B, l_in_channels, N_l)
        # l_mask: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        self.out_channels = out_channels or value_channels

        # language -> keys
        self.f_key = nn.Sequential(nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1))
        # visual -> queries
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )
        # language -> values
        self.f_value = nn.Sequential(nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1))
        # output projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x: (B, H*W, v_in_channels)
        # l: (B, l_in_channels, N_l)
        # l_mask: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)          # (B, v_in_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, 1, N_l)

        query = self.f_query(x)         # (B, key_channels, H*W)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)

        key = self.f_key(l)             # (B, key_channels, N_l)
        value = self.f_value(l)         # (B, value_channels, N_l)

        key = key * l_mask              # mask padding tokens
        value = value * l_mask

        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, N_l)

        sim_map = torch.matmul(query, key)                 # (B, heads, H*W, N_l)
        sim_map = (self.key_channels ** -0.5) * sim_map    # scaled dot product
        sim_map = sim_map + (1e4 * l_mask - 1e4)           # large negative on padding
        sim_map = F.softmax(sim_map, dim=-1)

        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, heads, H*W, value_channels//heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)
        out = out.permute(0, 2, 1)                              # (B, value_channels, HW)
        out = self.W(out)                                       # (B, out_channels, HW)
        out = out.permute(0, 2, 1)                              # (B, HW, out_channels)
        return out
