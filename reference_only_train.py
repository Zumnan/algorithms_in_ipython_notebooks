#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys
import time
import gc
from functools import reduce
import operator
from collections import OrderedDict
from typing import Optional  # <-- for Python 3.9 union types

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing

import cv2
import numpy as np
import torchvision
from transformers import BertModel

# Make repo root importable first
RRSIS_DIR = os.path.abspath(os.path.dirname(__file__))          # .../RRSIS
ROOT = os.path.abspath(os.path.join(RRSIS_DIR, '..'))           # repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project imports
from lib import segmentation
import transforms as T
import utils

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cudnn.benchmark = True

# ======================= DATASET SWITCH (optional local override) =======================
# Choices: 'auto', 'vdd_ris', 'rrsisd', 'lavt_risrs', 'risbench'
SELECTED_DATASET = 'lavt_risrs'

# Default roots for each dataset (edit these to your local paths as needed)
DATA_ROOTS = {
    'vdd_ris': "/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/",
    'rrsisd':  os.path.join(RRSIS_DIR, "refer", "data"),
    'lavt_risrs': os.path.join(RRSIS_DIR, "refer", "LAVT-RISRS"),
    'risbench': "/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/RISBench_dataset/",
}

def _maybe_override_dataset_from_file(args, selected_name: str):
    """Optionally set args.dataset_name and args.data_root from a simple in-file selector."""
    if not selected_name:
        return
    selected = (selected_name or "").lower().strip()
    if selected not in ("auto", "vdd_ris", "rrsisd", "lavt_risrs", "risbench"):
        print(f"[dataset switch] Unknown selection '{selected_name}', falling back to 'auto'.")
        selected = "auto"

    if not getattr(args, "dataset_name", None) or args.dataset_name == "auto":
        args.dataset_name = selected

    default_root = DATA_ROOTS.get(selected, "")
    if selected != "auto":
        if (not getattr(args, "data_root", None)) or (args.data_root and not os.path.isdir(args.data_root)):
            args.data_root = default_root

    if args.data_root:
        os.environ["RRSIS_DATA_ROOT"] = args.data_root

    print(f"[dataset switch] Using dataset='{args.dataset_name}'")
    print(f"[dataset switch] data_root='{args.data_root}'")

# ========================================================================
# ---------- BERT local snapshot helpers ----------
_DEFAULT_LOCAL_BERT = "/10T/students/doctor/2025/zum/models/bert-base-uncased/"
_BERT_REQUIRED_FILES = (
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "pytorch_model.bin",
)

def _use_local_bert_if_available(args):
    """
    If a local snapshot exists, force using it for BOTH the model and tokenizer,
    and enable offline mode to avoid any network calls.
    Priority:
      1) args.ck_bert if it is a directory
      2) env BERT_LOCAL_DIR if it exists
      3) _DEFAULT_LOCAL_BERT
    """
    candidates = []
    if getattr(args, "ck_bert", None):
        candidates.append(args.ck_bert)
    env_dir = os.getenv("BERT_LOCAL_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append(_DEFAULT_LOCAL_BERT)

    local_dir = None
    for c in candidates:
        if c and os.path.isdir(c):
            local_dir = os.path.abspath(c)
            break

    if local_dir is not None and (not getattr(args, "ck_bert", None) or args.ck_bert == "bert-base-uncased"):
        args.ck_bert = local_dir

    if os.path.isdir(getattr(args, "ck_bert", "")):
        if not getattr(args, "bert_tokenizer", None) or args.bert_tokenizer == "bert-base-uncased":
            args.bert_tokenizer = args.ck_bert

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        missing = [f for f in _BERT_REQUIRED_FILES if not os.path.exists(os.path.join(args.ck_bert, f))]
        if missing:
            print(f"[WARN] Local BERT folder found at {args.ck_bert} but missing files: {missing}")
        else:
            print(f"[BERT] Using local snapshot (offline): {args.ck_bert}")
            print(f"[BERT] Tokenizer path: {args.bert_tokenizer}")

# ---------- Swin pretrained checkpoint resolver ----------
_DEFAULT_SWIN_LOCAL = os.path.join(RRSIS_DIR, "pretrained_weights", "swin_base_patch4_window7_224.pth")

def _resolve_pretrained_swin(args):
    """Ensure args.pretrained_swin_weights points to an existing checkpoint."""
    ck = getattr(args, "pretrained_swin_weights", None)
    if ck and os.path.isfile(ck):
        resolved = os.path.abspath(ck)
        args.pretrained_swin_weights = resolved
        print(f"[Swin] Using user-provided pretrained weights: {resolved}")
        return

    env_ck = os.getenv("SWIN_LOCAL_WEIGHTS")
    if env_ck and os.path.isfile(env_ck):
        args.pretrained_swin_weights = os.path.abspath(env_ck)
        print(f"[Swin] Using env SWIN_LOCAL_WEIGHTS: {args.pretrained_swin_weights}")
        return

    if os.path.isfile(_DEFAULT_SWIN_LOCAL):
        args.pretrained_swin_weights = _DEFAULT_SWIN_LOCAL
        print(f"[Swin] Using local pretrained weights: {_DEFAULT_SWIN_LOCAL}")
    else:
        args.pretrained_swin_weights = ""
        print("[Swin] WARNING: No pretrained Swin weights found at "
              f"{_DEFAULT_SWIN_LOCAL}. Backbone will initialize randomly.")


def _ensure_transformers_offline():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _get_language_dim(args):
    return TEXT_DIM_CLIP if getattr(args, "language_encoder", "bert") == "clip" else TEXT_DIM_BERT

def _ensure_clip_support_or_fallback(args):
    """
    On very old transformers (e.g. 3.x with tokenizers==0.8.1rc1) CLIP does not
    exist at all, so trying to load it via AutoModel/AutoConfig gives
    KeyError('clip'). In that case we transparently fall back to BERT and keep
    the rest of the pipeline consistent.
    """
    if getattr(args, "language_encoder", "bert") != "clip":
        return

    try:
        # If this import works, the install actually has CLIP support.
        from transformers import CLIPTextModel  # type: ignore
        _ = CLIPTextModel
        return
    except Exception:
        import transformers as _tf
        print(f"[CLIP] WARNING: transformers {_tf.__version__} in this env does not expose "
              "CLIPTextModel/CLIPModel.")
        print("[CLIP]          Falling back to --language_encoder bert.")
        print("[CLIP]          To actually use CLIP you need a newer transformers "
              "version (>=4.x) and tokenizers>=0.10.x, ideally in a separate env.")
        args.language_encoder = "bert"


def _resolve_clip_model_class():
    """
    Return a *text-only* CLIP class.

    Preferred:
      1) CLIPTextModel  (true text tower, needs only input_ids/attention_mask)
      2) A wrapper around CLIPModel that exposes a text-only forward
      3) AutoModel as a last resort

    The returned class must support .from_pretrained(...) and its forward(...)
    must behave like CLIPTextModel (i.e. no pixel_values required and has
    .last_hidden_state in the output).
    """
    # 1) Best case: CLIPTextModel exists
    try:
        from transformers import CLIPTextModel as _ClipText
        return _ClipText
    except Exception:
        pass

    # 2) Fallback: CLIPModel exists but CLIPTextModel does not
    try:
        from transformers import CLIPModel as _Clip
        import torch.nn as nn

        class _ClipTextWrapper(nn.Module):
            """
            Wrap CLIPModel but expose only the text tower in forward().
            This makes it API-compatible with CLIPTextModel for our use:
            text_model(input_ids=..., attention_mask=...).last_hidden_state
            """
            def __init__(self, base):
                super().__init__()
                self.base = base
                # Keep a handle to the underlying text tower
                self.text_model = base.text_model

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                base = _Clip.from_pretrained(*args, **kwargs)
                return cls(base)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                # Call the underlying text tower directly, so no pixel_values needed
                return self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )

        return _ClipTextWrapper
    except Exception:
        pass

    # 3) Last resort: AutoModel
    try:
        from transformers import AutoModel as _Auto
        return _Auto
    except Exception as e:
        raise RuntimeError("transformers installation lacks CLIP text support.") from e



def _load_clip_text_encoder(args, distributed: bool):
    _ensure_transformers_offline()
    model_path = getattr(args, "clip_model_path", None)
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"[CLIP] Missing local model path: {model_path}")
    print(f"[CLIP] Loading offline CLIP text encoder from {model_path}")
    clip_cls = _resolve_clip_model_class()
    clip_model = clip_cls.from_pretrained(model_path, local_files_only=True)
    clip_model.cuda()
    if distributed:
        clip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)
        clip_model = torch.nn.parallel.DistributedDataParallel(
            clip_model, device_ids=[torch.cuda.current_device()]
        )
    return clip_model



def _load_clip_vision_encoder(args):
    """
    Optional ONNX-based CLIP vision encoder.

    - If visual_encoder != 'clip', this is never used.
    - If onnxruntime is not installed or the file is missing, we log once and
      cleanly return None. The rest of the pipeline continues with Swin only.
    """
    vision_path = getattr(args, "clip_vision_onnx", None)
    if getattr(args, "visual_encoder", "swin") != "clip":
        # Not requested -> no-op
        return None

    if not vision_path or not os.path.isfile(vision_path):
        print(f"[CLIP] Vision encoder path missing or not found: {vision_path}")
        print("[CLIP]       Vision tower will be disabled (Swin backbone only).")
        return None

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("[CLIP] onnxruntime is not installed; ONNX vision encoder will be disabled.")
        print("[CLIP] Install 'onnxruntime-gpu' (or 'onnxruntime') if you actually "
              "want to use the CLIP vision tower from ONNX.")
        return None
    except Exception as e:
        print(f"[CLIP] Failed to import onnxruntime ({e}); vision encoder disabled.")
        return None

    try:
        print(f"[CLIP] Loading offline vision encoder (ONNX): {vision_path}")
        session = ort.InferenceSession(
            vision_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        return session
    except Exception as e:
        print(f"[CLIP] Failed to create ONNX inference session: {e}")
        print("[CLIP] Vision tower will be disabled.")
        return None


# ---------- Dataset / transforms / metrics ----------
def get_dataset(image_set, transform, args):
    from data.newdataset_refer_bert import ReferDataset
    ds = ReferDataset(
        args,
        split=image_set,
        image_transforms=transform,
        target_transforms=None,
    )
    num_classes = 2
    return ds, num_classes

def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union

def get_transform(args, is_train=True):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)

# --- Baseline CE (kept for reference; not used after composite loss is introduced) ---
def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)

# ====================== Composite Loss Components (BERT-friendly) ======================
# Default/static weights
ALPHA_DICE   = 0.3     # Lseg = CE + ALPHA_DICE * Dice
ALPHA_ORTHO  = 0.05    # weight for Lortho
ALPHA_ALIGN  = 0.05    # weight for Lalign-lite

# Warmup controls (ramp dice/align over first K steps)
WARMUP_STEPS = 2000

# Optional focal-ize CE (kept off by default to avoid behavior changes)
USE_FOCAL_CE = False
FOCAL_GAMMA  = 2.0
FOCAL_ALPHA_FG = 0.75  # balance foreground a bit when using focal

TEXT_DIM_BERT = 768     # bert-base hidden size
TEXT_DIM_CLIP = 512     # openai/clip-vit-base-patch32 text hidden size
TEXT_HEAD_DIM = 256     # size of small text heads

def focal_ce_with_logits(logits, target, gamma=2.0, alpha_fg=0.75):
    """
    Binary focal CE implemented on 2-class logits.
    logits: [B,2,H,W] ; target: [B,H,W] in {0,1}
    """
    if logits.shape[1] != 2:
        raise ValueError("focal_ce_with_logits assumes 2-class logits.")
    # make prob for the target class
    log_prob = F.log_softmax(logits, dim=1)
    prob = torch.exp(log_prob)
    # gather target probs
    tgt = target.long()
    log_pt = log_prob.gather(1, tgt.unsqueeze(1)).squeeze(1)  # [B,H,W]
    pt = prob.gather(1, tgt.unsqueeze(1)).squeeze(1)          # [B,H,W]

    # alpha weighting (foreground emphasis)
    alpha = torch.ones_like(pt) * (1.0 - alpha_fg)
    alpha = torch.where(tgt == 1, torch.ones_like(pt) * alpha_fg, alpha)

    loss = -alpha * ((1 - pt) ** gamma) * log_pt
    return loss.mean()

class TextDecomposer(torch.nn.Module):
    """
    Two learnable linear heads that map a sentence embedding to two
    'semantic subspaces' t1 and t2, used in Lortho.
    """
    def __init__(self, dim_in: int, dim_out: int = 256):
        super().__init__()
        self.proj1 = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_in),
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_out, dim_out),
        )
        self.proj2 = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_in),
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_out, dim_out),
        )

    def forward(self, sent_vec: torch.Tensor):
        # sent_vec: [B, D]
        t1 = self.proj1(sent_vec)
        t2 = self.proj2(sent_vec)
        return t1, t2

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    Binary soft dice computed on the foreground channel of logits.
    logits: [B, C=2, H, W]
    target: [B, H, W] with {0,1}
    """
    probs = torch.softmax(logits, dim=1)[:, 1]        # foreground prob [B,H,W]
    target_f = (target > 0).float()
    if probs.shape[-2:] != target_f.shape[-2:]:
        probs = torch.nn.functional.interpolate(
            probs.unsqueeze(1), size=target_f.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
    intersection = (probs * target_f).sum(dim=(-1, -2))
    denom = probs.sum(dim=(-1, -2)) + target_f.sum(dim=(-1, -2))
    dice = (2 * intersection + eps) / (denom + eps)
    return (1.0 - dice).mean()

def squared_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    # a,b: [B,D]
    a = torch.nn.functional.normalize(a, dim=-1, eps=eps)
    b = torch.nn.functional.normalize(b, dim=-1, eps=eps)
    cos = (a * b).sum(dim=-1)           # [-1..1]
    return (cos ** 2).mean()

def sentence_mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor):
    """
    Mean-pool BERT token embeddings with attention mask.
    last_hidden: [B, T, D]
    attn_mask:   [B, T]  (1 = keep)
    return: [B, D]
    """
    mask = attn_mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled

class AlignLite(torch.nn.Module):
    """
    Projects (i) a global mask descriptor and (ii) the sentence embedding into the same space
    and aligns them with cosine loss. It’s 'weak' but helps without touching internals.
    """
    def __init__(self, txt_dim: int, hid: int = 256):
        super().__init__()
        self.txt_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(txt_dim), torch.nn.Linear(txt_dim, hid), torch.nn.ReLU(True),
            torch.nn.Linear(hid, hid)
        )
        # mask descriptor is 2D stats -> small MLP
        self.mask_proj = torch.nn.Sequential(
            torch.nn.Linear(4, hid), torch.nn.ReLU(True),
            torch.nn.Linear(hid, hid)
        )

    @staticmethod
    def mask_descriptor(mask_fg: torch.Tensor):
        """
        Tiny descriptor of predicted mask geometry:
        - foreground ratio
        - centroid (y,x) normalized
        - boundary ratio (fraction of foreground touching border)
        mask_fg: [B, H, W] in [0,1]
        """
        B, H, W = mask_fg.shape
        area = mask_fg.sum(dim=(1,2)).clamp_min(1e-6)
        ratio = area / float(H*W)

        ys = torch.linspace(0, 1, steps=H, device=mask_fg.device).view(1, H, 1)
        xs = torch.linspace(0, 1, steps=W, device=mask_fg.device).view(1, 1, W)
        cy = (mask_fg * ys).sum(dim=(1,2)) / area
        cx = (mask_fg * xs).sum(dim=(1,2)) / area

        top = mask_fg[:, 0, :].sum(dim=1)
        bot = mask_fg[:, -1, :].sum(dim=1)
        lef = mask_fg[:, :, 0].sum(dim=1)
        rig = mask_fg[:, :, -1].sum(dim=1)
        boundary = (top + bot + lef + rig) / (area + 1e-6)
        return torch.stack([ratio, cy, cx, boundary], dim=1)  # [B,4]

    def forward(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor, logits: torch.Tensor):
        # sentence -> embedding
        s = sentence_mean_pool(last_hidden, attn_mask)      # [B,D]
        s = self.txt_proj(s)                                # [B,H]

        # logits -> foreground prob -> descriptor
        pf = torch.softmax(logits, dim=1)[:, 1]             # [B,H,W]
        desc = self.mask_descriptor(pf)                     # [B,4]
        v = self.mask_proj(desc)                            # [B,H]

        # cosine alignment loss (maximize cosine => minimize 1-cos)
        s = torch.nn.functional.normalize(s, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        cos = (s * v).sum(dim=-1)
        return (1.0 - cos).mean()

def composite_loss(
    output_logits,
    target,
    last_hidden_states=None,
    attn_mask=None,
    text_decomposer=None,
    align_lite=None,
    alpha_dice: float = ALPHA_DICE,
    alpha_ortho: float = ALPHA_ORTHO,
    alpha_align: float = ALPHA_ALIGN,
):
    """
    L = Lseg + alpha_ortho * Lortho + alpha_align * Lalign-lite
    where Lseg = CE(or focal) + alpha_dice * Dice
    """
    # Lseg
    if USE_FOCAL_CE:
        l_ce = focal_ce_with_logits(output_logits, target, gamma=FOCAL_GAMMA, alpha_fg=FOCAL_ALPHA_FG)
    else:
        ce_w = torch.FloatTensor([0.9, 1.1]).to(output_logits.device)
        l_ce = nn.functional.cross_entropy(output_logits, target, weight=ce_w)
    l_dice = dice_loss_from_logits(output_logits, target)
    l_seg = l_ce + alpha_dice * l_dice

    l_ortho = torch.tensor(0., device=output_logits.device)
    l_align = torch.tensor(0., device=output_logits.device)

    if (last_hidden_states is not None) and (attn_mask is not None) and (text_decomposer is not None):
        s = sentence_mean_pool(last_hidden_states, attn_mask)  # [B,D]
        t1, t2 = text_decomposer(s)                             # [B,H], [B,H]
        l_ortho = squared_cosine_similarity(t1, t2)

    if (last_hidden_states is not None) and (attn_mask is not None) and (align_lite is not None):
        l_align = align_lite(last_hidden_states, attn_mask, output_logits)

    total = l_seg + alpha_ortho * l_ortho + alpha_align * l_align
    return total, {
        "l_ce": l_ce.detach(),
        "l_dice": l_dice.detach(),
        "l_ortho": l_ortho.detach(),
        "l_align": l_align.detach(),
        "l_seg": l_seg.detach(),
    }

# ====================== Eval ======================
@torch.no_grad()
def evaluate(model, data_loader, text_model, encoder_type: str = "bert", split_name="test", log_prefix="Eval", use_amp: bool = True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{log_prefix} [{split_name}]'
    total_its = 0
    acc_ious = 0

    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    imid = 0

    autocast = torch.cuda.amp.autocast

    for data in metric_logger.log_every(data_loader, 100, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = (
            image.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            sentences.cuda(non_blocking=True),
            attentions.cuda(non_blocking=True),
        )

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        with autocast(enabled=use_amp):
            if encoder_type == "clip" and text_model is not None:
                last_hidden_states = text_model(input_ids=sentences, attention_mask=attentions).last_hidden_state
                embedding = last_hidden_states.permute(0, 2, 1)
                attentions_img = attentions.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=attentions_img)
            elif text_model is not None:
                last_hidden_states = text_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                attentions_img = attentions.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=attentions_img)
            else:
                output = model(image, sentences, l_mask=attentions)

        iou, I, U = IoU(output, target)
        imid += 1
        acc_ious += iou
        mean_IoU.append(iou)
        cum_I += I
        cum_U += U
        for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
            seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
        seg_total += 1

    iou = acc_ious / max(total_its, 1)
    mean_IoU = np.array(mean_IoU)
    mIoU = float(np.mean(mean_IoU)) if len(mean_IoU) > 0 else 0.0
    overallIoU = float((cum_I / cum_U).item() * 100.0) if cum_U != 0 else 0.0

    print(f'[{split_name}] Final results:')
    print('  Mean IoU is %.2f' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
        results_str += '    precision@%s = %.2f\n' % (
            str(eval_seg_iou),
            seg_correct[n_eval_iou] * 100.0 / max(seg_total, 1),
        )
    results_str += '    overall IoU = %.2f\n' % (overallIoU)
    print(results_str)
    return 100.0 * float(iou), overallIoU  # (avg obj IoU %, overall IoU %)

# ====================== Training ======================
def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, text_model, encoder_type: str,
                    text_decomposer=None, align_lite=None,
                    use_amp: bool = True, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    """
    Returns updated `iterations` counter (global steps).
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    autocast = torch.cuda.amp.autocast
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = (
            image.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            sentences.cuda(non_blocking=True),
            attentions.cuda(non_blocking=True),
        )

        sentences = sentences.squeeze(1)
        attn_1d = attentions.squeeze(1)  # [B,T] for BERT & pooling

        # ---- warmup ramp for dice & align ----
        ramp = min(1.0, float(iterations) / float(max(WARMUP_STEPS, 1)))
        eff_alpha_dice  = ALPHA_DICE  * ramp
        eff_alpha_align = ALPHA_ALIGN * ramp
        eff_alpha_ortho = ALPHA_ORTHO

        with autocast(enabled=use_amp):
            if encoder_type == "clip" and text_model is not None:
                last_hidden_states = text_model(input_ids=sentences, attention_mask=attn_1d).last_hidden_state  # [B,T,D]
                embedding = last_hidden_states.permute(0, 2, 1)
                l_mask_img = attn_1d.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=l_mask_img)
                loss, parts = composite_loss(
                    output, target,
                    last_hidden_states=last_hidden_states,
                    attn_mask=attn_1d,           # use token mask for mean pooling
                    text_decomposer=text_decomposer,
                    align_lite=align_lite,
                    alpha_dice=eff_alpha_dice,
                    alpha_ortho=eff_alpha_ortho,
                    alpha_align=eff_alpha_align,
                )
            elif text_model is not None:
                last_hidden_states = text_model(sentences, attention_mask=attn_1d)[0]   # [B,T,D]
                embedding = last_hidden_states.permute(0, 2, 1)
                l_mask_img = attn_1d.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=l_mask_img)

                loss, parts = composite_loss(
                    output, target,
                    last_hidden_states=last_hidden_states,
                    attn_mask=attn_1d,           # use token mask for mean pooling
                    text_decomposer=text_decomposer,
                    align_lite=align_lite,
                    alpha_dice=eff_alpha_dice,
                    alpha_ortho=eff_alpha_ortho,
                    alpha_align=eff_alpha_align,
                )
            else:
                output = model(image, sentences, l_mask=attn_1d)
                loss, parts = composite_loss(
                    output, target,
                    alpha_dice=eff_alpha_dice,
                    alpha_ortho=eff_alpha_ortho,
                    alpha_align=eff_alpha_align,
                )

        optimizer.zero_grad(set_to_none=True)

        # AMP backward/step
        scaler.scale(loss).backward()
        # Unscale before clipping so clipping happens on true grads
        scaler.unscale_(optimizer)
        # ---- grad clip for stability ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"],
                             l_ce=parts["l_ce"].item(),
                             l_dice=parts["l_dice"].item(),
                             l_ortho=parts["l_ortho"].item(),
                             l_align=parts["l_align"].item())

        del image, target, sentences, attentions, loss, output, data
        if text_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return iterations

# ---------- BERT / DDP helpers ----------
def _load_bert(args, distributed: bool):
    """Load BERT from local snapshot or Hub; never wrap in DataParallel; wrap in DDP only if distributed."""
    model_class = BertModel
    ck = args.ck_bert
    local_only = os.path.isdir(ck)
    if local_only:
        print(f"[BERT] Loading from local snapshot: {ck} (offline)")
    bert = model_class.from_pretrained(ck, local_files_only=local_only)

    # DO NOT set bert.pooler = None here
    # You only use bert(...)[0] (sequence output), so we can leave pooler as-is.

    bert.cuda()
    if distributed:
        bert = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert)
        bert = torch.nn.parallel.DistributedDataParallel(
            bert, device_ids=[torch.cuda.current_device()]
        )
    return bert


def _is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _build_eval_loaders(args, transform, distributed, splits):
    """Create dataloaders for the provided eval splits."""
    loaders = OrderedDict()
    for split in splits:
        try:
            dataset_split, _ = get_dataset(split, transform, args=args)
        except Exception as e:
            print(f"[Eval] Skipping split '{split}' due to error: {e}")
            continue

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_split, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset_split)

        loaders[split] = torch.utils.data.DataLoader(
            dataset_split, batch_size=1, sampler=sampler, num_workers=args.workers
        )
    return loaders

# ====================== Main ======================
def main(args):
    # Optionally override dataset selection from the top-of-file constant
    _maybe_override_dataset_from_file(args, SELECTED_DATASET)

    # If the user requested CLIP but this transformers build doesn't have it,
    # switch back to BERT early so everything (dims, tokenizers, models) is consistent.
    _ensure_clip_support_or_fallback(args)

    if getattr(args, "language_encoder", "bert") == "clip":
        _ensure_transformers_offline()

    if getattr(args, "language_encoder", "bert") == "clip":
        _ensure_transformers_offline()

    # Confirm backbone file path
    try:
        from lib import backbone as _bb
        print("Using backbone module:", _bb.__file__)
    except Exception:
        pass

    # Force local BERT usage if available (and set offline env vars)
    _use_local_bert_if_available(args)

    # Resolve Swin pretrained weights path (your local file)
    _resolve_pretrained_swin(args)

    # Parse eval splits (comma-separated -> list, strip spaces)
    eval_splits = [s.strip() for s in getattr(args, "eval_splits", "val,test").split(",") if s.strip()]
    seen = set()
    eval_splits = [s for s in eval_splits if not (s in seen or seen.add(s))]
    primary_eval_split = "val" if "val" in eval_splits else (eval_splits[0] if eval_splits else args.split)

    # AMP toggle (default ON for A6000)
    use_amp = bool(getattr(args, "amp", True))
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print(f"[AMP] Automatic Mixed Precision enabled: {use_amp}")

    # datasets
    dataset, num_classes = get_dataset("train", get_transform(args=args, is_train=True), args=args)

    # distributed?
    distributed = _is_distributed()
    if not distributed:
        print("[DDP] Not initialized -> running SINGLE-PROCESS (no DDP).")
        print(f"[CUDA] Visible devices: {torch.cuda.device_count()} (using device 0 only)")

    # samplers
    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    # data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Build eval loaders for all requested splits (val/test by default)
    eval_transform = get_transform(args=args, is_train=False)
    eval_loaders = _build_eval_loaders(args, eval_transform, distributed, eval_splits)

    # compute language dim and attach to args for downstream components
    text_dim = _get_language_dim(args)
    args.language_dim = text_dim

    # optional: load CLIP vision encoder when requested (currently for offline readiness)
    clip_vision_session = None
    if getattr(args, "visual_encoder", "swin") == "clip":
        clip_vision_session = _load_clip_vision_encoder(args)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=(args.pretrained_swin_weights or ""), args=args)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
        )
        single_model = model.module
    else:
        single_model = model

    # Text encoder selection
    text_model = None
    single_text_model = None
    if args.model != 'lavt_one':
        if getattr(args, "language_encoder", "bert") == "clip":
            text_model = _load_clip_text_encoder(args, distributed)
            try:
                if hasattr(text_model, "vision_model"):
                    for p in text_model.vision_model.parameters():
                        p.requires_grad = False
            except Exception:
                pass
        else:
            text_model = _load_bert(args, distributed)
        single_text_model = text_model.module if hasattr(text_model, "module") else text_model
    else:
        if getattr(args, "language_encoder", "bert") != "bert":
            print("[WARN] lavt_one uses internal BERT encoder; ignoring language_encoder!=bert")

    # NEW: small text heads for losses
    text_decomposer = TextDecomposer(dim_in=text_dim, dim_out=TEXT_HEAD_DIM).cuda()
    align_lite = AlignLite(txt_dim=text_dim, hid=TEXT_HEAD_DIM).cuda()

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            if 'text_model' in checkpoint and single_text_model is not None:
                single_text_model.load_state_dict(checkpoint['text_model'])
            elif 'bert_model' in checkpoint and single_text_model is not None:
                single_text_model.load_state_dict(checkpoint['bert_model'])
        # try to resume heads if available
        if 'text_decomposer' in checkpoint:
            text_decomposer.load_state_dict(checkpoint['text_decomposer'])
        if 'align_lite' in checkpoint:
            align_lite.load_state_dict(checkpoint['align_lite'])

    # parameters to optimize
    backbone_no_decay = []
    backbone_decay = []
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        text_param_block = []
        if getattr(args, "language_encoder", "bert") == "clip":
            clip_text_module = getattr(single_text_model, "text_model", single_text_model)
            text_param_block = [{"params": [p for p in clip_text_module.parameters() if p.requires_grad]}]
        else:
            text_param_block = [{"params": reduce(operator.concat,
                              [[p for p in single_text_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])}]
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        ] + text_param_block
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # add loss-head params
    extra_params = list(text_decomposer.parameters()) + list(align_lite.parameters())
    if extra_params:
        params_to_optimize.append({"params": [p for p in extra_params if p.requires_grad]})

    # optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_primary_oIoU = -0.1
    best_epoch = -1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint.get('epoch', -1)
        best_primary_oIoU = checkpoint.get('best_primary_oIoU', best_primary_oIoU)
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        # Attempt to resume scaler if present
        if 'scaler' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
                print("[AMP] Loaded GradScaler state from checkpoint.")
            except Exception:
                print("[AMP] Could not load GradScaler state; continuing with fresh scaler.")
    else:
        resume_epoch = -999

    # Flag to allow eval-only without training
    eval_only = bool(getattr(args, "eval_only", False))

    # training/eval loops
    for epoch in range(max(0, resume_epoch + 1), args.epochs if not eval_only else 1):
        if _is_distributed() and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        if not eval_only:
            iterations = train_one_epoch(
                model, optimizer, data_loader, lr_scheduler,
                epoch, args.print_freq, iterations,
                text_model, getattr(args, "language_encoder", "bert"),
                text_decomposer=text_decomposer, align_lite=align_lite,
                use_amp=use_amp, scaler=scaler
            )

        # Evaluate on all requested splits
        split_metrics = {}
        for split_name, loader in eval_loaders.items():
            iou, overallIoU = evaluate(
                model, loader, text_model,
                encoder_type=getattr(args, "language_encoder", "bert"),
                split_name=split_name, log_prefix="Eval", use_amp=use_amp
            )
            split_metrics[split_name] = {"avg_obj_IoU": iou, "overallIoU": overallIoU}
            print(f'[{split_name}] Average object IoU {iou:.2f}')
            print(f'[{split_name}] Overall IoU {overallIoU:.2f}')

        # Choose primary split metric to track "best"
        primary_metric = split_metrics.get(primary_eval_split)
        if primary_metric is None and len(split_metrics) > 0:
            first_key = next(iter(split_metrics))
            primary_metric = split_metrics[first_key]
            print(f"[Eval] Primary split '{primary_eval_split}' not available; tracking '{first_key}' instead.")
            primary_eval_split = first_key  # update for logging

        if primary_metric is not None:
            current_oIoU = primary_metric["overallIoU"]
            save_checkpoint = (best_primary_oIoU < current_oIoU)
            if save_checkpoint:
                print('Better epoch: {}\n'.format(epoch))
                best_primary_oIoU = current_oIoU
                best_epoch = epoch

                dict_to_save = {
                    'model': single_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_primary_oIoU': best_primary_oIoU,
                    'best_epoch': best_epoch,
                    'tracked_split': primary_eval_split,
                    'split_metrics': split_metrics,
                    # also save loss heads to fully resume
                    'text_decomposer': text_decomposer.state_dict(),
                    'align_lite': align_lite.state_dict(),
                }
                if text_model is not None:
                    state = text_model.module.state_dict() if hasattr(text_model, "module") else text_model.state_dict()
                    dict_to_save['text_model'] = state
                    # backward compatibility
                    if getattr(args, "language_encoder", "bert") == "bert":
                        dict_to_save['bert_model'] = state
                if use_amp:
                    dict_to_save['scaler'] = scaler.state_dict()

                ck_name = f"model_best_{args.model_id}_{primary_eval_split}.pth"
                utils.save_on_master(dict_to_save, os.path.join(args.output_dir, ck_name))

        if eval_only:
            break

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if best_epoch >= 0:
        print(f"[Summary] Best {primary_eval_split} overallIoU: {best_primary_oIoU:.2f} at epoch {best_epoch}")

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # Try distributed init (safe if not configured; your utils should no-op if env isn't set)
    try:
        utils.init_distributed_mode(args)
    except Exception as e:
        print(f"[DDP] init_distributed_mode raised ({e}); continuing single-process.")
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
