"""
Training script for the LAVT/Nova stack on remote-sensing referring segmentation.

Single-GPU best practice defaults:
- Start with frozen BERT (most stable / least memory).
- If validation plateaus, automatically enable LoRA on BERT (rank 4 by default).
- Resume auto-detects LoRA from checkpoint and rebuilds model accordingly.

PyCharm-friendly: no shell args required; env vars can override defaults.
"""

import os
import sys
import random
import gc
import warnings
from typing import Optional, Tuple
import importlib

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.optim import lr_scheduler

import transforms as T
import utils
import transformers
from transformers import BertModel

# Keep legacy layout search order consistent
for subdir in ("lib", "loss", "data"):
    module_dir = os.path.join(os.path.dirname(__file__), subdir)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

import _utils

segmentation_lib = None
if importlib.util.find_spec("lib.segmentation") is not None:
    segmentation_lib = importlib.import_module("lib.segmentation")
elif importlib.util.find_spec("segmentation") is not None:
    segmentation_lib = importlib.import_module("segmentation")
else:
    print("Warning: Could not import segmentation from lib or local module")

# Loss (composite)
from loss import (
    ALPHA_ALIGN,
    ALPHA_DICE,
    ALPHA_ORTHO,
    WARMUP_STEPS,
    AlignLite,
    TextDecomposer,
    composite_loss,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------- Auto LoRA-on-plateau defaults (PyCharm-friendly) --------------------
AUTO_LORA = bool(int(os.environ.get("RRSIS_AUTO_LORA", "1")))
AUTO_LORA_RANK = int(os.environ.get("RRSIS_AUTO_LORA_RANK", "4"))  # safer default than 8
AUTO_LORA_PATIENCE = int(os.environ.get("RRSIS_AUTO_LORA_PATIENCE", "5"))
AUTO_LORA_MIN_EPOCH = int(os.environ.get("RRSIS_AUTO_LORA_MIN_EPOCH", "10"))
# ----------------------------------------------------------------------------------------

# Speed defaults (PyCharm-friendly):
# - AMP speeds training a lot on A6000 (disable via RRSIS_AMP=0)
# - TF32 + cudnn benchmark speed conv/matmul (disable via RRSIS_FAST=0)
USE_AMP = bool(int(os.environ.get("RRSIS_AMP", "1")))
FAST_MODE = bool(int(os.environ.get("RRSIS_FAST", "1")))

# Silence noisy torchvision warning (it can spam per-sample and slow epochs heavily)
warnings.filterwarnings(
    "ignore",
    message="Argument interpolation should be of type InterpolationMode instead of int.*",
)

try:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
except Exception:
    autocast = None  # type: ignore
    GradScaler = None  # type: ignore

# Optional: import LoRA utilities (tolerant to repo layout)
LoRAConfig = None
apply_lora_adapters = None
freeze_non_lora_params = None
try:
    from lib._utils import LoRAConfig as _LoRAConfig, apply_lora_adapters as _ala, freeze_non_lora_params as _fnlp  # type: ignore
    LoRAConfig, apply_lora_adapters, freeze_non_lora_params = _LoRAConfig, _ala, _fnlp
except Exception:
    try:
        from _utils import LoRAConfig as _LoRAConfig, apply_lora_adapters as _ala, freeze_non_lora_params as _fnlp  # type: ignore
        LoRAConfig, apply_lora_adapters, freeze_non_lora_params = _LoRAConfig, _ala, _fnlp
    except Exception:
        pass


def _clear_cuda_cache() -> None:
    """Drain CUDA caching allocator and IPC handles when available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _configure_mp_sharing() -> None:
    # Fix for: OSError: [Errno 9] Bad file descriptor / EOFError in pin_memory thread
    # Caused by FD-based tensor sharing on some HPC setups.
    # You can override with env var: RRSIS_MP_SHARING=file_descriptor
    try:
        import torch.multiprocessing as mp  # type: ignore
        strategy = os.environ.get("RRSIS_MP_SHARING", "file_system").strip()
        if strategy:
            mp.set_sharing_strategy(strategy)
            print(f"[mp] sharing_strategy={strategy}")
    except Exception as e:
        print(f"[mp] Could not set sharing strategy: {e}")


def _configure_torch_speed() -> None:
    # TF32 + cudnn benchmark improve throughput a lot on Ampere (A6000).
    # For strict determinism, set RRSIS_FAST=0.
    if not torch.cuda.is_available():
        return
    if FAST_MODE:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        except Exception:
            pass


def _ensure_transformers_offline() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _resolve_clip_model_class():
    clip_text_cls = getattr(transformers, "CLIPTextModel", None)
    if clip_text_cls is not None:
        return clip_text_cls

    clip_model_cls = getattr(transformers, "CLIPModel", None)
    if clip_model_cls is not None:
        class _ClipTextWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.text_model = base.text_model

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                base = clip_model_cls.from_pretrained(*args, **kwargs)
                return cls(base)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return _ClipTextWrapper

    auto_cls = getattr(transformers, "AutoModel", None)
    if auto_cls is not None:
        return auto_cls

    raise RuntimeError("transformers installation lacks CLIP text support.")


def _load_clip_text_encoder(args) -> nn.Module:
    _ensure_transformers_offline()
    model_path = getattr(args, "clip_model_path", None)
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"[CLIP] Missing local model path: {model_path}")
    print(f"[CLIP] Loading offline CLIP text encoder from {model_path}")
    clip_cls = _resolve_clip_model_class()
    clip_model = clip_cls.from_pretrained(model_path, local_files_only=True)
    clip_model.cuda()
    return clip_model


def _load_clip_vision_encoder(args):
    if getattr(args, "visual_encoder", "swin") != "clip":
        return None
    vision_path = getattr(args, "clip_vision_onnx", None)
    if not vision_path or not os.path.isfile(vision_path):
        print(f"[CLIP] Vision encoder path missing or not found: {vision_path}")
        print("[CLIP]       Vision tower will be disabled (Swin backbone only).")
        return None
    if importlib.util.find_spec("onnxruntime") is None:
        print("[CLIP] onnxruntime is not installed; ONNX vision encoder will be disabled.")
        print("[CLIP] Install 'onnxruntime-gpu' (or 'onnxruntime') to enable the CLIP vision tower.")
        return None
    import onnxruntime as ort
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


def seed_everything(seed: int = 2401) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep behavior fast by default; set RRSIS_FAST=0 for deterministic runs.
    if FAST_MODE:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _infer_lora_rank_from_state(state_dict) -> Optional[int]:
    """Detect LoRA rank from checkpoint keys (so test/resume rebuild matches)."""
    if not isinstance(state_dict, dict):
        return None
    for k, v in state_dict.items():
        if isinstance(k, str) and k.endswith("lora_A") and hasattr(v, "shape") and len(v.shape) >= 1:
            try:
                return int(v.shape[0])
            except Exception:
                return None
    return None


def _enable_lora_text_encoder(
    model: nn.Module,
    args,
    optimizer: torch.optim.Optimizer,
    *,
    rank: int,
    text_model: Optional[nn.Module] = None,
) -> bool:
    """Apply LoRA wrappers to a text encoder and register LoRA params in optimizer."""
    if LoRAConfig is None or apply_lora_adapters is None or freeze_non_lora_params is None:
        print("[LoRA] LoRA utilities not importable; cannot enable LoRA automatically.")
        return False
    if text_model is None:
        if not hasattr(model, "text_encoder") or getattr(model, "text_encoder") is None:
            print("[LoRA] Model has no text_encoder; nothing to LoRA-wrap.")
            return False
        text_encoder = getattr(model, "text_encoder")
    else:
        text_encoder = text_model

    # Capture device BEFORE wrapping so new LoRA params follow the same device.
    # This directly prevents the CPU/GPU mismatch you hit after Auto-LoRA triggers.
    try:
        device_before = next(text_encoder.parameters()).device
    except StopIteration:
        device_before = torch.device("cpu")

    try:
        cfg = LoRAConfig(
            rank=rank,
            alpha=getattr(args, "lora_alpha", 32.0),
            dropout=getattr(args, "lora_dropout", 0.1),
            full_match=getattr(args, "lora_full_match", False),
        )
    except Exception as e:
        print(f"[LoRA] Failed to construct LoRAConfig: {e}")
        return False

    replaced = apply_lora_adapters(text_encoder, cfg)
    trainable, frozen = freeze_non_lora_params(text_encoder)

    # Ensure any newly created LoRA params are on the same device as the text encoder was before.
    try:
        text_encoder.to(device_before)
    except Exception:
        pass

    # flip model to train text encoder graph (but only LoRA params are trainable)
    setattr(model, "train_text_encoder", True)
    try:
        text_encoder.train(True)
    except Exception:
        pass

    setattr(model, "lora_replacements", replaced)
    setattr(model, "lora_trainable", trainable)

    # register newly created LoRA params with optimizer (keep current lr)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    if not lora_params:
        print("[LoRA] No LoRA params found after wrapping; not enabling.")
        return False

    existing = set(id(p) for g in optimizer.param_groups for p in g.get("params", []))
    new_params = [p for p in lora_params if id(p) not in existing]
    if new_params:
        current_lr = optimizer.param_groups[0].get("lr", getattr(args, "lr", 3e-5))
        optimizer.add_param_group(
            {"params": new_params, "lr": current_lr, "weight_decay": getattr(args, "weight_decay", 1e-2)}
        )

    print(f"[LoRA] Enabled LoRA on BERT (rank={rank}). wrapped={replaced} adapter_params≈{trainable}")
    return True


def _save_checkpoint(
    output_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    text_model: Optional[nn.Module],
    encoder_type: str,
    best_overall_iou: float,
) -> None:
    """Persist only the best checkpoint to save disk space."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "checkpoint_best.pth")
    payload = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_overall_iou": best_overall_iou,
    }
    if text_model is not None:
        if encoder_type == "clip":
            payload["text_model"] = text_model.state_dict()
        else:
            payload["bert_model"] = text_model.state_dict()
    payload["language_encoder"] = encoder_type
    torch.save(payload, save_path)
    print(f"Best checkpoint saved to {save_path}")


def _load_refer_dataset(args, split: str, transform):
    if args.dataset == "refsegrs":
        from data.refsegrs_refer_bert import ReferDataset  # type: ignore
    elif args.dataset == "rrsisd":
        from rrsisd_refer_bert import ReferDataset  # type: ignore
    elif args.dataset == "risbench":
        from risbench_refer_bert import ReferDataset  # type: ignore
    elif args.dataset == "nwpu-refer":
        from nwpu_refer_bert import ReferDataset  # type: ignore
    elif args.dataset == "vdd_ris":
        from data.vdd_ris_refer_bert import ReferDataset  # type: ignore
    else:
        raise ValueError(f"No refer dataset is called [{args.dataset}]")

    eval_mode = split != "train"
    ds = ReferDataset(
        args,
        split=split,
        image_transforms=transform,
        target_transforms=None,
        eval_mode=eval_mode,
    )
    return ds, 2


def get_dataset(image_set: str, transform, args):
    return _load_refer_dataset(args, image_set, transform)


def IoU(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
    pred_labels = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred_labels, gt))
    union = torch.sum(torch.add(pred_labels, gt)) - intersection
    if intersection == 0 or union == 0:
        iou = 0.0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def _forward_model(
    model: nn.Module,
    batch,
    text_model: Optional[nn.Module] = None,
    encoder_type: str = "bert",
    train_text_encoder: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    image, target, sentences, attentions, target_masks, position_masks, _ = batch
    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    sentences = sentences.cuda(non_blocking=True)
    attentions = attentions.cuda(non_blocking=True)
    target_masks = target_masks.cuda(non_blocking=True)
    position_masks = position_masks.cuda(non_blocking=True)

    if sentences.dim() == 3 and sentences.size(1) == 1:
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        target_masks = target_masks.squeeze(1)
        position_masks = position_masks.squeeze(1)

    def _forward_single(sentence_tokens, attention_mask, target_mask, position_mask):
        if text_model is not None:
            if not train_text_encoder:
                text_model.eval()
                with torch.no_grad():
                    if encoder_type == "clip":
                        last_hidden_states = text_model(
                            input_ids=sentence_tokens,
                            attention_mask=attention_mask,
                        ).last_hidden_state
                    else:
                        last_hidden_states = text_model(sentence_tokens, attention_mask=attention_mask)[0]
            else:
                text_model.train(True)
                if encoder_type == "clip":
                    last_hidden_states = text_model(
                        input_ids=sentence_tokens,
                        attention_mask=attention_mask,
                    ).last_hidden_state
                else:
                    last_hidden_states = text_model(sentence_tokens, attention_mask=attention_mask)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            if not train_text_encoder:
                embedding = embedding.detach()
            attentions_exp = attention_mask.unsqueeze(dim=-1)
            if encoder_type == "clip" and hasattr(model, "text_encoder"):
                if not hasattr(model, "_clip_adapter"):
                    model._clip_adapter = _utils.LAVT(model.backbone, model.classifier)
                return model._clip_adapter(image, embedding, attentions_exp), last_hidden_states
            try:
                return model(image, embedding, attentions_exp, target_mask, position_mask), last_hidden_states
            except TypeError:
                return model(image, embedding, attentions_exp), last_hidden_states
        return model(image, sentence_tokens, attention_mask, target_mask, position_mask), None

    if sentences.dim() == 3:
        outputs = []
        hidden_states = []
        for sent_idx in range(sentences.size(1)):
            out, hidden = _forward_single(
                sentences[:, sent_idx, :],
                attentions[:, sent_idx, :],
                target_masks[:, sent_idx, :],
                position_masks[:, sent_idx, :],
            )
            outputs.append(out)
            hidden_states.append(hidden)
        output = torch.stack(outputs, dim=1).mean(dim=1)
        last_hidden = None
        if hidden_states and hidden_states[0] is not None:
            last_hidden = torch.stack(hidden_states, dim=1).mean(dim=1)
    else:
        output, last_hidden = _forward_single(sentences, attentions, target_masks, position_masks)
    attn_mask = attentions
    if last_hidden is None and hasattr(model, "last_hidden") and getattr(model, "last_hidden") is not None:
        last_hidden = getattr(model, "last_hidden")
        model_attn = getattr(model, "last_attn_mask", None)
        if model_attn is not None:
            attn_mask = model_attn
    return output, target, last_hidden, attn_mask


def evaluate(
    model: nn.Module,
    data_loader,
    text_model: Optional[nn.Module],
    encoder_type: str,
    epoch: int,
    text_decomposer: Optional[TextDecomposer],
    align_lite: Optional[AlignLite],
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val: "
    total_its = 0
    acc_ious = 0.0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    inference_ctx = getattr(torch, "inference_mode", torch.no_grad)

    with inference_ctx():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1

            if autocast is not None and torch.cuda.is_available() and USE_AMP:
                with autocast():
                    output, target, _, _ = _forward_model(
                        model,
                        data,
                        text_model,
                        encoder_type,
                        train_text_encoder=False,
                    )
            else:
                output, target, _, _ = _forward_model(
                    model,
                    data,
                    text_model,
                    encoder_type,
                    train_text_encoder=False,
                )

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1

        iou = acc_ious / max(total_its, 1)

    mean_IoU_arr = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU_arr) if mean_IoU_arr.size > 0 else 0.0
    print("Final results:")
    print("Mean IoU is %.2f\n" % (mIoU * 100.0))
    results_str = ""
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += "    precision@%s = %.2f\n" % (
            str(eval_seg_iou_list[n_eval_iou]),
            seg_correct[n_eval_iou] * 100.0 / max(seg_total, 1),
        )
    results_str += "    overall IoU = %.2f\n" % (cum_I * 100.0 / max(cum_U, 1))
    print(results_str)

    return 100 * iou, 100 * cum_I / max(cum_U, 1)


def _build_dataloader(dataset, *, batch_size: int, sampler, num_workers: int, pin_memory: bool, drop_last: bool):
    # DataLoader kwargs differ slightly across torch versions; keep it tolerant for Python 3.7 environments.
    base_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Enable throughput features when workers > 0.
    # Persistent workers can be disabled via RRSIS_PERSISTENT_WORKERS=0 if your torch build is finicky.
    persistent_ok = bool(int(os.environ.get("RRSIS_PERSISTENT_WORKERS", "1")))
    prefetch = int(os.environ.get("RRSIS_PREFETCH_FACTOR", "2"))
    if num_workers > 0 and persistent_ok:
        base_kwargs["persistent_workers"] = True
        base_kwargs["prefetch_factor"] = max(1, prefetch)

    try:
        return torch.utils.data.DataLoader(**base_kwargs)
    except TypeError:
        # Fallback for older torch builds that don't support persistent_workers/prefetch_factor
        base_kwargs.pop("persistent_workers", None)
        base_kwargs.pop("prefetch_factor", None)
        return torch.utils.data.DataLoader(**base_kwargs)


def train_one_epoch(
    model: nn.Module,
    optimizer,
    data_loader,
    scheduler,
    epoch: int,
    print_freq: int,
    iterations: int,
    text_model: Optional[nn.Module],
    encoder_type: str,
    train_text_encoder: bool,
    text_decomposer: Optional[TextDecomposer],
    align_lite: Optional[AlignLite],
    scaler,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = f'Epoch: [{epoch}]'
    train_loss = 0.0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        optimizer.zero_grad(set_to_none=True)

        if autocast is not None and scaler is not None and scaler.is_enabled():
            with autocast():
                output, target, last_hidden, attn_mask = _forward_model(
                    model,
                    data,
                    text_model,
                    encoder_type,
                    train_text_encoder=train_text_encoder,
                )
                ramp = min(1.0, float(iterations) / float(max(WARMUP_STEPS, 1)))
                loss, parts = composite_loss(
                    output,
                    target,
                    last_hidden_states=last_hidden,
                    attn_mask=attn_mask.squeeze(1) if attn_mask.dim() == 3 else attn_mask,
                    text_decomposer=text_decomposer,
                    align_lite=align_lite,
                    alpha_dice=ALPHA_DICE * ramp,
                    alpha_ortho=ALPHA_ORTHO,
                    alpha_align=ALPHA_ALIGN * ramp,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output, target, last_hidden, attn_mask = _forward_model(
                model,
                data,
                text_model,
                encoder_type,
                train_text_encoder=train_text_encoder,
            )
            ramp = min(1.0, float(iterations) / float(max(WARMUP_STEPS, 1)))
            loss, parts = composite_loss(
                output,
                target,
                last_hidden_states=last_hidden,
                attn_mask=attn_mask.squeeze(1) if attn_mask.dim() == 3 else attn_mask,
                text_decomposer=text_decomposer,
                align_lite=align_lite,
                alpha_dice=ALPHA_DICE * ramp,
                alpha_ortho=ALPHA_ORTHO,
                alpha_align=ALPHA_ALIGN * ramp,
            )
            loss.backward()
            optimizer.step()

        train_loss += float(loss.item())
        iterations += 1
        if parts:
            metric_logger.update(
                loss=float(loss.item()),
                lr=optimizer.param_groups[0]["lr"],
                l_ce=float(parts["l_ce"].item()),
                l_dice=float(parts["l_dice"].item()),
                l_ortho=float(parts["l_ortho"].item()),
                l_align=float(parts["l_align"].item()),
            )
        else:
            metric_logger.update(loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])

        del loss, output, target, data

    print(f"Training loss for epoch {epoch}: {train_loss / max(total_its, 1)}")


def _collect_debug_states(model: nn.Module) -> dict:
    debug = {}
    for module in model.modules():
        if hasattr(module, "debug_state") and callable(getattr(module, "debug_state")):
            try:
                name = module.__class__.__name__
                state = module.debug_state()
                if state:
                    debug[name] = state
            except Exception:
                continue
    return debug


def _print_debug_states(debug: dict) -> None:
    if not debug:
        print("[debug] No module debug_state available.")
        return
    print("[debug] Module diagnostics:")
    for name, state in debug.items():
        if isinstance(state, dict):
            entries = ", ".join(f"{k}={v}" for k, v in state.items())
        else:
            entries = str(state)
        print(f"[debug] - {name}: {entries}")


def main(args):
    if segmentation_lib is None:
        raise ImportError("segmentation module could not be imported; ensure it is available on sys.path")

    _configure_mp_sharing()
    _configure_torch_speed()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # If resuming, detect whether checkpoint contains LoRA and rebuild accordingly.
    resume_ckpt = None
    if getattr(args, "resume", ""):
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        resume_state = resume_ckpt.get("model", resume_ckpt)
        inferred_rank = _infer_lora_rank_from_state(resume_state)
        if inferred_rank is not None and getattr(args, "lora_rank", None) is None:
            args.lora_rank = inferred_rank
            print(f"[resume] Detected LoRA in checkpoint. Setting args.lora_rank={args.lora_rank} for correct rebuild.")

    print("\n[***] Set Datasets")
    dataset, _ = get_dataset("train", get_transform(args=args), args=args)
    dataset_test, _ = get_dataset("val", get_transform(args=args), args=args)

    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = _build_dataloader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = _build_dataloader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print("\n[***] Build Model")
    if getattr(args, "pretrained_swin_weights", "") and "window12" in str(args.pretrained_swin_weights) and not getattr(args, "window12", False):
        args.window12 = True
        print("[Swin] Detected window12 weights; enabling --window12 for compatible backbone init.")
    model = segmentation_lib.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    model.cuda()

    # Optional CLIP vision tower (ONNX). Loaded for offline readiness when requested.
    _ = _load_clip_vision_encoder(args)

    # Only instantiate external text encoder when needed (always for CLIP)
    encoder_type = getattr(args, "language_encoder", "bert")
    text_model: Optional[nn.Module]
    if encoder_type == "clip":
        text_model = _load_clip_text_encoder(args)
    elif getattr(model, "text_encoder", None) is None:
        text_model = BertModel.from_pretrained(args.ck_bert)
        text_model.pooler = None
        text_model.cuda()
    else:
        text_model = None

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt.get('model', resume_ckpt), strict=False)
        if text_model is not None:
            if 'text_model' in resume_ckpt:
                text_model.load_state_dict(resume_ckpt['text_model'])
            elif 'bert_model' in resume_ckpt:
                text_model.load_state_dict(resume_ckpt['bert_model'])

    # Print language encoder mode
    if encoder_type == "clip":
        print("[CLIP] Using CLIP text encoder.")
    elif getattr(args, "lora_rank", None) is not None:
        print(f"[BERT] LoRA is ON at start (rank={args.lora_rank}).")
    else:
        print("[BERT] Frozen at start (default). Auto-LoRA-on-plateau is "
              + ("ON" if AUTO_LORA else "OFF") + ".")

    # If LoRA already enabled at build-time, show counters (if model exposes them)
    if getattr(args, "lora_rank", None) is not None:
        lora_wrapped = getattr(model, "lora_replacements", None)
        lora_trainable = getattr(model, "lora_trainable", None)
        if lora_wrapped is not None:
            print(f"[LoRA] wrapped_layers={lora_wrapped} adapter_params≈{lora_trainable}")

    text_dim = 512 if encoder_type == "clip" else getattr(args, "lang_dim", 768)
    text_decomposer = TextDecomposer(dim_in=text_dim, dim_out=256).cuda()
    align_lite = AlignLite(txt_dim=text_dim, hid=256).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if GradScaler is not None and torch.cuda.is_available() and USE_AMP:
        scaler = GradScaler()
    else:
        scaler = None

    best_overall_iou = -float("inf")
    epochs_since_improve = 0
    lora_enabled_midrun = False
    train_text_encoder = False

    if getattr(args, "lora_rank", None) is not None:
        if encoder_type == "clip" and text_model is not None:
            ok = _enable_lora_text_encoder(model, args, optimizer, rank=args.lora_rank, text_model=text_model)
            if ok:
                train_text_encoder = True
                lora_enabled_midrun = True
        elif encoder_type == "bert" and getattr(model, "text_encoder", None) is not None:
            ok = _enable_lora_text_encoder(model, args, optimizer, rank=args.lora_rank)
            if ok:
                lora_enabled_midrun = True

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        train_one_epoch(
            model,
            optimizer,
            data_loader,
            scheduler,
            epoch,
            args.print_freq,
            0,
            text_model,
            encoder_type,
            train_text_encoder,
            text_decomposer,
            align_lite,
            scaler,
        )
        iou, overallIoU = evaluate(
            model,
            data_loader_test,
            text_model,
            encoder_type,
            epoch,
            text_decomposer,
            align_lite,
        )

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        debug = _collect_debug_states(model)
        _print_debug_states(debug)

        improved = overallIoU > best_overall_iou
        if improved:
            best_overall_iou = overallIoU
            epochs_since_improve = 0
            _save_checkpoint(
                args.output_dir,
                epoch,
                model,
                optimizer,
                scheduler,
                text_model,
                encoder_type,
                best_overall_iou,
            )
        else:
            epochs_since_improve += 1

        # Auto-enable LoRA if plateaued (only if we started frozen)
        if (
            AUTO_LORA
            and not lora_enabled_midrun
            and encoder_type in {"bert", "clip"}
            and getattr(args, "lora_rank", None) is None
            and epoch >= AUTO_LORA_MIN_EPOCH
            and epochs_since_improve >= AUTO_LORA_PATIENCE
        ):
            print(f"[Auto-LoRA] Plateau detected: no val improvement for {epochs_since_improve} epochs "
                  f"(epoch={epoch}). Enabling LoRA rank={AUTO_LORA_RANK} and continuing.")
            if encoder_type == "clip" and text_model is not None:
                ok = _enable_lora_text_encoder(model, args, optimizer, rank=AUTO_LORA_RANK, text_model=text_model)
                if ok:
                    train_text_encoder = True
            else:
                ok = _enable_lora_text_encoder(model, args, optimizer, rank=AUTO_LORA_RANK)
            if ok:
                args.lora_rank = AUTO_LORA_RANK
                lora_enabled_midrun = True
                epochs_since_improve = 0
                # ensure mode consistent for next epoch
                model.train()

        scheduler.step()

    print("Training completed.")


if __name__ == "__main__":
    from args import get_parser

    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
