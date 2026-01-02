"""
Test script for the LAVT/Nova stack (remote-sensing referring segmentation).

- Loads ONE split (typically --split test)
- Loads checkpoint (--resume), runs evaluation, prints mIoU/overallIoU/precision@τ
- Optional visualization saving to experiments/test_vis

PyCharm-friendly:
- If args.resume is empty, we auto-fill DEFAULT_RESUME_PATH so you can click Run
  without providing script parameters.
"""

import os
import sys
import time
import datetime
import gc
import importlib
import warnings
from typing import Optional

import cv2
import numpy as np
import torch
import torch.utils.data
from torch import nn

import transforms as T
import utils
import transformers
from transformers import BertModel
from torchvision.transforms import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_RESUME_PATH = "/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/checkpoints/checkpoint_best.pth"

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

save_dir = "experiments/test_vis"

# Speed defaults (PyCharm-friendly):
# - AMP speeds inference on A6000 (disable via RRSIS_AMP=0)
# - TF32 + cudnn benchmark speed conv/matmul (disable via RRSIS_FAST=0)
USE_AMP = bool(int(os.environ.get("RRSIS_AMP", "1")))
FAST_MODE = bool(int(os.environ.get("RRSIS_FAST", "1")))

try:
    from torch.cuda.amp import autocast  # type: ignore
except Exception:
    autocast = None  # type: ignore

# Silence noisy torchvision warning (it can spam per-sample and slow epochs heavily)
warnings.filterwarnings(
    "ignore",
    message="Argument interpolation should be of type InterpolationMode instead of int.*",
)


def _clear_cuda_cache() -> None:
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


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _infer_lora_rank_from_state(state_dict) -> Optional[int]:
    if not isinstance(state_dict, dict):
        return None
    for k, v in state_dict.items():
        if isinstance(k, str) and k.endswith("lora_A") and hasattr(v, "shape") and len(v.shape) >= 1:
            try:
                return int(v.shape[0])
            except Exception:
                return None
    return None


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

    ds = ReferDataset(
        args,
        split=split,
        image_transforms=transform,
        target_transforms=None,
        eval_mode=True,
    )
    return ds, 2


def get_dataset(image_set: str, transform, args):
    return _load_refer_dataset(args, image_set, transform)


def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def computeIoU(pred_seg: np.ndarray, gd_seg: np.ndarray):
    if pred_seg.ndim == 3:
        pred_seg = pred_seg[0]
    if gd_seg.ndim == 3:
        gd_seg = gd_seg[0]
    pred = pred_seg.astype(bool)
    gd = gd_seg.astype(bool)
    I = np.sum(np.logical_and(pred, gd))
    U = np.sum(np.logical_or(pred, gd))
    return I, U


def save_pred_targ_results(output_mask: np.ndarray, target: np.ndarray, image: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pred = output_mask[0, :, :]
    targ = target[0, :, :]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]

    im = F.normalize(image, mean=inv_mean, std=inv_std)
    im = im[0].cpu().detach().numpy().transpose(1, 2, 0)
    im = np.uint8(np.clip(im * 255.0, 0, 255))

    pred_red = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    pred_red[:, :] = (0, 0, 255)
    pred_mask3 = np.repeat(pred[:, :, np.newaxis], 3, axis=-1)
    pred_mask3 = np.uint8(pred_red * pred_mask3)
    pred_img = cv2.addWeighted(im, 0.5, pred_mask3, 0.5, 0)
    cv2.imwrite(save_path + "_pred.png", pred_img)

    targ_red = np.zeros((targ.shape[0], targ.shape[1], 3), dtype=np.uint8)
    targ_red[:, :] = (0, 0, 255)
    targ_mask3 = np.repeat(targ[:, :, np.newaxis], 3, axis=-1)
    targ_mask3 = np.uint8(targ_red * targ_mask3)
    targ_img = cv2.addWeighted(im, 0.5, targ_mask3, 0.5, 0)
    cv2.imwrite(save_path + "_targ.png", targ_img)


def _forward_model(model: nn.Module, batch, device: torch.device, text_model: Optional[nn.Module], encoder_type: str):
    image, target, sentences, attentions, target_masks, position_masks, save_prefix = batch

    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    sentences = sentences.to(device, non_blocking=True)
    attentions = attentions.to(device, non_blocking=True)
    target_masks = target_masks.to(device, non_blocking=True)
    position_masks = position_masks.to(device, non_blocking=True)

    sentences = sentences.squeeze(1)
    attentions = attentions.squeeze(1)
    target_masks = target_masks.squeeze(1)
    position_masks = position_masks.squeeze(1)

    if text_model is not None:
        if encoder_type == "clip":
            last_hidden_states = text_model(input_ids=sentences, attention_mask=attentions).last_hidden_state
        else:
            last_hidden_states = text_model(sentences, attention_mask=attentions)[0]
        embedding = last_hidden_states.permute(0, 2, 1).detach()
        attentions_img = attentions.unsqueeze(dim=-1)
        if encoder_type == "clip" and hasattr(model, "text_encoder"):
            if not hasattr(model, "_clip_adapter"):
                model._clip_adapter = _utils.LAVT(model.backbone, model.classifier)
            output = model._clip_adapter(image, embedding, attentions_img)
        else:
            try:
                output = model(image, embedding, attentions_img, target_masks, position_masks)
            except TypeError:
                output = model(image, embedding, attentions_img)
    else:
        output = model(image, sentences, attentions, target_masks, position_masks)
    return output, target, image, save_prefix


def _build_dataloader(dataset, *, batch_size: int, sampler, num_workers: int, pin_memory: bool):
    base_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Throughput features (can be disabled via env vars if your torch build is finicky)
    persistent_ok = bool(int(os.environ.get("RRSIS_PERSISTENT_WORKERS", "1")))
    prefetch = int(os.environ.get("RRSIS_PREFETCH_FACTOR", "2"))

    if num_workers > 0 and persistent_ok:
        base_kwargs["persistent_workers"] = True
        base_kwargs["prefetch_factor"] = max(1, prefetch)

    try:
        return torch.utils.data.DataLoader(**base_kwargs)
    except TypeError:
        base_kwargs.pop("persistent_workers", None)
        base_kwargs.pop("prefetch_factor", None)
        return torch.utils.data.DataLoader(**base_kwargs)


def evaluate(model: nn.Module, data_loader, device: torch.device, text_model: Optional[nn.Module], encoder_type: str):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    save_vis = bool(int(os.environ.get("RRSIS_SAVE_TEST_VIS", "0")))
    if save_vis:
        os.makedirs(save_dir, exist_ok=True)

    inference_ctx = getattr(torch, "inference_mode", torch.no_grad)

    start_time = time.time()
    with inference_ctx():
        for batch in metric_logger.log_every(data_loader, 100, header):
            if autocast is not None and torch.cuda.is_available() and USE_AMP:
                with autocast():
                    output, target_t, image_t, save_prefix = _forward_model(model, batch, device, text_model, encoder_type)
            else:
                output, target_t, image_t, save_prefix = _forward_model(model, batch, device, text_model, encoder_type)

            pred = output.argmax(1).cpu().numpy().astype(np.uint8)
            targ = target_t.cpu().numpy().astype(np.uint8)
            if targ.ndim == 4 and targ.shape[1] == 1:
                targ = targ[:, 0]

            B = pred.shape[0]
            for b in range(B):
                I, U = computeIoU(pred[b], targ[b])
                this_iou = 0.0 if U == 0 else float(I) / float(U)

                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U

                for k, thr in enumerate(eval_seg_iou_list):
                    seg_correct[k] += (this_iou >= thr)

                seg_total += 1

                if save_vis:
                    if isinstance(save_prefix, (list, tuple)):
                        name = str(save_prefix[b])
                    else:
                        name = str(save_prefix)
                    out_path = os.path.join(save_dir, f"{name}_{this_iou:.4f}")
                    save_pred_targ_results(
                        pred[b:b + 1],
                        targ[b:b + 1],
                        image_t[b:b + 1].cpu(),
                        out_path,
                    )

            del output, target_t, image_t

    mean_IoU_arr = np.array(mean_IoU, dtype=np.float32)
    mIoU = float(mean_IoU_arr.mean()) if mean_IoU_arr.size > 0 else 0.0
    overall = 0.0 if cum_U == 0 else (float(cum_I) / float(cum_U))

    print("Final results:")
    print("Mean IoU is %.2f\n" % (mIoU * 100.0))

    results_str = ""
    for k, thr in enumerate(eval_seg_iou_list):
        results_str += "    precision@%s = %.2f\n" % (str(thr), seg_correct[k] * 100.0 / max(seg_total, 1))
    results_str += "    overall IoU = %.2f\n" % (overall * 100.0)
    print(results_str)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Total test time {}".format(total_time_str))
    print("Test time for one sample %.4f sec" % (total_time / max(seg_total, 1)))


def main(args):
    if segmentation_lib is None:
        raise ImportError("segmentation module could not be imported; ensure it is available on sys.path")

    _configure_mp_sharing()
    _configure_torch_speed()

    if not getattr(args, "resume", ""):
        args.resume = DEFAULT_RESUME_PATH
        print(f"[test.py] args.resume was empty; using DEFAULT_RESUME_PATH:\n  {args.resume}")
    else:
        print(f"[test.py] Using resume from args:\n  {args.resume}")

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found at: {args.resume}")

    device = torch.device(args.device)

    # Load checkpoint first to detect LoRA
    checkpoint = torch.load(args.resume, map_location="cpu")
    model_state = checkpoint.get("model", checkpoint)
    model_state = _strip_module_prefix(model_state)

    inferred_rank = _infer_lora_rank_from_state(model_state)
    if inferred_rank is not None:
        if getattr(args, "lora_rank", None) is None:
            args.lora_rank = inferred_rank
            print(f"[test.py] Detected LoRA in checkpoint. Setting args.lora_rank={args.lora_rank} for correct rebuild.")
        elif int(args.lora_rank) != int(inferred_rank):
            print(f"[test.py] Warning: args.lora_rank={args.lora_rank} but checkpoint rank={inferred_rank}. "
                  f"Using args.lora_rank={args.lora_rank}.")

    dataset_test, _ = get_dataset(args.split, get_transform(args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = _build_dataloader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=getattr(args, "pin_mem", False),
    )

    build_fn = segmentation_lib.__dict__.get(args.model, None)
    if build_fn is None:
        build_fn = getattr(segmentation_lib, args.model, None)
    if build_fn is None:
        raise KeyError(f"Model factory '{args.model}' not found in segmentation module.")

    single_model = build_fn(pretrained=getattr(args, "pretrained_swin_weights", ""), args=args)

    missing, unexpected = single_model.load_state_dict(model_state, strict=False)

    print(f"[test.py] Loaded checkpoint keys. missing={len(missing)} unexpected={len(unexpected)}")
    if "epoch" in checkpoint:
        print(f"[test.py] checkpoint epoch = {checkpoint['epoch']}")
    if "best_overall_iou" in checkpoint:
        print(f"[test.py] best_overall_iou (val) = {checkpoint['best_overall_iou']}")

    _ = _load_clip_vision_encoder(args)

    encoder_type = getattr(args, "language_encoder", "bert")
    text_model: Optional[nn.Module] = None
    if encoder_type == "clip":
        text_model = _load_clip_text_encoder(args)
    elif getattr(single_model, "text_encoder", None) is None:
        text_model = BertModel.from_pretrained(args.ck_bert).to(device)

    if text_model is not None:
        if "text_model" in checkpoint:
            text_model.load_state_dict(checkpoint["text_model"])
        elif "bert_model" in checkpoint:
            text_model.load_state_dict(checkpoint["bert_model"])
        text_model = text_model.to(device)

    model = single_model.to(device)
    evaluate(model, data_loader_test, device=device, text_model=text_model, encoder_type=encoder_type)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()

    print("Image size: {}".format(str(args.img_size)))
    print("Dataset: {}, split: {}, model: {}".format(args.dataset, args.split, args.model))
    main(args)
