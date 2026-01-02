import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import transformers
from transformers import BertTokenizer
from refer.refer_ import REFER
import nltk

# Align NLTK data path with existing pipelines
_NLTK_DATA_PATH = "/10T/students/doctor/2025/zum/anaconda3/share/nltk_data"
if not os.path.exists(_NLTK_DATA_PATH):
    os.makedirs(_NLTK_DATA_PATH, exist_ok=True)
os.environ["NLTK_DATA"] = _NLTK_DATA_PATH

# Align NLTK data path with existing pipelines
_NLTK_DATA_PATH = "/10T/students/doctor/2025/zum/anaconda3/share/nltk_data"
if not os.path.exists(_NLTK_DATA_PATH):
    os.makedirs(_NLTK_DATA_PATH)
os.environ["NLTK_DATA"] = _NLTK_DATA_PATH


@dataclass
class Sample:
    image_name: str
    mask_name: Optional[str]
    sentence: str
    ref_id: Optional[int] = None


def _split_tokens(line: str) -> List[str]:
    for sep in ["###", "#", "\t", ",", "|"]:
        if sep in line:
            parts = line.split(sep)
            break
    else:
        parts = line.split()
    return [p.strip() for p in parts if p.strip()]


def _parse_metadata_line(line: str) -> Optional[Sample]:
    line = line.strip()
    if not line:
        return None

    parts = _split_tokens(line)
    if not parts:
        return None

    if len(parts) >= 3:
        image_name, mask_name, sentence = parts[0], parts[1], " ".join(parts[2:])
    elif len(parts) == 2:
        image_name, mask_name, sentence = parts[0], "", parts[1]
    else:
        image_name, mask_name, sentence = parts[0], "", ""

    image_name = os.path.basename(image_name)
    base_image, image_ext = os.path.splitext(image_name)
    if not image_ext:
        image_name = f"{image_name}.jpg"

    if not mask_name:
        mask_name = f"{os.path.splitext(image_name)[0]}.png"
    elif not os.path.splitext(mask_name)[1]:
        mask_name = f"{mask_name}.png"
    mask_name = os.path.basename(mask_name) if mask_name else mask_name

    return Sample(image_name=image_name, mask_name=mask_name, sentence=sentence)


def _normalize_seq(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    tensor = tensor.view(-1)
    if tensor.numel() >= max_len:
        return tensor[:max_len]
    # Pad with zeros if shorter
    pad = torch.zeros(max_len - tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _align_token_tensors(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.view(-1)
    attention_mask = attention_mask.view(-1)
    seq_len = min(max_len, input_ids.numel(), attention_mask.numel())
    input_ids = input_ids[:seq_len]
    attention_mask = attention_mask[:seq_len]
    # Pad if needed
    if seq_len < max_len:
        pad_ids = torch.zeros(max_len - seq_len, dtype=input_ids.dtype, device=input_ids.device)
        pad_mask = torch.zeros(max_len - seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        input_ids = torch.cat([input_ids, pad_ids], dim=0)
        attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
    return input_ids, attention_mask


def _safe_word_tokenize(text: str):
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _safe_pos_tag(tokens):
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        return [(tok, "NN") for tok in tokens]


def _load_tokenizer(args):
    cache_dir = getattr(args, "hf_cache_dir", None) or None
    language_encoder = getattr(args, "language_encoder", "bert")
    if language_encoder == "clip":
        clip_path = getattr(args, "clip_model_path", "")
        if not clip_path or not os.path.isdir(clip_path):
            raise RuntimeError(f"CLIP tokenizer path not found: {clip_path}")
        clip_tok_cls = getattr(transformers, "CLIPTokenizerFast", None) or getattr(transformers, "CLIPTokenizer", None)
        if clip_tok_cls is not None:
            return clip_tok_cls.from_pretrained(clip_path, cache_dir=cache_dir, local_files_only=True)
        auto_cls = getattr(transformers, "AutoTokenizer", None)
        if auto_cls is not None:
            return auto_cls.from_pretrained(clip_path, cache_dir=cache_dir, local_files_only=True)
        gpt_cls = getattr(transformers, "GPT2TokenizerFast", None) or getattr(transformers, "GPT2Tokenizer", None)
        if gpt_cls is not None:
            vocab_file = os.path.join(clip_path, "vocab.json")
            merges_file = os.path.join(clip_path, "merges.txt")
            tok = gpt_cls(vocab_file=vocab_file, merges_file=merges_file)
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
            return tok
        raise RuntimeError("CLIP tokenizer classes not available in this transformers build.")
    bert_path = getattr(args, "bert_tokenizer", "bert-base-uncased")
    local_only = os.path.isdir(bert_path)
    return BertTokenizer.from_pretrained(bert_path, cache_dir=cache_dir, local_files_only=local_only)


class ReferDataset(Dataset):
    def __init__(
        self,
        args,
        split: str = "train",
        image_transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
        eval_mode: bool = False,
    ) -> None:
        self.split = split
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.eval_mode = eval_mode

        self.max_tokens = getattr(args, "max_seq_length", 40)
        self.tokenizer = _load_tokenizer(args)

        self.refer = self._maybe_init_refer(args)
        self.src_dir, self.gt_dir, self.metadata_path = self._resolve_paths(args, split)
        if self.refer is not None:
            self.samples = self._load_samples_from_refer(split)
        else:
            self.samples = self._load_samples(self.metadata_path)

        if not self.samples:
            source_desc = (
                self.metadata_path if self.refer is None else f"REFER@{args.refer_data_root}"
            )
            raise ValueError(f"No samples loaded for split '{split}' from {source_desc}")

    def _maybe_init_refer(self, args) -> Optional[REFER]:
        """Instantiate REFER when refs metadata is available; otherwise fall back."""
        split_by = getattr(args, "splitBy", "unc")
        data_root = getattr(args, "refer_data_root", None)
        if data_root is None:
            return None

        refs_path = os.path.join(data_root, args.dataset, f"refs({split_by}).p")
        if not os.path.exists(refs_path):
            return None

        try:
            return REFER(data_root, args.dataset, split_by)
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"Warning: failed to initialize REFER for {args.dataset} at {data_root}: {exc}")
            return None

    def _resolve_paths(self, args, split: str) -> Tuple[str, str, str]:
        if split == "train":
            return (
                args.vdd_ris_train_src,
                args.vdd_ris_train_gt,
                args.vdd_ris_train_meta,
            )
        if split == "val":
            return (
                args.vdd_ris_val_src,
                args.vdd_ris_val_gt,
                args.vdd_ris_val_meta,
            )
        if split == "test":
            return (
                args.vdd_ris_test_src,
                args.vdd_ris_test_gt,
                args.vdd_ris_test_meta,
            )
        raise ValueError(f"Unsupported split '{split}' for VDD-RIS.")

    def _load_samples(self, metadata_path: str) -> List[Sample]:
        samples: List[Sample] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_metadata_line(line)
                if parsed is not None:
                    samples.append(parsed)
        return samples

    def _load_samples_from_refer(self, split: str) -> List[Sample]:
        ref_ids_for_split = self.refer.getRefIds(split=split)
        samples: List[Sample] = []
        for ref_id in ref_ids_for_split:
            ref_obj = self.refer.Refs[ref_id]
            img_info = self.refer.Imgs[ref_obj["image_id"]]
            image_name = img_info["file_name"]
            for sent in ref_obj.get("sentences", []):
                sentence_raw = sent.get("raw") or sent.get("sent") or ""
                samples.append(Sample(image_name=image_name, mask_name=None, sentence=sentence_raw, ref_id=ref_id))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_from_paths(self, sample: Sample) -> Tuple[Image.Image, Image.Image]:
        img_path = os.path.join(self.src_dir, sample.image_name)
        mask_path = os.path.join(self.gt_dir, sample.mask_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def _load_image_from_refer(self, sample: Sample) -> Tuple[Image.Image, Image.Image]:
        if sample.ref_id is None:
            raise ValueError("ref_id required when using REFER pipeline.")
        ref_obj = self.refer.Refs[sample.ref_id]
        img_info = self.refer.Imgs[ref_obj["image_id"]]
        img_path = os.path.join(self.refer.IMAGE_DIR, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        mask_data = self.refer.getMaskxml(ref_obj)
        mask_np = (np.array(mask_data["mask"]) > 0).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode="L")
        return image, mask

    def _tokenize_sentence(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0]

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if self.refer is not None and sample.ref_id is not None:
            image, mask = self._load_image_from_refer(sample)
        else:
            image, mask = self._load_image_from_paths(sample)

        if self.image_transforms is not None:
            image, mask = self.image_transforms(image, mask)
        else:
            image = TF.to_tensor(image)
            mask_np = (np.array(mask) > 0).astype(np.float32)
            mask = torch.from_numpy(mask_np).unsqueeze(0)

        if self.target_transforms is not None:
            mask = self.target_transforms(mask)

        input_ids, attention_mask = self._tokenize_sentence(sample.sentence)
        # Flatten any stray dimensions, align lengths, then pad/truncate to max_tokens for safety
        input_ids, attention_mask = _align_token_tensors(input_ids, attention_mask, self.max_tokens)
        input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
        attention_mask = attention_mask.unsqueeze(0)
        mask = (mask > 0).float()

        # Token-level masks aligned with attention length to avoid shape mismatches downstream.
        target_masks = attention_mask.clone()
        position_masks = attention_mask.clone()

        # CrossEntropy expects class indices (Long) shaped [H, W]
        target = (mask.squeeze(0) > 0).long()

        return image, target, input_ids, attention_mask, target_masks, position_masks, sample.sentence
