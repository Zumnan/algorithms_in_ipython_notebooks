import os
import random
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from nltk.tokenize import word_tokenize

import transformers
from transformers import BertTokenizer  # your current dependency


# ------------------------- NLTK offline (keep as-is) -------------------------
nltk_data_path = "/10T/students/doctor/2025/zum/anaconda3/share/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_path
# -----------------------------------------------------------------------------


_IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]


def _default_if_none(value: Optional[str], fallback: str) -> str:
    return fallback if value is None else value


def _ensure_exists(path: str, kind: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"RISBench {kind} not found at: {path}")


def _parse_phrase_line(line: str) -> Tuple[str, str]:
    """
    Supports:
      - "img.png<TAB>phrase..."
      - "img.png,phrase..."
      - "img.png phrase..."
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    if "\t" in line:
        image_part, phrase = line.split("\t", 1)
    elif "," in line:
        image_part, phrase = line.split(",", 1)
    else:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Unable to parse phrase line: {line}")
        image_part, phrase = parts

    return image_part.strip(), phrase.strip()


def _sanitize_prefix(s: str, max_len: int = 180) -> str:
    s = s.replace(os.sep, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z._-]+", "", s)
    return s[:max_len]


def _resolve_by_stem(folder: str, filename_in_phrase: str, exts=_IMG_EXTS) -> str:
    """
    Resolve a real file in `folder` using the stem from `filename_in_phrase`.
    Allows RGB and mask to have different extensions.
    """
    base = os.path.basename(filename_in_phrase.strip())
    stem, ext = os.path.splitext(base)

    # 1) Exact match if phrase already has an extension
    if ext:
        p = os.path.join(folder, base)
        if os.path.exists(p):
            return p

    # 2) Try common extensions
    for e in exts:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p

    # 3) Glob fallback
    g = glob.glob(os.path.join(folder, stem + ".*"))
    if g:
        return g[0]

    return ""


def add_random_boxes(img: Image.Image, min_num: int = 20, max_num: int = 60, size: int = 32) -> Image.Image:
    img_np = np.asarray(img).copy()
    H, W = img_np.shape[0], img_np.shape[1]
    h, w = size, size
    if H < h or W < w:
        return img

    num = random.randint(min_num, max_num)
    for _ in range(num):
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        img_np[y:y + h, x:x + w] = 0
    return Image.fromarray(img_np.astype("uint8"), "RGB")


def _safe_word_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _find_sublist(tokens: List[str], sub: List[str]) -> int:
    if not sub or len(sub) > len(tokens):
        return -1
    for i in range(len(tokens) - len(sub) + 1):
        if tokens[i:i + len(sub)] == sub:
            return i
    return -1


@dataclass
class _RISBenchSample:
    image_path: str
    mask_path: str
    sentence: str
    name_in_phrase: str  # used for overlap/leak checks


def _read_phrase_file(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                img_name, sent = _parse_phrase_line(line)
            except ValueError:
                continue
            if img_name and sent:
                pairs.append((img_name, sent))
    return pairs


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


class ReferDataset(data.Dataset):
    def __init__(
        self,
        args,
        image_transforms=None,
        target_transforms=None,
        split: str = "train",
        eval_mode: bool = False,
        exclude_val_from_train: bool = True,  # IMPORTANT: prevents leakage if overlaps exist
    ):
        self.classes: List[str] = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.eval_mode = eval_mode

        root = getattr(args, "risbench_data_root", None) or getattr(args, "refer_data_root", None)
        if root is None:
            raise ValueError("RISBench root not resolved; pass --refer_data_root or --risbench_data_root")

        img_dir = _default_if_none(getattr(args, "risbench_img_dir", None), os.path.join(root, "img_rgb"))
        mask_dir = _default_if_none(getattr(args, "risbench_mask_dir", None), os.path.join(root, "mask"))

        phrase_train = _default_if_none(getattr(args, "risbench_phrase_train", None), os.path.join(root, "output_phrase_train.txt"))
        phrase_val = _default_if_none(getattr(args, "risbench_phrase_val", None), os.path.join(root, "output_phrase_val.txt"))
        phrase_test = _default_if_none(getattr(args, "risbench_phrase_test", None), os.path.join(root, "output_phrase_test.txt"))

        _ensure_exists(img_dir, "image directory")
        _ensure_exists(mask_dir, "mask directory")

        # Phrase files: val is optional in some downloads, but you DO have it.
        if split == "train":
            _ensure_exists(phrase_train, "train phrase file")
            phrase_file = phrase_train
        elif split == "val":
            _ensure_exists(phrase_val, "val phrase file")
            phrase_file = phrase_val
        elif split == "test":
            _ensure_exists(phrase_test, "test phrase file")
            phrase_file = phrase_test
        else:
            raise ValueError(f"Unsupported split '{split}' for RISBench")

        # Tokenizer (offline-local path is fine)
        self.tokenizer = _load_tokenizer(args)

        # NLTK chunker (your PP grammar)
        grammar = r"""
        PP: {<IN><DT>?<JJ.*>?<NN>}
            {<IN><DT>?<JJ.*>?<JJ>}
            {<IN><DT>?<JJ.*><VBD>}
        """
        self._chunker = nltk.RegexpParser(grammar)

        # Load phrase pairs for requested split
        split_pairs = _read_phrase_file(phrase_file)

        # If training, optionally exclude anything listed in val (in case of overlap)
        val_names: Set[str] = set()
        if split == "train" and exclude_val_from_train and os.path.exists(phrase_val):
            val_pairs = _read_phrase_file(phrase_val)
            val_names = set(os.path.basename(n) for n, _ in val_pairs)

        # Build samples and compute max token length
        self.samples: List[_RISBenchSample] = []
        max_sentence_tokens = 0

        for name_in_phrase, sentence in split_pairs:
            base = os.path.basename(name_in_phrase)

            # Leakage guard: if someone accidentally includes val entries in train list
            if split == "train" and val_names and base in val_names:
                continue

            image_path = _resolve_by_stem(img_dir, base)
            mask_path = _resolve_by_stem(mask_dir, base)

            if not image_path or not mask_path:
                continue

            token_len = len(self.tokenizer.encode(text=sentence, add_special_tokens=True))
            if token_len > max_sentence_tokens:
                max_sentence_tokens = token_len

            self.samples.append(_RISBenchSample(
                image_path=image_path,
                mask_path=mask_path,
                sentence=sentence,
                name_in_phrase=base,
            ))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No RISBench samples loaded for split='{split}'. Check:\n"
                f"  phrase_file={phrase_file}\n  img_dir={img_dir}\n  mask_dir={mask_dir}\n"
            )

        # Token padding length
        self.max_tokens = min(max(22, max_sentence_tokens), 64)

        # Occlusion augmentation (20% of samples, train only)
        if split == "train":
            num_images_to_mask = int(len(self.samples) * 0.2)
            self.images_to_mask = set(random.sample(range(len(self.samples)), num_images_to_mask))
        else:
            self.images_to_mask = set()

        # Precompute lightweight arrays
        self._input_ids: List[List[int]] = []
        self._attention_masks: List[List[int]] = []
        self._target_masks: List[List[int]] = []
        self._position_masks: List[List[int]] = []
        self.pp_phrase: List[List[str]] = []

        for sample in self.samples:
            sentence_raw = sample.sentence

            input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)[: self.max_tokens]
            attn = [1] * len(input_ids)

            padded_ids = [0] * self.max_tokens
            padded_attn = [0] * self.max_tokens
            padded_ids[:len(input_ids)] = input_ids
            padded_attn[:len(attn)] = attn

            target_mask = padded_attn.copy()

            # Position mask via PP chunking
            position_mask = [0] * self.max_tokens
            tokenized_sentence = _safe_word_tokenize(sentence_raw)

            try:
                tagged = nltk.pos_tag(tokenized_sentence)
                tree = self._chunker.parse(tagged)
                pp_phrases = [
                    " ".join(word for word, _pos in subtree.leaves())
                    for subtree in tree.subtrees()
                    if subtree.label() == "PP"
                ]
            except LookupError:
                pp_phrases = []

            new_pp_phrase = [p for p in pp_phrases if not re.search(r"\bof\b", p)]

            if new_pp_phrase:
                for pp in new_pp_phrase:
                    pp_tokens = _safe_word_tokenize(pp)
                    start = _find_sublist(tokenized_sentence, pp_tokens)
                    if start >= 0:
                        s = start + 1  # +1 for [CLS]
                        e = min(start + 1 + len(pp_tokens), self.max_tokens)
                        for k in range(s, e):
                            position_mask[k] = 1

            if sum(position_mask) == 0:
                position_mask = padded_attn.copy()

            self._input_ids.append(padded_ids)
            self._attention_masks.append(padded_attn)
            self._target_masks.append(target_mask)
            self._position_masks.append(position_mask)
            self.pp_phrase.append(new_pp_phrase)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        img = Image.open(sample.image_path).convert("RGB")
        mask_img = Image.open(sample.mask_path).convert("L")

        mask_np = np.array(mask_img)
        annot = np.zeros(mask_np.shape, dtype=np.uint8)
        annot[mask_np > 0] = 1
        annot_pil = Image.fromarray(annot, mode="P")

        if self.split == "train" and index in self.images_to_mask:
            img = add_random_boxes(img)

        save_prefix = _sanitize_prefix(f"{os.path.splitext(os.path.basename(sample.image_path))[0]}_{sample.sentence}")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot_pil)
        else:
            target = annot_pil

        input_ids = torch.tensor(self._input_ids[index], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(self._attention_masks[index], dtype=torch.long).unsqueeze(0)
        target_mask = torch.tensor(self._target_masks[index], dtype=torch.long).unsqueeze(0)
        position_mask = torch.tensor(self._position_masks[index], dtype=torch.long).unsqueeze(0)

        return img, target, input_ids, attention_mask, target_mask, position_mask, save_prefix


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    for sp in ["train", "val", "test"]:
        ds = ReferDataset(args, split=sp, exclude_val_from_train=True)
        print(f"[{sp}] Loaded {len(ds)} samples; max_tokens={ds.max_tokens}")
