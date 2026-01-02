import os, sys, shutil, glob, json, pickle, warnings, re, xml.etree.ElementTree as ET
import torch.utils.data as data
import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2
from transformers import BertTokenizerFast as BertTokenizer
try:
    from transformers import CLIPTokenizerFast as _CLIPTokenizerFast
except Exception:  # pragma: no cover - defensive for older transformers
    _CLIPTokenizerFast = None
try:
    from transformers import CLIPTokenizer as _CLIPTokenizer
except Exception:  # pragma: no cover
    _CLIPTokenizer = None
try:  # transformers < 4.12 may lack CLIP classes entirely; keep a generic fallback
    from transformers import AutoTokenizer as _AutoTokenizer
except Exception:  # pragma: no cover
    _AutoTokenizer = None
try:  # GPT2 tokenizer can read CLIP's vocab/merges when CLIP classes are missing
    from transformers import GPT2TokenizerFast as _GPT2TokenizerFast
except Exception:  # pragma: no cover
    _GPT2TokenizerFast = None
try:
    from transformers import GPT2Tokenizer as _GPT2Tokenizer
except Exception:  # pragma: no cover
    _GPT2Tokenizer = None

# ---- optional COCO rle ----
try:
    from pycocotools import mask as maskUtils
    HAS_PYCOCO = True
except Exception:
    maskUtils = None
    HAS_PYCOCO = False
    warnings.warn("[newdataset_refer_bert] pycocotools not found. Falling back to bbox when needed.")

# ---------------- debug helpers ---------------- #
def _env_on(name, default=True):
    v = os.getenv(name, "")
    if v == "":  # not set
        return default
    return v not in ("0", "false", "False", "no", "NO")

def _dbg_enabled_for(kind: str):
    # Default: debug ON for vddris to diagnose splits; can disable via env
    if kind == "vddris":
        return _env_on("RRSIS_DEBUG", default=True)
    return _env_on("RRSIS_DEBUG", default=False)

def _dprint(ok: bool, *a, **k):
    if ok:
        print(*a, **k)

def _preview(paths, n=5):
    try:
        arr = sorted(list(paths)) if isinstance(paths, (list, tuple, set)) else []
        return arr[:n]
    except Exception:
        return []

# ---------------- small helpers ---------------- #
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _stem(p): return os.path.splitext(os.path.basename(str(p)))[0]

def _walk_up(start, max_hops=6):
    cur = os.path.abspath(start)
    for _ in range(max_hops):
        yield cur
        parent = os.path.dirname(cur)
        if parent == cur: break
        cur = parent

def _find_up_file(start, relpath):
    for anc in _walk_up(start):
        f = os.path.join(anc, relpath)
        if os.path.isfile(f): return f
    return None

def _find_up_dir(start, relpath):
    for anc in _walk_up(start):
        d = os.path.join(anc, relpath)
        if os.path.isdir(d): return d
    return None

def _has_vdd_vdd_layout(root):
    return any(os.path.isdir(os.path.join(root, s, "src")) for s in ("train", "val", "test")) \
           or os.path.isdir(os.path.join(root, "metadata"))

# ---------------- root detection ---------------- #
def _resolve_data_root(args):
    for k in ("data_root", "dataset_root"):
        if hasattr(args, k) and getattr(args, k):
            return os.path.abspath(getattr(args, k))
    env = os.getenv("RRSIS_DATA_ROOT")
    if env: return os.path.abspath(env)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "refer", "LAVT-RISRS")

def _looks_like_risbench(root: str) -> bool:
    """Strict check so RISBench under a parent with other datasets doesn't get misclassified."""
    must_dirs = [os.path.join(root, "img_rgb"), os.path.join(root, "mask")]
    must_files = [os.path.join(root, f"output_phrase_{s}.txt") for s in ("train", "val", "test")]
    return all(os.path.isdir(d) for d in must_dirs) and all(os.path.isfile(f) for f in must_files)

def _detect_dataset_kind(data_root: str) -> str:
    """
    Dataset-kind detection is ordered to avoid false positives when multiple datasets
    live under the same parent. We first check for a *self-contained* RISBench root,
    then VDD_RIS, then RRSIS-D, then RefSegRS.
    """
    root = os.path.abspath(data_root)
    base = os.path.basename(root).lower()

    # 1) RISBench (self-contained)
    if _looks_like_risbench(root) or "risbench" in base:
        return "risbench"

    # 2) VDD_RIS (accept various layouts found nearby)
    if _find_up_file(root, os.path.join("vdd_ris", "instances.json")) \
       or _find_up_dir(root, os.path.join("images", "vdd_ris")) \
       or _find_up_dir(root, os.path.join("VDD", "VDD")) \
       or _has_vdd_vdd_layout(root):
        return "vddris"

    # 3) RRSIS-D
    if base == "data" or os.path.exists(os.path.join(root, "rrsisd")) \
       or os.path.exists(os.path.join(root, "images", "rrsisd", "JPEGImages")):
        return "rrsisd"

    # 4) RefSegRS (LAVT-RISRS/new_dataset2)
    if "lavt-risrs" in base or os.path.exists(os.path.join(root, "new_dataset2")) \
       or os.path.exists(os.path.join(root, "LAVT-RISRS")):
        return "refsegrs"

    # Fallback
    return "refsegrs"

# ---------------- RefSegRS (unchanged) ---------------- #
def build_rsris_batches(data_root, split):
    im_dir = os.path.join(data_root, "new_dataset2", "images2")
    seg_dir = os.path.join(data_root, "new_dataset2", "masks2")
    setfile = {"train":"output_phrase_train.txt","val":"output_phrase_val3.txt","test":"output_phrase_test3.txt"}.get(split)
    if setfile is None: raise ValueError(f"Unknown split: {split}")
    split_file = os.path.join(data_root, "new_dataset2", setfile)
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}\n(resolved data_root={data_root})")
    imgs, labels, sents = [], [], []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if not parts: continue
            stem = parts[0]; sent = " ".join(parts[1:])
            imgs.append(os.path.join(im_dir, stem + ".tif"))
            labels.append(os.path.join(seg_dir, stem + ".tif"))
            sents.append(sent)
    print(f"Dataset Loaded (RefSegRS). split={split} | #samples={len(imgs)}")
    return imgs, labels, sents

# ---------------- COCO helpers ---------------- #
def _img_path_from_filename(images_base_dir: str, file_name: str) -> str:
    p1 = os.path.join(images_base_dir, file_name)
    if os.path.isfile(p1): return p1
    if os.path.isabs(file_name) and os.path.isfile(file_name): return file_name
    p2 = os.path.join(images_base_dir, os.path.basename(file_name))
    return p2 if os.path.isfile(p2) else p1

def _polygon_to_mask(size_wh, segmentation):
    W, H = size_wh; mask = Image.new("L", (W, H), 0); draw = ImageDraw.Draw(mask)
    for poly in segmentation or []:
        if not poly: continue
        pts = list(zip(poly[0::2], poly[1::2])); draw.polygon(pts, outline=255, fill=255)
    return np.array(mask, dtype=np.uint8)

def _bbox_to_mask(size_wh, bbox):
    W, H = size_wh; mask = Image.new("L", (W, H), 0); draw = ImageDraw.Draw(mask)
    x, y, w, h = bbox or [0, 0, 0, 0]; draw.rectangle([x, y, x + w, y + h], outline=255, fill=255)
    return np.array(mask, dtype=np.uint8)

def _decode_rle_to_mask(seg, W, H):
    if not HAS_PYCOCO: return None
    if isinstance(seg, list):
        if len(seg) == 0: return None
        any_uncompressed = any(isinstance(p.get("counts", None), (list, tuple)) for p in seg if isinstance(p, dict))
        m = maskUtils.decode(maskUtils.frPyObjects(seg, H, W) if any_uncompressed else seg)
        if m.ndim == 3: m = np.any(m, axis=2)
        return (m.astype(np.uint8) * 255)
    if isinstance(seg, dict) and "counts" in seg:
        m = maskUtils.decode(maskUtils.frPyObjects(seg, H, W) if isinstance(seg["counts"], (list, tuple)) else seg)
        if m.ndim == 3: m = m[..., 0]
        return (m.astype(np.uint8) * 255)
    return None

def _ann_to_mask_unified(ann, W, H):
    seg = ann.get("segmentation"); mask_u8 = None
    if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
        mask_u8 = _polygon_to_mask((W, H), seg)
    else:
        mask_u8 = _decode_rle_to_mask(seg, W, H)
    if mask_u8 is None or (isinstance(mask_u8, np.ndarray) and mask_u8.max() == 0):
        mask_u8 = _bbox_to_mask((W, H), ann.get("bbox", [0, 0, 0, 0]))
    if mask_u8.shape != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask_u8 > 0).astype(np.uint8) * 255

# ---------- sentence helpers ----------
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

def _text_from_sent_obj(s):
    if isinstance(s, dict): return s.get("raw") or s.get("sent") or ""
    return str(s)

def _prefer_english_sentence(sent_list):
    if not sent_list: return ""
    texts = [_text_from_sent_obj(s) for s in sent_list]
    for t in texts:
        if _ASCII_LETTER_RE.search(t or ""): return t
    for t in texts:
        if (t or "").strip(): return t
    return ""

def _ref_sentence_text(ref): return _prefer_english_sentence(ref.get("sentences", []))

def _split_ok(ref_split: str, desired: str) -> bool:
    rs = (ref_split or "").lower(); d = desired.lower()
    if d == "train": return rs in {"train", "trainval", "trn", "training"}
    if d == "val":   return rs in {"val", "valid", "validation", "minival", "valiadation"}
    if d == "test":  return rs in {"test", "testa", "testb", "testing", "testa&b"}
    return rs == d

def _load_pickle(path):
    with open(path, "rb") as f:
        try: return pickle.load(f)
        except Exception:
            f.seek(0); return pickle.load(f, encoding="latin1")

# ---------------- RRSIS-D (unchanged) ---------------- #
def build_rrsisd_batches(data_root, split):
    images_dir = os.path.join(data_root, "images", "rrsisd", "JPEGImages")
    refs_p = os.path.join(data_root, "rrsisd", "refs(unc).p")
    inst_json = os.path.join(data_root, "rrsisd", "instances.json")
    if not os.path.isfile(inst_json) or not os.path.isfile(refs_p):
        raise FileNotFoundError("RRSIS-D expected files not found.\n"
                                f"  instances: {inst_json}\n  refs: {refs_p}\n"
                                "Set data_root to parent '.../refer/data'.")
    cache_dir = os.path.join(data_root, "rrsisd", "_cache_masks")
    imgs, labels, sents = _build_from_coco_like(inst_json, refs_p, images_dir, split, cache_dir)
    print(f"Dataset Loaded (RRSIS-D). split={split} | #samples={len(imgs)}")
    return imgs, labels, sents

# ---------------- VDD_RIS builders ---------------- #
def _find_src_image_for_stem(src_dir: str, stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(src_dir, stem + ext)
        if os.path.isfile(p): return p
    # relaxed glob
    for p in sorted(glob.glob(os.path.join(src_dir, stem + "*"))):
        if os.path.isfile(p) and os.path.splitext(p)[1] in IMG_EXTS: return p
    return None

# replace the old _find_mask_for_stem with this more permissive version

def _find_mask_for_stem(anno_dir: str, stem: str):
    """
    Find a mask PNG in anno_dir whose basename contains the stem anywhere.
    Handles names like:
      train_DJI_0008_0_wall.png
      val_DJI_0008_2_roof.PNG
    """
    if not anno_dir or not os.path.isdir(anno_dir):
        return None

    # Fast exact tries
    for ext in (".png", ".PNG"):
        p = os.path.join(anno_dir, stem + ext)
        if os.path.isfile(p):
            return p

    # Common structured patterns
    patterns = [
        f"{stem}_*.png", f"{stem}_*.PNG",        # DJI_0008_*.png
        f"*_{stem}_*.png", f"*_{stem}_*.PNG",    # train_DJI_0008_*.png
        f"*{stem}*.png", f"*{stem}*.PNG",        # any position fallback
    ]
    for pat in patterns:
        cands = sorted(glob.glob(os.path.join(anno_dir, pat)))
        if cands:
            return cands[0]

    return None


def _read_sentence_from_xml(xml_dir_a: str, xml_dir_b: str, stem: str) -> str:
    candidates = []
    for d in (xml_dir_a, xml_dir_b):
        if not d or not os.path.isdir(d): continue
        exact = os.path.join(d, stem + ".xml")
        if os.path.isfile(exact): candidates.append(exact)
        else: candidates += glob.glob(os.path.join(d, stem + "*.xml"))[:1]
    for xmlp in candidates:
        try:
            tree = ET.parse(xmlp); root = tree.getroot(); texts = []
            for tag in ("raw", "sent", "sentence"):
                for el in root.iter(tag):
                    if el.text: texts.append(el.text.strip())
            if not texts:
                for el in root.iter():
                    if el.text and el.text.strip(): texts.append(el.text.strip())
            if texts: return _prefer_english_sentence(texts)
        except Exception: pass
    return ""

def _debug_scan_dir(dir_path, pat="*"):
    if not dir_path or not os.path.isdir(dir_path):
        return 0, []
    files = sorted(glob.glob(os.path.join(dir_path, pat)))
    return len(files), _preview(files, 5)

def _debug_refs_split_counts(refs, split):
    total = 0
    ok = 0
    splits = {}
    for r in refs if isinstance(refs, list) else []:
        s = (r.get("split", "") or "").lower()
        splits[s] = splits.get(s, 0) + 1
        total += 1
        if _split_ok(s, split):
            ok += 1
    return total, ok, splits

def _build_from_coco_like(instances_json, refs_pkl, images_base_dir, split, cache_dir, debug=False):
    with open(instances_json, "r", encoding="utf-8") as f: instances = json.load(f)
    refs = _load_pickle(refs_pkl)
    images_by_id = {im["id"]: im for im in instances.get("images", [])}
    anns_by_id   = {an["id"]: an for an in instances.get("annotations", [])}
    imgs, labels, sents = [], [], []; _ensure_dir(cache_dir)

    # Debug header
    if debug:
        tot_refs, ok_refs, split_hist = _debug_refs_split_counts(refs, split)
        _dprint(True, f"[VDD DEBUG][COCO] base={images_base_dir}")
        _dprint(True, f"[VDD DEBUG][COCO] refs: total={tot_refs}, matching_split={ok_refs}, split_histogram={split_hist}")
        _dprint(True, f"[VDD DEBUG][COCO] images_by_id={len(images_by_id)}, anns_by_id={len(anns_by_id)}")

    miss_img, empty_union, made = 0, 0, 0
    examples = []

    for ref in refs if isinstance(refs, list) else []:
        if not _split_ok(ref.get("split", ""), split): continue
        ann_ids = ref.get("ann_ids") or ref.get("annIds") or ref.get("annids")
        if ann_ids is None:
            single = ref.get("ann_id") or ref.get("annId") or ref.get("annid")
            ann_ids = [single] if single is not None else []
        if not ann_ids: continue

        img_id = ref.get("image_id") or ref.get("imageId") or ref.get("imageid")
        if img_id not in images_by_id: continue

        im = images_by_id[img_id]
        file_name = im.get("file_name") or im.get("filename") or ""
        img_path = _img_path_from_filename(images_base_dir, file_name)
        if not os.path.isfile(img_path):
            alt = os.path.join(images_base_dir, os.path.basename(file_name))
            if os.path.isfile(alt): img_path = alt
            else:
                miss_img += 1
                if len(examples) < 5:
                    examples.append(("MISS_IMG", file_name, images_base_dir))
                continue

        try:
            with Image.open(img_path) as im_pil: W, H = im_pil.size
        except Exception:
            arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if arr is None:
                miss_img += 1
                if len(examples) < 5:
                    examples.append(("MISS_IMG_READ", img_path))
                continue
            H, W = arr.shape[:2]

        try:
            ann_ids_sorted = [int(a) for a in ann_ids if int(a) in anns_by_id]
        except Exception:
            ann_ids_sorted = []
            for a in ann_ids:
                try:
                    aa = int(a)
                    if aa in anns_by_id: ann_ids_sorted.append(aa)
                except Exception: pass
        if not ann_ids_sorted: continue
        ann_ids_sorted.sort()

        base = os.path.splitext(os.path.basename(file_name))[0]
        mask_path = os.path.join(cache_dir, f"{base}_ann{'-'.join(map(str, ann_ids_sorted))}.png")
        if not os.path.isfile(mask_path):
            union = None
            for aid in ann_ids_sorted:
                ann = anns_by_id.get(aid);
                if ann is None: continue
                m = _ann_to_mask_unified(ann, W, H)
                union = (m > 0) if union is None else (union | (m > 0))
            if union is None or not union.any():
                empty_union += 1
                if len(examples) < 5:
                    examples.append(("EMPTY_UNION", file_name, ann_ids_sorted[:3]))
                continue
            Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(mask_path)

        imgs.append(img_path); labels.append(mask_path); sents.append(_ref_sentence_text(ref))
        made += 1
        if debug and len(examples) < 5:
            examples.append(("OK", os.path.basename(img_path), os.path.basename(mask_path)))

    if debug:
        _dprint(True, f"[VDD DEBUG][COCO] built={made}, miss_img={miss_img}, empty_union={empty_union}")
        if examples:
            _dprint(True, f"[VDD DEBUG][COCO] examples: {examples}")

    return imgs, labels, sents

def build_vddris_batches(data_root, split):
    """
    Robust VDD_RIS builder (with DEBUG):
      - Accepts data_root at .../VDD_RIS, .../VDD_RIS/VDD/VDD, or .../VDD_RIS/VDD/VDD/<split>
      - Prefers RGBs from VDD/VDD/<split>/src; never uses VDD/VDD/*/gt (all-black)
      - Masks from images/vdd_ris/annotations
      - Sentences from ann_split(_llama) XML if available; otherwise from refs; otherwise ''
    """
    # Ascend to find companion dirs/files
    vdd_vdd_root = _find_up_dir(data_root, os.path.join("VDD", "VDD"))
    if _has_vdd_vdd_layout(data_root): vdd_vdd_root = data_root
    images_root_dir = _find_up_dir(data_root, os.path.join("images", "vdd_ris"))
    anno_dir        = _find_up_dir(data_root, os.path.join("images", "vdd_ris", "annotations"))
    xml_a           = _find_up_dir(data_root, os.path.join("images", "vdd_ris", "ann_split"))
    xml_b           = _find_up_dir(data_root, os.path.join("images", "vdd_ris", "ann_split_llama"))
    inst_json       = _find_up_file(data_root, os.path.join("vdd_ris", "instances.json"))
    refs_p          = _find_up_file(data_root, os.path.join("vdd_ris", "refs(uow).p")) \
                      or _find_up_file(data_root, os.path.join("vdd_ris", "refs_llama(uow).p"))

    # DEBUG banner
    dbg = True  # vddris debug always on by default; can disable with RRSIS_DEBUG=0
    _dprint(dbg, "\n[VDD DEBUG] ====== PATH DISCOVERY ======")
    _dprint(dbg, f"[VDD DEBUG] data_root         : {data_root}")
    _dprint(dbg, f"[VDD DEBUG] vdd_vdd_root      : {vdd_vdd_root}")
    _dprint(dbg, f"[VDD DEBUG] images_root_dir   : {images_root_dir}")
    _dprint(dbg, f"[VDD DEBUG] anno_dir (png)    : {anno_dir}")
    _dprint(dbg, f"[VDD DEBUG] xml ann_split     : {xml_a}")
    _dprint(dbg, f"[VDD DEBUG] xml ann_split_llama: {xml_b}")
    _dprint(dbg, f"[VDD DEBUG] instances_json    : {inst_json}")
    _dprint(dbg, f"[VDD DEBUG] refs_p            : {refs_p}")

    # Quick directory stats
    if vdd_vdd_root:
        for s in ("train", "val", "test"):
            c, ex = _debug_scan_dir(os.path.join(vdd_vdd_root, s, "src"))
            _dprint(dbg, f"[VDD DEBUG] src {s:<5} count={c} examples={ex}")
    if anno_dir:
        c, ex = _debug_scan_dir(anno_dir, "*.png")
        _dprint(dbg, f"[VDD DEBUG] annotations png count={c} examples={ex}")
    if images_root_dir:
        c, ex = _debug_scan_dir(images_root_dir, "*")
        _dprint(dbg, f"[VDD DEBUG] images_root contents count={c} examples={ex}")

    imgs, labels, sents = [], [], []

    # -------- 1) COCO route (try split src first, then images_root) --------
    if inst_json and refs_p:
        _dprint(dbg, "\n[VDD DEBUG] ====== ROUTE 1: COCO (instances+refs -> rasterize) ======")
        bases_to_try = []
        split_src = os.path.join(vdd_vdd_root, split, "src") if vdd_vdd_root else None
        if split_src and os.path.isdir(split_src): bases_to_try.append(split_src)
        if images_root_dir and os.path.isdir(images_root_dir): bases_to_try.append(images_root_dir)
        if not bases_to_try:
            _dprint(dbg, "[VDD DEBUG][COCO] No candidate base dirs to search for RGBs.")
        cache_dir = _ensure_dir(os.path.join(os.path.dirname(inst_json), "_cache_masks"))
        for base in bases_to_try:
            _dprint(dbg, f"[VDD DEBUG][COCO] Trying base: {base}")
            i2, l2, s2 = _build_from_coco_like(instances_json=inst_json,
                                               refs_pkl=refs_p,
                                               images_base_dir=base,
                                               split=split,
                                               cache_dir=cache_dir,
                                               debug=True)
            if i2:
                _dprint(dbg, f"[VDD DEBUG][COCO] SUCCESS with base={base} -> {len(i2)} samples")
                imgs, labels, sents = i2, l2, s2
                break
            else:
                _dprint(dbg, f"[VDD DEBUG][COCO] No samples with base={base}")

    # -------- 2) refs + instances -> RGB; annotations PNG for masks --------
    if len(imgs) == 0 and inst_json and refs_p and anno_dir:
        _dprint(dbg, "\n[VDD DEBUG] ====== ROUTE 2: refs/instances for RGB + annotations PNG for masks ======")
        with open(inst_json, "r", encoding="utf-8") as f: instances = json.load(f)
        refs = _load_pickle(refs_p)
        images_by_id = {im["id"]: im for im in instances.get("images", [])}

        split_src = os.path.join(vdd_vdd_root, split, "src") if vdd_vdd_root else None
        tried, found_img, found_mask, made = 0, 0, 0, 0
        examples = []
        for ref in refs if isinstance(refs, list) else []:
            if not _split_ok(ref.get("split", ""), split): continue
            tried += 1
            img_id = ref.get("image_id") or ref.get("imageId") or ref.get("imageid")
            im = images_by_id.get(img_id, None)
            if not im: continue
            file_name = im.get("file_name") or im.get("filename") or ""
            # try split src first, then images_root_dir
            cand_img = None
            if split_src and os.path.isdir(split_src):
                gg = _img_path_from_filename(split_src, file_name)
                if os.path.isfile(gg): cand_img = gg
                else:
                    alt = os.path.join(split_src, os.path.basename(file_name))
                    if os.path.isfile(alt): cand_img = alt
            if cand_img is None and images_root_dir and os.path.isdir(images_root_dir):
                gg = _img_path_from_filename(images_root_dir, file_name)
                if os.path.isfile(gg): cand_img = gg
                else:
                    alt = os.path.join(images_root_dir, os.path.basename(file_name))
                    if os.path.isfile(alt): cand_img = alt
            if cand_img is None:
                if len(examples) < 5: examples.append(("MISS_IMG", file_name))
                continue
            found_img += 1

            base = _stem(file_name)
            mask_path = _find_mask_for_stem(anno_dir, base)
            if not mask_path:  # try by image stem after basename fallback
                mask_path = _find_mask_for_stem(anno_dir, _stem(os.path.basename(file_name)))
            if not mask_path:
                if len(examples) < 5: examples.append(("MISS_MASK", base))
                continue
            found_mask += 1

            imgs.append(cand_img)
            labels.append(mask_path)
            sents.append(_ref_sentence_text(ref) or _read_sentence_from_xml(xml_a, xml_b, base) or "")
            made += 1
            if len(examples) < 5: examples.append(("OK", os.path.basename(cand_img), os.path.basename(mask_path)))

        _dprint(dbg, f"[VDD DEBUG][R2] tried_refs={tried}, found_img={found_img}, found_mask={found_mask}, made={made}")
        if examples:
            _dprint(dbg, f"[VDD DEBUG][R2] examples: {examples}")

    # -------- 3) metadata + src + annotations --------
    if len(imgs) == 0 and vdd_vdd_root and anno_dir:
        _dprint(dbg, "\n[VDD DEBUG] ====== ROUTE 3: metadata + src + annotations ======")
        meta_txt = os.path.join(vdd_vdd_root, "metadata", f"{split}.txt")
        src_dir  = os.path.join(vdd_vdd_root, split, "src")
        if os.path.isfile(meta_txt) and os.path.isdir(src_dir):
            with open(meta_txt, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            _dprint(dbg, f"[VDD DEBUG][R3] meta file: {meta_txt}, lines={len(lines)}, head={lines[:5]}")
            tried, made = 0, 0
            examples = []
            for line in lines:
                tried += 1
                base = _stem(line.split()[0])
                img_path  = _find_src_image_for_stem(src_dir, base)
                mask_path = _find_mask_for_stem(anno_dir, base)
                if not img_path or not mask_path:
                    if len(examples) < 5:
                        examples.append(("MISS", base, bool(img_path), bool(mask_path)))
                    continue
                sent = _read_sentence_from_xml(xml_a, xml_b, base)
                imgs.append(img_path); labels.append(mask_path); sents.append(sent)
                made += 1
                if len(examples) < 5:
                    examples.append(("OK", os.path.basename(img_path), os.path.basename(mask_path)))
            _dprint(dbg, f"[VDD DEBUG][R3] tried_meta={tried}, made={made}")
            if examples:
                _dprint(dbg, f"[VDD DEBUG][R3] examples: {examples}")
        else:
            _dprint(dbg, f"[VDD DEBUG][R3] missing meta/src: meta_exists={os.path.isfile(meta_txt)} src_exists={os.path.isdir(src_dir)}")

    # -------- 4) last resort: scan src and pair by stem --------
    if len(imgs) == 0 and vdd_vdd_root and anno_dir:
        _dprint(dbg, "\n[VDD DEBUG] ====== ROUTE 4: scan src + pair by stem ======")
        src_dir = os.path.join(vdd_vdd_root, split, "src")
        if os.path.isdir(src_dir):
            cand_imgs = []
            for ext in IMG_EXTS:
                cand_imgs += sorted(glob.glob(os.path.join(src_dir, "*" + ext)))
            _dprint(dbg, f"[VDD DEBUG][R4] src_dir={src_dir}, candidates={len(cand_imgs)} preview={_preview(cand_imgs)}")
            made = 0; examples = []
            for img_path in cand_imgs:
                base = _stem(img_path)
                mask_path = _find_mask_for_stem(anno_dir, base)
                if not mask_path: continue
                sent = _read_sentence_from_xml(xml_a, xml_b, base)
                imgs.append(img_path); labels.append(mask_path); sents.append(sent); made += 1
                if len(examples) < 5:
                    examples.append(("OK", os.path.basename(img_path), os.path.basename(mask_path)))
            _dprint(dbg, f"[VDD DEBUG][R4] made={made}")
            if examples:
                _dprint(dbg, f"[VDD DEBUG][R4] examples: {examples}")

    print(f"Dataset Loaded (VDD_RIS). split={split} | #samples={len(imgs)}")
    return imgs, labels, sents

# ---------------- RISBench_dataset (NEW) ---------------- #
def build_risbench_batches(data_root, split):
    """
    RISBench loader:
      data_root/
        img_rgb/  (PNG RGB images)
        mask/     (PNG binary masks: black background, white segments)
        output_phrase_{train,val,test}.txt  (lines: '<filename.png> <free-form text>')
    """
    dbg = _dbg_enabled_for("risbench")
    im_dir = os.path.join(data_root, "img_rgb")
    seg_dir = os.path.join(data_root, "mask")
    setfile = {"train": "output_phrase_train.txt",
               "val":   "output_phrase_val.txt",
               "test":  "output_phrase_test.txt"}.get(split)
    split_file = os.path.join(data_root, setfile or "")

    # Quick sanity
    if dbg:
        _dprint(True, "\n[RISBench DEBUG] ====== PATHS ======")
        _dprint(True, f"[RISBench DEBUG] data_root : {data_root}")
        _dprint(True, f"[RISBench DEBUG] img_rgb   : {im_dir} (exists={os.path.isdir(im_dir)})")
        _dprint(True, f"[RISBench DEBUG] mask      : {seg_dir} (exists={os.path.isdir(seg_dir)})")
        _dprint(True, f"[RISBench DEBUG] split_file: {split_file} (exists={os.path.isfile(split_file)})")

        if os.path.isdir(im_dir):
            c, ex = _debug_scan_dir(im_dir, "*.png")
            _dprint(True, f"[RISBench DEBUG] img_rgb png count={c} examples={ex}")
        if os.path.isdir(seg_dir):
            c, ex = _debug_scan_dir(seg_dir, "*.png")
            _dprint(True, f"[RISBench DEBUG] mask png count={c} examples={ex}")

    if setfile is None:
        raise ValueError(f"Unknown split: {split}")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(
            f"[RISBench] Split file not found: {split_file}\n"
            f"(resolved data_root={data_root})"
        )
    if not os.path.isdir(im_dir) or not os.path.isdir(seg_dir):
        raise FileNotFoundError(
            f"[RISBench] Required directories missing.\n"
            f"  img_rgb: {im_dir}\n"
            f"  mask   : {seg_dir}"
        )

    imgs, labels, sents = [], [], []
    skipped = 0
    examples = []

    with open(split_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            # Split on first whitespace token being the filename, rest is sentence
            parts = line.split()
            if len(parts) < 2:
                continue
            image_id = parts[0]
            sent = " ".join(parts[1:])

            img_path = os.path.join(im_dir, image_id)
            mask_path = os.path.join(seg_dir, image_id)

            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                skipped += 1
                if len(examples) < 5:
                    examples.append(("MISS", image_id, os.path.isfile(img_path), os.path.isfile(mask_path)))
                continue

            imgs.append(img_path)
            labels.append(mask_path)
            sents.append(sent)
            if len(examples) < 5:
                examples.append(("OK", image_id))

    if dbg:
        _dprint(True, f"[RISBench DEBUG] built={len(imgs)} skipped_missing={skipped}")
        if examples:
            _dprint(True, f"[RISBench DEBUG] examples: {examples}")

    print(f"Dataset Loaded (RISBench). split={split} | #samples={len(imgs)}")
    return imgs, labels, sents

# ---------------- tokenizer loaders (unchanged) ---------------- #
_EXPECTED = {"tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"}
_EXPECTED_CLIP = {"tokenizer.json", "vocab.json", "merges.txt"}

def _looks_like_tokenizer_dir(path: str) -> bool:
    try:
        if not os.path.isdir(path): return False
        entries = set(os.listdir(path))
        return any(f in entries for f in _EXPECTED)
    except Exception: return False


def _looks_like_clip_tokenizer_dir(path: str) -> bool:
    try:
        if not os.path.isdir(path):
            return False
        entries = set(os.listdir(path))
        return all(f in entries for f in _EXPECTED_CLIP)
    except Exception:
        return False


def _load_clip_tokenizer(tok_path: str, cache_dir=None):
    """
    Load a CLIP tokenizer offline using whichever class the installed transformers provides.

    On old transformers (e.g. with Python 3.7) there may be no CLIPTokenizer* and
    AutoTokenizer may fail on model_type='clip'. In that case we fall back to a
    GPT-2 BPE tokenizer built directly from vocab.json + merges.txt.
    """
    # 1) Native CLIP tokenizers (if available in this transformers build)
    if _CLIPTokenizerFast is not None:
        return _CLIPTokenizerFast.from_pretrained(
            tok_path, cache_dir=cache_dir, local_files_only=True
        )
    if _CLIPTokenizer is not None:
        return _CLIPTokenizer.from_pretrained(
            tok_path, cache_dir=cache_dir, local_files_only=True
        )

    last_err = None

    # 2) Try AutoTokenizer, but don't die if it can't handle model_type='clip'
    if _AutoTokenizer is not None:
        try:
            return _AutoTokenizer.from_pretrained(
                tok_path, cache_dir=cache_dir, local_files_only=True
            )
        except Exception as e:
            print(
                "[Tokenizer] AutoTokenizer could not load CLIP (likely old transformers). "
                "Falling back to GPT2 BPE loader."
            )
            last_err = e

    # 3) GPT-2 BPE fallback: construct directly from vocab/merges to avoid config issues
    vocab_file = os.path.join(tok_path, "vocab.json")
    merges_file = os.path.join(tok_path, "merges.txt")

    for tok_cls in (_GPT2TokenizerFast, _GPT2Tokenizer):
        if tok_cls is None:
            continue
        try:
            # Direct constructor avoids needing a 'clip' config at all
            tok = tok_cls(vocab_file=vocab_file, merges_file=merges_file)
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
            print("[Tokenizer] Loaded CLIP vocab/merges using GPT2 tokenizer fallback.")
            return tok
        except Exception as e:
            last_err = e
            continue

    # If we get here, all strategies failed
    raise RuntimeError(
        "[Tokenizer] transformers installation does not provide CLIP tokenizer classes, "
        "and GPT2 vocab/merges fallback failed."
    ) from last_err


def _maybe_delete_if_corrupt(path: str) -> None:
    try:
        if os.path.isdir(path):
            entries = set(os.listdir(path))
            if len(entries) == 0 or not any(f in entries for f in _EXPECTED):
                shutil.rmtree(path); print(f"[Tokenizer] Removed corrupt/empty local dir: {path}")
    except Exception as e:
        print(f"[Tokenizer] Could not inspect/remove {path}: {e}")

def _find_in_hf_cache(model_id: str = "bert-base-uncased"):
    candidates = []
    for env_key in ("HF_HOME","HUGGINGFACE_HUB_CACHE","TRANSFORMERS_CACHE"):
        root = os.getenv(env_key)
        if root: candidates.append(root)
    home = os.path.expanduser("~")
    candidates += [os.path.join(home, ".cache", "huggingface", "hub"),
                   os.path.join(home, ".cache", "huggingface", "transformers")]
    for root in candidates:
        for d in glob.glob(os.path.join(root, f"models--{model_id}", "snapshots", "*")):
            if _looks_like_tokenizer_dir(d): return d
        for d in glob.glob(os.path.join(root, model_id, "*")):
            if _looks_like_tokenizer_dir(d): return d
    return None

def _ddp_info():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return True, dist.get_rank()
    except Exception: pass
    return False, 0

def _safe_load_tokenizer(name_or_path: str, args=None):
    cache_dir = getattr(args, "hf_cache_dir", None)
    encoder = getattr(args, "language_encoder", "bert")
    is_dist, rank = _ddp_info()

    if encoder == "clip":
        tok_path = name_or_path
        if not _looks_like_clip_tokenizer_dir(tok_path):
            alt = getattr(args, "clip_model_path", None)
            if alt and _looks_like_clip_tokenizer_dir(alt):
                tok_path = alt
        if not _looks_like_clip_tokenizer_dir(tok_path):
            raise RuntimeError(
                "[Tokenizer] CLIP tokenizer not found locally. Provide --clip_model_path pointing to the offline snapshot."
            )
        print(f"[Tokenizer] Using offline CLIP tokenizer: {tok_path}")
        return _load_clip_tokenizer(tok_path, cache_dir=cache_dir)

    if os.path.isdir(name_or_path) and _looks_like_tokenizer_dir(name_or_path):
        print(f"[Tokenizer] Using local tokenizer dir: {name_or_path}")
        return BertTokenizer.from_pretrained(name_or_path, cache_dir=cache_dir, local_files_only=True)
    shadow = os.path.abspath(os.path.join(os.getcwd(), name_or_path))
    if os.path.isdir(shadow): _maybe_delete_if_corrupt(shadow)
    def try_normal_download():
        if not is_dist or rank == 0:
            return BertTokenizer.from_pretrained(name_or_path, cache_dir=cache_dir)
        else:
            import torch.distributed as dist; dist.barrier()
            return BertTokenizer.from_pretrained(name_or_path, cache_dir=cache_dir)
    def try_local_fallbacks():
        bt = getattr(args, "bert_tokenizer", None)
        if bt and os.path.isdir(bt) and _looks_like_tokenizer_dir(bt):
            print(f"[Tokenizer] Loaded from local --bert_tokenizer: {bt}")
            return BertTokenizer.from_pretrained(bt, cache_dir=cache_dir, local_files_only=True)
        ck = getattr(args, "ck_bert", None)
        if ck and os.path.isdir(ck) and _looks_like_tokenizer_dir(ck):
            print(f"[Tokenizer] Loaded from local --ck_bert: {ck}")
            return BertTokenizer.from_pretrained(ck, cache_dir=cache_dir, local_files_only=True)
        snap = _find_in_hf_cache("bert-base-uncased")
        if snap:
            print(f"[Tokenizer] Loaded from HF cache snapshot: {snap}")
            return BertTokenizer.from_pretrained(snap, cache_dir=cache_dir, local_files_only=True)
        raise RuntimeError("Could not load a tokenizer offline.\n"
                           "Provide --bert_tokenizer /path/to/bert-base-uncased or "
                           "download a snapshot via huggingface-cli.")
    try: return try_normal_download()
    except Exception as e:
        print(f"[Tokenizer] Online load failed ({e}). Falling back to local-only resolution...")
        return try_local_fallbacks()

# ---------------- Dataset class (unchanged API) ---------------- #
class ReferDataset(data.Dataset):
    def __init__(self, args, image_transforms=None, target_transforms=None, split="train", eval_mode=False):
        self.classes = []; self.image_transforms = image_transforms
        self.target_transform = target_transforms; self.split = split
        self.max_tokens = 20; self.eval_mode = eval_mode

        self.data_root = _resolve_data_root(args)
        kind = _detect_dataset_kind(self.data_root)

        if kind == "refsegrs":
            self.imgs, self.labels, self.sentences = build_rsris_batches(self.data_root, self.split)
        elif kind == "rrsisd":
            self.imgs, self.labels, self.sentences = build_rrsisd_batches(self.data_root, self.split)
        elif kind == "vddris":
            self.imgs, self.labels, self.sentences = build_vddris_batches(self.data_root, self.split)
        elif kind == "risbench":
            self.imgs, self.labels, self.sentences = build_risbench_batches(self.data_root, self.split)
        else:
            raise RuntimeError(f"Unknown dataset kind resolved from root: {self.data_root}")

        lang_enc = getattr(args, "language_encoder", "bert")
        if lang_enc == "clip":
            tok_id = getattr(args, "clip_model_path", "")
        else:
            tok_id = getattr(args, "bert_tokenizer", "bert-base-uncased")

        self.tokenizer = _safe_load_tokenizer(tok_id, args=args)

        if lang_enc != "clip":
            ascii_english = sum(1 for s in self.sentences if _ASCII_LETTER_RE.search(s or ""))
            if ascii_english == 0 and "uncased" in str(tok_id):
                print("[WARN] No English sentences detected but tokenizer is 'bert-base-uncased'. "
                      "Consider using a multilingual or Chinese BERT, e.g. 'bert-base-chinese'.")

        self.input_ids, self.attention_masks = [], []
        for s in self.sentences:
            att = [0] * self.max_tokens; ids = [0] * self.max_tokens
            toks = self.tokenizer.encode(text=s, add_special_tokens=True)[: self.max_tokens]
            ids[: len(toks)] = toks; att[: len(toks)] = [1] * len(toks)
            self.input_ids.append([torch.tensor(ids).unsqueeze(0)])
            self.attention_masks.append([torch.tensor(att).unsqueeze(0)])

        if len(self.imgs) == 0 and kind == "vddris":
            # Extra diagnostics if nothing found for VDD_RIS
            diag = {
                "data_root": self.data_root, "kind": kind, "split": self.split,
                "vdd_vdd_found": bool(_find_up_dir(self.data_root, os.path.join("VDD","VDD")) or _has_vdd_vdd_layout(self.data_root)),
                "instances_json": bool(_find_up_file(self.data_root, os.path.join("vdd_ris","instances.json"))),
                "refs_uow": bool(_find_up_file(self.data_root, os.path.join("vdd_ris","refs(uow).p"))),
                "refs_llama": bool(_find_up_file(self.data_root, os.path.join("vdd_ris","refs_llama(uow).p"))),
                "images_vdd_ris_dir": bool(_find_up_dir(self.data_root, os.path.join("images","vdd_ris"))),
                "ann_png_dir": bool(_find_up_dir(self.data_root, os.path.join("images","vdd_ris","annotations"))),
                "meta_txt": bool(_find_up_file(self.data_root, os.path.join("VDD","VDD","metadata", f"{self.split}.txt"))),
                "src_dir": bool(_find_up_dir(self.data_root, os.path.join("VDD","VDD", self.split, "src")) or os.path.isdir(os.path.join(self.data_root, self.split, "src"))),
            }
            raise RuntimeError(f"[VDD_RIS] No samples found for split='{self.split}'. Diagnostics: {diag}")

    def get_classes(self): return self.classes
    def __len__(self): return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]; mask_path = self.labels[index]
        img = Image.open(img_path).convert("RGB")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            with Image.open(mask_path) as _m: mask = np.array(_m.convert("L"))
        ref = (np.array(mask) > 50)
        annot = np.zeros(ref.shape, dtype=np.uint8); annot[ref] = 1
        annot = Image.fromarray(annot, mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)
        else:
            target = annot

        if self.eval_mode:
            emb = [e.unsqueeze(-1) for e in self.input_ids[index]]
            att = [a.unsqueeze(-1) for a in self.attention_masks[index]]
            tensor_embeddings = torch.cat(emb, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            tensor_embeddings = self.input_ids[index][0]
            attention_mask = self.attention_masks[index][0]
        return img, target, tensor_embeddings, attention_mask
