"""Argument parser for the LAVT/Nova segmentation stack.

This module mirrors the import style used throughout the repository to avoid
path issues when running from packaged or local checkouts. Defaults point to
common local paths used in the original FIANet experiments while exposing
switches for enhancer, LoRA, and optimizer tweaks.
"""

import os
import argparse
from typing import Optional, List


class _ResolvingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that auto-resolves dataset roots after parsing."""

    def _post_process(self, parsed_args: argparse.Namespace) -> argparse.Namespace:
        parsed_args.refer_data_root = _resolve_data_root(parsed_args)

        # PyCharm-friendly LoRA toggle (no CLI needed):
        #   set env var RRSIS_LORA_RANK=4 (or 8) to enable LoRA from the IDE.
        env_rank = os.environ.get("RRSIS_LORA_RANK", "").strip()
        if parsed_args.lora_rank is None and env_rank:
            try:
                r = int(env_rank)
                parsed_args.lora_rank = r if r > 0 else None
            except Exception:
                pass

        # PyCharm-friendly performance defaults (no CLI needed):
        # - Treat workers=0 as AUTO (choose a sensible value for throughput)
        # - Enable pin memory by default for GPU input pipeline (disable via RRSIS_PIN_MEM=0)
        env_workers = os.environ.get("RRSIS_WORKERS", "").strip()
        if env_workers:
            try:
                parsed_args.workers = max(0, int(env_workers))
            except Exception:
                pass

        if getattr(parsed_args, "workers", 0) <= 0:
            cpu = os.cpu_count() or 8
            # Safer default than 8 for older PyTorch/Python 3.7 multiprocessing stability
            parsed_args.workers = min(4, max(2, cpu // 4))

        env_pin = os.environ.get("RRSIS_PIN_MEM", "1").strip()
        parsed_args.pin_mem = (env_pin != "0")

        return parsed_args

    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        parsed = super().parse_args(args=args, namespace=namespace)
        return self._post_process(parsed)

    def parse_known_args(self, args=None, namespace=None):  # type: ignore[override]
        parsed, extras = super().parse_known_args(args=args, namespace=namespace)
        return self._post_process(parsed), extras


def get_parser() -> argparse.ArgumentParser:
    parser = _ResolvingArgumentParser(
        description='Remote-sensing referring segmentation (LAVT/Nova stack)'
    )
    parser.add_argument(
        '--amsgrad',
        action='store_true',
        help='if true, set amsgrad to True in an Adam or AdamW optimizer.',
    )
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument(
        '--bert_tokenizer',
        default='/10T/students/doctor/2025/zum/models/bert-base-uncased/',
        help='Local path to BERT tokenizer',
    )
    parser.add_argument(
        '--ck_bert',
        default='/10T/students/doctor/2025/zum/models/bert-base-uncased/',
        help='Local path to pre-trained BERT weights',
    )
    parser.add_argument(
        '--hf_cache_dir',
        default='',
        help='Optional HF cache directory for tokenizer/models',
    )
    parser.add_argument(
        '--language_encoder',
        default='bert',
        choices=['bert', 'clip'],
        help='Select language encoder: bert (default) or clip.',
    )
    parser.add_argument(
        '--visual_encoder',
        default='swin',
        choices=['swin', 'clip'],
        help='Select vision encoder: swin backbone or CLIP vision tower (onnx).',
    )
    parser.add_argument(
        '--clip_model_path',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/Clipmodel/',
        help='Local offline snapshot for CLIP (openai/clip-vit-base-patch32).',
    )
    parser.add_argument(
        '--clip_vision_onnx',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/clip-resnet-101-visual.onnx',
        help='Local offline CLIP visual encoder (ONNX).',
    )
    parser.add_argument(
        '--dataset',
        choices=['rrsisd', 'refsegrs', 'risbench', 'nwpu-refer', 'vdd_ris'],
        default='rrsisd',
        help='Choose between the legacy RRSISD dataset, RefSegRS, RISBench, NWPU-refer, and VDD-RIS datasets.',
    )
    parser.add_argument(
        '--rrsisd_data_root',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/data',
        help='RRSISD dataset root directory',
    )
    parser.add_argument(
        '--nwpu_data_root',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/NWPU-refer',
        help='NWPU-refer dataset root directory (fixed paths used for images/annotations).',
    )
    parser.add_argument(
        '--refsegrs_data_root',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/LAVT-RISRS/new_dataset2',
        help='RefSegRS dataset root directory',
    )
    parser.add_argument(
        '--risbench_data_root',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/refer/RISBench_dataset',
        help='RISBench dataset root directory',
    )
    parser.add_argument(
        '--risbench_img_dir',
        default=None,
        help='Optional override for RISBench RGB image directory (defaults to <risbench_data_root>/img_rgb).',
    )
    parser.add_argument(
        '--risbench_mask_dir',
        default=None,
        help='Optional override for RISBench mask directory (defaults to <risbench_data_root>/mask).',
    )
    parser.add_argument(
        '--risbench_phrase_train',
        default=None,
        help='Optional override for RISBench training phrase file (defaults to <risbench_data_root>/output_phrase_train.txt).',
    )
    parser.add_argument(
        '--risbench_phrase_val',
        default=None,
        help='Optional override for RISBench validation phrase file (defaults to <risbench_data_root>/output_phrase_val.txt).',
    )
    parser.add_argument(
        '--risbench_phrase_test',
        default=None,
        help='Optional override for RISBench test phrase file (defaults to <risbench_data_root>/output_phrase_test.txt).',
    )
    parser.add_argument(
        '--vdd_ris_train_src',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/train/src/',
        help='VDD-RIS training RGB directory (jpg).',
    )
    parser.add_argument(
        '--vdd_ris_train_gt',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/train/gt/',
        help='VDD-RIS training mask directory (png).',
    )
    parser.add_argument(
        '--vdd_ris_val_src',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/val/src/',
        help='VDD-RIS validation RGB directory (jpg).',
    )
    parser.add_argument(
        '--vdd_ris_val_gt',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/val/gt/',
        help='VDD-RIS validation mask directory (png).',
    )
    parser.add_argument(
        '--vdd_ris_test_src',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/test/src/',
        help='VDD-RIS test RGB directory (jpg).',
    )
    parser.add_argument(
        '--vdd_ris_test_gt',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/test/gt/',
        help='VDD-RIS test mask directory (png).',
    )
    parser.add_argument(
        '--vdd_ris_train_meta',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/metadata/train.txt',
        help='VDD-RIS training metadata file.',
    )
    parser.add_argument(
        '--vdd_ris_val_meta',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/metadata/val.txt',
        help='VDD-RIS validation metadata file.',
    )
    parser.add_argument(
        '--vdd_ris_test_meta',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD/metadata/test.txt',
        help='VDD-RIS test metadata file.',
    )
    parser.add_argument(
        '--vdd_ris_root',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/VDD_RIS/VDD/VDD',
        help='Root directory for VDD-RIS dataset (contains train/val/test subfolders).',
    )
    parser.add_argument(
        '--refer_data_root',
        default=None,
        help='Override dataset root directory (otherwise selected based on --dataset)',
    )
    parser.add_argument(
        '--ddp_trained_weights',
        action='store_true',
        help='Only needs specified when testing,whether the weights to be loaded are from a DDP-trained model',
    )
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument(
        '--fusion_drop',
        default=0.0,
        type=float,
        help='dropout rate for fusion blocks in GeoFormer',
    )
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument(
        '--lr',
        default=3e-5,  # 3e-5 for rrsisd, 5e-5 for refsegrs 74_62@ epoch 19 when using 5e-5 on rrsisd
        type=float,
        help='the initial learning rate',
    )
    parser.add_argument(
        '--mha',
        default='',
        help=(
            'If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
            'where a, b, c, and d refer to the numbers of heads in stage-1,'
            'stage-2, stage-3, and stage-4 PWAMs'
        ),
    )
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='FIANet', help='name to identify the model')
    parser.add_argument('--num_classes', default=2, type=int, help='number of segmentation classes')
    parser.add_argument(
        '--lang_dim',
        default=768,
        type=int,
        help='language embedding dimension for the decoder',
    )
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='If true, pin memory when using the data loader.',
    )
    parser.add_argument(
        '--pretrained_swin_weights',
        default='/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/pretrained_weights/swin_base_patch4_window12_384_22k.pth',
        help='path to pre-trained Swin backbone weights',
    )
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument(
        '--resume',
        default='',
        help='resume from checkpoint',
    )
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument(
        '--splitBy',
        default='unc',
        help='change to umd or google when the dataset is G-Ref (RefCOCOg)',
    )
    parser.add_argument(
        '--swin_type',
        default='base',
        help='tiny, small, base, or large variants of the Swin Transformer',
    )
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-2,
        type=float,
        metavar='W',
        help='weight decay',
        dest='weight_decay',
    )
    parser.add_argument(
        '--window12',
        action='store_true',
        help=(
            "only needs specified when testing,"
            "when training, window size is inferred from pre-trained weights file name"
            "(containing 'window12'). Initialize Swin with window size 12 instead of the default 7."
        ),
    )
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    parser.add_argument(
        '--enhancer_blocks',
        default=3,  # 1 for RefSegRS, 3 for RRSIS-D
        type=int,
        help='number of NovaEnhancer blocks (replaces legacy TMEM)',
    )
    parser.add_argument(
        '--enhancer_heads',
        default=8,
        type=int,
        help='attention heads used inside NovaEnhancer blocks',
    )
    parser.add_argument(
        '--enhancer_downsample',
        default=4,
        type=int,
        help='pyramid downsample factor used by the spectral gather step',
    )
    parser.add_argument(
        '--enable_cam',
        action='store_true',
        help='enable Grad-CAM style attribution maps during enhancer forward pass',
    )
    parser.add_argument(
        '--lora_rank',
        default=None,
        type=int,
        help='enable LoRA adapters with the given rank (unset to disable)',
    )
    parser.add_argument(
        '--lora_alpha',
        default=32.0,
        type=float,
        help='LoRA scaling factor',
    )
    parser.add_argument(
        '--lora_dropout',
        default=0.1,
        type=float,
        help='LoRA dropout probability',
    )
    parser.add_argument(
        '--lora_full_match',
        action='store_true',
        help='use exact module-name matching when applying LoRA adapters',
    )

    parser.add_argument(
        '--decoder_dim',
        default=None,
        type=int,
        help='override decoder hidden dimension; defaults to backbone embed_dim when unset',
    )

    parser.add_argument(
        '--max_seq_length',
        default=22,
        type=int,
        help='maximum BERT token length for referring expressions',
    )
    return parser


def _resolve_data_root(args: argparse.Namespace) -> str:
    if args.refer_data_root:
        return args.refer_data_root
    if args.dataset == 'refsegrs':
        return args.refsegrs_data_root
    if args.dataset == 'risbench':
        return args.risbench_data_root
    if args.dataset == 'nwpu-refer':
        return args.nwpu_data_root
    if args.dataset == 'vdd_ris':
        return args.vdd_ris_root
    return args.rrsisd_data_root


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = get_parser()
    parsed_args = parser.parse_args(argv)
    parsed_args.refer_data_root = _resolve_data_root(parsed_args)
    return parsed_args


if __name__ == "__main__":
    _ = parse_args()
