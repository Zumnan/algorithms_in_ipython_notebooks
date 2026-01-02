import argparse
import os
import warnings


def _abspath_or_none(p):
    return os.path.abspath(p) if p else p


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')

    # ---------------- Core training ----------------
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad=True in Adam/AdamW.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.00005, type=float,
                        help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W',
                        dest='weight_decay', help='weight decay')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (iters)')

    # ---------------- Model ----------------
    parser.add_argument('--model', default='lavt', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--mha', default='',
                        help='Numbers of heads in stage-1..4 PWAMs; e.g., "4-4-4-4"')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--swin_type', default='base',
                        help='Swin variant: tiny, small, base, large')

    # ---------------- BERT / Tokenizer ----------------
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased',
                        help='HF name or local dir for tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased',
                        help='HF name or local dir for BERT weights (if used elsewhere)')
    parser.add_argument('--hf_cache_dir', default='',
                        help='Optional HF cache directory for tokenizer/models')
    parser.add_argument('--language_encoder', default='clip', choices=['bert', 'clip'],
                        help='Select language encoder: bert (default) or clip.')
    parser.add_argument('--visual_encoder', default='clip', choices=['swin', 'clip'],
                        help='Select vision encoder: swin backbone or CLIP vision tower (onnx).')
    parser.add_argument('--clip_model_path', default='/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/Clipmodel/',
                        help='Local offline snapshot for CLIP (openai/clip-vit-base-patch32).')
    parser.add_argument('--clip_vision_onnx', default='/10T/students/doctor/2025/zum/PyCharm-Remote/FIANet/clip-resnet-101-visual.onnx',
                        help='Local offline CLIP visual encoder (ONNX).')

    # ---------------- Data selection & roots ----------------
    # Unified selector to switch datasets explicitly.
    parser.add_argument(
        '--dataset_name',
        default='lavt_risrs',
        choices=['auto', 'vdd_ris', 'rrsisd', 'lavt_risrs', 'risbench'],
        help=(
            "Which dataset layout to use:\n"
            "  auto       : infer from --data_root (recommended)\n"
            "  vdd_ris    : VDD_RIS layout (e.g., refer/VDD_RIS)\n"
            "  rrsisd     : RRSIS-D 'data' layout (e.g., refer/data)\n"
            "  lavt_risrs : original LAVT-RISRS layout (e.g., refer/LAVT-RISRS)\n"
            "  risbench   : RISBench_dataset (e.g., refer/RISBench_dataset with img_rgb/, mask/, and output_phrase_{split}.txt)"
        )
    )
    parser.add_argument(
        '--data_root',
        default='',
        help=(
            'Parent dataset root. The loader will auto-detect layout when --dataset_name=auto. '
            'Examples:\n'
            '  VDD_RIS   : /path/to/refer/VDD_RIS\n'
            '  RRSIS-D   : /path/to/refer/data\n'
            '  LAVT-RISRS: /path/to/refer/LAVT-RISRS\n'
            '  RISBench  : /10T/students/doctor/2025/zum/PyCharm-Remote/RMSIN/RRSIS/refer/RISBench_dataset'
        )
    )
    # Back-compat alias (some scripts pass dataset_root instead)
    parser.add_argument(
        '--dataset_root',
        default='',
        help='Alias of --data_root. If both are given, --data_root takes precedence.'
    )

    # Old arg kept for compatibility (not used by the new loader, but harmless)
    parser.add_argument('--refer_data_root', default='./refer/data/',
                        help='[Deprecated] Old REFER root; prefer --data_root')

    # ---------------- Splits & testing ----------------
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='dataset split to use (esp. for standalone testing/inference)')

    # NEW: always-evaluate-on-these-splits during training (comma-separated list)
    parser.add_argument(
        '--eval_splits',
        default='val,test',
        help='Comma-separated splits to evaluate at checkpoints during training. Typical: "val,test".'
    )

    parser.add_argument('--device', default='cuda:0',
                        help='device string for single-machine testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint path')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='when testing: weights originated from DDP training')
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')

    # ---------------- Dataloader ----------------
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true',
                        help='pin memory for DataLoader')

    # ---------------- Output ----------------
    parser.add_argument('--output-dir', default='./checkpoints/',
                        help='directory for saving checkpoints')

    # ---------------- Legacy COCO-style refs options ----------------
    # Retained only because some downstream code may still read these.
    parser.add_argument('--dataset', default='refcoco',
                        help='[Legacy] refcoco family arg (unused by new loaders)')
    parser.add_argument('--splitBy', default='unc',
                        help='[Legacy] refcoco splitBy (unused by new loaders)')

    return parser


def _coalesce_roots(args):
    """
    Normalize/resolve roots. If only dataset_root is provided, copy to data_root.
    Warn if both are set and differ.
    """
    if args.data_root and args.dataset_root and \
       os.path.abspath(args.data_root) != os.path.abspath(args.dataset_root):
        warnings.warn(
            f"[args] Both --data_root and --dataset_root were set and differ.\n"
            f"Using --data_root={args.data_root}"
        )
    if not args.data_root and args.dataset_root:
        args.data_root = args.dataset_root

    # Normalize empty strings to None-like for downstream
    args.data_root = _abspath_or_none(args.data_root) if args.data_root else args.data_root
    return args


def _maybe_set_env(args):
    """
    Provide a gentle bridge for code that reads RRSIS_DATA_ROOT env.
    """
    if args.data_root and not os.getenv("RRSIS_DATA_ROOT"):
        os.environ["RRSIS_DATA_ROOT"] = args.data_root


def _print_dataset_choice(args):
    root = args.data_root or "(auto default)"
    eval_splits = getattr(args, "eval_splits", "val,test")
    print(f"[args] dataset_name={args.dataset_name} | data_root={root} | split={args.split} | eval_splits={eval_splits}")


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    args = _coalesce_roots(args)
    _maybe_set_env(args)
    _print_dataset_choice(args)
    return args


if __name__ == "__main__":
    _ = parse_args()
