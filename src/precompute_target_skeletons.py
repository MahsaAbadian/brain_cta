"""Precompute per-case target skeletons for clDice training.

Default method is 'lee' (skimage Lee thinning), which produces thin
single-voxel centerlines matching the TopBrain evaluation metric.
The 'soft' method uses the differentiable soft-skeletonize from losses.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from data_utils import read_num_classes_from_labelmap


def _import_skeletonize_lee():
    """Import skimage skeletonize with a NumPy 2.x compatibility shim.

    Some environments accidentally install scikit-image versions that still
    reference removed NumPy aliases (for example `np.float_`), which crashes at
    import time. We patch those aliases only when missing, then retry import.
    """
    try:
        from skimage.morphology import skeletonize  # pyright: ignore[reportMissingImports]
        return skeletonize
    except Exception as exc:
        err = str(exc)
        if "np.float_" in err and not hasattr(np, "float_"):
            np.float_ = np.float64  # type: ignore[attr-defined]
            try:
                from skimage.morphology import skeletonize  # pyright: ignore[reportMissingImports]
                return skeletonize
            except Exception as retry_exc:
                raise RuntimeError(
                    "Failed to import scikit-image skeletonize after NumPy compatibility shim. "
                    "Use compatible deps, e.g. `pip install 'numpy<2' 'scikit-image==0.21.0'` "
                    "or upgrade scikit-image to a NumPy-2-compatible release."
                ) from retry_exc
        raise RuntimeError(
            "Failed to import scikit-image skeletonize. "
            "Install compatible deps, e.g. `pip install 'numpy<2' 'scikit-image==0.21.0'`."
        ) from exc


def _parse_class_id_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if len(values) == 0:
        return None
    try:
        out = tuple(int(v) for v in values)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --class-ids='{raw}'. Expected comma-separated integers."
        ) from exc
    return tuple(sorted(set(out)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute soft target skeleton volumes for clDice."
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("training_data_resampled/labelsTr_topbrain_ct"),
        help="Directory containing <case_id>.nii.gz label files.",
    )
    parser.add_argument(
        "--labelmap-path",
        type=Path,
        default=Path("training_data_resampled/itksnap_labelmap_txt/labelmap_topbrain_ct.txt"),
        help="Path to labelmap text file for determining class count.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for <case_id>.npz skeleton files.",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated class IDs to skeletonize. "
            "Default is all foreground classes."
        ),
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="Optional comma-separated case IDs. Default processes all labels in label-dir.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Also precompute background skeleton (not recommended).",
    )
    parser.add_argument(
        "--method",
        choices=("lee", "soft"),
        default="lee",
        help=(
            "Skeletonization method. 'lee' uses skimage Lee thinning "
            "(thin, matches eval metric). 'soft' uses differentiable "
            "soft-skeletonize from losses.py (thicker, needs --iters)."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=12,
        help="Number of soft skeletonization iterations (only used with --method soft).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (only used with --method soft). Default auto-detect.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Output dtype for stored skeleton volumes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in out-dir.",
    )
    return parser.parse_args()


def _default_class_ids(
    num_classes: int,
    include_background: bool,
) -> tuple[int, ...]:
    start = 0 if include_background else 1
    return tuple(range(start, num_classes))


def _list_case_ids(label_dir: Path) -> list[str]:
    case_ids = []
    for path in sorted(label_dir.glob("*.nii.gz")):
        case_ids.append(path.name.replace(".nii.gz", ""))
    return case_ids


def main() -> int:
    args = parse_args()
    if not args.label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {args.label_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.iters < 0:
        raise ValueError("--iters must be >= 0")

    num_classes = read_num_classes_from_labelmap(args.labelmap_path)
    class_ids = _parse_class_id_list(args.class_ids)
    if class_ids is None:
        class_ids = _default_class_ids(num_classes, args.include_background)
    for class_id in class_ids:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"Invalid class id {class_id}; valid range is [0, {num_classes - 1}]"
            )
    if (not args.include_background) and any(c == 0 for c in class_ids):
        raise ValueError(
            "class-ids includes 0 but --include-background is not set."
        )

    use_lee = args.method == "lee"
    skeletonize_lee = None
    if use_lee:
        skeletonize_lee = _import_skeletonize_lee()

    device = None
    if not use_lee:
        import torch
        from losses import DiceCELoss

        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    case_ids = _list_case_ids(args.label_dir)
    if args.case_ids:
        requested = [tok.strip() for tok in args.case_ids.split(",") if tok.strip()]
        requested_set = set(requested)
        case_ids = [cid for cid in case_ids if cid in requested_set]
    if len(case_ids) == 0:
        raise RuntimeError(f"No .nii.gz files found in {args.label_dir}")
    print(
        f"method={args.method} cases={len(case_ids)} classes={class_ids} "
        f"{'iters=' + str(args.iters) + ' device=' + str(device) if not use_lee else ''} "
        f"dtype={args.dtype}"
    )

    for i, case_id in enumerate(case_ids, start=1):
        out_path = args.out_dir / f"{case_id}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(case_ids)}] skip existing {out_path.name}")
            continue

        label_path = args.label_dir / f"{case_id}.nii.gz"
        label_np = np.asanyarray(nib.load(str(label_path)).dataobj).astype(np.int64)
        skel_channels = np.zeros((len(class_ids), *label_np.shape), dtype=np.float32)

        for ch_idx, class_id in enumerate(class_ids):
            mask_bool = (label_np == class_id)

            if use_lee:
                skel_channels[ch_idx] = skeletonize_lee(
                    mask_bool, method="lee"
                ).astype(np.float32)
            else:
                mask_t = torch.from_numpy(
                    mask_bool.astype(np.float32)
                ).to(device=device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    skel_t = DiceCELoss._soft_skeletonize(mask_t, args.iters)
                skel_channels[ch_idx] = (
                    skel_t[0, 0].detach().cpu().numpy().astype(np.float32)
                )

        metadata = {
            "class_ids": np.asarray(class_ids, dtype=np.int16),
            "method": np.array([args.method]),
        }
        if not use_lee:
            metadata["iters"] = np.asarray([args.iters], dtype=np.int16)

        np.savez_compressed(
            out_path,
            skel=skel_channels.astype(out_dtype),
            **metadata,
        )
        print(
            f"[{i}/{len(case_ids)}] wrote {out_path.name} "
            f"shape={skel_channels.shape} "
            f"skel_voxels={int(skel_channels.sum())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
