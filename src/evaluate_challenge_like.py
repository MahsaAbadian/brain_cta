"""Challenge-like local evaluation with full-volume inference.

This script is designed to make local validation closer to TopBrain leaderboard
evaluation:
  1) Run sliding-window inference on full 3D validation cases.
  2) Save predicted segmentation masks as NIfTI files.
  3) Optionally run the official TopBrain evaluation package (if installed).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from data_utils import load_split_ids, read_num_classes_from_labelmap
from model_3d_unet import UNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run challenge-like full-volume validation and optional TopBrain metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to model weights. Supports either a raw state_dict file "
            "(e.g., runs/.../model_final_weights.pt) or a checkpoint dict "
            "with a model_state key."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/challenge_like_eval"),
        help="Output folder for predictions and evaluation artifacts.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("training_data_resampled/split"),
        help="Directory containing val_cases.txt.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("training_data_resampled/imagesTr_topbrain_ct"),
        help="Directory with preprocessed images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("training_data_resampled/labelsTr_topbrain_ct"),
        help="Directory with preprocessed labels.",
    )
    parser.add_argument(
        "--labelmap-path",
        type=Path,
        default=Path("training_data_resampled/itksnap_labelmap_txt/labelmap_topbrain_ct.txt"),
        help="ITK-SNAP label map file used to infer num_classes.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Sliding-window patch size (x y z).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=(64, 64, 64),
        help="Sliding-window stride (x y z).",
    )
    parser.add_argument(
        "--base-ch",
        type=int,
        default=32,
        help="UNet base channels (must match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--run-topbrain-eval",
        action="store_true",
        help="If set, attempts to run topbrain25_eval after predictions are written.",
    )
    parser.add_argument(
        "--topbrain-track",
        type=str,
        default="cta",
        help="Track string passed to topbrain25_eval.TopBrainEvaluation.",
    )
    return parser.parse_args()


def _axis_starts(dim: int, patch: int, stride: int) -> list[int]:
    if patch > dim:
        raise ValueError(f"Patch size {patch} is larger than dimension {dim}.")
    starts = list(range(0, dim - patch + 1, stride))
    if not starts or starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts


def _sliding_window_predict(
    model: torch.nn.Module,
    image_xyz: np.ndarray,
    num_classes: int,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    xdim, ydim, zdim = image_xyz.shape
    px, py, pz = patch_size
    sx, sy, sz = stride

    xs = _axis_starts(xdim, px, sx)
    ys = _axis_starts(ydim, py, sy)
    zs = _axis_starts(zdim, pz, sz)

    logits_sum = np.zeros((num_classes, xdim, ydim, zdim), dtype=np.float32)
    counts = np.zeros((xdim, ydim, zdim), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    patch = image_xyz[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz]
                    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    logits = model(patch_t)[0].detach().cpu().numpy()
                    logits_sum[:, x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz] += logits
                    counts[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz] += 1.0

    counts = np.maximum(counts, 1.0)
    avg_logits = logits_sum / counts[None, ...]
    pred = np.argmax(avg_logits, axis=0).astype(np.int16)
    return pred


def _run_topbrain_eval(
    *,
    track: str,
    expected_num_cases: int,
    predictions_path: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> bool:
    try:
        from topbrain25_eval.evaluation import TopBrainEvaluation
    except ImportError:
        print(
            "topbrain25_eval is not installed. Install with:\n"
            "  pip install git+https://github.com/CoWBenchmark/TopBrain_Eval_Metrics.git"
        )
        return False

    try:
        eval_run = TopBrainEvaluation(
            track=track,
            expected_num_cases=expected_num_cases,
            predictions_path=predictions_path,
            ground_truth_path=ground_truth_path,
            output_path=output_path,
        )
        eval_run.evaluate()
    except Exception as exc:  # pragma: no cover - depends on external package API
        print(
            "Failed to run topbrain25_eval automatically.\n"
            f"Reason: {exc}\n"
            "Double-check the expected track value for your installed version."
        )
        return False
    return True


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = out_dir / "predictions"
    gt_dir = out_dir / "ground-truth"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    _, val_case_ids = load_split_ids(args.split_dir)
    val_case_ids = sorted(set(val_case_ids))
    if not val_case_ids:
        raise RuntimeError(f"No validation cases found in {args.split_dir}")

    num_classes = read_num_classes_from_labelmap(args.labelmap_path)
    model = UNet3D(in_channels=1, num_classes=num_classes, base_ch=args.base_ch).to(device)

    weights_obj = torch.load(args.checkpoint, map_location=device)
    state_dict = (
        weights_obj["model_state"]
        if isinstance(weights_obj, dict) and "model_state" in weights_obj
        else weights_obj
    )
    model.load_state_dict(state_dict)
    model.eval()

    patch_size = tuple(int(v) for v in args.patch_size)
    stride = tuple(int(v) for v in args.stride)

    print(
        f"Evaluating {len(val_case_ids)} cases on {device} with "
        f"patch_size={patch_size}, stride={stride}"
    )
    for i, case_id in enumerate(val_case_ids, start=1):
        img_path = args.image_dir / f"{case_id}_0000.nii.gz"
        lbl_path = args.label_dir / f"{case_id}.nii.gz"
        if not img_path.exists() or not lbl_path.exists():
            raise FileNotFoundError(
                f"Missing case files for {case_id}: {img_path} or {lbl_path}"
            )

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))
        image = np.asanyarray(img_nii.dataobj).astype(np.float32)
        pred = _sliding_window_predict(
            model=model,
            image_xyz=image,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
            device=device,
        )

        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = gt_dir / f"{case_id}.nii.gz"

        nib.save(
            nib.Nifti1Image(pred, lbl_nii.affine, lbl_nii.header.copy()),
            str(pred_path),
        )
        shutil.copy2(lbl_path, gt_path)
        print(f"[{i:03d}/{len(val_case_ids):03d}] wrote {pred_path.name}")

    print(f"Prediction folder: {pred_dir}")
    print(f"Ground-truth folder: {gt_dir}")

    if args.run_topbrain_eval:
        eval_out = out_dir / "topbrain_eval_output"
        eval_out.mkdir(parents=True, exist_ok=True)
        ok = _run_topbrain_eval(
            track=args.topbrain_track,
            expected_num_cases=len(val_case_ids),
            predictions_path=pred_dir,
            ground_truth_path=gt_dir,
            output_path=eval_out,
        )
        if ok:
            print(f"TopBrain evaluation finished. Output: {eval_out}")
        else:
            print(
                "TopBrain evaluation was skipped because the package is missing. "
                "Predictions were still generated."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
