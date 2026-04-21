#!/usr/bin/env python3
"""Prepare nnUNet predictions for existing challenge-like evaluation flow."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy nnUNet predictions and matching GT labels into the same folder layout "
            "used by src/evaluate_challenge_like.py."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory produced by nnUNetv2_predict.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("training_data_resampled/split"),
        help="Directory containing val_cases.txt.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("training_data_resampled/labelsTr_topbrain_ct"),
        help="Ground-truth labels directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/nnunet/challenge_like_eval"),
        help="Output directory with predictions/ and ground-truth/ folders.",
    )
    parser.add_argument(
        "--run-topbrain-eval",
        action="store_true",
        help="Attempt to run topbrain25_eval after exporting files.",
    )
    parser.add_argument(
        "--topbrain-track",
        type=str,
        default="cta",
        help="Track string passed to topbrain25_eval.TopBrainEvaluation.",
    )
    return parser.parse_args()


def _load_val_ids(split_dir: Path) -> list[str]:
    val_path = split_dir / "val_cases.txt"
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing val split file: {val_path}")
    val_ids = [x.strip() for x in val_path.read_text().splitlines() if x.strip()]
    val_ids = sorted(set(val_ids))
    if not val_ids:
        raise RuntimeError(f"No validation ids found in {val_path}")
    return val_ids


def _resolve_prediction_file(pred_dir: Path, case_id: str) -> Path:
    candidates = [
        pred_dir / f"{case_id}.nii.gz",
        pred_dir / f"{case_id}_0000.nii.gz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No prediction found for case {case_id} in {pred_dir}. "
        f"Tried: {[p.name for p in candidates]}"
    )


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
    except Exception as exc:  # pragma: no cover - external package
        print(f"Failed to run topbrain25_eval automatically: {exc}")
        return False
    return True


def main() -> int:
    args = parse_args()
    val_ids = _load_val_ids(args.split_dir)
    out_pred_dir = args.out_dir / "predictions"
    out_gt_dir = args.out_dir / "ground-truth"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    for idx, case_id in enumerate(val_ids, start=1):
        pred_src = _resolve_prediction_file(args.prediction_dir, case_id)
        gt_src = args.label_dir / f"{case_id}.nii.gz"
        if not gt_src.is_file():
            raise FileNotFoundError(f"Missing ground truth for case {case_id}: {gt_src}")

        pred_dst = out_pred_dir / f"{case_id}.nii.gz"
        gt_dst = out_gt_dir / f"{case_id}.nii.gz"
        shutil.copy2(pred_src, pred_dst)
        shutil.copy2(gt_src, gt_dst)
        print(f"[{idx:03d}/{len(val_ids):03d}] exported {case_id}")

    print(f"Prediction folder: {out_pred_dir}")
    print(f"Ground-truth folder: {out_gt_dir}")

    if args.run_topbrain_eval:
        eval_out = args.out_dir / "topbrain_eval_output"
        eval_out.mkdir(parents=True, exist_ok=True)
        ok = _run_topbrain_eval(
            track=args.topbrain_track,
            expected_num_cases=len(val_ids),
            predictions_path=out_pred_dir,
            ground_truth_path=out_gt_dir,
            output_path=eval_out,
        )
        if ok:
            print(f"TopBrain evaluation finished. Output: {eval_out}")
        else:
            print("TopBrain evaluation skipped. Exported files are still available.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
