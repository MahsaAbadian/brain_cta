#!/usr/bin/env python3
"""Stitch per-fold validation predictions into one out-of-fold set and (optionally) run TopBrain eval.

Use this after training each fold independently. For each case, the prediction
from the fold where that case was held out is copied into ``<out-dir>/predictions/``
together with the matching ground-truth in ``<out-dir>/ground-truth/``. The
resulting folder pair is the same layout accepted by ``src/evaluate_challenge_like.py``
and ``topbrain25_eval.TopBrainEvaluation``.

Example (baseline):
    python scripts/eval_oof.py \
      --fold-pattern 'runs/cldice_kfold/fold_{fold}/challenge_like_eval/predictions' \
      --out-dir runs/cldice_kfold/oof --run-topbrain-eval

Example (nnUNet, using its built-in post-training validation dumps):
    python scripts/eval_oof.py \
      --fold-pattern \
      'nnUNet_results/Dataset501_TopBrainCTA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/validation' \
      --out-dir runs/nnunet/oof --run-topbrain-eval
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path("training_data_resampled/split/splits_final.json"),
        help="Canonical K-fold splits file.",
    )
    parser.add_argument(
        "--fold-pattern",
        type=str,
        required=True,
        help=(
            "Directory pattern containing {fold} that points at each fold's "
            "prediction directory. Example: 'runs/nnunet/predictions_fold{fold}'."
        ),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("training_data_resampled/labelsTr_topbrain_ct"),
        help="Ground-truth directory used to copy labels to out_dir/ground-truth.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination folder. Will contain predictions/ and ground-truth/.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=(
            "Optional comma-separated fold indices to include. Default is every "
            "fold present in splits_json."
        ),
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
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "If set, skip cases whose prediction file cannot be located. "
            "Default is to raise."
        ),
    )
    return parser.parse_args()


def _resolve_prediction_file(pred_dir: Path, case_id: str) -> Path | None:
    candidates = [
        pred_dir / f"{case_id}.nii.gz",
        pred_dir / f"{case_id}_0000.nii.gz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_folds(splits_json: Path, selection: list[int] | None) -> list[tuple[int, dict]]:
    if not splits_json.is_file():
        raise FileNotFoundError(f"Missing splits file: {splits_json}")
    folds = json.loads(splits_json.read_text())
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"Malformed splits file: {splits_json}")
    indexed = list(enumerate(folds))
    if selection is None:
        return indexed
    wanted = sorted({i for i in selection})
    for i in wanted:
        if i < 0 or i >= len(folds):
            raise IndexError(f"Fold {i} not in splits (have {len(folds)} folds).")
    return [indexed[i] for i in wanted]


def main() -> int:
    args = parse_args()
    if "{fold}" not in args.fold_pattern:
        raise ValueError("--fold-pattern must contain the '{fold}' placeholder.")

    selection: list[int] | None = None
    if args.folds:
        selection = [int(x) for x in args.folds.split(",") if x.strip()]

    folds = _load_folds(args.splits_json, selection)

    out_pred_dir = args.out_dir / "predictions"
    out_gt_dir = args.out_dir / "ground-truth"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, int] = {}
    missing: list[tuple[int, str]] = []

    for fold_idx, fold_entry in folds:
        pred_dir = Path(args.fold_pattern.format(fold=fold_idx))
        if not pred_dir.is_dir():
            raise FileNotFoundError(
                f"Fold {fold_idx} prediction dir missing: {pred_dir}"
            )
        val_ids = sorted(set(fold_entry["val"]))
        for case_id in val_ids:
            if case_id in seen:
                raise RuntimeError(
                    f"Case {case_id} appears in both fold {seen[case_id]} and "
                    f"fold {fold_idx}; splits are not a valid K-fold partition."
                )
            pred_src = _resolve_prediction_file(pred_dir, case_id)
            if pred_src is None:
                missing.append((fold_idx, case_id))
                if not args.allow_missing:
                    raise FileNotFoundError(
                        f"No prediction for case {case_id} in {pred_dir}"
                    )
                continue
            gt_src = args.label_dir / f"{case_id}.nii.gz"
            if not gt_src.is_file():
                raise FileNotFoundError(
                    f"Missing ground truth for case {case_id}: {gt_src}"
                )
            shutil.copy2(pred_src, out_pred_dir / f"{case_id}.nii.gz")
            shutil.copy2(gt_src, out_gt_dir / f"{case_id}.nii.gz")
            seen[case_id] = fold_idx

    if missing:
        print(f"Skipped {len(missing)} missing predictions (--allow-missing).")
        for fi, cid in missing[:10]:
            print(f"  fold {fi}: {cid}")
        if len(missing) > 10:
            print(f"  ...and {len(missing) - 10} more")

    print(f"OOF predictions: {out_pred_dir}")
    print(f"OOF ground-truth: {out_gt_dir}")
    print(f"Cases aggregated: {len(seen)}")

    if args.run_topbrain_eval:
        try:
            from topbrain25_eval.evaluation import TopBrainEvaluation
        except ImportError:
            print(
                "topbrain25_eval is not installed; skipping automated eval. "
                "Install with: pip install git+https://github.com/CoWBenchmark/TopBrain_Eval_Metrics.git"
            )
            return 0
        eval_out = args.out_dir / "topbrain_eval_output"
        eval_out.mkdir(parents=True, exist_ok=True)
        try:
            TopBrainEvaluation(
                track=args.topbrain_track,
                expected_num_cases=len(seen),
                predictions_path=out_pred_dir,
                ground_truth_path=out_gt_dir,
                output_path=eval_out,
            ).evaluate()
            print(f"TopBrain evaluation finished. Output: {eval_out}")
        except Exception as exc:  # pragma: no cover - external package
            print(f"TopBrain eval failed: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
