#!/usr/bin/env python3
"""Generate a canonical `splits_final.json` used by both the baseline and nnUNet.

The output is a list of ``{"train": [...], "val": [...]}`` dicts, one per fold,
which matches nnUNet v2's native split format. Seed and ordering are fixed so
that the same file is reproducible on any machine.

Example:
    python scripts/make_splits.py --n-folds 5 \
        --label-dir training_data_resampled/labelsTr_topbrain_ct \
        --output training_data_resampled/split/splits_final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an nnUNet-style splits_final.json with K folds shared by "
            "the baseline and nnUNet pipelines."
        )
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("training_data_resampled/labelsTr_topbrain_ct"),
        help="Directory whose *.nii.gz filenames define the case pool.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data_resampled/split/splits_final.json"),
        help="Destination splits_final.json.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed. 12345 matches nnUNet's internal default.",
    )
    parser.add_argument(
        "--holdout-file",
        type=Path,
        default=None,
        help=(
            "Optional text file (one case_id per line) that must never appear "
            "in any fold. Useful to carve off a true held-out test set."
        ),
    )
    parser.add_argument(
        "--group-regex",
        type=str,
        default=None,
        help=(
            "Optional regex whose first capture group maps a case_id to a "
            "group key. When provided, GroupKFold is used so the same subject "
            "never appears in both train and val of a fold. Example: "
            "'^(topcow_[a-z]+_\\d+)' to group by full case name."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def _discover_case_ids(label_dir: Path) -> list[str]:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    ids = sorted({p.name.removesuffix(".nii.gz") for p in label_dir.glob("*.nii.gz")})
    if not ids:
        raise RuntimeError(f"No *.nii.gz files found in {label_dir}")
    return ids


def _load_holdout_ids(holdout_file: Path | None) -> set[str]:
    if holdout_file is None:
        return set()
    if not holdout_file.is_file():
        raise FileNotFoundError(f"Holdout file not found: {holdout_file}")
    return {
        line.strip()
        for line in holdout_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def _groups_for_ids(case_ids: Sequence[str], pattern: str) -> list[str]:
    import re

    compiled = re.compile(pattern)
    groups: list[str] = []
    for cid in case_ids:
        match = compiled.search(cid)
        if match is None:
            raise ValueError(
                f"--group-regex {pattern!r} did not match case_id {cid!r}"
            )
        groups.append(match.group(1) if match.groups() else match.group(0))
    return groups


def _build_folds(
    case_ids: list[str],
    *,
    n_folds: int,
    seed: int,
    groups: list[str] | None,
) -> list[dict[str, list[str]]]:
    if n_folds < 2:
        raise ValueError(f"--n-folds must be >= 2, got {n_folds}")
    if n_folds > len(case_ids):
        raise ValueError(
            f"--n-folds={n_folds} exceeds number of cases ({len(case_ids)})."
        )

    case_arr = np.array(case_ids)
    if groups is not None:
        unique_groups = sorted(set(groups))
        if len(unique_groups) < n_folds:
            raise ValueError(
                f"GroupKFold needs at least {n_folds} distinct groups, "
                f"got {len(unique_groups)} from --group-regex."
            )
        splitter = GroupKFold(n_splits=n_folds)
        iterator = splitter.split(case_arr, groups=np.array(groups))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(case_arr)

    folds: list[dict[str, list[str]]] = []
    for train_idx, val_idx in iterator:
        folds.append(
            {
                "train": sorted(case_arr[train_idx].tolist()),
                "val": sorted(case_arr[val_idx].tolist()),
            }
        )
    return folds


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}. Re-run with --force to replace."
        )

    all_ids = _discover_case_ids(args.label_dir)
    holdout_ids = _load_holdout_ids(args.holdout_file)
    unknown_holdout = sorted(holdout_ids - set(all_ids))
    if unknown_holdout:
        raise ValueError(
            f"--holdout-file contains ids not present in label dir: {unknown_holdout}"
        )

    pool = [cid for cid in all_ids if cid not in holdout_ids]
    if len(pool) < args.n_folds:
        raise RuntimeError(
            f"Case pool after holdout has {len(pool)} ids; need >= {args.n_folds} "
            "for the requested fold count."
        )

    groups = (
        _groups_for_ids(pool, args.group_regex) if args.group_regex else None
    )
    folds = _build_folds(
        pool,
        n_folds=args.n_folds,
        seed=args.seed,
        groups=groups,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(folds, indent=2) + "\n")

    print(f"Wrote {args.output} with {len(folds)} folds over {len(pool)} cases.")
    if holdout_ids:
        print(f"Held-out cases excluded from all folds: {sorted(holdout_ids)}")
    for i, fold in enumerate(folds):
        print(
            f"  fold {i}: train={len(fold['train'])} val={len(fold['val'])}"
            f"  val={fold['val']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
