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
import random
from pathlib import Path
from typing import Sequence

import nibabel as nib
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
        "--stratify-by-class-presence",
        action="store_true",
        help=(
            "Build folds with a deterministic greedy multilabel stratifier "
            "based on foreground class presence in each label volume. This "
            "helps rare labels appear across validation folds when possible. "
            "Cannot be combined with --group-regex."
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


def _load_case_class_presence(
    label_dir: Path,
    case_ids: Sequence[str],
) -> dict[str, set[int]]:
    """Load foreground class-presence sets for each case."""
    presence: dict[str, set[int]] = {}
    for case_id in case_ids:
        label_path = label_dir / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for case {case_id}: {label_path}")
        label = np.asanyarray(nib.load(str(label_path)).dataobj)
        classes = {int(c) for c in np.unique(label).tolist() if int(c) > 0}
        presence[case_id] = classes
    return presence


def _build_stratified_folds_from_presence(
    case_ids: list[str],
    class_presence: dict[str, set[int]],
    *,
    n_folds: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    """Greedy multilabel stratification over per-case foreground labels."""
    if n_folds < 2:
        raise ValueError(f"--n-folds must be >= 2, got {n_folds}")
    if n_folds > len(case_ids):
        raise ValueError(
            f"--n-folds={n_folds} exceeds number of cases ({len(case_ids)})."
        )

    rng = random.Random(seed)
    shuffled_ids = case_ids[:]
    rng.shuffle(shuffled_ids)

    all_classes = sorted(set().union(*(class_presence.get(cid, set()) for cid in case_ids)))
    if not all_classes:
        return _build_folds(case_ids, n_folds=n_folds, seed=seed, groups=None)

    global_counts = {
        class_id: sum(class_id in class_presence.get(cid, set()) for cid in case_ids)
        for class_id in all_classes
    }
    target_fold_size = len(case_ids) / n_folds
    target_class_counts = {
        class_id: global_counts[class_id] / n_folds for class_id in all_classes
    }
    random_rank = {case_id: idx for idx, case_id in enumerate(shuffled_ids)}

    def _case_sort_key(case_id: str) -> tuple[int, int, int, str]:
        classes = class_presence.get(case_id, set())
        rarest = min((global_counts[c] for c in classes), default=len(case_ids) + 1)
        return (rarest, -len(classes), random_rank[case_id], case_id)

    ordered_ids = sorted(case_ids, key=_case_sort_key)
    fold_cases: list[list[str]] = [[] for _ in range(n_folds)]
    fold_class_counts = [
        {class_id: 0 for class_id in all_classes} for _ in range(n_folds)
    ]

    for case_id in ordered_ids:
        classes = class_presence.get(case_id, set())
        best_fold = 0
        best_score: tuple[float, int, int] | None = None
        for fold_idx in range(n_folds):
            next_size = len(fold_cases[fold_idx]) + 1
            size_score = ((next_size - target_fold_size) / max(target_fold_size, 1.0)) ** 2
            class_score = 0.0
            for class_id in classes:
                next_count = fold_class_counts[fold_idx][class_id] + 1
                target = max(target_class_counts[class_id], 1e-6)
                class_score += ((next_count - target) / target) ** 2
            score = (class_score + 0.05 * size_score, next_size, fold_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_idx

        fold_cases[best_fold].append(case_id)
        for class_id in classes:
            fold_class_counts[best_fold][class_id] += 1

    folds: list[dict[str, list[str]]] = []
    all_id_set = set(case_ids)
    for fold_idx in range(n_folds):
        val_ids = sorted(fold_cases[fold_idx])
        train_ids = sorted(all_id_set - set(val_ids))
        folds.append({"train": train_ids, "val": val_ids})
    return folds


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
    if args.stratify_by_class_presence and groups is not None:
        raise ValueError("--stratify-by-class-presence cannot be combined with --group-regex.")
    class_presence: dict[str, set[int]] | None = None
    if args.stratify_by_class_presence:
        class_presence = _load_case_class_presence(args.label_dir, pool)
        folds = _build_stratified_folds_from_presence(
            pool,
            class_presence,
            n_folds=args.n_folds,
            seed=args.seed,
        )
    else:
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
    if class_presence is not None:
        global_classes = sorted(set().union(*class_presence.values()))
        rare_classes = [
            class_id
            for class_id in global_classes
            if sum(class_id in class_presence[cid] for cid in pool) < args.n_folds
        ]
        if rare_classes:
            print(
                "Warning: these classes appear in fewer cases than folds and "
                f"cannot be present in every validation fold: {rare_classes}"
            )
    for i, fold in enumerate(folds):
        print(
            f"  fold {i}: train={len(fold['train'])} val={len(fold['val'])}"
            f"  val={fold['val']}"
        )
        if class_presence is not None:
            val_classes = sorted(
                set().union(*(class_presence[cid] for cid in fold["val"]))
                if fold["val"]
                else set()
            )
            print(f"    val foreground classes={val_classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
