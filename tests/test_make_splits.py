from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAKE_SPLITS_PATH = PROJECT_ROOT / "scripts" / "make_splits.py"
spec = importlib.util.spec_from_file_location("make_splits", MAKE_SPLITS_PATH)
assert spec is not None
make_splits = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(make_splits)


def test_stratified_folds_spread_class_presence() -> None:
    case_ids = [f"case_{i}" for i in range(6)]
    class_presence = {
        "case_0": {1, 2},
        "case_1": {1},
        "case_2": {1, 2},
        "case_3": {1},
        "case_4": {1, 3},
        "case_5": {1, 3},
    }

    folds = make_splits._build_stratified_folds_from_presence(
        case_ids,
        class_presence,
        n_folds=3,
        seed=123,
    )

    assert len(folds) == 3
    assert sorted(cid for fold in folds for cid in fold["val"]) == sorted(case_ids)
    assert [len(fold["val"]) for fold in folds] == [2, 2, 2]

    class_2_folds = [
        idx
        for idx, fold in enumerate(folds)
        if any(2 in class_presence[cid] for cid in fold["val"])
    ]
    class_3_folds = [
        idx
        for idx, fold in enumerate(folds)
        if any(3 in class_presence[cid] for cid in fold["val"])
    ]
    assert len(class_2_folds) == 2
    assert len(class_3_folds) == 2


def test_stratified_folds_fall_back_when_no_foreground() -> None:
    case_ids = [f"case_{i}" for i in range(4)]
    class_presence = {case_id: set() for case_id in case_ids}

    folds = make_splits._build_stratified_folds_from_presence(
        case_ids,
        class_presence,
        n_folds=2,
        seed=123,
    )

    assert len(folds) == 2
    assert sorted(cid for fold in folds for cid in fold["val"]) == sorted(case_ids)
