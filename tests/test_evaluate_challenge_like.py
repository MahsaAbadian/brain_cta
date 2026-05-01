from __future__ import annotations

import json
from pathlib import Path

from evaluate_challenge_like import _load_val_case_ids


def test_load_val_case_ids_prefers_splits_json(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "val_cases.txt").write_text("legacy_case\n")
    splits_json = split_dir / "splits_final.json"
    splits_json.write_text(
        json.dumps(
            [
                {"train": ["case_001"], "val": ["case_002", "case_003"]},
                {"train": ["case_002"], "val": ["case_001"]},
            ]
        )
        + "\n"
    )

    assert _load_val_case_ids(splits_json, split_dir, fold=0) == [
        "case_002",
        "case_003",
    ]


def test_load_val_case_ids_accepts_legacy_val_without_train(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "val_cases.txt").write_text("case_002\ncase_001\ncase_002\n")

    assert _load_val_case_ids(
        split_dir / "missing_splits_final.json", split_dir, fold=0
    ) == ["case_001", "case_002"]
