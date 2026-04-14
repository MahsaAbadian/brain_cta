from __future__ import annotations

import math

from topbrain_validation import (
    select_fixed_topbrain_subset,
    should_run_topbrain_full_eval,
    summarize_topbrain_case_metrics,
)


def test_select_fixed_topbrain_subset_is_deterministic() -> None:
    case_ids = [f"case_{i:03d}" for i in range(10)]
    subset_a = select_fixed_topbrain_subset(case_ids=case_ids, subset_size=4, seed=42)
    subset_b = select_fixed_topbrain_subset(case_ids=case_ids, subset_size=4, seed=42)
    assert subset_a == subset_b
    assert len(subset_a) == 4


def test_should_run_topbrain_full_eval_interval() -> None:
    assert not should_run_topbrain_full_eval(epoch=1, every_n_epochs=5)
    assert not should_run_topbrain_full_eval(epoch=4, every_n_epochs=5)
    assert should_run_topbrain_full_eval(epoch=5, every_n_epochs=5)
    assert should_run_topbrain_full_eval(epoch=10, every_n_epochs=5)


def test_summarize_topbrain_case_metrics_aggregates_values() -> None:
    case_metrics = [
        {
            "clsavg_dice": 0.80,
            "clsavg_cldice": 0.70,
            "clsavg_b0": 0.30,
            "clsavg_hd95": 9.0,
            "clsavg_invalid_neighbors": 1.5,
        },
        {
            "clsavg_dice": 0.60,
            "clsavg_cldice": 0.50,
            "clsavg_b0": 0.10,
            "clsavg_hd95": 11.0,
            "clsavg_invalid_neighbors": 2.5,
        },
    ]
    summary = summarize_topbrain_case_metrics(case_metrics=case_metrics, sideroad_f1=0.45)
    assert summary["num_cases"] == 2.0
    assert math.isclose(summary["clsavg_dice"], 0.70)
    assert math.isclose(summary["clsavg_cldice"], 0.60)
    assert math.isclose(summary["clsavg_b0"], 0.20)
    assert math.isclose(summary["clsavg_hd95"], 10.0)
    assert math.isclose(summary["clsavg_invalid_neighbors"], 2.0)
    assert math.isclose(summary["sideroad_f1"], 0.45)
