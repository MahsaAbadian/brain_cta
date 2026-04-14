from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOPBRAIN_METRIC_KEYS: tuple[str, ...] = (
    "clsavg_dice",
    "clsavg_cldice",
    "clsavg_b0",
    "clsavg_hd95",
    "clsavg_invalid_neighbors",
    "sideroad_f1",
)

_CASE_METRIC_SPECS: dict[str, tuple[str, str]] = {
    "clsavg_dice": ("Dice_ClsAvgDice", "Dice_"),
    "clsavg_cldice": ("clDice_ClsAvgclDice", "clDice_"),
    "clsavg_b0": ("B0err_ClsAvgB0err", "B0err_"),
    "clsavg_hd95": ("HD95_ClsAvgHD95", "HD95_"),
    "clsavg_invalid_neighbors": ("NbErr_ClsAvgNbErr", "NbErr_"),
}


def select_fixed_topbrain_subset(
    case_ids: list[str],
    subset_size: int,
    seed: int,
) -> list[str]:
    if subset_size <= 0 or not case_ids:
        return []
    if subset_size >= len(case_ids):
        return list(case_ids)
    rng = random.Random(seed)
    sampled = rng.sample(list(case_ids), k=subset_size)
    sampled.sort()
    return sampled


def should_run_topbrain_full_eval(epoch: int, every_n_epochs: int) -> bool:
    if every_n_epochs <= 0:
        raise ValueError("topbrain full-eval interval must be >= 1")
    return epoch % every_n_epochs == 0


def empty_topbrain_metrics() -> dict[str, float]:
    data = {key: float("nan") for key in TOPBRAIN_METRIC_KEYS}
    data["num_cases"] = 0.0
    return data


def summarize_topbrain_case_metrics(
    case_metrics: list[dict[str, float]],
    sideroad_f1: float | None,
) -> dict[str, float]:
    summary = empty_topbrain_metrics()
    if case_metrics:
        for out_key in _CASE_METRIC_SPECS:
            values = [m[out_key] for m in case_metrics if out_key in m and np.isfinite(m[out_key])]
            if values:
                summary[out_key] = float(np.mean(values))
        summary["num_cases"] = float(len(case_metrics))
    if sideroad_f1 is not None and np.isfinite(sideroad_f1):
        summary["sideroad_f1"] = float(sideroad_f1)
    return summary


def _extract_case_metric(raw_metrics: dict[str, Any], metric_key: str) -> float:
    expected_key, prefix = _CASE_METRIC_SPECS[metric_key]
    if expected_key in raw_metrics:
        return float(raw_metrics[expected_key])
    for key, value in raw_metrics.items():
        if key.startswith(prefix) and "ClsAvg" in key:
            return float(value)
    return float("nan")


class TopBrainRuntime:
    def __init__(self, track: str, package_root: Path | None = None):
        self.available = False
        self.error_message = ""
        self.track = track.lower()

        if package_root is None:
            package_root = Path(__file__).resolve().parent.parent / "TopBrain_Eval_Metrics-master"
        if package_root.exists():
            pkg_path = str(package_root)
            if pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)

        try:
            import pandas as pd
            import SimpleITK as sitk
            from topbrain25_eval.aggregate.aggregate_all_detection_dicts import (
                aggregate_all_detection_dicts,
            )
            from topbrain25_eval.constants import TRACK
            from topbrain25_eval.score_case_task_1_seg import score_case_task_1_seg
        except Exception as exc:  # pragma: no cover - depends on local env
            self.error_message = str(exc)
            return

        track_map = {
            "ct": TRACK.CT,
            "mr": TRACK.MR,
        }
        if self.track not in track_map:
            self.error_message = f"Unsupported TopBrain track: {track}"
            return

        self.available = True
        self._pd = pd
        self._sitk = sitk
        self._score_case_task_1_seg = score_case_task_1_seg
        self._aggregate_all_detection_dicts = aggregate_all_detection_dicts
        self._track_enum = track_map[self.track]

    def create_accumulator(self) -> "TopBrainAccumulator":
        return TopBrainAccumulator(self)


class TopBrainAccumulator:
    def __init__(self, runtime: TopBrainRuntime):
        self._runtime = runtime
        self._case_metrics: list[dict[str, float]] = []
        self._all_detection_dicts: list[dict[str, Any]] = []

    def add_case(self, *, case_id: str, pred_xyz: np.ndarray, label_path: Path) -> None:
        if not self._runtime.available:
            return
        _ = case_id

        gt = self._runtime._sitk.ReadImage(str(label_path))
        pred = self._runtime._sitk.GetImageFromArray(
            pred_xyz.transpose((2, 1, 0)).astype(np.uint8)
        )

        raw_metrics: dict[str, Any] = {}
        self._runtime._score_case_task_1_seg(
            track=self._runtime._track_enum,
            gt=gt,
            pred=pred,
            metrics_dict=raw_metrics,
        )

        case_result = {
            key: _extract_case_metric(raw_metrics, key) for key in _CASE_METRIC_SPECS
        }
        self._case_metrics.append(case_result)
        detection_dict = raw_metrics.get("all_detection_dicts")
        if isinstance(detection_dict, dict):
            self._all_detection_dicts.append(detection_dict)

    def finalize(self) -> dict[str, float]:
        if not self._runtime.available:
            return empty_topbrain_metrics()

        sideroad_f1: float | None = None
        if self._all_detection_dicts:
            detection_series = self._runtime._pd.Series(self._all_detection_dicts)
            detection_agg = self._runtime._aggregate_all_detection_dicts(
                self._runtime._track_enum, detection_series
            )
            f1_payload = detection_agg.get("f1_score", {})
            if isinstance(f1_payload, dict) and "mean" in f1_payload:
                sideroad_f1 = float(f1_payload["mean"])

        return summarize_topbrain_case_metrics(self._case_metrics, sideroad_f1)
