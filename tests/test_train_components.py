from __future__ import annotations

import torch
import nibabel as nib
import numpy as np

from train import (
    DiceCELoss,
    _compute_split_class_stats,
    _count_patch_class_hits,
    _grad_accum_divisor,
    _is_metric_improved,
    _mean_dice_for_classes,
    _should_step_optimizer,
)


def test_dice_ce_loss_forward_and_backward() -> None:
    criterion = DiceCELoss(num_classes=5, include_background=True)
    logits = torch.randn(2, 5, 16, 16, 16, requires_grad=True)
    target = torch.randint(0, 5, (2, 16, 16, 16), dtype=torch.long)

    loss = criterion(logits, target)
    assert torch.isfinite(loss).item()
    assert loss.ndim == 0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()


def test_count_patch_class_hits_counts_patch_presence_once() -> None:
    labels = torch.tensor(
        [
            [
                [[0, 1], [1, 1]],
                [[0, 0], [0, 0]],
            ],
            [
                [[2, 2], [3, 3]],
                [[2, 2], [3, 3]],
            ],
            [
                [[1, 4], [4, 4]],
                [[1, 1], [1, 1]],
            ],
        ],
        dtype=torch.long,
    )

    hits, patch_count = _count_patch_class_hits(labels, num_classes=5)

    assert patch_count == 3
    assert hits == [1, 2, 1, 1, 1]


def test_compute_split_class_stats_counts_cases_and_voxels(tmp_path) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    affine = np.eye(4)
    case_a = np.array([[[0, 1], [1, 2]]], dtype=np.int16)
    case_b = np.array([[[0, 0], [2, 2]]], dtype=np.int16)
    nib.save(nib.Nifti1Image(case_a, affine), label_dir / "case_a.nii.gz")
    nib.save(nib.Nifti1Image(case_b, affine), label_dir / "case_b.nii.gz")

    case_counts, voxel_counts, total_voxels = _compute_split_class_stats(
        label_dir=label_dir,
        case_ids=["case_a", "case_b"],
        num_classes=4,
    )

    assert total_voxels == 8
    assert case_counts == [2, 1, 2, 0]
    assert voxel_counts == [3, 2, 3, 0]


def test_mean_dice_for_classes_can_require_support() -> None:
    per_class_dice = [0.9, 0.2, 0.4, 1.0]
    support = [2, 1, 1, 0]

    assert _mean_dice_for_classes(per_class_dice, (1, 2, 3)) == 0.5333333333333333
    assert _mean_dice_for_classes(
        per_class_dice,
        (1, 2, 3),
        support_counts=support,
    ) == 0.30000000000000004


def test_is_metric_improved_respects_min_delta_and_nan() -> None:
    assert _is_metric_improved(0.51, 0.50, min_delta=0.0)
    assert _is_metric_improved(0.52, 0.50, min_delta=0.01)
    assert not _is_metric_improved(0.505, 0.50, min_delta=0.01)
    assert not _is_metric_improved(float("nan"), 0.50, min_delta=0.0)
    assert _is_metric_improved(0.10, float("-inf"), min_delta=0.0)


def test_should_step_optimizer_handles_accumulation_tail() -> None:
    step_batches = [
        idx
        for idx in range(1, 6)
        if _should_step_optimizer(idx, total_batches=5, grad_accum_steps=2)
    ]
    assert step_batches == [2, 4, 5]


def test_grad_accum_divisor_uses_tail_window_size() -> None:
    divisors = [
        _grad_accum_divisor(idx, total_batches=5, grad_accum_steps=2)
        for idx in range(1, 6)
    ]
    assert divisors == [2, 2, 2, 2, 1]
