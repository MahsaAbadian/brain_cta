from __future__ import annotations

import torch
import nibabel as nib
import numpy as np

from train import (
    DiceCELoss,
    _compute_split_class_stats,
    _count_patch_class_hits,
    _mean_dice_for_classes,
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
