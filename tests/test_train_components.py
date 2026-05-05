from __future__ import annotations

import torch

from train import DiceCELoss, _count_patch_class_hits


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
