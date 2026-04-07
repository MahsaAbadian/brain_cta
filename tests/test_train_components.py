from __future__ import annotations

import torch

from train import DiceCELoss


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
