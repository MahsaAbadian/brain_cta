"""Short sanity-check training script for quick pipeline debugging."""

from __future__ import annotations

import torch

from data_loader import build_train_val_loaders
from model_3d_unet import UNet3D
from train import DiceCELoss


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Intentionally tiny run so data/model issues show up fast.
    _, _, train_loader, _, num_classes = build_train_val_loaders(
        patch_size=(64, 64, 64),
        num_patches_per_volume=1,
        batch_size=1,
        num_workers=0,
    )

    model = UNet3D(in_channels=1, num_classes=num_classes, base_ch=16).to(device)
    criterion = DiceCELoss(
        num_classes=num_classes,
        dice_weight=1.0,
        ce_weight=1.0,
        include_background=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=6, eta_min=1e-6
    )

    # Use one fixed batch for a controlled "loss should decrease" sanity check.
    batch_x, batch_y, case_ids = next(iter(train_loader))
    print(f"case_ids={list(case_ids)}")
    print(f"raw batch_x shape={tuple(batch_x.shape)}")
    print(f"raw batch_y shape={tuple(batch_y.shape)}")

    # data_loader returns (B_volume, Patches, C, D, H, W) and (B_volume, Patches, D, H, W)
    batch_x = batch_x.flatten(0, 1).to(device)
    batch_y = batch_y.flatten(0, 1).to(device)
    print(f"flattened batch_x shape={tuple(batch_x.shape)}")
    print(f"flattened batch_y shape={tuple(batch_y.shape)}")

    model.train()
    losses: list[float] = []
    num_steps = 3

    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        lr = optimizer.param_groups[0]["lr"]
        print(f"step={step:02d} loss={loss_value:.6f} lr={lr:.8f}")

    print(
        f"loss trend: first={losses[0]:.6f} last={losses[-1]:.6f} "
        f"({'decreased' if losses[-1] < losses[0] else 'not decreased'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
