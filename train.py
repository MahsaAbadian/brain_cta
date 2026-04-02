import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import build_train_val_loaders
from model3dunet import UNet3D


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0, include_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, D, H, W), target: (N, D, H, W)
        ce_loss = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]

        dims = (0, 2, 3, 4)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

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