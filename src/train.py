"""Full baseline training script with validation, checkpoints, and metric logging."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import CTAPatchDataset, build_train_val_loaders
from data_utils import read_num_classes_from_labelmap
from model_3d_unet import UNet3D


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = True,
        eps: float = 1e-6,
        ce_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline 3D U-Net on resampled CTA data.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--num-patches-per-volume", type=int, default=2)
    parser.add_argument("--num-val-patches-per-volume", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument(
        "--overfit-case-id",
        type=str,
        default=None,
        help=(
            "Optional debug mode: use this single case ID for both train and val "
            "(intentional leakage) to verify the model can overfit."
        ),
    )
    parser.add_argument(
        "--overfit-disable-augment",
        action="store_true",
        help=(
            "Only used with --overfit-case-id. Disable training-time random flips "
            "to make one-case memorization easier and debugging cleaner."
        ),
    )
    return parser.parse_args()


def compute_class_weights(
    label_dir: Path,
    case_ids: list[str],
    num_classes: int,
) -> torch.Tensor:
    """Inverse-sqrt-frequency class weights for CrossEntropyLoss.

    Scans every training label volume, counts voxels per class, then returns
    ``w[c] = 1 / sqrt(freq[c])`` normalised so the weights sum to
    ``num_classes``.  Classes never seen get weight 0.
    """
    import nibabel as nib

    counts = np.zeros(num_classes, dtype=np.float64)
    for cid in case_ids:
        lbl_path = label_dir / f"{cid}.nii.gz"
        lbl = np.asanyarray(nib.load(str(lbl_path)).dataobj).astype(np.int64)
        for val, cnt in zip(*np.unique(lbl, return_counts=True)):
            if 0 <= val < num_classes:
                counts[val] += cnt

    total = counts.sum()
    freq = counts / max(total, 1.0)

    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = freq > 0
    weights[nonzero] = 1.0 / np.sqrt(freq[nonzero])

    # Normalise so weights sum to num_classes (keeps loss magnitude stable).
    w_sum = weights.sum()
    if w_sum > 0:
        weights *= num_classes / w_sum

    return torch.tensor(weights, dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_loader_batch(
    batch_x: torch.Tensor, batch_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # data_loader returns:
    # image: (B_volume, Patches, C, D, H, W)
    # label: (B_volume, Patches, D, H, W)
    x = batch_x.flatten(0, 1)
    y = batch_y.flatten(0, 1)
    return x, y


def _compute_per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> tuple[list[float], list[int]]:
    """Per-class Dice using the nnU-Net / MONAI / challenge-leaderboard convention.

    Edge cases (no smoothing epsilon — exact set comparison):
      * Both empty  → Dice = 1.0  (perfect agreement on absence)
      * One empty   → Dice = 0.0  (false positive or false negative)
      * Both present → 2|P∩G| / (|P|+|G|)
    Every class is always counted as a valid observation.
    """
    dice_vals: list[float] = []
    valid_counts: list[int] = []
    for c in range(num_classes):
        pred_c = pred == c
        tgt_c = target == c
        pred_count = pred_c.sum().item()
        tgt_count = tgt_c.sum().item()
        if pred_count == 0 and tgt_count == 0:
            dice_vals.append(1.0)
            valid_counts.append(1)
            continue
        denom = pred_count + tgt_count
        if denom == 0:
            dice_vals.append(1.0)
            valid_counts.append(1)
            continue
        inter = (pred_c & tgt_c).sum().item()
        dice = (2.0 * inter) / denom
        dice_vals.append(float(dice))
        valid_counts.append(1)
    return dice_vals, valid_counts


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_steps = 0
    for batch_x, batch_y, _ in loader:
        batch_x, batch_y = _flatten_loader_batch(batch_x, batch_y)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_steps += 1
    return running_loss / max(n_steps, 1)


def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[float]]:
    model.eval()
    running_loss = 0.0
    n_steps = 0

    per_class_sum = [0.0] * num_classes
    per_class_count = [0] * num_classes

    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x, batch_y = _flatten_loader_batch(batch_x, batch_y)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            running_loss += float(loss.item())
            n_steps += 1

            pred = torch.argmax(logits, dim=1)
            dice_vals, valid_counts = _compute_per_class_dice(
                pred=pred, target=batch_y, num_classes=num_classes
            )
            for c in range(num_classes):
                if valid_counts[c]:
                    per_class_sum[c] += dice_vals[c]
                    per_class_count[c] += 1

    avg_loss = running_loss / max(n_steps, 1)
    per_class_dice = [
        (per_class_sum[c] / per_class_count[c]) if per_class_count[c] > 0 else 0.0
        for c in range(num_classes)
    ]

    # Mean foreground Dice (exclude background class 0).
    fg_scores = [
        per_class_dice[c]
        for c in range(1, num_classes)
        if per_class_count[c] > 0
    ]
    mean_fg_dice = float(sum(fg_scores) / max(len(fg_scores), 1))

    return avg_loss, mean_fg_dice, per_class_dice


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_val_dice: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
        },
        path,
    )


def _build_overfit_loaders(
    *,
    case_id: str,
    patch_size: tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    num_patches_per_volume: int,
    num_val_patches_per_volume: int,
    disable_augment: bool,
) -> tuple[DataLoader, DataLoader, int]:
    image_dir = Path("training_data_resampled/imagesTr_topbrain_ct")
    label_dir = Path("training_data_resampled/labelsTr_topbrain_ct")
    labelmap_path = Path("training_data_resampled/itksnap_labelmap_txt/labelmap_topbrain_ct.txt")

    image_path = image_dir / f"{case_id}_0000.nii.gz"
    label_path = label_dir / f"{case_id}.nii.gz"
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Overfit case not found in resampled data: case_id={case_id} "
            f"(expected {image_path} and {label_path})"
        )

    num_classes = read_num_classes_from_labelmap(labelmap_path)
    train_ds = CTAPatchDataset(
        case_ids=[case_id],
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=patch_size,
        mode="train",
        preprocess_fn=None,
        do_augment=not disable_augment,
        num_classes=num_classes,
        num_patches=num_patches_per_volume,
    )
    val_ds = CTAPatchDataset(
        case_ids=[case_id],
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=patch_size,
        mode="val",
        preprocess_fn=None,
        do_augment=False,
        num_classes=num_classes,
        num_patches=num_val_patches_per_volume,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, num_classes


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    patch_size = tuple(int(x) for x in args.patch_size)
    label_dir = Path("training_data_resampled/labelsTr_topbrain_ct")

    if args.overfit_case_id:
        train_loader, val_loader, num_classes = _build_overfit_loaders(
            case_id=args.overfit_case_id,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_patches_per_volume=args.num_patches_per_volume,
            num_val_patches_per_volume=args.num_val_patches_per_volume,
            disable_augment=args.overfit_disable_augment,
        )
        train_case_ids = [args.overfit_case_id]
        print(
            "overfit mode enabled: "
            f"case_id={args.overfit_case_id} "
            f"augment={'off' if args.overfit_disable_augment else 'on'}"
        )
    else:
        train_ds, _, train_loader, val_loader, num_classes = build_train_val_loaders(
            patch_size=patch_size,
            num_patches_per_volume=args.num_patches_per_volume,
            num_val_patches_per_volume=args.num_val_patches_per_volume,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_case_ids = train_ds.case_ids
    print(
        f"num_classes={num_classes} patch_size={patch_size} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    ce_weights = compute_class_weights(label_dir, train_case_ids, num_classes).to(device)
    print(f"CE class weights (bg={ce_weights[0]:.3f}, min_fg={ce_weights[1:].min():.3f}, "
          f"max_fg={ce_weights[1:].max():.3f})")

    model = UNet3D(in_channels=1, num_classes=num_classes, base_ch=args.base_ch).to(device)
    criterion = DiceCELoss(
        num_classes=num_classes,
        dice_weight=1.0,
        ce_weight=1.0,
        include_background=False,
        ce_class_weights=ce_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6
    )

    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
                "val_mean_fg_dice",
            ]
        )

    best_val_dice = -1.0
    best_path = out_dir / "checkpoint_best.pt"
    latest_path = out_dir / "checkpoint_latest.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_mean_fg_dice, per_class_dice = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        with metrics_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{lr:.10f}",
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_mean_fg_dice:.6f}",
                ]
            )

        _save_checkpoint(
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_dice=best_val_dice,
        )
        if args.save_every > 0 and (epoch % args.save_every == 0):
            _save_checkpoint(
                path=out_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_dice=best_val_dice,
            )
        if val_mean_fg_dice > best_val_dice:
            best_val_dice = val_mean_fg_dice
            _save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_dice=best_val_dice,
            )

        elapsed = time.time() - epoch_start
        print(
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={lr:.2e} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_mean_fg_dice={val_mean_fg_dice:.6f} "
            f"time={elapsed:.1f}s"
        )
        if epoch == 1:
            print(
                "per_class_dice sample (classes 0..5): "
                + ", ".join(f"{d:.4f}" for d in per_class_dice[:6])
            )

    print(f"\nTraining complete. Best val_mean_fg_dice={best_val_dice:.6f}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved checkpoints: {latest_path}, {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())