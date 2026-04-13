"""Full baseline training script with validation and metric logging."""

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

from data_loader import (
    CTAPatchDataset,
    build_train_val_loaders,
    compute_rare_class_sampling_weights,
)
from data_utils import read_num_classes_from_labelmap
from model_3d_unet import UNet3D
from validation import validate_one_epoch


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
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--num-patches-per-volume", type=int, default=2)
    parser.add_argument(
        "--val-stride",
        type=int,
        nargs=3,
        default=None,
        help=(
            "Sliding-window stride (x y z) used only for full-volume validation. "
            "Default is patch_size // 2 per axis."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight for Dice term inside DiceCELoss.",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Weight for CE term inside DiceCELoss.",
    )
    parser.add_argument(
        "--ce-weight-min",
        type=float,
        default=None,
        help="Optional minimum clamp for CE class weights after normalization.",
    )
    parser.add_argument(
        "--ce-weight-max",
        type=float,
        default=None,
        help="Optional maximum clamp for CE class weights after normalization.",
    )
    parser.add_argument(
        "--rare-class-patch-prob",
        type=float,
        default=0.35,
        help=(
            "Probability that a train patch center is sampled from a rare foreground "
            "class (based on patient-level class presence). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--rare-class-weight-max",
        type=float,
        default=4.0,
        help=(
            "Maximum patient-presence inverse weight used for rare-class patch "
            "sampling. Higher increases focus on sparse classes."
        ),
    )
    return parser.parse_args()


def compute_class_weights(
    label_dir: Path,
    case_ids: list[str],
    num_classes: int,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> torch.Tensor:
    """Inverse-sqrt-frequency class weights for CrossEntropyLoss.

    Scans every training label volume, counts voxels per class, then returns
    ``w[c] = 1 / sqrt(freq[c])`` normalized so the weights sum to
    ``num_classes``. Classes never seen get weight 0.

    Optional clamp_min/clamp_max can tame extreme rare-class emphasis.
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

    # Normalize so weights sum to num_classes (keeps loss magnitude stable).
    w_sum = weights.sum()
    if w_sum > 0:
        weights *= num_classes / w_sum

    if clamp_min is not None or clamp_max is not None:
        lo = clamp_min if clamp_min is not None else -np.inf
        hi = clamp_max if clamp_max is not None else np.inf
        if lo > hi:
            raise ValueError(
                f"Invalid CE clamp range: min={clamp_min} > max={clamp_max}"
            )
        weights = np.clip(weights, lo, hi)

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


def _build_overfit_loaders(
    *,
    case_id: str,
    patch_size: tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    num_patches_per_volume: int,
    disable_augment: bool,
    rare_class_patch_prob: float,
    rare_class_weight_max: float,
) -> tuple[DataLoader, int]:
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
    rare_class_weights = compute_rare_class_sampling_weights(
        label_dir=label_dir,
        case_ids=[case_id],
        num_classes=num_classes,
        max_weight=rare_class_weight_max,
    )
    train_ds = CTAPatchDataset(
        case_ids=[case_id],
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=patch_size,
        preprocess_fn=None,
        do_augment=not disable_augment,
        num_classes=num_classes,
        num_patches=num_patches_per_volume,
        rare_class_prob=rare_class_patch_prob,
        rare_class_weights=rare_class_weights,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader, num_classes


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    patch_size = tuple(int(x) for x in args.patch_size)
    val_stride = (
        tuple(max(int(s), 1) for s in args.val_stride)
        if args.val_stride is not None
        else tuple(max(p // 2, 1) for p in patch_size)
    )
    label_dir = Path("training_data_resampled/labelsTr_topbrain_ct")
    image_dir = Path("training_data_resampled/imagesTr_topbrain_ct")

    if args.overfit_case_id:
        train_loader, num_classes = _build_overfit_loaders(
            case_id=args.overfit_case_id,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_patches_per_volume=args.num_patches_per_volume,
            disable_augment=args.overfit_disable_augment,
            rare_class_patch_prob=args.rare_class_patch_prob,
            rare_class_weight_max=args.rare_class_weight_max,
        )
        train_case_ids = [args.overfit_case_id]
        val_case_ids = [args.overfit_case_id]
        print(
            "overfit mode enabled: "
            f"case_id={args.overfit_case_id} "
            f"augment={'off' if args.overfit_disable_augment else 'on'}"
        )
    else:
        train_ds, val_case_ids, train_loader, num_classes = build_train_val_loaders(
            patch_size=patch_size,
            num_patches_per_volume=args.num_patches_per_volume,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rare_class_patch_prob=args.rare_class_patch_prob,
            rare_class_weight_max=args.rare_class_weight_max,
        )
        train_case_ids = train_ds.case_ids
    print(
        f"num_classes={num_classes} patch_size={patch_size} "
        f"val_stride={val_stride} train_batches={len(train_loader)} val_cases={len(val_case_ids)}"
    )
    print(
        "rare-class sampling: "
        f"prob={args.rare_class_patch_prob:.2f} "
        f"max_weight={args.rare_class_weight_max:.2f}"
    )

    ce_weights = compute_class_weights(
        label_dir,
        train_case_ids,
        num_classes,
        clamp_min=args.ce_weight_min,
        clamp_max=args.ce_weight_max,
    ).to(device)
    print(
        f"CE class weights (bg={ce_weights[0]:.3f}, min_fg={ce_weights[1:].min():.3f}, "
        f"max_fg={ce_weights[1:].max():.3f}, clamp_min={args.ce_weight_min}, "
        f"clamp_max={args.ce_weight_max})"
    )

    model = UNet3D(in_channels=1, num_classes=num_classes, base_ch=args.base_ch).to(device)
    criterion = DiceCELoss(
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
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
                *[f"val_dice_c{c:02d}" for c in range(num_classes)],
            ]
        )

    best_val_dice = -1.0
    final_weights_path = out_dir / "model_final_weights.pt"

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
            val_case_ids=val_case_ids,
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=patch_size,
            stride=val_stride,
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
                    *[f"{d:.6f}" for d in per_class_dice],
                ]
            )

        if val_mean_fg_dice > best_val_dice:
            best_val_dice = val_mean_fg_dice

        elapsed = time.time() - epoch_start
        print(
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={lr:.2e} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_mean_fg_dice={val_mean_fg_dice:.6f} "
            f"time={elapsed:.1f}s"
        )
        print(
            "per_class_dice: "
            + ", ".join(f"c{idx:02d}={d:.4f}" for idx, d in enumerate(per_class_dice))
        )

    torch.save(model.state_dict(), final_weights_path)
    print(f"\nTraining complete. Best val_mean_fg_dice={best_val_dice:.6f}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved final weights: {final_weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())