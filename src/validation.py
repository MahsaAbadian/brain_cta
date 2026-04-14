from __future__ import annotations

from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


def _axis_starts(dim: int, patch: int, step: int) -> list[int]:
    if patch > dim:
        raise ValueError(f"Patch size {patch} is larger than dimension {dim}.")
    starts = list(range(0, dim - patch + 1, step))
    if not starts or starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts


def _compute_per_class_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> tuple[list[float], list[int]]:
    """Per-class Dice using the nnU-Net / MONAI convention."""
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


def validate_one_epoch(
    model: nn.Module,
    val_case_ids: list[str],
    image_dir: Path,
    label_dir: Path,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    topbrain_case_callback: Callable[[str, np.ndarray, Path], None] | None = None,
) -> tuple[float, float, float, list[float]]:
    """Run one epoch of full-volume sliding-window validation."""
    model.eval()
    running_loss = 0.0
    n_steps = 0

    per_class_sum = [0.0] * num_classes
    per_class_count = [0] * num_classes
    gt_presence_count = [0] * num_classes
    pred_or_gt_presence_any = [False] * num_classes

    with torch.no_grad():
        for case_id in val_case_ids:
            image_path = image_dir / f"{case_id}_0000.nii.gz"
            label_path = label_dir / f"{case_id}.nii.gz"
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing validation files for {case_id}: {image_path} or {label_path}"
                )

            image = np.asanyarray(nib.load(str(image_path)).dataobj).astype(np.float32)
            target_np = np.asanyarray(nib.load(str(label_path)).dataobj).astype(np.int64)
            if image.shape != target_np.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: image={image.shape}, label={target_np.shape}"
                )

            xdim, ydim, zdim = image.shape
            px, py, pz = patch_size
            sx, sy, sz = stride

            xs = _axis_starts(xdim, px, sx)
            ys = _axis_starts(ydim, py, sy)
            zs = _axis_starts(zdim, pz, sz)

            logits_sum = np.zeros((num_classes, xdim, ydim, zdim), dtype=np.float32)
            counts = np.zeros((xdim, ydim, zdim), dtype=np.float32)

            # Sliding-window validation: infer on overlapping 3D patches,
            # then average logits in overlap regions to recover full-volume predictions.
            for x0 in xs:
                for y0 in ys:
                    for z0 in zs:
                        patch_img = image[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz]
                        patch_lbl = target_np[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz]
                        patch_x = torch.from_numpy(patch_img).unsqueeze(0).unsqueeze(0).to(device)
                        patch_y = torch.from_numpy(patch_lbl).unsqueeze(0).to(device)

                        logits = model(patch_x)
                        loss = criterion(logits, patch_y)
                        running_loss += float(loss.item())
                        n_steps += 1

                        logits_np = logits[0].detach().cpu().numpy()
                        logits_sum[:, x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz] += logits_np
                        counts[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz] += 1.0

            counts = np.maximum(counts, 1.0)
            avg_logits = logits_sum / counts[None, ...]
            pred_np = np.argmax(avg_logits, axis=0).astype(np.int64)
            if topbrain_case_callback is not None:
                topbrain_case_callback(case_id, pred_np, label_path)

            pred = torch.from_numpy(pred_np)
            target = torch.from_numpy(target_np)
            pred_or_gt_present_classes = set(np.unique(pred_np).tolist()) | set(
                np.unique(target_np).tolist()
            )
            for c in pred_or_gt_present_classes:
                if 0 <= int(c) < num_classes:
                    pred_or_gt_presence_any[int(c)] = True
            present_classes = np.unique(target_np)
            for c in present_classes:
                if 0 <= int(c) < num_classes:
                    gt_presence_count[int(c)] += 1
            dice_vals, valid_counts = _compute_per_class_dice(
                pred=pred, target=target, num_classes=num_classes
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
    fg_scores = [
        per_class_dice[c]
        for c in range(1, num_classes)
        if per_class_count[c] > 0 and pred_or_gt_presence_any[c]
    ]
    mean_fg_dice = float(sum(fg_scores) / max(len(fg_scores), 1))
    fg_all_cases_scores = [
        per_class_dice[c]
        for c in range(1, num_classes)
        if len(val_case_ids) > 0 and gt_presence_count[c] == len(val_case_ids)
    ]
    mean_fg_dice_all_cases_present = (
        float(sum(fg_all_cases_scores) / len(fg_all_cases_scores))
        if fg_all_cases_scores
        else float("nan")
    )
    return avg_loss, mean_fg_dice, mean_fg_dice_all_cases_present, per_class_dice
