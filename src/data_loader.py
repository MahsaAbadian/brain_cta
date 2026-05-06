from __future__ import annotations

"""Patch-based dataset and dataloader utilities for training on preprocessed data."""

import json
import random
from pathlib import Path
from typing import Callable, Literal, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import (
    extract_case_ids,
    load_split_ids,
    split_and_save,
    read_num_classes_from_labelmap,
)


def load_fold_from_splits_json(
    splits_json: Path, fold: int
) -> tuple[list[str], list[str]]:
    """Load a single fold from an nnUNet-style splits_final.json.

    The file must be a JSON list of ``{"train": [...], "val": [...]}`` dicts.
    Returns sorted, deduplicated train/val id lists. Validates disjointness.
    """
    if not splits_json.is_file():
        raise FileNotFoundError(f"splits_json not found: {splits_json}")
    folds = json.loads(splits_json.read_text())
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"splits_json is empty or not a list: {splits_json}")
    if fold < 0 or fold >= len(folds):
        raise IndexError(
            f"Requested fold {fold} is out of range; splits file has "
            f"{len(folds)} folds ({splits_json})."
        )
    entry = folds[fold]
    if not isinstance(entry, dict) or set(entry) != {"train", "val"}:
        raise ValueError(
            f"Fold {fold} in {splits_json} must be a dict with keys 'train' and 'val'."
        )
    train_ids = sorted({cid for cid in entry["train"] if cid})
    val_ids = sorted({cid for cid in entry["val"] if cid})
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            f"Fold {fold} has overlapping case ids: {sorted(overlap)}"
        )
    return train_ids, val_ids


def random_flip_3d(img: np.ndarray, lbl: np.ndarray, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Random axis flips, same transform on image and label.

    Axis 0 (left/right) is intentionally excluded: TopBrain volumes are stored
    in LPS orientation, so axis 0 separates anatomically right- vs left-sided
    structures (R-ICA / L-ICA, R-M1 / L-M1, ...). Flipping that axis without
    also swapping the paired R/L class labels would teach the model
    contradictory side assignments and destroy R/L discrimination, so we only
    flip the A/P (axis 1) and I/S (axis 2) axes here.
    """
    for axis in (1, 2):
        if random.random() < p:
            img = np.flip(img, axis=axis).copy()
            lbl = np.flip(lbl, axis=axis).copy()
    return img, lbl


def sample_patches_option_d(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple[int, int, int],
    num_patches: int = 4,
    fg_prob: float = 0.6,
    min_dist: float | None = None,
    rare_class_prob: float = 0.0,
    rare_class_weights: np.ndarray | None = None,
    return_starts: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]] | tuple[
    list[np.ndarray], list[np.ndarray], list[tuple[int, int, int]]
]:
    """
    Sample multiple patches from a single volume, ensuring they are not too close.
    Uses Foreground-Aware (Option B) + Minimum Center Distance (Option D).
    """
    px, py, pz = patch_size
    x, y, z = image.shape

    if min_dist is None:
        min_dist = max(patch_size) * 0.5  # half the max patch dimension

    # Pre-find foreground indices for Option B
    fg_indices = np.argwhere(label > 0)
    has_fg = len(fg_indices) > 0
    class_to_indices: dict[int, np.ndarray] = {}
    present_fg_classes = np.array([], dtype=np.int64)
    if has_fg and rare_class_weights is not None and rare_class_prob > 0.0:
        present_fg_classes = np.unique(label[label > 0]).astype(np.int64)
        class_to_indices = {
            int(c): np.argwhere(label == int(c)) for c in present_fg_classes.tolist()
        }

    chosen_centers = []
    out_images = []
    out_labels = []
    out_starts: list[tuple[int, int, int]] = []

    max_attempts = 50

    for _ in range(num_patches):
        patch_found = False
        for attempt in range(max_attempts):
            # 1) Decide foreground vs background/random
            is_fg = has_fg and (random.random() < fg_prob)
            use_rare_class = (
                has_fg
                and len(present_fg_classes) > 0
                and random.random() < rare_class_prob
            )

            if use_rare_class:
                # Sample a class inversely to patient-level presence and then
                # pick a voxel from that class. This increases coverage of
                # sparse classes without removing global foreground sampling.
                cls_weights = rare_class_weights[present_fg_classes]
                if np.sum(cls_weights) > 0:
                    probs = cls_weights / np.sum(cls_weights)
                    target_class = int(np.random.choice(present_fg_classes, p=probs))
                    idx_pool = class_to_indices.get(target_class, fg_indices)
                else:
                    idx_pool = fg_indices
                idx = random.randint(0, len(idx_pool) - 1)
                cx, cy, cz = idx_pool[idx]
                sx = cx - px // 2
                sy = cy - py // 2
                sz = cz - pz // 2

                sx = max(0, min(sx, x - px))
                sy = max(0, min(sy, y - py))
                sz = max(0, min(sz, z - pz))
            elif is_fg:
                # Pick a random foreground voxel as center
                idx = random.randint(0, len(fg_indices) - 1)
                cx, cy, cz = fg_indices[idx]

                # Shift center to top-left corner of the patch, ensuring it stays within bounds
                sx = cx - px // 2
                sy = cy - py // 2
                sz = cz - pz // 2

                sx = max(0, min(sx, x - px))
                sy = max(0, min(sy, y - py))
                sz = max(0, min(sz, z - pz))
            else:
                # Pure random
                sx = random.randint(0, x - px)
                sy = random.randint(0, y - py)
                sz = random.randint(0, z - pz)

            center = (sx + px / 2, sy + py / 2, sz + pz / 2)

            # 2) Check distance against already chosen centers
            too_close = False
            for cc in chosen_centers:
                dist = np.sqrt((center[0] - cc[0])**2 + (center[1] - cc[1])**2 + (center[2] - cc[2])**2)
                if dist < min_dist:
                    too_close = True
                    break

            if not too_close:
                chosen_centers.append(center)
                out_images.append(image[sx : sx + px, sy : sy + py, sz : sz + pz])
                out_labels.append(label[sx : sx + px, sy : sy + py, sz : sz + pz])
                out_starts.append((sx, sy, sz))
                patch_found = True
                break

        # If we failed to find a distant patch after max_attempts, just accept the last one
        if not patch_found:
            chosen_centers.append(center)
            out_images.append(image[sx : sx + px, sy : sy + py, sz : sz + pz])
            out_labels.append(label[sx : sx + px, sy : sy + py, sz : sz + pz])
            out_starts.append((sx, sy, sz))

    if return_starts:
        return out_images, out_labels, out_starts
    return out_images, out_labels


def compute_rare_class_sampling_weights(
    label_dir: Path,
    case_ids: Sequence[str],
    num_classes: int,
    max_weight: float = 4.0,
    mode: Literal["presence", "voxel", "hybrid"] = "hybrid",
) -> np.ndarray:
    """Compute rare-class sampling weights for foreground classes.

    Modes:
      - presence: inverse patient-level class presence rate.
      - voxel: inverse sqrt of total voxel frequency across train labels.
      - hybrid: geometric mean of presence and voxel weights.

    Foreground weights are normalized to mean 1 and clipped to [0.1, max_weight].
    """
    if max_weight <= 0:
        raise ValueError(f"max_weight must be > 0, got {max_weight}")
    if mode not in {"presence", "voxel", "hybrid"}:
        raise ValueError(f"Unknown rare class mode: {mode}")

    if len(case_ids) == 0:
        return np.ones(num_classes, dtype=np.float32)

    present_counts = np.zeros(num_classes, dtype=np.float64)
    voxel_counts = np.zeros(num_classes, dtype=np.float64)
    for cid in case_ids:
        lbl_path = label_dir / f"{cid}.nii.gz"
        lbl = np.asanyarray(nib.load(str(lbl_path)).dataobj).astype(np.int64)
        present_classes = np.unique(lbl)
        present_classes = present_classes[(present_classes >= 0) & (present_classes < num_classes)]
        present_counts[present_classes] += 1
        vals, counts = np.unique(lbl, return_counts=True)
        vals = vals.astype(np.int64)
        valid = (vals >= 0) & (vals < num_classes)
        voxel_counts[vals[valid]] += counts[valid]

    n_cases = float(len(case_ids))
    rates = present_counts / n_cases
    presence_weights = np.ones(num_classes, dtype=np.float64)
    for c in range(1, num_classes):
        if rates[c] > 0:
            presence_weights[c] = 1.0 / rates[c]
        else:
            presence_weights[c] = max_weight

    voxel_weights = np.ones(num_classes, dtype=np.float64)
    total_voxels = float(voxel_counts.sum())
    voxel_freq = voxel_counts / max(total_voxels, 1.0)
    for c in range(1, num_classes):
        if voxel_freq[c] > 0:
            voxel_weights[c] = 1.0 / np.sqrt(voxel_freq[c])
        else:
            voxel_weights[c] = max_weight

    if mode == "presence":
        weights = presence_weights
    elif mode == "voxel":
        weights = voxel_weights
    else:
        weights = np.sqrt(np.maximum(presence_weights * voxel_weights, 1e-12))

    fg = weights[1:]
    if fg.size > 0 and fg.mean() > 0:
        fg = fg / fg.mean()
    fg = np.clip(fg, 0.1, max_weight)
    weights[1:] = fg
    weights[0] = 0.0
    return weights.astype(np.float32)


class CTAPatchDataset(Dataset):
    """
    3D CTA patch dataset.

    Returns:
      image_tensor: shape (1, D, H, W), dtype float32
      label_tensor: shape (D, H, W), dtype int64
    """

    def __init__(
        self,
        *,
        case_ids: Sequence[str],
        image_dir: Path,
        label_dir: Path,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        do_augment: bool = True,
        num_classes: int = 41,
        num_patches: int = 4,
        rare_class_prob: float = 0.0,
        rare_class_weights: np.ndarray | None = None,
        target_skeleton_dir: Path | None = None,
    ) -> None:
        self.case_ids = list(case_ids)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.preprocess_fn = preprocess_fn
        self.do_augment = do_augment
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.rare_class_prob = float(np.clip(rare_class_prob, 0.0, 1.0))
        self.rare_class_weights = rare_class_weights
        self.target_skeleton_dir = target_skeleton_dir

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, str
    ]:
        case_id = self.case_ids[idx]
        image_path = self.image_dir / f"{case_id}_0000.nii.gz"
        label_path = self.label_dir / f"{case_id}.nii.gz"

        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        image = np.asanyarray(image_nii.dataobj).astype(np.float32)
        label = np.asanyarray(label_nii.dataobj).astype(np.int64)

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: image={image.shape}, label={label.shape}"
            )

        # CT normalization is now expected to happen offline in
        # src/preprocess_resample.py, so this hook is optional.
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)

        for dim, p in zip(image.shape, self.patch_size):
            if p > dim:
                raise ValueError(
                    f"Patch size {self.patch_size} is larger than volume shape {image.shape} for {case_id}"
                )

        sample_output = sample_patches_option_d(
            image,
            label,
            self.patch_size,
            num_patches=self.num_patches,
            rare_class_prob=self.rare_class_prob,
            rare_class_weights=self.rare_class_weights,
            return_starts=self.target_skeleton_dir is not None,
        )
        if self.target_skeleton_dir is None:
            images, labels = sample_output  # type: ignore[misc]
            starts: list[tuple[int, int, int]] = []
        else:
            images, labels, starts = sample_output  # type: ignore[misc]
        out_img_tensors = []
        out_lbl_tensors = []
        out_skel_tensors = []
        skel_volume: np.ndarray | None = None
        if self.target_skeleton_dir is not None:
            skel_path = self.target_skeleton_dir / f"{case_id}.npz"
            if not skel_path.is_file():
                raise FileNotFoundError(
                    f"Missing target skeleton file for {case_id}: {skel_path}"
                )
            with np.load(skel_path) as data:
                if "skel" not in data:
                    raise KeyError(f"Expected key 'skel' in {skel_path}")
                skel_volume = data["skel"].astype(np.float32)
            if skel_volume.ndim != 4:
                raise ValueError(
                    f"Invalid skeleton shape for {case_id}: {skel_volume.shape}, expected (C, D, H, W)"
                )
            if skel_volume.shape[1:] != image.shape:
                raise ValueError(
                    f"Skeleton/image shape mismatch for {case_id}: "
                    f"skel={skel_volume.shape[1:]}, image={image.shape}"
                )
        for patch_idx, (img, lbl) in enumerate(zip(images, labels)):
            skel_patch = None
            if skel_volume is not None:
                sx, sy, sz = starts[patch_idx]
                px, py, pz = self.patch_size
                skel_patch = skel_volume[:, sx : sx + px, sy : sy + py, sz : sz + pz]
            if self.do_augment:
                # Skip axis 0 (left/right) on purpose: volumes are LPS, so a
                # raw L/R flip without swapping paired class labels (R-ICA <->
                # L-ICA, R-M1 <-> L-M1, etc.) trains the head to predict the
                # right-sided class on the left side and vice versa. We only
                # flip A/P (axis 1) and I/S (axis 2). Skeleton patches are
                # (C, D, H, W), so the spatial axes are offset by one.
                for axis in (1, 2):
                    if random.random() < 0.5:
                        img = np.flip(img, axis=axis).copy()
                        lbl = np.flip(lbl, axis=axis).copy()
                        if skel_patch is not None:
                            skel_patch = np.flip(skel_patch, axis=axis + 1).copy()
            out_img_tensors.append(torch.from_numpy(img).float().unsqueeze(0))
            out_lbl_tensors.append(torch.from_numpy(lbl).long())
            if skel_patch is not None:
                out_skel_tensors.append(torch.from_numpy(skel_patch).float())

        image_t = torch.stack(out_img_tensors)  # (num_patches, 1, D, H, W)
        label_t = torch.stack(out_lbl_tensors)  # (num_patches, D, H, W)

        if label_t.min() < 0 or label_t.max() >= self.num_classes:
            raise ValueError(
                f"Label out of range for {case_id}: min={label_t.min().item()}, max={label_t.max().item()}, "
                f"num_classes={self.num_classes}"
            )

        if self.target_skeleton_dir is not None:
            skel_t = torch.stack(out_skel_tensors)  # (num_patches, C, D, H, W)
            return image_t, label_t, skel_t, case_id
        return image_t, label_t, case_id


def build_train_val_loaders(
    *,
    split_dir: Path = Path("training_data_resampled/split"),
    image_dir: Path = Path("training_data_resampled/imagesTr_topbrain_ct"),
    label_dir: Path = Path("training_data_resampled/labelsTr_topbrain_ct"),
    labelmap_path: Path = Path("training_data_resampled/itksnap_labelmap_txt/labelmap_topbrain_ct.txt"),
    split_ratio: float = 0.8,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    num_patches_per_volume: int = 4,
    batch_size: int = 1,
    num_workers: int = 0,
    rare_class_patch_prob: float = 0.35,
    rare_class_weight_max: float = 4.0,
    rare_class_mode: Literal["presence", "voxel", "hybrid"] = "hybrid",
    target_skeleton_dir: Path | None = None,
    splits_json: Path | None = None,
    fold: int = 0,
) -> tuple[CTAPatchDataset, list[str], DataLoader, int]:
    """
    Build CTA train dataset + loader and return validation case IDs.
    Reuses split + preprocessing utilities from data_utils.py.

    Split resolution order:
      1. If ``splits_json`` is given and exists: use fold ``fold`` from that JSON.
      2. Else if ``split_dir`` has a ``splits_final.json``: use fold ``fold``
         from it.
      3. Else if ``train_cases.txt`` / ``val_cases.txt`` exist: use them
         (legacy single-fold mode).
      4. Else: run a ratio split on every case found in ``image_dir``/``label_dir``.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    candidate_json = splits_json if splits_json is not None else split_dir / "splits_final.json"
    train_split = split_dir / "train_cases.txt"
    val_split = split_dir / "val_cases.txt"

    if candidate_json is not None and candidate_json.is_file():
        train_ids, val_ids = load_fold_from_splits_json(candidate_json, fold=fold)
    elif train_split.is_file() and val_split.is_file():
        train_ids, val_ids = load_split_ids(split_dir)
        # Safety: dedupe possibly duplicated IDs from older split generation.
        train_ids = sorted(set(train_ids))
        val_ids = sorted(set(val_ids))
        # Keep val disjoint from train if there is accidental overlap.
        val_ids = [cid for cid in val_ids if cid not in set(train_ids)]
        # If loaded split is heavily imbalanced after cleanup, rebuild in-memory.
        all_ids = sorted(set(train_ids).union(set(val_ids)))
        if all_ids:
            val_ratio = len(val_ids) / len(all_ids)
            if val_ratio < 0.15 or val_ratio > 0.35:
                rng = random.Random(42)
                shuffled = all_ids[:]
                rng.shuffle(shuffled)
                split_index = int(len(shuffled) * split_ratio)
                train_ids = shuffled[:split_index]
                val_ids = shuffled[split_index:]
    else:
        case_ids = extract_case_ids(image_dir=image_dir, label_dir=label_dir)
        # extract_case_ids currently appends duplicates; dedupe for safe split.
        case_ids = sorted(set(case_ids))
        train_ids, val_ids = split_and_save(
            case_ids, split_ratio=split_ratio, output_dir=split_dir
        )

    num_classes = read_num_classes_from_labelmap(labelmap_path)
    rare_class_weights = compute_rare_class_sampling_weights(
        label_dir=label_dir,
        case_ids=train_ids,
        num_classes=num_classes,
        max_weight=rare_class_weight_max,
        mode=rare_class_mode,
    )

    train_ds = CTAPatchDataset(
        case_ids=train_ids,
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=patch_size,
        preprocess_fn=None,
        do_augment=True,
        num_classes=num_classes,
        num_patches=num_patches_per_volume,
        rare_class_prob=rare_class_patch_prob,
        rare_class_weights=rare_class_weights,
        target_skeleton_dir=target_skeleton_dir,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_ds, val_ids, train_loader, num_classes


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_ds, val_case_ids, train_loader, num_classes = build_train_val_loaders(
        patch_size=(128, 128, 128),
        batch_size=1,
        num_workers=0,
    )

    print(f"num_classes={num_classes}")
    print(f"train cases={len(train_ds)}, val cases={len(val_case_ids)}")

    sample_x, sample_y, sample_case = train_ds[0]
    print("\nDataset sample check:")
    print(f"  case_id={sample_case}")
    print(f"  image shape={tuple(sample_x.shape)}, dtype={sample_x.dtype}")
    print(f"  label shape={tuple(sample_y.shape)}, dtype={sample_y.dtype}")
    print(
        f"  label range=({sample_y.min().item()}, {sample_y.max().item()})"
    )

    batch_x, batch_y, batch_case_ids = next(iter(train_loader))
    print("\nTrain loader batch check:")
    print(f"  image batch shape={tuple(batch_x.shape)}, dtype={batch_x.dtype}")
    print(f"  label batch shape={tuple(batch_y.shape)}, dtype={batch_y.dtype}")
    print(f"  case_ids={list(batch_case_ids)}")

    fake_logits = torch.randn(
        batch_x.shape[0] * batch_x.shape[1],
        num_classes,
        batch_x.shape[3],
        batch_x.shape[4],
        batch_x.shape[5],
    )
    print("\nPlaceholder model pass shape check:")
    print(f"  logits shape={tuple(fake_logits.shape)}")
