from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from data_loader import build_train_val_loaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "training_data_resampled"
IMAGE_DIR = DATA_ROOT / "imagesTr_topbrain_ct"
LABEL_DIR = DATA_ROOT / "labelsTr_topbrain_ct"
SPLIT_DIR = DATA_ROOT / "split"


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.slow
@pytest.mark.integration
def test_resampled_dataset_split_integrity() -> None:
    if not SPLIT_DIR.exists():
        pytest.skip("Resampled split directory not found.")

    train_raw = _read_ids(SPLIT_DIR / "train_cases.txt")
    val_raw = _read_ids(SPLIT_DIR / "val_cases.txt")

    train = sorted(set(train_raw))
    val = sorted(set(val_raw))

    assert len(train) == len(train_raw), "Duplicate case IDs in train split."
    assert len(val) == len(val_raw), "Duplicate case IDs in val split."
    assert set(train).isdisjoint(set(val)), "Train/val overlap detected."

    all_ids = sorted(set(train) | set(val))
    assert all_ids, "No IDs found in split files."
    val_ratio = len(val) / len(all_ids)
    assert 0.15 <= val_ratio <= 0.35, f"Unexpected val ratio: {val_ratio:.2f}"


@pytest.mark.slow
@pytest.mark.integration
def test_resampled_dataset_files_spacing_and_ranges() -> None:
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        pytest.skip("Resampled CT image/label directories not found.")

    case_ids = sorted(
        p.name.replace("_0000.nii.gz", "")
        for p in IMAGE_DIR.glob("*_0000.nii.gz")
    )
    assert case_ids, "No resampled CT image files found."

    spacing_tol = 0.02
    target_spacing = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    for cid in case_ids:
        img_path = IMAGE_DIR / f"{cid}_0000.nii.gz"
        lbl_path = LABEL_DIR / f"{cid}.nii.gz"
        assert img_path.exists(), f"Missing image: {img_path}"
        assert lbl_path.exists(), f"Missing label: {lbl_path}"

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))
        img = np.asanyarray(img_nii.dataobj, dtype=np.float32)
        lbl = np.asanyarray(lbl_nii.dataobj)

        assert img.shape == lbl.shape, f"Shape mismatch for {cid}: {img.shape} vs {lbl.shape}"

        spacing = np.sqrt(np.sum(img_nii.affine[:3, :3] ** 2, axis=0))
        assert np.all(np.abs(spacing - target_spacing) <= spacing_tol), (
            f"Spacing mismatch for {cid}: {spacing}"
        )

        assert float(img.min()) >= -1e-4, f"Image min out of range for {cid}: {img.min()}"
        assert float(img.max()) <= 1.0 + 1e-4, f"Image max out of range for {cid}: {img.max()}"

        lmin, lmax = int(lbl.min()), int(lbl.max())
        assert lmin >= 0 and lmax <= 40, f"Label range out of bounds for {cid}: [{lmin}, {lmax}]"


@pytest.mark.integration
def test_data_loader_shapes_and_label_bounds() -> None:
    if not DATA_ROOT.exists():
        pytest.skip("Resampled data root not found.")

    train_ds, val_case_ids, train_loader, num_classes = build_train_val_loaders(
        patch_size=(96, 96, 96),
        num_patches_per_volume=4,
        batch_size=1,
        num_workers=0,
    )

    assert num_classes == 41
    assert len(train_ds) > 0
    assert len(val_case_ids) > 0

    sample_x, sample_y, _ = train_ds[0]
    assert tuple(sample_x.shape) == (4, 1, 96, 96, 96)
    assert tuple(sample_y.shape) == (4, 96, 96, 96)
    assert sample_x.dtype == torch.float32
    assert sample_y.dtype == torch.int64
    assert int(sample_y.min()) >= 0
    assert int(sample_y.max()) < num_classes

    batch_x, batch_y, _ = next(iter(train_loader))
    assert tuple(batch_x.shape) == (1, 4, 1, 96, 96, 96)
    assert tuple(batch_y.shape) == (1, 4, 96, 96, 96)
