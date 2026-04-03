
"""Utility functions for dataset splits, preprocessing, cropping, and resampling."""

import random
from pathlib import Path
import logging
import numpy as np
from scipy.ndimage import zoom


logger = logging.getLogger("data_split")

def extract_case_ids(image_dir: Path, label_dir: Path, image_suffix: str = "_0000.nii.gz", label_suffix: str = ".nii.gz") -> list[str]:
    """
    Extract case IDs from CT image directory. This also validates that both image and label files exist for each case.
    """
    case_ids = []
    if label_dir.is_dir():
        for label_file in label_dir.glob(f"*{label_suffix}"):
            case_id = label_file.name.replace(label_suffix, "")
            if(image_dir.is_dir()):
                image_file = image_dir / f"{case_id}{image_suffix}"
                if image_file.exists():
                    case_ids.append(case_id)
                else:
                    logger.error(f"Image file {image_file} not found for lable {label_file}")
                case_ids.append(case_id)
            else:
                logger.error(f"Image directory {image_dir} not found")
        return case_ids
    else:
        raise FileNotFoundError(f"Image directory {image_dir} not found")


def split_and_save(case_ids: list[str], split_ratio: float = 0.8, output_dir: Path = Path("training_data/split")) -> tuple[list[str], list[str]]:
    """
    Split case IDs into training and validation sets.
    """
    random.shuffle(case_ids)
    split_index = int(len(case_ids) * split_ratio)
    train_ids, val_ids = case_ids[:split_index], case_ids[split_index:]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_cases.txt").write_text("\n".join(train_ids) + "\n")
    (output_dir / "val_cases.txt").write_text("\n".join(val_ids) + "\n")
    return train_ids, val_ids

def load_split_ids(output_dir: Path) -> tuple[list[str], list[str]]:
    """
    Load train and validation case IDs from a text file.
    """
    if output_dir.is_dir():
        train_ids = (output_dir / "train_cases.txt").read_text().splitlines()
        val_ids = (output_dir / "val_cases.txt").read_text().splitlines()
        return train_ids, val_ids
    else:
        raise FileNotFoundError(f"Output directory {output_dir} not found")

if __name__ == "__main__":
    image_dir = Path("training_data/imagesTr_topbrain_ct")
    label_dir = Path("training_data/labelsTr_topbrain_ct")
    case_ids = extract_case_ids(image_dir, label_dir)
    train_ids, val_ids = split_and_save(case_ids, split_ratio=0.8, output_dir=Path("training_data/split"))
    print("Count:", len(case_ids))
    print("First few:", case_ids[:5])
    print("Train count:", len(train_ids))
    print("Val count:", len(val_ids))
    print("First few train:", train_ids[:5])
    print("First few val:", val_ids[:5])

def preprocess_ct(x, low=-100.0, high=400.0):
    """Clip CT intensities to a vessel-focused window and normalize to [0, 1]."""
    x = np.clip(x, low, high)
    x = (x - low) / (high - low)
    return x.astype(np.float32)


def read_num_classes_from_labelmap(labelmap_path: Path) -> int:
    """Read ITK-SNAP label map and return max_label + 1."""
    max_idx = 0
    for line in labelmap_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first_token = line.split()[0]
        try:
            idx = int(first_token)
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


def random_crop_3d(
    image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Random crop for training. Arrays are expected in (x, y, z)."""
    px, py, pz = patch_size
    sx = random.randint(0, image.shape[0] - px)
    sy = random.randint(0, image.shape[1] - py)
    sz = random.randint(0, image.shape[2] - pz)
    return (
        image[sx : sx + px, sy : sy + py, sz : sz + pz],
        label[sx : sx + px, sy : sy + py, sz : sz + pz],
    )


def center_crop_3d(
    image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Center crop for validation. Arrays are expected in (x, y, z)."""
    px, py, pz = patch_size
    sx = (image.shape[0] - px) // 2
    sy = (image.shape[1] - py) // 2
    sz = (image.shape[2] - pz) // 2
    return (
        image[sx : sx + px, sy : sy + py, sz : sz + pz],
        label[sx : sx + px, sy : sy + py, sz : sz + pz],
    )


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Extract voxel spacing in mm from a NIfTI affine."""
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).astype(np.float32)

def resample_to_spacing(
    volume: np.ndarray,
    current_spacing: np.ndarray,
    target_spacing=(0.6, 0.6, 0.6),
    is_label=False,
) -> np.ndarray:
    """Resample an image or label volume to a target spacing."""
    target_spacing = np.array(target_spacing, dtype=np.float32)
    zoom_factors = current_spacing / target_spacing
    order = 0 if is_label else 1  # nearest for labels, linear for images
    return zoom(volume, zoom=zoom_factors, order=order)