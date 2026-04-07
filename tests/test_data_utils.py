from __future__ import annotations

from pathlib import Path

import numpy as np

from data_utils import (
    center_crop_3d,
    load_split_ids,
    preprocess_ct,
    read_num_classes_from_labelmap,
    resample_to_spacing,
    spacing_from_affine,
    split_and_save,
)


def test_preprocess_ct_clips_and_normalizes() -> None:
    x = np.array([-500.0, -100.0, 150.0, 400.0, 1000.0], dtype=np.float32)
    out = preprocess_ct(x, low=-100.0, high=400.0)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32


def test_spacing_from_affine_extracts_mm() -> None:
    affine = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.75, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    spacing = spacing_from_affine(affine)
    np.testing.assert_allclose(spacing, np.array([0.5, 0.5, 0.75], dtype=np.float32))


def test_resample_to_spacing_linear_image() -> None:
    vol = np.random.rand(10, 12, 8).astype(np.float32)
    current_spacing = np.array([1.2, 1.2, 1.2], dtype=np.float32)
    out = resample_to_spacing(vol, current_spacing, target_spacing=(0.6, 0.6, 0.6), is_label=False)
    assert out.shape == (20, 24, 16)
    assert out.dtype == np.float32


def test_resample_to_spacing_nearest_label_preserves_integer_classes() -> None:
    vol = np.zeros((8, 8, 8), dtype=np.int16)
    vol[2:6, 2:6, 2:6] = 3
    current_spacing = np.array([1.2, 1.2, 1.2], dtype=np.float32)
    out = resample_to_spacing(vol, current_spacing, target_spacing=(0.6, 0.6, 0.6), is_label=True)
    uniq = np.unique(out)
    assert out.shape == (16, 16, 16)
    assert set(uniq.tolist()) <= {0, 3}


def test_split_and_load_roundtrip(tmp_path: Path) -> None:
    ids = [f"case_{i:03d}" for i in range(10)]
    train_ids, val_ids = split_and_save(ids[:], split_ratio=0.8, output_dir=tmp_path)
    loaded_train, loaded_val = load_split_ids(tmp_path)
    assert len(train_ids) == 8
    assert len(val_ids) == 2
    assert loaded_train == train_ids
    assert loaded_val == val_ids


def test_center_crop_3d_shape() -> None:
    image = np.random.rand(32, 40, 28).astype(np.float32)
    label = np.random.randint(0, 4, size=(32, 40, 28), dtype=np.int16)
    out_img, out_lbl = center_crop_3d(image, label, patch_size=(16, 20, 12))
    assert out_img.shape == (16, 20, 12)
    assert out_lbl.shape == (16, 20, 12)


def test_read_num_classes_from_labelmap(tmp_path: Path) -> None:
    labelmap = tmp_path / "labelmap.txt"
    labelmap.write_text(
        "\n".join(
            [
                "# ITK-SNAP label map",
                "0 0 0 0 0 background",
                "1 255 0 0 1 vessel_1",
                "40 0 255 0 1 vessel_40",
            ]
        )
        + "\n"
    )
    assert read_num_classes_from_labelmap(labelmap) == 41
