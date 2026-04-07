from __future__ import annotations

import numpy as np
import pytest

from preprocess_resample import _make_target_affine


def test_make_target_affine_preserves_origin_and_orientation() -> None:
    old_affine = np.array(
        [
            [0.0, -0.5, 0.0, 10.0],
            [0.5, 0.0, 0.0, 20.0],
            [0.0, 0.0, 0.75, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = _make_target_affine(old_affine, target_spacing=(0.6, 0.6, 0.6))

    # Origin unchanged
    np.testing.assert_allclose(out[:3, 3], old_affine[:3, 3], rtol=1e-6, atol=1e-6)

    # New spacing equals target
    spacing = np.sqrt(np.sum(out[:3, :3] ** 2, axis=0))
    np.testing.assert_allclose(spacing, np.array([0.6, 0.6, 0.6], dtype=np.float32), atol=1e-6)

    # Direction columns remain collinear (same orientation axes)
    for i in range(3):
        a = old_affine[:3, i]
        b = out[:3, i]
        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            assert abs(cos) == pytest.approx(1.0, abs=1e-5)


def test_make_target_affine_rejects_degenerate_basis() -> None:
    bad_affine = np.eye(4, dtype=np.float32)
    bad_affine[:3, 0] = 0.0
    with pytest.raises(ValueError, match="Invalid affine basis"):
        _make_target_affine(bad_affine, target_spacing=(0.6, 0.6, 0.6))
