from __future__ import annotations

import torch

from model_3d_unet import UNet3D


def test_unet3d_output_shape_matches_input_spatial_shape() -> None:
    model = UNet3D(in_channels=1, num_classes=41, base_ch=8)
    x = torch.randn(2, 1, 64, 64, 64)
    y = model(x)
    assert y.shape == (2, 41, 64, 64, 64)


def test_unet3d_handles_non_divisible_spatial_dimensions() -> None:
    model = UNet3D(in_channels=1, num_classes=41, base_ch=8)
    x = torch.randn(1, 1, 62, 70, 66)
    y = model(x)
    assert y.shape == (1, 41, 62, 70, 66)


def test_unet3d_residual_blocks_output_shape_matches_input_spatial_shape() -> None:
    model = UNet3D(in_channels=1, num_classes=41, base_ch=8, residual_blocks=True)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    assert y.shape == (1, 41, 64, 64, 64)
