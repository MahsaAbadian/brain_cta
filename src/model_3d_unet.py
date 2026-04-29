# model_3d_unet.py
from __future__ import annotations

"""Compact 3D U-Net baseline for multiclass vessel segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvBlock3D(nn.Module):
    """(Conv3d -> InstanceNorm -> LeakyReLU) x2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResConvBlock3D(nn.Module):
    """Residual 3D conv block with projection skip when channels change."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.skip(x)
        return self.act(out)


class UNet3D(nn.Module):
    """
    3D U-Net for multiclass vessel segmentation.
    Default channels: 16 -> 32 -> 64 -> 128 -> 256 (bottleneck).
    Uses InstanceNorm + LeakyReLU (stable with batch_size=1).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 41,
        base_ch: int = 16,
        use_checkpoint: bool = False,
        deep_supervision: bool = False,
        residual_blocks: bool = False,
    ):
        super().__init__()
        # When enabled, encoder/decoder ConvBlocks are run via
        # torch.utils.checkpoint: their forward activations are NOT stored for
        # backward; instead the forward is re-run during backward to rebuild
        # them. Trades ~25-35% extra compute per step for a large drop in peak
        # activation memory. Results are bit-exact vs. non-checkpointed.
        self.use_checkpoint = use_checkpoint
        # If enabled, return additional decoder-scale logits during training only.
        self.deep_supervision = deep_supervision
        self.residual_blocks = residual_blocks
        block_cls = ResConvBlock3D if residual_blocks else ConvBlock3D

        # Encoder
        self.enc1 = block_cls(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = block_cls(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = block_cls(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = block_cls(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = block_cls(base_ch * 8, base_ch * 16)

        # Decoder (transpose conv upsample + skip concat + conv block)
        self.up4 = nn.ConvTranspose3d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = block_cls(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = block_cls(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = block_cls(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = block_cls(base_ch * 2, base_ch)

        # Final 1x1x1 conv -> logits
        self.out_conv = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        if self.deep_supervision:
            self.ds2 = nn.Conv3d(base_ch * 2, num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(base_ch * 4, num_classes, kernel_size=1)
            self.ds4 = nn.Conv3d(base_ch * 8, num_classes, kernel_size=1)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Safety crop/pad to match skip tensor size if odd dimensions cause mismatch.
        Usually not needed for sizes divisible by 16, but useful anyway.
        """
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a ConvBlock3D with optional gradient checkpointing.

        Checkpointing is only applied during training; at eval time we use the
        standard forward so activation memory is already minimal and we avoid
        recompute overhead.
        """
        if self.use_checkpoint and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # Encoder path
        e1 = self._run_block(self.enc1, x)
        e2 = self._run_block(self.enc2, self.pool1(e1))
        e3 = self._run_block(self.enc3, self.pool2(e2))
        e4 = self._run_block(self.enc4, self.pool3(e3))

        b = self._run_block(self.bottleneck, self.pool4(e4))

        # Decoder path mirrors the encoder and fuses skip features at each scale.
        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self._run_block(self.dec4, torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self._run_block(self.dec3, torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self._run_block(self.dec2, torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self._run_block(self.dec1, torch.cat([d1, e1], dim=1))

        logits = self.out_conv(d1)  # raw logits
        if self.training and self.deep_supervision:
            return [logits, self.ds2(d2), self.ds3(d3), self.ds4(d4)]
        return logits


if __name__ == "__main__":
    # quick shape sanity test
    model = UNet3D(in_channels=1, num_classes=41, base_ch=16)
    x = torch.randn(2, 1, 128, 128, 128)  # (B, C, D, H, W)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)  # expected: (2, 41, 128, 128, 128)