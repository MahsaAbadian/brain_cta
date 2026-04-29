"""Project-local nnU-Net v2 trainer variants for TopBrain experiments.

nnU-Net only auto-discovers trainers that live inside its installed package.
The companion ``run_custom_training.py`` wrapper imports this module directly
and patches discovery so we can keep custom loss experiments in this repo.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc


class TopBrainDiceCEclDiceLoss(nn.Module):
    """CE + soft Dice + differentiable clDice for multi-class vessel labels."""

    def __init__(
        self,
        *,
        batch_dice: bool,
        include_background: bool = False,
        ignore_label: int | None = None,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 0.25,
        cldice_iters: int = 8,
        cldice_channel_chunk: int = 4,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if cldice_iters < 1:
            raise ValueError(f"cldice_iters must be >= 1, got {cldice_iters}")
        if cldice_channel_chunk < 0:
            raise ValueError(
                f"cldice_channel_chunk must be >= 0, got {cldice_channel_chunk}"
            )
        self.batch_dice = batch_dice
        self.include_background = include_background
        self.ignore_label = ignore_label
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_cldice = weight_cldice
        self.cldice_iters = cldice_iters
        self.cldice_channel_chunk = cldice_channel_chunk
        self.smooth = smooth

    @staticmethod
    def _soft_erode(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            e1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            e2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
            e3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
            return torch.minimum(torch.minimum(e1, e2), e3)
        if x.ndim == 4:
            e1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
            e2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
            return torch.minimum(e1, e2)
        raise ValueError(
            f"Expected 2D/3D network output, got tensor shape {tuple(x.shape)}"
        )

    @staticmethod
    def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        if x.ndim == 4:
            return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        raise ValueError(
            f"Expected 2D/3D network output, got tensor shape {tuple(x.shape)}"
        )

    @classmethod
    def _soft_open(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._soft_dilate(cls._soft_erode(x))

    @classmethod
    def _soft_skeletonize(cls, x: torch.Tensor, iters: int) -> torch.Tensor:
        opened = cls._soft_open(x)
        skel = F.relu(x - opened)
        for _ in range(iters):
            x = cls._soft_erode(x)
            opened = cls._soft_open(x)
            delta = F.relu(x - opened)
            skel = skel + (1.0 - skel) * delta
        return skel

    @classmethod
    def _soft_skeletonize_checkpointed(
        cls, x: torch.Tensor, iters: int
    ) -> torch.Tensor:
        return checkpoint(cls._soft_skeletonize, x, iters, use_reentrant=False)

    def _prepare_target(
        self, target: torch.Tensor, num_classes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if target.ndim < 3:
            raise ValueError(f"Invalid target shape: {tuple(target.shape)}")
        if target.shape[1] != 1:
            raise ValueError(
                "This trainer expects class-index targets shaped (B, 1, ...), "
                f"got {tuple(target.shape)}"
            )

        target_labels = target[:, 0].long()
        valid_mask = None
        safe_target = target_labels
        if self.ignore_label is not None:
            valid_mask = target_labels != self.ignore_label
            safe_target = torch.where(valid_mask, target_labels, 0)

        one_hot = F.one_hot(safe_target, num_classes=num_classes)
        one_hot = one_hot.movedim(-1, 1).float()
        if valid_mask is not None:
            one_hot = one_hot * valid_mask.unsqueeze(1)
        return target_labels, one_hot, valid_mask

    def _dice_loss(
        self,
        probs: torch.Tensor,
        target_1h: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            probs = probs * valid_mask.unsqueeze(1)

        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]

        spatial_dims = tuple(range(2, probs.ndim))
        dims = (0, *spatial_dims) if self.batch_dice else spatial_dims
        inter = (probs.float() * target_1h.float()).sum(dims)
        denom = probs.float().sum(dims) + target_1h.float().sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def _cldice_loss(
        self,
        probs: torch.Tensor,
        target_1h: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            probs = probs * valid_mask.unsqueeze(1)

        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]

        chunk_size = self.cldice_channel_chunk
        num_channels = probs.shape[1]
        if chunk_size <= 0 or chunk_size >= num_channels:
            return self._cldice_loss_chunk(probs, target_1h)

        losses = []
        weights = []
        for start in range(0, num_channels, chunk_size):
            end = min(start + chunk_size, num_channels)
            loss = self._cldice_loss_chunk(probs[:, start:end], target_1h[:, start:end])
            losses.append(loss)
            weights.append(end - start)
        weighted = sum(w * loss for w, loss in zip(weights, losses))
        return weighted / sum(weights)

    def _cldice_loss_chunk(
        self, probs: torch.Tensor, target_1h: torch.Tensor
    ) -> torch.Tensor:
        probs_skel = self._soft_skeletonize_checkpointed(probs, self.cldice_iters).float()
        target_skel = self._soft_skeletonize_checkpointed(
            target_1h, self.cldice_iters
        ).float()
        probs = probs.float()
        target_1h = target_1h.float()

        dims = (0, *tuple(range(2, probs.ndim))) if self.batch_dice else tuple(range(2, probs.ndim))
        tprec = (probs_skel * target_1h).sum(dims) / (
            probs_skel.sum(dims) + self.smooth
        )
        tsens = (target_skel * probs).sum(dims) / (
            target_skel.sum(dims) + self.smooth
        )
        cldice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cldice.mean()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_labels, target_1h, valid_mask = self._prepare_target(
            target, net_output.shape[1]
        )
        if self.weight_ce != 0 and (valid_mask is None or valid_mask.any()):
            ce_loss = F.cross_entropy(
                net_output,
                target_labels,
                ignore_index=self.ignore_label if self.ignore_label is not None else -100,
            )
        else:
            ce_loss = torch.zeros((), dtype=net_output.dtype, device=net_output.device)

        probs = torch.softmax(net_output, dim=1)
        dice_loss = (
            self._dice_loss(probs, target_1h, valid_mask)
            if self.weight_dice != 0
            else torch.zeros((), dtype=net_output.dtype, device=net_output.device)
        )
        cldice_loss = (
            self._cldice_loss(probs, target_1h, valid_mask)
            if self.weight_cldice != 0
            else torch.zeros((), dtype=net_output.dtype, device=net_output.device)
        )
        return (
            self.weight_ce * ce_loss
            + self.weight_dice * dice_loss
            + self.weight_cldice * cldice_loss
        )


class nnUNetTrainerTopBrainClDice(nnUNetTrainer):
    """nnU-Net trainer using a TopBrain vessel-aware custom loss."""

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError(
                "nnUNetTrainerTopBrainClDice supports class-index labels, not region labels."
            )

        loss = TopBrainDiceCEclDiceLoss(
            batch_dice=self.configuration_manager.batch_dice,
            include_background=os.environ.get(
                "TOPBRAIN_NNUNET_INCLUDE_BACKGROUND", "0"
            )
            == "1",
            ignore_label=self.label_manager.ignore_label,
            weight_ce=_env_float("TOPBRAIN_NNUNET_CE_WEIGHT", 1.0),
            weight_dice=_env_float("TOPBRAIN_NNUNET_DICE_WEIGHT", 1.0),
            weight_cldice=_env_float("TOPBRAIN_NNUNET_CLDICE_WEIGHT", 0.25),
            cldice_iters=_env_int("TOPBRAIN_NNUNET_CLDICE_ITERS", 8),
            cldice_channel_chunk=_env_int("TOPBRAIN_NNUNET_CLDICE_CHANNEL_CHUNK", 4),
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
