"""Loss functions used by training and evaluation pipelines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        cldice_weight: float = 0.0,
        cldice_iters: int = 12,
        cldice_class_ids: tuple[int, ...] | None = None,
        include_background: bool = True,
        eps: float = 1e-6,
        ce_class_weights: torch.Tensor | None = None,
        cldice_channel_chunk: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cldice_weight = cldice_weight
        self.cldice_iters = cldice_iters
        self.cldice_class_ids = cldice_class_ids
        self.include_background = include_background
        self.eps = eps
        # cldice_channel_chunk:
        #   0  -> process all clDice channels at once (legacy behavior).
        #   >0 -> process at most this many channels per skeletonize call,
        #         accumulating per-class numerators/denominators across chunks.
        # Soft-skeletonize is channel-independent (max_pool3d / pointwise ops),
        # so chunked evaluation is mathematically identical to all-at-once but
        # with peak memory scaled by ~chunk/total_channels.
        if cldice_channel_chunk < 0:
            raise ValueError(
                f"cldice_channel_chunk must be >= 0, got {cldice_channel_chunk}"
            )
        self.cldice_channel_chunk = int(cldice_channel_chunk)
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights)
        self._cldice_channel_indices = self._build_cldice_channel_indices()

    def _build_cldice_channel_indices(self) -> tuple[int, ...] | None:
        if self.cldice_class_ids is None:
            return None
        indices: list[int] = []
        for class_id in self.cldice_class_ids:
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(
                    f"Invalid clDice class id {class_id}; valid range is [0, {self.num_classes - 1}]"
                )
            if not self.include_background and class_id == 0:
                continue
            idx = class_id if self.include_background else class_id - 1
            if idx < 0:
                continue
            indices.append(idx)
        deduped = tuple(sorted(set(indices)))
        if len(deduped) == 0:
            raise ValueError(
                "clDice class selection is empty after background filtering. "
                "Provide at least one foreground class id."
            )
        return deduped

    @staticmethod
    def _soft_erode(x: torch.Tensor) -> torch.Tensor:
        e1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        e2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        e3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        return torch.minimum(torch.minimum(e1, e2), e3)

    @staticmethod
    def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

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
    def _skeletonize_checkpointed(cls, x: torch.Tensor, iters: int) -> torch.Tensor:
        """Gradient-checkpointed skeletonize: frees intermediates during forward."""
        return checkpoint(cls._soft_skeletonize, x, iters, use_reentrant=False)

    def _select_cldice_channels(
        self,
        probs: torch.Tensor,
        target_1h: torch.Tensor,
        target_skel: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self._cldice_channel_indices is None:
            return probs, target_1h, target_skel
        idx = list(self._cldice_channel_indices)
        probs_sel = probs[:, idx]
        target_sel = target_1h[:, idx]
        if target_skel is None:
            return probs_sel, target_sel, None
        if target_skel.shape[1] == target_1h.shape[1]:
            target_skel_sel = target_skel[:, idx]
        elif target_skel.shape[1] == len(idx):
            target_skel_sel = target_skel
        else:
            raise ValueError(
                "target_skel channel dimension must match either full target channels "
                f"({target_1h.shape[1]}) or selected clDice channels ({len(idx)}), "
                f"got {target_skel.shape[1]}."
            )
        return probs_sel, target_sel, target_skel_sel

    def _cldice_loss(
        self,
        probs: torch.Tensor,
        target_1h: torch.Tensor,
        target_skel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs, target_1h, target_skel = self._select_cldice_channels(
            probs, target_1h, target_skel
        )

        num_channels = probs.shape[1]
        chunk = self.cldice_channel_chunk
        if chunk <= 0 or chunk >= num_channels:
            # Fast path: one big skeletonize call, legacy behavior.
            return self._cldice_loss_chunk(probs, target_1h, target_skel)

        # Chunked path: loop over contiguous channel groups and accumulate
        # per-class numerator / denominator pairs separately for tprec and
        # tsens. Each chunk's skeletonize graph is an independent autograd
        # subgraph (it calls _skeletonize_checkpointed internally), so during
        # backward the recompute activations from chunk i are freed before
        # chunk i+1's are rebuilt. Peak memory is therefore set by `chunk`,
        # not by `num_channels`.
        tprec_num_parts: list[torch.Tensor] = []
        tprec_den_parts: list[torch.Tensor] = []
        tsens_num_parts: list[torch.Tensor] = []
        tsens_den_parts: list[torch.Tensor] = []
        dims = (0, 2, 3, 4)
        for start in range(0, num_channels, chunk):
            end = min(start + chunk, num_channels)
            probs_c = probs[:, start:end]
            target_1h_c = target_1h[:, start:end]
            target_skel_c = (
                target_skel[:, start:end] if target_skel is not None else None
            )
            probs_skel_c = self._skeletonize_checkpointed(probs_c, self.cldice_iters)
            if target_skel_c is None:
                target_skel_c = self._skeletonize_checkpointed(
                    target_1h_c, self.cldice_iters
                )
            probs_skel_c = probs_skel_c.float()
            target_skel_c = target_skel_c.float()
            probs_c = probs_c.float()
            target_1h_c = target_1h_c.float()
            tprec_num_parts.append((probs_skel_c * target_1h_c).sum(dims))
            tprec_den_parts.append(probs_skel_c.sum(dims))
            tsens_num_parts.append((target_skel_c * probs_c).sum(dims))
            tsens_den_parts.append(target_skel_c.sum(dims))

        tprec_num = torch.cat(tprec_num_parts)
        tprec_den = torch.cat(tprec_den_parts)
        tsens_num = torch.cat(tsens_num_parts)
        tsens_den = torch.cat(tsens_den_parts)
        tprec = tprec_num / (tprec_den + self.eps)
        tsens = tsens_num / (tsens_den + self.eps)
        cldice_score = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return 1.0 - cldice_score.mean()

    def _cldice_loss_chunk(
        self,
        probs: torch.Tensor,
        target_1h: torch.Tensor,
        target_skel: torch.Tensor | None,
    ) -> torch.Tensor:
        """Original, all-channels-at-once clDice computation for a single chunk."""
        probs_skel = self._skeletonize_checkpointed(probs, self.cldice_iters)
        if target_skel is None:
            target_skel = self._skeletonize_checkpointed(target_1h, self.cldice_iters)

        probs_skel = probs_skel.float()
        target_skel = target_skel.float()
        probs = probs.float()
        target_1h = target_1h.float()

        dims = (0, 2, 3, 4)
        tprec = (probs_skel * target_1h).sum(dims) / (probs_skel.sum(dims) + self.eps)
        tsens = (target_skel * probs).sum(dims) / (target_skel.sum(dims) + self.eps)
        cldice_score = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return 1.0 - cldice_score.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        target_skel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        if not self.include_background:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]
            if target_skel is not None and target_skel.shape[1] == self.num_classes:
                target_skel = target_skel[:, 1:]

        dims = (0, 2, 3, 4)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice.mean()
        cldice_loss = (
            self._cldice_loss(probs, target_1h, target_skel=target_skel)
            if self.cldice_weight > 0.0
            else torch.zeros((), dtype=logits.dtype, device=logits.device)
        )

        return (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.cldice_weight * cldice_loss
        )
