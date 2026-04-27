"""Full baseline training script with validation and metric logging."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from data_loader import (
    CTAPatchDataset,
    build_train_val_loaders,
    compute_rare_class_sampling_weights,
)
from data_utils import (
    read_class_names_from_labelmap,
    read_num_classes_from_labelmap,
)
from losses import DiceCELoss
from model_3d_unet import UNet3D
from topbrain_validation import (
    TOPBRAIN_METRIC_KEYS,
    TopBrainRuntime,
    empty_topbrain_metrics,
    select_fixed_topbrain_subset,
    should_run_topbrain_full_eval,
)
from validation import validate_one_epoch


def _parse_class_id_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if len(tokens) == 0:
        return None
    try:
        values = tuple(int(tok) for tok in tokens)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --cldice-class-ids='{raw}'. Expected comma-separated integers."
        ) from exc
    return tuple(sorted(set(values)))


def _parse_rare_class_mode(raw: str) -> Literal["presence", "voxel", "hybrid"]:
    mode = raw.strip().lower()
    if mode not in {"presence", "voxel", "hybrid"}:
        raise ValueError(
            f"Invalid --rare-class-mode='{raw}'. Expected one of: presence, voxel, hybrid."
        )
    return mode  # type: ignore[return-value]


def _resolve_expected_cldice_class_ids(
    *,
    num_classes: int,
    include_background: bool,
    cldice_class_ids: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if cldice_class_ids is None:
        start = 0 if include_background else 1
        return tuple(range(start, num_classes))

    expected: list[int] = []
    for class_id in sorted(set(cldice_class_ids)):
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"Invalid clDice class id {class_id}; valid range is [0, {num_classes - 1}]"
            )
        if (not include_background) and class_id == 0:
            continue
        expected.append(class_id)
    if len(expected) == 0:
        raise ValueError(
            "clDice class selection is empty after background filtering. "
            "Provide at least one foreground class id."
        )
    return tuple(expected)


def _validate_target_skeleton_dir(
    *,
    target_skeleton_dir: Path,
    train_case_ids: list[str],
    expected_class_ids: tuple[int, ...],
) -> None:
    missing = [
        case_id
        for case_id in train_case_ids
        if not (target_skeleton_dir / f"{case_id}.npz").is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            "Missing precomputed target skeleton files for training cases in "
            f"{target_skeleton_dir}: {preview}{more}"
        )

    sample_path = target_skeleton_dir / f"{train_case_ids[0]}.npz"
    with np.load(sample_path) as sample:
        if "skel" not in sample:
            raise KeyError(f"Expected key 'skel' in {sample_path}")
        skel = sample["skel"]
        if skel.ndim != 4:
            raise ValueError(
                f"Invalid precomputed skeleton shape in {sample_path}: "
                f"{skel.shape}. Expected (C, D, H, W)."
            )
        if skel.shape[0] != len(expected_class_ids):
            raise ValueError(
                "Precomputed skeleton channel mismatch. "
                f"Expected {len(expected_class_ids)} channels from clDice class IDs "
                f"{expected_class_ids}, got {skel.shape[0]} in {sample_path}."
            )

        if "class_ids" in sample:
            file_class_ids = tuple(int(x) for x in sample["class_ids"].tolist())
            if file_class_ids != expected_class_ids:
                raise ValueError(
                    "Precomputed skeleton class_ids metadata mismatch. "
                    f"Expected {expected_class_ids}, got {file_class_ids} in {sample_path}."
                )
        method = str(sample["method"][0]) if "method" in sample else "unknown"
    print(
        "validated precomputed target skeletons: "
        f"dir={target_skeleton_dir} channels={len(expected_class_ids)} method={method}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline 3D U-Net on resampled CTA data.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--num-patches-per-volume", type=int, default=2)
    parser.add_argument(
        "--val-stride",
        type=int,
        nargs=3,
        default=None,
        help=(
            "Sliding-window stride (x y z) used only for full-volume validation. "
            "Default is patch_size // 2 per axis."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Device to use for training. 'auto' prefers CUDA but falls back to CPU "
            "if CUDA initialization fails."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path("training_data_resampled/split/splits_final.json"),
        help=(
            "Canonical K-fold splits file (nnUNet format). When the file "
            "exists, the requested --fold is used. Falls back to legacy "
            "train_cases.txt/val_cases.txt when this file is missing."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index into --splits-json (ignored in legacy mode).",
    )
    parser.add_argument(
        "--fold-subdir",
        action="store_true",
        help=(
            "If set, training artifacts go under <out-dir>/fold_<fold> instead "
            "of <out-dir>. Recommended when sweeping all folds into one parent "
            "run directory."
        ),
    )
    parser.add_argument(
        "--overfit-case-id",
        type=str,
        default=None,
        help=(
            "Optional debug mode: use this single case ID for both train and val "
            "(intentional leakage) to verify the model can overfit."
        ),
    )
    parser.add_argument(
        "--overfit-disable-augment",
        action="store_true",
        help=(
            "Only used with --overfit-case-id. Disable training-time random flips "
            "to make one-case memorization easier and debugging cleaner."
        ),
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight for Dice term inside DiceCELoss.",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Weight for CE term inside DiceCELoss.",
    )
    parser.add_argument(
        "--tversky-weight",
        type=float,
        default=1.0,
        help="Weight for Tversky/Focal-Tversky term inside DiceCELoss.",
    )
    parser.add_argument(
        "--tversky-alpha",
        type=float,
        default=0.3,
        help="Tversky FN penalty weight (higher penalizes missed positives more).",
    )
    parser.add_argument(
        "--tversky-beta",
        type=float,
        default=0.7,
        help="Tversky FP penalty weight (higher penalizes false positives more).",
    )
    parser.add_argument(
        "--tversky-gamma",
        type=float,
        default=1.0,
        help="Focal exponent for Tversky term (1.0 = plain Tversky).",
    )
    parser.add_argument(
        "--cldice-weight",
        type=float,
        default=1.0,
        help="Weight for clDice term inside DiceCELoss (0 disables clDice).",
    )
    parser.add_argument(
        "--cldice-iters",
        type=int,
        default=12,
        help="Number of iterative soft-skeletonization steps used by clDice.",
    )
    parser.add_argument(
        "--cldice-class-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated class IDs to include in clDice. "
            "Example: '1,2,7'. Dice/CE still use all classes."
        ),
    )
    parser.add_argument(
        "--cldice-target-skeleton-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing precomputed target skeleton files "
            "named <case_id>.npz with key 'skel'."
        ),
    )
    parser.add_argument(
        "--cldice-channel-chunk",
        type=int,
        default=0,
        help=(
            "Process clDice skeletonization in chunks of this many channels "
            "(0 = all channels at once, legacy behavior). Soft-skeletonize is "
            "channel-independent so chunking is bit-exact but cuts peak "
            "skeletonize memory by roughly chunk/total_channels. Use small "
            "values (1-4) to trade extra step time for OOM relief."
        ),
    )
    parser.add_argument(
        "--ce-weight-min",
        type=float,
        default=None,
        help="Optional minimum clamp for CE class weights after normalization.",
    )
    parser.add_argument(
        "--ce-weight-max",
        type=float,
        default=None,
        help="Optional maximum clamp for CE class weights after normalization.",
    )
    parser.add_argument(
        "--enable-ce-class-weights",
        action="store_true",
        help=(
            "Enable inverse-sqrt frequency class weights for CrossEntropyLoss. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--rare-class-patch-prob",
        type=float,
        default=0.35,
        help=(
            "Probability that a train patch center is sampled from a rare foreground "
            "class (based on patient-level class presence). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--rare-class-weight-max",
        type=float,
        default=4.0,
        help=(
            "Maximum patient-presence inverse weight used for rare-class patch "
            "sampling. Higher increases focus on sparse classes."
        ),
    )
    parser.add_argument(
        "--rare-class-mode",
        type=str,
        choices=("presence", "voxel", "hybrid"),
        default="hybrid",
        help=(
            "Rare-class weighting mode for patch-center sampling: "
            "'presence' (inverse patient presence), "
            "'voxel' (inverse sqrt voxel frequency), "
            "'hybrid' (geometric mean of both)."
        ),
    )
    parser.add_argument(
        "--topbrain-disable",
        action="store_true",
        help="Disable TopBrain metric computation during validation.",
    )
    parser.add_argument(
        "--topbrain-per-epoch",
        action="store_true",
        help=(
            "Run TopBrain during training validation (subset every epoch; full every N). "
            "Default is to run TopBrain once after the last training epoch only."
        ),
    )
    parser.add_argument(
        "--topbrain-track",
        type=str,
        default="ct",
        choices=("ct", "mr"),
        help="TopBrain track used for metric definitions.",
    )
    parser.add_argument(
        "--topbrain-eval-every-n-epochs",
        type=int,
        default=5,
        help=(
            "When --topbrain-per-epoch is set: run full-validation TopBrain metrics every N epochs. "
            "Ignored when TopBrain runs only at end."
        ),
    )
    parser.add_argument(
        "--topbrain-subset-size",
        type=int,
        default=4,
        help=(
            "When --topbrain-per-epoch is set: fixed validation subset size for each epoch's TopBrain."
        ),
    )
    parser.add_argument(
        "--topbrain-full-max-cases",
        type=int,
        default=None,
        help=(
            "Optional cap on number of validation cases in full TopBrain runs. "
            "When set, a deterministic subset is used."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable CUDA mixed-precision training (autocast + GradScaler). "
            "Significantly reduces activation memory and speeds up training on "
            "modern GPUs. Requires --device=cuda (or auto-resolved to cuda)."
        ),
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help=(
            "Enable gradient (activation) checkpointing on UNet encoder/decoder "
            "blocks. Bit-exact equivalent of standard training, trades ~25-35%% "
            "extra compute per step for a large drop in peak activation memory. "
            "Useful when you want larger effective batch without AMP."
        ),
    )
    parser.add_argument(
        "--deep-supervision",
        action="store_true",
        help=(
            "Enable training-only deep supervision heads on decoder scales "
            "(dec2/dec3/dec4). Adds weighted multi-scale loss with nnUNet-style "
            "weights [1, 1/2, 1/4, 1/8] normalized to sum to 1."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=("fp16", "bf16"),
        default="fp16",
        help=(
            "Autocast dtype when --amp is set. 'bf16' is safer (no GradScaler "
            "needed for dynamic range) on Ampere+ GPUs; 'fp16' is the default."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Save a full (model+optimizer+scheduler+RNG+epoch) training "
            "checkpoint every N epochs for resumability. 0 (default) disables "
            "periodic checkpointing. Best-val weights are saved independently "
            "whenever validation Dice improves."
        ),
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=3,
        help=(
            "When --checkpoint-every > 0, retain this many most recent periodic "
            "checkpoints and delete older ones. Best-val and final weights are "
            "kept regardless."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional path to a periodic checkpoint (.pt) produced by "
            "--checkpoint-every. Resumes training with restored model, "
            "optimizer, scheduler, GradScaler, and RNG state. Note: keep the "
            "same --epochs (cosine schedule T_max) as the original run for "
            "consistent LR decay."
        ),
    )
    parser.add_argument(
        "--load-weights",
        type=Path,
        default=None,
        help=(
            "Optional path to weights-only model state_dict (.pt), such as "
            "model_final_weights.pt or model_best_weights.pt. Initializes the "
            "model from those weights but starts a fresh optimizer, scheduler, "
            "epoch counter, and RNG state. Mutually exclusive with --resume."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="topbrain",
        help="W&B project name when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name override.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Optional comma-separated W&B tags, e.g. 'ps128,cldice,ablationA'.",
    )
    return parser.parse_args()


def _parse_wandb_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _wandb_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            cfg[key] = str(value)
        elif isinstance(value, tuple):
            cfg[key] = list(value)
        else:
            cfg[key] = value
    return cfg


def _log_wandb_summary_charts(
    *,
    wandb_run: Any,
    history: dict[str, list[float]],
    class_labels: list[str],
) -> None:
    """Emit end-of-run overlay charts to W&B.

    Per-epoch scalars are already logged live; this just adds custom panels
    that overlay multiple series on a single chart:

    - ``summary/train_vs_val_loss``: total train loss vs total val loss.
    - ``summary/train_loss_components``: each DiceCELoss term over epochs
      (CE, Dice, Tversky, clDice).
    - ``summary/val_loss_components``: the same for validation.
    - ``summary/val_per_class_dice_history``: per-class Dice over epochs so
      it's easy to spot classes that never converge.
    """
    if wandb is None:
        return

    epochs = history.get("epoch", [])
    if not epochs:
        return

    def _sanitize(values: list[float]) -> list[float]:
        return [v if np.isfinite(v) else 0.0 for v in values]

    try:
        wandb_run.log(
            {
                "summary/train_vs_val_loss": wandb.plot.line_series(
                    xs=epochs,
                    ys=[_sanitize(history["train_loss"]), _sanitize(history["val_loss"])],
                    keys=["train_loss", "val_loss"],
                    title="Train vs Val Loss",
                    xname="epoch",
                )
            }
        )

        train_series = [
            _sanitize(history[f"train_loss_{name}"])
            for name in DiceCELoss.COMPONENT_NAMES
        ]
        val_series = [
            _sanitize(history[f"val_loss_{name}"])
            for name in DiceCELoss.COMPONENT_NAMES
        ]
        wandb_run.log(
            {
                "summary/train_loss_components": wandb.plot.line_series(
                    xs=epochs,
                    ys=train_series,
                    keys=[f"train_{name}" for name in DiceCELoss.COMPONENT_NAMES],
                    title="Train loss components (CE / Dice / Tversky / clDice)",
                    xname="epoch",
                ),
                "summary/val_loss_components": wandb.plot.line_series(
                    xs=epochs,
                    ys=val_series,
                    keys=[f"val_{name}" for name in DiceCELoss.COMPONENT_NAMES],
                    title="Val loss components (CE / Dice / Tversky / clDice)",
                    xname="epoch",
                ),
            }
        )

        num_classes = len(class_labels)
        per_class_series = [
            _sanitize(history[f"val_dice_c{c:02d}"]) for c in range(num_classes)
        ]
        keys = [f"c{c:02d}_{class_labels[c]}" for c in range(num_classes)]
        wandb_run.log(
            {
                "summary/val_per_class_dice_history": wandb.plot.line_series(
                    xs=epochs,
                    ys=per_class_series,
                    keys=keys,
                    title="Per-class Dice over epochs",
                    xname="epoch",
                )
            }
        )
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"warning: failed to log W&B summary charts: {exc}")


def compute_class_weights(
    label_dir: Path,
    case_ids: list[str],
    num_classes: int,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> torch.Tensor:
    """Inverse-sqrt-frequency class weights for CrossEntropyLoss.

    Scans every training label volume, counts voxels per class, then returns
    ``w[c] = 1 / sqrt(freq[c])`` normalized so the weights sum to
    ``num_classes``. Classes never seen get weight 0.

    Optional clamp_min/clamp_max can tame extreme rare-class emphasis.
    """
    import nibabel as nib

    counts = np.zeros(num_classes, dtype=np.float64)
    for cid in case_ids:
        lbl_path = label_dir / f"{cid}.nii.gz"
        lbl = np.asanyarray(nib.load(str(lbl_path)).dataobj).astype(np.int64)
        for val, cnt in zip(*np.unique(lbl, return_counts=True)):
            if 0 <= val < num_classes:
                counts[val] += cnt

    total = counts.sum()
    freq = counts / max(total, 1.0)

    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = freq > 0
    weights[nonzero] = 1.0 / np.sqrt(freq[nonzero])

    # Normalize so weights sum to num_classes (keeps loss magnitude stable).
    w_sum = weights.sum()
    if w_sum > 0:
        weights *= num_classes / w_sum

    if clamp_min is not None or clamp_max is not None:
        lo = clamp_min if clamp_min is not None else -np.inf
        hi = clamp_max if clamp_max is not None else np.inf
        if lo > hi:
            raise ValueError(
                f"Invalid CE clamp range: min={clamp_min} > max={clamp_max}"
            )
        weights = np.clip(weights, lo, hi)

    return torch.tensor(weights, dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve user device choice with safe CUDA fallback for --device=auto."""
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device=cuda requested, but CUDA is not available in this environment."
            )
        try:
            # Force a tiny allocation so runtime/driver issues fail early.
            torch.empty(1, device="cuda")
        except Exception as exc:
            raise RuntimeError(
                "--device=cuda requested, but CUDA failed to initialize."
            ) from exc
        return torch.device("cuda")

    # --device=auto
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(
                "warning: CUDA was detected but failed to initialize; "
                f"falling back to CPU ({exc})."
            )
    return torch.device("cpu")


def _flatten_loader_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_skel: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # data_loader returns:
    # image: (B_volume, Patches, C, D, H, W)
    # label: (B_volume, Patches, D, H, W)
    x = batch_x.flatten(0, 1)
    y = batch_y.flatten(0, 1)
    skel = batch_skel.flatten(0, 1) if batch_skel is not None else None
    return x, y, skel


def _topbrain_csv_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_ran",
        f"{prefix}_num_cases",
        *[f"{prefix}_{k}" for k in TOPBRAIN_METRIC_KEYS],
    ]


def _topbrain_csv_values(ran: bool, metrics: dict[str, float]) -> list[str]:
    values = [str(int(ran)), f"{int(metrics['num_cases'])}"]
    for key in TOPBRAIN_METRIC_KEYS:
        value = metrics[key]
        values.append("" if not np.isfinite(value) else f"{value:.6f}")
    return values


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: "torch.amp.GradScaler | None",
    epoch: int,
    best_val_dice: float,
) -> None:
    """Atomically write a full training checkpoint to `path`.

    Captures everything needed to resume training bit-identically (modulo
    non-deterministic CUDA kernels): model/optimizer/scheduler/scaler state,
    best-val tracking, and RNG states for python/numpy/torch-CPU/torch-CUDA.
    Writes to a .tmp file first and then renames to avoid leaving a corrupt
    checkpoint if the process is killed mid-save.
    """
    state: dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": (
            scaler.state_dict()
            if (scaler is not None and scaler.is_enabled())
            else None
        ),
        "best_val_dice": best_val_dice,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: "torch.amp.GradScaler | None",
    device: torch.device,
) -> tuple[int, float]:
    """Restore full training state from a checkpoint written by _save_checkpoint.

    Returns (last_completed_epoch, best_val_dice). Caller should start the next
    training epoch at `last_completed_epoch + 1`.
    """
    state = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if (
        scaler is not None
        and scaler.is_enabled()
        and state.get("scaler") is not None
    ):
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch_cpu"])
    cuda_rng = state.get("rng_torch_cuda")
    if torch.cuda.is_available() and cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return int(state["epoch"]), float(state["best_val_dice"])


def _load_model_weights(path: Path, *, model: nn.Module, device: torch.device) -> None:
    """Load weights-only model state produced by torch.save(model.state_dict())."""
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)


def _rotate_periodic_checkpoints(out_dir: Path, keep: int) -> None:
    """Delete periodic checkpoints beyond the `keep` most recent.

    Only matches the `checkpoint_epoch*.pt` naming pattern, so best-val and
    final weights (different filenames) are never touched.
    """
    if keep <= 0:
        return
    checkpoints = sorted(out_dir.glob("checkpoint_epoch*.pt"))
    for old in checkpoints[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Train for one epoch and return per-component and aggregate statistics.

    Returned dict always contains ``total`` (mean optimization loss) and
    ``grad_norm`` (mean L2 gradient norm across steps). When the criterion
    exposes ``forward_components``, the dict also contains per-term losses
    keyed by ``ce``, ``dice``, ``tversky``, ``cldice`` so that W&B/CSV can
    track each loss curve individually.
    """
    ds_base_weights = torch.tensor(
        [1.0, 0.5, 0.25, 0.125], dtype=torch.float32, device=device
    )
    ds_base_weights = ds_base_weights / ds_base_weights.sum()

    def _resize_target_for_logits(
        target_full: torch.Tensor, logits_scale: torch.Tensor
    ) -> torch.Tensor:
        if target_full.shape[1:] == logits_scale.shape[2:]:
            return target_full
        target_small = F.interpolate(
            target_full.unsqueeze(1).float(),
            size=logits_scale.shape[2:],
            mode="nearest",
        ).squeeze(1)
        return target_small.long()

    def _resize_skeleton_for_logits(
        skel_full: torch.Tensor | None, logits_scale: torch.Tensor
    ) -> torch.Tensor | None:
        if skel_full is None:
            return None
        if skel_full.shape[2:] == logits_scale.shape[2:]:
            return skel_full
        return F.interpolate(
            skel_full,
            size=logits_scale.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def _multiscale_loss(
        logits_out: torch.Tensor | list[torch.Tensor],
        target_full: torch.Tensor,
        skel_full: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(logits_out, torch.Tensor):
            return criterion(logits_out, target_full, target_skel=skel_full)

        logits_scales = logits_out
        if len(logits_scales) == 0:
            raise ValueError("Deep supervision logits list is empty.")

        if len(logits_scales) > ds_base_weights.numel:
            raise ValueError(
                f"Deep supervision returned {len(logits_scales)} scales, "
                f"but only {ds_base_weights.numel} weights are defined."
            )
        weights = ds_base_weights[: len(logits_scales)]
        weights = weights / weights.sum()
        total = torch.zeros((), dtype=logits_scales[0].dtype, device=logits_scales[0].device)
        for i, logits_i in enumerate(logits_scales):
            target_i = _resize_target_for_logits(target_full, logits_i)
            skel_i = _resize_skeleton_for_logits(skel_full, logits_i)
            total = total + weights[i] * criterion(logits_i, target_i, target_skel=skel_i)
        return total

    # Track the unweighted per-term losses returned by DiceCELoss. We use the
    # finest-scale (primary) logits for reporting when deep supervision is on
    # so that the numbers line up with validation (which runs without DS).
    supports_components = hasattr(criterion, "forward_components")

    def _primary_logits(logits_out: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(logits_out, torch.Tensor):
            return logits_out
        return logits_out[0]

    model.train()
    running_loss = 0.0
    running_grad_norm = 0.0
    component_sums: dict[str, float] = {}
    n_steps = 0
    use_amp = scaler is not None and amp_dtype is not None and device.type == "cuda"
    for batch in loader:
        batch_skel = None
        if len(batch) == 4:
            batch_x, batch_y, batch_skel, _ = batch
        else:
            batch_x, batch_y, _ = batch
        batch_x, batch_y, batch_skel = _flatten_loader_batch(batch_x, batch_y, batch_skel)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_skel is not None:
            batch_skel = batch_skel.to(device)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(batch_x)
                loss = _multiscale_loss(logits, batch_y, batch_skel)
            scaler.scale(loss).backward()
            # Unscale before computing the real-scale gradient norm so the
            # value we log matches the step actually applied.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = _multiscale_loss(logits, batch_y, batch_skel)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            optimizer.step()

        if supports_components:
            with torch.no_grad():
                _, components = criterion.forward_components(
                    _primary_logits(logits), batch_y, target_skel=batch_skel
                )
            for name, value in components.items():
                component_sums[name] = (
                    component_sums.get(name, 0.0) + float(value.item())
                )

        running_loss += float(loss.item())
        if torch.isfinite(grad_norm):
            running_grad_norm += float(grad_norm.item())
        n_steps += 1

    divisor = max(n_steps, 1)
    stats: dict[str, float] = {
        "total": running_loss / divisor,
        "grad_norm": running_grad_norm / divisor,
    }
    for name, total in component_sums.items():
        stats[name] = total / divisor
    return stats


def _build_overfit_loaders(
    *,
    case_id: str,
    patch_size: tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    num_patches_per_volume: int,
    disable_augment: bool,
    rare_class_patch_prob: float,
    rare_class_weight_max: float,
    rare_class_mode: Literal["presence", "voxel", "hybrid"],
    target_skeleton_dir: Path | None,
) -> tuple[DataLoader, int]:
    image_dir = Path("training_data_resampled/imagesTr_topbrain_ct")
    label_dir = Path("training_data_resampled/labelsTr_topbrain_ct")
    labelmap_path = Path("training_data_resampled/itksnap_labelmap_txt/labelmap_topbrain_ct.txt")

    image_path = image_dir / f"{case_id}_0000.nii.gz"
    label_path = label_dir / f"{case_id}.nii.gz"
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Overfit case not found in resampled data: case_id={case_id} "
            f"(expected {image_path} and {label_path})"
        )

    num_classes = read_num_classes_from_labelmap(labelmap_path)
    rare_class_weights = compute_rare_class_sampling_weights(
        label_dir=label_dir,
        case_ids=[case_id],
        num_classes=num_classes,
        max_weight=rare_class_weight_max,
        mode=rare_class_mode,
    )
    train_ds = CTAPatchDataset(
        case_ids=[case_id],
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=patch_size,
        preprocess_fn=None,
        do_augment=not disable_augment,
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
    return train_loader, num_classes


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.load_weights is not None:
        raise ValueError("--resume and --load-weights are mutually exclusive.")
    if args.load_weights is not None and not args.load_weights.is_file():
        raise FileNotFoundError(f"--load-weights file not found: {args.load_weights}")

    cldice_class_ids = _parse_class_id_list(args.cldice_class_ids)
    rare_class_mode = _parse_rare_class_mode(args.rare_class_mode)
    target_skeleton_dir = args.cldice_target_skeleton_dir
    if target_skeleton_dir is not None and not target_skeleton_dir.is_dir():
        raise FileNotFoundError(
            f"--cldice-target-skeleton-dir does not exist: {target_skeleton_dir}"
        )
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    print(f"device={device}")

    out_dir: Path = args.out_dir
    if args.fold_subdir:
        out_dir = out_dir / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise ImportError(
                "W&B logging requested with --wandb, but 'wandb' is not installed. "
                "Install it with: .venv/bin/pip install wandb"
            )
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=_parse_wandb_tags(args.wandb_tags),
            config=_wandb_config_from_args(args),
            job_type="train",
            dir=str(out_dir),
        )
        wandb_run.config.update({"resolved_device": str(device)}, allow_val_change=True)
        # Make `epoch` the default x-axis for every chart so train/* and val/*
        # scalars end up aligned in the same per-section dashboard.
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("train/*", step_metric="epoch")
        wandb_run.define_metric("val/*", step_metric="epoch")
        wandb_run.define_metric("optim/*", step_metric="epoch")
        wandb_run.define_metric("topbrain/*", step_metric="epoch")
        wandb_run.define_metric("sys/*", step_metric="epoch")
        wandb_run.define_metric("time/*", step_metric="epoch")
        # Highlight which scalars should drive the "best" summary panel.
        wandb_run.define_metric("val/mean_fg_dice", summary="max")
        wandb_run.define_metric("val_mean_fg_dice", summary="max")
        wandb_run.define_metric("val/loss_total", summary="min")
        wandb_run.define_metric("train/loss_total", summary="min")
        print(
            "W&B enabled: "
            f"entity={args.wandb_entity} project={args.wandb_project} "
            f"name={args.wandb_run_name} mode={os.environ.get('WANDB_MODE', 'online')}"
        )
        if os.environ.get("WANDB_MODE", "").lower() == "offline":
            print("W&B offline mode active. Sync later with: wandb sync wandb/offline-run-*")

    patch_size = tuple(int(x) for x in args.patch_size)
    val_stride = (
        tuple(max(int(s), 1) for s in args.val_stride)
        if args.val_stride is not None
        else tuple(max(p // 2, 1) for p in patch_size)
    )
    label_dir = Path("training_data_resampled/labelsTr_topbrain_ct")
    image_dir = Path("training_data_resampled/imagesTr_topbrain_ct")

    if args.overfit_case_id:
        train_loader, num_classes = _build_overfit_loaders(
            case_id=args.overfit_case_id,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_patches_per_volume=args.num_patches_per_volume,
            disable_augment=args.overfit_disable_augment,
            rare_class_patch_prob=args.rare_class_patch_prob,
            rare_class_weight_max=args.rare_class_weight_max,
            rare_class_mode=rare_class_mode,
            target_skeleton_dir=target_skeleton_dir,
        )
        train_case_ids = [args.overfit_case_id]
        val_case_ids = [args.overfit_case_id]
        print(
            "overfit mode enabled: "
            f"case_id={args.overfit_case_id} "
            f"augment={'off' if args.overfit_disable_augment else 'on'}"
        )
    else:
        train_ds, val_case_ids, train_loader, num_classes = build_train_val_loaders(
            patch_size=patch_size,
            num_patches_per_volume=args.num_patches_per_volume,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rare_class_patch_prob=args.rare_class_patch_prob,
            rare_class_weight_max=args.rare_class_weight_max,
            rare_class_mode=rare_class_mode,
            target_skeleton_dir=target_skeleton_dir,
            splits_json=args.splits_json,
            fold=args.fold,
        )
        train_case_ids = train_ds.case_ids
        if args.splits_json is not None and args.splits_json.is_file():
            print(
                f"splits: {args.splits_json} fold={args.fold} "
                f"train={len(train_case_ids)} val={len(val_case_ids)}"
            )
    print(
        f"num_classes={num_classes} patch_size={patch_size} "
        f"val_stride={val_stride} train_batches={len(train_loader)} val_cases={len(val_case_ids)}"
    )
    print(
        "rare-class sampling: "
        f"prob={args.rare_class_patch_prob:.2f} "
        f"max_weight={args.rare_class_weight_max:.2f} "
        f"mode={rare_class_mode}"
    )

    if args.topbrain_eval_every_n_epochs <= 0:
        raise ValueError("--topbrain-eval-every-n-epochs must be >= 1")
    if args.cldice_weight < 0:
        raise ValueError("--cldice-weight must be >= 0")
    if args.tversky_weight < 0:
        raise ValueError("--tversky-weight must be >= 0")
    if args.tversky_alpha < 0:
        raise ValueError("--tversky-alpha must be >= 0")
    if args.tversky_beta < 0:
        raise ValueError("--tversky-beta must be >= 0")
    if args.tversky_alpha + args.tversky_beta <= 0:
        raise ValueError("--tversky-alpha + --tversky-beta must be > 0")
    if args.tversky_gamma <= 0:
        raise ValueError("--tversky-gamma must be > 0")
    if args.cldice_iters < 0:
        raise ValueError("--cldice-iters must be >= 0")
    if args.topbrain_subset_size < 0:
        raise ValueError("--topbrain-subset-size must be >= 0")
    if args.topbrain_full_max_cases is not None and args.topbrain_full_max_cases <= 0:
        raise ValueError("--topbrain-full-max-cases must be >= 1 when provided")

    ce_weights = None
    if args.enable_ce_class_weights:
        ce_weights = compute_class_weights(
            label_dir,
            train_case_ids,
            num_classes,
            clamp_min=args.ce_weight_min,
            clamp_max=args.ce_weight_max,
        ).to(device)
        if num_classes > 1:
            print(
                f"CE class weights enabled (bg={ce_weights[0]:.3f}, "
                f"min_fg={ce_weights[1:].min():.3f}, max_fg={ce_weights[1:].max():.3f}, "
                f"clamp_min={args.ce_weight_min}, clamp_max={args.ce_weight_max})"
            )
        else:
            print(
                f"CE class weights enabled (w0={ce_weights[0]:.3f}, "
                f"clamp_min={args.ce_weight_min}, clamp_max={args.ce_weight_max})"
            )
    else:
        print("CE class weights disabled (using unweighted CrossEntropyLoss).")
        if args.ce_weight_min is not None or args.ce_weight_max is not None:
            print(
                "Ignoring --ce-weight-min/--ce-weight-max because "
                "--enable-ce-class-weights is not set."
            )

    model = UNet3D(
        in_channels=1,
        num_classes=num_classes,
        base_ch=args.base_ch,
        use_checkpoint=bool(args.grad_checkpoint),
        deep_supervision=bool(args.deep_supervision),
    ).to(device)
    if args.load_weights is not None:
        _load_model_weights(args.load_weights, model=model, device=device)
        print(
            f"loaded model weights from {args.load_weights}; "
            "optimizer, scheduler, epoch counter, and RNG start fresh."
        )
    if wandb_run is not None:
        # Weights & Biases will automatically log weight and gradient
        # histograms every `log_freq` steps. This gives per-layer histograms
        # in the "Gradients" / "Parameters" panels of the run dashboard.
        try:
            wandb.watch(model, log="all", log_freq=100, log_graph=False)
        except Exception as exc:  # pragma: no cover - best-effort extra logging
            print(f"warning: wandb.watch failed ({exc}); skipping gradient histograms.")
    if args.grad_checkpoint:
        print("gradient checkpointing enabled on UNet encoder/decoder blocks")
    if args.deep_supervision:
        print("deep supervision enabled (training-only multi-scale logits)")
    criterion = DiceCELoss(
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
        cldice_weight=args.cldice_weight,
        cldice_iters=args.cldice_iters,
        cldice_class_ids=cldice_class_ids,
        include_background=False,
        ce_class_weights=ce_weights,
        cldice_channel_chunk=args.cldice_channel_chunk,
    )
    if target_skeleton_dir is not None:
        expected_class_ids = _resolve_expected_cldice_class_ids(
            num_classes=num_classes,
            include_background=False,
            cldice_class_ids=cldice_class_ids,
        )
        _validate_target_skeleton_dir(
            target_skeleton_dir=target_skeleton_dir,
            train_case_ids=train_case_ids,
            expected_class_ids=expected_class_ids,
        )
    print(
        "loss weights: "
        f"ce={args.ce_weight:.3f} dice={args.dice_weight:.3f} "
        f"tversky={args.tversky_weight:.3f} "
        f"(alpha={args.tversky_alpha:.3f}, beta={args.tversky_beta:.3f}, gamma={args.tversky_gamma:.3f}) "
        f"cldice={args.cldice_weight:.3f} cldice_iters={args.cldice_iters} "
        f"cldice_class_ids={cldice_class_ids} "
        f"cldice_channel_chunk={args.cldice_channel_chunk} "
        f"precomputed_target_skeletons={target_skeleton_dir}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6
    )

    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    # bf16 has fp32's dynamic range, so GradScaler is unnecessary (and unsupported
    # by torch.amp.GradScaler for bf16 on CUDA). Use a no-op scaler semantics by
    # passing enabled=False when amp_dtype is bf16.
    scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype is torch.float16))
        if amp_enabled
        else None
    )
    if args.amp and not amp_enabled:
        print("warning: --amp ignored because device is not CUDA.")
    if amp_enabled:
        print(f"AMP enabled: dtype={args.amp_dtype} grad_scaler={scaler.is_enabled() if scaler else False}")

    topbrain_requested = not args.topbrain_disable
    topbrain_runtime = TopBrainRuntime(track=args.topbrain_track) if topbrain_requested else None
    topbrain_enabled = bool(topbrain_runtime and topbrain_runtime.available)
    topbrain_subset_ids: list[str] = []
    topbrain_subset_ids_set: set[str] = set()
    if topbrain_requested and args.topbrain_per_epoch:
        topbrain_subset_ids = select_fixed_topbrain_subset(
            case_ids=val_case_ids,
            subset_size=args.topbrain_subset_size,
            seed=args.seed,
        )
        topbrain_subset_ids_set = set(topbrain_subset_ids)
    if topbrain_requested:
        print(
            "TopBrain validation: "
            f"enabled={topbrain_enabled} "
            f"track={args.topbrain_track} "
            f"when={'each_epoch' if args.topbrain_per_epoch else 'end_only'} "
            f"subset_size={len(topbrain_subset_ids_set)} "
            f"full_every_n={args.topbrain_eval_every_n_epochs} "
            f"full_max_cases={args.topbrain_full_max_cases}"
        )
        if topbrain_runtime and not topbrain_runtime.available:
            print(f"TopBrain disabled (import/runtime error): {topbrain_runtime.error_message}")

    resuming = args.resume is not None
    if resuming and not args.resume.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    # Resolve per-class display names from the ITK-SNAP labelmap so per-class
    # dice can be logged with readable keys (e.g. val_dice_c05_R-M1) and
    # rendered as a labelled bar chart in W&B.
    labelmap_path = Path(
        f"training_data_resampled/itksnap_labelmap_txt/labelmap_topbrain_{args.topbrain_track}.txt"
    )
    class_names = read_class_names_from_labelmap(labelmap_path, num_classes=num_classes)

    def _class_label(idx: int) -> str:
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        # Sanitize a bit so it reads cleanly in W&B chart axes / CSV headers.
        safe = name.replace(" ", "_").replace('"', "").strip("_") or f"class_{idx}"
        return safe

    # Only (re)write the CSV header if the file does not already contain rows.
    # When resuming, we append so prior epoch history is preserved.
    metrics_has_rows = metrics_csv.is_file() and metrics_csv.stat().st_size > 0
    if not metrics_has_rows:
        with metrics_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "val_loss",
                    "train_grad_norm",
                    *[f"train_loss_{name}" for name in DiceCELoss.COMPONENT_NAMES],
                    *[f"val_loss_{name}" for name in DiceCELoss.COMPONENT_NAMES],
                    "val_mean_fg_dice",
                    "val_mean_fg_dice_all_cases_present",
                    *[
                        f"val_dice_c{c:02d}_{_class_label(c)}"
                        for c in range(num_classes)
                    ],
                    *_topbrain_csv_columns("tb_subset"),
                    *_topbrain_csv_columns("tb_full"),
                ]
            )

    best_val_dice = -1.0
    start_epoch = 1
    final_weights_path = out_dir / "model_final_weights.pt"
    best_weights_path = out_dir / "model_best_weights.pt"

    # In-memory history used to build multi-line overlay charts at end-of-run
    # (train-vs-val total, per-component train, per-component val, per-class
    # dice history). Scalars are still logged per-epoch for live line charts,
    # this just adds dedicated custom panels once everything is finished.
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mean_fg_dice": [],
    }
    for name in DiceCELoss.COMPONENT_NAMES:
        history[f"train_loss_{name}"] = []
        history[f"val_loss_{name}"] = []
    for c in range(num_classes):
        history[f"val_dice_c{c:02d}"] = []

    if resuming:
        last_completed_epoch, best_val_dice = _load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = last_completed_epoch + 1
        print(
            f"resumed from {args.resume}: "
            f"last_completed_epoch={last_completed_epoch} "
            f"best_val_dice={best_val_dice:.6f} "
            f"next_epoch={start_epoch}"
        )
        if start_epoch > args.epochs:
            print(
                f"checkpoint is already past --epochs={args.epochs}; nothing "
                "to train. Increase --epochs to continue training."
            )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        run_topbrain_this_epoch = topbrain_enabled and args.topbrain_per_epoch
        run_topbrain_full = (
            run_topbrain_this_epoch
            and should_run_topbrain_full_eval(epoch, args.topbrain_eval_every_n_epochs)
        )
        topbrain_subset_acc = (
            topbrain_runtime.create_accumulator() if run_topbrain_this_epoch else None
        )
        topbrain_full_acc = (
            topbrain_runtime.create_accumulator() if run_topbrain_full and topbrain_runtime else None
        )
        full_case_ids = val_case_ids
        if run_topbrain_full and args.topbrain_full_max_cases is not None:
            full_case_ids = select_fixed_topbrain_subset(
                case_ids=val_case_ids,
                subset_size=min(args.topbrain_full_max_cases, len(val_case_ids)),
                seed=args.seed + 10_000,
            )
        full_case_ids_set = set(full_case_ids)

        def _topbrain_case_callback(case_id: str, pred_np: np.ndarray, label_path: Path) -> None:
            if topbrain_subset_acc is not None and case_id in topbrain_subset_ids_set:
                topbrain_subset_acc.add_case(
                    case_id=case_id,
                    pred_xyz=pred_np,
                    label_path=label_path,
                )
            if topbrain_full_acc is not None and case_id in full_case_ids_set:
                topbrain_full_acc.add_case(
                    case_id=case_id,
                    pred_xyz=pred_np,
                    label_path=label_path,
                )

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_dtype=amp_dtype if amp_enabled else None,
        )
        train_loss = train_stats["total"]
        train_grad_norm = train_stats.get("grad_norm", float("nan"))
        train_components = {
            name: train_stats[name]
            for name in DiceCELoss.COMPONENT_NAMES
            if name in train_stats
        }
        (
            val_loss,
            val_mean_fg_dice,
            val_mean_fg_dice_all_cases_present,
            per_class_dice,
            val_components,
        ) = validate_one_epoch(
            model=model,
            val_case_ids=val_case_ids,
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=patch_size,
            stride=val_stride,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            topbrain_case_callback=_topbrain_case_callback if run_topbrain_this_epoch else None,
        )

        subset_metrics = (
            topbrain_subset_acc.finalize()
            if topbrain_subset_acc is not None
            else empty_topbrain_metrics()
        )
        full_metrics = (
            topbrain_full_acc.finalize() if topbrain_full_acc is not None else empty_topbrain_metrics()
        )

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        def _fmt_component(src: dict[str, float], name: str) -> str:
            value = src.get(name)
            if value is None or not np.isfinite(value):
                return ""
            return f"{value:.6f}"

        with metrics_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{lr:.10f}",
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    "" if not np.isfinite(train_grad_norm) else f"{train_grad_norm:.6f}",
                    *[
                        _fmt_component(train_components, name)
                        for name in DiceCELoss.COMPONENT_NAMES
                    ],
                    *[
                        _fmt_component(val_components, name)
                        for name in DiceCELoss.COMPONENT_NAMES
                    ],
                    f"{val_mean_fg_dice:.6f}",
                    (
                        ""
                        if not np.isfinite(val_mean_fg_dice_all_cases_present)
                        else f"{val_mean_fg_dice_all_cases_present:.6f}"
                    ),
                    *[f"{d:.6f}" for d in per_class_dice],
                    *_topbrain_csv_values(
                        run_topbrain_this_epoch and bool(topbrain_subset_ids_set),
                        subset_metrics,
                    ),
                    *_topbrain_csv_values(run_topbrain_full, full_metrics),
                ]
            )

        improved_best = val_mean_fg_dice > best_val_dice
        if improved_best:
            best_val_dice = val_mean_fg_dice
            torch.save(model.state_dict(), best_weights_path)
            if wandb_run is not None:
                wandb_run.summary["best_val_mean_fg_dice"] = float(best_val_dice)
                wandb_run.summary["best_epoch"] = int(epoch)

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch:04d}.pt"
            _save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_dice=best_val_dice,
            )
            _rotate_periodic_checkpoints(out_dir, args.keep_last_checkpoints)

        elapsed = time.time() - epoch_start
        best_tag = " [new best]" if improved_best else ""
        print(
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={lr:.2e} grad_norm={train_grad_norm:.3f} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_mean_fg_dice={val_mean_fg_dice:.6f}"
            f"{best_tag} "
            "val_mean_fg_dice_all_cases_present="
            f"{val_mean_fg_dice_all_cases_present:.6f} "
            f"time={elapsed:.1f}s"
        )
        if train_components or val_components:
            train_bits = ", ".join(
                f"{name}={train_components[name]:.4f}"
                for name in DiceCELoss.COMPONENT_NAMES
                if name in train_components
            )
            val_bits = ", ".join(
                f"{name}={val_components[name]:.4f}"
                for name in DiceCELoss.COMPONENT_NAMES
                if name in val_components
            )
            print(f"loss_components: train[{train_bits}] val[{val_bits}]")
        print(
            "per_class_dice: "
            + ", ".join(
                f"c{idx:02d}_{_class_label(idx)}={d:.4f}"
                for idx, d in enumerate(per_class_dice)
            )
        )
        if run_topbrain_this_epoch:
            print(
                "topbrain_subset: "
                f"cases={int(subset_metrics['num_cases'])} "
                f"dice={subset_metrics['clsavg_dice']:.4f} "
                f"cldice={subset_metrics['clsavg_cldice']:.4f} "
                f"b0={subset_metrics['clsavg_b0']:.4f} "
                f"hd95={subset_metrics['clsavg_hd95']:.4f} "
                f"nb_err={subset_metrics['clsavg_invalid_neighbors']:.4f} "
                f"f1={subset_metrics['sideroad_f1']:.4f}"
            )
            if run_topbrain_full:
                print(
                    "topbrain_full: "
                    f"cases={int(full_metrics['num_cases'])} "
                    f"dice={full_metrics['clsavg_dice']:.4f} "
                    f"cldice={full_metrics['clsavg_cldice']:.4f} "
                    f"b0={full_metrics['clsavg_b0']:.4f} "
                    f"hd95={full_metrics['clsavg_hd95']:.4f} "
                    f"nb_err={full_metrics['clsavg_invalid_neighbors']:.4f} "
                    f"f1={full_metrics['sideroad_f1']:.4f}"
                )

        if wandb_run is not None:
            log_payload: dict[str, Any] = {
                "epoch": int(epoch),
                "time/epoch_sec": float(elapsed),
                "optim/lr": float(lr),
                "optim/grad_norm": float(train_grad_norm),
                # Top-level mirrors so the summary panel picks them up by default.
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mean_fg_dice": float(val_mean_fg_dice),
                # Namespaced metrics give clean per-section dashboards:
                #   train/loss_*  vs  val/loss_*    (train-vs-val overlay)
                #   val/dice/class/<name>           (per-class dice)
                "train/loss_total": float(train_loss),
                "val/loss_total": float(val_loss),
                "val/mean_fg_dice": float(val_mean_fg_dice),
                "val/best_mean_fg_dice": float(best_val_dice),
            }
            for name, value in train_components.items():
                if np.isfinite(value):
                    log_payload[f"train/loss_{name}"] = float(value)
            for name, value in val_components.items():
                if np.isfinite(value):
                    log_payload[f"val/loss_{name}"] = float(value)
            if np.isfinite(val_mean_fg_dice_all_cases_present):
                log_payload["val_mean_fg_dice_all_cases_present"] = float(
                    val_mean_fg_dice_all_cases_present
                )
                log_payload["val/mean_fg_dice_all_cases_present"] = float(
                    val_mean_fg_dice_all_cases_present
                )

            # Per-class Dice: log as both flat scalars (auto line charts) and a
            # bar chart table keyed by class name so W&B renders a labelled
            # per-class view.
            per_class_rows: list[list[Any]] = []
            for idx, d in enumerate(per_class_dice):
                safe_name = _class_label(idx)
                log_payload[f"val_dice_c{idx:02d}"] = float(d)
                log_payload[f"val/dice/class/{safe_name}"] = float(d)
                per_class_rows.append([f"{idx:02d}_{safe_name}", float(d)])
            per_class_table = wandb.Table(
                data=per_class_rows, columns=["class", "dice"]
            )
            log_payload["val/per_class_dice_bar"] = wandb.plot.bar(
                per_class_table,
                label="class",
                value="dice",
                title=f"Per-class Dice (epoch {epoch})",
            )

            # GPU memory (rank 0) so we can spot OOM cliffs on the same axis.
            if device.type == "cuda":
                try:
                    log_payload["sys/gpu_mem_allocated_gb"] = (
                        float(torch.cuda.memory_allocated(device)) / (1024**3)
                    )
                    log_payload["sys/gpu_mem_reserved_gb"] = (
                        float(torch.cuda.memory_reserved(device)) / (1024**3)
                    )
                    log_payload["sys/gpu_mem_max_allocated_gb"] = (
                        float(torch.cuda.max_memory_allocated(device)) / (1024**3)
                    )
                except Exception:
                    pass

            if run_topbrain_this_epoch:
                log_payload["tb_subset_ran"] = 1
                log_payload["tb_subset_num_cases"] = int(subset_metrics["num_cases"])
                for key in TOPBRAIN_METRIC_KEYS:
                    value = subset_metrics[key]
                    if np.isfinite(value):
                        log_payload[f"tb_subset_{key}"] = float(value)
                        log_payload[f"topbrain/subset/{key}"] = float(value)
            else:
                log_payload["tb_subset_ran"] = 0

            if run_topbrain_full:
                log_payload["tb_full_ran"] = 1
                log_payload["tb_full_num_cases"] = int(full_metrics["num_cases"])
                for key in TOPBRAIN_METRIC_KEYS:
                    value = full_metrics[key]
                    if np.isfinite(value):
                        log_payload[f"tb_full_{key}"] = float(value)
                        log_payload[f"topbrain/full/{key}"] = float(value)
            else:
                log_payload["tb_full_ran"] = 0

            wandb_run.log(log_payload, step=epoch)

        # Always append to in-memory history (even when W&B is disabled) so we
        # can keep an internal record and still build charts if a run later
        # opts in to W&B sync.
        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_mean_fg_dice"].append(float(val_mean_fg_dice))
        for name in DiceCELoss.COMPONENT_NAMES:
            history[f"train_loss_{name}"].append(
                float(train_components.get(name, float("nan")))
            )
            history[f"val_loss_{name}"].append(
                float(val_components.get(name, float("nan")))
            )
        for c in range(num_classes):
            value = per_class_dice[c] if c < len(per_class_dice) else float("nan")
            history[f"val_dice_c{c:02d}"].append(float(value))

    final_epoch_ran_topbrain_full = (
        topbrain_enabled
        and args.topbrain_per_epoch
        and should_run_topbrain_full_eval(args.epochs, args.topbrain_eval_every_n_epochs)
    )
    need_final_topbrain = topbrain_enabled and topbrain_runtime and (
        not args.topbrain_per_epoch or not final_epoch_ran_topbrain_full
    )
    if need_final_topbrain:
        print(
            "running TopBrain validation after training "
            "(end-of-run metrics on the validation set)."
        )
        final_topbrain_acc = topbrain_runtime.create_accumulator()
        final_case_ids = val_case_ids
        if args.topbrain_full_max_cases is not None:
            final_case_ids = select_fixed_topbrain_subset(
                case_ids=val_case_ids,
                subset_size=min(args.topbrain_full_max_cases, len(val_case_ids)),
                seed=args.seed + 10_000,
            )
        final_case_ids_set = set(final_case_ids)

        def _final_topbrain_case_callback(case_id: str, pred_np: np.ndarray, label_path: Path) -> None:
            if case_id in final_case_ids_set:
                final_topbrain_acc.add_case(
                    case_id=case_id,
                    pred_xyz=pred_np,
                    label_path=label_path,
                )

        (
            _final_val_loss,
            _final_val_mean_fg_dice,
            _final_val_mean_fg_dice_all_cases_present,
            _final_per_class_dice,
            _final_val_components,
        ) = validate_one_epoch(
            model=model,
            val_case_ids=final_case_ids,
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=patch_size,
            stride=val_stride,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            topbrain_case_callback=_final_topbrain_case_callback,
        )
        final_metrics = final_topbrain_acc.finalize()
        print(
            "topbrain_final: "
            f"cases={int(final_metrics['num_cases'])} "
            f"dice={final_metrics['clsavg_dice']:.4f} "
            f"cldice={final_metrics['clsavg_cldice']:.4f} "
            f"b0={final_metrics['clsavg_b0']:.4f} "
            f"hd95={final_metrics['clsavg_hd95']:.4f} "
            f"nb_err={final_metrics['clsavg_invalid_neighbors']:.4f} "
            f"f1={final_metrics['sideroad_f1']:.4f}"
        )

    torch.save(model.state_dict(), final_weights_path)
    if wandb_run is not None:
        wandb_run.summary["best_val_mean_fg_dice"] = float(best_val_dice)
        _log_wandb_summary_charts(
            wandb_run=wandb_run,
            history=history,
            class_labels=[_class_label(c) for c in range(num_classes)],
        )
        artifact_name = f"{out_dir.name}-{wandb_run.id}-artifacts"
        artifact = wandb.Artifact(name=artifact_name, type="training_outputs")
        for path in (best_weights_path, final_weights_path, metrics_csv):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(f"\nTraining complete. Best val_mean_fg_dice={best_val_dice:.6f}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved final weights: {final_weights_path}")
    if best_weights_path.is_file():
        print(f"Saved best-val weights: {best_weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())