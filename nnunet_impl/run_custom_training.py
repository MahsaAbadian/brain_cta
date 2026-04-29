#!/usr/bin/env python3
"""Run nnU-Net v2 with project-local custom trainers.

This mirrors ``nnUNetv2_train`` for the common single-GPU/CPU case, but keeps
custom trainer code in this repository instead of copying files into the
installed ``nnunetv2`` package.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from typing import Any

import torch

from nnunet_impl.custom_trainer import nnUNetTrainerTopBrainClDice


CUSTOM_TRAINERS = {
    nnUNetTrainerTopBrainClDice.__name__: nnUNetTrainerTopBrainClDice,
}


def _patch_trainer_lookup() -> None:
    import nnunetv2.run.run_training as run_training_module

    original_lookup = run_training_module.recursive_find_python_class

    def lookup(folder: str, class_name: str, current_module: str) -> Any:
        if class_name in CUSTOM_TRAINERS:
            return CUSTOM_TRAINERS[class_name]
        return original_lookup(folder, class_name, current_module)

    run_training_module.recursive_find_python_class = lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nnU-Net v2 with TopBrain project-local custom trainers."
    )
    parser.add_argument("dataset_name_or_id")
    parser.add_argument("configuration")
    parser.add_argument("fold")
    parser.add_argument("-tr", default="nnUNetTrainerTopBrainClDice")
    parser.add_argument("-p", default="nnUNetPlans")
    parser.add_argument("-pretrained_weights", default=None)
    parser.add_argument("-num_gpus", type=int, default=1)
    parser.add_argument("--npz", action="store_true")
    parser.add_argument("--c", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--val_best", action="store_true")
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument(
        "-device",
        choices=("cpu", "cuda", "mps"),
        default="cuda",
        help="Same meaning as nnUNetv2_train -device.",
    )
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    if device_name == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    return torch.device("mps")


def main() -> int:
    args = parse_args()
    if args.num_gpus != 1 and args.tr in CUSTOM_TRAINERS:
        raise NotImplementedError(
            "Project-local custom trainers are wired for single-process training. "
            "Use -num_gpus 1, or install the trainer into the nnunetv2 package "
            "for DDP discovery."
        )

    _patch_trainer_lookup()

    from nnunetv2.run.run_training import run_training

    run_training(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        args.tr,
        args.p,
        args.pretrained_weights,
        args.num_gpus,
        args.npz,
        args.c,
        args.val,
        args.disable_checkpointing,
        args.val_best,
        device=_resolve_device(args.device),
    )
    return 0


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    raise SystemExit(main())
