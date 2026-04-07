from __future__ import annotations

"""Offline preprocessing script that resamples data and normalizes CT volumes."""

import argparse
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from data_utils import preprocess_ct, resample_to_spacing, spacing_from_affine


def _make_target_affine(old_affine: np.ndarray, target_spacing: tuple[float, float, float]) -> np.ndarray:
    """
    Keep orientation/origin from old affine and only replace voxel spacings.
    """
    new_affine = old_affine.copy()
    basis = old_affine[:3, :3]
    col_norms = np.linalg.norm(basis, axis=0)
    if np.any(col_norms <= 0):
        raise ValueError("Invalid affine basis; cannot derive orientation vectors.")
    unit_dirs = basis / col_norms
    new_affine[:3, :3] = unit_dirs * np.asarray(target_spacing, dtype=np.float32)
    return new_affine


def _resample_pair(
    image_path: Path,
    label_path: Path,
    out_image_path: Path,
    out_label_path: Path,
    target_spacing: tuple[float, float, float],
    normalize_ct: bool,
) -> None:
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))

    image = np.asanyarray(image_nii.dataobj).astype(np.float32)
    label = np.asanyarray(label_nii.dataobj).astype(np.int64)

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: image={image.shape}, label={label.shape}"
        )

    # Read spacing from the image affine so preprocessing stays data-driven.
    spacing = spacing_from_affine(image_nii.affine)
    image_rs = resample_to_spacing(image, spacing, target_spacing, is_label=False)
    label_rs = resample_to_spacing(label, spacing, target_spacing, is_label=True)
    if normalize_ct:
        image_rs = preprocess_ct(image_rs)
    label_rs = np.rint(label_rs).astype(np.int16)

    out_affine = _make_target_affine(image_nii.affine, target_spacing)
    img_hdr = image_nii.header.copy()
    lbl_hdr = label_nii.header.copy()
    img_hdr.set_zooms((*target_spacing, *img_hdr.get_zooms()[3:]))
    lbl_hdr.set_zooms((*target_spacing, *lbl_hdr.get_zooms()[3:]))

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image_rs.astype(np.float32), out_affine, img_hdr), str(out_image_path))
    nib.save(nib.Nifti1Image(label_rs, out_affine, lbl_hdr), str(out_label_path))

    print(
        f"[ok] {image_path.stem} spacing {np.round(spacing, 4)} -> {target_spacing} "
        f"shape {image.shape} -> {image_rs.shape}"
    )


def _resample_dataset_dirs(
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    target_spacing: tuple[float, float, float],
) -> int:
    out_image_dir = output_root / image_dir.name
    out_label_dir = output_root / label_dir.name

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_files = sorted(image_dir.glob("*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    n_done = 0
    normalize_ct = image_dir.name.endswith("_ct")
    print(f"\n[dataset] {image_dir.name} + {label_dir.name}")
    for image_path in image_files:
        case_id = image_path.name.replace("_0000.nii.gz", "")
        label_path = label_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            print(f"[skip] missing label for {case_id}: {label_path}")
            continue

        out_image_path = out_image_dir / image_path.name
        out_label_path = out_label_dir / label_path.name
        _resample_pair(
            image_path=image_path,
            label_path=label_path,
            out_image_path=out_image_path,
            out_label_path=out_label_path,
            target_spacing=target_spacing,
            normalize_ct=normalize_ct,
        )
        n_done += 1

    print(f"[done] {n_done} pairs -> {out_image_dir} / {out_label_dir}")
    return n_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline isotropic resampling for TopBrain datasets."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("training_data"),
        help="Input root that contains imagesTr_* and labelsTr_* folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_data_resampled"),
        help="Output root with mirrored imagesTr_*/labelsTr_* folders.",
    )
    parser.add_argument(
        "--sx",
        type=float,
        default=0.6,
        help="Target spacing in x (mm).",
    )
    parser.add_argument(
        "--sy",
        type=float,
        default=0.6,
        help="Target spacing in y (mm).",
    )
    parser.add_argument(
        "--sz",
        type=float,
        default=0.6,
        help="Target spacing in z (mm).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Optional dataset suffix to process (repeatable), e.g. "
            "--dataset topbrain_ct --dataset topbrain_mr. "
            "If omitted, all imagesTr_* with matching labelsTr_* are processed."
        ),
    )
    parser.add_argument(
        "--copy-metadata",
        action="store_true",
        help="Copy split/, itksnap_labelmap_txt/, README.txt, and License.txt to output root if present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_spacing = (args.sx, args.sy, args.sz)
    input_root = args.input_root
    output_root = args.output_root
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    if args.dataset:
        suffixes = args.dataset
    else:
        suffixes = sorted(
            p.name.replace("imagesTr_", "")
            for p in input_root.glob("imagesTr_*")
            if (input_root / p.name.replace("imagesTr_", "labelsTr_")).is_dir()
        )

    if not suffixes:
        raise FileNotFoundError(
            f"No dataset pairs found under {input_root}. Expected imagesTr_* with matching labelsTr_*."
        )

    total = 0
    for suffix in suffixes:
        image_dir = input_root / f"imagesTr_{suffix}"
        label_dir = input_root / f"labelsTr_{suffix}"
        total += _resample_dataset_dirs(
            image_dir=image_dir,
            label_dir=label_dir,
            output_root=output_root,
            target_spacing=target_spacing,
        )

    if args.copy_metadata:
        for rel in ("split", "itksnap_labelmap_txt", "README.txt", "License.txt"):
            src = input_root / rel
            dst = output_root / rel
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"[copy] {src} -> {dst}")
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"[copy] {src} -> {dst}")

    print(f"\nDone. Resampled total pairs: {total}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
