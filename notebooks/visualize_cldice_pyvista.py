"""
3D PyVista visualization for clDice-style skeletons.

Usage (from project root):
  .venv/bin/python notebooks/visualize_cldice_pyvista.py --case-id topcow_ct_005 --show
  .venv/bin/python notebooks/visualize_cldice_pyvista.py --case-id topcow_ct_005 --cldice-iters 5
  .venv/bin/python notebooks/visualize_cldice_pyvista.py --case-id topcow_ct_005 --label-id 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

project_root = Path(__file__).resolve().parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root / "TopBrain_Eval_Metrics-master") not in sys.path:
    sys.path.insert(0, str(project_root / "TopBrain_Eval_Metrics-master"))
if str(project_root / "clDice-master" / "cldice_metric") not in sys.path:
    sys.path.insert(0, str(project_root / "clDice-master" / "cldice_metric"))
if str(project_root / "clDice-master" / "cldice_loss" / "pytorch") not in sys.path:
    sys.path.insert(0, str(project_root / "clDice-master" / "cldice_loss" / "pytorch"))

from losses import DiceCELoss

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "PyVista is not installed. Install with: pip install pyvista vtk"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize clDice soft skeletons in 3D with PyVista."
    )
    parser.add_argument("--case-id", default="topcow_ct_005")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("training_data_resampled"),
        help="Root containing imagesTr_topbrain_ct and labelsTr_topbrain_ct.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Patch size (x y z) centered around foreground.",
    )
    parser.add_argument(
        "--cldice-iters",
        type=int,
        default=12,
        help="Soft skeletonization iterations.",
    )
    parser.add_argument(
        "--label-id",
        type=int,
        default=None,
        help="Optional single label id. If omitted, uses merged foreground (label > 0).",
    )
    parser.add_argument(
        "--skeleton-thresh",
        type=float,
        default=0.10,
        help="Threshold used to binarize the soft skeleton for 3D point rendering.",
    )
    parser.add_argument(
        "--skeleton-step",
        type=int,
        default=1,
        help="Subsample factor for skeleton points (1 = keep all, 2 = every second point).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("notebooks/debug_cldice_output"),
    )
    parser.add_argument(
        "--screenshot",
        default="cldice_pyvista.png",
        help="Screenshot filename saved in out-dir.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive window. Without this flag, runs off-screen screenshot only.",
    )
    return parser.parse_args()


def _load_case_patch(
    data_root: Path,
    case_id: str,
    patch_size: tuple[int, int, int],
    label_id: int | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image_path = data_root / "imagesTr_topbrain_ct" / f"{case_id}_0000.nii.gz"
    label_path = data_root / "labelsTr_topbrain_ct" / f"{case_id}.nii.gz"

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))

    image = np.asanyarray(image_nii.dataobj).astype(np.float32)
    label = np.asanyarray(label_nii.dataobj).astype(np.int64)
    spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:3])

    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch image={image.shape}, label={label.shape}")

    image = np.clip(image, -100.0, 400.0)
    image = (image + 100.0) / 500.0

    if label_id is None:
        fg_mask = label > 0
    else:
        fg_mask = label == label_id

    fg_idx = np.argwhere(fg_mask)
    if fg_idx.size == 0:
        if label_id is None:
            raise ValueError(f"No foreground found in {case_id}.")
        raise ValueError(f"Label {label_id} not found in {case_id}.")

    center = fg_idx.mean(axis=0).astype(int)
    starts = [
        max(0, min(c - p // 2, s - p))
        for c, p, s in zip(center, patch_size, image.shape)
    ]
    sx, sy, sz = starts
    px, py, pz = patch_size

    image_patch = image[sx : sx + px, sy : sy + py, sz : sz + pz]
    fg_patch = fg_mask[sx : sx + px, sy : sy + py, sz : sz + pz].astype(np.float32)
    return image_patch, fg_patch, spacing


def _make_cell_grid(mask_xyz: np.ndarray, spacing_xyz: tuple[float, float, float]) -> pv.ImageData:
    grid = pv.ImageData()
    grid.dimensions = np.array(mask_xyz.shape) + 1
    grid.spacing = spacing_xyz
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["mask"] = mask_xyz.ravel(order="F")
    return grid


def _build_skeleton_points(
    *,
    skel_bin: np.ndarray,
    fg_np: np.ndarray,
    spacing: tuple[float, float, float],
    step: int,
) -> tuple[pv.PolyData | None, np.ndarray]:
    edt_mm = distance_transform_edt(fg_np > 0.5, sampling=spacing)
    diam_mm = 2.0 * edt_mm[skel_bin]
    if diam_mm.size == 0:
        return None, diam_mm

    skel_ijk = np.argwhere(skel_bin)
    skel_xyz = skel_ijk.astype(np.float32) * np.asarray(spacing, dtype=np.float32)
    if step > 1:
        keep = np.arange(len(skel_xyz))[::step]
        skel_xyz = skel_xyz[keep]
        diam_mm = diam_mm[keep]

    skel_points = pv.PolyData(skel_xyz)
    skel_points["diameter_mm"] = diam_mm
    return skel_points, diam_mm


def _resolve_eval_skeletonize():
    """Return the eval-style 3D skeletonize callable.

    Prefer TopBrain's implementation. If unavailable (for example due to missing
    optional deps in local venv), fall back to skimage Lee skeletonization,
    which matches the TopBrain compatibility path.
    """
    try:
        from topbrain25_eval.metrics.cls_avg_clDice import skeletonize_3d as topbrain_skeletonize_3d  # pyright: ignore[reportMissingImports]

        return topbrain_skeletonize_3d, "topbrain25_eval.metrics.cls_avg_clDice.skeletonize_3d"
    except Exception as exc:
        from skimage.morphology import skeletonize as skimage_skeletonize  # pyright: ignore[reportMissingImports]

        def _fallback_skeletonize_3d(image: np.ndarray) -> np.ndarray:
            return skimage_skeletonize(image, method="lee")

        print(
            "TopBrain import unavailable; falling back to skimage Lee skeletonization "
            f"for eval view. Reason: {exc}"
        )
        return _fallback_skeletonize_3d, "skimage.morphology.skeletonize(method='lee')"


def _resolve_cldice_master_soft_skeletonize():
    """Return clDice-master's differentiable SoftSkeletonize callable."""
    try:
        from soft_skeleton import SoftSkeletonize  # pyright: ignore[reportMissingImports]

        def _cldice_master_soft(image: np.ndarray, iters: int) -> np.ndarray:
            x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            module = SoftSkeletonize(num_iter=iters)
            with torch.no_grad():
                skel = module(x)[0, 0].cpu().numpy()
            return skel

        return _cldice_master_soft, "clDice-master.cldice_loss.pytorch.SoftSkeletonize"
    except Exception as exc:
        def _fallback_soft(image: np.ndarray, iters: int) -> np.ndarray:
            x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                skel = DiceCELoss._soft_skeletonize(x, iters)[0, 0].cpu().numpy()
            return skel

        print(f"clDice-master SoftSkeletonize import unavailable; falling back to DiceCELoss soft skeleton. Reason: {exc}")
        return _fallback_soft, "DiceCELoss._soft_skeletonize (fallback)"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    skeletonize_eval_3d, eval_impl_name = _resolve_eval_skeletonize()
    soft_skeletonize_cldice_master, cldice_master_impl_name = _resolve_cldice_master_soft_skeletonize()

    image_np, fg_np, spacing = _load_case_patch(
        data_root=args.data_root,
        case_id=args.case_id,
        patch_size=tuple(args.patch_size),
        label_id=args.label_id,
    )

    fg_t = torch.from_numpy(fg_np).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        ours_soft = DiceCELoss._soft_skeletonize(fg_t, args.cldice_iters)[0, 0].cpu().numpy()
        cldice_master_soft = soft_skeletonize_cldice_master(fg_np, args.cldice_iters)
    ours_bin = ours_soft >= float(args.skeleton_thresh)
    eval_bin = skeletonize_eval_3d((fg_np > 0.5).astype(bool)).astype(bool)
    cldice_master_bin = cldice_master_soft >= float(args.skeleton_thresh)

    ours_points, ours_diam_mm = _build_skeleton_points(
        skel_bin=ours_bin,
        fg_np=fg_np,
        spacing=spacing,
        step=args.skeleton_step,
    )
    eval_points, eval_diam_mm = _build_skeleton_points(
        skel_bin=eval_bin,
        fg_np=fg_np,
        spacing=spacing,
        step=args.skeleton_step,
    )
    cldice_master_points, cldice_master_diam_mm = _build_skeleton_points(
        skel_bin=cldice_master_bin,
        fg_np=fg_np,
        spacing=spacing,
        step=args.skeleton_step,
    )

    if ours_points is None and eval_points is None and cldice_master_points is None:
        print("No skeleton points found for any method. Try lowering --skeleton-thresh.")
        return 1

    grid = _make_cell_grid(fg_np, spacing)
    vessel_surface = grid.threshold(value=0.5, scalars="mask").extract_surface(
        algorithm="dataset_surface"
    )

    plotter = pv.Plotter(shape=(1, 3), off_screen=not args.show)

    plotter.subplot(0, 0)
    plotter.add_axes(line_width=2)
    plotter.add_text(
        (
            "Ours: DiceCELoss._soft_skeletonize\n"
            f"iters={args.cldice_iters}  thresh={args.skeleton_thresh:.2f}\n"
            f"skeleton_voxels={int(ours_bin.sum())}"
        ),
        font_size=9,
    )
    plotter.add_mesh(
        vessel_surface,
        color="seagreen",
        opacity=0.18,
        smooth_shading=True,
        name="vessel_surface_ours",
    )
    if ours_points is not None:
        plotter.add_mesh(
            ours_points,
            scalars="diameter_mm",
            cmap="plasma",
            point_size=5.0,
            render_points_as_spheres=True,
            name="ours_skeleton_points",
            scalar_bar_args={"title": "Diameter (mm)"},
        )

    plotter.subplot(0, 1)
    plotter.add_axes(line_width=2)
    plotter.add_text(
        (
            f"Eval: {eval_impl_name}\n"
            f"skeleton_voxels={int(eval_bin.sum())}"
        ),
        font_size=9,
    )
    plotter.add_mesh(
        vessel_surface,
        color="seagreen",
        opacity=0.18,
        smooth_shading=True,
        name="vessel_surface_eval",
    )
    if eval_points is not None:
        plotter.add_mesh(
            eval_points,
            scalars="diameter_mm",
            cmap="viridis",
            point_size=5.0,
            render_points_as_spheres=True,
            name="eval_skeleton_points",
            scalar_bar_args={"title": "Diameter (mm)"},
        )

    plotter.subplot(0, 2)
    plotter.add_axes(line_width=2)
    plotter.add_text(
        (
            f"clDice-master soft: {cldice_master_impl_name}\n"
            f"iters={args.cldice_iters}  thresh={args.skeleton_thresh:.2f}\n"
            f"skeleton_voxels={int(cldice_master_bin.sum())}"
        ),
        font_size=9,
    )
    plotter.add_mesh(
        vessel_surface,
        color="seagreen",
        opacity=0.18,
        smooth_shading=True,
        name="vessel_surface_cldice_master",
    )
    if cldice_master_points is not None:
        plotter.add_mesh(
            cldice_master_points,
            scalars="diameter_mm",
            cmap="magma",
            point_size=5.0,
            render_points_as_spheres=True,
            name="cldice_master_skeleton_points",
            scalar_bar_args={"title": "Diameter (mm)"},
        )

    plotter.link_views()

    screenshot_path = args.out_dir / args.screenshot
    plotter.show(screenshot=str(screenshot_path), auto_close=not args.show)
    if args.show:
        plotter.close()

    print("=" * 64)
    print("PyVista clDice visualization")
    print("=" * 64)
    print(f"Case:            {args.case_id}")
    print(f"Label:           {'all foreground (>0)' if args.label_id is None else args.label_id}")
    print(f"Patch shape:     {fg_np.shape}")
    print(f"Spacing (mm):    {spacing}")
    print(f"clDice iters:    {args.cldice_iters}")
    print(f"Skel threshold:  {args.skeleton_thresh}")
    print(f"Ours voxels:     {int(ours_bin.sum())}")
    print(f"Eval voxels:     {int(eval_bin.sum())}")
    print(f"clDice voxels:   {int(cldice_master_bin.sum())}")
    if ours_diam_mm.size > 0:
        print(f"Ours max diam:   {float(np.max(ours_diam_mm)):.3f} mm")
    if eval_diam_mm.size > 0:
        print(f"Eval max diam:   {float(np.max(eval_diam_mm)):.3f} mm")
    if cldice_master_diam_mm.size > 0:
        print(f"clDice max diam: {float(np.max(cldice_master_diam_mm)):.3f} mm")
    print(f"Screenshot:      {screenshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
