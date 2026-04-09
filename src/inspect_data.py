# Run from project root with the project venv:
#   .venv/bin/python src/inspect_data.py
#   .venv/bin/python src/inspect_data.py --data-root training_data_resampled --already-preprocessed
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import matplotlib
import re



project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

matplotlib.use("Agg")  # no display needed; saves to PNG only
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Inspect raw or preprocessed TopBrain cases.")
parser.add_argument("--case-id", default="topcow_ct_005")
parser.add_argument("--data-root", default="training_data")
parser.add_argument(
    "--already-preprocessed",
    action="store_true",
    help="Use this when inspecting data saved by src/preprocess_resample.py.",
)
parser.add_argument(
    "--report-class-volumes",
    action="store_true",
    help=(
        "Compute dataset-level average class volumes from labelsTr_topbrain_ct and "
        "print a per-class table (voxels and mm^3)."
    ),
)
args = parser.parse_args()


def _parse_itksnap_label_names(labelmap_path: Path) -> dict[int, str]:
    """Parse class index -> name from an ITK-SNAP labelmap file."""
    if not labelmap_path.exists():
        return {}

    names: dict[int, str] = {}
    line_re = re.compile(r'^\s*(\d+)\s+.*"([^"]+)"\s*$')
    for line in labelmap_path.read_text().splitlines():
        m = line_re.match(line)
        if m is None:
            continue
        idx = int(m.group(1))
        names[idx] = m.group(2)
    return names


def compute_average_class_volumes(
    *,
    labels_dir: Path,
    labelmap_path: Path | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute per-class average volume across all CTA label files.

    Returns one row per class with:
      - avg_voxels_per_case
      - avg_mm3_per_case
      - present_cases / num_cases
    """
    label_files = sorted(labels_dir.glob('*.nii.gz'))
    if not label_files:
        raise FileNotFoundError(f'No label files found in {labels_dir}')

    # First pass: discover max class id across dataset.
    max_class = 0
    for p in label_files:
        lbl = np.asanyarray(nib.load(str(p)).dataobj)
        if lbl.size == 0:
            continue
        max_class = max(max_class, int(np.max(lbl)))

    num_classes = max_class + 1
    total_voxels = np.zeros(num_classes, dtype=np.float64)
    total_mm3 = np.zeros(num_classes, dtype=np.float64)
    present_cases = np.zeros(num_classes, dtype=np.int64)

    for p in label_files:
        nii = nib.load(str(p))
        lbl = np.asanyarray(nii.dataobj).astype(np.int64)
        spacing = np.sqrt(np.sum(nii.affine[:3, :3] ** 2, axis=0))
        voxel_mm3 = float(np.prod(spacing))

        cls_ids, counts = np.unique(lbl, return_counts=True)
        for cls_id, cnt in zip(cls_ids.astype(int), counts.astype(np.int64)):
            if cls_id < 0 or cls_id >= num_classes:
                continue
            total_voxels[cls_id] += float(cnt)
            total_mm3[cls_id] += float(cnt) * voxel_mm3
            present_cases[cls_id] += 1

    num_cases = len(label_files)
    names = _parse_itksnap_label_names(labelmap_path) if labelmap_path else {}

    rows: list[dict[str, float | int | str]] = []
    for c in range(num_classes):
        rows.append(
            {
                'class_id': c,
                'class_name': names.get(c, ''),
                'present_cases': int(present_cases[c]),
                'num_cases': num_cases,
                'avg_voxels_per_case': float(total_voxels[c] / num_cases),
                'avg_mm3_per_case': float(total_mm3[c] / num_cases),
            }
        )
    return rows


def print_average_class_volumes(rows: list[dict[str, float | int | str]]) -> None:
    """Pretty-print per-class average volume table to stdout."""
    print('\n=== Average Class Volumes (dataset-level, per case) ===')
    print('class  name           present   avg_voxels/case   avg_mm3/case')
    print('-----  -------------  -------  -----------------  --------------')
    for r in rows:
        cls_id = int(r['class_id'])
        name = str(r['class_name']) if r['class_name'] else '-'
        present = f"{int(r['present_cases'])}/{int(r['num_cases'])}"
        avg_vox = float(r['avg_voxels_per_case'])
        avg_mm3 = float(r['avg_mm3_per_case'])
        print(f"{cls_id:>5}  {name[:13]:<13}  {present:>7}  {avg_vox:>17.1f}  {avg_mm3:>14.1f}")


case_id = args.case_id
data_root = project_root / args.data_root
out_dir = project_root / "data_inspection"
out_dir.mkdir(parents=True, exist_ok=True)

if args.report_class_volumes:
    labelmap_path = data_root / "itksnap_labelmap_txt" / "labelmap_topbrain_ct.txt"
    labels_dir = data_root / "labelsTr_topbrain_ct"
    rows = compute_average_class_volumes(
        labels_dir=labels_dir,
        labelmap_path=labelmap_path if labelmap_path.exists() else None,
    )
    print_average_class_volumes(rows)


img_path = data_root / "imagesTr_topbrain_ct" / f"{case_id}_0000.nii.gz"
lbl_path = data_root / "labelsTr_topbrain_ct" / f"{case_id}.nii.gz"

img_nii = nib.load(str(img_path))
lbl_nii = nib.load(str(lbl_path))
img = np.asanyarray(img_nii.dataobj)
lbl = np.asanyarray(lbl_nii.dataobj)

print(f"Case: {case_id}")
print("Image shape:", img.shape, "dtype:", img.dtype)
print("Label shape:", lbl.shape, "dtype:", lbl.dtype)
print("Shape match?:", img.shape == lbl.shape)
u = np.unique(lbl)
print("Label min/max:", u.min(), u.max(), "n_unique:", len(u))

# Affine: voxel -> world (spacing and orientation)
affine = img_nii.affine
print("Affine shape:", affine.shape)
spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
print("Spacing (mm):", spacing)

# ---------------------------------------------------------------------------
# Visualize center slices (axial, coronal, sagittal)
# ---------------------------------------------------------------------------
if args.already_preprocessed:
    vmin, vmax = 0.0, 1.0
else:
    # For raw CT, choose a display window using center/width convention:
    # values below vmin look black, values above vmax look white.
    # This only affects visualization in matplotlib, not the stored image data.
    w_center = 200
    w_width = 800
    vmin = w_center - w_width / 2
    vmax = w_center + w_width / 2

nx, ny, nz = img.shape
mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(
    img[mid_x, :, :].T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    # aspect=1: one screen pixel = one data voxel, so anisotropy is visible
    aspect=1,
)
axes[0].set_title("Sagittal (x mid)")
axes[0].axis("off")

axes[1].imshow(
    img[:, mid_y, :].T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    aspect=1,
)
axes[1].set_title("Coronal (y mid)")
axes[1].axis("off")

axes[2].imshow(
    img[:, :, mid_z].T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    aspect=1,
)
axes[2].set_title("Axial (z mid)")
axes[2].axis("off")

plt.suptitle(f"CTA: {img_path.name}")
plt.tight_layout()
out_path = out_dir / "inspect_data_slices.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path}")

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
for ax, (sl, title) in zip(
    axes2,
    [
        (lbl[mid_x, :, :].T, "Sagittal (labels)"),
        (lbl[:, mid_y, :].T, "Coronal (labels)"),
        (lbl[:, :, mid_z].T, "Axial (labels)"),
    ],
):
    ax.imshow(sl, cmap="nipy_spectral", origin="lower", vmin=0, vmax=40, aspect=1)
    ax.set_title(title)
    ax.axis("off")
plt.suptitle(f"Labels: {case_id}.nii.gz")
plt.tight_layout()
out_labels = out_dir / "inspect_data_labels.png"
plt.savefig(out_labels, dpi=120, bbox_inches="tight")
print(f"Saved: {out_labels}")

# Overlay labels on top of CTA for alignment check
fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
for ax, (img_sl, lbl_sl, title) in zip(
    axes3,
    [
        (img[mid_x, :, :].T, lbl[mid_x, :, :].T, "Sagittal (overlay)"),
        (img[:, mid_y, :].T, lbl[:, mid_y, :].T, "Coronal (overlay)"),
        (img[:, :, mid_z].T, lbl[:, :, mid_z].T, "Axial (overlay)"),
    ],
):
    ax.imshow(img_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=1)
    # Hide background class (0) so only vessel labels are overlaid
    lbl_masked = np.ma.masked_where(lbl_sl == 0, lbl_sl)
    ax.imshow(
        lbl_masked,
        cmap="nipy_spectral",
        origin="lower",
        vmin=1,
        vmax=40,
        alpha=0.45,
        aspect=1,
    )
    ax.set_title(title)
    ax.axis("off")
plt.suptitle(f"Overlay: {case_id}")
plt.tight_layout()
out_overlay = out_dir / "inspect_data_overlay.png"
plt.savefig(out_overlay, dpi=120, bbox_inches="tight")
print(f"Saved: {out_overlay}")

# plt.show()  # uncomment to open windows when running in a GUI

if args.already_preprocessed:
    # In preprocessed mode, the loaded image already contains the offline
    # normalization output from src/preprocess_resample.py.
    img_pre = img.astype(np.float32)
    print("Loaded preprocessed min/max/mean:", img_pre.min(), img_pre.max(), img_pre.mean())

    # Compare against the matching raw CT case when it exists.
    raw_img_path = project_root / "training_data" / "imagesTr_topbrain_ct" / f"{case_id}_0000.nii.gz"
    if raw_img_path.exists():
        raw_img_nii = nib.load(str(raw_img_path))
        raw_img = np.asanyarray(raw_img_nii.dataobj).astype(np.float32)
        raw_nx, raw_ny, raw_nz = raw_img.shape
        raw_mid_x, raw_mid_y, raw_mid_z = raw_nx // 2, raw_ny // 2, raw_nz // 2
        raw_w_center = 200
        raw_w_width = 800
        raw_vmin = raw_w_center - raw_w_width / 2
        raw_vmax = raw_w_center + raw_w_width / 2

        fig_compare, axes_compare = plt.subplots(2, 3, figsize=(12, 8))
        compare_specs = [
            ("Sagittal", raw_img[raw_mid_x, :, :].T, img_pre[mid_x, :, :].T),
            ("Coronal",  raw_img[:, raw_mid_y, :].T,  img_pre[:, mid_y, :].T),
            ("Axial",    raw_img[:, :, raw_mid_z].T,  img_pre[:, :, mid_z].T),
        ]
        for col, (title, raw_sl, pre_sl) in enumerate(compare_specs):
            # Raw row: aspect=1 so the anisotropic voxel grid is visible as-is
            axes_compare[0, col].imshow(
                raw_sl,
                cmap="gray",
                origin="lower",
                vmin=raw_vmin,
                vmax=raw_vmax,
                aspect=1,
            )
            axes_compare[0, col].set_title(f"{title} (raw)")
            axes_compare[0, col].axis("off")

            axes_compare[1, col].imshow(
                pre_sl, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, aspect=1
            )
            axes_compare[1, col].set_title(f"{title} (preprocessed)")
            axes_compare[1, col].axis("off")

        plt.suptitle(f"Raw vs Preprocessed: {case_id}")
        plt.tight_layout()
        out_compare = out_dir / "inspect_data_raw_vs_preprocessed.png"
        plt.savefig(out_compare, dpi=120, bbox_inches="tight")
        print(f"Saved: {out_compare}")
    else:
        print(f"Raw comparison image not found: {raw_img_path}")

    # visualize preprocessed center slices
    fig4, axes4 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (sl, title) in zip(
        axes4,
        [
            (img_pre[mid_x, :, :].T, "Sagittal (preprocessed)"),
            (img_pre[:, mid_y, :].T, "Coronal (preprocessed)"),
            (img_pre[:, :, mid_z].T, "Axial (preprocessed)"),
        ],
    ):
        ax.imshow(sl, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, aspect=1)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Preprocessed CTA: {case_id}")
    plt.tight_layout()
    out_pre = out_dir / "inspect_data_preprocessed.png"
    plt.savefig(out_pre, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_pre}")

    fig5, axes5 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (img_sl, lbl_sl, title) in zip(
        axes5,
        [
            (img_pre[mid_x, :, :].T, lbl[mid_x, :, :].T, "Sagittal (pre + overlay)"),
            (img_pre[:, mid_y, :].T, lbl[:, mid_y, :].T, "Coronal (pre + overlay)"),
            (img_pre[:, :, mid_z].T, lbl[:, :, mid_z].T, "Axial (pre + overlay)"),
        ],
    ):
        ax.imshow(img_sl, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, aspect=1)
        lbl_masked = np.ma.masked_where(lbl_sl == 0, lbl_sl)
        ax.imshow(
            lbl_masked,
            cmap="nipy_spectral",
            origin="lower",
            vmin=1,
            vmax=40,
            alpha=0.45,
            aspect=1,
        )
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Preprocessed Overlay: {case_id}")
    plt.tight_layout()
    out_pre_overlay = out_dir / "inspect_data_preprocessed_overlay.png"
    plt.savefig(out_pre_overlay, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_pre_overlay}")