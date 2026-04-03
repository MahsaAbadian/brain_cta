# Run from project root with the project venv:
#   .venv/bin/python src/inspect_data.py
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import matplotlib


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_utils import preprocess_ct
matplotlib.use("Agg")  # no display needed; saves to PNG only
import matplotlib.pyplot as plt

case_id = "topcow_ct_005"
out_dir = Path(__file__).resolve().parent


img_path = project_root / "training_data/imagesTr_topbrain_ct" / f"{case_id}_0000.nii.gz"
lbl_path = project_root / "training_data/labelsTr_topbrain_ct" / f"{case_id}.nii.gz"

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
# Visualize center slices (axial, coronal, sagittal) with a CT vessel window
# ---------------------------------------------------------------------------
w_center = 200
w_width = 800
vmin = w_center - w_width / 2
vmax = w_center + w_width / 2

nx, ny, nz = img.shape
mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
sx, sy, sz = spacing[0], spacing[1], spacing[2]
# aspect is height/width of data scale
asp_sag, asp_cor, asp_ax = sz / sy, sz / sx, sy / sx

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(
    img[mid_x, :, :].T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    aspect=asp_sag,
)
axes[0].set_title("Sagittal (x mid)")
axes[0].axis("off")

axes[1].imshow(
    img[:, mid_y, :].T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    aspect=asp_cor,
)
axes[1].set_title("Coronal (y mid)")
axes[1].axis("off")

axes[2].imshow(
    img[:, :, mid_z].T,
    cmap="gray",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    aspect=asp_ax,
)
axes[2].set_title("Axial (z mid)")
axes[2].axis("off")

plt.suptitle(f"CTA: {img_path.name}")
plt.tight_layout()
out_path = out_dir / "inspect_data_slices.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path}")

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
for ax, (sl, title, asp) in zip(
    axes2,
    [
        (lbl[mid_x, :, :].T, "Sagittal (labels)", asp_sag),
        (lbl[:, mid_y, :].T, "Coronal (labels)", asp_cor),
        (lbl[:, :, mid_z].T, "Axial (labels)", asp_ax),
    ],
):
    ax.imshow(sl, cmap="nipy_spectral", origin="lower", vmin=0, vmax=40, aspect=asp)
    ax.set_title(title)
    ax.axis("off")
plt.suptitle(f"Labels: {case_id}.nii.gz")
plt.tight_layout()
out_labels = out_dir / "inspect_data_labels.png"
plt.savefig(out_labels, dpi=120, bbox_inches="tight")
print(f"Saved: {out_labels}")

# Overlay labels on top of CTA for alignment check
fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
for ax, (img_sl, lbl_sl, title, asp) in zip(
    axes3,
    [
        (img[mid_x, :, :].T, lbl[mid_x, :, :].T, "Sagittal (overlay)", asp_sag),
        (img[:, mid_y, :].T, lbl[:, mid_y, :].T, "Coronal (overlay)", asp_cor),
        (img[:, :, mid_z].T, lbl[:, :, mid_z].T, "Axial (overlay)", asp_ax),
    ],
):
    ax.imshow(img_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=asp)
    # Hide background class (0) so only vessel labels are overlaid
    lbl_masked = np.ma.masked_where(lbl_sl == 0, lbl_sl)
    ax.imshow(
        lbl_masked,
        cmap="nipy_spectral",
        origin="lower",
        vmin=1,
        vmax=40,
        alpha=0.45,
        aspect=asp,
    )
    ax.set_title(title)
    ax.axis("off")
plt.suptitle(f"Overlay: {case_id}")
plt.tight_layout()
out_overlay = out_dir / "inspect_data_overlay.png"
plt.savefig(out_overlay, dpi=120, bbox_inches="tight")
print(f"Saved: {out_overlay}")

# plt.show()  # uncomment to open windows when running in a GUI

# Preprocess with shared function from data_utils
img_pre = preprocess_ct(img, low=-100.0, high=400.0)
print("Preprocessed min/max/mean:", img_pre.min(), img_pre.max(), img_pre.mean())

# visualize preprocessed center slices
fig4, axes4 = plt.subplots(1, 3, figsize=(12, 4))
for ax, (sl, title, asp) in zip(
    axes4,
    [
        (img_pre[mid_x, :, :].T, "Sagittal (preprocessed)", asp_sag),
        (img_pre[:, mid_y, :].T, "Coronal (preprocessed)", asp_cor),
        (img_pre[:, :, mid_z].T, "Axial (preprocessed)", asp_ax),
    ],
):
    ax.imshow(sl, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, aspect=asp)
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(f"Preprocessed CTA: {case_id}")
plt.tight_layout()
out_pre = out_dir / "inspect_data_preprocessed.png"
plt.savefig(out_pre, dpi=120, bbox_inches="tight")
print(f"Saved: {out_pre}")

fig5, axes5 = plt.subplots(1, 3, figsize=(12, 4))
for ax, (img_sl, lbl_sl, title, asp) in zip(
    axes5,
    [
        (img_pre[mid_x, :, :].T, lbl[mid_x, :, :].T, "Sagittal (pre + overlay)", asp_sag),
        (img_pre[:, mid_y, :].T, lbl[:, mid_y, :].T, "Coronal (pre + overlay)", asp_cor),
        (img_pre[:, :, mid_z].T, lbl[:, :, mid_z].T, "Axial (pre + overlay)", asp_ax),
    ],
):
    ax.imshow(img_sl, cmap="gray", origin="lower", vmin=0.0, vmax=1.0, aspect=asp)
    lbl_masked = np.ma.masked_where(lbl_sl == 0, lbl_sl)
    ax.imshow(lbl_masked, cmap="nipy_spectral", origin="lower", vmin=1, vmax=40, alpha=0.45, aspect=asp)
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(f"Preprocessed Overlay: {case_id}")
plt.tight_layout()
out_pre_overlay = out_dir / "inspect_data_preprocessed_overlay.png"
plt.savefig(out_pre_overlay, dpi=120, bbox_inches="tight")
print(f"Saved: {out_pre_overlay}")