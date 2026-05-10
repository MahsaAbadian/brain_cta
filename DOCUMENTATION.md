# Project Workflow and Notes

This document records the current project workflow and the implementation details behind each stage.

## Project stage order

The project is now organized in this order:
1. Data inspection / validation
2. One-time preprocessing: resampling + CT normalization
3. Inspect and validate preprocessed data
4. Data loading
5. Model
6. Training
7. Evaluation

## 1) Data inspection / validation

### File
- `src/inspect_data.py`

### What this script currently does
- Loads one CTA image and its matching label by `case_id`.
  - Image path pattern: `training_data/imagesTr_topbrain_ct/<case_id>_0000.nii.gz`
  - Label path pattern: `training_data/labelsTr_topbrain_ct/<case_id>.nii.gz`
- Uses `nibabel` to read both NIfTI files.
- Converts loaded data to NumPy arrays via `np.asanyarray(...dataobj)`.
- Adds project root to `sys.path` so it can import shared utilities from root-level modules.
- Imports and uses shared preprocessing:
  - `from data_utils import preprocess_ct`
- Prints key checks:
  - image shape + dtype
  - label shape + dtype
  - shape match (`True/False`)
  - label min/max + number of unique labels
  - affine matrix shape
  - voxel spacing in mm
- Applies a CTA display window:
  - center = `200`
  - width = `800`
  - derived `vmin/vmax` for display
- Visualizes middle slices for 3 planes:
  - sagittal
  - coronal
  - axial
- Displays slices with `aspect=1` to preserve voxel-grid view (no display aspect correction).
- Saves raw-data figures to `data_inspection/`:
  - `inspect_data_slices.png` (CTA grayscale slices)
  - `inspect_data_labels.png` (label slices)
  - `inspect_data_overlay.png` (labels overlaid on CTA)
- In overlay, label `0` (background) is hidden; non-zero labels are shown with alpha blending.
- Applies preprocessing using shared utility:
  - `preprocess_ct(img, low=-100.0, high=400.0)` (clip + normalize to `[0, 1]`)
- Prints preprocessed summary stats:
  - min / max / mean
- Saves preprocessed-data figures:
  - `inspect_data_preprocessed.png` (preprocessed grayscale slices)
  - `inspect_data_preprocessed_overlay.png` (preprocessed image with label overlay)
- Can also be pointed at `training_data_resampled/` to inspect the preprocessed dataset.

### Current runtime command
- From project root:
  - `.venv/bin/python src/inspect_data.py`
  - `.venv/bin/python src/inspect_data.py --data-root training_data_resampled --already-preprocessed`

### Notes
- Matplotlib backend is set to `"Agg"` (non-interactive) so this script saves images without opening windows.
- `plt.show()` is intentionally commented out for non-GUI runs.
- Script is path-safe from different working directories because it resolves:
  - `project_root = Path(__file__).resolve().parent.parent`
  - output folder as `project_root / "data_inspection"`

---

## 2) One-time preprocessing: resampling + CT normalization

### File
- `src/preprocess_resample.py`

### Purpose
- Read voxel spacing from each case's NIfTI affine.
- Resample images and labels once to the target spacing (default `0.6, 0.6, 0.6` mm).
- Apply CT normalization offline (clip to HU window and normalize to `[0, 1]`) for CT datasets.
- Save training-ready data to `training_data_resampled/` so runtime training stays fast and consistent.

### Recommended command (from project root)
- `.venv/bin/python src/preprocess_resample.py --copy-metadata`

### Other common command variants
- Explicit spacing:
  - `.venv/bin/python src/preprocess_resample.py --sx 0.6 --sy 0.6 --sz 0.6 --copy-metadata`
- Process only one dataset suffix:
  - `.venv/bin/python src/preprocess_resample.py --dataset topbrain_ct --copy-metadata`
- Process both CT and MR explicitly:
  - `.venv/bin/python src/preprocess_resample.py --dataset topbrain_ct --dataset topbrain_mr --copy-metadata`

### Output structure
- `training_data_resampled/imagesTr_*`
- `training_data_resampled/labelsTr_*`
- `training_data_resampled/split`
- `training_data_resampled/itksnap_labelmap_txt`
- `training_data_resampled/README.txt`
- `training_data_resampled/License.txt`

### Shared helper functions used by preprocessing
- Core helpers live in `src/data_utils.py`:
  - `preprocess_ct(...)`
  - `spacing_from_affine(...)`
  - `resample_to_spacing(...)`
- These helpers are consumed by `src/preprocess_resample.py` and reused by inspection where needed.

---

## 3) Inspect and validate preprocessed data

After running `src/preprocess_resample.py`, the next step is to inspect the saved outputs in `training_data_resampled/`.

### Current validation goal
- Confirm image/label alignment is preserved after resampling.
- Confirm CT images are already normalized and saved in the processed dataset.
- Confirm labels still contain valid integer class ids.

### Recommended runtime command
- From project root:
  - `.venv/bin/python src/inspect_data.py --data-root training_data_resampled --already-preprocessed`

### Dataset conventions validated so far

- Data is 3D.
- For tested case `topcow_ct_005`:
  - image shape: `(332, 417, 184)`
  - label shape: `(332, 417, 184)`
  - shape match: `True`
- Label values include background and vessel classes:
  - min label: `0`
  - max label: `40`
  - unique label count observed in tested case: `34`

---

## 4) Data loading

### File
- `src/data_loader.py`

### Purpose
- Build datasets and dataloaders entirely from the preprocessed `training_data_resampled/` root.
- Read images, labels, split files, and label maps from the same processed dataset root.
- Sample training patches and center-crop validation patches.
- Apply data augmentation during training only.

### Important note
- CT normalization is not done on the fly in the dataloader.
- The loader expects CT images in `training_data_resampled/` to already be normalized by `src/preprocess_resample.py`.

### Augmentation policy: no left/right flip
- All resampled volumes are stored in **LPS** orientation, so array axis 0 is the
  left/right axis (`nib.aff2axcodes(...)` returns `('L', 'P', 'S')` for every
  case).
- The training-time augmentation in `CTAPatchDataset.__getitem__` (and the
  helper `random_flip_3d`) only flips axes 1 (A/P) and 2 (I/S). Axis 0 is
  intentionally skipped.
- **Why**: the labelmap distinguishes paired vessels by side (`R-ICA` vs
  `L-ICA`, `R-M1` vs `L-M1`, `R-A1A2` vs `L-A1A2`, ..., `R-BVR` vs `L-BVR`).
  A naive L/R flip swaps the visual side of every voxel **without** swapping
  the paired class IDs, which trains the model to predict the right-sided
  class on the left side and vice versa for ~50% of patches. The result is
  catastrophic R/L confusion on every bilateral structure.
- **Diagnostic signature** (from the `thin_vessel_phase3_arch` run): purely
  midline classes (BA, SSS, ICVs, StS, VoG) had smooth monotone learning
  curves that plateaued at 0.5–0.85 Dice, while every bilateral pair (R/L
  ICA, R/L M1, R/L P1P2, ...) oscillated wildly between ~0.10 and ~0.65
  every other epoch, settling around 0.37–0.41 only after the cosine LR
  schedule decayed enough to damp the oscillation. The contrast made the
  augmentation the unambiguous root cause.
- **After the fix** (`thin_vessel_phase4_lrfix` run, identical
  hyperparameters): bilateral classes converge smoothly to ~0.65–0.70 Dice,
  `val_mean_fg_dice` lifts from ~0.30 to ~0.50, and
  `val_mean_fg_dice_all_cases_present` from ~0.40 to ~0.65.
- **Follow-up option**: re-enable axis-0 flip with a paired-class label
  swap (R↔L LUT over the ~16 paired class pairs, nnUNet-style). This
  recovers the lost augmentation diversity and effectively doubles
  per-pair training data. Tracked in `PROJECT_TODO.md`.

---

## 5) Model

### File
- `src/model_3d_unet.py`

### Purpose
- Define the baseline compact 3D U-Net used in current experiments.

### Architecture implemented
- Model class: `UNet3D(in_channels=1, num_classes=41, base_ch=16)`.
- Input tensor format: `(B, C, D, H, W)` with `C=1` for CTA.
- Output tensor format: `(B, 41, D, H, W)` as **raw logits** (no activation in `forward`).
- Encoder:
  - 4 resolution levels (`enc1`..`enc4`) using `ConvBlock3D` (two `3x3x3` convs + BatchNorm + ReLU).
  - Downsampling via `MaxPool3d(2)` after each level.
- Bottleneck:
  - `ConvBlock3D` at deepest scale (`base_ch*16`).
- Decoder:
  - `ConvTranspose3d(..., kernel_size=2, stride=2)` upsampling.
  - Skip connections via channel-wise concat with encoder features.
  - Decoder conv blocks to fuse skip + upsampled features.
- Final layer:
  - `1x1x1` conv to project features to 41 class logits.
- Shape safety:
  - `_match_size(...)` uses interpolation when needed so decoder tensors match skip-connection spatial sizes.

### Why this model was chosen
1. **Task fit for volumetric vessels**: Vessel anatomy is 3D and topology-sensitive. A 3D U-Net captures continuity across slices better than 2D models.
2. **Localization + context balance**: U-Net skip connections preserve fine vessel boundaries while the deep encoder learns larger anatomical context.
3. **Stable baseline for small datasets**: For ~25 CTA training volumes, a compact `base_ch=16` U-Net is strong enough without being too large to train.
4. **Compatible with current patch pipeline**: Works directly with patch tensors from `src/data_loader.py` after flattening to `(B_eff, 1, D, H, W)`.
5. **Loss compatibility**: Raw logits are correct for `CrossEntropyLoss` (and Dice+CE variants that also expect logits).

### Forward-pass sanity target
- Given input batch `(B, 1, 96, 96, 96)`, expected output is `(B, 41, 96, 96, 96)`.
- Current `__main__` block in `src/model_3d_unet.py` already verifies this shape behavior.

---

## 6) Training

### File
- `src/train.py`

### Purpose
- Run full baseline training on the preprocessed CTA dataset.
- Includes epoch loop, validation, checkpoint saving, and metric logging.

### Current runtime command
- From project root:
  - `.venv/bin/python src/train.py`
  - `.venv/bin/python src/train.py --epochs 40 --out-dir runs/baseline_e40`
### Notes
- Builds CTA train/val loaders from `training_data_resampled/`.
- Uses `UNet3D` + `DiceCELoss` objective.
- Logs per-epoch metrics to `metrics.csv`.
- Saves `checkpoint_latest.pt`, `checkpoint_best.pt`, and optional periodic epoch checkpoints.
- Detailed argument and loss documentation is now in `TRAINING.md`.

---

## 7) Evaluation

This stage is not fully implemented yet.

### Current evaluation goals
- Add quantitative validation metrics such as Dice and per-class Dice.
- Save qualitative prediction overlays for validation cases.
- Review failure cases on thin and small vessels.
- Define what counts as a successful baseline compared with future experiments.

---

## Experiments and ablations

This section covers the main design choices to compare once the end-to-end baseline is running reliably.

### A) Patch sampling strategy

When loading 3D crops (patches) for training, we have a few options to ensure the model sees useful, non-redundant data.

#### Option A: Pure Random Cropping
- **How it works**: Pick random `(x, y, z)` starting coordinates for every patch.
- **Pros**: Very easy to implement.
- **Cons**: Brain vessels are sparse (most of the head is brain tissue, bone, or air). Pure random crops will yield many patches with only background (label `0`), leading to slow training and class imbalance.

#### Option B: Foreground-Aware Cropping (Balanced Random)
- **How it works**: Pre-calculate coordinates of all vessel voxels. For a given batch, pick a patch center from the foreground (vessel) voxels $P\%$ of the time, and a completely random center $(1-P)\%$ of the time. (e.g., $P=60\%$).
- **Pros**: Ensures the model always sees vessels during training while still learning background context.
- **Cons**: Patches might overlap heavily if sampled independently, showing the model redundant data.

#### Option C: Foreground-Aware + Non-Overlapping (or Low-Overlap) Sampling via IoU
- **How it works**: Like Option B, but we also enforce a rule: before accepting a new patch for an epoch/batch, check its overlap (Intersection over Union, IoU) against already-chosen patches for that volume. Reject it if the overlap is too high.
- **Pros**: Solves class imbalance *and* exact redundancy. Maximizes data efficiency.
- **Cons**: Slightly more complex to implement and computationally heavy(requires tracking selected boxes and computing 3D IoU).

#### Option D: Foreground-Aware + Minimum Center Distance (Simplified Low-Overlap)
- **How it works**: A simpler alternative to Option C. We track the center coordinates of patches already chosen for a given volume in the current epoch. If a newly sampled center is too close (e.g., Euclidean distance < half the patch size) to an existing center, we reject and resample.
- **Pros**: Computationally much cheaper and simpler to implement than full volume IoU overlap checks. Effectively prevents severe redundancy.
- **Cons**: Requires keeping state (a list of chosen centers) per volume during the epoch, and distance doesn't perfectly map to exact voxel overlap if aspect ratios vary.

#### Option E: Deterministic Grid (Tiling)
- **How it works**: Divide the volume into a fixed grid of patches (often with slight overlap).
- **Pros**: Guarantees coverage of the entire volume exactly once.
- **Cons**: Lacks translation invariance (the network always sees vessels at the exact same relative grid offsets). Usually reserved for **Validation/Inference**, not training.

### Current choice: Option D 


*Reasoning*:
1. **Sparsity**: Vessels occupy a tiny fraction of the brain volume. Pure random sampling would result in "empty" patches >90% of the time.
2. **Class Imbalance**: Forcing the center of the patch to be a vessel voxel ensures the model receives gradient signals for the challenging minority classes.
3. **Redundancy Mitigation (Option D element)**: While true IoU non-overlap tracking (Option C) is complex for a simple dataloader, we can approximate it by simply keeping the patch size reasonably large and sampling fewer patches per volume per epoch, or explicitly implementing a minimum center distance check (Option D), naturally reducing the chance of severe overlap without needing an explicit IoU tracker.

**Implementation Plan**:
Modify the dataloader so that for a training crop, we first randomly decide if this crop should be "foreground" (e.g., 60% chance) or "background/random" (40% chance). If foreground, we pick a random non-zero voxel from the label array and center the patch on it.

---

### B) Patch size selection

When extracting 3D sub-volumes for training, we use a fixed patch size. Our baseline choices are typically `(96, 96, 96)` or `(128, 128, 128)`.

*Reasoning for these sizes*:
1. **Hardware Memory Limits (GPU VRAM)**: A full CTA volume (e.g., `332 x 417 x 184`) is too large to fit in GPU memory for a 3D U-Net, especially with 41 output classes. We must crop.
2. **Anatomical Context**: The model needs to see enough surrounding tissue to distinguish *which* specific vessel it is looking at (e.g., distinguishing M1 from M2 based on branching). 
   - A `96x96x96` patch at `0.4x0.4x0.75mm` spacing covers `~38x38x72mm` of physical space.
   - A `128x128x128` patch covers `~51x51x96mm`, providing even more context, which is highly beneficial if GPU memory allows.
3. **Network Architecture (Divisibility)**: 3D U-Nets use repeated downsampling (usually max pooling by 2). The patch dimensions should be cleanly divisible by $2^N$ (where $N$ is the number of pooling layers, e.g., 4 or 5) to ensure skip connections match perfectly in the decoder. Both 96 ($2^5 \times 3$) and 128 ($2^7$) are excellent choices.

*Strategy*: Start with `(96, 96, 96)` to maximize context. If not enough we should increase!

---

## Future model variants

This section discusses common U-Net variants that could replace or extend the current `UNet3D` baseline.

### A) Deeper/Wider 3D U-Net
- **What changes**: Increase `base_ch` (e.g., 16 -> 32) and/or add more encoder-decoder levels.
- **Why it may help**:
  - More capacity can improve multiclass vessel discrimination (41 classes).
  - Larger receptive field can capture broader vascular context.
- **Why it may hurt**:
  - Much higher VRAM usage and slower training.
  - Increased overfitting risk with only ~25 CTA volumes.
- **Recommendation**: Useful only after baseline is stable and if hardware allows.

### B) Residual 3D U-Net (ResUNet3D)
- **What changes**: Replace plain conv blocks with residual blocks (`x + F(x)`).
- **Why it may help**:
  - Easier optimization in deeper networks.
  - Often improves gradient flow and convergence stability.
- **Why it may hurt**:
  - More implementation complexity.
  - Slight compute/memory increase.
- **Recommendation**: Strong next candidate once baseline metrics plateau.

### C) Attention U-Net (3D Attention Gates)
- **What changes**: Attention gates on skip connections to suppress irrelevant features.
- **Why it may help**:
  - Could focus decoder on vessel regions and reduce false positives.
  - Helpful when structures are small/thin (as in vessels).
- **Why it may hurt**:
  - Extra parameters and compute.
  - Gains are dataset-dependent; can be marginal on small datasets.
- **Recommendation**: Worth trying after baseline + residual variant.

### D) nnU-Net-style configuration (framework approach)
- **What changes**: Use automated design rules (patch size, spacing, augmentations, architecture settings).
- **Why it may help**:
  - Very strong out-of-the-box medical segmentation baseline.
  - Reduces manual hyperparameter trial-and-error.
- **Why it may hurt**:
  - Less educational if goal is to learn implementation details deeply.
  - Less control over every internal choice.
- **Recommendation**: Excellent benchmarking reference, even if final training remains custom.
- **Current repo status**: A separate official `nnunetv2` pipeline is available; see `NNUNET.md` for commands and folder conventions.

### E) 2D U-Net or 2.5D U-Net
- **What changes**: Train slice-wise (2D) or multi-slice pseudo-3D inputs.
- **Why it may help**:
  - Much lower memory footprint.
  - Faster training/debug cycle.
- **Why it may hurt**:
  - Loses full 3D vessel continuity/topology information.
  - Usually weaker for thin connected vessel networks than true 3D models.
- **Recommendation**: Useful fallback for limited hardware, but not preferred primary model.

### F) Swin-UNETR / Transformer-based 3D segmentation
- **What changes**: Transformer encoder with U-Net-style decoder.
- **Why it may help**:
  - Better long-range context modeling.
  - Can improve global anatomical consistency.
- **Why it may hurt**:
  - Data-hungry and compute-heavy.
  - Higher tuning complexity and training instability risk on small datasets.
- **Recommendation**: Not first choice for current data scale; consider later only if strong resources are available.

### Practical model roadmap for this project
1. **Start** with current compact `UNet3D` baseline (`base_ch=16`).
2. If underfitting or limited performance, try **ResUNet3D**.
3. If false positives remain high, test **Attention U-Net**.
4. Keep **nnU-Net** as an external benchmark to sanity-check expected performance.
5. Use transformer models only as advanced experiments after strong baselines.

---
