## 1. Data Augmentation

#### 1.1.1 Random 3D Rotation

Rotates the patch by a random angle in the range [−15°, +15°] around a randomly chosen orthogonal plane (XY, XZ, or YZ). The image is interpolated using cubic spline (order 3) to preserve intensity fidelity, while the label and skeleton maps use nearest-neighbor interpolation (order 0) to avoid creating invalid fractional class labels.

**Why this helps thin vessels:** Thin elongated structures benefit because the model sees them at multiple orientations rather than always axis-aligned. This is particularly important because the 3D U-Net uses axis-aligned convolution kernels, and a vessel running perfectly along one axis is learned very differently from one at a slight angle.

- **Implementation:** `scipy.ndimage.rotate()` with `reshape=False` and `mode='reflect'`
- **CLI flag:** `--aug-rotation-prob` (default: 0.5)

#### 1.1.2 Random Elastic Deformation

Applies smooth, random spatial warping via dense displacement fields. Three independent displacement fields (one per axis) are generated from uniform noise and smoothed with a Gaussian filter (σ = 3.0), then scaled by an amplitude factor (α = 10.0). The resulting deformation is applied using `scipy.ndimage.map_coordinates`.

**Why this helps thin vessels:** Elastic deformation simulates the natural anatomical variability in vessel curvature and branching angles between patients. With only 25 training volumes, the model would otherwise memorize specific vessel curvatures, but elastic augmentation exposes it to smoothly warped variants that approximate inter-patient variability.

- **Implementation:** Gaussian-smoothed displacement fields + `map_coordinates`
- **CLI flag:** `--aug-elastic-prob` (default: 0.3, lower than rotation due to higher computational cost)

#### 1.1.3 Random Intensity Jitter

Scales the entire patch intensity by a random factor drawn from [1 − factor, 1 + factor] with factor = 0.1 (i.e., ±10% brightness variation). This is applied to the image only.

**Why this helps:** CTA vessel brightness depends on contrast agent concentration and timing, which varies across patients and scan protocols. Intensity jitter forces the model to rely on structural features (shape, topology) rather than absolute intensity values for vessel detection.

- **Implementation:** Simple multiplicative scaling
- **CLI flag:** `--aug-jitter-prob` (default: 0.5)

#### 1.1.4 Random Gamma Augmentation

Applies a random power-law transformation: `sign(x) · |x|^γ` where γ is drawn uniformly from [0.7, 1.5]. Values below 1.0 brighten dark regions (expanding the contrast range for faint vessels), while values above 1.0 darken them (simulating under-perfusion). The sign-preserving formulation handles the normalized intensity range correctly.

**Why this helps:** Gamma augmentation specifically addresses the contrast variation problem for thin vessels, where vessel-to-background contrast can be marginal. By training with both enhanced and reduced contrast, the model learns to detect vessels across a wider range of signal-to-noise conditions.

- **Implementation:** Sign-preserving power transform
- **CLI flag:** `--aug-gamma-prob` (default: 0.5)

### 1.2 Augmentation Pipeline Order

Within each training patch, augmentations are applied in the following order:

```
1. Random axis flips (p=0.5 per axis, already in baseline)
2. Random 3D rotation
3. Random elastic deformation
4. Random intensity jitter (image only)
5. Random gamma augmentation (image only)
```

Spatial augmentations (steps 1–3) are applied first so that intensity augmentations (steps 4–5) operate on the geometrically transformed image. All spatial augmentations jointly transform the image, label, and (if present) precomputed skeleton tensors to maintain voxel-level correspondence.

## 2. Loss Functions

### 2.1 Loss Function Architecture

The `DiceCELoss` class serves as the central loss orchestrator. It computes all loss terms and combines them as a weighted sum:

```
L_total = w_CE · L_CE + w_Dice · L_Dice + w_Tversky · L_Tversky + w_clDice · L_clDice
        + w_SkelRecall · L_SkelRecall + w_cbDice · L_cbDice + w_CAS · L_CAS
```

Each term is independently gated: setting its weight to 0 disables it entirely (no computation overhead). The default configuration uses only CE + Dice; all other terms are opt-in.

### 2.2 Baseline Losses (from master)

#### 2.2.1 Cross-Entropy Loss

Standard pixel-wise cross-entropy with optional inverse-sqrt-frequency class weighting. The class weights can be enabled via `--enable-ce-class-weights` and clamped with `--ce-weight-min` / `--ce-weight-max` to prevent extreme emphasis on very rare classes.

- **CLI flag:** `--ce-weight` (default: 1.0)

#### 2.2.2 Dice Loss

Per-class Dice computed on softmax probabilities and one-hot ground truth, averaged across classes. Uses the nnU-Net/MONAI convention where both-empty classes score 1.0. Background is excluded (`include_background=False`).

```
L_Dice = 1 - (1/C) * Σ_c [ (2 * Σ_i p_ci · g_ci + ε) / (Σ_i p_ci + Σ_i g_ci + ε) ]
```

- **CLI flag:** `--dice-weight` (default: 1.0)

#### 2.2.3 Centerline Dice (clDice)

clDice computes Dice on the soft-skeletonized prediction and ground truth, rewarding topologically correct thin-tube predictions. The soft skeletonization is performed iteratively via morphological erosion and opening operations using 3D max-pooling kernels. Gradient checkpointing is used to reduce memory consumption during the iterative skeletonization.

```
clDice = (2 · Tprec · Tsens) / (Tprec + Tsens)
```

where Tprec (topology precision) measures how much of the predicted skeleton overlaps with the ground truth, and Tsens (topology sensitivity) measures how much of the ground truth skeleton is covered by the prediction.

- **CLI flag:** `--cldice-weight` (default: 0.0)
- **Additional flags:** `--cldice-iters`, `--cldice-class-ids`, `--cldice-channel-chunk`, `--cldice-target-skeleton-dir`

### 2.3 New Loss Functions

#### 2.3.1 Tversky / Focal Tversky Loss

A generalization of Dice loss with asymmetric penalties for false positives (FP) and false negatives (FN):

```
Tversky(c) = (TP_c + ε) / (TP_c + α · FN_c + β · FP_c + ε)

L_Tversky = (1/C) * Σ_c (1 - Tversky(c))^γ
```

Setting β > α (default: α=0.3, β=0.7) penalizes false negatives more than false positives. For thin vessels, the dominant failure mode is missing the structure entirely (FN), not predicting false vessels (FP), so this asymmetry directly targets the problem. The focal exponent γ (default: 1.0) can be raised above 1 to focus gradient on the hardest classes each step.

- **CLI flags:** `--tversky-weight` (default: 0.0), `--tversky-alpha`, `--tversky-beta`, `--tversky-gamma`

#### 2.3.2 SkelRecall Loss

Optimizes recall purely on skeleton voxels — measuring what fraction of the ground truth centerline is covered by the prediction:

```
L_SkelRecall = 1 - (1/C) * Σ_c [ Σ_i (s_ci^GT · p_ci) / (Σ_i s_ci^GT + ε) ]
```

where s^GT is the precomputed ground truth skeleton. This loss heavily penalizes disconnected breaks along the vessel centerline — even a single missed voxel on a thin vessel centerline incurs a significant penalty.

- **Requires:** Precomputed target skeletons via `--cldice-target-skeleton-dir`
- **CLI flag:** `--skelrecall-weight` (default: 0.0)
- **Reference:** Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures," 2024

#### 2.3.3 Centerline Boundary Dice (cbDice)

Extends clDice by combining centerline topology with a boundary-aware term. The boundary is extracted using a morphological gradient approximation (3D max-pool dilation minus the original mask). The combined score balances connectivity (via clDice) with surface accuracy (via boundary Dice):

```
cbDice = α · clDice + (1 - α) · BoundaryDice
```

with α = 0.5 by default. This addresses the limitation of pure clDice, which rewards correct topology but can tolerate surface inaccuracies. cbDice ensures that the predicted vessel is both topologically correct *and* geometrically accurate at the boundary.

- **Requires:** Precomputed target skeletons via `--cldice-target-skeleton-dir`
- **CLI flag:** `--cbdice-weight` (default: 0.0)
- **Reference:** Shit et al., "cbDice — a novel boundary-aware loss for centerline Dice," 2024

#### 2.3.4 Connectivity-Aware Surrogate (CAS) Loss

Penalizes differences in local neighborhood connectivity density between prediction and ground truth. For each class channel, a 3×3×3 uniform convolution kernel computes the local vessel density (fraction of neighboring voxels that are positive). The loss then minimizes the absolute difference between predicted and true connectivity densities:

```
D_c = Conv3D(x_c, 1_{3x3x3}) / 27      (local density)

L_CAS = (1/C) * Σ_c mean( |D_c^pred - D_c^GT| )
```

This loss does *not* require precomputed skeletons, making it simpler to deploy. It forces the model to predict locally contiguous vessel segments rather than scattered isolated voxels.

- **CLI flag:** `--cas-weight` (default: 0.0)
- **Reference:** Stucki et al., "Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes," 2022

### 2.4 Loss Component Logging

All seven loss components are tracked independently throughout training:

| Metric | CSV Column | W&B Key | Console |
|--------|-----------|---------|---------|
| Cross-Entropy | `train_loss_ce`, `val_loss_ce` | `train/loss_ce`, `val/loss_ce` | ✅ |
| Dice | `train_loss_dice`, `val_loss_dice` | `train/loss_dice`, `val/loss_dice` | ✅ |
| Tversky | `train_loss_tversky`, `val_loss_tversky` | `train/loss_tversky`, `val/loss_tversky` | ✅ |
| clDice | `train_loss_cldice`, `val_loss_cldice` | `train/loss_cldice`, `val/loss_cldice` | ✅ |
| SkelRecall | `train_loss_skelrecall`, `val_loss_skelrecall` | `train/loss_skelrecall`, `val/loss_skelrecall` | ✅ |
| cbDice | `train_loss_cbdice`, `val_loss_cbdice` | `train/loss_cbdice`, `val/loss_cbdice` | ✅ |
| CAS | `train_loss_cas`, `val_loss_cas` | `train/loss_cas`, `val/loss_cas` | ✅ |

**Note:** Skeleton-dependent losses (SkelRecall, cbDice) report as zero during validation because precomputed skeleton volumes are not loaded in the validation loop. The primary validation signal comes from per-class Dice and non-skeleton loss terms.


## 3. Configuration Reference

### 3.1 Example: Baseline (CE + Dice only)

```bash
python src/train.py --epochs 200 --out-dir runs/baseline
```

### 3.2 Example: Full augmentation + topology-aware training

```bash
python src/train.py \
  --epochs 200 \
  --out-dir runs/topo_aware \
  --aug-rotation-prob 0.5 \
  --aug-elastic-prob 0.3 \
  --aug-jitter-prob 0.5 \
  --aug-gamma-prob 0.5 \
  --tversky-weight 0.5 --tversky-alpha 0.3 --tversky-beta 0.7 \
  --cldice-weight 0.3 \
  --skelrecall-weight 0.2 \
  --cbdice-weight 0.1 \
  --cas-weight 0.1 \
  --cldice-target-skeleton-dir training_data_resampled/skeletons
```

### 3.3 Complete Flag Reference

| Category | Flag | Type | Default | Description |
|----------|------|------|---------|-------------|
| **Augmentation** | `--aug-rotation-prob` | float | 0.5 | 3D rotation probability |
| | `--aug-elastic-prob` | float | 0.3 | Elastic deformation probability |
| | `--aug-jitter-prob` | float | 0.5 | Intensity jitter probability |
| | `--aug-gamma-prob` | float | 0.5 | Gamma augmentation probability |
| **Loss Weights** | `--dice-weight` | float | 1.0 | Dice loss weight |
| | `--ce-weight` | float | 1.0 | Cross-entropy weight |
| | `--tversky-weight` | float | 0.0 | Tversky loss weight (0 = disabled) |
| | `--cldice-weight` | float | 0.0 | clDice weight (0 = disabled) |
| | `--skelrecall-weight` | float | 0.0 | SkelRecall weight (0 = disabled) |
| | `--cbdice-weight` | float | 0.0 | cbDice weight (0 = disabled) |
| | `--cas-weight` | float | 0.0 | CAS loss weight (0 = disabled) |
| **Tversky** | `--tversky-alpha` | float | 0.3 | FN penalty weight |
| | `--tversky-beta` | float | 0.7 | FP penalty weight |
| | `--tversky-gamma` | float | 1.0 | Focal exponent (1.0 = plain Tversky) |
| **clDice** | `--cldice-iters` | int | 12 | Soft-skeletonization iterations |
| | `--cldice-class-ids` | str | None | Comma-separated class IDs for clDice |
| | `--cldice-channel-chunk` | int | 0 | Memory-saving channel chunking |
| | `--cldice-target-skeleton-dir` | Path | None | Precomputed skeleton directory |

---

## 4. References

1. Shit, S., et al. "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." *CVPR*, 2021. arXiv:2003.07311.
2. Kirchhoff, Y., et al. "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures." *arXiv*, 2024. arXiv:2404.03010.
3. Shit, S., et al. "cbDice: A Boundary-Centerline Dice Loss for Vessel Segmentation." *arXiv*, 2024. arXiv:2412.12120.
4. Stucki, N., et al. "Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes." *NeurIPS*, 2022. arXiv:2206.07486.
5. Salehi, S. S. M., et al. "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks." *MLMI Workshop, MICCAI*, 2017. arXiv:1706.05721.
