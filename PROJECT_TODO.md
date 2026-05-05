# Project Todo

This file tracks what is already done, what is currently underway, and what still needs to happen to complete the project.

## Done

- Set up a collaboration-friendly repository with data excluded from git.
- Added project-facing documentation in `README.md`.
- Kept challenge-specific instructions separate in `Competision_README.md`.
- Added dataset metadata files (`training_data/README.txt`, `training_data/License.txt`) to the repo.
- Built a data inspection script in `src/inspect_data.py`.
- Verified key dataset properties such as shape, label range, and voxel spacing.
- Confirmed the raw data is anisotropic.
- Added one-time offline preprocessing in `src/preprocess_resample.py` (resampling + CT normalization).
- Ran `src/preprocess_resample.py` on the full dataset and created `training_data_resampled/`.
- Visually inspected several resampled CTA image/label pairs and confirmed alignment, label validity, and normalized CT intensity range.
- Removed runtime resampling from the dataloader.
- Updated `src/data_loader.py` to read from resampled data folders by default.
- Implemented the patch-based CTA dataset and dataloader pipeline.
- Implemented the baseline 3D U-Net in `src/model_3d_unet.py`.
- Implemented a full training script in `src/train.py` (epochs + validation + checkpoints + metrics).
- Added inline comments and module-level explanations in core source files.
- Aligned the documentation with the current `src/` workflow.
- Added proper tests with `pytest`.
- Added overfit debug mode and optional augmentation disable in `src/train.py`.
- Switched Dice edge-case handling to nnU-Net/MONAI convention.
- Added CE class weighting (inverse-sqrt frequency) with optional clamp controls.
- Added CLI knobs for loss balancing (`--dice-weight`, `--ce-weight`, `--ce-weight-min`, `--ce-weight-max`).
- Added per-class Dice logging to console and `metrics.csv`.
- Added present-only foreground Dice tracking in validation (plus `val_mean_fg_dice_all` diagnostic metric).
- Added challenge-like local evaluation script `src/evaluate_challenge_like.py` for full-volume inference.
- Added per-class training patch hit diagnostics to confirm rare/thin classes are sampled each epoch.
- Added Part A metric diagnostics:
  - per-class validation GT/prediction support counts per epoch
  - train/val class-frequency report (`class_frequency.csv`)
  - near-zero support flags for split-level class support
- Added thin-vessel mean Dice summary metrics for the known failing group.
- Added early stopping for faster experiment cycles when validation metrics plateau.
- Added gradient accumulation support for larger effective batches under GPU memory limits.
- Added stratified K-fold split generation based on foreground class presence.

## In Progress

- Improve low-performing and zero-Dice sparse classes while preserving current strong classes.
- Align local model selection with challenge-style full-case metrics.
- Stabilize runtime (current large model is slow) without losing segmentation quality.

## Diagnosed Problems (from epoch 119 analysis)

### Metric reliability issues

- **Fake Dice=1.0 on absent-in-val classes**: c15 (3rd-A2, 4/25 patients), c16 (3rd-A3, 4/25),
  c28 (L-AICA, 8/25), c31 (R-AChA, 5/25), c32 (L-AChA, 6/25) all show Dice=1.0 not because
  the model learned them, but because the 5-patient val set likely contains zero examples of
  these classes. Both GT and prediction are empty → Dice=1.0 by convention. The model has
  never actually predicted these structures.
- **Round Dice values (0.2/0.4/0.6/0.8) are 5-patient averages, not real scores**: Each step
  of 0.2 corresponds to one extra val patient where the model is either fully right or fully
  wrong. These are noise, not signal. A single misclassified patient moves any class by 0.2.
- **`val_mean_fg_dice_all` (0.428) is inflated** by absent-class 1.0 scores. The real picture
  is `val_mean_fg_dice` (0.339, present-only), and even that is noisy.

### Thin vessel failures (genuine, not metric noise)

These are always-present, reasonably sized structures where the model consistently scores <0.2.
They are the primary drag on `val_mean_fg_dice`:

| Class | Name | Patients | Avg voxels | Dice (ep119) |
|-------|------|----------|------------|--------------|
| c10   | Acom      | 21/25 |    94 | 0.043 |
| c06   | L-ICA     | 24/25 | 1,267 | 0.052 |
| c03   | L-P1P2    | 25/25 | 1,065 | 0.077 |
| c11   | R-A1A2    | 25/25 |   876 | 0.092 |
| c12   | L-A1A2    | 25/25 |   892 | 0.098 |
| c02   | R-P1P2    | 25/25 | 1,022 | 0.175 |
| c04   | R-ICA     | 25/25 | 1,406 | 0.181 |
| c23   | R-VA      | 25/25 | 4,424 | 0.113 |
| c25   | R-SCA     | 25/25 |   316 | 0.200 |

Root cause: thin elongated vessels are geometrically unforgiving — a 1-voxel spatial offset on a
2-3 voxel wide vessel collapses Dice because both `|P|` and `|G|` are tiny. A 96³ patch also
captures only a short cross-section of a long vessel, so the model rarely sees enough spatial
context to distinguish similar-looking vessel branches.

## Sparse / Zero-Class Recovery Plan

### A) Metric and diagnosis tasks

- Done: add per-class support report per epoch:
  - number of validation cases where GT class is present
  - number of validation cases where prediction is positive
  - helps separate true failure from class-absent artifacts.
- Done: add class-frequency table for train and val splits side-by-side.
- Done: flag classes with near-zero support to avoid over-interpreting noisy Dice.

### B) Data split and validation robustness

- Try 3-fold or 5-fold cross-validation to reduce split bias on rare classes.
- Done: build stratified split(s) based on class presence so rare labels appear in train and val.
- For final reporting, aggregate fold-level per-class Dice and variance.

### C) Sampling strategy for rare classes

- Add class-aware patch sampling:
  - sample patch centers from selected rare classes with configurable probability.
- Ensure each train epoch sees a minimum number of patches containing each targeted rare class.
- Keep a mixed strategy:
  - foreground-aware global sampling +
  - rare-class targeted sampling.
- Add sampling diagnostics log:
  - per-class patch hit counts in each epoch.

### D) Loss strategy for rare classes

- Sweep CE/Dice balance:
  - test `ce_weight` in `{0.25, 0.5, 0.75, 1.0}` with fixed Dice weight.
- Sweep CE clamp max:
  - test `ce_weight_max` in `{1.5, 2.0, 3.0}`.
- Compare alternative class-weight formulas:
  - inverse frequency
  - inverse-sqrt frequency (current)
  - effective-number weighting.
- Evaluate focal CE variant (or focal Tversky) for hard rare classes.

### E) Architecture and optimization

- **Do not use a bigger model** (`base_ch=32` was tested and made results worse — more
  parameters overfit the small dataset and rare classes especially suffer).
- Try residual blocks inside `ConvBlock3D` (see Thin Vessel Plan section G) — more capacity
  per parameter without the overfitting cost of wider channels.
- Done: add gradient accumulation to emulate larger batch behavior if GPU memory is tight.
- Tune LR schedule/warmup for stability on rare classes.
- Done: add early-stop-on-plateau logic for faster iteration cycles.

### F) Inference and post-processing

- Tune sliding-window stride and overlap for better boundary consistency.
- Test connected-component cleanup per class (remove tiny isolated false positives).
- For classes prone to misses, test class-specific minimum component retention rules.
- Save qualitative overlays for top failing classes each experiment.

## Thin Vessel Improvement Plan

Thin vessels (ICA, P1P2, A1A2, Acom, VA, SCA) are geometrically 2-4 voxels wide in the
resampled data. Standard patch training and Dice loss both work against them. The strategies
below are ordered roughly from easiest to hardest to implement.

### G) Patch sampling for thin vessels

- Center training patches directly on thin-vessel voxels using class-aware sampling
  (already partially in place via `rare_class_prob`; confirm it targets the failing classes).
- Verify per-class patch hit counts per epoch (add to sampling diagnostics log) — confirm
  c02/c03/c04/c06/c10/c11/c12/c23 are being sampled, not just incidentally included.
- Try higher `--rare-class-patch-prob` (e.g. 0.5–0.7) specifically for the thin-vessel group.
- Try centering on vessel *centerline* voxels only (not any foreground voxel) so the vessel
  is always near the patch center rather than at the edge.

### H) Loss modifications for thin vessels

- **Tversky loss** instead of standard Dice: `T = TP / (TP + α*FP + β*FN)`. Set `β > α`
  (e.g. β=0.7, α=0.3) to penalize false negatives more than false positives. For thin vessels,
  missing the structure entirely (FN) is the dominant failure mode.
- **Focal Tversky loss**: raise the Tversky score to a power `γ > 1` to focus gradient on the
  hardest (lowest-Tversky) classes each step.
- **Per-class Dice weighting**: instead of equal weight per class in the Dice term, weight each
  class inversely by its average voxel count so thin vessels get more gradient than large ones
  like SSS.
- **Boundary / surface loss** as an auxiliary term: punishes predictions that are spatially
  offset from the GT surface, which is exactly the failure mode for thin tubes.

### I) Architecture changes for thin vessels

- **Residual blocks** (swap `ConvBlock3D` → `ResConvBlock3D`): better gradient flow through
  the encoder helps early layers learn fine vessel features without vanishing gradients.
- **Deep supervision**: attach auxiliary segmentation heads at each decoder level and compute
  loss at all scales. This forces the network to represent vessel structure even in the
  lower-resolution decoder stages, giving thin vessels more gradient signal.
- **Larger input patch with same output patch** (encoder-decoder asymmetry): feed a 128³ or
  160³ patch to the encoder but only supervise the center 96³. This gives the model more
  spatial context (e.g. to tell left ICA from right ICA based on surrounding anatomy) while
  keeping memory cost moderate.
- **Anisotropy-aware convolutions**: if resampling to isotropic spacing introduces interpolation
  artifacts on thin structures, try using elongated kernels (e.g. `1×1×3` + `1×3×1` + `3×1×1`)
  in the first encoder block to capture vessel orientation before mixing channels.

### J) Data augmentation for thin vessels

- **Elastic deformation**: random smooth spatial warping teaches the model that vessel shape
  varies across patients, reducing overfitting to specific vessel trajectories in the training set.
- **Intensity jitter and gamma augmentation**: thin vessels are visible due to contrast
  enhancement; making the model robust to brightness variation helps generalization.
- **Random rotation** (small angles, ±15°): thin elongated structures benefit because the model
  sees them at multiple orientations rather than always axis-aligned.
- Ensure augmentations are applied consistently to image and label together (already done for
  flips; extend to rotations and elastic deformation).

### K) Inference improvements for thin vessels

- **Smaller sliding-window stride**: use stride=32 instead of 64 on the thin-vessel axes so
  predictions are averaged over more overlapping patches, smoothing out boundary errors.
- **Gaussian weighting of patch contributions**: weight central voxels of each patch higher
  than edge voxels when aggregating predictions (standard in nnU-Net). Edge voxels of a patch
  have less context and are noisier for thin structures.
- **Morphological post-processing per class**: after argmax, apply a small closing operation
  (e.g. ball radius 1) to thin vessel predictions to fill 1-voxel gaps that break connectivity.
- **Connected-component filtering**: for thin-vessel classes, keep only the largest connected
  component and discard tiny isolated blobs (likely false positives from other vessel classes).

### L) Evaluation fixes to stop being misled

- Switch primary training-time metric from patch Dice to **full-volume Dice** using
  `src/evaluate_challenge_like.py` on the val split after each experiment.
- Implement **stratified cross-validation** (3-fold or 5-fold) so thin vessel Dice estimates
  are averaged over more patients and left-right asymmetry noise is reduced.
- Report per-class Dice with the number of val patients where GT was present alongside it,
  so fake 1.0 scores are immediately visible.
- Done: add a "thin vessel mean Dice" summary metric = mean Dice over {c02, c03, c04, c06, c10,
  c11, c12, c23, c25} only — the group that is currently failing — to track improvement
  specifically for these structures.

## After Baseline Works

- Compare different patch sizes.
- Compare different numbers of patches per volume.
- Tune foreground sampling probability.
- Evaluate augmentation choices.
- Compare different target resampling spacings.
- Save qualitative prediction overlays for validation cases.
- Analyze failure cases, especially on thin and small vessels.

## Model Improvement Ideas

- Try residual blocks.
- Try attention U-Net.
- Try improved loss weighting for class imbalance.
- Try stronger or better-targeted augmentation.
- Compare against an nnU-Net-style baseline if time allows.
- Evaluate deep supervision.
- Evaluate boundary-aware auxiliary loss.

## Future Improvements (from TopBrain Lessons)

Source: [TopBrain Tutorials — Lessons from TopCoW](https://topbrain2025.grand-challenge.org/tutorials/#:~:text=Benefits%20of%20mixed,architecture%20%5BLink%5D)

### M) Mixed modality training for CTA

- Train with **both MRA and CTA** modalities together rather than CTA alone.
  - Top-performing teams in TopCoW found mixed-modality training improved CTA performance specifically.
  - Build a combined dataloader that samples from both MRA and CTA training sets.
  - Consider modality conditioning (e.g. a one-hot modality token or separate input channel) so the
    model can distinguish the two input domains.
  - Evaluate CTA-only vs mixed-modality models on the CTA validation leaderboard to confirm the gain.

### N) Topological loss functions (centerline-based)

Thin vessels fail primarily because standard Dice is insensitive to connectivity and centerline
accuracy. Centerline-aware losses directly penalize topology errors:

- **clDice (centerline Dice)**: computes Dice on the skeletonised prediction and GT, rewarding
  topologically correct thin-tube predictions. Well-suited for vessels where connectivity matters
  more than volumetric overlap. [[paper]](https://arxiv.org/abs/2003.07311)
- **CAS (connectivity-aware surrogate) loss**: a differentiable surrogate for topological
  connectivity, penalising breaks in predicted vessel paths.
  [[paper]](https://arxiv.org/abs/2206.07486)
- **SkelRecall loss**: optimises recall on the skeleton voxels so the predicted vessel covers the
  full length of the GT centreline, reducing missed vessel segments.
  [[paper]](https://arxiv.org/abs/2404.03010)
- **cbDice (centerline boundary Dice)**: extends clDice with boundary-sensitive weighting,
  combining topological correctness with surface accuracy.
  [[paper]](https://arxiv.org/abs/2412.12120)

Implementation note: all of these can be added as auxiliary loss terms on top of the existing
Dice + CE combination (e.g. `total_loss = dice + ce + λ_topo * topo_loss`). Start with clDice
as it has the most readily available open-source implementations.

### O) Connectivity-based architectural optimizations

- **Lung-airway-style connectivity module**: originally designed for airway tree segmentation,
  enforces long-range connectivity between predicted vessel segments. Adapt the approach for
  cerebrovascular trees. [[paper]](https://arxiv.org/abs/2209.01084)
- **NexToU architecture (neighborhood relation)**: incorporates explicit neighborhood-relation
  modeling so the network understands spatial adjacency between vessel branches, which helps
  correctly label branching points and avoid topological errors.
  [[paper]](https://arxiv.org/abs/2305.15911)

## Final Deliverables

- Prepare a stable baseline result.
- Document baseline vs improved model comparisons.
- Create figures for:
  - raw vs resampled data
  - training curves
  - qualitative predictions
  - metric tables
- Write the final report and clearly state each team member's contribution.

## Collaboration Checklist

- Make sure all collaborators can install dependencies from `requirements.txt`.
- Make sure all collaborators know where to place raw data.
- Make sure all collaborators run preprocessing before training.
- Keep large medical data local and out of git.
- Push documentation and code changes regularly to the shared repo.
