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
- Moved the short pipeline sanity-check into `src/train_sanity_check.py`.
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

## In Progress

- Improve low-performing and zero-Dice sparse classes while preserving current strong classes.
- Align local model selection with challenge-style full-case metrics.
- Stabilize runtime (current large model is slow) without losing segmentation quality.

## Immediate Next (Highest Priority)

- Run challenge-like evaluation on current best checkpoints using `src/evaluate_challenge_like.py`.
- Store per-experiment eval outputs in separate folders and compare the same checkpoint family.
- Pick one primary selection metric for experiments:
  - training-time: `val_mean_fg_dice` (present-only), and
  - challenge-like: official TopBrain metrics (when package is available).
- Build a short experiment tracker CSV with:
  - checkpoint path
  - loss weights/clamps
  - patch/stride settings
  - challenge-like metrics summary.

## Sparse / Zero-Class Recovery Plan

### A) Metric and diagnosis tasks

- Add per-class "support" report per epoch:
  - number of validation batches where GT class is present
  - number of predicted-positive batches
  - helps separate true failure from class-absent artifacts.
- Add class-frequency table for train and val splits side-by-side.
- Flag classes with near-zero support to avoid over-interpreting noisy Dice.

### B) Data split and validation robustness

- Try 3-fold or 5-fold cross-validation to reduce split bias on rare classes.
- Build stratified split(s) based on class presence so rare labels appear in train and val.
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

- Compare `base_ch=32` vs lighter model (speed/quality tradeoff).
- Test gradient accumulation to emulate larger batch behavior if GPU memory is tight.
- Tune LR schedule/warmup for stability on rare classes.
- Add early-stop-on-plateau logic for faster iteration cycles.

### F) Inference and post-processing

- Tune sliding-window stride and overlap for better boundary consistency.
- Test connected-component cleanup per class (remove tiny isolated false positives).
- For classes prone to misses, test class-specific minimum component retention rules.
- Save qualitative overlays for top failing classes each experiment.

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
