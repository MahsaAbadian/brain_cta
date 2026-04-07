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

## In Progress

- Finalizing the documentation so the project flow is fully clear for collaborators.
- Keeping the codebase structure consistent after moving active code into `src/`.
- Preparing the repo so another collaborator can clone it and get started with minimal setup.

## Next
- Test the full dataloader on the resampled dataset and confirm patch shapes, label ranges, and split integrity.
- Add validation metrics such as Dice and per-class Dice.
- Train the first real baseline CTA model and save checkpoints/results.

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
