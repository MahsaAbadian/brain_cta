# Brain CTA Project

This repository is for local research, baseline comparison, and collaboration on 3D brain vessel segmentation. It is not primarily organized as a challenge-submission repo anymore. Submission-specific notes were moved to `Competision_README.md`.

## Project Goal

The current goal is to build and compare a 3D U-Net baseline for TopBrain vessel segmentation, with a preprocessing pipeline that resamples the raw NIfTI volumes once and then trains on the resampled data.

## Repository Overview

- `README.md`: quick start for collaborators using this repo.
- `TRAINING.md`: detailed training documentation (loss, parameters, metrics, checkpoints).
- `DOCUMENTATION.md`: detailed project notes, design decisions, and implementation progress.
- `Competision_README.md`: archived challenge/submission instructions from the original template.
- `requirements.txt`: Python dependencies for the project environment.
- `src/preprocess_resample.py`: one-time offline resampling script that converts raw NIfTI data into the resampled training folders.
- `src/data_loader.py`: dataset and dataloader code used for loading patches for training/validation.
- `src/data_utils.py`: helper functions for preprocessing, split handling, cropping, spacing, and label utilities.
- `src/model_3d_unet.py`: 3D U-Net model definition.
- `src/train.py`: full baseline training script (epochs, validation, checkpoints, metrics).
- `src/train_sanity_check.py`: tiny pipeline sanity-check run for quick debugging.
- `src/inspect_data.py`: script for visualizing scans, labels, overlays, and spacing information.
- `training_data/README.txt`: dataset description from the TopBrain release.
- `training_data/License.txt`: dataset license information from the TopBrain release.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

## Get the Data

Use the TopBrain 2025 data release: https://zenodo.org/records/16878417
This repo includes `training_data/README.txt` and `training_data/License.txt`, but not the actual scans or labels.

Main reference:
- `https://topbrain2025.grand-challenge.org`

## Put the Raw Data Here

Place the downloaded files in these folders:

- `training_data/imagesTr_topbrain_ct/*.nii.gz`
- `training_data/labelsTr_topbrain_ct/*.nii.gz`
- `training_data/imagesTr_topbrain_mr/*.nii.gz`
- `training_data/labelsTr_topbrain_mr/*.nii.gz`

These data folders are ignored by git, so collaborators need to download the dataset locally.

## Preprocess Once

Run offline isotropic resampling once before training:

```bash
.venv/bin/python src/preprocess_resample.py --copy-metadata
```

This writes resampled data to:

- `training_data_resampled/imagesTr_topbrain_ct/*.nii.gz`
- `training_data_resampled/labelsTr_topbrain_ct/*.nii.gz`
- `training_data_resampled/imagesTr_topbrain_mr/*.nii.gz`
- `training_data_resampled/labelsTr_topbrain_mr/*.nii.gz`

## Train

The current loader defaults point to the resampled CTA folders, so after preprocessing you can run:

```bash
.venv/bin/python src/train.py
```

`train.py` now runs a full baseline loop with validation, checkpoint saving, and metric logging.
Detailed training behavior and argument reference: `TRAINING.md`.

For a quick pipeline smoke check, run:

```bash
.venv/bin/python src/train_sanity_check.py
```

## Run Tests

The project uses `pytest` with tests under `tests/`.

Run all tests:

```bash
.venv/bin/python -m pytest -q
```

Run only fast tests (skip slow full-dataset checks):

```bash
.venv/bin/python -m pytest -q -m "not slow"
```

Run only slow/integration tests:

```bash
.venv/bin/python -m pytest -q -m "slow or integration"
```

## Collaboration Notes

- Large datasets and generated outputs are excluded from git via `.gitignore`.
- Code and documentation should be committed; raw and resampled medical volumes should stay local.
- If someone clones the repo fresh, they should:
  1. install dependencies
  2. place raw data in `training_data/`
  3. run `src/preprocess_resample.py`
  4. run `src/train.py`
