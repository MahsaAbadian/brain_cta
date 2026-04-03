# Brain CTA Project

This repository is for local research, baseline comparison, and collaboration on 3D brain vessel segmentation. It is not primarily organized as a challenge-submission repo anymore. Submission-specific notes were moved to `Competision_README.md`.

## Project Goal

The current goal is to build and compare a 3D U-Net baseline for TopBrain vessel segmentation, with a preprocessing pipeline that resamples the raw NIfTI volumes once and then trains on the resampled data.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

## Get the Data

Use the TopBrain 2025 data release. The included `training_data/README.txt` describes the dataset structure and license.

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
.venv/bin/python preprocess_resample.py --copy-metadata
```

This writes resampled data to:

- `training_data_resampled/imagesTr_topbrain_ct/*.nii.gz`
- `training_data_resampled/labelsTr_topbrain_ct/*.nii.gz`
- `training_data_resampled/imagesTr_topbrain_mr/*.nii.gz`
- `training_data_resampled/labelsTr_topbrain_mr/*.nii.gz`

## Train

The current loader defaults point to the resampled CTA folders, so after preprocessing you can run:

```bash
.venv/bin/python train.py
```

At the moment, `train.py` is a short sanity-check training run that verifies the pipeline, loss, and model wiring. It is useful as a baseline check before adding longer experiment scripts.

## Useful Files

- `preprocess_resample.py`: one-time offline resampling
- `data_loader.py`: dataset and dataloader logic
- `data_utils.py`: preprocessing and helper functions
- `train.py`: baseline training/sanity script
- `data_inspection/inspect_data.py`: visual data inspection
- `DOCUMENTATION.md`: detailed project notes and implementation decisions
- `Competision_README.md`: archived challenge/submission instructions

## Collaboration Notes

- Large datasets and generated outputs are excluded from git via `.gitignore`.
- Code and documentation should be committed; raw and resampled medical volumes should stay local.
- If someone clones the repo fresh, they should:
  1. install dependencies
  2. place raw data in `training_data/`
  3. run `preprocess_resample.py`
  4. run `train.py`
