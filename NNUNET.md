# nnUNet v2 Separate Pipeline

This project includes a `nnunetv2` pipeline in `nnunet_impl/` that:

- Reuses existing data preprocessing outputs in `training_data_resampled/`.
- Shares a canonical K-fold `splits_final.json` with the baseline in `src/` so
  both pipelines see identical folds for apples-to-apples comparison.
- Keeps nnUNet training/inference isolated from `src/train.py`.

## 0) Generate canonical splits (once)

Both pipelines read the same file:

```bash
python scripts/make_splits.py --n-folds 5
```

Output: `training_data_resampled/split/splits_final.json`, a list of
`{"train": [...], "val": [...]}` entries (nnUNet's native split format).

Advanced options:

- `--holdout-file path/to/test_cases.txt` to carve off a held-out test set.
- `--group-regex '^(.*)_\d+$'` for patient/group-aware folds (`GroupKFold`).
- `--seed 12345` (default; matches nnUNet's internal seed).
- `--force` to overwrite an existing file.

## 1) Install nnUNet v2

```bash
pip install nnunetv2
```

## 2) Build nnUNet dataset + run planning/preprocessing

```bash
bash nnunet_impl/run_plan_and_preprocess.sh 501 TopBrainCTA
```

This command:

1. Converts preprocessed data into nnUNet layout via `nnunet_impl/convert_dataset.py`.
2. Copies `splits_final.json` (all folds) into both the raw and preprocessed
   dataset directories so `nnUNetv2_train` uses your folds verbatim.
3. Runs `nnUNetv2_plan_and_preprocess`.

Default folders used (override with env vars if needed):

- `NNUNET_RAW_DIR=nnUNet_raw`
- `NNUNET_PREPROCESSED_DIR=nnUNet_preprocessed`
- `NNUNET_RESULTS_DIR=nnUNet_results`

Useful `convert_dataset.py` flags:

- `--write-per-fold-imagesval` to also create `imagesVal_fold0`, `imagesVal_fold1`,
  etc. for explicit per-fold `nnUNetv2_predict` runs.
- `--link-mode copy` on filesystems without symlink/hardlink support.

## 3) Train nnUNet

Single fold:

```bash
bash nnunet_impl/run_train.sh 501 3d_fullres 0
```

All folds in sequence:

```bash
bash nnunet_impl/run_train_all_folds.sh 501 3d_fullres 0,1,2,3,4
```

- `501` = dataset id
- `3d_fullres` = configuration
- Third arg = comma-separated fold indices.

## 4) Predict validation cases

nnUNet writes per-fold validation predictions automatically to
`nnUNet_results/Dataset501_TopBrainCTA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_<k>/validation/`
after each fold's training finishes. For most K-fold workflows you don't need
to run prediction separately.

If you *do* want standalone prediction (e.g. to reuse the checkpoint on other
inputs) and have written per-fold imagesVal folders:

```bash
bash nnunet_impl/run_predict_all_folds.sh 501 3d_fullres 0,1,2,3,4
```

Outputs land in `runs/nnunet/predictions_fold<k>/`.

## 5) Out-of-fold (OOF) evaluation

The cleanest way to compare pipelines is to stitch each fold's validation
predictions together so every case is predicted exactly once. Use
`scripts/eval_oof.py` with a `{fold}` placeholder in the directory pattern.

nnUNet (built-in validation dumps):

```bash
python scripts/eval_oof.py \
  --fold-pattern 'nnUNet_results/Dataset501_TopBrainCTA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/validation' \
  --out-dir runs/nnunet/oof \
  --run-topbrain-eval --topbrain-track cta
```

nnUNet (predictions produced by `run_predict_all_folds.sh`):

```bash
python scripts/eval_oof.py \
  --fold-pattern 'runs/nnunet/predictions_fold{fold}' \
  --out-dir runs/nnunet/oof \
  --run-topbrain-eval --topbrain-track cta
```

Single-fold export (legacy path) is still available via
`nnunet_impl/export_for_eval.py --prediction-dir ... --fold <k>`.

## 6) Baseline K-fold (mirrors the nnUNet loop)

```bash
bash scripts/train_baseline_all_folds.sh runs/cldice_kfold 0,1,2,3,4 \
  --epochs 40 --cldice-weight 1.0 --cldice-iters 12
```

This writes `runs/cldice_kfold/fold_0/`, `.../fold_1/`, etc., each containing
the normal `metrics.csv` / `model_final_weights.pt`. Use `scripts/eval_oof.py`
the same way to stitch per-fold predictions once you've run the baseline's
challenge-like evaluation per fold.

## Notes

- The baseline's `src/train.py` now accepts `--splits-json` and `--fold`. When
  `training_data_resampled/split/splits_final.json` exists it is used by
  default; otherwise the legacy `train_cases.txt` / `val_cases.txt` path is
  preserved.
- nnUNet's default split (random 5-fold, seed 12345) is overridden the moment
  `splits_final.json` is present in its preprocessed dataset directory.
- If your filesystem does not support symlinks/hardlinks, run conversion with
  `--link-mode copy`.
