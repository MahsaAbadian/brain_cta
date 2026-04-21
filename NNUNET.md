# nnUNet v2 Separate Pipeline

This project now includes a separate `nnunetv2` pipeline in `nnunet_impl/` that:

- Reuses existing data preprocessing outputs in `training_data_resampled/`.
- Reuses existing train/val split files in `training_data_resampled/split/`.
- Keeps nnUNet training/inference isolated from `src/train.py`.

## 1) Install nnUNet v2

Install in the same environment you use for this repo:

```bash
pip install nnunetv2
```

## 2) Build nnUNet dataset + run planning/preprocessing

```bash
bash nnunet_impl/run_plan_and_preprocess.sh 501 TopBrainCTA
```

This command:

1. Converts current preprocessed data into nnUNet layout via `nnunet_impl/convert_dataset.py`.
2. Writes `splits_final.json` from current `train_cases.txt` / `val_cases.txt`.
3. Runs `nnUNetv2_plan_and_preprocess`.

Default folders used (override with env vars if needed):

- `NNUNET_RAW_DIR=nnUNet_raw`
- `NNUNET_PREPROCESSED_DIR=nnUNet_preprocessed`
- `NNUNET_RESULTS_DIR=nnUNet_results`

## 3) Train nnUNet

```bash
bash nnunet_impl/run_train.sh 501 3d_fullres 0
```

- `501` = dataset id
- `3d_fullres` = configuration
- `0` = fold

Use additional folds by changing the fold argument.

## 4) Predict validation cases

```bash
bash nnunet_impl/run_predict.sh 501 3d_fullres 0
```

By default this predicts from `imagesVal` generated from your validation split.

## 5) Export outputs for existing challenge-like evaluation

```bash
python nnunet_impl/export_for_eval.py \
  --prediction-dir runs/nnunet/predictions_fold0 \
  --out-dir runs/nnunet/challenge_like_eval
```

Optional TopBrain eval:

```bash
python nnunet_impl/export_for_eval.py \
  --prediction-dir runs/nnunet/predictions_fold0 \
  --out-dir runs/nnunet/challenge_like_eval \
  --run-topbrain-eval --topbrain-track cta
```

## Notes

- The custom baseline pipeline in `src/train.py` remains unchanged.
- This setup is intended for fair side-by-side benchmarking using the same split and evaluation style.
- If your filesystem does not support symlinks/hardlinks, run conversion with `--link-mode copy`.
