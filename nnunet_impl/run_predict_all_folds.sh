#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash nnunet_impl/run_predict_all_folds.sh [dataset_id] [configuration] [folds]
#
# For each fold, this predicts the cases that were held out as that fold's
# validation set. Requires nnunet_impl/convert_dataset.py to have been run
# with --write-per-fold-imagesval so that imagesVal_fold<k> directories exist.

DATASET_ID="${1:-501}"
CONFIGURATION="${2:-3d_fullres}"
FOLDS_ARG="${3:-0,1,2,3,4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-nnUNet_raw}"
DATASET_PREFIX="$(printf "Dataset%03d_" "${DATASET_ID}")"
MATCHES=("${NNUNET_RAW_DIR}/${DATASET_PREFIX}"*)
if [[ ! -d "${MATCHES[0]}" ]]; then
  echo "Cannot find raw dataset folder for id ${DATASET_ID} under ${NNUNET_RAW_DIR}" >&2
  exit 1
fi
RAW_DATASET_DIR="${MATCHES[0]}"

IFS=',' read -r -a FOLDS <<<"${FOLDS_ARG}"

for FOLD in "${FOLDS[@]}"; do
  FOLD_TRIMMED="$(echo "${FOLD}" | tr -d '[:space:]')"
  if [[ -z "${FOLD_TRIMMED}" ]]; then
    continue
  fi
  INPUT_DIR="${RAW_DATASET_DIR}/imagesVal_fold${FOLD_TRIMMED}"
  OUTPUT_DIR="runs/nnunet/predictions_fold${FOLD_TRIMMED}"
  if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Skipping fold ${FOLD_TRIMMED}: ${INPUT_DIR} does not exist." >&2
    echo "Re-run nnunet_impl/convert_dataset.py with --write-per-fold-imagesval." >&2
    continue
  fi
  echo "=========================================="
  echo "Predicting fold ${FOLD_TRIMMED}"
  echo "  input:  ${INPUT_DIR}"
  echo "  output: ${OUTPUT_DIR}"
  echo "=========================================="
  bash "${SCRIPT_DIR}/run_predict.sh" \
    "${DATASET_ID}" "${CONFIGURATION}" "${FOLD_TRIMMED}" \
    "${INPUT_DIR}" "${OUTPUT_DIR}"
done

echo "Finished predictions for folds: ${FOLDS_ARG}"
