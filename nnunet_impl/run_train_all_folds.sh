#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash nnunet_impl/run_train_all_folds.sh [dataset_id] [configuration] [folds]
#
# Examples:
#   bash nnunet_impl/run_train_all_folds.sh 501 3d_fullres         # folds 0..4
#   bash nnunet_impl/run_train_all_folds.sh 501 3d_fullres 0,2,4   # subset
#
# Environment variables forwarded to run_train.sh:
#   NNUNET_RAW_DIR, NNUNET_PREPROCESSED_DIR, NNUNET_RESULTS_DIR

DATASET_ID="${1:-501}"
CONFIGURATION="${2:-3d_fullres}"
FOLDS_ARG="${3:-0,1,2,3,4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -r -a FOLDS <<<"${FOLDS_ARG}"

for FOLD in "${FOLDS[@]}"; do
  FOLD_TRIMMED="$(echo "${FOLD}" | tr -d '[:space:]')"
  if [[ -z "${FOLD_TRIMMED}" ]]; then
    continue
  fi
  echo "=========================================="
  echo "Training fold ${FOLD_TRIMMED} (${CONFIGURATION})"
  echo "=========================================="
  bash "${SCRIPT_DIR}/run_train.sh" "${DATASET_ID}" "${CONFIGURATION}" "${FOLD_TRIMMED}"
done

echo "Finished all requested folds: ${FOLDS_ARG}"
