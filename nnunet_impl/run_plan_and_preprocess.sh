#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash nnunet_impl/run_plan_and_preprocess.sh [dataset_id] [dataset_name]
#
# Optional environment variables:
#   NNUNET_RAW_DIR
#   NNUNET_PREPROCESSED_DIR
#   NNUNET_RESULTS_DIR
#   LINK_MODE (symlink|hardlink|copy)

DATASET_ID="${1:-501}"
DATASET_NAME="${2:-TopBrainCTA}"
LINK_MODE="${LINK_MODE:-symlink}"

NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-nnUNet_results}"

export nnUNet_raw="${NNUNET_RAW_DIR}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED_DIR}"
export nnUNet_results="${NNUNET_RESULTS_DIR}"

python "nnunet_impl/convert_dataset.py" \
  --dataset-id "${DATASET_ID}" \
  --dataset-name "${DATASET_NAME}" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --link-mode "${LINK_MODE}"

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  --verify_dataset_integrity

echo "Finished planning/preprocessing for dataset ${DATASET_ID} (${DATASET_NAME})."
