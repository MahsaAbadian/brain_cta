#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash nnunet_impl/run_train.sh [dataset_id] [configuration] [fold]
#
# Example:
#   bash nnunet_impl/run_train.sh 501 3d_fullres 0
#
# Optional W&B env passthrough (for trainers that read wandb settings):
#   NNUNET_WANDB=1
#   NNUNET_WANDB_PROJECT=topbrain
#   NNUNET_WANDB_ENTITY=my-team
#   NNUNET_WANDB_RUN_NAME=exp_ps128
#   NNUNET_WANDB_TAGS=baseline,cldice
#   NNUNET_WANDB_MODE=offline
#
# Optional extra train args:
#   NNUNET_TRAIN_EXTRA_ARGS="--npz --c"

DATASET_ID="${1:-501}"
CONFIGURATION="${2:-3d_fullres}"
FOLD="${3:-0}"

NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-nnUNet_results}"

export nnUNet_raw="${NNUNET_RAW_DIR}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED_DIR}"
export nnUNet_results="${NNUNET_RESULTS_DIR}"

if [[ "${NNUNET_WANDB:-0}" == "1" ]]; then
  export WANDB_PROJECT="${NNUNET_WANDB_PROJECT:-${WANDB_PROJECT:-topbrain}}"
  if [[ -n "${NNUNET_WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY="${NNUNET_WANDB_ENTITY}"
  fi
  if [[ -n "${NNUNET_WANDB_RUN_NAME:-}" ]]; then
    export WANDB_NAME="${NNUNET_WANDB_RUN_NAME}"
  fi
  if [[ -n "${NNUNET_WANDB_TAGS:-}" ]]; then
    export WANDB_TAGS="${NNUNET_WANDB_TAGS}"
  fi
  if [[ -n "${NNUNET_WANDB_MODE:-}" ]]; then
    export WANDB_MODE="${NNUNET_WANDB_MODE}"
  fi
  echo "W&B env enabled: project=${WANDB_PROJECT} entity=${WANDB_ENTITY:-unset} mode=${WANDB_MODE:-online}"
fi

DATASET_PREFIX="$(printf "Dataset%03d_" "${DATASET_ID}")"
MATCHES=("${NNUNET_RAW_DIR}/${DATASET_PREFIX}"*)
if [[ ! -d "${MATCHES[0]}" ]]; then
  echo "Cannot find raw dataset folder for id ${DATASET_ID} under ${NNUNET_RAW_DIR}" >&2
  exit 1
fi
RAW_DATASET_DIR="${MATCHES[0]}"
DATASET_FOLDER="$(basename "${RAW_DATASET_DIR}")"

RAW_SPLIT_FILE="${RAW_DATASET_DIR}/splits_final.json"
PREPROCESSED_DATASET_DIR="${NNUNET_PREPROCESSED_DIR}/${DATASET_FOLDER}"
PREPROCESSED_SPLIT_FILE="${PREPROCESSED_DATASET_DIR}/splits_final.json"

if [[ -f "${RAW_SPLIT_FILE}" ]]; then
  mkdir -p "${PREPROCESSED_DATASET_DIR}"
  cp "${RAW_SPLIT_FILE}" "${PREPROCESSED_SPLIT_FILE}"
fi

TRAIN_CMD=(nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}")
if [[ -n "${NNUNET_TRAIN_EXTRA_ARGS:-}" ]]; then
  # Split extra args by shell words (quotes supported in env value).
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${NNUNET_TRAIN_EXTRA_ARGS} )
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
