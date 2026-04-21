#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash nnunet_impl/run_predict.sh [dataset_id] [configuration] [fold] [input_dir] [output_dir]
#
# Example:
#   bash nnunet_impl/run_predict.sh 501 3d_fullres 0
#
# Optional W&B env passthrough (for tools/scripts that read wandb settings):
#   NNUNET_WANDB=1
#   NNUNET_WANDB_PROJECT=topbrain
#   NNUNET_WANDB_ENTITY=my-team
#   NNUNET_WANDB_RUN_NAME=predict_fold0
#   NNUNET_WANDB_TAGS=nnunet,predict,fold0
#   NNUNET_WANDB_MODE=offline
#
# Optional extra predict args:
#   NNUNET_PREDICT_EXTRA_ARGS="--save_probabilities"

DATASET_ID="${1:-501}"
CONFIGURATION="${2:-3d_fullres}"
FOLD="${3:-0}"
INPUT_DIR="${4:-}"
OUTPUT_DIR="${5:-}"

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

if [[ -z "${INPUT_DIR}" ]]; then
  INPUT_DIR="${RAW_DATASET_DIR}/imagesVal"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="runs/nnunet/predictions_fold${FOLD}"
fi

mkdir -p "${OUTPUT_DIR}"

PREDICT_CMD=(
  nnUNetv2_predict
  -i "${INPUT_DIR}"
  -o "${OUTPUT_DIR}"
  -d "${DATASET_ID}"
  -c "${CONFIGURATION}"
  -f "${FOLD}"
)
if [[ -n "${NNUNET_PREDICT_EXTRA_ARGS:-}" ]]; then
  # Split extra args by shell words (quotes supported in env value).
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${NNUNET_PREDICT_EXTRA_ARGS} )
  PREDICT_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${PREDICT_CMD[*]}"
"${PREDICT_CMD[@]}"

echo "Predictions written to ${OUTPUT_DIR}"
