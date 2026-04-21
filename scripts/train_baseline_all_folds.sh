#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_baseline_all_folds.sh <parent_out_dir> [folds] [extra train.py args...]
#
# Example:
#   bash scripts/train_baseline_all_folds.sh runs/cldice_kfold 0,1,2,3,4 \
#        --epochs 40 --cldice-weight 1.0 --cldice-iters 12
#
# Each fold's artifacts are written to <parent_out_dir>/fold_<k>/.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <parent_out_dir> [folds] [extra train.py args...]" >&2
  exit 1
fi

PARENT_OUT_DIR="$1"; shift
FOLDS_ARG="${1:-0,1,2,3,4}"
if [[ $# -ge 1 ]]; then shift; fi
EXTRA_ARGS=("$@")

SPLITS_JSON="${SPLITS_JSON:-training_data_resampled/split/splits_final.json}"

if [[ ! -f "${SPLITS_JSON}" ]]; then
  echo "Splits file not found: ${SPLITS_JSON}" >&2
  echo "Generate it first: python scripts/make_splits.py --n-folds 5" >&2
  exit 1
fi

IFS=',' read -r -a FOLDS <<<"${FOLDS_ARG}"

for FOLD in "${FOLDS[@]}"; do
  FOLD_TRIMMED="$(echo "${FOLD}" | tr -d '[:space:]')"
  if [[ -z "${FOLD_TRIMMED}" ]]; then
    continue
  fi
  echo "=========================================="
  echo "Baseline training fold ${FOLD_TRIMMED}"
  echo "=========================================="
  python src/train.py \
    --out-dir "${PARENT_OUT_DIR}" \
    --fold-subdir \
    --splits-json "${SPLITS_JSON}" \
    --fold "${FOLD_TRIMMED}" \
    "${EXTRA_ARGS[@]}"
done

echo "Finished all baseline folds: ${FOLDS_ARG}"
echo "Artifacts under: ${PARENT_OUT_DIR}/fold_*"
