#!/usr/bin/env bash
# Evaluate baselines for one run, then refresh the all-dataset comparison table.
# Usage: bash scripts/run_epf_baseline_comparison.sh <DATASET> <RUN_NAME>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}; run: bash scripts/setup_env.sh" >&2
  exit 1
fi

DATASET="${1:-}"
RUN_NAME="${2:-}"
if [[ -z "${DATASET}" || -z "${RUN_NAME}" ]]; then
  echo "Usage: bash scripts/run_epf_baseline_comparison.sh <DATASET> <RUN_NAME>" >&2
  echo "  Example: bash scripts/run_epf_baseline_comparison.sh EPF_BE Full_1" >&2
  exit 1
fi

PRED="outputs/${DATASET}/runs/${RUN_NAME}/predictions.csv"
if [[ ! -f "${PRED}" ]]; then
  echo "[error] Missing predictions: ${PRED}" >&2
  exit 1
fi

ROOT_CMP="outputs/comparison"
DS_DIR="${ROOT_CMP}/${DATASET}"

# Paper Table-1 dataset order (all local comparison datasets).
ALL_DATASETS=(
  EPF_BE EPF_DE EPF_FR EPF_NP EPF_PJM
  ETTh1 ETTm1 windy_power sunny_power MOPEX
)

echo "[info] dataset=${DATASET}"
echo "[info] run_name=${RUN_NAME}"
echo "[info] predictions=${PRED}"
echo "[info] output-dir=${DS_DIR}"

bash scripts/run.sh scripts/eval_baselines_table.py \
  --datasets "${DATASET}" \
  --output-dir "${DS_DIR}" \
  --from-archive "CastMind_current=${PRED}"

"${PY}" scripts/assemble_comparison_table.py \
  --datasets "${DATASET}" \
  --comparison-dir "${DS_DIR}" \
  --out-md "${DS_DIR}/table.md" \
  --out-copy-md "${DS_DIR}/table_local.md" \
  --out-csv "${DS_DIR}/table.csv"

mkdir -p "${ROOT_CMP}"
cp "${DS_DIR}/${DATASET}_baselines.json" "${ROOT_CMP}/"
cp "${DS_DIR}/${DATASET}_castmind_CastMind_current.json" "${ROOT_CMP}/"

"${PY}" scripts/assemble_comparison_table.py \
  --datasets "${ALL_DATASETS[@]}" \
  --comparison-dir "${ROOT_CMP}" \
  --out-md "${ROOT_CMP}/table.md" \
  --out-copy-md "${ROOT_CMP}/table_local.md" \
  --out-csv "${ROOT_CMP}/table.csv"

echo "[info] refreshed ${ROOT_CMP}/table.md (all datasets)"
echo "[info] per-dataset table: ${DS_DIR}/table.md"
