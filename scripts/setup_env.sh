#!/usr/bin/env bash
# Create/refresh the main project .venv (Sundial-compatible: transformers==4.40.1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip
pip install -r "${ROOT}/requirements.txt"

# Force Sundial-compatible stack last (Sundial README: transformers==4.40.1).
pip install "transformers==4.40.1" "tokenizers>=0.19,<0.20" "huggingface-hub>=0.23,<1.0"

echo "[ok] Main env: ${VENV}"
echo "     python: $(command -v python)"
python - <<'PY'
import transformers
print(f"     transformers={transformers.__version__}")
maj = int(str(transformers.__version__).split(".", 1)[0])
raise SystemExit(0 if maj < 5 else 1)
PY
echo "Activate: source .venv/bin/activate"
echo "Run:     python run_experiment.py --dataset ETTh1"
echo "   or:   bash scripts/run.sh --dataset ETTh1"
