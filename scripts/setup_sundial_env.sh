#!/usr/bin/env bash
# Sundial (thuml/sundial-base-128m) needs transformers 4.x APIs.
# Main CastMind env may use transformers>=5; keep an isolated side venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-sundial"
MAIN_PY="${ROOT}/.venv/bin/python"
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${MAIN_PY}" ]]; then
    PYTHON="${MAIN_PY}"
  else
    PYTHON="python3"
  fi
fi

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip

# Runtime deps (include Prophet so case-library matches main-env baseline pool).
pip install \
  "pydantic-ai>=0.0.15" \
  "pydantic>=2.8" \
  "python-dotenv>=1.0" \
  "pyyaml>=6.0.2" \
  "pandas>=2.2" \
  "numpy>=1.26" \
  "scipy>=1.11" \
  "statsmodels>=0.14" \
  "statsforecast>=2.0.2" \
  "tsfeatures>=0.4.5" \
  "reformer_pytorch>=1.4.4" \
  "pyclustering>=0.10.1" \
  "scikit-learn>=1.3" \
  "accelerate" \
  "safetensors" \
  "chronos-forecasting>=1.4.0" \
  "prophet>=1.1.7"

# Prefer copying torch from the main project venv (avoids huge flaky downloads).
SP_MAIN="${ROOT}/.venv/lib/python3.11/site-packages"
SP_SD="${VENV}/lib/python3.11/site-packages"
if [[ -d "${SP_MAIN}/torch" && ! -d "${SP_SD}/torch" ]]; then
  echo "[info] Copying torch from main .venv"
  cp -a "${SP_MAIN}/torch" "${SP_SD}/"
  cp -a "${SP_MAIN}"/torch-*.dist-info "${SP_SD}/" 2>/dev/null || true
fi
# Fallback: pip install torch if still missing.
python -c "import torch" 2>/dev/null || pip install "torch>=2.5.1"

# Force Sundial-compatible stack last (README: transformers==4.40.1).
pip install "transformers==4.40.1" "tokenizers>=0.19,<0.20" "huggingface-hub>=0.23,<1.0"

echo "[ok] Sundial side env: ${VENV}"
echo "     python: $(command -v python)"
python - <<'PY'
import transformers
print(f"     transformers={transformers.__version__}")
maj = int(str(transformers.__version__).split(".", 1)[0])
raise SystemExit(0 if maj < 5 else 1)
PY
echo "Use: bash scripts/run_with_sundial_env.sh --dataset ETTh1"
echo "Note: main .venv still skips Sundial on transformers>=5."
