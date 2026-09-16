#!/usr/bin/env bash
# Run CastMind with the main project .venv (transformers==4.40.1, Sundial-capable).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}"
  echo "        Run: bash scripts/setup_env.sh"
  exit 1
fi

export ORCHESTRATION_MODE="${ORCHESTRATION_MODE:-deterministic}"
export CASTMIND_RUNTIME="${CASTMIND_RUNTIME:-main}"
# Auto-archive to outputs/_archive/<dataset>_… after a finished run (set 0 to disable).
export CASTMIND_AUTO_ARCHIVE="${CASTMIND_AUTO_ARCHIVE:-1}"

cd "${ROOT}"
echo "[info] Using main env: ${VENV}"
echo "[info] ORCHESTRATION_MODE=${ORCHESTRATION_MODE}"
echo "[info] CASTMIND_RUNTIME=${CASTMIND_RUNTIME} AUTO_ARCHIVE=${CASTMIND_AUTO_ARCHIVE}"
"${PY}" - <<'PY'
import transformers
from castmind.models.base import get_default_models
maj = int(str(transformers.__version__).split(".", 1)[0])
names = [getattr(m, "alias", type(m).__name__) for m in get_default_models()]
print(f"[info] transformers={transformers.__version__} (major={maj})")
print(f"[info] baseline pool ({len(names)}): {', '.join(names)}")
if "Sundial" not in names:
    print(
        "[warn] Sundial not in pool "
        "(need foundation_models/sundial-base-128m and transformers 4.40.x)"
    )
if maj >= 5:
    print("[warn] transformers>=5: Sundial will be skipped. Re-run scripts/setup_env.sh")
PY

# Default: run_experiment.py. Or: scripts/run.sh scripts/eval_baselines_table.py --datasets ETTh1
if [[ "${1:-}" == *.py ]]; then
  SCRIPT="$1"
  shift
  if [[ "${SCRIPT}" != /* ]]; then
    SCRIPT="${ROOT}/${SCRIPT}"
  fi
  exec "${PY}" "${SCRIPT}" "$@"
fi
exec "${PY}" "${ROOT}/run_experiment.py" "$@"
