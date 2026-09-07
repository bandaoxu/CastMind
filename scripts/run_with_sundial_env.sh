#!/usr/bin/env bash
# Run CastMind with the Sundial-compatible env (transformers==4.40.1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-sundial"
PY="${VENV}/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}"
  echo "        Run: bash scripts/setup_sundial_env.sh"
  exit 1
fi

export ORCHESTRATION_MODE="${ORCHESTRATION_MODE:-deterministic}"
export CASTMIND_RUNTIME="${CASTMIND_RUNTIME:-sundial}"
# Auto-archive to outputs/_archive/<dataset>_… after a finished run (set 0 to disable).
export CASTMIND_AUTO_ARCHIVE="${CASTMIND_AUTO_ARCHIVE:-1}"

cd "${ROOT}"
echo "[info] Using Sundial side env: ${VENV}"
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
    print("[warn] Sundial not in pool (need foundation_models/sundial-base-128m and transformers 4.40.x)")
PY

exec "${PY}" "${ROOT}/run_experiment.py" "$@"
