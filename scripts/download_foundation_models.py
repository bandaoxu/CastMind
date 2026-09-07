#!/usr/bin/env python3
"""Download Chronos-Bolt-Base and Sundial weights used by CastMind (AlphaCast §4.1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FM = ROOT / "castmind" / "foundation_models"

# Paper/base-line Chronos size is bolt-base (~205M params, ~821MB weights).
# Do NOT download bolt-small into a directory named chronos-bolt-base.
CHRONOS_REPO = "amazon/chronos-bolt-base"
CHRONOS_DIR = FM / "chronos-bolt-base"
SUNDIAL_REPO = "thuml/sundial-base-128m"
SUNDIAL_DIR = FM / "sundial-base-128m"

# bolt-small is ~190MB; bolt-base safetensors is ~821MB.
_MIN_CHRONOS_BASE_BYTES = 500_000_000


def snapshot(repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    print(f"[ok] {dest}")


def _chronos_looks_like_base(dest: Path) -> bool:
    weights = dest / "model.safetensors"
    if not weights.is_file():
        return False
    if weights.stat().st_size < _MIN_CHRONOS_BASE_BYTES:
        return False
    cfg_path = dest / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            d_model = int(cfg.get("d_model") or 0)
            # t5-efficient-small ≈ 512; base ≈ 768
            if d_model and d_model < 700:
                return False
        except Exception:
            pass
    return True


def ensure_chronos(*, force: bool = False) -> None:
    if not force and _chronos_looks_like_base(CHRONOS_DIR):
        print(f"[skip] Chronos bolt-base already present at {CHRONOS_DIR}")
        return
    if CHRONOS_DIR.exists() and not _chronos_looks_like_base(CHRONOS_DIR):
        print(
            f"[warn] {CHRONOS_DIR} does not look like chronos-bolt-base "
            f"(likely bolt-small). Re-downloading {CHRONOS_REPO}..."
        )
    snapshot(CHRONOS_REPO, CHRONOS_DIR)
    if not _chronos_looks_like_base(CHRONOS_DIR):
        raise RuntimeError(
            f"Downloaded Chronos weights at {CHRONOS_DIR} still fail bolt-base checks. "
            f"Verify HF access to {CHRONOS_REPO}."
        )


def ensure_sundial(*, force: bool = False) -> None:
    marker = SUNDIAL_DIR / "config.json"
    if not force and marker.is_file():
        print(f"[skip] Sundial already present at {SUNDIAL_DIR}")
        return
    snapshot(SUNDIAL_REPO, SUNDIAL_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--skip-sundial", action="store_true")
    parser.add_argument("--skip-chronos", action="store_true")
    args = parser.parse_args()

    FM.mkdir(parents=True, exist_ok=True)
    if not args.skip_chronos:
        ensure_chronos(force=args.force)
    if not args.skip_sundial:
        try:
            ensure_sundial(force=args.force)
        except Exception as exc:
            print(f"[warn] Sundial download skipped: {exc}")


if __name__ == "__main__":
    main()
