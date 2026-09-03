#!/usr/bin/env python3
"""Download Chronos (and optionally Sundial) weights used by CastMind."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FM = ROOT / "castmind" / "foundation_models"


def snapshot(repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    print(f"[ok] {dest}")


def main() -> None:
    FM.mkdir(parents=True, exist_ok=True)
    snapshot("amazon/chronos-bolt-small", FM / "chronos-bolt-base")
    try:
        snapshot("thuml/sundial-base-128m", FM / "sundial-base-128m")
    except Exception as exc:
        print(f"[warn] Sundial download skipped: {exc}")


if __name__ == "__main__":
    main()
