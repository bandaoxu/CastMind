#!/usr/bin/env python3
"""Install official / external DL checkpoints into CastMind's expected paths.

Usage:
  .venv/bin/python scripts/install_official_checkpoints.py \\
      --source /path/to/downloaded_ckpts \\
      --dataset ETTh1

Expected source layout (either is accepted):
  <source>/Autoformer.pth ... TimesNet.pth
  <source>/<dataset>/Autoformer.pth ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "castmind" / "DeepLearningCheckpoints"
REQUIRED = ("Autoformer", "DLinear", "PatchTST", "TimesNet", "iTransformer")


def _find_model_file(src: Path, dataset: str, name: str) -> Path | None:
    candidates = [
        src / f"{name}.pth",
        src / dataset / f"{name}.pth",
        src / name / f"{name}.pth",
        src / f"{name}.pt",
        src / dataset / f"{name}.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def install_dataset(source: Path, dataset: str, *, copy: bool, force: bool) -> list[str]:
    dest_dir = DEST_ROOT / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    missing: list[str] = []
    for name in REQUIRED:
        src_file = _find_model_file(source, dataset, name)
        if src_file is None:
            missing.append(name)
            continue
        dest = dest_dir / f"{name}.pth"
        if dest.exists() or dest.is_symlink():
            if not force:
                print(f"[skip] {dest} exists (use --force to replace)")
                installed.append(name)
                continue
            dest.unlink()
        if copy:
            shutil.copy2(src_file, dest)
            print(f"[copy] {src_file} -> {dest}")
        else:
            dest.symlink_to(src_file.resolve())
            print(f"[link] {src_file} -> {dest}")
        installed.append(name)
    if missing:
        raise SystemExit(f"[error] missing checkpoint files for {dataset}: {', '.join(missing)}")
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing official .pth files")
    parser.add_argument("--dataset", required=True, help="Dataset name matching config.yaml (e.g. ETTh1)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    parser.add_argument("--force", action="store_true", help="Replace existing destinations")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"[error] source is not a directory: {source}")

    installed = install_dataset(source, args.dataset, copy=args.copy, force=args.force)
    print(f"[ok] {args.dataset}: {', '.join(installed)}")
    print(f"[ok] destination: {DEST_ROOT / args.dataset}")
    print(
        "[note] Ensure config.yaml checkpoints for this dataset point at "
        f"castmind/DeepLearningCheckpoints/{args.dataset}/*.pth"
    )
    print(f"[note] If checkpoint widths differ from local light ckpts, update "
        "castmind.models.base.dl_backbone_hparams to match before inference.")


if __name__ == "__main__":
    main()
