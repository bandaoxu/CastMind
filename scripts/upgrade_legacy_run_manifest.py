#!/usr/bin/env python3
"""Upgrade a legacy run_manifest.json so --resume can pass today's fingerprint check."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from castmind.config import apply_ablation, load_config  # noqa: E402
from castmind.data_loader import infer_target_column  # noqa: E402
from castmind.run_layout import (  # noqa: E402
    build_run_fingerprint,
    load_run_manifest,
    runs_root,
    upgrade_legacy_run_manifest,
    verify_resume_manifest,
)


def _resolve_dataset(cfg, dataset: str):
    token = dataset.strip().lower()
    alias_lookup = {}
    for ds in cfg.datasets:
        for alias in ds.all_aliases():
            alias_lookup.setdefault(alias, ds)
    if token not in alias_lookup:
        known = ", ".join(sorted({ds.name for ds in cfg.datasets}))
        raise SystemExit(f"Unknown dataset {dataset!r}. Known: {known}")
    return alias_lookup[token]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade a legacy run_manifest to the current fingerprint schema."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. EPF_FR")
    parser.add_argument("--run-name", required=True, help="Run folder name, e.g. Full_1")
    parser.add_argument("--config", default="config.yaml", help="Experiment config path")
    parser.add_argument(
        "--ablation",
        default=None,
        help="Ablation id (same as the original run; default: none / full)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print changed keys without writing",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ablation_id = apply_ablation(cfg, args.ablation or os.getenv("CASTMIND_ABLATION"))
    cfg.run_name = args.run_name
    ds = _resolve_dataset(cfg, args.dataset)
    # Match run_experiment.py: --dataset narrows cfg.datasets before fingerprinting.
    cfg.datasets = [ds]

    run_dir = runs_root(cfg.output_dir, ds.name) / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    stored = load_run_manifest(run_dir)
    if stored is None:
        raise SystemExit(f"Missing run_manifest.json under {run_dir}")

    lib_id = str(stored.get("lib_id") or "")
    if not lib_id:
        raise SystemExit("Legacy upgrade refused: stored manifest missing lib_id")

    train_df = pd.read_csv(ds.training_csv)
    target_column = str(stored.get("target_column") or infer_target_column(train_df, ds.name))
    # Align env-derived fingerprint fields with the stored run when present.
    if stored.get("model") and not (os.getenv("MODEL") or "").strip():
        os.environ["MODEL"] = str(stored["model"])
    if stored.get("orchestration_mode") and not (os.getenv("ORCHESTRATION_MODE") or "").strip():
        os.environ["ORCHESTRATION_MODE"] = str(stored["orchestration_mode"])
    current = build_run_fingerprint(
        cfg,
        dataset_name=ds.name,
        training_csv=ds.training_csv,
        test_csv=ds.test_csv,
        target_column=target_column,
        look_back=int(ds.look_back),
        sliding_window=int(ds.sliding_window),
        predicted_window=int(ds.predicted_window),
        ablation_id=ablation_id or "",
        lib_id=lib_id,
    )

    try:
        upgraded = upgrade_legacy_run_manifest(run_dir, current, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    changed = upgraded.pop("_upgrade_changed_keys", [])
    mode = "dry-run" if args.dry_run else "wrote"
    print(f"[info] {mode}: {run_dir / 'run_manifest.json'}")
    print(f"[info] changed_keys={changed or '(none; already aligned)'}")
    if not args.dry_run:
        print(f"[info] backup={run_dir / 'run_manifest.pre_upgrade.json'}")

    try:
        verify_resume_manifest(upgraded, current)
    except RuntimeError as exc:
        print(f"[error] post-upgrade verify failed: {exc}", file=sys.stderr)
        return 1
    print("[info] verify_resume_manifest: ok")
    if args.dry_run:
        print("[info] Re-run without --dry-run to write, then --resume the experiment.")
    else:
        print(
            f"[info] Next: ORCHESTRATION_MODE=llm bash scripts/run.sh "
            f"--dataset {ds.name} --run-name {args.run_name} --resume"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
