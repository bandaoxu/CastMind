#!/usr/bin/env python3
"""Backfill metrics.json from existing predictions.csv (no LLM re-run).

Scans outputs/_archive/* and optionally outputs/<DatasetName>/, aligns against
data/<ds>/test.csv, and writes metrics.json next to predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from castmind.config import ABLATION_CHOICES, load_config
from castmind.data_loader import TIME_COL
from castmind.eval import align_predictions, mae, mse, smape

ABLATION_SUFFIXES = tuple(sorted(ABLATION_CHOICES, key=len, reverse=True))


def _dataset_names(config_path: Path) -> List[str]:
    cfg = load_config(str(config_path))
    # Longest first so EPF_NP matches before EPF if both existed.
    return sorted((d.name for d in cfg.datasets), key=len, reverse=True)


def parse_archive_tag(tag: str, dataset_names: List[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (dataset, ablation_or_None, llm_model_guess_or_None)."""
    dataset = ""
    rest = tag
    for name in dataset_names:
        if tag == name or tag.startswith(name + "_"):
            dataset = name
            rest = tag[len(name) :].lstrip("_")
            break
    if not dataset:
        # Fallback: first token before _llm_ / _deterministic_
        m = re.match(r"^(.+?)_(?:llm|deterministic|sundial)_", tag)
        dataset = m.group(1) if m else tag.split("_")[0]

    ablation = None
    for ab in ABLATION_SUFFIXES:
        suf = "_" + ab
        if rest.endswith(suf):
            ablation = ab
            rest = rest[: -len(suf)]
            break
        if rest == ab:
            ablation = ab
            rest = ""
            break

    llm_model = None
    # Common slugs from _archive_tag model_slug = re.sub(r'[^a-z0-9]+', '', model)
    slug_map = {
        "deepseekchat": "deepseek-chat",
        "deepseek": "deepseek-chat",
        "gpt41": "gpt-4.1",
        "gpt4omini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
    }
    m = re.search(r"(?:^|_)llm_([a-z0-9]+)", rest)
    if m:
        llm_model = slug_map.get(m.group(1), m.group(1))
    return dataset, ablation, llm_model


def compute_metrics(dataset: str, pred_path: Path) -> Dict:
    test_path = ROOT / "data" / dataset / "test.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"missing test csv: {test_path}")
    test_df = pd.read_csv(test_path)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
    pred_df = pd.read_csv(pred_path)
    if "time_stamp" in pred_df.columns:
        pred_df["time_stamp"] = pd.to_datetime(pred_df["time_stamp"])
    y_true, y_pred = align_predictions(test_df, pred_df, dataset)
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "n": int(len(y_true)),
    }


def write_metrics(
    out_dir: Path,
    dataset: str,
    scores: Dict,
    *,
    ablation: Optional[str],
    llm_model: Optional[str],
    source: str,
    force: bool,
) -> Optional[Path]:
    path = out_dir / "metrics.json"
    if path.exists() and not force:
        print(f"[skip] {path} exists")
        return None
    payload = {
        "dataset": dataset,
        "MSE": float(scores["MSE"]),
        "MAE": float(scores["MAE"]),
        "sMAPE": float(scores["sMAPE"]),
        "n": int(scores["n"]),
        "model": "LLM",
        "orchestration_mode": "llm",
        "llm_model": llm_model,
        "ablation": ablation,
        "max_steps": None,
        "ablation_flags": None,
        "backfilled": True,
        "source": source,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[ok] {path}  n={payload['n']} MSE={payload['MSE']:.6f} MAE={payload['MAE']:.6f}"
        + (f" ablation={ablation}" if ablation else "")
    )
    return path


def collect_targets(
    archive: bool,
    working: bool,
    only: Optional[List[str]],
    dataset_names: List[str],
) -> List[Tuple[Path, str, Optional[str], Optional[str], str]]:
    """List of (dir, dataset, ablation, llm_model, source_label)."""
    found: List[Tuple[Path, str, Optional[str], Optional[str], str]] = []
    only_set = {x.strip() for x in (only or []) if x.strip()}

    if archive:
        root = ROOT / "outputs" / "_archive"
        if root.is_dir():
            for pred in sorted(root.glob("*/predictions.csv")):
                tag = pred.parent.name
                if only_set and tag not in only_set and not any(tag.startswith(o) for o in only_set):
                    continue
                ds, ab, model = parse_archive_tag(tag, dataset_names)
                found.append((pred.parent, ds, ab, model, f"_archive/{tag}"))

    if working:
        out = ROOT / "outputs"
        for name in dataset_names:
            pred = out / name / "predictions.csv"
            if not pred.is_file():
                continue
            if only_set and name not in only_set:
                continue
            found.append((pred.parent, name, None, None, f"outputs/{name}"))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill metrics.json from predictions.csv")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing metrics.json")
    parser.add_argument("--no-archive", action="store_true", help="Skip outputs/_archive")
    parser.add_argument("--working", action="store_true", help="Also backfill outputs/<DatasetName>/")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only process matching archive tag or dataset name (repeatable).",
    )
    args = parser.parse_args()

    dataset_names = _dataset_names(Path(args.config))
    targets = collect_targets(
        archive=not args.no_archive,
        working=args.working,
        only=args.only or None,
        dataset_names=dataset_names,
    )
    if not targets:
        print("[warn] No prediction directories found.")
        return 1

    ok = 0
    fail = 0
    for out_dir, dataset, ablation, llm_model, source in targets:
        pred = out_dir / "predictions.csv"
        try:
            scores = compute_metrics(dataset, pred)
            wrote = write_metrics(
                out_dir,
                dataset,
                scores,
                ablation=ablation,
                llm_model=llm_model,
                source=source,
                force=args.force,
            )
            if wrote is not None:
                ok += 1
        except Exception as exc:
            fail += 1
            print(f"[error] {out_dir}: {exc}")

    print(f"[done] wrote={ok} failed={fail} scanned={len(targets)}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
