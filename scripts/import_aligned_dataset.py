#!/usr/bin/env python3
"""Import paper-aligned CSVs into CastMind data/ with AlphaCast Table 6 splits.

MemCast's advertised Google Drive pack does NOT contain Windy/Sunny/MOPEX
(see scripts/memcast_drive_inventory.json). Use this script once you obtain
the real series from authors or the iFLYTEK contest.

Examples:
  .venv/bin/python scripts/import_aligned_dataset.py \\
      --dataset windy_power --csv /path/to/windy_full.csv

  .venv/bin/python scripts/import_aligned_dataset.py \\
      --dataset MOPEX --csv /path/to/mopex.csv \\
      --rename MAP=P,CPE=E,streamflow=Q,Tmax=Tmax,Tmin=Tmin

Target must end up as the last numeric column (CastMind infer_target_column).
Preferred targets: windy/sunny → real_power; MOPEX → streamflow.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# AlphaCast Table 6 lengths used by prepare_data.py
SPLITS: Dict[str, Tuple[int, int, int]] = {
    "windy_power": (16896, 2496, 4896),
    "sunny_power": (16896, 2496, 4896),
    "MOPEX": (8544, 1344, 2544),
}

PREFERRED_COLUMNS: Dict[str, list] = {
    "windy_power": [
        "date",
        "direct_radiation",
        "wind_direction_80m",
        "wind_speed_80m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "real_power",
    ],
    "sunny_power": [
        "date",
        "direct_radiation",
        "wind_direction_80m",
        "wind_speed_80m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "real_power",
    ],
    "MOPEX": ["date", "MAP", "CPE", "Tmax", "Tmin", "streamflow"],
}


def _parse_rename(spec: str | None) -> Dict[str, str]:
    """Parse dest=src pairs: MAP=P,streamflow=Q → rename P→MAP, Q→streamflow."""
    if not spec:
        return {}
    mapping: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --rename item '{part}', expected dest=src")
        dest, src = part.split("=", 1)
        mapping[src.strip()] = dest.strip()
    return mapping


def _archive_existing(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = DATA / "_archive_proxy" / f"{out_dir.name}_{stamp}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir), str(archive))
    print(f"[archive] previous {out_dir.name} → {archive}")


def write_split(df: pd.DataFrame, out_dir: Path, splits: Tuple[int, int, int], source: str) -> None:
    train_n, val_n, test_n = splits
    needed = train_n + val_n + test_n
    if len(df) < needed:
        raise ValueError(f"{out_dir.name}: need ≥{needed} rows for splits {splits}, got {len(df)}")
    train = df.iloc[: train_n + val_n].copy()
    test = df.iloc[train_n + val_n : needed].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[:needed].to_csv(out_dir / f"{out_dir.name}.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    meta = {
        "source": source,
        "rows_full": int(len(df)),
        "rows_used": needed,
        "rows_train_plus_val": int(len(train)),
        "rows_test": int(len(test)),
        "splits": {"train": train_n, "val": val_n, "test": test_n},
        "columns": list(df.columns),
        "start": str(df["date"].iloc[0]),
        "end": str(df["date"].iloc[needed - 1]),
        "aligned": True,
        "note": "Imported via scripts/import_aligned_dataset.py (Table 6 prefix split).",
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] {out_dir.name}: train={len(train)} test={len(test)} cols={list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(SPLITS.keys()))
    parser.add_argument("--csv", required=True, type=Path, help="Full series CSV (date + features + target last)")
    parser.add_argument(
        "--rename",
        default=None,
        help="Optional dest=src pairs, comma-separated (e.g. real_power=target,MAP=P)",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Overwrite data/<dataset> without moving the previous folder aside",
    )
    parser.add_argument(
        "--keep-extra-columns",
        action="store_true",
        help="Do not drop columns outside the preferred schema (still requires date + target)",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df.columns = [str(c).strip() for c in df.columns]
    rename = _parse_rename(args.rename)
    if rename:
        missing = [s for s in rename if s not in df.columns]
        if missing:
            raise SystemExit(f"--rename sources missing from CSV: {missing}; have {list(df.columns)}")
        df = df.rename(columns=rename)

    if "date" not in df.columns:
        # common contest aliases
        for alt in ("time", "Time", "datetime", "Date", "timestamp"):
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    if "date" not in df.columns:
        raise SystemExit(f"Need a date/time column; got {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    preferred = PREFERRED_COLUMNS[args.dataset]
    target_name = preferred[-1]
    if target_name not in df.columns:
        # allow last column as target if user already ordered it
        raise SystemExit(
            f"Target column '{target_name}' missing after rename. "
            f"Columns={list(df.columns)}. Use --rename {target_name}=<src>."
        )

    if not args.keep_extra_columns:
        missing_feats = [c for c in preferred if c not in df.columns]
        if missing_feats:
            print(f"[warn] preferred columns missing (kept what exists): {missing_feats}")
        ordered = [c for c in preferred if c in df.columns]
        # ensure target is last
        ordered = [c for c in ordered if c != target_name] + [target_name]
        df = df[ordered].copy()
    else:
        cols = [c for c in df.columns if c != target_name]
        if "date" in cols:
            cols = ["date"] + [c for c in cols if c != "date"]
        df = df[cols + [target_name]].copy()

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        print(f"[info] dropped {before - len(df)} rows with NaNs")

    out_dir = DATA / args.dataset
    if not args.no_archive:
        _archive_existing(out_dir)
    elif out_dir.exists():
        shutil.rmtree(out_dir)

    write_split(df, out_dir, SPLITS[args.dataset], source=str(args.csv.resolve()))


if __name__ == "__main__":
    main()
