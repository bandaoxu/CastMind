#!/usr/bin/env python3
"""Backup + audit + migrate outputs/_archive into outputs/<ds>/runs/<readable_name>/."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from castmind.run_layout import (  # noqa: E402
    derive_run_name_from_archive_tag,
    runs_root,
    write_run_manifest,
)


def _audit_predictions(pred_path: Path, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(pred_path),
        "exists": pred_path.is_file(),
        "null_ts": 0,
        "n_rows": 0,
        "n_unique_ts": 0,
        "duplicate_ts": 0,
        "metrics_n": None if not metrics else metrics.get("n"),
        "status": "missing",
    }
    if not pred_path.is_file():
        return result
    try:
        df = pd.read_csv(pred_path)
    except Exception as exc:
        result["status"] = f"unreadable:{exc}"
        return result
    result["n_rows"] = int(len(df))
    if "time_stamp" not in df.columns:
        result["status"] = "no_time_stamp"
        return result
    ts = pd.to_datetime(df["time_stamp"], errors="coerce")
    result["null_ts"] = int(ts.isna().sum())
    valid = ts.dropna()
    result["n_unique_ts"] = int(valid.nunique())
    result["duplicate_ts"] = int(len(valid) - valid.nunique())
    metrics_n = result["metrics_n"]
    if result["null_ts"] > 0:
        result["status"] = "damaged_null_ts"
    elif result["duplicate_ts"] > 0:
        result["status"] = "needs_recompute_metrics"
    elif metrics_n is not None and int(metrics_n) != result["n_unique_ts"]:
        result["status"] = "needs_recompute_metrics"
    elif result["n_unique_ts"] == 0:
        result["status"] = "empty"
    else:
        result["status"] = "format_checked"
    return result


def _load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _dataset_from_archive_tag(tag: str, known: List[str]) -> Optional[str]:
    for name in sorted(known, key=len, reverse=True):
        if tag == name or tag.startswith(name + "_"):
            return name
    return None


def migrate(
    output_dir: Path,
    *,
    datasets: Optional[List[str]],
    dry_run: bool,
    copy_case_from_working: bool,
) -> int:
    archive_root = output_dir / "_archive"
    if not archive_root.is_dir():
        print(f"[warn] No archive directory at {archive_root}")
        return 0

    # Discover dataset names from archive tags + working dirs.
    known = set(datasets or [])
    for p in output_dir.iterdir():
        if p.is_dir() and p.name not in {"_archive", "comparison", "_smoke", "diagnostics"}:
            if not p.name.startswith("_"):
                known.add(p.name)
    for p in archive_root.iterdir():
        if p.is_dir():
            ds = _dataset_from_archive_tag(p.name, list(known) or [p.name.split("_")[0]])
            if ds:
                known.add(ds)
            # Heuristic for EPF_* / ETTh1 style
            parts = p.name.split("_")
            if parts[0] == "EPF" and len(parts) >= 2:
                known.add(f"EPF_{parts[1]}")
            elif parts[0] in {"ETTh1", "ETTm1", "MOPEX"}:
                known.add(parts[0])
            elif parts[0] in {"windy", "sunny"} and len(parts) >= 2:
                known.add(f"{parts[0]}_{parts[1]}")

    known_list = sorted(known)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report: List[Dict[str, Any]] = []

    for tag_dir in sorted(archive_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        ds_name = _dataset_from_archive_tag(tag, known_list)
        if datasets and ds_name not in datasets:
            continue
        if ds_name is None:
            report.append({"tag": tag, "status": "unknown_dataset", "action": "skip"})
            print(f"[skip] Cannot map archive tag to dataset: {tag}")
            continue

        metrics = _load_metrics(tag_dir / "metrics.json")
        audit = _audit_predictions(tag_dir / "predictions.csv", metrics)
        run_name = derive_run_name_from_archive_tag(tag, ds_name)
        dest = runs_root(str(output_dir), ds_name) / run_name
        backup = output_dir / ds_name / f"_legacy_backup_{ts}" / "archive" / tag

        action = "skip"
        if audit["status"] == "format_checked":
            action = "migrate"
        elif audit["status"] == "needs_recompute_metrics":
            action = "migrate_recompute"
        elif audit["status"] in {"damaged_null_ts", "empty", "missing", "no_time_stamp"} or str(
            audit["status"]
        ).startswith("unreadable"):
            action = "backup_only"
        else:
            action = "backup_only"

        entry = {
            "tag": tag,
            "dataset": ds_name,
            "run_name": run_name,
            "audit": audit,
            "action": action,
            "dest": str(dest),
            "backup": str(backup),
        }
        report.append(entry)
        print(
            f"[{action}] {tag} -> {ds_name}/runs/{run_name} "
            f"(status={audit['status']} unique_ts={audit['n_unique_ts']} null={audit['null_ts']})"
        )

        if dry_run:
            continue

        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copytree(tag_dir, backup)

        if action.startswith("migrate"):
            if dest.exists():
                print(f"  [warn] destination exists, skipping copy: {dest}")
                entry["action"] = "skipped_exists"
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tag_dir, dest)
            manifest = {
                "dataset": ds_name,
                "run_name": run_name,
                "migrated_from": str(tag_dir),
                "migrated_at": datetime.now(timezone.utc).isoformat(),
                "audit_status": audit["status"],
                "experiment_validity": "unverified",
                "auxiliary_log_provenance": "unverified",
                "read_only_source": True,
                "metrics_n": audit.get("metrics_n"),
                "n_unique_ts": audit.get("n_unique_ts"),
            }
            if action == "migrate_recompute":
                manifest["needs_recompute_metrics"] = True
            if audit["status"] != "format_checked":
                manifest["needs_rerun"] = audit["status"] not in {"needs_recompute_metrics"}
            write_run_manifest(dest, manifest)

            # Optionally copy flat working-dir case artifacts into case_libraries/legacy_from_working
            if copy_case_from_working:
                working = output_dir / ds_name
                case_files = [
                    "memory.json",
                    "case_base.json",
                    "case_neighbor.json",
                    "cluster_base.json",
                    "case_feature_neighbor.json",
                    "case_feature_scaler.json",
                ]
                if any((working / f).is_file() for f in case_files):
                    legacy_lib = output_dir / ds_name / "case_libraries" / "legacy_from_working"
                    legacy_lib.mkdir(parents=True, exist_ok=True)
                    for f in case_files:
                        src = working / f
                        if src.is_file():
                            shutil.copy2(src, legacy_lib / f)

    report_path = output_dir / f"_migrate_report_{ts}.json"
    if not dry_run:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[info] Wrote {report_path}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Limit to dataset(s); repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Audit only; do not copy.")
    parser.add_argument(
        "--copy-case-from-working",
        action="store_true",
        help="Also snapshot flat working-dir case_*.json into case_libraries/legacy_from_working.",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    raise SystemExit(
        migrate(
            out,
            datasets=args.dataset or None,
            dry_run=bool(args.dry_run),
            copy_case_from_working=bool(args.copy_case_from_working),
        )
    )


if __name__ == "__main__":
    main()
