#!/usr/bin/env python3
"""Evaluate each baseline independently on the test set (Table-1 style rows).

Uses the same look_back / predicted_window / sliding_window protocol as
run_experiment.py LLM/deterministic coverage of test_df.iloc[look_back:].
Does NOT build a case library or call the LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from castmind.config import DatasetConfig, load_config
from castmind.data_loader import TIME_COL, infer_target_column
from castmind.eval import align_predictions, mae, mse, smape
from castmind.models.base import (
    configure_deep_learning_runtime,
    get_default_models,
)
from castmind.utils.time import resolve_season_length


def _eval_one_model(
    model,
    test_df: pd.DataFrame,
    target_col: str,
    look_back: int,
    horizon: int,
    stride: int,
    season_length: int,
) -> Dict[str, Any]:
    y = test_df[target_col].to_numpy(dtype=float)
    ts = pd.to_datetime(test_df[TIME_COL])
    total_len = max(0, len(test_df) - look_back)
    if total_len == 0:
        return {
            "model": model.alias,
            "MSE": float("nan"),
            "MAE": float("nan"),
            "sMAPE": float("nan"),
            "n_points": 0,
            "error": "empty test horizon",
        }

    rows: List[Dict[str, Any]] = []
    current_len = 0
    current_collected = 0
    while current_collected < total_len:
        step_horizon = min(horizon, total_len - current_collected)
        offset = current_len
        lookback_end = offset + look_back
        if lookback_end > len(y):
            break
        window = y[offset:lookback_end]
        window_ts = ts.iloc[offset:lookback_end]
        fut_start = lookback_end
        fut_end = lookback_end + step_horizon
        if fut_end > len(y):
            step_horizon = max(0, len(y) - fut_start)
            fut_end = fut_start + step_horizon
        if step_horizon <= 0:
            break
        fut_ts = ts.iloc[fut_start:fut_end]
        try:
            model.fit(window, season_length=season_length, timestamps=window_ts)
            pred = np.asarray(
                model.predict(step_horizon, future_timestamps=fut_ts),
                dtype=float,
            ).reshape(-1)
            if len(pred) != step_horizon:
                pred = pred[:step_horizon]
            for i in range(len(pred)):
                rows.append(
                    {
                        "time_stamp": fut_ts.iloc[i],
                        "predicted_ans": float(pred[i]),
                        "window_offset": offset,
                        "horizon_index": i,
                    }
                )
        except Exception as exc:
            return {
                "model": model.alias,
                "MSE": float("nan"),
                "MAE": float("nan"),
                "sMAPE": float("nan"),
                "n_points": 0,
                "error": str(exc),
            }

        current_collected += step_horizon
        current_len += stride
        if current_len + look_back > len(y) and current_collected < total_len:
            # Match run_experiment: keep advancing until budget filled when possible
            if stride <= 0:
                break

    if not rows:
        return {
            "model": model.alias,
            "MSE": float("nan"),
            "MAE": float("nan"),
            "sMAPE": float("nan"),
            "n_points": 0,
            "error": "no predictions",
        }

    pred_df = pd.DataFrame(rows)
    y_true, y_pred = align_predictions(test_df, pred_df, None)
    if len(y_true) == 0:
        return {
            "model": model.alias,
            "MSE": float("nan"),
            "MAE": float("nan"),
            "sMAPE": float("nan"),
            "n_points": 0,
            "error": "alignment produced empty arrays",
        }
    return {
        "model": model.alias,
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "n_points": int(len(y_true)),
        "error": None,
    }


def eval_dataset(
    ds: DatasetConfig,
    model_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    train_df = pd.read_csv(ds.training_csv)
    test_df = pd.read_csv(ds.test_csv)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
    test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)
    target_col = infer_target_column(test_df, ds.name)
    train_target = pd.to_numeric(
        train_df[infer_target_column(train_df, ds.name)], errors="coerce"
    ).dropna().to_numpy(dtype=float)
    season_length = max(
        1,
        int(resolve_season_length(train_target, frequency=getattr(ds, "frequency", None))),
    )

    configure_deep_learning_runtime(ds.checkpoints, ds.predicted_window)
    models = get_default_models()
    if model_filter:
        wanted = {m.lower() for m in model_filter}
        models = [m for m in models if m.alias.lower() in wanted]

    look_back = int(ds.look_back)
    horizon = int(ds.predicted_window)
    stride = int(ds.sliding_window) if ds.sliding_window and ds.sliding_window > 0 else horizon

    print(
        f"\n=== {ds.name} L={look_back} H={horizon} stride={stride} "
        f"models={len(models)} season={season_length} ==="
    )
    results: List[Dict[str, Any]] = []
    for model in models:
        print(f"  evaluating {model.alias} ...", flush=True)
        row = _eval_one_model(
            model,
            test_df,
            target_col,
            look_back,
            horizon,
            stride,
            season_length,
        )
        if row.get("error"):
            print(f"    [warn] {model.alias}: {row['error']}")
        else:
            print(
                f"    MSE={row['MSE']:.4f} MAE={row['MAE']:.4f} "
                f"n={row['n_points']}"
            )
        results.append(row)

    configure_deep_learning_runtime(None, None)
    return {
        "dataset": ds.name,
        "look_back": look_back,
        "predicted_window": horizon,
        "sliding_window": stride,
        "season_length": season_length,
        "pool": [m.alias for m in models],
        "results": results,
    }


def metrics_from_predictions(
    dataset_name: str,
    test_csv: str,
    predictions_csv: Path,
    label: str,
) -> Dict[str, Any]:
    test_df = pd.read_csv(test_csv)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
    pred_df = pd.read_csv(predictions_csv)
    y_true, y_pred = align_predictions(test_df, pred_df, dataset_name)
    return {
        "model": label,
        "MSE": mse(y_true, y_pred) if len(y_true) else float("nan"),
        "MAE": mae(y_true, y_pred) if len(y_true) else float("nan"),
        "sMAPE": smape(y_true, y_pred) if len(y_true) else float("nan"),
        "n_points": int(len(y_true)),
        "source": str(predictions_csv),
        "error": None if len(y_true) else "empty alignment",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"])
    parser.add_argument("--models", nargs="+", default=None, help="Subset of aliases")
    parser.add_argument(
        "--output-dir",
        default="outputs/comparison",
        help="Directory for JSON/CSV fragments",
    )
    parser.add_argument(
        "--from-archive",
        nargs="*",
        default=None,
        help="Optional CastMind prediction CSVs: label=path or bare path",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    wanted = {n.lower() for n in args.datasets}
    datasets = [d for d in cfg.datasets if d.name.lower() in wanted]
    if not datasets:
        raise SystemExit(f"No datasets matched {args.datasets}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        payload = eval_dataset(ds, model_filter=args.models)
        json_path = out_dir / f"{ds.name}_baselines.json"
        if args.models and json_path.is_file():
            try:
                prev = json.loads(json_path.read_text(encoding="utf-8"))
                by_name = {str(r["model"]): r for r in (prev.get("results") or [])}
                for r in payload["results"]:
                    by_name[str(r["model"])] = r
                payload["results"] = list(by_name.values())
                payload["pool"] = sorted(by_name.keys())
                if prev.get("season_length") is not None:
                    payload["season_length"] = prev.get("season_length") or payload.get("season_length")
                # Prefer latest season from this run when full semantics matter
                payload["season_length"] = max(
                    1, int(payload.get("season_length") or prev.get("season_length") or 1)
                )
            except Exception as exc:
                print(f"[warn] could not merge prior baselines JSON: {exc}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[info] wrote {json_path}")

        for r in payload["results"]:
            all_rows.append(
                {
                    "dataset": ds.name,
                    "model": r["model"],
                    "MSE": r["MSE"],
                    "MAE": r["MAE"],
                    "sMAPE": r.get("sMAPE"),
                    "n_points": r.get("n_points"),
                    "error": r.get("error"),
                    "source": "baseline_eval",
                }
            )

        if args.from_archive is not None:
            # Copy per dataset — mutating args.from_archive would reuse the first
            # dataset's paths for every later dataset (empty-alignment / wrong MSE).
            specs = list(args.from_archive)
            if len(specs) == 0:
                # Auto-discover common archives for this dataset
                archive_root = ROOT / "outputs" / "_archive"
                defaults = [
                    (f"CastMind_llm_deepseek", archive_root / f"{ds.name}_llm_deepseek" / "predictions.csv"),
                    (f"CastMind_current", ROOT / "outputs" / ds.name / "predictions.csv"),
                ]
                for label, path in defaults:
                    if path.is_file():
                        specs.append(f"{label}={path}")

            for spec in specs:
                if "=" in spec:
                    label, path_s = spec.split("=", 1)
                else:
                    path_s = spec
                    label = Path(path_s).parent.name
                path = Path(path_s)
                if not path.is_file():
                    print(f"[warn] skip missing archive predictions: {path}")
                    continue
                row = metrics_from_predictions(ds.name, ds.test_csv, path, label)
                print(
                    f"  archive {label}: MSE={row['MSE']:.4f} MAE={row['MAE']:.4f} "
                    f"n={row['n_points']}"
                )
                cast_path = out_dir / f"{ds.name}_castmind_{label}.json"
                with open(cast_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, indent=2)
                all_rows.append(
                    {
                        "dataset": ds.name,
                        "model": row["model"],
                        "MSE": row["MSE"],
                        "MAE": row["MAE"],
                        "sMAPE": row.get("sMAPE"),
                        "n_points": row.get("n_points"),
                        "error": row.get("error"),
                        "source": row.get("source"),
                    }
                )

    frag = out_dir / "table_fragment.csv"
    pd.DataFrame(all_rows).to_csv(frag, index=False)
    print(f"[info] wrote {frag}")


if __name__ == "__main__":
    main()
