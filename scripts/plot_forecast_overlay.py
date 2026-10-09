#!/usr/bin/env python3
"""Plot Ground Truth / CastMind / Base Model forecast overlays.

Default panels: ETTh1, ETTm1, EPF_NP, EPF_PJM (real datasets with ready Full runs).
Excludes long-horizon proxy sets (MOPEX / windy_power / sunny_power).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FONT = (
    "PingFang SC"
    if "PingFang SC" in {f.name for f in fm.fontManager.ttflist}
    else "Heiti TC"
)
plt.rcParams["font.sans-serif"] = [FONT, "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GT_COLOR = "#1f77b4"
AGENT_COLOR = "#ff7f0e"
BASE_COLOR = "#2ca02c"

DEFAULT_PANELS: List[Dict[str, str]] = [
    {
        "dataset": "ETTh1",
        "title": "ETTh1",
        "run_dir": "outputs/ETTh1/runs/Full_deepseek_1",
        "test_csv": "data/ETTh1/test.csv",
    },
    {
        "dataset": "ETTm1",
        "title": "ETTm1",
        "run_dir": "outputs/ETTm1/runs/Full_1",
        "test_csv": "data/ETTm1/test.csv",
    },
    {
        "dataset": "EPF_NP",
        "title": "EPF_NP",
        "run_dir": "outputs/EPF_NP/runs/Full_2",
        "test_csv": "data/EPF_NP/test.csv",
    },
    {
        "dataset": "EPF_PJM",
        "title": "EPF_PJM",
        "run_dir": "outputs/EPF_PJM/runs/Full_1",
        "test_csv": "data/EPF_PJM/test.csv",
    },
]


def _infer_target_column(df: pd.DataFrame) -> str:
    exclude = {"date", "time_stamp", "predicted_ans", "features_used"}
    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        raise ValueError("Unable to infer target column")
    return cols[-1]


def _load_aligned(
    *,
    test_csv: Path,
    run_dir: Path,
) -> pd.DataFrame:
    pred = pd.read_csv(run_dir / "predictions.csv", parse_dates=["time_stamp"])
    pred = pred.sort_values(["window_offset", "horizon_index", "time_stamp"]).reset_index(drop=True)

    test = pd.read_csv(test_csv, parse_dates=["date"])
    target = _infer_target_column(test)
    gt = test[["date", target]].rename(columns={"date": "time_stamp", target: "ground_truth"})

    merged = pred.merge(gt, on="time_stamp", how="inner")
    if merged.empty:
        raise RuntimeError(f"No GT∩pred overlap for {run_dir}")

    basemodel_path = run_dir / "basemodel_results.json"
    if not basemodel_path.is_file():
        raise FileNotFoundError(basemodel_path)
    basemodel = json.loads(basemodel_path.read_text(encoding="utf-8"))
    if not isinstance(basemodel, list):
        raise ValueError(f"Unexpected basemodel_results.json shape in {basemodel_path}")

    base_rows: List[Dict[str, object]] = []
    for entry in basemodel:
        start = entry.get("start_timestamp")
        ref = entry.get("reference_prediction")
        if not start or not isinstance(ref, list) or not ref:
            continue
        t0 = pd.Timestamp(start)
        win = pred.loc[pred["time_stamp"] == t0]
        if win.empty:
            continue
        woff = int(win.iloc[0]["window_offset"])
        wpred = pred.loc[pred["window_offset"] == woff].sort_values("horizon_index")
        n = min(len(wpred), len(ref))
        for i in range(n):
            base_rows.append(
                {
                    "time_stamp": wpred.iloc[i]["time_stamp"],
                    "base_prediction": float(ref[i]),
                }
            )

    if not base_rows:
        raise RuntimeError(f"No stamped reference_prediction aligned for {run_dir}")

    candidates = pd.DataFrame(base_rows)
    if (candidates.groupby("time_stamp")["base_prediction"].nunique() > 1).any():
        raise ValueError("Conflicting auxiliary predictions: establish run provenance before plotting")

    base_df = (
        pd.DataFrame(base_rows)
        .drop_duplicates(subset=["time_stamp"], keep="last")
        .sort_values("time_stamp")
    )
    out = merged.merge(base_df, on="time_stamp", how="inner")
    if out.empty:
        raise RuntimeError(f"No GT∩pred∩base overlap for {run_dir}")
    return out.sort_values("time_stamp").reset_index(drop=True)


def _pick_segment(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    # Prefer an early contiguous block with non-flat GT (paper-like visual).
    gt = df["ground_truth"].to_numpy(dtype=float)
    best_i = 0
    best_score = -1.0
    step = max(1, max_points // 4)
    for i in range(0, len(df) - max_points + 1, step):
        seg = gt[i : i + max_points]
        score = float(np.nanstd(seg))
        if score > best_score:
            best_score = score
            best_i = i
    return df.iloc[best_i : best_i + max_points].reset_index(drop=True)


def plot_panels(
    panels: Sequence[Dict[str, str]],
    *,
    max_points: int,
    out_path: Path,
    subtitle: str,
) -> Path:
    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    fig.suptitle(
        "CastMind vs Base Model (local Full runs)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        run_dir = ROOT / panel["run_dir"]
        test_csv = ROOT / panel["test_csv"]
        aligned = _load_aligned(test_csv=test_csv, run_dir=run_dir)
        seg = _pick_segment(aligned, max_points)
        x = np.arange(len(seg))
        ax.plot(x, seg["ground_truth"], color=GT_COLOR, lw=1.6, label="Ground Truth")
        ax.plot(x, seg["prediction"], color=AGENT_COLOR, lw=1.5, label="CastMind")
        ax.plot(x, seg["base_prediction"], color=BASE_COLOR, lw=1.4, label="Base Model")
        ax.set_title(panel.get("title") or panel["dataset"], fontsize=11)
        ax.set_xlabel("Time step")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="best", fontsize=8, frameon=False)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="Points per panel (paper Fig.7-like window).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/comparison/forecast_overlay_real_datasets.png",
    )
    args = parser.parse_args()

    subtitle = (
        "Local CastMind/DeepSeek Full runs; Base = stamped reference_prediction. "
        "Excludes proxy sets (MOPEX/windy/sunny). "
        "EPF_DE/BE/FR omitted until Train-only Full runs exist."
    )
    out = plot_panels(
        DEFAULT_PANELS,
        max_points=args.max_points,
        out_path=args.out if args.out.is_absolute() else ROOT / args.out,
        subtitle=subtitle,
    )
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
