#!/usr/bin/env python3
"""Assemble a Table-1-style markdown/CSV from paper refs + local baseline/CastMind evals.

Default: paper-rows-only — main table matches AlphaCast Table 1 method names.
System pool (`get_default_models`) is also the Table 1 baseline set.
Numeric gaps vs paper are expected without author DL checkpoints / GPT-5.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Paper Table 1 display order (AlphaCast first).
PAPER_ROW_ORDER = [
    "AlphaCast",
    "Sundial",
    "Chronos",
    "DLinear",
    "PatchTST",
    "TimesNet",
    "TimeXer",
    "Transformer",
    "Autoformer",
    "Prophet",
    "SNaive",
    "ARIMA",
    "CES",
    "CrostonClassic",
    "Optimizers",
    "HistoricAverage",
]

# Paper display name -> local get_default_models() alias
# AlphaCast local cell = outputs/<ds>/predictions.csv (CastMind_current).
PAPER_TO_LOCAL = {
    "AlphaCast": "CastMind_current",
    "Sundial": "Sundial",
    "Chronos": "Chronos",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
    "TimeXer": "TimeXer",
    "Transformer": "iTransformer",
    "Autoformer": "Autoformer",
    "Prophet": "Prophet",
    "SNaive": "SeasonalNaive",
    "ARIMA": "AutoARIMA",
    "CES": "AutoCES",
    "CrostonClassic": "CrostonClassic",
    "Optimizers": "DynamicOptimizedTheta",
    "HistoricAverage": "HistoricAverage",
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(mse: Optional[float], mae: Optional[float]) -> str:
    if mse is None or mae is None:
        return "—"
    try:
        if not (mse == mse and mae == mae):
            return "—"
        return f"{float(mse):.3f} / {float(mae):.3f}"
    except (TypeError, ValueError):
        return "—"


def _collect_local(dataset: str, comparison_dir: Path) -> Dict[str, Dict[str, Any]]:
    local_by_alias: Dict[str, Dict[str, Any]] = {}
    base_json = comparison_dir / f"{dataset}_baselines.json"
    if base_json.is_file():
        payload = _load_json(base_json)
        for r in payload.get("results") or []:
            local_by_alias[str(r["model"])] = r
    for path in sorted(comparison_dir.glob(f"{dataset}_castmind_*.json")):
        row = _load_json(path)
        local_by_alias[str(row["model"])] = row
    return local_by_alias


def assemble_dataset(
    dataset: str,
    comparison_dir: Path,
    paper_ref: Dict[str, Any],
    paper_rows_only: bool,
) -> List[Dict[str, Any]]:
    paper_ds = (paper_ref.get("datasets") or {}).get(dataset) or {}
    # Support both paper display keys and local alias keys in the JSON.
    raw_methods: Dict[str, Dict[str, float]] = dict(paper_ds.get("methods") or {})
    aliases = dict(paper_ds.get("paper_name_aliases") or {})
    # Invert aliases: local -> paper display if needed
    paper_by_display: Dict[str, Dict[str, float]] = {}
    for k, v in raw_methods.items():
        display = k
        # If key is a local alias that maps from a paper name, keep as-is when already display
        paper_by_display[display] = v

    # Normalize: if JSON uses SeasonalNaive etc., map to paper display for lookup
    local_to_paper = {v: k for k, v in PAPER_TO_LOCAL.items() if v}
    for k, v in list(paper_by_display.items()):
        if k in local_to_paper:
            paper_by_display[local_to_paper[k]] = v

    local_by_alias = _collect_local(dataset, comparison_dir)
    rows_out: List[Dict[str, Any]] = []

    for display in PAPER_ROW_ORDER:
        paper = paper_by_display.get(display) or raw_methods.get(display)
        # Also try local alias key in paper JSON
        local_alias = PAPER_TO_LOCAL.get(display)
        if paper is None and local_alias:
            paper = raw_methods.get(local_alias)

        local = None
        if local_alias:
            local = local_by_alias.get(local_alias)

        rows_out.append(
            {
                "method": display,
                "local_alias": local_alias,
                "paper_MSE": None if not paper else paper.get("MSE"),
                "paper_MAE": None if not paper else paper.get("MAE"),
                "local_MSE": None if not local else local.get("MSE"),
                "local_MAE": None if not local else local.get("MAE"),
                "local_n": None if not local else local.get("n_points"),
                "local_error": None if not local else local.get("error"),
            }
        )

    if not paper_rows_only:
        # Append any other local models not already shown
        shown = {r["local_alias"] for r in rows_out if r.get("local_alias")}
        shown |= set(PAPER_TO_LOCAL.values())
        for alias, local in sorted(local_by_alias.items()):
            if alias in shown or alias.startswith("CastMind_"):
                continue
            rows_out.append(
                {
                    "method": alias,
                    "local_alias": alias,
                    "paper_MSE": None,
                    "paper_MAE": None,
                    "local_MSE": local.get("MSE"),
                    "local_MAE": local.get("MAE"),
                    "local_n": local.get("n_points"),
                    "local_error": local.get("error"),
                }
            )

    return rows_out


def write_markdown(
    datasets: List[str],
    assembled: Dict[str, List[Dict[str, Any]]],
    out_md: Path,
    paper_ref: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# CastMind 本机 vs 论文 Table 1 风格对比")
    lines.append("")
    lines.append(
        "> **同构进度：** 季节按 frequency（如 ETTh `h→24`）；统计 AutoARIMA=StatsForecast；"
        "DL 用 `DeepLearningCheckpoints/<ds>/` light 五模型（非作者权重；本轮无 TimeXer）；"
        "本机 **AlphaCast** 行 = `outputs/<ds>/predictions.csv`（CastMind_current / DeepSeek）。"
        "**不宣称逐格复现：** DeepSeek≠GPT-5；light≠作者 `.pth`；"
        "EPF 短序覆盖不全；WP/SP/MOPEX 为代理数据。"
        " **EPF 数据：** 2026-09-13 已修 Price/外生列对调；基线应与论文同量级。"
    )
    lines.append(">")
    lines.append(
        f"> 论文参考：{paper_ref.get('source', 'AlphaCast Table 1')}。"
        " 单元格：`MSE / MAE`（越低越好）。"
        " Transformer = 本机 iTransformer；Optimizers = DynamicOptimizedTheta。"
    )
    lines.append("")

    for ds in datasets:
        rows = assembled[ds]
        paper_key = ((paper_ref.get("datasets") or {}).get(ds) or {}).get("paper_key", ds)
        lines.append(f"## {ds}（论文列：{paper_key}）")
        lines.append("")
        lines.append("| 方法 | 论文 MSE/MAE | 本机 MSE/MAE | 本机点数 |")
        lines.append("|------|-------------:|-------------:|---------:|")

        full_n = max((int(r["local_n"]) for r in rows if r.get("local_n")), default=0)
        local_scores = []
        for r in rows:
            mse_v = r.get("local_MSE")
            n = r.get("local_n")
            name = str(r["method"])
            if mse_v is None or mse_v != mse_v:
                continue
            if "partial" in name.lower():
                continue
            if full_n and n is not None and int(n) < full_n:
                continue
            local_scores.append((name, float(mse_v)))
        best_local = min(local_scores, key=lambda x: x[1])[0] if local_scores else None

        for r in rows:
            paper_cell = _fmt(r.get("paper_MSE"), r.get("paper_MAE"))
            local_cell = _fmt(r.get("local_MSE"), r.get("local_MAE"))
            if r["method"] == best_local and local_cell != "—":
                local_cell = f"**{local_cell}**"
            n = r.get("local_n")
            n_s = "—" if n is None else str(n)
            if r.get("local_error"):
                n_s = "err"
            lines.append(f"| {r['method']} | {paper_cell} | {local_cell} | {n_s} |")
        lines.append("")

    lines.append("## 口径说明")
    lines.append("")
    lines.append("- **方法集 / 主表行 / 系统池**：对齐论文 Table 1。")
    lines.append(
        "- **可对齐：** L/H=96、原始 OT、`season_length` 按 frequency（小时=24）、"
        "StatsForecast 统计栈（含 AutoARIMA）。"
    )
    lines.append(
        "- **近似对齐：** DL 为 TSLib 风格自训 `*_official`（无作者官方权重时的上限）。"
    )
    lines.append(
        "- **不可对齐：** 作者主表 `.pth`；本机 AlphaCast 为 DeepSeek 系统行，不宣称与 GPT-5 逐格一致。"
    )
    lines.append(
        "- **AlphaCast（本机）**：仅用当前 `outputs/<ds>/`（与 `_archive/<ds>_llm_deepseekchat` 同步）；"
        "不另列历史 `*_llm_deepseek` 行。"
    )
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"])
    parser.add_argument("--comparison-dir", default="outputs/comparison")
    parser.add_argument("--paper-ref", default="scripts/paper_table1_reference.json")
    parser.add_argument("--out-md", default="docs/comparison_table.md")
    parser.add_argument("--out-copy-md", default="outputs/comparison/table1_local.md")
    parser.add_argument("--out-csv", default="outputs/comparison/table1_local.csv")
    parser.add_argument(
        "--all-local-rows",
        action="store_true",
        help="Also append leftover local aliases if present in baselines JSON (legacy).",
    )
    args = parser.parse_args()

    paper_ref = _load_json(Path(args.paper_ref))
    comparison_dir = Path(args.comparison_dir)
    assembled: Dict[str, List[Dict[str, Any]]] = {}
    flat: List[Dict[str, Any]] = []
    paper_rows_only = not args.all_local_rows

    for ds in args.datasets:
        rows = assemble_dataset(ds, comparison_dir, paper_ref, paper_rows_only)
        assembled[ds] = rows
        for r in rows:
            flat.append({"dataset": ds, **r})

    write_markdown(args.datasets, assembled, Path(args.out_md), paper_ref)
    write_markdown(args.datasets, assembled, Path(args.out_copy_md), paper_ref)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "local_alias",
                "paper_MSE",
                "paper_MAE",
                "local_MSE",
                "local_MAE",
                "local_n",
                "local_error",
            ],
        )
        writer.writeheader()
        writer.writerows(flat)

    print(f"[info] wrote {args.out_md}")
    print(f"[info] wrote {args.out_copy_md}")
    print(f"[info] wrote {out_csv}")


if __name__ == "__main__":
    main()
