#!/usr/bin/env python3
"""Smoke-test feature-vector case library (incremental, parallel to raw neighbor).

Usage (from repo root):
  .venv/bin/python scripts/smoke_feature_case_library.py
  .venv/bin/python scripts/smoke_feature_case_library.py --dataset ETTh1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from castmind.agents.common import prepare_investor_packet
from castmind.agents.knowledge import build_context_lookup, build_knowledge_lookup
from castmind.config import load_config
from castmind.data_loader import TIME_COL
from castmind.tools.analysis import FEATURE_VECTOR_KEYS, analyze_training


REQUIRED_CASE_KEYS = (
    "look_back_window",
    "pred_window",
    "feature_vector_raw",
    "feature_vector_norm",
    "best_model",
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--windows", type=int, default=3, help="Train windows to analyze")
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    # Smoke the innovation path (Full defaults this off; CLI --ablation feature_case turns it on).
    cfg.use_feature_case_library = True
    ds = next((d for d in cfg.datasets if d.name == args.dataset), None)
    if ds is None:
        print(f"[error] dataset {args.dataset} not in config")
        return 1

    L = int(ds.look_back)
    H = int(ds.predicted_window)
    stride = int(ds.sliding_window) if ds.sliding_window else H
    n_win = max(1, int(args.windows))
    need = L + H + (n_win - 1) * stride

    train_df = pd.read_csv(ds.training_csv)
    train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL])
    train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)
    if len(train_df) < need:
        print(f"[error] train rows {len(train_df)} < need {need}")
        return 1
    train_df = train_df.iloc[:need].copy()

    smoke_root = ROOT / "outputs" / "_smoke" / "feature_case"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(smoke_root)

    print(f"[info] analyze_training on {len(train_df)} rows -> {smoke_root}/{ds.name}")
    analyze_training(
        train_df,
        L,
        H,
        cfg.output_dir,
        ds.name,
        sliding_window=stride,
        method="weighted",
        num_clusters=min(3, max(1, n_win)),
        dataset_cfg=ds,
    )

    ds_out = smoke_root / ds.name
    feat_path = ds_out / "case_feature_neighbor.json"
    scaler_path = ds_out / "case_feature_scaler.json"
    _assert(feat_path.exists(), f"missing {feat_path}")
    _assert(scaler_path.exists(), f"missing {scaler_path}")

    cases = json.loads(feat_path.read_text(encoding="utf-8"))
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    _assert(isinstance(cases, list) and len(cases) >= 1, "case_feature_neighbor empty")
    _assert(isinstance(scaler, dict), "scaler not a dict")
    _assert(scaler.get("feature_keys") == list(FEATURE_VECTOR_KEYS), "feature_keys mismatch")
    _assert(len(scaler.get("mean") or []) == len(FEATURE_VECTOR_KEYS), "mean length")
    _assert(len(scaler.get("std") or []) == len(FEATURE_VECTOR_KEYS), "std length")

    sample = cases[0]
    for key in REQUIRED_CASE_KEYS:
        _assert(key in sample, f"case missing key {key}")
    _assert(len(sample["look_back_window"]) == L, "look_back length")
    _assert(len(sample["pred_window"]) == H, "pred length")
    _assert(len(sample["feature_vector_norm"]) == len(FEATURE_VECTOR_KEYS), "norm length")
    _assert(set(sample["feature_vector_raw"].keys()) == set(FEATURE_VECTOR_KEYS), "raw keys")
    print(f"[ok] feature case files: n={len(cases)} keys={len(FEATURE_VECTOR_KEYS)}")

    briefing = ""
    if ds.context_prompt_file and os.path.exists(ds.context_prompt_file):
        briefing = Path(ds.context_prompt_file).read_text(encoding="utf-8")
    briefings = {ds.name: briefing} if briefing else build_context_lookup([ds])
    knowledge = build_knowledge_lookup([ds.name])

    packet = prepare_investor_packet(
        cfg,
        ds,
        briefings,
        0,
        H,
        knowledge_lookup=knowledge,
        include_case_evidence=True,
    )
    _assert(packet.get("include_case_evidence") is True, "include_case_evidence")
    _assert("reference_prediction" in packet, "missing reference_prediction")
    _assert("neighbor_lookback" in packet and "neighbor_pred" in packet, "missing raw neighbor")
    _assert("X_auxiliary" in packet and "X_neighbor" in packet, "missing X_* aliases")
    _assert(packet.get("feature_neighbor_lookback") is not None, "feature_neighbor_lookback")
    _assert(packet.get("feature_neighbor_pred") is not None, "feature_neighbor_pred")
    _assert(packet.get("feature_neighbor_distance") is not None, "feature_neighbor_distance")
    _assert(isinstance(packet.get("feature_neighbor_model_weights"), dict), "model_weights")
    _assert(packet.get("case_feature_neighbor_size") == len(cases), "case_feature_neighbor_size")
    _assert("X_auxiliary_feature" in packet and "X_neighbor_feature" in packet, "X_*_feature")
    ref_f = packet.get("reference_prediction_feature")
    if ref_f is not None:
        _assert(len(ref_f) == H, "reference_prediction_feature length")
    print("[ok] prepare_investor_packet injects old + feature fields")

    # Compatibility: missing feature JSON must not break old case evidence.
    feat_path.unlink()
    scaler_path.unlink()
    packet2 = prepare_investor_packet(
        cfg,
        ds,
        briefings,
        0,
        H,
        knowledge_lookup=knowledge,
        include_case_evidence=True,
    )
    _assert(packet2.get("include_case_evidence") is True, "compat include_case_evidence")
    _assert(packet2.get("neighbor_lookback") is not None or packet2.get("reference_prediction") is not None,
            "compat: old case evidence missing")
    _assert(packet2.get("feature_neighbor_lookback") is None, "compat: feature lookback should be None")
    _assert(packet2.get("reference_prediction_feature") is None, "compat: feature aux should be None")
    print("[ok] missing feature JSON: old case path still works")

    # use_case_library=false gates both old and new case evidence.
    cfg.use_case_library = False
    # Restore feature files so we can prove the gate blocks them too.
    feat_path.write_text(json.dumps(cases), encoding="utf-8")
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    packet3 = prepare_investor_packet(
        cfg,
        ds,
        briefings,
        0,
        H,
        knowledge_lookup=knowledge,
        include_case_evidence=True,
    )
    _assert(packet3.get("include_case_evidence") is False, "no_case gate")
    _assert(packet3.get("reference_prediction") is None, "no_case: reference_prediction")
    _assert(packet3.get("neighbor_lookback") is None, "no_case: neighbor")
    _assert(packet3.get("feature_neighbor_lookback") is None, "no_case: feature neighbor")
    _assert(packet3.get("reference_prediction_feature") is None, "no_case: feature aux")
    print("[ok] use_case_library=false blocks old + feature case injection")

    summary = {
        "dataset": ds.name,
        "n_feature_cases": len(cases),
        "look_back": L,
        "predicted_window": H,
        "feature_neighbor_distance": packet.get("feature_neighbor_distance"),
        "feature_neighbor_model_weights": packet.get("feature_neighbor_model_weights"),
        "has_reference_prediction_feature": ref_f is not None,
        "reference_prediction_feature_len": len(ref_f) if isinstance(ref_f, list) else None,
    }
    out_path = smoke_root / "smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] wrote {out_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[error] {exc}")
        raise SystemExit(2)
