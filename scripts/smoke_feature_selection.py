#!/usr/bin/env python3
"""Smoke-test Investigator F_selected (paper Eq. 6 approx) on ETTh1 look-back window.

Usage (from repo root):
  .venv/bin/python scripts/smoke_feature_selection.py
  .venv/bin/python scripts/smoke_feature_selection.py --mode paper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from castmind.config import load_config
from castmind.agents.common import prepare_investor_packet
from castmind.agents.knowledge import build_context_lookup, build_knowledge_lookup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "rules", "paper"), default="rules")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--window-offset", type=int, default=0)
    parser.add_argument(
        "--feedback",
        default="",
        help="Optional Reflector-style feedback to test re-select nudges",
    )
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.feature_selection = args.mode
    ds = next((d for d in cfg.datasets if d.name == args.dataset), None)
    if ds is None:
        print(f"[error] dataset {args.dataset} not in config")
        return 1

    briefing = ""
    if ds.context_prompt_file and os.path.exists(ds.context_prompt_file):
        briefing = Path(ds.context_prompt_file).read_text(encoding="utf-8")
    briefings = {ds.name: briefing} if briefing else build_context_lookup([ds])
    knowledge = build_knowledge_lookup([ds.name])

    # Ensure case library exists for packet assembly.
    case_path = ROOT / cfg.output_dir / ds.name / "case_base.json"
    if not case_path.exists():
        print(f"[warn] Missing {case_path}; prepare_investor_packet may rebuild / fail on models.")

    feedback = args.feedback.strip() or None
    packet = prepare_investor_packet(
        cfg,
        ds,
        briefings,
        args.window_offset,
        int(ds.predicted_window),
        reflective_feedback=feedback,
        knowledge_lookup=knowledge,
        include_case_evidence=False,
    )
    out = {
        "feature_selection_mode": packet.get("feature_selection_mode"),
        "selection_method": packet.get("selection_method"),
        "selection_rationale": packet.get("selection_rationale"),
        "selected_features": packet.get("selected_features"),
        "feature_weights": packet.get("feature_weights"),
        "n_features_full": len(packet.get("features_full") or {}),
        "features_selected_values": packet.get("features_selected_values"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    smoke_dir = ROOT / cfg.output_dir / "_smoke" / "feature_selection"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    path = smoke_dir / f"{args.dataset}_{args.mode}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] wrote {path}")
    if args.mode != "off" and not out.get("selected_features"):
        print("[error] expected non-empty selected_features")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
