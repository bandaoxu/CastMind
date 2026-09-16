#!/usr/bin/env python3
"""Build group-talk PPT from the official visual template.

Preferred path: clone docs/AlphaCast论文阅读与复现进展汇报.pptx and rewrite copy.
Fallback docs: see docs/alphacast_talk.md (Marp) if template is missing.
"""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    template = ROOT / "docs" / "AlphaCast论文阅读与复现进展汇报.pptx"
    builder = ROOT / "scripts" / "build_group_talk_from_template.py"
    if not template.is_file():
        raise SystemExit(
            f"[error] template missing: {template}\n"
            "Place the styled PPT template under docs/, then re-run."
        )
    runpy.run_path(str(builder), run_name="__main__")


if __name__ == "__main__":
    main()
