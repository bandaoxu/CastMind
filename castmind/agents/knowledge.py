"""Load Knowledge Base (K) and Contextual Repository (E) text stores.

Paper §3.3.3 / §3.3.4: K and E are separate inputs to Generator Eq. (8).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
_KB_DIR = _ROOT / "prompts" / "knowledge_base"
_CTX_DIR = _ROOT / "prompts" / "contextual_briefings"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def load_knowledge(dataset_name: str, knowledge_prompt_file: Optional[str] = None) -> str:
    """Domain knowledge K: shared conceptual + optional dataset empirical file."""
    parts = []
    shared = _KB_DIR / "_shared.txt"
    if shared.exists():
        parts.append(_read_text(shared))
    if knowledge_prompt_file:
        p = Path(os.path.expandvars(knowledge_prompt_file))
        if p.exists():
            parts.append(_read_text(p))
    else:
        ds_path = _KB_DIR / f"{dataset_name}.txt"
        if ds_path.exists():
            parts.append(_read_text(ds_path))
    return "\n\n".join(p for p in parts if p)


def load_context(dataset_name: str, context_prompt_file: Optional[str] = None) -> str:
    """Contextual repository E: time/domain situation text for the dataset."""
    if context_prompt_file:
        p = Path(os.path.expandvars(context_prompt_file))
        if p.exists():
            return _read_text(p)
    ds_path = _CTX_DIR / f"{dataset_name}.txt"
    if ds_path.exists():
        return _read_text(ds_path)
    return ""


def build_knowledge_lookup(dataset_names: list[str]) -> Dict[str, str]:
    return {name: load_knowledge(name) for name in dataset_names}


def build_context_lookup(
    datasets: list,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ds in datasets:
        name = getattr(ds, "name", str(ds))
        ctx_file = getattr(ds, "context_prompt_file", None)
        out[name] = load_context(name, ctx_file)
    return out
