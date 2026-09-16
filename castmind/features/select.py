"""Investigator-side feature selection approximating paper Eq. (6): F_selected = S(F, I_input).

Paper does not specify S(·). Modes:
  - rules: deterministic heuristic (reproducible)
  - paper: LLM selection with rules fallback
  - off: no Investigator selection (legacy)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Preference order for rule fallback / padding (must exist in F when used).
_RULE_PRIORITY = [
    "seasonal_strength",
    "basic_std",
    "x_acf1",
    "seas_acf1",
    "basic_mean",
    "lumpiness",
    "spectral_entropy",
    "entropy",
    "x_acf10",
    "diff1_acf1",
    "basic_skew",
    "basic_kurt",
    "basic_min",
    "basic_max",
    "basic_count",
    "crossing_points",
    "flat_spots",
    "diff1_acf10",
    "diff2_acf1",
    "diff2_acf10",
]


def normalize_feature_selection_mode(raw: Optional[str]) -> str:
    mode = (raw or "off").strip().lower()
    if mode in ("paper", "rules", "off"):
        return mode
    return "off"


def select_features_rules(
    features: Dict[str, Any],
    *,
    min_count: int = 3,
    reflective_feedback: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float], str]:
    """Heuristic S(F): prefer seasonality / volatility / short-lag ACF; pad from priority list."""
    available = [str(k) for k in features.keys()] if isinstance(features, dict) else []
    if not available:
        return [], {}, "no features available"

    chosen: List[str] = []
    feedback_l = (reflective_feedback or "").lower()

    def _add(name: str) -> None:
        if name in available and name not in chosen:
            chosen.append(name)

    # Feedback-driven nudges (Reflector → Investigator re-select).
    if feedback_l:
        if any(tok in feedback_l for tok in ("season", "周期", "日", "hour", "periodic")):
            _add("seasonal_strength")
            _add("seas_acf1")
        if any(tok in feedback_l for tok in ("volatil", "std", "noise", "波动", "var")):
            _add("basic_std")
            _add("lumpiness")
        if any(tok in feedback_l for tok in ("trend", "acf", "autocorr", "相关")):
            _add("x_acf1")
            _add("diff1_acf1")

    try:
        ss = float(features.get("seasonal_strength")) if "seasonal_strength" in features else None
        if ss is not None and ss >= 0.3:
            _add("seasonal_strength")
            _add("seas_acf1")
    except Exception:
        pass
    try:
        if "basic_std" in features:
            _add("basic_std")
    except Exception:
        pass
    _add("x_acf1")
    _add("basic_mean")

    for name in _RULE_PRIORITY:
        if len(chosen) >= min_count:
            break
        _add(name)

    if len(chosen) < min_count:
        for name in available:
            if len(chosen) >= min_count:
                break
            _add(name)

    chosen = chosen[: max(min_count, len(chosen))]
    chosen = chosen[:8]
    n = len(chosen)
    weights = {name: 1.0 / n for name in chosen} if n else {}
    rationale = (
        f"rules: selected {chosen}"
        + (" with reflective feedback" if reflective_feedback else "")
    )
    return chosen, weights, rationale


def _parse_llm_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    raise ValueError("LLM feature selection did not return JSON object")


def select_features_llm(
    features: Dict[str, Any],
    *,
    dataset_name: str,
    briefing: str = "",
    reflective_feedback: Optional[str] = None,
    min_count: int = 3,
) -> Tuple[List[str], Dict[str, float], str]:
    """Call chat completions API; validate names against F; fall back to rules on any failure."""
    available = [str(k) for k in features.keys()] if isinstance(features, dict) else []
    if not available:
        return [], {}, "no features available"

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return select_features_rules(
            features, min_count=min_count, reflective_feedback=reflective_feedback
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return select_features_rules(
            features, min_count=min_count, reflective_feedback=reflective_feedback
        )

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model = os.getenv("MODEL") or "deepseek-chat"
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    feature_preview = {
        k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
        for k, v in list(features.items())[:40]
    }
    system = (
        "You are the AlphaCast InvestigatorAgent feature selector. "
        "Given task context and a feature dictionary F, return F_selected = S(F, I_input) "
        "as JSON only: "
        '{"selected_features":[...], "feature_weights":{...}, "rationale":"..."} '
        f"Pick at least {min_count} and at most 8 names that exist in F. "
        "Weights must be non-negative and should sum to about 1.0."
    )
    user_parts = [
        f"dataset: {dataset_name}",
        f"available_feature_names: {available}",
        f"feature_values: {json.dumps(feature_preview, ensure_ascii=False)}",
    ]
    if briefing:
        user_parts.append(f"dataset_briefing:\n{briefing[:2000]}")
    if reflective_feedback:
        user_parts.append(
            "Reflector rejected the previous forecast. Re-select features using this feedback:\n"
            + reflective_feedback[:2000]
        )
    user_parts.append("Return JSON only.")

    try:
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        data = _parse_llm_json(content)
        names_raw = data.get("selected_features") or []
        weights_raw = data.get("feature_weights") or {}
        rationale = str(data.get("rationale") or "llm selection")
        names = [str(n) for n in names_raw if str(n) in available]
        for name in _RULE_PRIORITY:
            if len(names) >= min_count:
                break
            if name in available and name not in names:
                names.append(name)
        for name in available:
            if len(names) >= min_count:
                break
            if name not in names:
                names.append(name)
        names = names[:8]
        weights: Dict[str, float] = {}
        if isinstance(weights_raw, dict):
            for k, v in weights_raw.items():
                if k in names:
                    try:
                        weights[k] = float(v)
                    except Exception:
                        continue
        if len(weights) != len(names) or sum(weights.values()) <= 0:
            w = 1.0 / len(names) if names else 0.0
            weights = {n: w for n in names}
        else:
            total = float(sum(weights.values())) or 1.0
            weights = {k: float(v) / total for k, v in weights.items()}
        return names, weights, f"llm: {rationale}"
    except Exception as exc:
        names, weights, rationale = select_features_rules(
            features, min_count=min_count, reflective_feedback=reflective_feedback
        )
        return names, weights, f"rules_fallback_after_llm_error({exc}): {rationale}"


def select_features_for_mode(
    mode: str,
    features: Dict[str, Any],
    *,
    dataset_name: str,
    briefing: str = "",
    reflective_feedback: Optional[str] = None,
    min_count: int = 3,
) -> Tuple[List[str], Dict[str, float], str, str]:
    """Returns (names, weights, rationale, method_tag)."""
    mode_n = normalize_feature_selection_mode(mode)
    if mode_n == "off" or not features:
        return [], {}, "feature_selection=off", "off"
    if mode_n == "rules":
        names, weights, rationale = select_features_rules(
            features, min_count=min_count, reflective_feedback=reflective_feedback
        )
        return names, weights, rationale, "rules"
    names, weights, rationale = select_features_llm(
        features,
        dataset_name=dataset_name,
        briefing=briefing,
        reflective_feedback=reflective_feedback,
        min_count=min_count,
    )
    method = "llm" if rationale.startswith("llm:") else "rules_fallback"
    return names, weights, rationale, method
