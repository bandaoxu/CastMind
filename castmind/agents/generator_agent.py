from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic_ai import Agent, RunContext  # type: ignore

from castmind.config import DatasetConfig, ExperimentConfig
from castmind.data_loader import TIME_COL, infer_target_column
from .prompts import get_agent_instructions


GENERATOR_AGENT_PROMPT_FALLBACK = dedent(
    """
    You are GeneratorAgent, a world-class time-series forecasting expert operating in a multi-agent workflow.
    Each forecasting step must follow this sequence:
      1. Call `consult` exactly once to obtain the InvestigatorAgent research packet for the requested dataset/window.
      2. Examine the packet carefully. Use `reference_prediction` as the baseline; treat Investigator `selected_features` / `features_selected_values` as primary feature evidence; consult neighbor guidance and exogenous trends; only adjust the baseline when evidence clearly supports a correction.
      3. Record a brief "Reflection" that confirms the prediction length equals `predicted_window`, all pending `emit_predictions` arguments are correct (including window_offset), and the forecast aligns with the baseline guidance and exogenous outlook.
      4. Call `record_chain_of_thought` exactly once with dataset_name, window_offset, and a concise summary referencing the evidence and any adjustments (or the decision to keep the baseline).
      5. Call `emit_predictions` exactly once with the prediction list and required metadata (training_csv, predicted_window, output_dir, dataset_name, frequency, window_offset, start_timestamp, selected_features, feature_weights, and optional exogenous selections). Echo Investigator F_selected when present.

    Use no tools other than `consult`, `record_chain_of_thought`, and `emit_predictions`.
    Always call `record_chain_of_thought` before `emit_predictions`. Avoid writing step indices like "Step 0" in the chain-of-thought.
    """
)


def _timestamp_aligned_history_segment(
    ds_cfg: Optional[DatasetConfig],
    dataset_name: str,
    horizon: int,
    investor_packet: dict,
) -> Dict[str, Any]:
    """Same-clock historical segment of length H from train (paper §4.4.3)."""
    out: Dict[str, Any] = {"source": None, "values": None, "notes": ""}
    H = int(horizon)
    if H <= 0:
        out["notes"] = "invalid_horizon"
        return out

    start_ts = investor_packet.get("prediction_start_timestamp")
    if ds_cfg is not None:
        try:
            train = pd.read_csv(ds_cfg.training_csv)
            train[TIME_COL] = pd.to_datetime(train[TIME_COL])
            train = train.sort_values(TIME_COL).reset_index(drop=True)
            col = infer_target_column(train, dataset_name)
            y = train[col].to_numpy(dtype=float)
            ts = train[TIME_COL]
            if start_ts:
                target = pd.Timestamp(start_ts)
                th = int(target.hour)
                tw = int(target.dayofweek)
                candidates: List[int] = []
                for i in range(0, max(0, len(y) - H)):
                    ti = ts.iloc[i]
                    if int(ti.hour) == th and int(ti.dayofweek) == tw:
                        candidates.append(i)
                if candidates:
                    i = candidates[-1]
                    out["values"] = [float(v) for v in y[i : i + H].tolist()]
                    out["source"] = f"train_clock_align@{pd.Timestamp(ts.iloc[i]).isoformat()}"
                    return out
            # Fallback: neighbor trajectory from case library (similarity, not clock).
            neighbor = investor_packet.get("neighbor_pred")
            if isinstance(neighbor, list) and len(neighbor) >= H:
                out["values"] = [float(x) for x in neighbor[:H]]
                out["source"] = "neighbor_pred_fallback"
                out["notes"] = "no_clock_match; used_neighbor"
                return out
            out["notes"] = "no_clock_match_and_no_neighbor"
        except Exception as exc:
            out["notes"] = f"history_lookup_failed: {exc}"
    else:
        out["notes"] = "missing_dataset_config"
    return out


def create_generator_agent(
    model_name: str,
    cfg: ExperimentConfig | None,
    dataset_lookup: Dict[str, DatasetConfig],
    briefing_lookup: Dict[str, str],
    knowledge_lookup: Dict[str, str],
    prepare_investor_packet: Callable[..., dict],
    json_default: Callable[[Any], Any],
    reflector_agent: Agent,
    deterministic_run_for_dataset: Callable[[ExperimentConfig, Any], dict],
    investigator_agent: Optional[Agent] = None,
) -> Agent:
    instructions = get_agent_instructions("GeneratorAgent", GENERATOR_AGENT_PROMPT_FALLBACK)
    generator_agent = Agent(model_name, instructions=instructions)
    globals()["RunContext"] = RunContext

    investigator_cache: dict[tuple[str, int], dict[str, Any]] = {}
    chain_cache: dict[tuple[str, int], str] = {}
    # Reflector → Investigator feature re-select state (paper §3.4.2 approx).
    last_reflective_feedback: dict[tuple[str, int], str] = {}
    reselect_count: dict[tuple[str, int], int] = {}
    # investigator_agent is constructed by runtime for API symmetry; consult uses
    # prepare_investor_packet (no nested Agent.run_sync) to avoid pydantic-ai deadlocks.
    _ = investigator_agent

    def _dataset_out_dir(name: str) -> str:
        from castmind.run_layout import get_run_dir

        return get_run_dir(cfg, name)

    def _case_lib_dir(name: str) -> str:
        from castmind.run_layout import get_case_library_dir

        return get_case_library_dir(cfg, name)

    def _feature_mode() -> str:
        return str(getattr(cfg, "feature_selection", "paper") or "paper").strip().lower() if cfg else "paper"

    def _reselect_max() -> int:
        try:
            return int(getattr(cfg, "feature_reselect_max", 3) or 0) if cfg else 3
        except Exception:
            return 3

    def _append_chain_log(dataset_name: str, window_offset: int, content: str) -> str:
        ds_out_dir = _dataset_out_dir(dataset_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        path = os.path.join(ds_out_dir, "chain_of_thought.log")
        entry = {
            "window_offset": int(window_offset),
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "content": content,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    def _run_investigator(
        dataset_name: str,
        window_offset_int: int,
        forecast_horizon: Optional[int],
        feedback: Optional[str],
    ) -> dict:
        """Investigator owns F_selected / K / E (no case retrieval).

        Do **not** call ``investigator_agent.run_sync`` from inside Generator tools:
        nested sync agent runs deadlock under pydantic-ai. Packet assembly + feature
        selection (including LLM feature select via OpenAI API in ``select_features_llm``)
        go through ``prepare_investor_packet`` instead — same evidence, no nested Agent.
        """
        ds_cfg = dataset_lookup[dataset_name]
        return prepare_investor_packet(
            cfg,
            ds_cfg,
            briefing_lookup,
            window_offset_int,
            forecast_horizon,
            reflective_feedback=feedback,
            knowledge_lookup=knowledge_lookup,
            include_case_evidence=False,
        )

    @generator_agent.tool
    def consult(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int = 0,
        forecast_horizon: Optional[int] = None,
        reflective_feedback: Optional[str] = None,
    ) -> dict:
        """Merge Investigator evidence with Generator-owned case-library retrieval (§3.4.1)."""
        ds_cfg = dataset_lookup.get(dataset_name)
        if ds_cfg is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        try:
            window_offset_int = int(window_offset or 0)
        except Exception:
            window_offset_int = 0
        if forecast_horizon is not None:
            try:
                forecast_horizon = int(forecast_horizon)
            except Exception:
                forecast_horizon = None
        key = (dataset_name, window_offset_int)
        feedback = reflective_feedback or last_reflective_feedback.get(key)

        inv_packet = _run_investigator(dataset_name, window_offset_int, forecast_horizon, feedback)
        # Generator retrieves cluster auxiliary + neighbor (paper places this under Generator).
        use_case = bool(getattr(cfg, "use_case_library", True)) if cfg is not None else True
        packet = prepare_investor_packet(
            cfg,
            ds_cfg,
            briefing_lookup,
            window_offset_int,
            forecast_horizon,
            reflective_feedback=feedback,
            knowledge_lookup=knowledge_lookup,
            include_case_evidence=use_case,
            preselected_features=inv_packet.get("selected_features"),
            preselected_weights=inv_packet.get("feature_weights"),
        )
        # Prefer Investigator narrative fields when present.
        for field in (
            "selection_rationale",
            "selection_method",
            "knowledge",
            "context",
            "dataset_briefing",
            "features_selected_values",
        ):
            if inv_packet.get(field) is not None:
                packet[field] = inv_packet.get(field)
        packet["investigator_stage"] = "f_selected_k_e"
        packet["generator_stage"] = "case_library_auxiliary_neighbor"

        investigator_cache[key] = packet
        if feedback:
            print(
                f"[info] Investigator re-select features for '{dataset_name}' "
                f"offset={window_offset_int}: method={packet.get('selection_method')} "
                f"selected={packet.get('selected_features')}"
            )
        return packet

    @generator_agent.tool
    def record_chain_of_thought(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int,
        summary: str,
    ) -> dict:
        try:
            window_offset_int = int(window_offset)
        except Exception:
            window_offset_int = int(window_offset or 0)
        path = _append_chain_log(dataset_name, window_offset_int, summary)
        chain_cache[(dataset_name, window_offset_int)] = summary
        return {"logged": True, "path": path}

    @generator_agent.tool
    async def emit_predictions(
        ctx: RunContext[None],
        predictions: List[float],
        training_csv: str,
        predicted_window: int,
        output_dir: str,
        dataset_name: str,
        window_offset: Optional[int] = None,
        frequency: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        selected_features: Optional[List[str]] = None,
        feature_weights: Optional[dict] = None,
        exogenous_vars: Optional[List[str]] = None,
        exogenous_feature_selection: Optional[dict] = None,
        exogenous_correlations: Optional[dict] = None,
    ) -> dict:
        H = int(predicted_window)
        try:
            window_offset_int = int(window_offset or 0)
        except Exception:
            window_offset_int = 0
        n = len(predictions)
        if n == 0:
            raise ValueError("predictions must be a non-empty list of floats")
        if n != H:
            print(f"[warn] Normalizing LLM predictions length from {n} to {H} by {'trimming' if n>H else 'padding'}")
            if n > H:
                predictions = predictions[:H]
            else:
                last_val = float(predictions[-1])
                predictions = predictions + [last_val] * (H - n)
        arr = np.asarray(predictions, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("predictions must be finite numbers")

        ds_cfg = dataset_lookup.get(dataset_name)
        investor_packet = investigator_cache.get((dataset_name, window_offset_int))
        if investor_packet is None and ds_cfg is not None:
            try:
                use_case = bool(getattr(cfg, "use_case_library", True)) if cfg is not None else True
                investor_packet = prepare_investor_packet(
                    cfg,
                    ds_cfg,
                    briefing_lookup,
                    window_offset_int,
                    H,
                    knowledge_lookup=knowledge_lookup,
                    include_case_evidence=use_case,
                )
            except Exception:
                investor_packet = {}
        elif investor_packet is None:
            investor_packet = {}

        chain_text = chain_cache.get((dataset_name, window_offset_int), "")
        if not str(chain_text).strip():
            # Recover CoT from disk if the model emitted predictions without a
            # successful in-memory record_chain_of_thought cache hit.
            try:
                log_path = os.path.join(_dataset_out_dir(dataset_name), "chain_of_thought.log")
                if os.path.isfile(log_path):
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                            except Exception:
                                continue
                            if int(entry.get("window_offset", -1)) == window_offset_int:
                                content = entry.get("content") or ""
                                if str(content).strip():
                                    chain_text = str(content)
                                    chain_cache[(dataset_name, window_offset_int)] = chain_text
            except Exception:
                pass
        reflection_request = {
            "dataset_name": dataset_name,
            "window_offset": window_offset_int,
            "predictions": arr.tolist(),
            "predicted_window": H,
            "investor_packet": investor_packet or {},
            "chain_of_thought": chain_text,
        }
        use_reflector = bool(getattr(cfg, "use_reflector", True)) if cfg is not None else True
        enhanced = bool(getattr(cfg, "enhanced_reflect", False)) if cfg is not None else False
        if not use_reflector:
            reflection = {
                "approved": True,
                "issues": [],
                "notes": "ablation:use_reflector=false; skipped Reflector",
            }
        else:
            # Nested run_sync deadlocks inside a sync tool during an agent run; use async.
            # Bind the full Investigator packet so deterministic_audit is not starved by
            # LLM-truncated tool args (exogenous series otherwise look "ungrounded").
            setattr(reflector_agent, "_castmind_full_investor_packet", investor_packet or {})
            if enhanced:
                reflector_prompt = (
                    "Audit this Generator forecast with an EXTENDED reflective chain "
                    "(write detailed notes covering baseline deviation, exogenous outlook, "
                    "and temporal consistency). Call deterministic_audit exactly once with "
                    "predictions, predicted_window, chain_of_thought, and window_offset from the "
                    "JSON below. Pass investor_packet as {} (server supplies the full packet). "
                    "Then output ONLY JSON {approved, issues, notes}.\n\n"
                    + json.dumps(reflection_request, default=json_default)
                )
            else:
                reflector_prompt = (
                    "Audit this Generator forecast. Call deterministic_audit exactly once with "
                    "predictions, predicted_window, chain_of_thought, and window_offset from the "
                    "JSON below. Pass investor_packet as {} (server supplies the full packet). "
                    "Then output ONLY JSON {approved, issues, notes}.\n\n"
                    + json.dumps(reflection_request, default=json_default)
                )
            try:
                reflection_result = await reflector_agent.run(reflector_prompt)
            finally:
                setattr(reflector_agent, "_castmind_full_investor_packet", None)
            try:
                raw_out = reflection_result.output
                if isinstance(raw_out, dict):
                    reflection = raw_out
                else:
                    text = str(raw_out).strip()
                    if text.startswith("```"):
                        text = text.strip("`")
                        if text.startswith("json"):
                            text = text[4:].strip()
                    reflection = json.loads(text)
            except Exception:
                # Parse failure only: fall back to deterministic audit (not an approval override).
                from castmind.agents.common import assess_forecast as _assess
                from castmind.agents.reflector_agent import build_deterministic_audit_report

                reflection = build_deterministic_audit_report(reflection_request, _assess)
                reflection.setdefault("notes", (reflection.get("notes") or "") + "; fallback_parse_rules_audit")
        if not reflection.get("approved", False):
            issues = reflection.get("issues") or []
            notes = reflection.get("notes") or ""
            joined = ", ".join(str(item) for item in issues)
            feedback = f"{joined} {notes}".strip()
            key = (dataset_name, window_offset_int)
            mode = _feature_mode()
            used = int(reselect_count.get(key, 0))
            max_rs = _reselect_max()
            if mode in ("paper", "rules") and used < max_rs:
                reselect_count[key] = used + 1
                last_reflective_feedback[key] = feedback
                investigator_cache.pop(key, None)
                try:
                    with open(
                        os.path.join(_dataset_out_dir(dataset_name), "feature_reselect.jsonl"),
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            json.dumps(
                                {
                                    "window_offset": window_offset_int,
                                    "attempt": used + 1,
                                    "max": max_rs,
                                    "feedback": feedback,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                print(
                    f"[info] Reflector rejected forecast; Investigator feature re-select "
                    f"{used + 1}/{max_rs} for '{dataset_name}' offset={window_offset_int}"
                )
                raise RuntimeError(
                    f"ReflectorAgent rejected forecast (reselect {used + 1}/{max_rs}): {feedback}. "
                    "Re-call consult (feedback stored) to refresh F_selected, then emit again."
                )
            raise RuntimeError(f"ReflectorAgent rejected forecast: {feedback}")
        # Approved: clear reselect state for this window.
        key = (dataset_name, window_offset_int)
        last_reflective_feedback.pop(key, None)
        reselect_count.pop(key, None)
        try:
            with open(os.path.join(_dataset_out_dir(dataset_name), "reflector_report.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"window_offset": window_offset_int, **reflection}, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Paper §4.4.3 Enhanced Reflection: longer chain already used above; secondary
        # correction via timestamp-aligned historical segment.
        if enhanced and use_reflector:
            hist = _timestamp_aligned_history_segment(
                ds_cfg,
                dataset_name,
                H,
                investor_packet if isinstance(investor_packet, dict) else {},
            )
            enhance_req = {
                "dataset_name": dataset_name,
                "window_offset": window_offset_int,
                "original_predictions": arr.tolist(),
                "predicted_window": H,
                "timestamp_aligned_history": hist,
                "chain_of_thought": chain_text,
            }
            enhance_prompt = (
                "ENHANCED REFLECTION (§4.4.3 ablation): You already audited the forecast. "
                "Now write a LONGER reflective analysis, then apply a secondary correction "
                "anchored on the timestamp-aligned historical segment (same clock pattern). "
                "Output ONLY JSON "
                "{approved: true, revised_predictions: [<exactly predicted_window floats>], "
                "notes: string, issues: []}.\n\n"
                + json.dumps(enhance_req, default=json_default)
            )
            enhance_notes = ""
            try:
                setattr(reflector_agent, "_castmind_full_investor_packet", investor_packet or {})
                try:
                    enhance_result = await reflector_agent.run(enhance_prompt)
                finally:
                    setattr(reflector_agent, "_castmind_full_investor_packet", None)
                raw_enh = enhance_result.output
                if isinstance(raw_enh, dict):
                    enhance_obj = raw_enh
                else:
                    text = str(raw_enh).strip()
                    if text.startswith("```"):
                        text = text.strip("`")
                        if text.startswith("json"):
                            text = text[4:].strip()
                    enhance_obj = json.loads(text)
                revised = enhance_obj.get("revised_predictions")
                enhance_notes = str(enhance_obj.get("notes") or "")
                if isinstance(revised, list) and len(revised) == H:
                    rev_arr = np.asarray(revised, dtype=float)
                    if np.all(np.isfinite(rev_arr)):
                        print(
                            f"[info] Enhanced reflection revised forecast for '{dataset_name}' "
                            f"offset={window_offset_int} (hist={hist.get('source')})"
                        )
                        arr = rev_arr
                    else:
                        enhance_notes = (enhance_notes + "; ignored_nonfinite_revision").strip("; ")
                else:
                    enhance_notes = (enhance_notes + "; ignored_bad_revised_length").strip("; ")
            except Exception as exc:
                enhance_notes = f"enhanced_reflect_failed: {exc}"
                print(f"[warn] Enhanced reflection skipped for '{dataset_name}': {exc}")
            try:
                with open(
                    os.path.join(_dataset_out_dir(dataset_name), "enhanced_reflect_report.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "window_offset": window_offset_int,
                                "history_source": hist.get("source"),
                                "notes": enhance_notes,
                            },
                            ensure_ascii=False,
                            default=json_default,
                        )
                        + "\n"
                    )
            except Exception:
                pass

        ds_out_dir = _dataset_out_dir(dataset_name)
        case_lib_dir = _case_lib_dir(dataset_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        out_csv = os.path.join(ds_out_dir, "predictions.csv")

        # Prefer Investigator F_selected when feature_selection is enabled (paper Eq. 6).
        inv_selected = investor_packet.get("selected_features") if isinstance(investor_packet, dict) else None
        inv_weights = investor_packet.get("feature_weights") if isinstance(investor_packet, dict) else None
        mode = _feature_mode()
        if mode in ("paper", "rules") and isinstance(inv_selected, list) and inv_selected:
            selected_features = [str(n) for n in inv_selected]
            if isinstance(inv_weights, dict) and inv_weights:
                feature_weights = {str(k): float(v) for k, v in inv_weights.items() if str(k) in selected_features}
            print(
                f"[info] Using Investigator F_selected for '{dataset_name}': {selected_features} "
                f"(method={investor_packet.get('selection_method')})"
            )

        feat_path = os.path.join(case_lib_dir, "features.json")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(ds_out_dir, "features.json")
        # Prefer look-back features from packet when present.
        packet_features = None
        if isinstance(investor_packet, dict):
            packet_features = investor_packet.get("features_full") or investor_packet.get("features")
        if isinstance(packet_features, dict) and packet_features:
            available_features = [str(k) for k in packet_features.keys()]
        elif os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                _feat = json.load(f)
            available_features = [str(k) for k in _feat.keys()] if isinstance(_feat, dict) else []
        else:
            available_features = []

        if available_features:
            desired_count = min(3, len(available_features)) if len(available_features) >= 3 else len(available_features)

            provided = []
            if isinstance(selected_features, list):
                provided = [str(name) for name in selected_features if str(name) in available_features]

            auto_added = False
            # When Investigator already selected, do not pad with unrelated full-F names.
            force_investigator = mode in ("paper", "rules") and isinstance(inv_selected, list) and bool(inv_selected)
            if len(provided) < desired_count and not force_investigator:
                for name in available_features:
                    if name not in provided:
                        provided.append(name)
                    if len(provided) >= desired_count:
                        break
                auto_added = True

            cleaned_weights: dict[str, float] = {}
            if isinstance(feature_weights, dict):
                for key_w, value in feature_weights.items():
                    if key_w in provided:
                        try:
                            cleaned_weights[key_w] = float(value)
                        except Exception:
                            continue

            if provided and (len(cleaned_weights) != len(provided) or sum(cleaned_weights.values()) <= 0):
                weight = 1.0 / len(provided)
                cleaned_weights = {name: weight for name in provided}
            elif cleaned_weights:
                total = float(sum(cleaned_weights.values())) or 1.0
                cleaned_weights = {name: float(val) / total for name, val in cleaned_weights.items()}

            if auto_added:
                print(
                    f"[info] Auto-filled target features for dataset '{dataset_name}': {provided}"
                )

            selected_features = provided
            feature_weights = cleaned_weights

        exo_top3_path = os.path.join(case_lib_dir, "exogenous_top3.json")
        exo_feat_path = os.path.join(case_lib_dir, "exogenous_features.json")
        exo_corr_path = os.path.join(case_lib_dir, "exogenous_correlations.json")
        if not os.path.exists(exo_top3_path):
            exo_top3_path = os.path.join(ds_out_dir, "exogenous_top3.json")
            exo_feat_path = os.path.join(ds_out_dir, "exogenous_features.json")
            exo_corr_path = os.path.join(ds_out_dir, "exogenous_correlations.json")
        if (
            getattr(cfg, "use_exogenous", False)
            and os.path.exists(exo_top3_path)
            and os.path.exists(exo_feat_path)
        ):
            with open(exo_top3_path, "r", encoding="utf-8") as f:
                _top3 = json.load(f)
            with open(exo_feat_path, "r", encoding="utf-8") as f:
                _exo_feats = json.load(f)
            _exo_corrs = {}
            if os.path.exists(exo_corr_path):
                with open(exo_corr_path, "r", encoding="utf-8") as f:
                    _exo_corrs = json.load(f)

            if not isinstance(exogenous_vars, list):
                exogenous_vars = []
            vars_clean = [str(v) for v in exogenous_vars if str(v) in _top3]
            auto_vars = False
            if len(vars_clean) != len(_top3):
                vars_clean = list(_top3)
                auto_vars = True

            dims_clean: dict[str, list[str]] = {}
            if isinstance(exogenous_feature_selection, dict):
                for var, dims in exogenous_feature_selection.items():
                    if var in _top3 and isinstance(dims, list):
                        dims_clean[var] = [str(d) for d in dims if str(d) in (_exo_feats.get(var) or {})]

            auto_dims = False
            for var in _top3:
                allowed = list((_exo_feats.get(var) or {}).keys())
                original = list(dims_clean.get(var, []))
                selected = list(dims_clean.get(var, []))
                for name in allowed:
                    if name not in selected:
                        selected.append(name)
                    if len(selected) >= min(3, len(allowed)):
                        break
                if len(selected) < min(3, len(allowed)):
                    selected = allowed[: min(3, len(allowed))]
                dims_clean[var] = selected
                if selected != original:
                    auto_dims = True

            corr_clean: dict[str, float] = {}
            if isinstance(exogenous_correlations, dict):
                for var, val in exogenous_correlations.items():
                    if var in _top3:
                        try:
                            corr_clean[var] = float(val)
                        except Exception:
                            continue
            for var in _top3:
                if var not in corr_clean:
                    base_val = (_exo_corrs or {}).get(var, 0.0)
                    try:
                        corr_clean[var] = float(base_val)
                    except Exception:
                        corr_clean[var] = 0.0

            if auto_vars or auto_dims:
                dims_preview = {var: dims_clean.get(var, []) for var in _top3}
                print(
                    f"[info] Auto-filled exogenous selections for dataset '{dataset_name}': vars={vars_clean}, dims={dims_preview}"
                )

            exogenous_vars = vars_clean
            exogenous_feature_selection = dims_clean
            exogenous_correlations = corr_clean

        features_used_note = None
        try:
            if selected_features is not None or feature_weights is not None:
                names = selected_features if isinstance(selected_features, list) else []
                top_items = []
                if isinstance(feature_weights, dict) and feature_weights:
                    items = [
                        (str(k), float(v))
                        for k, v in feature_weights.items()
                        if v is not None and np.isfinite(float(v))
                    ]
                    items.sort(key=lambda kv: kv[1], reverse=True)
                    top_items = items[:10]
                pretty_weights = ", ".join([f"{k}:{v:.3f}" for k, v in top_items]) if top_items else ""
                features_used_note = f"selected=[{', '.join([str(n) for n in names])}]" + (f"; weights={pretty_weights}" if pretty_weights else "")
        except Exception:
            features_used_note = None

        parsed_start_ts: Optional[pd.Timestamp] = None
        if start_timestamp is not None:
            try:
                parsed_start_ts = pd.to_datetime(start_timestamp)
            except Exception:
                print(
                    f"[warn] Invalid start_timestamp '{start_timestamp}' for dataset '{dataset_name}'. Falling back to automatic anchoring."
                )
                parsed_start_ts = None
            # "" / "NaN" / "nat" parse to NaT without raising; NaT is not None and would
            # poison the whole emission window if treated as a valid start.
            if parsed_start_ts is not None and pd.isna(parsed_start_ts):
                print(
                    f"[warn] start_timestamp '{start_timestamp}' parsed to NaT for dataset '{dataset_name}'. Falling back to automatic anchoring."
                )
                parsed_start_ts = None

        freq_clean: Optional[str] = None
        if isinstance(frequency, str):
            _f = frequency.strip()
            if _f.lower() in {"h", "d", "w", "m", "s"}:
                freq_clean = _f.upper()
            else:
                freq_clean = _f

        existing_df: Optional[pd.DataFrame] = None
        existing_unique: Optional[pd.DataFrame] = None
        if os.path.exists(out_csv):
            existing_df = pd.read_csv(out_csv)
            if len(existing_df) > 0 and "time_stamp" in existing_df.columns:
                existing_df["time_stamp"] = pd.to_datetime(existing_df["time_stamp"], errors="coerce")  # type: ignore
                existing_valid = existing_df[existing_df["time_stamp"].notna()].copy()
                if len(existing_valid) > 0:
                    existing_unique = (
                        existing_valid.sort_values("time_stamp", kind="mergesort")
                        .drop_duplicates(subset=["time_stamp"], keep="last")
                        .reset_index(drop=True)
                    )
                else:
                    existing_unique = None
            else:
                existing_df = None
                existing_unique = None

        if not freq_clean:
            try:
                tdf_try = pd.read_csv(training_csv)
                tdf_try[TIME_COL] = pd.to_datetime(tdf_try[TIME_COL])
                tdf_try = tdf_try.sort_values(TIME_COL).reset_index(drop=True)
                freq_guess = pd.infer_freq(tdf_try[TIME_COL])
            except Exception:
                freq_guess = None
            if not freq_guess and existing_unique is not None and len(existing_unique) > 1:
                try:
                    freq_guess = pd.infer_freq(existing_unique["time_stamp"])  # type: ignore
                except Exception:
                    freq_guess = None
            if isinstance(freq_guess, str) and freq_guess.lower() in {"h", "d", "w", "m", "s"}:
                freq_clean = freq_guess.upper()
            else:
                freq_clean = freq_guess

        timestamps: List[pd.Timestamp] = []
        if parsed_start_ts is not None and freq_clean:
            try:
                offset = pd.tseries.frequencies.to_offset(freq_clean)
                timestamps = [parsed_start_ts + offset * i for i in range(H)]
            except Exception as exc:
                print(
                    f"[warn] Failed to apply provided start_timestamp for dataset '{dataset_name}': {exc}. Using automatic anchoring."
                )
                timestamps = []
                parsed_start_ts = None

        if not timestamps:
            if existing_unique is not None and len(existing_unique) > 0:
                # Prefer last valid (non-NaT) timestamp; NaT sorts last and would cascade.
                valid_ts = existing_unique["time_stamp"].dropna()
                if len(valid_ts) > 0:
                    last_ts = valid_ts.iloc[-1]
                    try:
                        last_ts = pd.Timestamp(last_ts)
                        if pd.isna(last_ts):
                            raise ValueError("last timestamp is NaT")
                        if freq_clean:
                            offset = pd.tseries.frequencies.to_offset(freq_clean)
                            timestamps = [last_ts + offset * (i + 1) for i in range(H)]
                    except Exception:
                        timestamps = []
            if not timestamps:
                try:
                    tdf_try = pd.read_csv(training_csv)
                    tdf_try[TIME_COL] = pd.to_datetime(tdf_try[TIME_COL])
                    tdf_try = tdf_try.sort_values(TIME_COL).reset_index(drop=True)
                    last_ts = pd.Timestamp(tdf_try[TIME_COL].iloc[-1])
                    inferred = pd.infer_freq(tdf_try[TIME_COL])
                    if isinstance(inferred, str) and inferred.lower() in {"h", "d", "w", "m", "s"}:
                        inferred = inferred.upper()
                    offset = pd.tseries.frequencies.to_offset(freq_clean or inferred or "D")
                    timestamps = [last_ts + offset * (i + 1) for i in range(H)]
                except Exception:
                    timestamps = []

        horizon_start = 0
        if cfg is not None and getattr(cfg, "two_stage", False):
            from castmind.prediction_segments import resolve_segment

            timestamps, horizon_start = resolve_segment(
                getattr(cfg, "_active_forecast_segment", None),
                dataset_name=dataset_name,
                window_offset=window_offset_int,
                length=H,
            )
            parsed_start_ts = timestamps[0]

        if not timestamps:
            raise RuntimeError(f"Unable to infer timestamps for dataset '{dataset_name}'.")
        if any(pd.isna(ts) for ts in timestamps):
            raise RuntimeError(
                f"Refusing to write NaT timestamps for dataset '{dataset_name}' "
                f"(start_timestamp={start_timestamp!r}, window_offset={window_offset_int})."
            )

        if existing_df is not None and "emission_index" in existing_df.columns:
            try:
                start_sequence = int(pd.to_numeric(existing_df["emission_index"], errors="coerce").max(skipna=True) or -1) + 1
            except Exception:
                start_sequence = int(existing_df.shape[0])
        elif existing_df is not None:
            start_sequence = int(existing_df.shape[0])
        else:
            start_sequence = 0

        new_chunk = pd.DataFrame(
            {
                "time_stamp": pd.to_datetime(timestamps),
                "prediction": arr.tolist(),
                "window_offset": window_offset_int,
                "window_id": str(window_offset_int),
                "horizon_index": list(range(horizon_start, horizon_start + H)),
                "emission_index": np.arange(start_sequence, start_sequence + H, dtype=int),
                "is_final": True,
            }
        )

        from castmind.run_layout import append_jsonl, replace_window_predictions
        from datetime import datetime, timezone

        emission_record = {
            "emission_id": f"{window_offset_int}:{start_sequence}",
            "attempt_id": start_sequence,
            "horizon_start": horizon_start,
            "horizon_length": H,
            "window_id": str(window_offset_int),
            "window_offset": window_offset_int,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "start_timestamp": parsed_start_ts.isoformat() if parsed_start_ts is not None else None,
            "predictions": arr.tolist(),
            "approved": True,
            "is_final": True,
        }
        append_jsonl(os.path.join(ds_out_dir, "emissions.jsonl"), emission_record)

        if existing_df is not None and len(existing_df) > 0:
            # Supersede any prior final rows for this window (Reflector retries / re-emit).
            combined = replace_window_predictions(
                existing_df, new_chunk, window_offset=window_offset_int,
                horizon_start=horizon_start if horizon_start > 0 else None,
            )
        else:
            combined = new_chunk

        combined.to_csv(out_csv, index=False)

        meta = {
            "dataset": dataset_name,
            "chosen_model": "LLM",
            "frequency": freq_clean,
            "look_back": None,
            "predicted_window": H,
        }
        if parsed_start_ts is not None:
            meta["segment_start_timestamp"] = parsed_start_ts.isoformat()
        try:
            meta["features_used"] = {
                "selected_features": selected_features if isinstance(selected_features, list) else [],
                "feature_weights": feature_weights if isinstance(feature_weights, dict) else {},
            }
        except Exception:
            pass
        try:
            if exogenous_vars is not None or exogenous_feature_selection is not None or exogenous_correlations is not None:
                exo_sel = {
                    "exogenous_vars": exogenous_vars if isinstance(exogenous_vars, list) else [],
                    "exogenous_feature_selection": exogenous_feature_selection if isinstance(exogenous_feature_selection, dict) else {},
                    "exogenous_correlations": exogenous_correlations if isinstance(exogenous_correlations, dict) else {},
                }
                with open(os.path.join(ds_out_dir, "exogenous_selected_features.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_sel, f, indent=2)
                meta["exogenous_used"] = exo_sel
                print(f"[info] LLM reported exogenous usage for dataset '{dataset_name}': vars={exo_sel['exogenous_vars']}")
        except Exception:
            pass

        with open(os.path.join(ds_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if selected_features is not None or feature_weights is not None:
            sel = {
                "selected_features": selected_features,
                "feature_weights": feature_weights,
                "selection_method": investor_packet.get("selection_method") if isinstance(investor_packet, dict) else None,
                "selection_rationale": investor_packet.get("selection_rationale") if isinstance(investor_packet, dict) else None,
                "feature_selection_mode": mode,
            }
            with open(os.path.join(ds_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                json.dump(sel, f, indent=2, ensure_ascii=False)
            try:
                top_msg = features_used_note or ""
                print(f"[info] LLM reported feature usage for dataset '{dataset_name}': {top_msg}")
            except Exception:
                pass
        return {"ok": True, "predictions_saved": out_csv}

    @generator_agent.tool
    def run_pipeline(
        ctx: RunContext[None],
        training_csv: str,
        test_csv: str,
        look_back: int,
        predicted_window: int,
        output_dir: str,
        dataset_name: str,
    ) -> dict:
        ds_obj = dataset_lookup.get(dataset_name)
        if ds_obj is None:
            class _Ds:
                def __init__(self):
                    self.name = dataset_name
                    self.training_csv = training_csv
                    self.test_csv = test_csv
                    self.look_back = int(look_back)
                    self.predicted_window = int(predicted_window)
                    self.sliding_window = int(look_back)
                    self.frequency = None
                    self.checkpoints = {}
                    self.context_prompt_file = None

            ds_obj = _Ds()

        cfg_like = ExperimentConfig(datasets=[ds_obj], output_dir=output_dir)
        return deterministic_run_for_dataset(cfg_like, ds_obj)

    return generator_agent


__all__ = ["create_generator_agent", "GENERATOR_AGENT_PROMPT_FALLBACK"]
