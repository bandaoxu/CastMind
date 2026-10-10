from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from textwrap import dedent, indent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from castmind.config import (
    DatasetConfig,
    ExperimentConfig,
    ABLATION_CHOICES,
    ablation_flags_dict,
    apply_ablation,
    load_config,
)
from castmind.data_loader import TIME_COL, infer_target_column
from castmind.agents.runtime import (
    build_agent_or_none,
    clear_resume_state,
    deterministic_run_for_dataset,
    load_resume_state,
    save_resume_state,
)
from castmind.agents.knowledge import build_context_lookup, build_knowledge_lookup
from castmind.eval import align_predictions, mae, mse, smape
from castmind.tools.analysis import AnalyzeResult, analyze_training
from castmind.features import extract_target_features, extract_exogenous_features

from castmind.run_layout import (
    bind_dataset_paths,
    build_run_fingerprint,
    case_library_ready,
    resolve_experiment_paths,
    write_library_manifest,
    write_run_manifest,
)


def _env_flag_true(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_cached_analysis(case_lib_dir: str) -> AnalyzeResult:
    memory_path = os.path.join(case_lib_dir, "memory.json")
    with open(memory_path, "r", encoding="utf-8") as f:
        memory = json.load(f)
    if not isinstance(memory, dict):
        memory = {}
    return AnalyzeResult(memory=memory, case_base=[], case_neighbors=[])


def _archive_tag(dataset_name: str) -> str:
    """Stable archive folder name for overwriteable slots under outputs/_archive/."""
    override = (os.getenv("CASTMIND_ARCHIVE_NAME") or "").strip()
    if override:
        return override
    mode = (os.getenv("ORCHESTRATION_MODE") or "llm").strip().lower()
    runtime = (os.getenv("CASTMIND_RUNTIME") or "main").strip().lower()
    model = (os.getenv("MODEL") or "llm").strip().lower()
    model_slug = re.sub(r"[^a-z0-9]+", "", model) or "llm"
    ablation = (os.getenv("CASTMIND_ABLATION") or "").strip().lower()
    ablation_suffix = f"_{ablation}" if ablation in ABLATION_CHOICES else ""
    if mode == "llm" and runtime in {"main", "primary", "default"}:
        return f"{dataset_name}_llm_{model_slug}{ablation_suffix}"
    if mode == "llm" and runtime in {"sundial", "side", "sideenv"}:
        return f"{dataset_name}_llm_sundial{ablation_suffix}"
    if mode == "deterministic" and runtime in {"sundial", "side", "sideenv"}:
        return f"{dataset_name}_sundial_deterministic{ablation_suffix}"
    return f"{dataset_name}_{mode}_{runtime}{ablation_suffix}"


def _parse_max_steps() -> Optional[int]:
    raw = (os.getenv("CASTMIND_MAX_STEPS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"CASTMIND_MAX_STEPS must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"CASTMIND_MAX_STEPS must be >= 1, got {value}")
    return value


def _write_metrics_json(
    metrics_dir: str,
    dataset_name: str,
    metrics_row: Dict,
    *,
    n: int,
    ablation_id: str = "",
    ablation_flags: Optional[Dict[str, bool]] = None,
    max_steps: Optional[int] = None,
    run_name: Optional[str] = None,
) -> Path:
    """Persist MSE/MAE/sMAPE next to predictions."""
    out = Path(metrics_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "run_name": run_name,
        "MSE": float(metrics_row["MSE"]),
        "MAE": float(metrics_row["MAE"]),
        "sMAPE": float(metrics_row["sMAPE"]),
        "n": int(n),
        "model": str(metrics_row.get("model", "")),
        "orchestration_mode": (os.getenv("ORCHESTRATION_MODE") or "llm").strip().lower(),
        "llm_model": (os.getenv("MODEL") or "").strip() or None,
        "ablation": ablation_id or None,
        "max_steps": int(max_steps) if max_steps is not None else None,
        "ablation_flags": ablation_flags,
    }
    path = out / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[info] Wrote {path} (n={payload['n']} MSE={payload['MSE']:.6f} MAE={payload['MAE']:.6f})")
    return path


def archive_dataset_outputs(output_dir: str, dataset_name: str, run_dir: Optional[str] = None) -> Optional[Path]:
    """Optional legacy copy into outputs/_archive/<tag> (disabled by default when using run dirs)."""
    flag = (os.getenv("CASTMIND_AUTO_ARCHIVE") or "0").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return None

    src = Path(run_dir) if run_dir else Path(output_dir) / dataset_name
    if not src.is_dir():
        print(f"[warn] Auto-archive skipped: missing source directory {src}")
        return None

    tag = _archive_tag(dataset_name)
    dest = Path(output_dir) / "_archive" / tag
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"[info] Archived {src} -> {dest} (overwrite)")
    return dest


def _load_dataset_brief(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        print(f"[warn] Failed to read dataset briefing '{path}': {exc}")
        return ""
    return text.strip()


def run_experiment(
    config_path: str,
    dataset_selectors: Optional[List[str]] = None,
    ablation: Optional[str] = None,
    run_name: Optional[str] = None,
    resume: bool = False,
) -> None:
    # Load environment from .env if present
    load_dotenv(override=False)

    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    env_run = (os.getenv("CASTMIND_RUN_NAME") or "").strip() or None
    cfg.run_name = (run_name or env_run or "").strip() or None
    cfg.resume = bool(resume) or _env_flag_true("CASTMIND_RESUME")

    env_ablation = (os.getenv("CASTMIND_ABLATION") or "").strip() or None
    ablation_id = apply_ablation(cfg, ablation or env_ablation)
    if ablation_id:
        os.environ["CASTMIND_ABLATION"] = ablation_id
    flags = ablation_flags_dict(cfg)
    print(f"[info] ablation_flags={flags}" + (f" ablation={ablation_id}" if ablation_id else " (full)"))
    if cfg.run_name:
        print(f"[info] run_name={cfg.run_name}" + (" resume=1" if cfg.resume else ""))
    max_steps = _parse_max_steps()
    if max_steps is not None:
        print(f"[info] CASTMIND_MAX_STEPS={max_steps} (early stop after N windows; not a full eval)")

    selectors = [s.strip().lower() for s in (dataset_selectors or []) if s]
    if selectors:
        alias_lookup: Dict[str, DatasetConfig] = {}
        for ds in cfg.datasets:
            for alias in ds.all_aliases():
                alias_lookup.setdefault(alias, ds)

        missing = [token for token in selectors if token not in alias_lookup]
        if missing:
            raise ValueError(f"Unknown dataset selector(s): {', '.join(missing)}")

        selected: List[DatasetConfig] = []
        seen_names = set()
        for token in selectors:
            ds = alias_lookup[token]
            if ds.name not in seen_names:
                selected.append(ds)
                seen_names.add(ds.name)

        if not selected:
            raise ValueError("No datasets resolved from provided selectors.")

        cfg.datasets = selected
        print(f"[info] Running datasets: {', '.join(ds.name for ds in cfg.datasets)}")
    
    dataset_briefings: Dict[str, str] = build_context_lookup(cfg.datasets)
    knowledge_lookup: Dict[str, str] = build_knowledge_lookup([ds.name for ds in cfg.datasets])

    # Set up logfire
    # logfire.configure()
    # logfire.instrument_pydantic_ai()

    mode = os.getenv("ORCHESTRATION_MODE", "llm").lower()
    use_agent = mode == "llm"

    agent = build_agent_or_none(cfg, dataset_briefings, knowledge_lookup) if use_agent else None
    if use_agent and agent is None:
        print("[warn] LLM orchestration unavailable or misconfigured; falling back to deterministic mode.")

    rows = []
    any_resume_required = False
    for ds in cfg.datasets:
        print(f"\n=== Processing dataset: {ds.name} ===")
        dataset_brief = dataset_briefings.get(ds.name, "")
        knowledge_brief = knowledge_lookup.get(ds.name, "")
        formatted_brief = ""
        parts = []
        if knowledge_brief:
            parts.append("Knowledge (K):\n" + indent(knowledge_brief, "  "))
        if dataset_brief:
            parts.append("Context (E):\n" + indent(dataset_brief, "  "))
        if parts:
            formatted_brief = "\n\n".join(parts)
        try:
            train_df = pd.read_csv(ds.training_csv)
            train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL])
            train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)
        except Exception as exc:
            print(f"[warn] Failed to load training data for dataset '{ds.name}': {exc}. Using deterministic fallback.")
            # Still isolate under runs/<name> when possible.
            try:
                resolved_fb = resolve_experiment_paths(
                    cfg,
                    dataset_name=ds.name,
                    training_csv=ds.training_csv,
                    test_csv=ds.test_csv,
                    target_column="unknown",
                    look_back=int(ds.look_back),
                    sliding_window=int(ds.sliding_window),
                    predicted_window=int(ds.predicted_window),
                    ablation_id=ablation_id,
                    force_new_library=False,
                )
                bind_dataset_paths(cfg, resolved_fb)
                run_dir_fb = str(resolved_fb.run_dir)
            except Exception as path_exc:
                print(f"[error] Cannot create run directory: {path_exc}")
                raise SystemExit(2) from path_exc
            det_row = deterministic_run_for_dataset(cfg, ds)
            rows.append(det_row)
            _write_metrics_json(
                run_dir_fb,
                ds.name,
                det_row,
                n=int(det_row.get("n") or 0),
                ablation_id=ablation_id,
                ablation_flags=flags,
                max_steps=max_steps,
                run_name=cfg.run_name,
            )
            continue

        look_back = int(ds.look_back)
        predicted_window = int(ds.predicted_window)
        force_analyze = _env_flag_true("CASTMIND_FORCE_ANALYZE")
        target_col = infer_target_column(train_df, ds.name)

        try:
            resolved = resolve_experiment_paths(
                cfg,
                dataset_name=ds.name,
                training_csv=ds.training_csv,
                test_csv=ds.test_csv,
                target_column=target_col,
                look_back=look_back,
                sliding_window=int(ds.sliding_window),
                predicted_window=predicted_window,
                ablation_id=ablation_id,
                force_new_library=force_analyze,
            )
        except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
            print(f"[error] {exc}")
            raise SystemExit(2) from exc

        bind_dataset_paths(cfg, resolved)
        case_lib_dir = str(resolved.case_library_dir)
        ds_out_dir = str(resolved.run_dir)
        print(
            f"[info] run_dir={ds_out_dir} case_library={case_lib_dir} "
            f"lib_id={resolved.lib_id}"
        )

        skip_analyze = not force_analyze and case_library_ready(case_lib_dir)

        try:
            if skip_analyze:
                print(
                    f"[info] Skipping analyze_training for '{ds.name}' "
                    f"(case library {resolved.lib_id} on disk; set CASTMIND_FORCE_ANALYZE=1 to rebuild)."
                )
                analysis = _load_cached_analysis(case_lib_dir)
            else:
                analysis = analyze_training(
                    train_df,
                    look_back,
                    predicted_window,
                    cfg.output_dir,
                    ds.name,
                    ds.sliding_window,
                    method="weighted",
                    num_clusters=6,
                    dataset_cfg=ds,
                    case_out_dir=case_lib_dir,
                )
                write_library_manifest(
                    Path(case_lib_dir),
                    fingerprint=resolved.lib_id,
                    training_csv=ds.training_csv,
                    target_column=target_col,
                    look_back=look_back,
                    sliding_window=int(ds.sliding_window),
                    predicted_window=predicted_window,
                )
        except Exception as exc:
            print(f"[warn] Training analysis failed for dataset '{ds.name}': {exc}. Using deterministic fallback.")
            det_row = deterministic_run_for_dataset(cfg, ds)
            rows.append(det_row)
            _write_metrics_json(
                ds_out_dir,
                ds.name,
                det_row,
                n=int(det_row.get("n") or 0),
                ablation_id=ablation_id,
                ablation_flags=flags,
                max_steps=max_steps,
                run_name=cfg.run_name,
            )
            continue

        if not cfg.resume:
            fingerprint = build_run_fingerprint(
                cfg,
                dataset_name=ds.name,
                training_csv=ds.training_csv,
                test_csv=ds.test_csv,
                target_column=target_col,
                look_back=look_back,
                sliding_window=int(ds.sliding_window),
                predicted_window=predicted_window,
                ablation_id=ablation_id,
                lib_id=resolved.lib_id,
            )
            fingerprint["case_library_dir"] = case_lib_dir
            fingerprint["run_name"] = resolved.run_name
            write_run_manifest(Path(ds_out_dir), fingerprint)

        frequency = None
        if isinstance(analysis.memory, dict):
            frequency = analysis.memory.get("frequency")

        # Dynamically infer the target column for downstream feature computation
        y = train_df[target_col].to_numpy(dtype=float)
        sel_cfg = getattr(cfg, "feature_selection_override", None)

        features = {}
        if getattr(cfg, "use_features", True):
            try:
                features = extract_target_features(y, frequency)
                with open(os.path.join(ds_out_dir, "features.json"), "w", encoding="utf-8") as f:
                    json.dump(features, f, indent=2)
                if isinstance(sel_cfg, dict):
                    with open(os.path.join(ds_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                        json.dump(sel_cfg, f, indent=2)
            except Exception as exc:
                features = {}
                print(f"[warn] Failed to compute target features for dataset '{ds.name}': {exc}")

        if features:
            try:
                basic_keys = [
                    "basic_count",
                    "basic_mean",
                    "basic_std",
                    "basic_min",
                    "basic_max",
                    "basic_skew",
                    "basic_kurt",
                ]
                print(f"- Frequency: {frequency} | Target feature count: {len(features)}")
                for key in basic_keys:
                    if key in features:
                        try:
                            print(f"- {key}: {float(features[key]):.6g}")
                        except Exception:
                            print(f"- {key}: {features[key]}")
                extras = [
                    k
                    for k in features.keys()
                    if k not in set(basic_keys + ["spectral_entropy", "seasonal_strength"])
                ]
                if "spectral_entropy" in features:
                    try:
                        print(f"- spectral_entropy: {float(features['spectral_entropy']):.6g}")
                    except Exception:
                        print(f"- spectral_entropy: {features['spectral_entropy']}")
                if "seasonal_strength" in features:
                    try:
                        print(f"- seasonal_strength: {float(features['seasonal_strength']):.6g}")
                    except Exception:
                        print(f"- seasonal_strength: {features['seasonal_strength']}")
                if extras:
                    extras = sorted(extras)
                    preview = ", ".join(extras[:20]) + (" ..." if len(extras) > 20 else "")
                    print(f"- Other feature keys: {preview}")
            except Exception:
                pass

        exo_features = {}
        exo_corr = {}
        exo_top3 = []
        exo_columns = {}
        if getattr(cfg, "use_exogenous", False):
            try:
                (
                    exo_features,
                    exo_corr,
                    exo_top3,
                    exo_columns,
                ) = extract_exogenous_features(
                    train_df, target_col, ds.name, frequency
                )
                with open(os.path.join(ds_out_dir, "exogenous_features.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_features, f, indent=2)
                with open(
                    os.path.join(ds_out_dir, "exogenous_correlations.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(exo_corr, f, indent=2)
                with open(os.path.join(ds_out_dir, "exogenous_top3.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_top3, f, indent=2)
                with open(os.path.join(ds_out_dir, "exogenous_columns.json"), "w", encoding="utf-8") as f:
                    json.dump(exo_columns, f, indent=2)

                print(
                    f"- Exogenous variables enabled. Top-3: {', '.join(exo_top3) if exo_top3 else 'None'}"
                )
                if exo_corr:
                    ordered = sorted(exo_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)
                    for name, val in ordered[:10]:
                        try:
                            print(f"  · corr({name}) = {float(val):.4f}")
                        except Exception:
                            print(f"  · corr({name}) = {val}")
            except Exception as exc:
                exo_features, exo_corr, exo_top3, exo_columns = {}, {}, [], {}
                print(f"[warn] Exogenous variable processing failed for dataset '{ds.name}': {exc}")

        try:
            test_df = pd.read_csv(ds.test_csv)
            test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
            test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)
        except Exception as exc:
            print(f"[warn] Failed to load test data for dataset '{ds.name}': {exc}. Using deterministic fallback.")
            rows.append(deterministic_run_for_dataset(cfg, ds))
            continue

        target_df = test_df.iloc[look_back:].reset_index(drop=True)

        out_csv = os.path.join(ds_out_dir, "predictions.csv")
        agent_success = False
        resume_required = False

        if agent is not None:
            resume_state = load_resume_state(ds_out_dir) if cfg.resume else None
            if not cfg.resume and os.path.exists(out_csv):
                # New run dirs should be empty; refuse rather than wipe.
                raise SystemExit(
                    f"[error] predictions.csv already exists under {ds_out_dir}. "
                    "Use --resume or a new --run-name."
                )

            total_len = len(target_df)
            if total_len == 0:
                print(
                    f"[info] Dataset '{ds.name}' has no forecast horizon after applying look_back; skipping LLM orchestration."
                )
                agent_success = True
            else:
                horizon = predicted_window
                stride = ds.sliding_window
                training_literal = json.dumps(ds.training_csv)
                output_literal = json.dumps(cfg.output_dir)
                dataset_literal = json.dumps(ds.name)
                max_net_failures = 3
                max_other_failures = 3

                def _is_network_error(ex: Exception) -> bool:
                    txt = str(ex).lower()
                    return any(
                        marker in txt
                        for marker in [
                            "timeout",
                            "timed out",
                            "connection",
                            "network",
                            "temporarily unavailable",
                            "connection reset",
                            "dns",
                            "host unreachable",
                            "429",
                            "502",
                            "503",
                            "504",
                        ]
                    )

                # Use integer prediction budget equal to remaining target length
                # to avoid float slicing and overshoot from overlapping windows.
                total_needed = int(total_len)
                current_len = 0
                current_collected = 0
                step_index = 0

                resume_state_valid = False
                if resume_state is not None:
                    try:
                        same_total = int(resume_state.get("total_len", total_len)) == total_len
                        same_stride = int(resume_state.get("stride", stride)) == stride
                        same_horizon = int(resume_state.get("horizon", horizon)) == horizon
                        if same_total and same_stride and same_horizon:
                            resume_state_valid = True
                        else:
                            print(
                                f"[warn] Ignoring stored resume state for dataset '{ds.name}' due to configuration mismatch; starting fresh."
                            )
                    except Exception:
                        pass

                if resume_state_valid:
                    stored_total_needed = resume_state.get("total_needed")
                    if stored_total_needed is not None:
                        try:
                            total_needed = int(stored_total_needed)
                        except Exception:
                            total_needed = int(total_len)
                    current_len = int(resume_state.get("current_len", 0))
                    step_index = int(resume_state.get("step_index", 0))
                    current_collected = int(resume_state.get("current_collected", 0))
                    if os.path.exists(out_csv):
                        try:
                            _existing = pd.read_csv(out_csv)
                            if "time_stamp" in _existing.columns and len(_existing) > 0:
                                _existing["time_stamp"] = pd.to_datetime(_existing["time_stamp"])
                                existing_unique = (
                                    _existing.sort_values("time_stamp", kind="mergesort")
                                    .drop_duplicates(subset=["time_stamp"], keep="last")
                                )
                                unique_len = len(existing_unique)
                            else:
                                unique_len = len(_existing)
                            current_collected = min(unique_len, current_collected or unique_len)
                        except Exception:
                            pass
                    current_collected = min(current_collected, total_needed)
                    print(
                        f"[info] Resuming LLM orchestration for dataset '{ds.name}' from step {step_index} with {current_collected}/{total_needed} predictions already collected."
                    )
                else:
                    if resume_state is not None and not cfg.resume:
                        clear_resume_state(ds_out_dir)
                    if not cfg.resume:
                        current_len = 0
                        current_collected = 0
                        step_index = 0
                    elif not resume_state_valid:
                        raise SystemExit(
                            f"[error] --resume for '{ds.name}' but resume state is missing or "
                            f"mismatched under {ds_out_dir}."
                        )

                llm_failed = False
                early_stop_due_to_bounds = False
                early_stop_max_steps = False

                while current_collected < total_needed:
                    if max_steps is not None and step_index >= max_steps:
                        early_stop_max_steps = True
                        print(
                            f"[info] Dataset '{ds.name}': CASTMIND_MAX_STEPS={max_steps} reached "
                            f"after {step_index} window(s); stopping early for ablation smoke."
                        )
                        break

                    step_horizon = min(horizon, total_needed - current_collected)
                    window_offset = current_len
                    use_two_stage = bool(getattr(cfg, "two_stage", False))

                    def _build_step_prompt(
                        *,
                        stage_horizon: int,
                        stage_label: str,
                        prior_half: Optional[List[float]] = None,
                    ) -> str:
                        sections: List[str] = []
                        if formatted_brief:
                            sections.append(formatted_brief)
                        # Bind segment placement in Python, not in LLM-generated arguments.
                        segment_start = len(prior_half) if prior_half is not None else 0
                        if use_two_stage:
                            cfg._active_forecast_segment = {
                                "dataset": ds.name,
                                "window_offset": window_offset,
                                "horizon_start": segment_start,
                                "length": stage_horizon,
                                "timestamps": [
                                    pd.Timestamp(t).isoformat()
                                    for t in test_df[TIME_COL].iloc[
                                        window_offset + look_back + segment_start:
                                        window_offset + look_back + segment_start + stage_horizon
                                    ]
                                ],
                            }
                        two_stage_extra = ""
                        if use_two_stage and prior_half is None:
                            two_stage_extra = dedent(
                                f"""
                                TWO-STAGE ABLATION (stage 1 of 2): Emit ONLY the first half of the
                                horizon ({stage_horizon} points). Do not emit the full {step_horizon}-point
                                forecast in this call. A separate paused turn will generate the remainder.
                                """
                            ).strip()
                        elif use_two_stage and prior_half is not None:
                            two_stage_extra = dedent(
                                f"""
                                TWO-STAGE ABLATION (stage 2 of 2): The previous turn already emitted the
                                first half ({len(prior_half)} points): {json.dumps(prior_half)}.
                                Continuity was intentionally interrupted (paper Table 4). Now emit ONLY
                                the remaining {stage_horizon} points. Call consult with
                                forecast_horizon={stage_horizon}. The runner binds this segment to
                                full-window indices {segment_start} through {segment_start + stage_horizon - 1}.
                                When emitting, omit start_timestamp (or set null); the runner supplies
                                the exact segment timestamps, including on retries.
                                """
                            ).strip()
                        sections.append(
                            dedent(
                                f"""
                                You are forecasting the dataset {ds.name} (step {step_index}{stage_label}).

                                Step configuration:
                                  - window_offset: {window_offset}
                                  - look_back length: {look_back}
                                  - remaining targets: {total_needed - current_collected}
                                  - forecast horizon for this call: {stage_horizon}

                                {two_stage_extra}

                                Required actions:
                                  1. Call tool.consult exactly once with dataset_name={dataset_literal}, window_offset={window_offset}, forecast_horizon={stage_horizon} to fetch the InvestigatorAgent packet.
                                  2. Analyse the packet: anchor on `reference_prediction`, compare it to neighbor hints and exogenous trends, and decide whether a careful adjustment is justified. Capture the main signals in a short internal plan.
                                  3. Before emitting predictions, write a brief "Reflection" confirming the prediction list will have length {stage_horizon}, that every argument you will pass to tool.emit_predictions (dataset_name, output_dir, predicted_window, window_offset, start_timestamp, selected_features, feature_weights, exogenous selections) is correct, and that the forecast stays consistent with the baseline guidance and exogenous outlook.
                                  4. Log the reasoning by calling tool.record_chain_of_thought exactly once with dataset_name={dataset_literal}, window_offset={window_offset}, and a concise reasoning summary referencing the evidence and any adjustments (or the decision to keep the baseline).
                                  5. Call tool.emit_predictions exactly once with:
                                       - predictions: a list of {stage_horizon} floats,
                                       - training_csv: {training_literal},
                                       - predicted_window: {stage_horizon},
                                       - output_dir: {output_literal},
                                       - dataset_name: {dataset_literal},
                                       - window_offset: {window_offset},
                                       - frequency: reuse the 'frequency' field from the context (use null if missing),
                                       - start_timestamp: reuse 'prediction_start_timestamp' from the context when provided (stage-2 two_stage: prefer null to continue),
                                       - selected_features: when the Investigator packet already provides `selected_features`, echo that list; otherwise choose ≥3 feature names from the provided dictionary when available (use [] if none),
                                       - feature_weights: echo Investigator `feature_weights` when present; otherwise non-negative weights for those features that sum to 1.0 (use {{}} if none),
                                       - exogenous_vars / exogenous_feature_selection / exogenous_correlations: when exogenous context is present, echo the listed variables, pick ≥3 dimension names per variable, and report the provided correlations.

                                Rules:
                                  - Only use consult, record_chain_of_thought, and emit_predictions.
                                  - Treat this step independently; rely only on the context you just loaded.
                                """
                            ).strip()
                        )
                        return "\n\n".join(s for s in sections if s)

                    def _run_agent_prompt(prompt: str) -> bool:
                        """Return True on success; set llm_failed/resume_required on hard failure."""
                        nonlocal llm_failed, resume_required
                        net_failures = 0
                        other_failures = 0
                        while True:
                            try:
                                agent.run_sync(prompt)
                                return True
                            except Exception as exc:
                                if _is_network_error(exc) and net_failures < max_net_failures:
                                    net_failures += 1
                                    print(
                                        f"[warn] Network error during LLM call for dataset '{ds.name}' step {step_index} (attempt {net_failures}/{max_net_failures}). Retrying..."
                                    )
                                    time.sleep(1.0)
                                    continue
                                if not _is_network_error(exc) and other_failures < max_other_failures:
                                    other_failures += 1
                                    print(
                                        f"[warn] LLM call failed for dataset '{ds.name}' step {step_index} (attempt {other_failures}/{max_other_failures}): {exc}. Retrying..."
                                    )
                                    time.sleep(1.0)
                                    continue
                                print(
                                    f"[warn] LLM orchestration failed for dataset '{ds.name}' step {step_index}: {exc}. Saving progress for resume."
                                )
                                llm_failed = True
                                resume_required = True
                                return False

                    if use_two_stage and step_horizon >= 2:
                        half1 = step_horizon // 2
                        half2 = step_horizon - half1
                        print(
                            f"[info] two_stage ablation: step {step_index} split {step_horizon} -> {half1}+{half2}"
                        )
                        if not _run_agent_prompt(
                            _build_step_prompt(stage_horizon=half1, stage_label=", two_stage=1/2")
                        ):
                            break
                        # Read first-half predictions for stage-2 context.
                        prior_half: List[float] = []
                        if os.path.exists(out_csv):
                            try:
                                _pdf = pd.read_csv(out_csv)
                                pred_col = next(
                                    (
                                        c
                                        for c in ("predicted_ans", "prediction", "forecast", "value")
                                        if c in _pdf.columns
                                    ),
                                    None,
                                )
                                if pred_col is not None and {"window_offset", "horizon_index"}.issubset(_pdf.columns):
                                    _first = _pdf.loc[
                                        (pd.to_numeric(_pdf["window_offset"], errors="coerce") == window_offset)
                                        & (pd.to_numeric(_pdf["horizon_index"], errors="coerce") < half1)
                                    ].sort_values("horizon_index")
                                    if _first["horizon_index"].tolist() == list(range(half1)):
                                        prior_half = [float(x) for x in _first[pred_col].tolist()]
                            except Exception:
                                prior_half = []
                        if len(prior_half) != half1:
                            raise RuntimeError("Cannot start two-stage continuation without the complete first half")
                        if not _run_agent_prompt(
                            _build_step_prompt(
                                stage_horizon=half2,
                                stage_label=", two_stage=2/2",
                                prior_half=prior_half or None,
                            )
                        ):
                            break
                    else:
                        if use_two_stage and step_horizon < 2:
                            print(
                                f"[warn] two_stage requested but horizon={step_horizon}<2; falling back to single emit."
                            )
                        if not _run_agent_prompt(
                            _build_step_prompt(stage_horizon=step_horizon, stage_label="")
                        ):
                            break

                    if llm_failed:
                        break

                    if not os.path.exists(out_csv):
                        print(
                            f"[warn] LLM did not emit predictions for dataset '{ds.name}' (step {step_index})."
                        )
                        llm_failed = True
                        resume_required = True
                        break

                    pred_df_full = pd.read_csv(out_csv)
                    if "time_stamp" not in pred_df_full.columns or len(pred_df_full) == 0:
                        print(
                            f"[warn] Predictions file for dataset '{ds.name}' is empty or missing 'time_stamp' column after step {step_index}."
                        )
                        llm_failed = True
                        resume_required = True
                        break

                    pred_df_full["time_stamp"] = pd.to_datetime(pred_df_full["time_stamp"])
                    unique_pred_df = (
                        pred_df_full.sort_values("time_stamp", kind="mergesort")
                        .drop_duplicates(subset=["time_stamp"], keep="last")
                        .reset_index(drop=True)
                    )
                    unique_count = len(unique_pred_df)
                    if unique_count > total_needed:
                        current_collected = int(total_needed)
                        early_stop_due_to_bounds = True
                        print(
                            f"[info] Dataset '{ds.name}': unique forecast coverage reached {total_needed}; proceeding to evaluation with full emission history retained."
                        )
                        break
                        
                    basemodel_path = os.path.join(ds_out_dir, "basemodel_results.json")
                    if os.path.exists(basemodel_path):
                        with open(basemodel_path, "r", encoding="utf-8") as f:
                            basemodel_results = json.load(f) or []
                    else:
                        basemodel_results = []

                    new_result = basemodel_results[-1] if basemodel_results else {}
                    if new_result and 'start_timestamp' not in new_result:
                        idx = current_len + look_back
                        if idx < len(test_df):
                            new_result['start_timestamp'] = test_df[TIME_COL].iloc[idx].isoformat()
                        else:
                            # Out-of-bounds start; stop and evaluate with current results
                            early_stop_due_to_bounds = True
                            print(
                                f"[warn] Start timestamp index {idx} is out-of-bounds for dataset '{ds.name}'. Stopping predictions and proceeding to evaluation."
                            )
                            break
                        
                    basemodel_results[-1] = new_result
                    with open(os.path.join(ds_out_dir, "basemodel_results.json"), "w", encoding="utf-8") as f:
                        json.dump(basemodel_results, f, indent=2)

                    new_collected = unique_count
                    added = new_collected - current_collected
                    if added <= 0:
                        print(
                            f"[warn] No additional predictions were added for dataset '{ds.name}' at step {step_index}."
                        )
                        llm_failed = True
                        resume_required = True
                        break

                    current_collected = new_collected
                    current_len += stride
                    step_index += 1

                    save_resume_state(
                        ds_out_dir,
                        {
                            "current_collected": int(current_collected),
                            "current_len": int(current_len),
                            "step_index": int(step_index),
                            "total_needed": int(total_needed),
                            "total_len": int(total_len),
                            "stride": int(stride),
                            "horizon": int(horizon),
                        },
                    )
                    print(
                        f"[info] Dataset '{ds.name}': collected {current_collected}/{total_needed} predictions via LLM."
                    )

                if not llm_failed and (
                    current_collected >= total_needed
                    or early_stop_due_to_bounds
                    or early_stop_max_steps
                ):
                    clear_resume_state(ds_out_dir)
                    pred_df_full = pd.read_csv(out_csv)
                    if "time_stamp" in pred_df_full.columns:
                        pred_df_full["time_stamp"] = pd.to_datetime(pred_df_full["time_stamp"])
                        unique_pred_df = (
                            pred_df_full.sort_values("time_stamp", kind="mergesort")
                            .drop_duplicates(subset=["time_stamp"], keep="last")
                            .reset_index(drop=True)
                        )
                        if len(unique_pred_df) < total_needed:
                            print(
                                f"[warn] Dataset '{ds.name}': only {len(unique_pred_df)} unique timestamps collected (expected {total_needed}). Evaluation will proceed with available data."
                            )
                        elif len(unique_pred_df) > total_needed:
                            print(
                                f"[info] Dataset '{ds.name}': collected {len(unique_pred_df)} unique timestamps; evaluation will use full emission history while reporting metrics."
                            )
                    else:
                        print(
                            f"[warn] Dataset '{ds.name}': predictions file missing 'time_stamp' column during evaluation; attempting best-effort alignment."
                        )
                    y_true, y_pred = align_predictions(target_df, pred_df_full, ds.name)
                    metrics_row = {
                        "dataset": ds.name,
                        "MSE": mse(y_true, y_pred),
                        "MAE": mae(y_true, y_pred),
                        "sMAPE": smape(y_true, y_pred),
                        "model": "LLM",
                    }
                    rows.append(metrics_row)
                    _write_metrics_json(
                        ds_out_dir,
                        ds.name,
                        metrics_row,
                        n=len(y_true),
                        ablation_id=ablation_id,
                        ablation_flags=flags,
                        max_steps=max_steps,
                        run_name=cfg.run_name,
                    )
                    agent_success = True

                    try:
                        sel_path = os.path.join(ds_out_dir, "selected_features.json")
                        if os.path.exists(sel_path):
                            with open(sel_path, "r", encoding="utf-8") as f:
                                sel = json.load(f)
                            sf = sel.get("selected_features")
                            fw = sel.get("feature_weights")
                            if isinstance(sf, list):
                                print(f"=== LLM feature usage | dataset: {ds.name} ===")
                                print(f"- selected_features: {', '.join(sf)}")
                            if isinstance(fw, dict) and fw:
                                try:
                                    ordered = [
                                        (str(k), float(v))
                                        for k, v in fw.items()
                                        if v is not None and np.isfinite(float(v))
                                    ]
                                    ordered.sort(key=lambda kv: kv[1], reverse=True)
                                    preview = ", ".join([f"{k}:{v:.3f}" for k, v in ordered[:10]])
                                    if preview:
                                        suffix = " ..." if len(ordered) > 10 else ""
                                        print(f"- feature_weights: {preview}{suffix}")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    try:
                        if getattr(cfg, "use_exogenous", False):
                            exo_sel_path = os.path.join(ds_out_dir, "exogenous_selected_features.json")
                            if os.path.exists(exo_sel_path):
                                with open(exo_sel_path, "r", encoding="utf-8") as f:
                                    exo_sel = json.load(f)
                                vars_used = exo_sel.get("exogenous_vars") or []
                                dims_map = exo_sel.get("exogenous_feature_selection", {})
                                cors_map = exo_sel.get("exogenous_correlations", {})
                                print(f"=== LLM exogenous usage | dataset: {ds.name} ===")
                                print(f"- variables: {', '.join(vars_used) if vars_used else 'None'}")
                                if isinstance(dims_map, dict):
                                    for var, dims in dims_map.items():
                                        if isinstance(dims, list):
                                            try:
                                                corr_val = cors_map.get(var)
                                                if corr_val is not None:
                                                    print(f"- {var}: r={float(corr_val):.4f}; dimensions -> {', '.join(dims)}")
                                                else:
                                                    print(f"- {var}: dimensions -> {', '.join(dims)}")
                                            except Exception:
                                                print(f"- {var}: dimensions -> {', '.join(dims)}")
                    except Exception:
                        pass

                elif llm_failed and resume_required:
                    any_resume_required = True
                    save_resume_state(
                        ds_out_dir,
                        {
                            "current_collected": int(current_collected),
                            "current_len": int(current_len),
                            "step_index": int(step_index),
                            "total_needed": int(total_needed),
                            "total_len": int(total_len),
                            "stride": int(stride),
                            "horizon": int(horizon),
                        },
                    )
                    print(
                        f"[info] Saved partial LLM results for dataset '{ds.name}' under {ds_out_dir}. "
                        f"Resume with the same --run-name and --resume (from step {step_index})."
                    )

        if agent_success:
            continue

        if agent is not None and resume_required:
            continue

        det_row = deterministic_run_for_dataset(cfg, ds)
        rows.append(det_row)
        _write_metrics_json(
            ds_out_dir,
            ds.name,
            det_row,
            n=int(det_row.get("n") or 0),
            ablation_id=ablation_id,
            ablation_flags=flags,
            max_steps=max_steps,
            run_name=cfg.run_name,
        )

    summary = pd.DataFrame(rows)
    print("\n=== Experiment Summary ===")
    print(summary.to_string(index=False))

    # Optional legacy _archive copy (off by default; isolated runs/ is the source of truth).
    if not summary.empty and "dataset" in summary.columns:
        for ds_name in summary["dataset"].astype(str).unique():
            run_dir = getattr(cfg, "run_dirs", {}).get(ds_name)
            archive_dataset_outputs(cfg.output_dir, ds_name, run_dir=run_dir)

    if any_resume_required:
        print(
            "[error] One or more datasets stopped with partial LLM progress (resume state saved). "
            "Re-run with the same --run-name and --resume. Exiting with code 1."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run time series agent experiments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Dataset name or alias to run (repeatable).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets defined in config (default).",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=list(ABLATION_CHOICES),
        help=(
            "Ablation variant: toolset off "
            "(no_feature|no_knowledge|no_case|no_reflect) or reasoning-length "
            "(two_stage|enhanced_reflect). Default is Full."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Isolated experiment folder under outputs/<ds>/runs/<name>/ (e.g. Full_1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing --run-name directory (same name; do not invent a new one).",
    )

    args, unknown = parser.parse_known_args()

    selector_tokens: List[str] = []
    if not args.all:
        selector_tokens.extend(args.datasets)

        for token in unknown:
            if token.startswith("--") and len(token) > 2:
                selector_tokens.append(token[2:])
            else:
                raise ValueError(f"Unrecognized argument '{token}'. Use --dataset or known aliases.")

    run_experiment(
        args.config,
        selector_tokens if selector_tokens else None,
        ablation=args.ablation,
        run_name=args.run_name,
        resume=bool(args.resume),
    )
