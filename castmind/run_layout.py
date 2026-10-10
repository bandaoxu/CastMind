"""Isolated experiment run directories and case-library fingerprinting."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from castmind.config import ABLATION_CHOICES, ExperimentConfig, ablation_flags_dict

CASE_LIBRARY_CORE = (
    "memory.json",
    "case_base.json",
    "case_neighbor.json",
    "cluster_base.json",
)
CASE_LIBRARY_OPTIONAL = (
    "cases_stats.json",
    "features.json",
    "exogenous_features.json",
    "exogenous_correlations.json",
    "exogenous_top3.json",
    "exogenous_columns.json",
)

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def dataset_root(output_dir: str, dataset_name: str) -> Path:
    return Path(output_dir) / dataset_name


def runs_root(output_dir: str, dataset_name: str) -> Path:
    return dataset_root(output_dir, dataset_name) / "runs"


def case_libraries_root(output_dir: str, dataset_name: str) -> Path:
    return dataset_root(output_dir, dataset_name) / "case_libraries"


def sanitize_run_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("run name must be non-empty")
    if not _RUN_NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid run name {name!r}. Use ASCII letters/digits/_./- "
            f"(e.g. Full_1, no_case_1, no_knowledge_1)."
        )
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Invalid run name {name!r}")
    return cleaned


def ablation_run_prefix(ablation_id: str = "") -> str:
    """Human-readable type prefix for run folders."""
    key = (ablation_id or "").strip().lower()
    if key in ABLATION_CHOICES:
        return key
    return "Full"


def list_existing_run_names(output_dir: str, dataset_name: str) -> List[str]:
    root = runs_root(output_dir, dataset_name)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def suggest_next_run_name(
    output_dir: str,
    dataset_name: str,
    prefix: str,
) -> str:
    prefix = sanitize_run_name(prefix) if prefix else "Full"
    existing = list_existing_run_names(output_dir, dataset_name)
    max_n = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for name in existing:
        m = pattern.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{prefix}_{max_n + 1}"


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_file_hashes() -> Dict[str, Optional[str]]:
    root = Path(__file__).resolve().parent.parent / "prompts"
    out: Dict[str, Optional[str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = sha256_file(path)
    return out


def case_library_fingerprint(
    *,
    training_csv: str,
    target_column: str,
    look_back: int,
    sliding_window: int,
    predicted_window: int,
) -> str:
    train_hash = sha256_file(training_csv) if os.path.isfile(training_csv) else "missing"
    payload = {
        "train_sha256": train_hash,
        "target_column": str(target_column),
        "look_back": int(look_back),
        "sliding_window": int(sliding_window),
        "predicted_window": int(predicted_window),
    }
    digest = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return f"train_{digest[:16]}"


def case_library_ready(lib_dir: str | Path) -> bool:
    lib = Path(lib_dir)
    required = list(CASE_LIBRARY_CORE)
    for name in required:
        path = lib / name
        if not path.is_file() or path.stat().st_size <= 2:
            return False
    return True


def _load_library_manifest(lib_dir: Path) -> Optional[Dict[str, Any]]:
    path = lib_dir / "library_manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _manifest_matches_current(
    manifest: Dict[str, Any],
    *,
    training_csv: str,
    target_column: str,
    look_back: int,
    sliding_window: int,
    predicted_window: int,
) -> bool:
    train_hash = sha256_file(training_csv) if os.path.isfile(training_csv) else None
    try:
        same = (
            manifest.get("train_sha256") == train_hash
            and not manifest.get("feature_case", False)
            and str(manifest.get("target_column")) == str(target_column)
            and int(manifest.get("look_back")) == int(look_back)
            and int(manifest.get("sliding_window")) == int(sliding_window)
            and int(manifest.get("predicted_window")) == int(predicted_window)
        )
    except (TypeError, ValueError):
        return False
    return bool(same)


def find_matching_case_library(output_dir, dataset_name, fingerprint, *,
                               training_csv="", target_column="", look_back=0,
                               sliding_window=0, predicted_window=0):
    """Only reuse a documented library with matching inputs; never certify old artifacts."""
    root = case_libraries_root(output_dir, dataset_name)
    if not root.is_dir():
        return None
    candidates = sorted(root.iterdir(), key=lambda p: (p.name != fingerprint, p.name))
    for candidate in candidates:
        if not candidate.is_dir() or not case_library_ready(candidate):
            continue
        manifest = _load_library_manifest(candidate)
        if manifest and _manifest_matches_current(
            manifest, training_csv=training_csv, target_column=target_column,
            look_back=look_back, sliding_window=sliding_window, predicted_window=predicted_window
        ):
            expected = manifest.get("artifact_sha256")
            if expected and all((candidate / name).is_file() and sha256_file(candidate / name) == digest
                                for name, digest in expected.items()):
                return candidate
    return None


def write_library_manifest(
    lib_dir: Path,
    *,
    fingerprint: str,
    training_csv: str,
    target_column: str,
    look_back: int,
    sliding_window: int,
    predicted_window: int,
) -> Path:
    lib_dir.mkdir(parents=True, exist_ok=True)
    train_hash = sha256_file(training_csv) if os.path.isfile(training_csv) else None
    payload = {
        "lib_id": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {name: sha256_file(lib_dir / name) for name in CASE_LIBRARY_CORE},
        "training_csv": str(training_csv),
        "train_sha256": train_hash,
        "target_column": target_column,
        "look_back": int(look_back),
        "sliding_window": int(sliding_window),
        "predicted_window": int(predicted_window),
    }
    path = lib_dir / "library_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@dataclass
class ResolvedPaths:
    dataset_name: str
    run_name: str
    resume: bool
    run_dir: Path
    case_library_dir: Path
    lib_id: str
    manifest: Dict[str, Any] = field(default_factory=dict)


def build_run_fingerprint(
    cfg: ExperimentConfig,
    *,
    dataset_name: str,
    training_csv: str,
    test_csv: str,
    target_column: str,
    look_back: int,
    sliding_window: int,
    predicted_window: int,
    ablation_id: str,
    lib_id: str,
) -> Dict[str, Any]:
    from dataclasses import asdict
    config_state = asdict(cfg)
    for key in ("run_name", "resume", "run_dirs", "case_library_dirs", "output_dir"):
        config_state.pop(key, None)
    source_root = Path(__file__).resolve().parent.parent
    source_hashes = {str(p.relative_to(source_root)): sha256_file(p)
                     for p in sorted((source_root / "castmind").rglob("*.py"))}
    source_hashes["run_experiment.py"] = sha256_file(source_root / "run_experiment.py")
    return {
        "runtime_config": config_state,
        "source_hashes": source_hashes,
        "dataset": dataset_name,
        "run_name": getattr(cfg, "run_name", None),
        "ablation": ablation_id or None,
        "ablation_flags": ablation_flags_dict(cfg),
        "model": (os.getenv("MODEL") or "").strip() or None,
        "orchestration_mode": (os.getenv("ORCHESTRATION_MODE") or "llm").strip().lower(),
        "training_csv": training_csv,
        "train_sha256": sha256_file(training_csv) if os.path.isfile(training_csv) else None,
        "test_csv": test_csv,
        "test_sha256": sha256_file(test_csv) if os.path.isfile(test_csv) else None,
        "target_column": target_column,
        "look_back": int(look_back),
        "sliding_window": int(sliding_window),
        "predicted_window": int(predicted_window),
        "lib_id": lib_id,
        "prompt_hashes": _prompt_file_hashes(),
        "feature_selection": str(getattr(cfg, "feature_selection", "paper") or "paper"),
        "use_exogenous": bool(getattr(cfg, "use_exogenous", False)),
    }


def write_run_manifest(run_dir: Path, payload: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    full = dict(payload)
    full.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_run_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


_FINGERPRINT_KEYS = (
    "runtime_config",
    "source_hashes",
    "dataset",
    "ablation",
    "ablation_flags",
    "model",
    "orchestration_mode",
    "train_sha256",
    "test_sha256",
    "target_column",
    "look_back",
    "sliding_window",
    "predicted_window",
    "lib_id",
    "prompt_hashes",
    "feature_selection",
    "use_exogenous",
)


def verify_resume_manifest(
    stored: Dict[str, Any],
    current: Dict[str, Any],
) -> None:
    mismatches: List[str] = []
    for key in _FINGERPRINT_KEYS:
        left = stored.get(key)
        right = current.get(key)
        if left != right:
            mismatches.append(f"{key}: stored={left!r} current={right!r}")
    if mismatches:
        detail = "; ".join(mismatches[:8])
        more = f" (+{len(mismatches) - 8} more)" if len(mismatches) > 8 else ""
        raise RuntimeError(
            "Resume refused: run_manifest fingerprint mismatch. "
            f"{detail}{more}. Use a new --run-name for a different experiment."
        )


_LEGACY_ABLATION_KEYS = frozenset({"use_feature_case_library"})
_UPGRADE_FROM_TAG = "pre_d54be1e_shape"


def merge_legacy_fingerprint(
    stored: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a legacy run_manifest fingerprint into the current schema.

    Overlapping fingerprint fields must match (with special cases for
    ``prompt_hashes`` subset equality and stripped legacy ablation flags).
    Missing fingerprint keys are filled from ``current``. Non-fingerprint
    fields from ``stored`` are preserved.
    """
    if not isinstance(stored, dict) or not isinstance(current, dict):
        raise RuntimeError("Legacy upgrade refused: stored/current must be objects")

    upgraded = dict(stored)
    changed: List[str] = []

    for key in _FINGERPRINT_KEYS:
        if key not in stored:
            upgraded[key] = current.get(key)
            changed.append(key)
            continue

        left = stored.get(key)
        right = current.get(key)

        if key == "prompt_hashes":
            if not isinstance(left, dict):
                raise RuntimeError(
                    "Legacy upgrade refused: prompt_hashes stored value is not an object"
                )
            if not isinstance(right, dict):
                raise RuntimeError(
                    "Legacy upgrade refused: prompt_hashes current value is not an object"
                )
            for prompt_key, digest in left.items():
                if prompt_key not in right:
                    raise RuntimeError(
                        f"Legacy upgrade refused: prompt_hashes key {prompt_key!r} "
                        "missing from current fingerprint"
                    )
                if digest != right[prompt_key]:
                    raise RuntimeError(
                        f"Legacy upgrade refused: prompt_hashes mismatch for {prompt_key!r}"
                    )
            if left != right:
                upgraded[key] = dict(right)
                changed.append(key)
            continue

        if key == "ablation_flags":
            if not isinstance(left, dict):
                raise RuntimeError(
                    "Legacy upgrade refused: ablation_flags stored value is not an object"
                )
            if not isinstance(right, dict):
                raise RuntimeError(
                    "Legacy upgrade refused: ablation_flags current value is not an object"
                )
            trimmed = {k: v for k, v in left.items() if k not in _LEGACY_ABLATION_KEYS}
            if trimmed != right:
                raise RuntimeError(
                    "Legacy upgrade refused: ablation_flags mismatch after dropping "
                    f"legacy keys {_LEGACY_ABLATION_KEYS}: "
                    f"stored={trimmed!r} current={right!r}"
                )
            if left != right:
                upgraded[key] = dict(right)
                changed.append(key)
            continue

        if left != right:
            raise RuntimeError(
                f"Legacy upgrade refused: {key} mismatch "
                f"(stored={left!r} current={right!r})"
            )

    if not changed and all(stored.get(k) == current.get(k) for k in _FINGERPRINT_KEYS):
        # Already current-shaped; still allow stamping when called to rewrite.
        pass

    upgraded["legacy_manifest_upgraded_at"] = datetime.now(timezone.utc).isoformat()
    upgraded["legacy_manifest_upgrade_from"] = _UPGRADE_FROM_TAG
    upgraded["_upgrade_changed_keys"] = changed
    return upgraded


def upgrade_legacy_run_manifest(
    run_dir: Path,
    current: Dict[str, Any],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Upgrade ``run_manifest.json`` under ``run_dir`` to the current fingerprint schema.

    Backs up the original to ``run_manifest.pre_upgrade.json`` before writing.
    Returns the upgraded payload (including transient ``_upgrade_changed_keys``).
    """
    run_dir = Path(run_dir)
    stored = load_run_manifest(run_dir)
    if stored is None:
        raise RuntimeError(f"Legacy upgrade refused: missing run_manifest.json under {run_dir}")

    upgraded = merge_legacy_fingerprint(stored, current)
    changed = list(upgraded.pop("_upgrade_changed_keys", []))

    if dry_run:
        upgraded["_upgrade_changed_keys"] = changed
        return upgraded

    backup = run_dir / "run_manifest.pre_upgrade.json"
    if not backup.is_file():
        backup.write_text(
            json.dumps(stored, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    write_run_manifest(run_dir, upgraded)
    upgraded["_upgrade_changed_keys"] = changed
    return upgraded


def resolve_experiment_paths(
    cfg: ExperimentConfig,
    *,
    dataset_name: str,
    training_csv: str,
    test_csv: str,
    target_column: str,
    look_back: int,
    sliding_window: int,
    predicted_window: int,
    ablation_id: str,
    force_new_library: bool = False,
) -> ResolvedPaths:
    """Resolve run dir + case library; enforce new/resume directory rules."""
    run_name_raw = (getattr(cfg, "run_name", None) or os.getenv("CASTMIND_RUN_NAME") or "").strip()
    resume = bool(getattr(cfg, "resume", False)) or (
        (os.getenv("CASTMIND_RESUME") or "").strip().lower() in {"1", "true", "yes", "on"}
    )
    prefix = ablation_run_prefix(
        ablation_id,
    )
    if not run_name_raw:
        suggestion = suggest_next_run_name(cfg.output_dir, dataset_name, prefix)
        raise ValueError(
            f"--run-name is required for dataset '{dataset_name}'. "
            f"Example: --run-name {suggestion}"
            + (" --resume" if resume else "")
        )
    run_name = sanitize_run_name(run_name_raw)
    run_dir = runs_root(cfg.output_dir, dataset_name) / run_name

    fingerprint = case_library_fingerprint(
        training_csv=training_csv,
        target_column=target_column,
        look_back=look_back,
        sliding_window=sliding_window,
        predicted_window=predicted_window,
    )
    if force_new_library:
        # Force a unique lib_id so analyze writes a fresh library.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        fingerprint = f"{fingerprint}_force_{stamp}"

    matched = None if force_new_library else find_matching_case_library(
        cfg.output_dir,
        dataset_name,
        fingerprint,
        training_csv=training_csv,
        target_column=target_column,
        look_back=look_back,
        sliding_window=sliding_window,
        predicted_window=predicted_window,
    )
    case_lib_dir = matched if matched is not None else (
        case_libraries_root(cfg.output_dir, dataset_name) / fingerprint
    )

    if matched is None and case_lib_dir.exists() and not resume:
        fingerprint += "_new_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        case_lib_dir = case_libraries_root(cfg.output_dir, dataset_name) / fingerprint

    if resume and force_new_library:
        raise ValueError("Cannot rebuild a case library while resuming; start a new run")
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"--resume requested but run directory does not exist: {run_dir}"
            )
        stored = load_run_manifest(run_dir)
        if stored is None:
            raise RuntimeError(
                f"--resume refused: missing run_manifest.json under {run_dir}"
            )
        # Prefer the library recorded in the manifest when present.
        stored_lib = stored.get("case_library_dir") or stored.get("lib_id")
        if stored.get("case_library_dir"):
            case_lib_dir = Path(str(stored["case_library_dir"]))
            fingerprint = str(stored.get("lib_id") or case_lib_dir.name)
        elif stored.get("lib_id"):
            fingerprint = str(stored["lib_id"])
            case_lib_dir = case_libraries_root(cfg.output_dir, dataset_name) / fingerprint
        current = build_run_fingerprint(
            cfg,
            dataset_name=dataset_name,
            training_csv=training_csv,
            test_csv=test_csv,
            target_column=target_column,
            look_back=look_back,
            sliding_window=sliding_window,
            predicted_window=predicted_window,
            ablation_id=ablation_id,
            lib_id=fingerprint,
        )
        verify_resume_manifest(stored, current)
        manifest = _load_library_manifest(case_lib_dir)
        artifacts = (manifest or {}).get("artifact_sha256", {})
        if not case_library_ready(case_lib_dir) or not artifacts or not all(
            (case_lib_dir / name).is_file() and sha256_file(case_lib_dir / name) == digest
            for name, digest in artifacts.items()
        ):
            raise RuntimeError("Resume refused: missing or changed case library")
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                f"Pass --resume to continue, or choose a new name "
                f"(e.g. {suggest_next_run_name(cfg.output_dir, dataset_name, prefix)})."
            )
        run_dir.mkdir(parents=True, exist_ok=False)

    return ResolvedPaths(
        dataset_name=dataset_name,
        run_name=run_name,
        resume=resume,
        run_dir=run_dir,
        case_library_dir=Path(case_lib_dir),
        lib_id=fingerprint,
    )


def bind_dataset_paths(cfg: ExperimentConfig, resolved: ResolvedPaths) -> None:
    """Attach per-dataset paths onto cfg for agents/tools."""
    run_dirs = getattr(cfg, "run_dirs", None)
    if not isinstance(run_dirs, dict):
        run_dirs = {}
        setattr(cfg, "run_dirs", run_dirs)
    case_dirs = getattr(cfg, "case_library_dirs", None)
    if not isinstance(case_dirs, dict):
        case_dirs = {}
        setattr(cfg, "case_library_dirs", case_dirs)
    run_dirs[resolved.dataset_name] = str(resolved.run_dir)
    case_dirs[resolved.dataset_name] = str(resolved.case_library_dir)


def get_run_dir(cfg: Optional[ExperimentConfig], dataset_name: str, output_dir: Optional[str] = None) -> str:
    if cfg is not None:
        run_dirs = getattr(cfg, "run_dirs", None)
        if isinstance(run_dirs, dict) and dataset_name in run_dirs:
            return str(run_dirs[dataset_name])
        run_name = (getattr(cfg, "run_name", None) or "").strip()
        base = output_dir or cfg.output_dir
        if run_name:
            return str(runs_root(base, dataset_name) / sanitize_run_name(run_name))
        return str(dataset_root(base, dataset_name))
    base = output_dir or "outputs"
    return str(Path(base) / dataset_name)


def get_case_library_dir(
    cfg: Optional[ExperimentConfig],
    dataset_name: str,
    output_dir: Optional[str] = None,
) -> str:
    if cfg is not None:
        case_dirs = getattr(cfg, "case_library_dirs", None)
        if isinstance(case_dirs, dict) and dataset_name in case_dirs:
            return str(case_dirs[dataset_name])
        # Fall back to legacy flat layout (pre-migration / deterministic smoke).
        base = output_dir or cfg.output_dir
        return str(dataset_root(base, dataset_name))
    base = output_dir or "outputs"
    return str(Path(base) / dataset_name)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def replace_window_predictions(
    existing_df,
    new_chunk,
    *,
    window_offset: int,
    horizon_start: Optional[int] = None,
):
    """Replace a whole window, or only its continuation from horizon_start.

    Starting a window supersedes stale later segments; a continuation preserves
    the first segment. Re-emitting the same continuation is idempotent.
    """
    import pandas as pd

    if existing_df is None or len(existing_df) == 0:
        return new_chunk
    df = existing_df.copy()
    if "window_offset" in df.columns:
        mask = pd.to_numeric(df["window_offset"], errors="coerce") != int(window_offset)
        if horizon_start is not None:
            if "horizon_index" not in df.columns:
                raise ValueError("Cannot continue a window without horizon_index")
            indices = pd.to_numeric(df["horizon_index"], errors="coerce")
            prefix = df.loc[~mask & (indices < horizon_start)]
            if (len(prefix) != horizon_start
                    or sorted(pd.to_numeric(prefix["horizon_index"]).tolist()) != list(range(horizon_start))):
                raise ValueError("Cannot continue a window with missing or duplicate prefix indices")
            mask = mask | (indices < horizon_start)
        kept = df.loc[mask].copy()
    else:
        kept = df
    return pd.concat([kept, new_chunk], ignore_index=True)


# Archive → readable run name mapping used by migrate script.
ARCHIVE_NAME_MAP = {
    "llm_deepseekchat": "Full_1",
    "llm_deepseekchat_2": "Full_2",
    "llm_deepseek": "Full_deepseek_1",
    "feature_case": "feature_case_v1_1",
    "feature_case_v2": "feature_case_v2_1",
    "no_case": "no_case_1",
    "no_reflect": "no_reflect_1",
    "no_feature": "no_feature_1",
    "no_knowledge": "no_knowledge_1",
    "two_stage": "two_stage_1",
    "enhanced_reflect": "enhanced_reflect_1",
}


def derive_run_name_from_archive_tag(tag: str, dataset_name: str) -> str:
    """Map outputs/_archive/<tag> to a readable runs/<name>."""
    prefix = f"{dataset_name}_"
    rest = tag[len(prefix) :] if tag.startswith(prefix) else tag

    # Known exact / suffix matches (longest first).
    known_pairs = [
        ("llm_deepseekchat_2", "Full_2"),
        ("llm_deepseekchat", "Full_1"),
        ("llm_deepseek_2", "Full_deepseek_2"),
        ("llm_deepseek", "Full_deepseek_1"),
        ("feature_case_v2", "feature_case_v2_1"),
        ("feature_case", "feature_case_v1_1"),
        ("no_case", "no_case_1"),
        ("no_reflect", "no_reflect_1"),
        ("no_feature", "no_feature_1"),
        ("no_knowledge", "no_knowledge_1"),
        ("two_stage", "two_stage_1"),
        ("enhanced_reflect", "enhanced_reflect_1"),
    ]
    for suffix, name in known_pairs:
        if rest == suffix or rest.endswith("_" + suffix):
            # e.g. llm_deepseekchat_no_case
            if rest.endswith("_" + suffix) and rest != suffix:
                head = rest[: -(len(suffix) + 1)]
                # Prefer ablation-specific folders over Full.
                if suffix.startswith("llm_deepseek"):
                    abl = head.replace("llm_deepseekchat_", "").replace("llm_deepseek_", "")
                    if abl in ARCHIVE_NAME_MAP:
                        return ARCHIVE_NAME_MAP[abl]
                    if abl:
                        return f"{abl}_1"
            return name

    m = re.match(
        r"^llm_(?:deepseekchat|deepseek)(?:_(?P<ablation>[a-z0-9]+))?(?:_(?P<n>\d+))?$",
        rest,
    )
    if m:
        abl = m.group("ablation")
        n = m.group("n") or "1"
        if not abl:
            return f"Full_{n}"
        if abl in ARCHIVE_NAME_MAP:
            base = ARCHIVE_NAME_MAP[abl].rsplit("_", 1)[0]
            return f"{base}_{n}"
        return f"{abl}_{n}"

    if rest.startswith("llm_"):
        slug = rest[4:]
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_") or "migrated"
        return f"{slug}_1"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", rest).strip("_") or "migrated"
    return f"{slug}_1"


__all__ = [
    "ARCHIVE_NAME_MAP",
    "CASE_LIBRARY_CORE",
    "ResolvedPaths",
    "ablation_run_prefix",
    "append_jsonl",
    "bind_dataset_paths",
    "build_run_fingerprint",
    "case_libraries_root",
    "case_library_fingerprint",
    "case_library_ready",
    "derive_run_name_from_archive_tag",
    "find_matching_case_library",
    "get_case_library_dir",
    "get_run_dir",
    "list_existing_run_names",
    "load_run_manifest",
    "merge_legacy_fingerprint",
    "replace_window_predictions",
    "resolve_experiment_paths",
    "runs_root",
    "sanitize_run_name",
    "sha256_file",
    "suggest_next_run_name",
    "upgrade_legacy_run_manifest",
    "verify_resume_manifest",
    "write_library_manifest",
    "write_run_manifest",
]
