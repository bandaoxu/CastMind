import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml


@dataclass
class DatasetConfig:
    name: str
    training_csv: str
    test_csv: str
    look_back: int
    predicted_window: int
    sliding_window: int
    frequency: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    checkpoints: Dict[str, str] = field(default_factory=dict)
    context_prompt_file: Optional[str] = None

    def all_aliases(self) -> List[str]:
        base = {self.name.lower()}
        base.update(str(alias).lower() for alias in self.aliases)
        return sorted(base)


@dataclass
class ExperimentConfig:
    datasets: List[DatasetConfig]
    output_dir: str = "outputs"
    # New optional fields
    use_features: bool = True
    # Ablation gates (default True = Full Model; CLI --ablation flips one off)
    use_knowledge: bool = True
    use_case_library: bool = True
    use_reflector: bool = True
    # Reasoning-length ablations (paper §4.4.2 / §4.4.3); default off = Full continuous path
    two_stage: bool = False
    enhanced_reflect: bool = False
    feature_selection_override: Optional[Dict] = None
    # Investigator F_selected: off | rules | paper (LLM + rules fallback)
    feature_selection: str = "paper"
    # Max Reflector→Investigator feature re-selects per window (paper loop)
    feature_reselect_max: int = 3
    # Exogenous variable processing switch
    use_exogenous: bool = False
    sel_model: Optional[str] = None
    # Feature-vector case library (incremental; default off = Full).
    # Enable via --ablation feature_case (same pattern as two_stage).
    use_feature_case_library: bool = False
    feature_neighbor_top_k: int = 5
    feature_neighbor_auxiliary: bool = True


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw_datasets = raw.get("datasets", [])
    if isinstance(raw_datasets, dict):
        dataset_entries = []
        for key, value in raw_datasets.items():
            if not isinstance(value, dict):
                continue
            entry = {"name": value.get("name", key), **value}
            dataset_entries.append(entry)
        raw_datasets = dataset_entries
    datasets = [DatasetConfig(**d) for d in raw_datasets]
    output_dir = raw.get("output_dir", "outputs")
    # New fields with defaults
    use_features = bool(raw.get("use_features", True))
    use_knowledge = bool(raw.get("use_knowledge", True))
    use_case_library = bool(raw.get("use_case_library", True))
    use_reflector = bool(raw.get("use_reflector", True))
    two_stage = bool(raw.get("two_stage", False))
    enhanced_reflect = bool(raw.get("enhanced_reflect", False))
    feature_selection_override = raw.get("feature_selection_override")
    feature_selection = str(raw.get("feature_selection", "paper") or "paper").strip().lower()
    if feature_selection not in ("off", "rules", "paper"):
        feature_selection = "paper"
    try:
        feature_reselect_max = int(raw.get("feature_reselect_max", 3))
    except Exception:
        feature_reselect_max = 3
    feature_reselect_max = max(0, feature_reselect_max)
    use_exogenous = bool(raw.get("use_exogenous", False))
    use_feature_case_library = bool(raw.get("use_feature_case_library", False))
    try:
        feature_neighbor_top_k = int(raw.get("feature_neighbor_top_k", 5))
    except Exception:
        feature_neighbor_top_k = 5
    feature_neighbor_top_k = max(1, feature_neighbor_top_k)
    feature_neighbor_auxiliary = bool(raw.get("feature_neighbor_auxiliary", True))
    sel_model_raw = raw.get("SEL_MODEL")
    sel_model = None
    if sel_model_raw is not None:
        sel_model_str = str(sel_model_raw).strip()
        sel_model = sel_model_str or None

    # Expand env vars and absolute paths
    for d in datasets:
        d.training_csv = os.path.expandvars(d.training_csv or "")
        d.test_csv = os.path.expandvars(d.test_csv or "")
        if d.context_prompt_file:
            d.context_prompt_file = os.path.expandvars(d.context_prompt_file)
        if d.checkpoints:
            d.checkpoints = {
                str(model): os.path.expandvars(path)
                for model, path in d.checkpoints.items()
                if path
            }
        d.aliases = DatasetConfig.all_aliases(d)
    return ExperimentConfig(
        datasets=datasets,
        output_dir=output_dir,
        use_features=use_features,
        use_knowledge=use_knowledge,
        use_case_library=use_case_library,
        use_reflector=use_reflector,
        two_stage=two_stage,
        enhanced_reflect=enhanced_reflect,
        feature_selection_override=feature_selection_override,
        feature_selection=feature_selection,
        feature_reselect_max=feature_reselect_max,
        use_exogenous=use_exogenous,
        sel_model=sel_model,
        use_feature_case_library=use_feature_case_library,
        feature_neighbor_top_k=feature_neighbor_top_k,
        feature_neighbor_auxiliary=feature_neighbor_auxiliary,
    )


ABLATION_CHOICES = (
    "no_feature",
    "no_knowledge",
    "no_case",
    "no_reflect",
    "two_stage",
    "enhanced_reflect",
    "feature_case",
)


def apply_ablation(cfg: ExperimentConfig, ablation: Optional[str]) -> str:
    """Apply one ablation variant. Returns normalized ablation id or ''."""
    if ablation is None:
        return ""
    key = str(ablation).strip().lower()
    if not key:
        return ""
    if key not in ABLATION_CHOICES:
        raise ValueError(
            f"Unknown ablation '{ablation}'. Expected one of: {', '.join(ABLATION_CHOICES)}"
        )
    if key == "no_feature":
        cfg.use_features = False
    elif key == "no_knowledge":
        cfg.use_knowledge = False
    elif key == "no_case":
        cfg.use_case_library = False
    elif key == "no_reflect":
        cfg.use_reflector = False
    elif key == "two_stage":
        cfg.two_stage = True
    elif key == "enhanced_reflect":
        cfg.enhanced_reflect = True
    elif key == "feature_case":
        cfg.use_feature_case_library = True
    return key


def ablation_flags_dict(cfg: ExperimentConfig) -> Dict[str, bool]:
    return {
        "use_features": bool(getattr(cfg, "use_features", True)),
        "use_knowledge": bool(getattr(cfg, "use_knowledge", True)),
        "use_case_library": bool(getattr(cfg, "use_case_library", True)),
        "use_reflector": bool(getattr(cfg, "use_reflector", True)),
        "two_stage": bool(getattr(cfg, "two_stage", False)),
        "enhanced_reflect": bool(getattr(cfg, "enhanced_reflect", False)),
        "use_feature_case_library": bool(getattr(cfg, "use_feature_case_library", False)),
    }
