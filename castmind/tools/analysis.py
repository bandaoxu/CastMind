# castmind/tools/analysis.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from collections import Counter
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids

from ..data_loader import TIME_COL, infer_target_column
from ..config import DatasetConfig
from ..models.base import (
    ForecastModel,
    configure_deep_learning_runtime,
    get_default_models,
)
from ..utils.similarity import zscore, top1_most_similar,top1_most_similar_neighbor, top1_most_similar_cluster
from ..utils.time import (
    AnalysisMemory,
    CaseEntry,
    ClusterEntry,
    CaseNeighbor,
    FeatureCaseNeighbor,
    resolve_season_length,
)
from ..features import extract_target_features

FEATURE_VECTOR_KEYS = [
    "basic_count",
    "basic_mean",
    "basic_std",
    "basic_min",
    "basic_max",
    "basic_skew",
    "basic_kurt",
    "spectral_entropy",
    "crossing_points",
    "flat_spots",
    "lumpiness",
    "entropy",
    "x_acf1",
    "x_acf10",
    "diff1_acf1",
    "diff1_acf10",
    "diff2_acf1",
    "diff2_acf10",
    "seas_acf1",
    "seasonal_strength",
]


def feature_dict_to_vector(
    features: Dict,
    keys: Optional[List[str]] = None,
) -> np.ndarray:
    key_list = keys or FEATURE_VECTOR_KEYS
    vals: List[float] = []
    for k in key_list:
        v = features.get(k, 0.0) if isinstance(features, dict) else 0.0
        try:
            f = float(v)
        except Exception:
            f = 0.0
        if not np.isfinite(f):
            f = 0.0
        vals.append(f)
    return np.asarray(vals, dtype=float)


def fit_feature_scaler(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if vectors.size == 0:
        n = len(FEATURE_VECTOR_KEYS)
        return np.zeros(n, dtype=float), np.ones(n, dtype=float)
    mean = np.nanmean(vectors, axis=0)
    std = np.nanstd(vectors, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((np.isfinite(std)) & (std > 1e-8), std, 1.0)
    return mean.astype(float), std.astype(float)


def transform_feature_vector(
    vector: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    out = (np.asarray(vector, dtype=float) - mean) / std
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(float)


def _scaler_arrays(scaler: Dict) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(scaler.get("mean") or [], dtype=float)
    std = np.asarray(scaler.get("std") or [], dtype=float)
    n = len(FEATURE_VECTOR_KEYS)
    if mean.size != n:
        mean = np.zeros(n, dtype=float)
    if std.size != n:
        std = np.ones(n, dtype=float)
    std = np.where((np.isfinite(std)) & (std > 1e-8), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return mean, std


def choose_feature_neighbors_topk(
    cases: List[FeatureCaseNeighbor],
    current_features: Dict,
    scaler: Dict,
    k: int = 5,
) -> List[Tuple[FeatureCaseNeighbor, float]]:
    if not cases or k <= 0:
        return []
    mean, std = _scaler_arrays(scaler if isinstance(scaler, dict) else {})
    raw = feature_dict_to_vector(current_features)
    cur = transform_feature_vector(raw, mean, std)
    scored: List[Tuple[FeatureCaseNeighbor, float]] = []
    for case in cases:
        vec = np.asarray(case.feature_vector_norm, dtype=float)
        if vec.size != cur.size:
            continue
        dist = float(np.linalg.norm(cur - vec))
        if not np.isfinite(dist):
            continue
        scored.append((case, dist))
    scored.sort(key=lambda t: t[1])
    return scored[: int(k)]


def choose_feature_neighbor_by_similarity(
    cases: List[FeatureCaseNeighbor],
    current_features: Dict,
    scaler: Dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[FeatureCaseNeighbor]]:
    top = choose_feature_neighbors_topk(cases, current_features, scaler, k=1)
    if not top:
        return None, None, float("inf"), None
    case, dist = top[0]
    return (
        np.asarray(case.look_back_window, dtype=float),
        np.asarray(case.pred_window, dtype=float),
        float(dist),
        case,
    )


@dataclass
class AnalyzeResult:
    memory: AnalysisMemory
    case_base: List[CaseEntry]
    case_neighbors: List[CaseNeighbor]

def sliding_windows(
    y: np.ndarray,
    ts: pd.Series,
    L: int,
    H: int,
    step: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]]:
    out: List[Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]] = []
    stride = int(step) if step and step > 0 else (L + H)  # Default: no overlap
    max_start = len(y) - (L + H)
    if max_start < 0: return out
    ts = pd.to_datetime(ts)
    for s in range(0, max_start + 1, stride):
        x = y[s : s + L]
        fut = y[s + L : s + L + H]
        ts_x = ts.iloc[s : s + L]
        ts_fut = ts.iloc[s + L : s + L + H]
        out.append((x, fut, ts_x, ts_fut))
    return out

def evaluate_models_on_window(
    models: List[ForecastModel],
    x: np.ndarray,
    fut: np.ndarray,
    ts_x: pd.Series,
    ts_fut: pd.Series,
    season_length: Optional[int],
) -> Tuple[str, float]:
    best_name, best_err = "SeasonalNaive", float("inf")
    for m in models:
        try:
            # Pass only historical timestamps so deep models can build x_mark_enc
            m.fit(x, season_length=season_length, timestamps=ts_x)
            pred = m.predict(len(fut), future_timestamps=ts_fut)
            err = float(np.mean((pred - fut) ** 2))
            if err < best_err:
                best_err, best_name = err, m.alias
        except Exception as e:
            print(f"[warn] Skipping model {m.alias}: {e}")
            continue
    return best_name, best_err

# The caller always provides train_df with unchanged parameters; timestamps are now handled internally.
def analyze_training(
    train_df: pd.DataFrame,
    look_back: int,
    predicted_window: int,
    output_dir: str,
    dataset_name: str,
    sliding_window: Optional[int] = None,
    method: str = "weighted",
    num_clusters: Optional[int] = 6,
    dataset_cfg: Optional[DatasetConfig] = None,
) -> AnalyzeResult:
    target_col = infer_target_column(train_df, dataset_name)
    y = train_df[target_col].to_numpy(dtype=float)
    ts_all = pd.to_datetime(train_df[TIME_COL])

    cfg_freq = getattr(dataset_cfg, "frequency", None) if dataset_cfg is not None else None
    inferred_freq = pd.infer_freq(train_df[TIME_COL]) if len(train_df) > 1 else None
    freq = cfg_freq or inferred_freq
    memory: AnalysisMemory = {
        "max": float(np.max(y)) if len(y) else 0.0,
        "min": float(np.min(y)) if len(y) else 0.0,
        "mean": float(np.mean(y)) if len(y) else 0.0,
        "variance": float(np.var(y)) if len(y) else 0.0,
        "periodicity_lag": int(resolve_season_length(y, frequency=freq)),
        "series_length": int(len(y)),
        "frequency": freq,
    }

    if dataset_cfg is not None:
        configure_deep_learning_runtime(
            dataset_cfg.checkpoints,
            dataset_cfg.predicted_window,
        )
    else:
        configure_deep_learning_runtime(None, None)

    models = get_default_models()
    L, H = look_back, predicted_window
    stride = int(sliding_window) if sliding_window and sliding_window > 0 else None

    cases: List[CaseEntry] = []
    cases_neighbors: List[CaseNeighbor] = []
    clusters: List[CaseEntry] = []
    
    cases_stats : Dict[str, int] = {}
    feature_case_temp: List[Dict] = []
    feature_vectors_raw: List[np.ndarray] = []
    
    for x, fut, ts_x, ts_fut in sliding_windows(y, ts_all, L, H, step=stride):
        if len(x) < L or len(fut) < H: continue
        best_model, _ = evaluate_models_on_window(models, x, fut, ts_x, ts_fut, season_length=memory["periodicity_lag"])
        cases.append(CaseEntry(window=zscore(x).tolist(), best_model=best_model))
        cases_stats.setdefault(best_model, 0)
        cases_stats[best_model] += 1
        cases_neighbors.append(CaseNeighbor(look_back_window=x.tolist(), pred_window=fut.tolist()))

        # Feature-vector case library (incremental; does not replace raw neighbor).
        try:
            window_features = extract_target_features(
                np.asarray(x, dtype=float),
                memory.get("frequency") if isinstance(memory, dict) else None,
            )
            if not isinstance(window_features, dict):
                window_features = {}
        except Exception as exc:
            print(f"[warn] Feature extraction failed for a training window: {exc}")
            window_features = {}
        raw_vec = feature_dict_to_vector(window_features)
        feature_vectors_raw.append(raw_vec)
        feature_case_temp.append(
            {
                "look_back_window": x.tolist(),
                "pred_window": fut.tolist(),
                "feature_vector_raw": {
                    k: float(raw_vec[i]) for i, k in enumerate(FEATURE_VECTOR_KEYS)
                },
                "best_model": best_model,
            }
        )
    
    # AlphaCast.pdf §3.3.5: K-means case-library clustering.
    clustered = cluster_by_kmeans(cases, method=method, num_clusters=num_clusters)
    clusters.extend(clustered)

    ds_out = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_out, exist_ok=True)
    with open(os.path.join(ds_out, "cases_stats.json"), "w", encoding="utf-8") as f:
        json.dump(cases_stats, f, indent=2)
    with open(os.path.join(ds_out, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    with open(os.path.join(ds_out, "case_base.json"), "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cases], f, indent=2)
    with open(os.path.join(ds_out, "case_neighbor.json"), "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cases_neighbors], f, indent=2)  
    with open(os.path.join(ds_out, "cluster_base.json"), "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in clusters], f, indent=2)

    # Fit feature scaler on train windows and persist feature-neighbor library.
    if feature_vectors_raw:
        feature_matrix = np.vstack(feature_vectors_raw)
    else:
        feature_matrix = np.empty((0, len(FEATURE_VECTOR_KEYS)), dtype=float)
    mean, std = fit_feature_scaler(feature_matrix)
    feature_cases_out: List[Dict] = []
    for item, raw_vec in zip(feature_case_temp, feature_vectors_raw):
        norm_vec = transform_feature_vector(raw_vec, mean, std)
        feature_cases_out.append({**item, "feature_vector_norm": norm_vec.tolist()})
    with open(os.path.join(ds_out, "case_feature_neighbor.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cases_out, f, indent=2)
    with open(os.path.join(ds_out, "case_feature_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_keys": list(FEATURE_VECTOR_KEYS),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "method": "zscore_train_windows",
            },
            f,
            indent=2,
        )
        
    print(f"Wrting case base to {os.path.join(ds_out, 'case_base.json')}")
    print(f"Wrting cluster base to {os.path.join(ds_out, 'cluster_base.json')}")
    print(f"Writing feature case neighbor to {os.path.join(ds_out, 'case_feature_neighbor.json')}")

    result = AnalyzeResult(memory=memory, case_base=cases, case_neighbors=cases_neighbors)

    # Clear model-specific context after analysis to avoid leaking to other datasets.
    configure_deep_learning_runtime(None, None)
    return result

def choose_model_by_similarity(cases: List[CaseEntry], current_window: np.ndarray) -> str:
    candidates = [(np.asarray(c.window, dtype=float), c.best_model) for c in cases]
    model_name, _ = top1_most_similar(zscore(current_window), candidates)
    return model_name

def choose_neighbor_by_similarity(cases: List[CaseNeighbor], current_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    candidates = [(np.asarray(c.look_back_window, dtype=float), np.asarray(c.pred_window, dtype=float)) for c in cases]
    neighbor_lookback, neighbor_pred = top1_most_similar_neighbor(current_window, candidates)
    return neighbor_lookback, neighbor_pred

def choose_cluster_by_similarity(clusters: List[ClusterEntry], current_window: np.ndarray) -> ClusterEntry:
    candidates = [c for c in clusters if c.window]
    best_cluster = top1_most_similar_cluster(zscore(current_window), candidates)
    return best_cluster

def _assign_cluster_models(
    cases: List[CaseEntry],
    cluster_indices_list: List[List[int]],
    center_windows: List,
    method: Optional[str],
) -> List[ClusterEntry]:
    centers: List[ClusterEntry] = [
        ClusterEntry(window=w, best_model={}, total_weight=0) for w in center_windows
    ]
    if method == "voting":
        for gi, cluster_indices in enumerate(cluster_indices_list):
            group_cases = [cases[idx] for idx in cluster_indices]
            if group_cases:
                counts = Counter(c.best_model for c in group_cases)
                centers[gi].best_model = {counts.most_common(1)[0][0]: 1}
                centers[gi].total_weight = 1
    elif method == "weighted":
        # Paper Eq. (7): cluster-local models weighted by win frequency.
        # w_i = n_i / sum_j n_j (no count>3 filter; that was an upstream heuristic).
        for gi, cluster_indices in enumerate(cluster_indices_list):
            group_cases = [cases[idx] for idx in cluster_indices]
            if group_cases:
                counts = Counter(c.best_model for c in group_cases)
                centers[gi].best_model = dict(counts)
                centers[gi].total_weight = sum(counts.values())
    return centers


def cluster_by_kmeans(
    cases: List[CaseEntry],
    method: Optional[str] = "voting",
    num_clusters: Optional[int] = 6,
) -> List[ClusterEntry]:
    """Cluster cases with K-means (AlphaCast.pdf §3.3.5) and return cluster-mean centers."""
    if not cases:
        return []

    k = int(num_clusters) if (num_clusters and num_clusters > 0) else 4
    k = max(1, min(k, len(cases)))
    window_vectors = np.asarray(
        [c.window.tolist() if hasattr(c.window, "tolist") else list(c.window) for c in cases],
        dtype=float,
    )

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(window_vectors)
        centers_arr = km.cluster_centers_
    except Exception:
        # Lightweight fallback when sklearn is unavailable.
        rng = np.random.default_rng(0)
        centers_arr = window_vectors[rng.choice(len(window_vectors), size=k, replace=False)].copy()
        labels = np.zeros(len(window_vectors), dtype=int)
        for _ in range(20):
            dists = ((window_vectors[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            for j in range(k):
                members = window_vectors[labels == j]
                if len(members):
                    centers_arr[j] = members.mean(axis=0)

    cluster_indices_list = [[i for i, lab in enumerate(labels) if int(lab) == j] for j in range(k)]
    cluster_indices_list = [idxs for idxs in cluster_indices_list if idxs]
    center_windows = [centers_arr[j].tolist() for j in range(len(cluster_indices_list))]
    print(f"[info] Case-library clustering: K-means (k={len(cluster_indices_list)}) [AlphaCast.pdf]")
    return _assign_cluster_models(cases, cluster_indices_list, center_windows, method)


def cluster_by_kmedoid(
    cases: List[CaseEntry],
    metric: Optional[str] = "cosine",
    method: Optional[str] = "voting",
    num_clusters: Optional[int] = 6,
) -> List[ClusterEntry]:
    """
    Cluster cases with K-Medoids and return the medoid CaseEntry objects.
    """
    if not cases:
        return []

    k = int(num_clusters) if (num_clusters and num_clusters > 0) else 4
    k = max(1, min(k, len(cases)))

    # Convert NumPy arrays to Python lists for pyclustering compatibility
    window_vectors = [c.window.tolist() if hasattr(c.window, 'tolist') else list(c.window) for c in cases]
    
    # Run pyclustering's K-Medoids implementation.
    # Note: pyclustering does not support cosine distance, so we use Euclidean distance.
    import random
    random.seed(0)  # Ensure reproducibility
    initial_medoids = random.sample(range(len(cases)), k)
    kmedoids_instance = kmedoids(window_vectors, initial_medoids, ccore=False)
    kmedoids_instance.process()
    
    # Retrieve clustering results
    clusters = kmedoids_instance.get_clusters()
    medoid_indices = kmedoids_instance.get_medoids()

    centers: List[ClusterEntry] = [ClusterEntry(window=cases[i].window, best_model={}, total_weight=0) for i in medoid_indices]
    print(f"[info] Case-library clustering: K-Medoids (k={len(medoid_indices)}) [legacy]")
    return _assign_cluster_models(
        cases,
        clusters,
        [c.window for c in centers],
        method,
    )
