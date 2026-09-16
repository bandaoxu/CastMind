from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from ..data_loader import TIME_COL, TARGET_COL


class AnalysisMemory(TypedDict):
    max: float
    min: float
    mean: float
    variance: float
    periodicity_lag: int
    series_length: int
    frequency: Optional[str]


@dataclass
class CaseEntry:
    window: List[float]  # z-scored L-length vector
    best_model: str
    
@dataclass
class ClusterEntry:
    window: List[float]  # z-scored L-length vector
    best_model: Dict[str, int]  # model_name -> weight
    total_weight: int

@dataclass
class CaseNeighbor:
    look_back_window: List[float]  # z-scored L-length vector
    pred_window: List[float]

def season_length_from_frequency(freq: Optional[str]) -> Optional[int]:
    """Map dataset frequency to a conventional seasonal period (paper-style)."""
    if freq is None:
        return None
    f = str(freq).strip().lower()
    if f in {"h", "hh", "hour", "hourly"}:
        return 24
    if f in {"t", "min", "minute", "1min", "1t"}:
        return 1440
    if f in {"15min", "15t", "15minutely"}:
        return 96
    if f in {"d", "day", "daily"}:
        return 7
    if f in {"w", "week", "weekly"}:
        return 52
    if f in {"m", "month", "monthly", "ms"}:
        return 12
    # pandas infer_freq often returns 'H' / '15T' etc.
    if f.endswith("h") and f[:-1].isdigit():
        hours = int(f[:-1]) or 1
        return max(1, 24 // hours)
    if "min" in f or f.endswith("t"):
        return 96
    return None


def estimate_periodicity(y: np.ndarray, max_lag: Optional[int] = None) -> int:
    """ACF peak lag. Prefer resolve_season_length() when a frequency is known."""
    if len(y) < 3:
        return 1
    if max_lag is None:
        max_lag = max(2, min(len(y) // 2, 7 * 24))
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    autocorr = np.correlate(y, y, mode="full")[len(y) - 1 : len(y) - 1 + max_lag]
    if len(autocorr) < 2:
        return 1
    # Skip lag-1: raw ACF almost always peaks at 1 and collapses SeasonalNaive.
    search = autocorr[2:] if len(autocorr) > 2 else autocorr[1:]
    if len(search) == 0:
        return 1
    lag = int(np.argmax(search) + (2 if len(autocorr) > 2 else 1))
    return max(1, lag)


def resolve_season_length(
    y: np.ndarray,
    frequency: Optional[str] = None,
    max_lag: Optional[int] = None,
) -> int:
    """Frequency-first season length for stats baselines / case library."""
    freq_default = season_length_from_frequency(frequency)
    if freq_default is not None:
        return int(freq_default)
    lag = estimate_periodicity(y, max_lag=max_lag)
    if lag <= 1 and len(y) >= 48:
        return 24
    return max(1, int(lag))


def generate_future_timestamps(last_ts: pd.Timestamp, h: int, freq: Optional[str]) -> List[pd.Timestamp]:
    if freq is None:
        # Fallback: assume uniform daily spacing and increment by i
        return [last_ts + pd.Timedelta(days=i) for i in range(1, h + 1)]
    return list(pd.date_range(start=last_ts, periods=h + 1, freq=freq)[1:])