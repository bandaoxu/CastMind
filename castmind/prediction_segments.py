"""Deterministic placement of two-stage forecast emissions."""
from typing import Any

import pandas as pd


def resolve_segment(segment: Any, *, dataset_name: str, window_offset: int, length: int):
    """Reject missing/stale segment state instead of guessing from the last CSV row."""
    if not isinstance(segment, dict):
        raise ValueError("Two-stage emission requires runner segment context")
    if (segment.get("dataset") != dataset_name
            or segment.get("window_offset") != window_offset
            or segment.get("length") != length):
        raise ValueError("Two-stage emission does not match runner segment context")
    start = segment.get("horizon_start")
    if not isinstance(start, int) or start < 0 or length <= 0:
        raise ValueError("Invalid two-stage segment bounds")
    timestamps = list(pd.to_datetime(segment.get("timestamps", [])))
    if (len(timestamps) != length or any(pd.isna(t) for t in timestamps)
            or any(a >= b for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError("Invalid two-stage segment timestamps")
    return timestamps, start
