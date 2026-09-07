#!/usr/bin/env python3
"""Download and split the CastMind experiment datasets."""
from __future__ import annotations

import io
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EPF_URL = "https://zenodo.org/records/4624805/files/{name}.csv"
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
USGS = "https://waterservices.usgs.gov/nwis/dv/"

EPF_SPLITS = (10224, 1584, 3024)
ETTH_SPLITS = (8544, 1344, 2544)
ETTM_SPLITS = (16896, 2496, 4896)
POWER_SPLITS = (16896, 2496, 4896)
# AlphaCast Table 6 (same lengths as ETTh1); daily hydrology needs ≥12432 points.
MOPEX_SPLITS = ETTH_SPLITS
MOPEX_START = "1980-01-01"
MOPEX_END = "2014-12-31"
MOPEX_SITE = "01022500"
MOPEX_NEEDED = sum(MOPEX_SPLITS)

EPF_COLUMNS = {
    "NP": ["date", "system_load_forecast", "wind_power_forecast", "Price"],
    "PJM": ["date", "system_load_forecast", "comed_load_forecast", "Price"],
    "BE": ["date", "generation_forecast", "system_load_forecast", "Price"],
    "FR": ["date", "generation_forecast", "system_load_forecast", "Price"],
    "DE": ["date", "wind_power_forecast", "ampirion_zonal_load_forecast", "Price"],
}


def _urlopen(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": "CastMind-setup/1.0"})
    last_err = None
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as exc:
            last_err = exc
            print(f"[retry] {attempt + 1}/4 {exc}")
    raise last_err


def _curl_json(url: str) -> dict:
    import subprocess

    cmd = ["curl", "-fsSL", "--max-time", "120", url]
    raw = subprocess.check_output(cmd)
    return json.loads(raw.decode("utf-8"))


def download_csv(url: str) -> pd.DataFrame:
    print(f"[download] {url}")
    with _urlopen(url) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def write_split(df: pd.DataFrame, out_dir: Path, splits: Tuple[int, int, int]) -> None:
    train_n, val_n, test_n = splits
    needed = train_n + val_n + test_n
    if len(df) < needed:
        raise ValueError(f"{out_dir.name}: need {needed} rows, got {len(df)}")
    train = df.iloc[: train_n + val_n].copy()
    test = df.iloc[train_n + val_n : needed].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{out_dir.name}.csv"
    if "date" in df.columns:
        df.to_csv(full_path, index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    meta = {
        "rows_full": int(len(df)),
        "rows_train_plus_val": int(len(train)),
        "rows_test": int(len(test)),
        "splits": {"train": train_n, "val": val_n, "test": test_n},
        "columns": list(df.columns),
        "start": str(df["date"].iloc[0]) if "date" in df.columns else None,
        "end": str(df["date"].iloc[needed - 1]) if "date" in df.columns else None,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] {out_dir.name}: train={len(train)} test={len(test)}")


def prepare_epf(name: str) -> None:
    df = download_csv(EPF_URL.format(name=name))
    df.columns = [c.strip() for c in df.columns]
    time_col = df.columns[0]
    df = df.rename(columns={time_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    value_cols = [c for c in df.columns if c != "date"]
    if len(value_cols) < 3:
        raise ValueError(f"{name}: expected price + 2 exogenous columns, got {value_cols}")
    exo1, exo2, price = value_cols[0], value_cols[1], value_cols[2]
    # Price must be last so infer_target_column picks it.
    out = pd.DataFrame(
        {
            "date": df["date"],
            EPF_COLUMNS[name][1]: pd.to_numeric(df[exo1], errors="coerce"),
            EPF_COLUMNS[name][2]: pd.to_numeric(df[exo2], errors="coerce"),
            "Price": pd.to_numeric(df[price], errors="coerce"),
        }
    )
    write_split(out, DATA / f"EPF_{name}", EPF_SPLITS)


def prepare_ett(name: str, splits: Tuple[int, int, int]) -> None:
    df = download_csv(ETT_URL.format(name=name))
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    write_split(df, DATA / name, splits)


def _open_meteo(params: Dict[str, str]) -> dict:
    from urllib.parse import urlencode

    url = f"{OPEN_METEO}?{urlencode(params)}"
    print(f"[download] {url}")
    try:
        with _urlopen(url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        payload = _curl_json(url)
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", payload))
    return payload


def _wind_power(speed: pd.Series, rated: float = 12.0, cut_in: float = 3.0, cut_out: float = 25.0) -> pd.Series:
    v = speed.clip(lower=0)
    frac = ((v - cut_in) / (rated - cut_in)).clip(lower=0, upper=1) ** 3
    frac = frac.where(v <= cut_out, 0.0)
    return (100.0 * frac).rename("real_power")


def _solar_power(radiation: pd.Series) -> pd.Series:
    return (100.0 * (radiation.clip(lower=0) / 800.0).clip(upper=1)).rename("real_power")


def prepare_power(kind: str) -> None:
    """Build 15-min wind/solar series from Open-Meteo (iFLYTEK contest files are not public)."""
    if kind == "windy_power":
        lat, lon = 43.85, 87.62
        out_name = "windy_power"
    else:
        lat, lon = 38.43, 100.83
        out_name = "sunny_power"

    common = {
        "latitude": str(lat),
        "longitude": str(lon),
        "start_date": "2023-01-01",
        "end_date": "2023-09-15",
        "timezone": "UTC",
    }
    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "direct_radiation",
        "wind_speed_10m",
        "wind_speed_80m",
        "wind_direction_10m",
        "wind_direction_80m",
    ]
    try:
        payload = _open_meteo({**common, "minutely_15": ",".join(variables)})
        if "minutely_15" not in payload:
            raise KeyError(payload.get("reason", "minutely_15 unavailable"))
        df = pd.DataFrame(payload["minutely_15"]).rename(columns={"time": "date"})
    except Exception as exc:
        print(f"[warn] 15-min Open-Meteo failed ({exc}); falling back to hourly interpolation")
        payload = _open_meteo({**common, "hourly": ",".join(variables)})
        hourly = payload.get("hourly") or {}
        if "time" not in hourly:
            raise RuntimeError(payload.get("reason", "hourly unavailable"))
        df = pd.DataFrame(hourly).rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        value_cols = [c for c in df.columns if c != "date"]
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
        df = df.set_index("date").sort_index()
        df = df.resample("15min").interpolate(method="time").ffill().bfill().reset_index()

    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "wind_speed_80m" not in df.columns or df["wind_speed_80m"].isna().all():
        if "wind_speed_10m" in df.columns:
            df["wind_speed_80m"] = pd.to_numeric(df["wind_speed_10m"], errors="coerce") * 1.25
    if "wind_direction_80m" not in df.columns or df["wind_direction_80m"].isna().all():
        if "wind_direction_10m" in df.columns:
            df["wind_direction_80m"] = df["wind_direction_10m"]
    keep = [
        "date",
        "direct_radiation",
        "wind_direction_80m",
        "wind_speed_80m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
    ]
    df = df[keep].copy()
    df = df.ffill().bfill()
    if kind == "windy_power":
        df["real_power"] = _wind_power(df["wind_speed_80m"])
    else:
        df["real_power"] = _solar_power(df["direct_radiation"])
    write_split(df, DATA / out_name, POWER_SPLITS)


def prepare_mopex() -> None:
    """Daily hydrology stand-in: USGS streamflow + Open-Meteo weather for basin 01022500.

    Splits follow AlphaCast Table 6: (8544, 1344, 2544). Date range is long enough
    to yield at least 12432 valid daily rows after merge/dropna.
    """
    site = MOPEX_SITE
    url = (
        f"{USGS}?sites={site}&startDT={MOPEX_START}&endDT={MOPEX_END}"
        "&parameterCd=00060&format=json"
    )
    print(f"[download] {url}")
    with _urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    values = payload["value"]["timeSeries"][0]["values"][0]["value"]
    flow = pd.DataFrame(
        {
            "date": [row["dateTime"][:10] for row in values],
            "streamflow": [
                float(row["value"]) if row["value"] not in ("", "-999999") else math.nan
                for row in values
            ],
        }
    )
    flow["date"] = pd.to_datetime(flow["date"])

    weather = _open_meteo(
        {
            "latitude": "44.61",
            "longitude": "-68.41",
            "start_date": MOPEX_START,
            "end_date": MOPEX_END,
            "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,temperature_2m_min",
            "timezone": "UTC",
        }
    )["daily"]
    met = pd.DataFrame(
        {
            "date": pd.to_datetime(weather["time"]),
            "MAP": weather["precipitation_sum"],
            "CPE": weather["et0_fao_evapotranspiration"],
            "Tmax": weather["temperature_2m_max"],
            "Tmin": weather["temperature_2m_min"],
        }
    )
    df = met.merge(flow, on="date", how="inner").dropna()
    df = df[["date", "MAP", "CPE", "Tmax", "Tmin", "streamflow"]].reset_index(drop=True)
    if len(df) < MOPEX_NEEDED:
        raise ValueError(
            f"MOPEX: need ≥{MOPEX_NEEDED} rows for Table 6 splits {MOPEX_SPLITS}, "
            f"got {len(df)} after merge ({MOPEX_START}..{MOPEX_END}, site {site}). "
            "Extend MOPEX_START/MOPEX_END or choose another basin."
        )
    write_split(df, DATA / "MOPEX", MOPEX_SPLITS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-mopex",
        action="store_true",
        help="Rebuild MOPEX even if data/MOPEX/train.csv already exists.",
    )
    parser.add_argument(
        "--only-mopex",
        action="store_true",
        help="Only prepare MOPEX (implies rebuilding that dataset).",
    )
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("ETTh1", lambda: prepare_ett("ETTh1", ETTH_SPLITS)),
        ("ETTm1", lambda: prepare_ett("ETTm1", ETTM_SPLITS)),
        ("EPF_NP", lambda: prepare_epf("NP")),
        ("EPF_PJM", lambda: prepare_epf("PJM")),
        ("EPF_BE", lambda: prepare_epf("BE")),
        ("EPF_FR", lambda: prepare_epf("FR")),
        ("EPF_DE", lambda: prepare_epf("DE")),
        ("windy_power", lambda: prepare_power("windy_power")),
        ("sunny_power", lambda: prepare_power("sunny_power")),
        ("MOPEX", prepare_mopex),
    ]
    if args.only_mopex:
        jobs = [("MOPEX", prepare_mopex)]
        args.force_mopex = True

    failed = []
    for name, fn in jobs:
        train_csv = DATA / name / "train.csv"
        force = args.force_mopex and name == "MOPEX"
        if train_csv.exists() and name not in {"windy_power", "sunny_power"} and not force:
            # Skip stale MOPEX if split_meta is not Table 6 lengths.
            if name == "MOPEX":
                meta_path = DATA / "MOPEX" / "split_meta.json"
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    splits = meta.get("splits") or {}
                    if (
                        int(splits.get("train", -1)) != MOPEX_SPLITS[0]
                        or int(splits.get("val", -1)) != MOPEX_SPLITS[1]
                        or int(splits.get("test", -1)) != MOPEX_SPLITS[2]
                    ):
                        print(
                            f"[info] MOPEX split_meta {splits} != Table 6 {MOPEX_SPLITS}; rebuilding"
                        )
                        force = True
                except Exception:
                    force = True
            if not force:
                print(f"[skip] {name} already prepared")
                continue
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"[error] {name}: {exc}")
    if failed:
        raise SystemExit(f"failed datasets: {', '.join(failed)}")
    print("[done] all datasets prepared under data/")


if __name__ == "__main__":
    main()