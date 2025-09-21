# src/forex/io.py

import os
import hashlib
import pandas as pd
from typing import Optional

def _cache_key_for(path: str, resample: str = "B") -> str:
    base = os.path.basename(path)
    try:
        mtime = int(os.path.getmtime(path))
    except OSError:
        mtime = 0
    raw = f"{base}|{mtime}|{resample}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    stem, _ = os.path.splitext(base)
    return f"{stem}_{h}.parquet"

def load_minute_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns={
        "DateTime Stamp": "Time",
        "Bar CLOSE Bid Quote": "Close",
        "Bar OPEN Bid Quote": "Open",
        "Bar HIGH Bid Quote": "High",
        "Bar LOW Bid Quote": "Low",
    })
    df["Time"] = pd.to_datetime(df["Time"], format="%Y%m%d %H%M%S")
    df = df.sort_values("Time").set_index("Time")
    return df

def to_daily_usd_to_gbp(df_minute: pd.DataFrame, resample: str = "B") -> pd.DataFrame:
    daily = df_minute["Close"].resample(resample).last().dropna().to_frame("GBPUSD")
    daily["Rate_USD_to_GBP"] = 1.0 / daily["GBPUSD"]
    # Ensure the frequency is explicitly set on the index
    daily = daily.asfreq(resample)
    return daily

def load_or_build_daily(path_csv: str, cache_dir: Optional[str] = "data/interim",
                        resample: str = "B", force_rebuild: bool = False) -> pd.DataFrame:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, _cache_key_for(path_csv, resample=resample))
    else:
        cache_path = None

    if (not force_rebuild) and cache_path and os.path.exists(cache_path):
        daily = pd.read_parquet(cache_path)
        # Parquet doesn't retain index freq; restore it here to avoid ValueWarning later
        daily = daily.asfreq(resample)
        return daily

    # Build fresh
    df_min = load_minute_csv(path_csv)
    daily = to_daily_usd_to_gbp(df_min, resample=resample)

    if cache_path:
        daily.to_parquet(cache_path, compression="snappy", index=True)
    return daily
