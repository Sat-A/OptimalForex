import os
import pandas as pd

def load_minute_csv(path: str) -> pd.DataFrame:
    """
    Reads semicolon CSV with columns:
    DateTime Stamp;Bar OPEN Bid Quote;...;Bar CLOSE Bid Quote;Volume
    """
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
    """
    Resamples minute GBPUSD quotes to daily 'Close' using last observation on business days,
    then converts to USD->GBP as Rate_USD_to_GBP = 1 / GBPUSD.
    """
    daily = df_minute["Close"].resample(resample).last().dropna().to_frame("GBPUSD")
    daily["Rate_USD_to_GBP"] = 1.0 / daily["GBPUSD"]
    return daily

