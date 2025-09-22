from __future__ import annotations
import numpy as np
import pandas as pd
from collections import namedtuple

from .models import RandomWalkEWMA, ArimaPointForecaster

Decision = namedtuple("Decision", ["name", "date", "rate", "extra"])

# -------- Baselines --------
def perfect_foresight(series: pd.Series, start: str, deadline: str) -> Decision:
    p = series.loc[start:deadline]
    idx = p.idxmax()
    return Decision("perfect_foresight", idx, float(p.loc[idx]), {})

def first_day(series: pd.Series, start: str, deadline: str) -> Decision:
    p = series.loc[start:deadline]
    idx = p.index[0]
    return Decision("first_day", idx, float(p.loc[idx]), {})

def last_day(series: pd.Series, start: str, deadline: str) -> Decision:
    p = series.loc[start:deadline]
    idx = p.index[-1]
    return Decision("last_day", idx, float(p.loc[idx]), {})

def historical_average(series: pd.Series, start: str, deadline: str) -> Decision:
    p_idx = series.loc[start:deadline].index
    for t in p_idx:
        hist = series.loc[:t]
        if len(hist) < 2:
            continue
        live = float(hist.iloc[-1])
        avg = float(hist.iloc[:-1].mean())
        if live > avg:
            return Decision("historical_average", t, live, {"avg": avg})
    t = p_idx[-1]
    return Decision("historical_average", t, float(series.loc[t]), {"avg": float(series.loc[:t].mean())})

def arima_trigger(series: pd.Series, start: str, deadline: str, order=(1,0,0), min_history: int = 30, max_iter: int = 100, trend: str = "n") -> Decision:
    fore = ArimaPointForecaster(order=order, max_iter=max_iter, trend=trend)
    p_idx = series.loc[start:deadline].index
    for t in p_idx:
        hist = series.loc[:t]
        if len(hist) < min_history:
            continue
        f = fore.forecast_next(hist)
        live = float(hist.iloc[-1])
        if f is not None and live > f:
            return Decision("arima_trigger", t, live, {"forecast": f})
    t = p_idx[-1]
    return Decision("arima_trigger", t, float(series.loc[t]), {"forecast": None})

# -------- Optimal stopping --------
def _risk_measure(x: np.ndarray, risk: str = "mean", alpha: float = 0.10) -> float:
    if risk == "cvar":
        q = np.quantile(x, alpha)
        tail = x[x <= q]
        return float(np.mean(tail)) if tail.size else float(q)
    return float(np.mean(x))

def compute_thresholds(series: pd.Series, start: str, deadline: str,
                       sampler: RandomWalkEWMA, min_history_days: int = 30,
                       risk: str = "mean", alpha: float = 0.10, n_samples: int = 2000) -> pd.Series:
    p = series.loc[start:deadline]
    days = p.index
    b = pd.Series(index=days, dtype=float)
    if len(days) == 0:
        raise ValueError("Empty window for thresholds.")
    # Deadline: must convert
    b.iloc[-1] = -np.inf

    for i in range(len(days)-2, -1, -1):
        t = days[i]
        hist = series.loc[:t]
        if len(hist) < min_history_days:
            b.iloc[i] = np.inf  # don't convert yet
            continue
        last_rate = float(hist.iloc[-1])
        r_next = sampler.sample_next(hist, n_samples=n_samples)
        # continuation value at t+1: max(R_{t+1}, b_{t+1})
        b_next = b.iloc[i+1]
        cont = np.maximum(r_next, b_next)
        b.iloc[i] = _risk_measure(cont, risk=risk, alpha=alpha)
    return b

def optimal_stopping(series: pd.Series, start: str, deadline: str,
                     sampler_params: dict, min_history_days: int = 30,
                     risk: str = "mean", alpha: float = 0.10, n_samples: int = 2000) -> Decision:
    sampler = RandomWalkEWMA(
        ewma_lambda=sampler_params.get("ewma_lambda", 0.94),
        drift_window=sampler_params.get("drift_window", 60),
        seed=sampler_params.get("seed", None),
    )
    thresholds = compute_thresholds(
        series, start, deadline, sampler,
        min_history_days=min_history_days, risk=risk, alpha=alpha, n_samples=n_samples
    )
    p = series.loc[start:deadline]
    for t, r in p.items():
        b_t = float(thresholds.loc[t])
        if r >= b_t:
            return Decision("optimal_stopping", t, float(r),
                            {"threshold": b_t, "thresholds": thresholds})
    # fallback last day
    t = p.index[-1]
    return Decision("optimal_stopping", t, float(p.iloc[-1]),
                    {"threshold": float(thresholds.loc[t]), "thresholds": thresholds})
