import os
import json
import time
import pandas as pd

from .metrics import calculate_gbp, regret_vs_hindsight
from . import plotting as plotting_mod
from . import strategies as S

def _run_strategies_for_window(series, start, deadline, amount_usd, fees, forecaster_cfg, strategies_cfg, save_plots, outdir):
    # Always compute perfect foresight first for regret baseline
    dec_pf = S.perfect_foresight(series, start, deadline)
    gbp_pf = calculate_gbp(amount_usd, dec_pf.rate, fees)

    decisions = [dec_pf]
    for entry in strategies_cfg:
        if isinstance(entry, str):
            name, params = entry, {}
        else:
            name, params = entry["name"], entry.get("params", {})

        if name == "first_day":
            d = S.first_day(series, start, deadline)
        elif name == "last_day":
            d = S.last_day(series, start, deadline)
        elif name == "historical_average":
            d = S.historical_average(series, start, deadline)
        elif name == "arima_trigger":
            order = tuple(params.get("order", (1,0,0)))
            min_hist = int(params.get("min_history", 30))
            d = S.arima_trigger(series, start, deadline, order=order, min_history=min_hist)
        elif name == "optimal_stopping":
            risk = params.get("risk", "mean")
            alpha = float(params.get("alpha", 0.10))
            min_hist = int(params.get("min_history_days", 30))
            sampler_params = {
                "ewma_lambda": forecaster_cfg["params"].get("ewma_lambda", 0.94),
                "drift_window": forecaster_cfg["params"].get("drift_window", 60),
                "seed": forecaster_cfg["params"].get("seed", None),
            }
            n_samples = int(forecaster_cfg["params"].get("samples", 2000))
            d = S.optimal_stopping(series, start, deadline,
                                   sampler_params=sampler_params,
                                   min_history_days=min_hist,
                                   risk=risk, alpha=alpha, n_samples=n_samples)
        elif name == "perfect_foresight":
            # already computed
            d = dec_pf
        else:
            raise ValueError(f"Unknown strategy: {name}")

        decisions.append(d)

    # Build results rows
    rows = []
    for dec in decisions:
        gbp = calculate_gbp(amount_usd, dec.rate, fees)
        rows.append({
            "start": start,
            "deadline": deadline,
            "strategy": dec.name,
            "date": dec.date.date(),
            "rate": dec.rate,
            "gbp_received": gbp,
            "regret_vs_pf": regret_vs_hindsight(gbp, gbp_pf),
        })

    # Plot if requested
    if save_plots:
        plotting_mod.plot_window(series, start, deadline, decisions, outdir)

    return rows

def run_many(series, windows, amount_usd, fees, forecaster_cfg, strategies_cfg, save_plots, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Save config snapshot
    cfg_snap = {
        "windows": windows,
        "amount_usd": amount_usd,
        "fees": fees,
        "forecaster": forecaster_cfg,
        "strategies": strategies_cfg,
        "timestamp": time.time(),
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg_snap, f, indent=2)

    all_rows = []
    for w in windows:
        rows = _run_strategies_for_window(
            series, w["start"], w["deadline"],
            amount_usd, fees, forecaster_cfg, strategies_cfg,
            save_plots, outdir
        )
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)
