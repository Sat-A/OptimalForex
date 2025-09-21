import argparse
import os
import time
import yaml
import numpy as np
import pandas as pd

from src.forex.io import load_or_build_daily
from src.forex.backtest import run_many

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiments/config.yaml")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Ignore parquet cache and rebuild daily data")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("seed", 123))

    # Load & cache daily data (fast path via parquet)
    daily = load_or_build_daily(
        path_csv=cfg["data"]["file"],
        cache_dir="data/interim",
        resample=cfg["data"].get("resample", "B"),
        force_rebuild=args.force_rebuild_cache
    )

    # Prepare output dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    exp_name = cfg.get("experiment_name", "exp")
    outdir = os.path.join(cfg["output"]["dir"], f"{ts}-{exp_name}")
    os.makedirs(outdir, exist_ok=True)

    # Run backtests
    results = run_many(
        series=daily["Rate_USD_to_GBP"],
        windows=cfg["windows"],
        amount_usd=cfg["amount_usd"],
        fees=cfg["fees"],
        forecaster_cfg=cfg["models"],
        strategies_cfg=cfg["strategies"],
        save_plots=cfg["output"].get("save_plots", True),
        outdir=outdir,
    )

    # Save raw results
    results.to_csv(os.path.join(outdir, "results.csv"), index=False)

    # Build leaderboard with richer stats (exclude perfect_foresight from the ranking view if you want)
    def pct(x): return 100.0 * x

    agg = (results
           .groupby("strategy")
           .agg(mean_gbp=("gbp_received", "mean"),
                std_gbp=("gbp_received", "std"),
                median_gbp=("gbp_received", "median"),
                mean_pct_of_pf=("pct_of_pf", "mean"),
                mean_regret_pct=("regret_pct", "mean"),
                win_rate_vs_last=("beat_last_day", "mean"),
                median_days_to_convert=("days_from_start", "median"),
                n_windows=("strategy", "count"))
           .reset_index())

    # Order by mean GBP (desc), but show % of PF too
    agg = agg.sort_values("mean_gbp", ascending=False)

    # Pretty print
    print("\n=== Leaderboard across windows ===")
    print("(win_rate_vs_last compares each strategy to converting on the last day)\n")
    # Format a subset for display
    disp = agg.copy()
    disp["mean_pct_of_pf"] = disp["mean_pct_of_pf"].map(lambda v: f"{pct(v):.1f}%")
    disp["mean_regret_pct"] = disp["mean_regret_pct"].map(lambda v: f"{pct(v):+.2f}%")
    disp["win_rate_vs_last"] = disp["win_rate_vs_last"].map(lambda v: f"{pct(v):.1f}%")
    disp["mean_gbp"] = disp["mean_gbp"].map(lambda v: f"{v:,.2f}")
    disp["std_gbp"] = disp["std_gbp"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    disp["median_gbp"] = disp["median_gbp"].map(lambda v: f"{v:,.2f}")
    disp["median_days_to_convert"] = disp["median_days_to_convert"].map(lambda v: f"{int(v)}")
    disp = disp[["strategy","n_windows","mean_gbp","std_gbp","median_gbp",
                 "mean_pct_of_pf","mean_regret_pct","win_rate_vs_last",
                 "median_days_to_convert"]]
    print(disp.to_string(index=False))

    # # Also show per-window top performer overview
    # winners = (results.loc[results.groupby(["start","deadline"])["gbp_received"].idxmax()]
    #            .groupby("strategy").size().rename("wins").reset_index()
    #            .sort_values("wins", ascending=False))
    # print("\n=== Per-window winners (count) ===")
    # print(winners.to_string(index=False))

    print(f"\nSaved outputs to: {outdir}")

if __name__ == "__main__":
    main()
