import argparse
import os
import time
import yaml
import numpy as np
import pandas as pd

from src.forex.io import load_minute_csv, to_daily_usd_to_gbp
from src.forex.backtest import run_many

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiments/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("seed", 123))

    # Load & prepare data
    df_min = load_minute_csv(cfg["data"]["file"])
    series = to_daily_usd_to_gbp(df_min, resample=cfg["data"].get("resample", "B"))

    # Prepare output dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    exp_name = cfg.get("experiment_name", "exp")
    outdir = os.path.join(cfg["output"]["dir"], f"{ts}-{exp_name}")
    os.makedirs(outdir, exist_ok=True)

    # Run backtests
    results = run_many(
        series=series["Rate_USD_to_GBP"],
        windows=cfg["windows"],
        amount_usd=cfg["amount_usd"],
        fees=cfg["fees"],
        forecaster_cfg=cfg["models"],
        strategies_cfg=cfg["strategies"],
        save_plots=cfg["output"].get("save_plots", True),
        outdir=outdir,
    )

    # Save & print summary
    results.to_csv(os.path.join(outdir, "results.csv"), index=False)
    leaderboard = (
        results.groupby("strategy")["gbp_received"]
        .mean()
        .sort_values(ascending=False)
        .to_frame()
    )
    print("\nLeaderboard (mean GBP across windows):\n")
    print(leaderboard.to_string())
    print(f"\nSaved results to: {outdir}")

if __name__ == "__main__":
    main()
