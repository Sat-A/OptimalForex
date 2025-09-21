import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_window(series: pd.Series, start: str, deadline: str, decisions, outdir: str):
    seg = series.loc[start:deadline]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seg.index, seg.values, linewidth=2, label="USD→GBP rate", alpha=0.8)

    # Mark decisions
    markers = {"perfect_foresight": "*", "first_day": "o", "last_day": "s",
               "historical_average": "^", "arima_trigger": "D", "optimal_stopping": "X"}
    for d in decisions:
        m = markers.get(d.name, "o")
        ax.scatter(d.date, d.rate, s=140, marker=m, edgecolors="black", zorder=5, label=d.name)

        # Optional thresholds for optimal stopping
        if d.name == "optimal_stopping" and isinstance(d.extra.get("thresholds"), pd.Series):
            th = d.extra["thresholds"]
            ax.plot(th.index, th.values, linestyle="--", linewidth=1.5, label="threshold")

    ax.set_title(f"Window {start} → {deadline}")
    ax.set_xlabel("Date")
    ax.set_ylabel("GBP per 1 USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    fig.savefig(os.path.join(outdir, "plots", f"{start}_to_{deadline}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
