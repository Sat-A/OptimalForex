import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================
# Configuration
# =============================
CSV_FILE = r"HISTDATA_COM_ASCII_GBPUSD_M12024\DAT_ASCII_GBPUSD_M1_2024.csv"
AMOUNT_USD = 10000
CONVERSION_COST_PERCENT = 0.2  # applied once when converting
CONVERSION_COST_FIXED = 0
START_DATE = "2024-02-01"
DEADLINE_DATE = "2024-12-31"

# Optimal-stopping / forecasting config
MIN_HISTORY_DAYS = 30              # minimum history before we trust forecasts
FORECAST_SAMPLES = 2000            # Monte-Carlo samples for next-day rate
EWMA_LAMBDA = 0.94                 # EWMA volatility decay (RiskMetrics-style)
DRIFT_WINDOW = 60                  # window (days) to estimate drift; shrinks toward 0
USE_CVAR = True                   # set True for risk-aware thresholds
CVAR_ALPHA = 0.10                  # worst 10% tail if CVaR enabled
RNG_SEED = 123

np.random.seed(RNG_SEED)

# =============================
# Utilities
# =============================
def load_and_prepare_data(path):
    """Loads minute data (semicolon separated), converts to daily close and USD->GBP rate."""
    try:
        df = pd.read_csv(path, sep=";")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return None

    # Normalize expected columns from your sample
    df = df.rename(columns={
        "DateTime Stamp": "Time",
        "Bar CLOSE Bid Quote": "Close",
        "Bar OPEN Bid Quote": "Open",
        "Bar HIGH Bid Quote": "High",
        "Bar LOW Bid Quote": "Low",
    })
    # Parse times like 20240101 170000
    df["Time"] = pd.to_datetime(df["Time"], format="%Y%m%d %H%M%S")
    df = df.sort_values("Time").set_index("Time")

    # Aggregate to daily last close (GBPUSD), then compute USD->GBP
    daily = df["Close"].resample("B").last().dropna().to_frame()  # Business days
    daily["Rate_USD_to_GBP"] = 1.0 / daily["Close"]
    daily = daily.drop(columns=["Close"])
    daily.index.name = "Date"
    return daily

def calculate_gbp(amount_usd, rate_usd_to_gbp):
    """Final GBP amount after fees."""
    fee = (amount_usd * CONVERSION_COST_PERCENT / 100.0) + CONVERSION_COST_FIXED
    net_usd = amount_usd - fee
    return net_usd * rate_usd_to_gbp

def realized_log_returns(series):
    """Compute daily log returns of the USD->GBP rate."""
    return np.log(series).diff().dropna()

def ewma_vol(log_rets, lam=EWMA_LAMBDA):
    """EWMA volatility estimate (annualization not needed for one-step)."""
    # If too short, fallback to simple std
    if len(log_rets) == 0:
        return 0.0
    w = np.array([(1 - lam) * lam ** i for i in range(len(log_rets)-1, -1, -1)])
    w /= w.sum()
    mu = np.sum(w * log_rets.values)
    var = np.sum(w * (log_rets.values - mu) ** 2)
    return np.sqrt(max(var, 0.0))

def shrink_to_zero(x, shrink=0.5):
    """Simple shrinkage of drift toward 0 to avoid overfitting."""
    return (1 - shrink) * x

def sample_next_rate(last_rate, hist_rates, n_samples=FORECAST_SAMPLES,
                     drift_window=DRIFT_WINDOW, lam=EWMA_LAMBDA):
    """
    Distributional forecaster:
    - Random walk in log space with EWMA vol and small drift estimated from recent window.
    - Returns Monte-Carlo samples of next day's USD->GBP rate.
    """
    hist = hist_rates.dropna()
    if len(hist) < 2:
        return np.full(n_samples, last_rate)

    log_rets = realized_log_returns(hist)
    if len(log_rets) == 0:
        return np.full(n_samples, last_rate)

    vol = ewma_vol(log_rets, lam=lam)
    if np.isnan(vol) or vol == 0:
        vol = max(1e-6, log_rets.std())

    # Drift estimate (shrink toward 0 because FX ~ random walk)
    if len(log_rets) >= drift_window:
        mu_hat = log_rets.tail(drift_window).mean()
    else:
        mu_hat = log_rets.mean()
    mu_hat = shrink_to_zero(mu_hat, shrink=0.75)

    # Monte-Carlo one-day-ahead samples in log space
    eps = np.random.normal(loc=mu_hat, scale=vol, size=n_samples)
    samples = last_rate * np.exp(eps)
    return samples

def risk_measure(x, use_cvar=USE_CVAR, alpha=CVAR_ALPHA):
    """
    Risk functional for continuation value:
    - mean (risk-neutral) or CVaR_alpha of the distribution provided in x.
    """
    x = np.asarray(x)
    if not use_cvar:
        return float(np.mean(x))
    # CVaR of the *distribution of continuation value*
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return float(np.mean(tail)) if len(tail) > 0 else float(q)

# =============================
# Baseline Strategies (yours)
# =============================
def run_perfect_foresight_strategy(df, start_date, deadline_date):
    period_df = df.loc[start_date:deadline_date]
    best_day = period_df['Rate_USD_to_GBP'].idxmax()
    best_rate = period_df['Rate_USD_to_GBP'].max()
    return {"name": "Perfect Foresight", "date": best_day, "rate": best_rate}

def run_first_day_strategy(df, start_date, deadline_date):
    first_day = df.loc[start_date:deadline_date].index[0]
    first_rate = df.loc[first_day, "Rate_USD_to_GBP"]
    return {"name": "First Day Conversion", "date": first_day, "rate": first_rate}

def run_last_day_strategy(df, start_date, deadline_date):
    last_day = df.loc[start_date:deadline_date].index[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "Last Day Conversion", "date": last_day, "rate": last_rate}

def run_historical_average_strategy(df, start_date, deadline_date):
    decision_days = df.loc[start_date:deadline_date].index
    for current_day in decision_days:
        hist = df.loc[:current_day, "Rate_USD_to_GBP"]
        if len(hist) < 2:
            continue
        live_rate = hist.iloc[-1]
        historical_avg = hist.iloc[:-1].mean()  # strictly before today
        if live_rate > historical_avg:
            return {"name": "Historical Average", "date": current_day, "rate": live_rate}
    last_day = decision_days[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "Historical Average", "date": last_day, "rate": last_rate}

# =============================
# ARIMA Trigger (your original, kept for comparison)
# =============================
def run_arima_strategy(df, start_date, deadline_date, order=(1, 0, 0), min_history=30):
    decision_days = df.loc[start_date:deadline_date].index
    for current_day in decision_days:
        history = df.loc[:current_day, "Rate_USD_to_GBP"]
        if len(history) < min_history:
            continue
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1).iloc[0]
            live_rate = history.iloc[-1]
            if live_rate > forecast:
                return {"name": "ARIMA Forecast", "date": current_day, "rate": live_rate}
        except Exception:
            continue
    last_day = decision_days[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "ARIMA Forecast", "date": last_day, "rate": last_rate}

# =============================
# NEW: Optimal-Stopping Threshold Policy
# =============================
def compute_thresholds(df, start_date, deadline_date,
                       min_history_days=MIN_HISTORY_DAYS,
                       n_samples=FORECAST_SAMPLES,
                       lam=EWMA_LAMBDA,
                       use_cvar=USE_CVAR,
                       alpha=CVAR_ALPHA):
    """
    Backward induction for thresholds b_t on each decision day (business days).
    We use a one-step predictive distribution:
        V_{t+1} = max( R_{t+1}, b_{t+1} ) with b_T = -inf (must convert at T).
    b_t = RiskFunctional( V_{t+1} ), where RiskFunctional = mean or CVaR.
    """
    period = df.loc[start_date:deadline_date].copy()
    days = period.index
    rates = df["Rate_USD_to_GBP"]

    T = len(days) - 1
    if T < 0:
        raise ValueError("No decision days in the selected window.")

    b = pd.Series(index=days, dtype=float)
    # At deadline T: always convert, threshold = -inf (ensures trigger)
    b.iloc[T] = -np.inf

    # Work backward: t = T-1 ... 0
    cont_next = None  # we'll derive from b_{t+1}
    for i in range(T - 1, -1, -1):
        t = days[i]
        # Historical data available up to day t
        hist = rates.loc[:t]
        # If not enough history, fall back to simple heuristic: threshold = current rate
        if len(hist) < min_history_days:
            # This prevents premature conversion in the forward pass
            b.iloc[i] = np.inf  # "don't convert yet"
            continue

        last_rate = hist.iloc[-1]

        # Sample R_{t+1} from our distributional forecaster based on history
        r_next_samples = sample_next_rate(
            last_rate=last_rate,
            hist_rates=hist,
            n_samples=n_samples,
            drift_window=DRIFT_WINDOW,
            lam=lam,
        )

        # Continuation at t+1 is max(R_{t+1}, b_{t+1})
        b_next = b.iloc[i + 1] if np.isfinite(b.iloc[i + 1]) else -np.inf
        cont_samples = np.maximum(r_next_samples, b_next)

        # Risk functional gives the threshold b_t
        b.iloc[i] = risk_measure(cont_samples, use_cvar=use_cvar, alpha=alpha)

    return b

def run_optimal_stopping_strategy(df, start_date, deadline_date, **kwargs):
    """
    Forward pass: convert on first day R_t >= b_t, else convert on deadline.
    """
    thresholds = compute_thresholds(df, start_date, deadline_date, **kwargs)
    period = df.loc[start_date:deadline_date]
    for t, row in period.iterrows():
        r = row["Rate_USD_to_GBP"]
        b_t = thresholds.loc[t]
        if r >= b_t:
            return {"name": f"Optimal Stopping ({'CVaR' if USE_CVAR else 'Mean'})",
                    "date": t, "rate": r, "threshold": b_t, "thresholds": thresholds}
    # Fallback: last day
    t = period.index[-1]
    r = period.iloc[-1]["Rate_USD_to_GBP"]
    return {"name": f"Optimal Stopping ({'CVaR' if USE_CVAR else 'Mean'})",
            "date": t, "rate": r, "threshold": thresholds.loc[t], "thresholds": thresholds}

# =============================
# Plotting
# =============================
def plot_strategy_comparison(df, results, start_date, deadline_date):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    plot_df = df.loc[start_date:deadline_date]
    ax.plot(plot_df.index, plot_df["Rate_USD_to_GBP"], color='#555', alpha=0.8, linewidth=2, label="Daily Rate (USD→GBP)")

    marker_styles = ['*', 'o', 's', '^', 'v', 'D', 'P', 'X', 'h', '8']
    colors = plt.cm.tab10.colors

    for idx, r in enumerate(results):
        name, date, rate = r['name'], r['date'], r['rate']
        marker = marker_styles[idx % len(marker_styles)]
        color = colors[idx % len(colors)]
        label_text = f"{name}: {rate:.5f} on {date.strftime('%b %d')}"
        ax.scatter(date, rate, marker=marker, s=160, color=color, zorder=5, edgecolors='black',
                   label=label_text)

        # If this is the optimal-stopping strategy, optionally draw thresholds
        if "thresholds" in r:
            th = r["thresholds"].copy()
            ax.plot(th.index, th.values, linestyle='--', linewidth=1.5, alpha=0.9,
                    label=f"{name} Threshold")

    ax.set_title(f"Conversion Strategies ({start_date} → {deadline_date})", fontsize=18, fontweight='bold', pad=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Rate (GBP per 1 USD)", fontsize=14)
    ax.legend(loc="best", fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, which='major', linestyle='--', linewidth=0.6)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()

# =============================
# Main
# =============================
if __name__ == "__main__":
    df_daily = load_and_prepare_data(CSV_FILE)
    if df_daily is None or df_daily.empty:
        raise SystemExit

    # Run strategies
    results = [
        run_perfect_foresight_strategy(df_daily, START_DATE, DEADLINE_DATE),
        run_first_day_strategy(df_daily, START_DATE, DEADLINE_DATE),
        run_last_day_strategy(df_daily, START_DATE, DEADLINE_DATE),
        run_historical_average_strategy(df_daily, START_DATE, DEADLINE_DATE),
        run_arima_strategy(df_daily, START_DATE, DEADLINE_DATE, order=(1,0,0), min_history=MIN_HISTORY_DAYS),
        run_optimal_stopping_strategy(df_daily, START_DATE, DEADLINE_DATE,
                                      min_history_days=MIN_HISTORY_DAYS,
                                      n_samples=FORECAST_SAMPLES,
                                      lam=EWMA_LAMBDA,
                                      use_cvar=USE_CVAR,
                                      alpha=CVAR_ALPHA),
    ]

    # Calculate GBP received
    benchmark_gbp = 0.0
    for r in results:
        r['gbp_received'] = calculate_gbp(AMOUNT_USD, r['rate'])
        if r['name'] == 'Perfect Foresight':
            benchmark_gbp = r['gbp_received']
    for r in results:
        r['difference'] = r['gbp_received'] - benchmark_gbp

    # Results table
    results_df = pd.DataFrame(results)
    results_df['date'] = pd.to_datetime(results_df['date']).dt.date
    cols = ['name', 'date', 'rate', 'gbp_received', 'difference']
    print("=== Strategy Comparison Results ===")
    print(f"Converting ${AMOUNT_USD:,.2f} to GBP between {START_DATE} and {DEADLINE_DATE}\n")
    print(results_df[cols].sort_values('gbp_received', ascending=False).to_string(index=False))

    # Plot
    plot_strategy_comparison(df_daily, results, START_DATE, DEADLINE_DATE)
