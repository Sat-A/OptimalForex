# CurrencyConvert_fixed.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# =============================
# Configuration
# =============================
CSV_FILE = "HISTDATA_COM_ASCII_GBPUSD_M12024/DAT_ASCII_GBPUSD_M1_2024.csv"  # adjust path
AMOUNT_USD = 10000
CONVERSION_COST_PERCENT = 0.2   # percentage fee on USD amount
CONVERSION_COST_FIXED = 0.0     # fixed GBP fee applied after conversion
START_DATE = "2024-06-01"
DEADLINE_DATE = "2024-08-31"

# Simulation / modelling params
MIN_HISTORY_DAYS = 30
N_PATHS = 500                # reduce/increase for speed/accuracy
RISK_AVERSION = 0.25          # lambda used for CVaR penalty
CVAR_ALPHA = 0.05             # tail for CVaR
ARIMA_ORDER = (3, 1, 0)
SEED = 42

np.random.seed(SEED)

# =============================
# Utilities
# =============================
def load_and_prepare_data(path):
    """Load CSV and produce daily close in USD->GBP (GBP per 1 USD)."""
    try:
        df = pd.read_csv(path, sep=";")
    except FileNotFoundError:
        raise
    # Rename & parse — adapt if your columns differ
    df = df.rename(columns={"DateTime Stamp": "Time", "Bar CLOSE Bid Quote": "Close"})
    df["Time"] = pd.to_datetime(df["Time"], format="%Y%m%d %H%M%S")
    df = df.sort_values("Time").set_index("Time")
    # Resample to daily at last tick (close)
    df_daily = df['Close'].resample('D').last().dropna().to_frame()
    # CSV likely has GBPUSD (price of 1 GBP in USD) so USD->GBP = 1 / GBPUSD
    df_daily['Rate_USD_to_GBP'] = 1.0 / df_daily['Close']
    return df_daily

def effective_rate(mid_rate, margin_bps=0.0):
    """Apply percentage margin (in basis points) to mid rate."""
    return mid_rate * (1.0 - margin_bps / 10000.0)

def calculate_gbp(amount_usd, rate_usd_to_gbp, conversion_cost_percent=CONVERSION_COST_PERCENT,
                  conversion_cost_fixed=CONVERSION_COST_FIXED):
    """Compute GBP received after fees.
       - percentage cost is charged on USD then converted.
       - fixed fee assumed in GBP deducted at the end."""
    fee_usd = amount_usd * (conversion_cost_percent / 100.0)
    net_usd = amount_usd - fee_usd
    gbp = net_usd * rate_usd_to_gbp - conversion_cost_fixed
    return gbp

# =============================
# Forecasting helpers
# =============================
def fit_arima(history_series, order=ARIMA_ORDER):
    """Fit ARIMA on the log-rate (stabilises variance). Returns model_fit and residuals."""
    log_hist = np.log(history_series)
    model = ARIMA(log_hist, order=order)
    fit = model.fit()
    resid = fit.resid.dropna().values  # additive residuals on log scale
    return fit, resid

def fit_ets(history_series):
    """Fit ETS (additive on log rates). Return fitted object and residuals."""
    log_hist = np.log(history_series)
    model = ExponentialSmoothing(log_hist, trend='add', seasonal=None, damped_trend=False)
    fit = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
    resid = (log_hist - fit.fittedvalues).dropna().values
    return fit, resid

# =============================
# Fixed: robust simulate_future_paths
# =============================
def simulate_future_paths(last_rate,
                          history_returns,
                          arima_fit=None, arima_resid=None,
                          ets_fit=None, ets_resid=None,
                          horizon=10, n_paths=N_PATHS):
    """
    Simulate future paths for USD->GBP rate.
    - history_returns: pandas Series of pct returns (or numpy array) on which we base bootstraps.
    - Returns array shape (n_paths, horizon) of simulated rates (not log rates).
    """
    # Normalise input types: get historical log-returns array
    if isinstance(history_returns, pd.Series):
        # history_returns already pct_change series
        hist_ret = history_returns.dropna().values
    else:
        # could be numpy array or list
        hist_ret = np.array(history_returns).ravel()

    # Convert to log-returns using log1p if we have pct returns; if empty, create tiny gaussian noise
    if hist_ret.size > 0:
        hist_log_returns = np.log1p(hist_ret)  # stable for pct returns
    else:
        hist_log_returns = np.random.normal(loc=0.0, scale=1e-4, size=500)

    # Build residual pool
    residuals_pool = np.array([], dtype=float)
    if arima_resid is not None and len(arima_resid) > 0:
        residuals_pool = np.concatenate([residuals_pool, np.array(arima_resid).ravel()])
    if ets_resid is not None and len(ets_resid) > 0:
        residuals_pool = np.concatenate([residuals_pool, np.array(ets_resid).ravel()])
    # If no model residuals available, use historical log returns as residual-like noise
    if residuals_pool.size == 0:
        residuals_pool = hist_log_returns.copy()

    # final fallback
    if residuals_pool.size == 0:
        residuals_pool = np.random.normal(loc=0.0, scale=1e-4, size=1000)

    sims = np.zeros((n_paths, horizon))
    last_log = np.log(last_rate)

    for p in range(n_paths):
        current_log = last_log
        for h in range(horizon):
            # compute model drift estimate (difference between 1-step forecast and current_log)
            drifts = []
            if arima_fit is not None:
                try:
                    arima_fc = arima_fit.get_forecast(steps=1)
                    # predicted_mean is on log scale (we fitted log rates)
                    pred = arima_fc.predicted_mean.iloc[-1]
                    drifts.append(pred - current_log)
                except Exception:
                    pass
            if ets_fit is not None:
                try:
                    ets_fc = ets_fit.forecast(1)
                    pred = ets_fc[-1]
                    drifts.append(pred - current_log)
                except Exception:
                    pass

            model_drift = np.mean(drifts) if len(drifts) > 0 else 0.0

            # sample components
            res = np.random.choice(residuals_pool)
            boot_log_ret = np.random.choice(hist_log_returns) if hist_log_returns.size > 0 else 0.0

            # combine: weights chosen to blend model drift and historical returns + residual noise
            next_log = current_log + 0.6 * model_drift + 0.4 * boot_log_ret + 0.6 * res
            sims[p, h] = np.exp(next_log)
            current_log = next_log

    return sims

# =============================
# Fixed: robust optimal stopping approximation
# =============================
def compute_optimal_stopping_policy(df, start_date, deadline_date,
                                    amount_usd=AMOUNT_USD,
                                    conversion_cost_percent=CONVERSION_COST_PERCENT,
                                    conversion_cost_fixed=CONVERSION_COST_FIXED,
                                    n_paths=N_PATHS,
                                    min_history=MIN_HISTORY_DAYS,
                                    risk_aversion=RISK_AVERSION,
                                    cvar_alpha=CVAR_ALPHA):
    """
    For each date t in the decision window, compute whether to SEND or WAIT using Monte Carlo approx.
    Returns a DataFrame indexed by date with columns: rate, send_now_gbp, continuation_gbp, decision.
    """
    dates = pd.date_range(start=start_date, end=deadline_date, freq='D')
    rows = []
    df_all = df.copy()

    for today in dates:
        if today not in df_all.index:
            # skip marketless days (holidays); alternatively you could forward-fill
            continue

        # prepare history up to today (inclusive)
        history = df_all.loc[:today, "Rate_USD_to_GBP"]

        # get model fits if enough history
        arima_fit = None; arima_resid = None; ets_fit = None; ets_resid = None
        if len(history) >= min_history:
            try:
                arima_fit, arima_resid = fit_arima(history)
            except Exception:
                arima_fit = None; arima_resid = None
            try:
                ets_fit, ets_resid = fit_ets(history)
            except Exception:
                ets_fit = None; ets_resid = None

        today_rate = history.iloc[-1]
        send_now_gbp = calculate_gbp(amount_usd, today_rate,
                                     conversion_cost_percent, conversion_cost_fixed)

        # terminal day: must send
        if today == pd.to_datetime(deadline_date):
            rows.append({
                'date': today, 'rate': today_rate,
                'send_now_gbp': send_now_gbp, 'continuation_gbp': np.nan,
                'decision': 'SEND (deadline)'
            })
            break

        # days remaining
        days_remaining = (pd.to_datetime(deadline_date) - today).days
        if days_remaining <= 0:
            rows.append({
                'date': today, 'rate': today_rate,
                'send_now_gbp': send_now_gbp, 'continuation_gbp': 0.0,
                'decision': 'SEND'
            })
            continue

        # build historical returns series (pct change) to pass as Series
        hist_rates = df_all.loc[:today, "Rate_USD_to_GBP"]
        hist_returns_series = hist_rates.pct_change().dropna()
        # pass series (simulate will accept both series and arrays)
        sims = simulate_future_paths(
            last_rate=today_rate,
            history_returns=hist_returns_series,
            arima_fit=arima_fit, arima_resid=arima_resid,
            ets_fit=ets_fit, ets_resid=ets_resid,
            horizon=days_remaining,
            n_paths=n_paths
        )

        # For each simulated path compute the pathwise best GBP value (if you could pick best day)
        # Convert simulated rates to GBP after fees
        # sims shape: (n_paths, days_remaining)
        path_gbp_matrix = np.apply_along_axis(
            lambda row: calculate_gbp(amount_usd, row, conversion_cost_percent, conversion_cost_fixed),
            axis=1, arr=sims
        )  # shape (n_paths, days_remaining)

        per_path_best = path_gbp_matrix.max(axis=1)  # best future outcome per path

        # compute continuation: mean minus CVaR-style downside penalty
        cont_mean = per_path_best.mean()
        tail_quantile = np.quantile(per_path_best, cvar_alpha)
        tail = per_path_best[per_path_best <= tail_quantile]
        cvar = tail.mean() if tail.size > 0 else tail_quantile
        # risk-adjustment: penalise downside by risk_aversion * (mean - CVaR)
        continuation_value = cont_mean - risk_aversion * (cont_mean - cvar)

        decision = 'SEND' if send_now_gbp >= continuation_value else 'WAIT'

        rows.append({
            'date': today, 'rate': today_rate,
            'send_now_gbp': send_now_gbp,
            'continuation_gbp': continuation_value,
            'decision': decision
        })

    decisions_df = pd.DataFrame(rows).set_index('date')
    return decisions_df

# =============================
# Baseline strategies (same as before)
# =============================
def run_first_day(df, start_date, deadline_date):
    s = df.loc[start_date:deadline_date]
    d = s.index[0]
    rate = s.loc[d, 'Rate_USD_to_GBP']
    return {'name': 'First Day', 'date': d, 'rate': rate,
            'gbp': calculate_gbp(AMOUNT_USD, rate)}

def run_last_day(df, start_date, deadline_date):
    s = df.loc[start_date:deadline_date]
    d = s.index[-1]
    rate = s.loc[d, 'Rate_USD_to_GBP']
    return {'name': 'Last Day', 'date': d, 'rate': rate,
            'gbp': calculate_gbp(AMOUNT_USD, rate)}

def run_perfect_foresight(df, start_date, deadline_date):
    s = df.loc[start_date:deadline_date]
    idx = s['Rate_USD_to_GBP'].idxmax()
    rate = s.loc[idx, 'Rate_USD_to_GBP']
    return {'name': 'Perfect Foresight', 'date': idx, 'rate': rate,
            'gbp': calculate_gbp(AMOUNT_USD, rate)}

def run_historical_avg_trigger(df, start_date, deadline_date):
    s = df.loc[start_date:deadline_date]
    for d in s.index:
        hist = df.loc[:d, 'Rate_USD_to_GBP']
        if len(hist) < 2:
            continue
        live = hist.iloc[-1]
        avg = hist.iloc[:-1].mean()
        if live > avg:
            return {'name': 'Hist Avg Trigger', 'date': d, 'rate': live,
                    'gbp': calculate_gbp(AMOUNT_USD, live)}
    # fallback
    return run_last_day(df, start_date, deadline_date)

def run_arima_one_step(df, start_date, deadline_date, order=ARIMA_ORDER):
    s = df.loc[start_date:deadline_date]
    for d in s.index:
        hist = df.loc[:d, 'Rate_USD_to_GBP']
        if len(hist) < MIN_HISTORY_DAYS:
            continue
        try:
            fit, _ = fit_arima(hist, order=order)
            # predicted_mean is log-rate; exponentiate
            fc_log = fit.get_forecast(steps=1).predicted_mean.iloc[-1]
            fc = np.exp(fc_log)
            live = hist.iloc[-1]
            if live > fc:
                return {'name': 'ARIMA one-step', 'date': d, 'rate': live,
                        'gbp': calculate_gbp(AMOUNT_USD, live)}
        except Exception:
            continue
    return run_last_day(df, start_date, deadline_date)

# =============================
# Plot helpers and metrics
# =============================
def plot_decisions_and_rates(df, decisions_df, start_date, deadline_date, strategy_points):
    fig, ax = plt.subplots(figsize=(14,6))
    plot_df = df.loc[start_date:deadline_date]
    ax.plot(plot_df.index, plot_df['Rate_USD_to_GBP'], linewidth=2, alpha=0.8, label='Daily rate (USD->GBP)')

    # plot decisions (SEND days)
    send_days = decisions_df[decisions_df['decision'].str.startswith('SEND')]
    ax.scatter(send_days.index, send_days['rate'], marker='X', s=120, color='tab:orange', edgecolor='k', label='Policy SEND(s)')

    marker_styles = {'First Day':'o', 'Last Day':'s', 'Perfect Foresight':'*', 'Hist Avg Trigger':'^', 'ARIMA one-step':'D', 'MC Optimal Stopping':'P'}
    for p in strategy_points:
        ax.scatter(p['date'], p['rate'], marker=marker_styles.get(p['name'],'o'), s=140, label=f"{p['name']} ({p['date'].date()})", edgecolor='k')

    ax.set_title(f"Rates and Conversion Decisions {start_date} → {deadline_date}", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Rate (GBP per 1 USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def print_results_table(points):
    df = pd.DataFrame(points)
    df['date'] = df['date'].dt.date
    df = df[['name','date','rate','gbp']]
    df = df.sort_values('gbp', ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))

# =============================
# Main orchestration
# =============================
if __name__ == "__main__":
    df = load_and_prepare_data(CSV_FILE)
    df_window = df.loc[START_DATE:DEADLINE_DATE].copy()

    decisions = compute_optimal_stopping_policy(df, START_DATE, DEADLINE_DATE,
                                               amount_usd=AMOUNT_USD,
                                               conversion_cost_percent=CONVERSION_COST_PERCENT,
                                               conversion_cost_fixed=CONVERSION_COST_FIXED,
                                               n_paths=N_PATHS,
                                               min_history=MIN_HISTORY_DAYS,
                                               risk_aversion=RISK_AVERSION,
                                               cvar_alpha=CVAR_ALPHA)

    # choose first SEND recommended
    send_rows = decisions[decisions['decision'].str.startswith('SEND')]
    if not send_rows.empty:
        chosen_date = send_rows.index[0]
        chosen_rate = send_rows.loc[chosen_date, 'rate']
        chosen_gbp = send_rows.loc[chosen_date, 'send_now_gbp']
        policy_point = {'name':'MC Optimal Stopping', 'date': chosen_date, 'rate': chosen_rate, 'gbp': chosen_gbp}
    else:
        tmp = run_last_day(df, START_DATE, DEADLINE_DATE)
        policy_point = {'name':'MC Optimal Stopping (fallback last)', 'date': tmp['date'], 'rate': tmp['rate'], 'gbp': tmp['gbp']}

    baselines = [
        run_perfect_foresight(df, START_DATE, DEADLINE_DATE),
        run_first_day(df, START_DATE, DEADLINE_DATE),
        run_last_day(df, START_DATE, DEADLINE_DATE),
        run_historical_avg_trigger(df, START_DATE, DEADLINE_DATE),
        run_arima_one_step(df, START_DATE, DEADLINE_DATE)
    ]

    strategy_points = baselines + [policy_point]

    print("\n=== Strategy comparison (GBP received) ===\n")
    print_results_table(strategy_points)

    plot_decisions_and_rates(df, decisions, START_DATE, DEADLINE_DATE, strategy_points)

    print("\n=== Decision log (sample) ===")
    display_cols = ['rate','send_now_gbp','continuation_gbp','decision']
    print(decisions[display_cols].head(20).to_string())

    perfect_gbp = baselines[0]['gbp']
    print("\n=== Performance vs perfect foresight ===")
    for p in strategy_points:
        diff = p['gbp'] - perfect_gbp
        print(f"{p['name']}: GBP {p['gbp']:.2f} (diff vs perfect: {diff:.2f})")
