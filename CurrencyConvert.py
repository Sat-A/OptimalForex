import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =============================
# Configuration
# =============================
CSV_FILE = "HISTDATA_COM_ASCII_GBPUSD_M12024\DAT_ASCII_GBPUSD_M1_2024.csv"
AMOUNT_USD = 10000
CONVERSION_COST_PERCENT = 0.2
CONVERSION_COST_FIXED = 0
START_DATE = "2024-06-01"
DEADLINE_DATE = "2024-12-01"

# =============================
# Setup Functions (from previous step)
# =============================
def load_and_prepare_data(path):
    try:
        df = pd.read_csv(path, sep=";")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return None
    df = df.rename(columns={"DateTime Stamp": "Time", "Bar CLOSE Bid Quote": "Close"})
    df["Time"] = pd.to_datetime(df["Time"], format="%Y%m%d %H%M%S")
    df = df.sort_values("Time").set_index("Time")
    df_daily = df['Close'].resample('D').last().dropna().to_frame()
    df_daily['Rate_USD_to_GBP'] = 1 / df_daily['Close']
    return df_daily

def calculate_gbp(amount_usd, rate_usd_to_gbp):
    fee = (amount_usd * CONVERSION_COST_PERCENT / 100.0) + CONVERSION_COST_FIXED
    net_usd = amount_usd - fee
    return net_usd * rate_usd_to_gbp

# =============================
# Strategy 1: Perfect Foresight (The Benchmark)
# =============================
def run_perfect_foresight_strategy(df, start_date, deadline_date):
    period_df = df.loc[start_date:deadline_date]
    best_day = period_df['Rate_USD_to_GBP'].idxmax()
    best_rate = period_df['Rate_USD_to_GBP'].max()
    return {"name": "Perfect Foresight", "date": best_day, "rate": best_rate}

# =============================
# Strategy 2: ARIMA Forecast (Original Model)
# =============================
def forecast_rates(hist, horizon_steps):
    try:
        model = ARIMA(hist, order=(1, 1, 1)).fit()
        forecast = model.get_forecast(steps=horizon_steps)
        return forecast.predicted_mean, forecast.conf_int()
    except Exception:
        last_val = hist.iloc[-1]
        future_index = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_steps, freq='D')
        return pd.Series([last_val] * horizon_steps, index=future_index), pd.DataFrame({f"lower {hist.name}": [last_val] * horizon_steps, f"upper {hist.name}": [last_val] * horizon_steps}, index=future_index)

def run_arima_strategy(df, start_date, deadline_date):
    decision_days = df.loc[start_date:deadline_date].index
    for current_day in decision_days:
        hist = df.loc[:current_day, "Rate_USD_to_GBP"]
        live_rate = hist.iloc[-1]
        horizon = len(df.loc[current_day:deadline_date]) - 1 or 1
        forecast_mean, forecast_ci = forecast_rates(hist, horizon)
        
        trigger_threshold = forecast_mean.mean() + 0.75 * (forecast_ci.iloc[:, 1].max() - forecast_mean.mean())
        if live_rate >= trigger_threshold:
            return {"name": "ARIMA Forecast", "date": current_day, "rate": live_rate}
            
    # If never triggered, convert on the last day
    last_day = decision_days[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "ARIMA Forecast", "date": last_day, "rate": last_rate}

# =============================
# Strategy 3 & 4: First Day / Last Day
# =============================
def run_first_day_strategy(df, start_date, deadline_date):
    first_day = df.loc[start_date:deadline_date].index[0]
    first_rate = df.loc[first_day, "Rate_USD_to_GBP"]
    return {"name": "First Day Conversion", "date": first_day, "rate": first_rate}

def run_last_day_strategy(df, start_date, deadline_date):
    last_day = df.loc[start_date:deadline_date].index[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "Last Day Conversion", "date": last_day, "rate": last_rate}

# =============================
# Strategy 5: Historical Average
# =============================
def run_historical_average_strategy(df, start_date, deadline_date):
    decision_days = df.loc[start_date:deadline_date].index
    for current_day in decision_days:
        hist = df.loc[:current_day, "Rate_USD_to_GBP"]
        live_rate = hist.iloc[-1]
        historical_avg = hist.iloc[:-1].mean() # Average of all data *before* today
        if live_rate > historical_avg:
            return {"name": "Historical Average", "date": current_day, "rate": live_rate}
    
    # If never triggered, convert on the last day
    last_day = decision_days[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "Historical Average", "date": last_day, "rate": last_rate}

# =============================
# Plotting
# =============================
def plot_strategy_comparison(df, results, start_date, deadline_date):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    period_df = df.loc[start_date:deadline_date]
    ax.plot(period_df.index, period_df["Rate_USD_to_GBP"], color='grey', alpha=0.8, label="Daily Rate")

    markers = {'Perfect Foresight': ('*', 200, 'gold'), 'ARIMA Forecast': ('o', 100, 'crimson'), 
               'First Day Conversion': ('^', 100, 'forestgreen'), 'Last Day Conversion': ('v', 100, 'darkorange'),
               'Historical Average': ('s', 100, 'royalblue')}
    
    for r in results:
        name, date, rate = r['name'], r['date'], r['rate']
        marker, size, color = markers[name]
        ax.scatter(date, rate, marker=marker, s=size, color=color, zorder=5, edgecolors='black', label=f"{name}: {rate:.5f}")

    ax.set_title(f"Comparison of Conversion Strategies [{start_date} to {deadline_date}]", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rate (GBP per 1 USD)", fontsize=12)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
    plt.show()

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    df_daily = load_and_prepare_data(CSV_FILE)

    if df_daily is not None:
        # Run all strategies
        results = [
            run_perfect_foresight_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_arima_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_first_day_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_last_day_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_historical_average_strategy(df_daily, START_DATE, DEADLINE_DATE)
        ]

        # Calculate GBP received for each strategy
        benchmark_gbp = 0
        for r in results:
            r['gbp_received'] = calculate_gbp(AMOUNT_USD, r['rate'])
            if r['name'] == 'Perfect Foresight':
                benchmark_gbp = r['gbp_received']
        
        for r in results:
            r['difference'] = r['gbp_received'] - benchmark_gbp
            
        # Display results in a table
        results_df = pd.DataFrame(results)
        results_df['date'] = results_df['date'].dt.date
        print("=== Strategy Comparison Results ===")
        print(results_df[['name', 'date', 'rate', 'gbp_received', 'difference']].sort_values('gbp_received', ascending=False).to_string(index=False))

        # Plot the comparison
        plot_strategy_comparison(df_daily, results, START_DATE, DEADLINE_DATE)