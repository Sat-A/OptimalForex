"""
USD → GBP Conversion Optimiser
Realistic simulation with rolling forecast + plots
(Corrected and Improved Version)
"""

# =============================
# Imports
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =============================
# Configuration / Inputs
# =============================
# Path to the CSV file provided by the user.
# IMPORTANT: Replace with the actual path to your file.
# Example: "C:/Users/YourUser/Downloads/DAT_ASCII_GBPUSD_M1_2024.csv"
CSV_FILE = "CurrencyConversion/HISTDATA_COM_ASCII_GBPUSD_M12024/DAT_ASCII_GBPUSD_M1_2024.csv"

AMOUNT_USD = 10000
CONVERSION_COST_PERCENT = 0.2
CONVERSION_COST_FIXED = 0

START_DATE = "2024-06-01"    # When you start watching
DEADLINE_DATE = "2024-06-20" # Latest conversion date

# =============================
# Load and Prepare Data
# =============================
def load_and_prepare_data(path):
    """
    Loads minute-level data, standardises it, resamples to daily frequency,
    and calculates the correct conversion rate (USD to GBP).
    """
    try:
        df = pd.read_csv(path, sep=";")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        print("Please update the CSV_FILE variable with the correct path.")
        return None

    # Standardise column names
    df = df.rename(columns={
        "DateTime Stamp": "Time",
        "Bar OPEN Bid Quote": "Open",
        "Bar HIGH Bid Quote": "High",
        "Bar LOW Bid Quote": "Low",
        "Bar CLOSE Bid Quote": "Close",
        "Volume": "Volume"
    })

    # Parse datetime and sort
    df["Time"] = pd.to_datetime(df["Time"], format="%Y%m%d %H%M%S")
    df = df.sort_values("Time").set_index("Time")

    # **IMPROVEMENT 1: Resample to daily data**
    # The decision is daily, so we should use daily data.
    # We take the last closing price of each day.
    # .dropna() removes non-trading days (weekends, holidays).
    df_daily = df['Close'].resample('D').last().dropna().to_frame()

    # **IMPROVEMENT 2: Use the correct rate for USD -> GBP conversion**
    # The 'Close' column is GBP/USD (how many USD for 1 GBP).
    # To convert USD to GBP, we need the inverse rate (1 / Close).
    df_daily['Rate_USD_to_GBP'] = 1 / df_daily['Close']

    return df_daily

# =============================
# Cost Calculation
# =============================
def calculate_gbp(amount_usd, rate_usd_to_gbp, cost_percent, cost_fixed):
    """Calculates the final GBP amount after costs."""
    fee = (amount_usd * cost_percent / 100.0) + cost_fixed
    net_usd = amount_usd - fee
    # We multiply by the USD_to_GBP rate.
    return net_usd * rate_usd_to_gbp

# =============================
# Forecasting
# =============================
def forecast_rates(hist, horizon_steps):
    """
    Forecasts future rates using an ARIMA model.
    Falls back to a simple last-value forecast if ARIMA fails.
    """
    try:
        # A simple ARIMA(p,d,q) model. (1,1,1) is a common starting point.
        model = ARIMA(hist, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=horizon_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        return forecast_mean, forecast_ci
    except Exception:
        # Fallback if ARIMA fails (e.g., too little data)
        last_val = hist.iloc[-1]
        # Create a compatible forecast structure
        future_index = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_steps, freq='D')
        forecast_mean = pd.Series([last_val] * horizon_steps, index=future_index)
        forecast_ci = pd.DataFrame({
            f"lower {hist.name}": [last_val] * horizon_steps,
            f"upper {hist.name}": [last_val] * horizon_steps
        }, index=future_index)
        return forecast_mean, forecast_ci

# =============================
# Strategy
# =============================
def should_convert(live_rate, forecast_mean, forecast_ci):
    """
    Decision logic: convert if the current rate is exceptionally high
    compared to the forecast.
    """
    # The trigger is set at 75% of the way towards the maximum expected rate in the forecast.
    # This is a strategy to capture unusually good rates without being too greedy.
    trigger_threshold = forecast_mean.mean() + 0.75 * (forecast_ci.iloc[:, 1].max() - forecast_mean.mean())

    if live_rate >= trigger_threshold:
        return True, trigger_threshold # Convert now
    else:
        return False, trigger_threshold # Wait

# =============================
# Simulation
# =============================
def run_simulation(df, start_date, deadline_date):
    """
    Runs the day-by-day simulation to decide when to convert.
    """
    # **FIX 1: Iterate over actual trading days in the decision window**
    decision_days = df.loc[start_date:deadline_date].index
    if decision_days.empty:
        print("Error: No trading data available for the specified date range.")
        return [], None, None

    decisions = []
    converted = False
    convert_day = None
    convert_rate = None

    for current_day in decision_days:
        # Get all historical data up to the current day
        hist = df.loc[:current_day, "Rate_USD_to_GBP"]
        live_rate = hist.iloc[-1]

        # Calculate remaining trading days until the deadline
        remaining_days_df = df.loc[current_day:deadline_date]
        horizon = len(remaining_days_df) - 1
        if horizon <= 0:
            horizon = 1 # Forecast at least one step ahead

        forecast_mean, forecast_ci = forecast_rates(hist, horizon)

        # Make the decision
        sell_now, threshold = should_convert(live_rate, forecast_mean, forecast_ci)

        decisions.append({
            "date": current_day,
            "live_rate": live_rate,
            "forecast_mean": forecast_mean.mean(),
            "threshold": threshold,
            "decision": "CONVERT" if sell_now else "WAIT"
        })

        if sell_now:
            converted = True
            convert_day = current_day
            convert_rate = live_rate
            print(f"Decision on {current_day.date()}: CONVERT. Live rate {live_rate:.5f} exceeded threshold {threshold:.5f}.")
            break # Stop simulation after converting
        else:
            print(f"Decision on {current_day.date()}: WAIT. Live rate {live_rate:.5f} is below threshold {threshold:.5f}.")


    # **FIX 2: Handle deadline force-sell correctly**
    # If deadline is reached and not converted, convert on the last available trading day.
    if not converted:
        last_trading_day = decision_days[-1]
        live_rate = df.loc[last_trading_day, "Rate_USD_to_GBP"]
        convert_day = last_trading_day
        convert_rate = live_rate
        decisions.append({
            "date": last_trading_day,
            "live_rate": live_rate,
            "forecast_mean": live_rate,
            "threshold": live_rate,
            "decision": "FORCED CONVERT (Deadline)"
        })
        print(f"Deadline reached. Forced conversion on {last_trading_day.date()} at rate {live_rate:.5f}.")


    return decisions, convert_day, convert_rate

# =============================
# Plotting
# =============================
def plot_simulation(df, decisions, convert_day, convert_rate, start_date, deadline_date):
    """Visualises the simulation results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot historical data for context
    context_start = pd.to_datetime(start_date) - pd.Timedelta(days=30)
    ax.plot(df.loc[context_start:deadline_date].index, df.loc[context_start:deadline_date, "Rate_USD_to_GBP"],
            color='grey', alpha=0.6, label="Historical Rate")

    # Highlight the decision period
    dec_df = pd.DataFrame(decisions).set_index("date")
    ax.plot(dec_df.index, dec_df["live_rate"], "o-", color='royalblue', label="Observed Rate (Decision Period)")

    # Mark conversion point
    if convert_day:
        final_gbp = calculate_gbp(AMOUNT_USD, convert_rate, CONVERSION_COST_PERCENT, CONVERSION_COST_FIXED)
        ax.axvline(convert_day, color="crimson", linestyle="--", lw=1.5,
                    label=f"Conversion Day: {convert_day.date()}")
        ax.scatter(convert_day, convert_rate, color="crimson", s=150, zorder=5, marker="*",
                   label=f"Conversion Rate: {convert_rate:.5f}\nReceived: £{final_gbp:,.2f}")

    ax.set_title(f"USD to GBP Conversion Optimiser [{start_date} to {deadline_date}]", fontsize=16)
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
        decisions, convert_day, convert_rate = run_simulation(df_daily, START_DATE, DEADLINE_DATE)

        if convert_day:
            print("\n=== Simulation Results ===")
            print(f"Start Date: {START_DATE}, Deadline: {DEADLINE_DATE}")
            print(f"Optimal Conversion Day: {convert_day.date()}")
            print(f"Conversion Rate (GBP per USD): {convert_rate:.5f}")
            final_amount = calculate_gbp(AMOUNT_USD, convert_rate, CONVERSION_COST_PERCENT, CONVERSION_COST_FIXED)
            print(f"Amount Received: £{final_amount:,.2f}")

            plot_simulation(df_daily, decisions, convert_day, convert_rate, START_DATE, DEADLINE_DATE)