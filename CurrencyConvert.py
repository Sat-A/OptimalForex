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
# DEADLINE_DATE = "2024-12-01" # Your original deadline
DEADLINE_DATE = "2024-08-31" # Using a shorter period for a focused example

# =============================
# Setup Functions
# =============================
def load_and_prepare_data(path):
    """Loads and prepares the forex data from the CSV file."""
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
    """Calculates the final GBP amount after fees."""
    fee = (amount_usd * CONVERSION_COST_PERCENT / 100.0) + CONVERSION_COST_FIXED
    net_usd = amount_usd - fee
    return net_usd * rate_usd_to_gbp

# =============================
# Strategy 1: Perfect Foresight (The Benchmark)
# =============================
def run_perfect_foresight_strategy(df, start_date, deadline_date):
    """Finds the absolute best day to convert in hindsight."""
    period_df = df.loc[start_date:deadline_date]
    best_day = period_df['Rate_USD_to_GBP'].idxmax()
    best_rate = period_df['Rate_USD_to_GBP'].max()
    return {"name": "Perfect Foresight", "date": best_day, "rate": best_rate}

# =============================
# Strategy 2 & 3: First Day / Last Day
# =============================
def run_first_day_strategy(df, start_date, deadline_date):
    """Converts on the first possible day."""
    first_day = df.loc[start_date:deadline_date].index[0]
    first_rate = df.loc[first_day, "Rate_USD_to_GBP"]
    return {"name": "First Day Conversion", "date": first_day, "rate": first_rate}

def run_last_day_strategy(df, start_date, deadline_date):
    """Waits and converts on the last possible day."""
    last_day = df.loc[start_date:deadline_date].index[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "Last Day Conversion", "date": last_day, "rate": last_rate}

# =============================
# Strategy 4: Historical Average
# =============================
def run_historical_average_strategy(df, start_date, deadline_date):
    """Converts on the first day the live rate is above the historical average."""
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
# Strategy 5: ARIMA Forecast (NEW)
# =============================
def run_arima_strategy(df, start_date, deadline_date, order=(5, 1, 0), min_history=30):
    """
    Converts when the live rate is better than the next day's forecast.
    - order (p,d,q): ARIMA model parameters.
    - min_history: Minimum number of days required to train the model.
    """
    decision_days = df.loc[start_date:deadline_date].index
    
    for current_day in decision_days:
        # Get all historical data up to the current day
        history = df.loc[:current_day, "Rate_USD_to_GBP"]
        
        # Ensure we have enough data to train the model
        if len(history) < min_history:
            continue
            
        try:
            # Train the ARIMA model on the historical data
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            
            # Forecast one step ahead (for the next day)
            forecast = model_fit.forecast(steps=1).iloc[0]
            
            # Get the live rate for the current day
            live_rate = history.iloc[-1]
            
            # If today's rate is better than tomorrow's forecast, convert now
            if live_rate > forecast:
                return {"name": "ARIMA Forecast", "date": current_day, "rate": live_rate}
                
        except Exception as e:
            # If the model fails for any reason, just move to the next day
            # print(f"ARIMA model failed on {current_day.date()}: {e}")
            continue

    # If the condition was never met, convert on the last day as a fallback
    last_day = decision_days[-1]
    last_rate = df.loc[last_day, "Rate_USD_to_GBP"]
    return {"name": "ARIMA Forecast", "date": last_day, "rate": last_rate}


# =============================
# Plotting
# =============================
def plot_strategy_comparison(df, results, start_date, deadline_date):
    """Visualises the daily rate and the conversion points for each strategy."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    plot_df = df.loc[start_date:deadline_date]
    ax.plot(plot_df.index, plot_df["Rate_USD_to_GBP"], color='#555', alpha=0.7, linewidth=2, label="Daily Rate (USD to GBP)")

    marker_styles = ['*', 'o', 's', '^', 'v', 'D', 'P', 'X', 'h', '8']
    colors = plt.cm.tab10.colors
    
    for idx, r in enumerate(results):
        name, date, rate = r['name'], r['date'], r['rate']
        marker = marker_styles[idx % len(marker_styles)]
        color = colors[idx % len(colors)]
        label_text = f"{name}: {rate:.5f} on {date.strftime('%b %d')}"
        ax.scatter(date, rate, marker=marker, s=200, color=color, zorder=5, edgecolors='black',
                   label=label_text)

    ax.set_title(f"Comparison of Conversion Strategies ({start_date} to {deadline_date})", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Rate (GBP per 1 USD)", fontsize=14)
    ax.legend(loc="best", fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    df_daily = load_and_prepare_data(CSV_FILE)

    if df_daily is not None:
        # Run all strategies, including the new ARIMA strategy
        results = [
            run_perfect_foresight_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_first_day_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_last_day_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_historical_average_strategy(df_daily, START_DATE, DEADLINE_DATE),
            run_arima_strategy(df_daily, START_DATE, DEADLINE_DATE) # <-- NEW STRATEGY ADDED HERE
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
        print(f"Converting ${AMOUNT_USD:,.2f} to GBP between {START_DATE} and {DEADLINE_DATE}\n")
        print(results_df[['name', 'date', 'rate', 'gbp_received', 'difference']].sort_values('gbp_received', ascending=False).to_string(index=False))

        # Plot the comparison
        plot_strategy_comparison(df_daily, results, START_DATE, DEADLINE_DATE)