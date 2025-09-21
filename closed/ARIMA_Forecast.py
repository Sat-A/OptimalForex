import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import itertools
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D # Import Line2D for custom legend

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
class Config:
    CSV_FILE = "HISTDATA_COM_ASCII_GBPUSD_M12024/DAT_ASCII_GBPUSD_M1_2024.csv"
    PICKLE_FILE = "HISTDATA_COM_ASCII_GBPUSD_M12024/processed_data.pkl"
    RESAMPLE_PERIOD = 'D'
    FORCE_RELOAD = True
    ARIMA_ORDER = (1, 0, 0)
    START_DATE = '2024-10-01'
    END_DATE = '2024-10-20'

# =============================================================================
# Data Loading and Preparation
# =============================================================================
def load_and_prepare_data(csv_path: str, pickle_path: str, resample_period: str, force_reload: bool = False) -> pd.DataFrame:
    if not force_reload and os.path.exists(pickle_path):
        print(f"Loaded resampled data from '{pickle_path}'.")
        return pd.read_pickle(pickle_path)
    
    print("Loading data from CSV...")
    try:
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None
        
    df = df.rename(columns={"DateTime Stamp": "Time", "Bar CLOSE Bid Quote": "Close"})    
    df["Time"] = pd.to_datetime(df["Time"], format="%Y%m%d %H%M%S")
    df = df.sort_values("Time").set_index("Time")
    
    print(f"Resampling data to '{resample_period}' frequency...")
    df_resampled = df['Close'].resample(resample_period).last().dropna().to_frame()
    df_resampled['Rate_USD_to_GBP'] = 1 / df_resampled['Close']
    
    df_resampled.to_pickle(pickle_path)
    print(f"Resampled data saved to '{pickle_path}'.")
    
    return df_resampled[['Rate_USD_to_GBP']]

# =============================================================================
# ARIMA Forecasting
# =============================================================================
def generate_daily_multi_step_forecasts(ts: pd.Series, order: tuple, start_date: str, end_date: str) -> dict:
    print(f"\n--- Generating daily multi-step forecasts from {start_date} to {end_date}... ---")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    all_forecasts = {}
    forecast_days = pd.date_range(start=start_date, end=end_date)

    for current_day in forecast_days:
        train_end_date = current_day - pd.Timedelta(days=1)
        history = ts.loc[:train_end_date]
        
        if history.empty:
            print(f"Skipping {current_day.date()}: Not enough historical data.")
            continue
        
        history = history.asfreq('D')
        steps_to_forecast = (end_date - current_day).days + 1
        
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            forecast_series = model_fit.forecast(steps=steps_to_forecast)
            all_forecasts[current_day] = forecast_series
            print(f"Generated forecast starting from {current_day.date()}")
        except Exception as e:
            print(f"Could not generate forecast for {current_day.date()}: {e}")
            continue

    return all_forecasts

# =============================================================================
# Plotting (NEW AND IMPROVED)
# =============================================================================
def plot_daily_forecasts(actual: pd.Series, all_forecasts: dict, start_date: str, end_date: str) -> None:
    """
    Plots the actual values against the series of daily-updated forecasts,
    highlighting the start point of each forecast.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # 1. Plot historical and actual data
    hist_start_date = pd.to_datetime(start_date) - pd.DateOffset(days=30)
    historical_data = actual.loc[hist_start_date:start_date]
    ax.plot(historical_data.index, historical_data, color='grey', label='Historical Data', linestyle='--')

    actual_slice = actual.loc[start_date:end_date]
    ax.plot(actual_slice.index, actual_slice, 'o-', color='black', label='Actual Values', linewidth=2.5, markersize=5, zorder=10)

    # 2. Plot each daily forecast with a start-point marker
    if all_forecasts:
        cmap = get_cmap('viridis', len(all_forecasts))
        for i, (forecast_date, forecast_series) in enumerate(all_forecasts.items()):
            # Plot the forecast line
            ax.plot(forecast_series.index, forecast_series,
                    color=cmap(i),
                    alpha=0.6,
                    linewidth=1.5)
            
            # Add a prominent marker at the start of the forecast
            ax.plot(forecast_series.index[0], forecast_series.iloc[0],
                    marker='o',
                    color=cmap(i),
                    markersize=7,
                    zorder=5)

    # 3. Create a cleaner, more informative legend
    legend_elements = [
        Line2D([0], [0], color='grey', linestyle='--', label='Historical Data'),
        Line2D([0], [0], color='black', marker='o', lw=2.5, label='Actual Values'),
        Line2D([0], [0], color='purple', marker='o', lw=1.5, alpha=0.6, label='Daily Forecasts (w/ Start Point)')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # 4. Formatting
    ax.set_title('Daily Updated ARIMA Forecasts with Start-Point Indicators', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rate (USD to GBP)', fontsize=12)
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.6)
    fig.autofmt_xdate(rotation=45)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution Workflow
# =============================================================================
def main():
    ts_df = load_and_prepare_data(
        csv_path=Config.CSV_FILE,
        pickle_path=Config.PICKLE_FILE,
        resample_period=Config.RESAMPLE_PERIOD,
        force_reload=Config.FORCE_RELOAD
    )
    if ts_df is None: return
    
    ts = ts_df['Rate_USD_to_GBP']
    
    all_forecasts = generate_daily_multi_step_forecasts(
        ts=ts, 
        order=Config.ARIMA_ORDER, 
        start_date=Config.START_DATE,
        end_date=Config.END_DATE
    )
    
    if not all_forecasts:
        print("No forecasts were generated. Cannot plot results.")
        return
        
    plot_daily_forecasts(ts, all_forecasts, Config.START_DATE, Config.END_DATE)

if __name__ == "__main__":
    main()