import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import itertools
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
class Config:
    """A class to hold all configuration parameters for the script."""
    # --- File Paths ---
    CSV_FILE = "HISTDATA_COM_ASCII_GBPUSD_M12024/DAT_ASCII_GBPUSD_M1_2024.csv"
    PICKLE_FILE = "HISTDATA_COM_ASCII_GBPUSD_M12024/processed_data.pkl"
    
    # --- Data Loading ---
    RESAMPLE_PERIOD = 'D' # 'D' for Daily, 'H' for Hourly, '5T' for 5-minute
    FORCE_RELOAD = True  # Set to True to ignore pickle and reload from CSV
    
    # --- ARIMA Model ---
    ARIMA_ORDER = (1, 0, 0) # Default order (p, d, q)
    FORECAST_PERIOD = '30D' # Period to forecast and validate against (e.g., '30D')
    
    # --- Analysis Options ---
    RUN_FULL_ANALYSIS = True  # Set to True to run stationarity tests and find best order

# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================
def load_and_prepare_data(csv_path: str, pickle_path: str, resample_period: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Loads forex data, resamples it to a specified frequency, and saves it to a pickle file.
    """
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
# 2. Time Series Analysis
# =============================================================================
def perform_series_analysis(ts: pd.Series) -> None:
    """
    Performs and plots stationarity tests (ADF) and autocorrelation plots (ACF/PACF).
    """
    print("\n--- Performing Time Series Analysis ---")
    # ADF Test for Stationarity
    result = adfuller(ts.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] > 0.05:
        print("Result: The series is likely non-stationary (p-value > 0.05). Differencing may be required.")
    else:
        print("Result: The series is likely stationary (p-value <= 0.05).")

    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(ts.dropna(), ax=ax1, lags=40)
    plot_pacf(ts.dropna(), ax=ax2, lags=40)
    plt.tight_layout()
    plt.show()

def find_best_arima_order(ts: pd.Series) -> tuple:
    """
    Performs a grid search to find the best ARIMA (p,d,q) order based on AIC.
    """
    print("\n--- Finding best ARIMA order (Grid Search)... ---")
    p = d = q = range(0, 3)
    pdq_combinations = list(itertools.product(p, d, q))
    best_aic, best_order = float("inf"), None

    for order in pdq_combinations:
        try:
            model = ARIMA(ts, order=order).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
        except Exception as e:
            continue
            
    print(f"Best Order: {best_order} with AIC: {best_aic:.2f}")
    return best_order

# =============================================================================
# 3. ARIMA Forecasting
# =============================================================================
def generate_rolling_forecast(ts: pd.Series, order: tuple, forecast_period: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generates a rolling forecast for a specified period of a time series.
    """
    print(f"\n--- Generating rolling forecast for the last {forecast_period}... ---")
    test_data = ts.last(forecast_period)
    history = ts.loc[:test_data.index[0] - pd.Timedelta(days=1)].tolist()
    
    predictions, conf_intervals = [], []

    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        
        forecast = model_fit.get_forecast(steps=1)
        yhat = forecast.predicted_mean[0]
        conf = forecast.conf_int(alpha=0.05)
        
        predictions.append(yhat)
        conf_intervals.append(conf[0])
        history.append(test_data[t])
        
    predictions_series = pd.Series(predictions, index=test_data.index)
    conf_df = pd.DataFrame(conf_intervals, index=test_data.index)
    
    return predictions_series, conf_df

# =============================================================================
# 4. Plotting
# =============================================================================
def plot_forecast_vs_actual(actual: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame) -> None:
    """
    Plots the actual time series against the forecasted values with confidence intervals.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(actual.index, actual, 'o-', label='Actual Values', color='#006699')
    ax.plot(forecast.index, forecast, 'o-', label='Rolling Forecast', color='#FF6600')

    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    color='#FF6600', alpha=0.2, label='95% Confidence Interval')

    ax.set_title('ARIMA Rolling Forecast vs. Actual Values', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rate (USD to GBP)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, which='major', linestyle='--', linewidth=0.6)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution Workflow
# =============================================================================
def main() -> None:
    """Main function to run the entire analysis and forecasting workflow."""
    
    # 1. Load Data
    ts_df = load_and_prepare_data(
        csv_path=Config.CSV_FILE,
        pickle_path=Config.PICKLE_FILE,
        resample_period=Config.RESAMPLE_PERIOD,
        force_reload=Config.FORCE_RELOAD
    )
    if ts_df is None:
        return
    
    ts = ts_df['Rate_USD_to_GBP']
    
    # 2. (Optional) Perform Analysis to determine ARIMA order
    arima_order = Config.ARIMA_ORDER
    if Config.RUN_FULL_ANALYSIS:
        perform_series_analysis(ts)
        best_order = find_best_arima_order(ts)
        if best_order: arima_order = best_order
    
    # 3. Generate Forecast
    forecasts, conf_intervals = generate_rolling_forecast(
        ts=ts, 
        order=arima_order, 
        forecast_period=Config.FORECAST_PERIOD
    )
    
    # 4. Plot Results
    actual_values = ts.last(Config.FORECAST_PERIOD)
    plot_forecast_vs_actual(actual_values, forecasts, conf_intervals)

if __name__ == "__main__":
    main()