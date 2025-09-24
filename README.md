
# Forex Conversion Optimal Stopping

A research and development project exploring **USD → GBP conversion timing** as an **optimal stopping problem**, using time-series forecasting, statistical baselines, and reinforcement-inspired methods.

The project evaluates whether one can systematically outperform naïve benchmarks (e.g., converting on the first or last day) by combining **ARIMA forecasts**, **historical averages**, and **Monte Carlo–based dynamic programming**.


## Features

* **Modular Python framework** for currency conversion timing.
* **Backtesting engine** with reproducible experiments, caching, and efficient data I/O (Parquet).
* **Forecasting & decision modules**: ARIMA, historical baselines, Monte Carlo optimal stopping.
* **Performance evaluation** against perfect-foresight benchmark.
* **Scalable repo design** for experimentation and extension.


## Usage

Run experiments with:

```bash
python run.py --config experiments/config.yaml
```

## Example Results

Here’s how the system performs compared to benchmarks:

```
=== Leaderboard across windows ===
(win_rate_vs_last compares each strategy to converting on the last day)

          strategy  n_windows mean_gbp std_gbp median_gbp mean_pct_of_pf mean_regret_pct win_rate_vs_last median_days_to_convert
 perfect_foresight         20 7,909.73  128.84   7,908.97         100.0%          +0.00%           100.0%                      8
  optimal_stopping         20 7,883.75  126.98   7,890.31          99.7%          -0.33%            75.0%                      2
historical_average         20 7,835.88  147.95   7,892.14          99.1%          -0.93%            85.0%                     15
     arima_trigger         20 7,825.01  139.51   7,863.90          98.9%          -1.07%           100.0%                     16
          last_day         20 7,825.01  139.51   7,863.90          98.9%          -1.07%           100.0%                     16
         first_day         20 7,815.03  122.92   7,853.42          98.8%          -1.19%            40.0%                      0
```


## Graphs

* Strategy comparison vs. perfect foresight
  ![Placeholder graph 1](docs/images/strategy_comparison.png)

* Distribution of regrets across windows
  ![Placeholder graph 2](docs/images/regret_distribution.png)


## Repository Structure

```
OptimalForex/
├── data/                # Raw & processed exchange rate data
├── experiments/         # Backtest results and logs
├── src/
│   └── forex/           # Core framework modules
└── run.py               # Main entry point
```

---

## Next Steps

* Add more forecasting models (Prophet, LSTM).
* Extend to multi-currency support.
* Explore reinforcement learning approaches.


