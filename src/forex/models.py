import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional

# ---------- Helpers for RandomWalkEWMA ----------
def _log_returns(series: pd.Series) -> pd.Series:
    return np.log(series).diff().dropna()

def _ewma_vol(log_rets: pd.Series, lam: float) -> float:
    if len(log_rets) == 0:
        return 0.0
    w = np.array([(1 - lam) * lam ** i for i in range(len(log_rets)-1, -1, -1)])
    w /= w.sum()
    mu = np.sum(w * log_rets.values)
    var = np.sum(w * (log_rets.values - mu) ** 2)
    return float(np.sqrt(max(var, 0.0)))

def _shrink(x: float, shrink: float = 0.75) -> float:
    return (1 - shrink) * float(x)

# ---------- Models ----------
class RandomWalkEWMA:
    """
    One-step distributional forecaster:
    - log random-walk with EWMA volatility and small drift (shrunk toward 0)
    - returns Monte-Carlo samples for R_{t+1}
    """
    def __init__(self, ewma_lambda: float = 0.94, drift_window: int = 60, seed: Optional[int] = None):
        self.lam = ewma_lambda
        self.drift_window = drift_window
        self.rng = np.random.default_rng(seed)

    def sample_next(self, y_hist: pd.Series, n_samples: int = 2000) -> np.ndarray:
        y = y_hist.dropna()
        if len(y) < 2:
            return np.full(n_samples, float(y.iloc[-1]) if len(y) else np.nan)

        lr = _log_returns(y)
        vol = _ewma_vol(lr, self.lam)
        if not np.isfinite(vol) or vol == 0:
            vol = max(1e-6, float(lr.std()))

        if len(lr) >= self.drift_window:
            mu_hat = float(lr.tail(self.drift_window).mean())
        else:
            mu_hat = float(lr.mean())
        mu_hat = _shrink(mu_hat, 0.75)

        eps = self.rng.normal(loc=mu_hat, scale=vol, size=n_samples)
        return y.iloc[-1] * np.exp(eps)

class ArimaPointForecaster:
    """Point forecaster for baseline ARIMA trigger."""
    def __init__(self, order=(1,0,0)):
        self.order = tuple(order)

    def forecast_next(self, y_hist: pd.Series) -> Optional[float]:
        if len(y_hist) < sum(self.order) + max(1, self.order[0]):
            return None
        try:
            model = ARIMA(y_hist, order=self.order)
            fit = model.fit()
            return float(fit.forecast(steps=1).iloc[0])
        except Exception:
            return None
