import pandas as pd
import numpy as np
from typing import List, Optional

def load_prices_yahoo(tickers: List[str], start: Optional[str], end: Optional[str], price_col: str="Adj Close") -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not installed or import failed.") from e

    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if price_col in data:
        df = data[price_col]
    else:
        # Fallback to Close if Adj Close absent
        df = data["Close"]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna(how="all").ffill().dropna()
    return df

def synthetic_prices(tickers: List[str], length: int=2500, seed: int=42) -> pd.DataFrame:
    """Geometric Brownian motion with random covariances to allow offline execution."""
    rng = np.random.default_rng(seed)
    n = len(tickers); dt=1/252
    mu = rng.uniform(0.05, 0.12, n)    # annual drifts
    vol = rng.uniform(0.10, 0.30, n)   # annual vols

    A = rng.standard_normal((n, n))
    cov = A @ A.T
    cov = cov / np.max(np.abs(cov))
    D = np.diag(vol)
    cov = D @ cov @ D

    chol = np.linalg.cholesky(cov + 1e-8*np.eye(n))
    prices = np.zeros((length, n))
    prices[0] = 100*(1 + rng.random(n))

    for t in range(1, length):
        z = rng.standard_normal(n)
        dz = chol @ z
        ret = (mu - 0.5*np.diag(cov))*dt + dz*np.sqrt(dt)
        prices[t] = prices[t-1] * np.exp(ret)

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=length)
    return pd.DataFrame(prices, index=idx, columns=tickers)

def get_prices(tickers, start, end, price_col="Adj Close") -> pd.DataFrame:
    try:
        return load_prices_yahoo(tickers, start, end, price_col)
    except Exception:
        return synthetic_prices(tickers)
