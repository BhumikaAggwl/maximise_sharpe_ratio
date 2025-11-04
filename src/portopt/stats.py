import pandas as pd
import numpy as np

def compute_returns(prices: pd.DataFrame, return_type: str="log") -> pd.DataFrame:
    if return_type == "log":
        rets = (prices / prices.shift(1)).apply(np.log).dropna()
    else:
        rets = prices.pct_change().dropna()
    return rets

def annualize_stats(rets: pd.DataFrame, trading_days: int=252):
    mu = rets.mean() * trading_days
    Sigma = rets.cov() * trading_days
    Corr = rets.corr()
    return mu, Sigma, Corr
