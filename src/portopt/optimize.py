import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize

@dataclass
class OptResult:
    weights: pd.Series
    sr: float
    ret: float
    vol: float

def portfolio_performance(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float) -> Tuple[float,float,float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sr  = (ret - rf) / vol if vol > 0 else -np.inf
    return ret, vol, sr

def max_sharpe(mu_vec: pd.Series, Sigma_df: pd.DataFrame, rf: float, long_only: bool=True) -> OptResult:
    n = len(mu_vec)
    mu = mu_vec.values
    Sigma = Sigma_df.values
    w0 = np.ones(n) / n

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)]*n if long_only else [(-1.0, 1.0)]*n

    def neg_sharpe(w):
        _, _, sr = portfolio_performance(w, mu, Sigma, rf)
        return -sr

    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 10_000, 'ftol': 1e-12})
    w = res.x
    ret, vol, sr = portfolio_performance(w, mu, Sigma, rf)
    if long_only:
        # sanitize any tiny negatives from numerical noise
        w = np.maximum(w, 0)
        w = w / max(w.sum(), 1e-12)

    return OptResult(weights=pd.Series(w, index=mu_vec.index), sr=sr, ret=ret, vol=vol)

def min_variance_for_target_return(mu_vec: pd.Series, Sigma_df: pd.DataFrame, target: float, rf: float, long_only=True) -> OptResult:
    n = len(mu_vec)
    mu = mu_vec.values
    Sigma = Sigma_df.values
    w0 = np.ones(n)/n

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: w @ mu - target}
    ]
    bounds = [(0.0, 1.0)]*n if long_only else [(-1.0, 1.0)]*n

    def var_obj(w): return w @ Sigma @ w

    res = minimize(var_obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 10_000, 'ftol': 1e-12})
    w = res.x
    ret, vol, sr = portfolio_performance(w, mu, Sigma, rf)
    return OptResult(weights=pd.Series(w, index=mu_vec.index), sr=sr, ret=ret, vol=vol)

def efficient_frontier(mu_vec: pd.Series, Sigma_df: pd.DataFrame, rf: float, points: int=60, long_only=True):
    rmin = float(mu_vec.min()) * 0.6
    rmax = float(mu_vec.max()) * 1.2
    targets = np.linspace(rmin, rmax, points)
    risks, rets, srs = [], [], []
    for t in targets:
        try:
            res = min_variance_for_target_return(mu_vec, Sigma_df, t, rf, long_only=long_only)
            risks.append(res.vol); rets.append(res.ret); srs.append(res.sr)
        except Exception:
            pass
    return np.array(risks), np.array(rets), np.array(srs), targets

def tangency_unconstrained(mu_vec: pd.Series, Sigma_df: pd.DataFrame, rf: float) -> pd.Series:
    one = np.ones(len(mu_vec))
    Sigma_inv = np.linalg.pinv(Sigma_df.values)
    w = Sigma_inv @ (mu_vec.values - rf*one)
    w = w / np.sum(w)
    return pd.Series(w, index=mu_vec.index)
