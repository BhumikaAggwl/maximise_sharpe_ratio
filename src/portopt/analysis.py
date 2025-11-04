import pandas as pd
import numpy as np
from .optimize import max_sharpe

def metrics_for_weights(w: pd.Series, mu: pd.Series, Sigma: pd.DataFrame, rf: float):
    ret = float(w.values @ mu.values)
    vol = float(np.sqrt(w.values @ Sigma.values @ w.values))
    sr  = (ret - rf)/max(vol,1e-12)
    return ret, vol, sr

def comparison_table(mu: pd.Series, Sigma: pd.DataFrame, rf: float, long_only=True) -> pd.DataFrame:
    opt = max_sharpe(mu, Sigma, rf, long_only=long_only)
    w_eq = pd.Series(np.ones(len(mu))/len(mu), index=mu.index)
    ret_eq, vol_eq, sr_eq = metrics_for_weights(w_eq, mu, Sigma, rf)

    single_stats = []
    for t in mu.index:
        w_single = pd.Series([1.0 if x==t else 0.0 for x in mu.index], index=mu.index)
        single_stats.append((t, *metrics_for_weights(w_single, mu, Sigma, rf)))
    best_single = sorted(single_stats, key=lambda x: x[3])[-1]

    comp = pd.DataFrame([
        {"Portfolio":"Max Sharpe", "Return": opt.ret, "Volatility": opt.vol, "Sharpe": opt.sr},
        {"Portfolio":"Equal Weight","Return": ret_eq,"Volatility": vol_eq,"Sharpe": sr_eq},
        {"Portfolio": f"Single Asset ({best_single[0]})",
         "Return": best_single[1], "Volatility": best_single[2], "Sharpe": best_single[3]},
    ]).set_index("Portfolio").sort_values("Sharpe", ascending=False)

    return opt, comp
