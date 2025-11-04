import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from portopt.config import Config
from portopt.utils import ensure_dir
from portopt.data import get_prices
from portopt.stats import compute_returns, annualize_stats
from portopt.optimize import max_sharpe, efficient_frontier, tangency_unconstrained
from portopt.analysis import comparison_table
from portopt.plots import (
    plot_frontier, plot_allocation_pie,
    plot_sensitivity_rf, plot_sensitivity_shrinkage
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","TLT","GLD"])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--risk_free", type=float, default=0.02)
    p.add_argument("--price_col", default="Adj Close")
    p.add_argument("--return_type", choices=["log","simple"], default="log")
    p.add_argument("--trading_days", type=int, default=252)
    p.add_argument("--long_only", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        tickers=args.tickers, start=args.start, end=args.end,
        risk_free=args.risk_free, price_col=args.price_col,
        return_type=args.return_type, trading_days=args.trading_days,
        long_only=args.long_only
    )

    out_dir = Path("outputs")
    figs = out_dir / "figures"
    tabs = out_dir / "tables"
    ensure_dir(figs.as_posix()); ensure_dir(tabs.as_posix())

    # 1) Data
    prices = get_prices(cfg.tickers, cfg.start, cfg.end, cfg.price_col)

    # 2) Stats
    rets = compute_returns(prices, cfg.return_type)
    mu, Sigma, Corr = annualize_stats(rets, cfg.trading_days)

    # Save summary tables
    mu.to_frame("mu").T.to_csv(tabs/"expected_returns.csv", index=False)
    Sigma.to_csv(tabs/"covariance.csv")
    Corr.to_csv(tabs/"correlation.csv")

    # 3) Optimization
    opt = max_sharpe(mu, Sigma, cfg.risk_free, long_only=cfg.long_only)
    opt.weights.to_frame("Weight").to_csv(tabs/"optimal_weights.csv")

    # 4) Efficient frontier
    risks, rets_front, srs, targets = efficient_frontier(mu, Sigma, cfg.risk_free, points=60, long_only=cfg.long_only)
    plot_frontier(risks, rets_front, opt.ret, opt.vol, (figs/"frontier.png").as_posix())

    # 5) Compare vs baselines
    opt_res, comp = comparison_table(mu, Sigma, cfg.risk_free, long_only=cfg.long_only)
    comp.round(6).to_csv(tabs/"performance_comparison.csv")

    # 6) Sensitivity: risk-free
    rf_grid = np.linspace(0.0, 0.05, 11)
    sharpe_vals = []
    for rf in rf_grid:
        o = max_sharpe(mu, Sigma, rf, long_only=cfg.long_only)
        sharpe_vals.append(o.sr)
    plot_sensitivity_rf(rf_grid, sharpe_vals, (figs/"sensitivity_rf.png").as_posix())

    # 7) Sensitivity: mean shrinkage
    alphas = np.linspace(0, 1, 11)
    grand_mean = float(mu.mean())
    sparsity = []
    for a in alphas:
        mu_shrunk = (1-a)*mu + a*grand_mean
        o = max_sharpe(mu_shrunk, Sigma, cfg.risk_free, long_only=cfg.long_only)
        sparsity.append(int((o.weights.values < 1e-4).sum()))
    plot_sensitivity_shrinkage(alphas, sparsity, (figs/"sensitivity_shrinkage.png").as_posix())

    # 8) Validation: unconstrained tangency (for reference)
    w_tan = tangency_unconstrained(mu, Sigma, cfg.risk_free)
    w_tan.to_frame("Weight").to_csv(tabs/"tangency_unconstrained.csv")

    # 9) Allocation pie
    plot_allocation_pie(opt.weights, (figs/"allocation_pie.png").as_posix())

    print("Done. Outputs written to ./outputs")

if __name__ == "__main__":
    main()
