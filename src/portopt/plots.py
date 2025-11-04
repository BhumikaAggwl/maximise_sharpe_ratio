import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_frontier(risks, rets, opt_ret, opt_vol, path: str):
    plt.figure(figsize=(7,5))
    plt.scatter(risks, rets, s=18, alpha=0.7, label="Efficient Frontier")
    plt.scatter([opt_vol], [opt_ret], s=100, marker='*', label="Max Sharpe")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Return (μ)")
    plt.title("Efficient Frontier (Long-only)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()

def plot_allocation_pie(weights: pd.Series, path: str):
    plt.figure(figsize=(6,6))
    plt.pie(weights.values, labels=weights.index, autopct="%1.1f%%", startangle=90)
    plt.title("Optimal Allocation (Max Sharpe)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()

def plot_sensitivity_rf(rf_grid, sharpe_vals, path: str):
    plt.figure(figsize=(7,5))
    plt.plot(rf_grid, sharpe_vals, marker='o')
    plt.xlabel("Risk-free rate")
    plt.ylabel("Max Sharpe")
    plt.title("Sensitivity: Sharpe vs Risk-free rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()

def plot_sensitivity_shrinkage(alphas, sparsity, path: str):
    plt.figure(figsize=(7,5))
    plt.plot(alphas, sparsity, marker='s')
    plt.xlabel("Shrinkage toward grand mean (alpha)")
    plt.ylabel("# near-zero weights")
    plt.title("Sensitivity: Weight sparsity vs return shrinkage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
