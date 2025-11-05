# Portfolio Optimization — Maximizing Sharpe Ratio 

This project builds a full workflow to **maximize Sharpe ratio** under standard Markowitz assumptions with **long-only** weights and a **full investment** constraint.

## Math

- We have `n` assets with:
  - Expected returns (annualized) vector: **μ** ∈ ℝⁿ  
  - Covariance (annualized) matrix: **Σ** ∈ ℝⁿˣⁿ, symmetric positive semidefinite  
  - Portfolio weights: **w** ∈ ℝⁿ
- Constraints:
  - Full investment:  ∑ᵢ wᵢ = 1
  - Long-only:        wᵢ ≥ 0  ∀i
- Portfolio return:  
  **Rₚ = wᵀ μ**
- Portfolio volatility:  
  **σₚ = sqrt(wᵀ Σ w)**
- Sharpe ratio (annualized):  
  **SR(w) = (Rₚ − r_f) / σₚ = (wᵀ μ − r_f) / sqrt(wᵀ Σ w)**

**Max-Sharpe problem (long-only):**
maximize_w (wᵀ μ − r_f) / sqrt(wᵀ Σ w)
subject to ∑ wᵢ = 1, wᵢ ≥ 0

sql
Copy code

**Analytical tangency (unconstrained, validation only)**  
If shorting allowed and only equality constraint, tangency direction is:
w* ∝ Σ⁻¹ (μ − r_f 1)

pgsql
Copy code
We normalize to sum to one.

## What’s included

- Data: from Yahoo Finance via `yfinance` (if available). If not, uses a **synthetic GBM fallback** so everything still runs offline.
- Estimates: daily returns → annualize to get **μ**, **Σ**, and correlation matrix.
- Optimizer: `scipy.optimize.minimize` (SLSQP) for **max Sharpe** and **min variance for target return** (for efficient frontier).
- Plots: efficient frontier with **max-Sharpe** point, allocation pie, sensitivity charts.
- Benchmarks: equal-weight and best single-asset baselines.
- Sensitivity: vary risk-free rate and apply **μ shrinkage** toward its grand mean.
- Outputs: CSVs in `outputs/tables` and PNGs in `outputs/figures`.

## Quickstart

```bash
# 1) Create env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
```bash
# 2) Run end-to-end
python scripts/run_end_to_end.py \
  --tickers AAPL MSFT TLT GLD \
  --start 2015-01-01 \
  --end   2025-11-01 \
  --risk_free 0.02 \
  --return_type log \
  --trading_days 252
```
# Outputs:
```bash
outputs/tables/optimal_weights.csv

outputs/tables/performance_comparison.csv

outputs/figures/frontier.png

outputs/figures/allocation_pie.png

outputs/figures/sensitivity_rf.png

outputs/figures/sensitivity_shrinkage.png
```

# Notes
```bash
If you want to allow shorting, set long_only=False in the code paths.

For robust stats, consider:

Return shrinkage (already included demo)

Covariance shrinkage (e.g., Ledoit–Wolf)

Rolling windows / out-of-sample validation.
```
yaml
Copy code

---

```bash
# requirements.txt

```txt
```
numpy>=1.23
pandas>=1.5
matplotlib>=3.7
scipy>=1.10
yfinance>=0.2.36
