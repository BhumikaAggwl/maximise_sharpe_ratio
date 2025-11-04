import numpy as np
import pandas as pd
from portopt.optimize import max_sharpe

def test_max_sharpe_basic():
    mu = pd.Series([0.10, 0.05], index=["A","B"])
    Sigma = pd.DataFrame([[0.04,0.01],[0.01,0.02]], index=["A","B"], columns=["A","B"])
    rf = 0.02
    res = max_sharpe(mu, Sigma, rf, long_only=True)
    assert abs(res.weights.sum() - 1) < 1e-6
    assert (res.weights.values >= -1e-8).all()
