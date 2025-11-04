import pandas as pd
import numpy as np
from portopt.stats import compute_returns, annualize_stats

def test_returns_shapes():
    prices = pd.DataFrame({
        "A":[100,101,102,103],
        "B":[50,50,49,50]
    }, index=pd.date_range("2020-01-01", periods=4))
    rets = compute_returns(prices, "log")
    assert rets.shape == (3,2)
    mu, Sigma, Corr = annualize_stats(rets, 252)
    assert mu.shape[0] == 2
    assert Sigma.shape == (2,2)
    assert Corr.shape == (2,2)
