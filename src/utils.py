# Python file
import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - risk_free_rate) / std_return if std_return != 0 else 0
