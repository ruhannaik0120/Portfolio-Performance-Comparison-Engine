"""
Backtest utilities — transaction cost modelling, turnover tracking.
"""

import numpy as np


def apply_transaction_costs(daily_return, new_weights, prev_weights, cost_rate=0.001):
    """
    Deduct transaction costs proportional to turnover from a daily return.

    Parameters:
        daily_return (float): Raw portfolio return for the day.
        new_weights (array-like): Target weights after rebalance.
        prev_weights (array-like): Weights before rebalance.
        cost_rate (float): Cost per unit of turnover (default 0.1 %).

    Returns:
        net_return (float): Return after transaction costs.
        turnover (float): Turnover for this rebalance.
    """
    new_weights = np.asarray(new_weights)
    prev_weights = np.asarray(prev_weights)
    turnover = np.sum(np.abs(new_weights - prev_weights))
    net_return = daily_return - cost_rate * turnover
    return net_return, turnover


def total_turnover_from_weight_series(weight_series):
    """
    Compute total turnover from a list / array of weight vectors over time.

    Parameters:
        weight_series (list[np.array]): Sequence of weight vectors.

    Returns:
        float: Cumulative turnover.
    """
    total = 0.0
    for i in range(1, len(weight_series)):
        total += np.sum(np.abs(weight_series[i] - weight_series[i - 1]))
    return total
