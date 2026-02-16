"""
Strategy wrappers — each returns (portfolio_returns, cumulative, metadata_dict).
All heavy computation lives here, keeping main.py presentation-only.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.portfolio import (
    calculate_daily_returns,
    equal_weight_portfolio,
    custom_weight_portfolio,
    walk_forward_rp_vol_targeted,
)
from src.metrics import (
    cumulative_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    drawdown_series,
    rolling_volatility,
    portfolio_risk_decomposition,
    compute_risk_parity_weights,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _metrics_dict(port_returns, cum_returns):
    """Standard metrics dict for a strategy."""
    return {
        "Return": annualized_return(port_returns),
        "Volatility": annualized_volatility(port_returns),
        "Sharpe": sharpe_ratio(port_returns),
        "Max Drawdown": max_drawdown(cum_returns),
    }


# ------------------------------------------------------------------
# Product strategies
# ------------------------------------------------------------------

def equal_weight_strategy(returns):
    """Equal-weight daily-rebalanced benchmark."""
    port = equal_weight_portfolio(returns)
    cum = cumulative_returns(port)
    dd = drawdown_series(cum)
    return port, cum, dd, _metrics_dict(port, cum)


def static_risk_parity_strategy(returns):
    """Full-sample risk-parity weights applied statically."""
    rp_weights = compute_risk_parity_weights(returns)
    port = custom_weight_portfolio(returns, rp_weights)
    cum = cumulative_returns(port)
    dd = drawdown_series(cum)
    metrics = _metrics_dict(port, cum)
    metrics["weights"] = rp_weights
    return port, cum, dd, metrics


def vol_targeted_risk_parity_strategy(returns, target_vol=0.15):
    """Static risk-parity with volatility targeting (scale capped at 1)."""
    rp_weights = compute_risk_parity_weights(returns)
    rp_port = custom_weight_portfolio(returns, rp_weights)
    roll_vol = rolling_volatility(rp_port)

    scaled = []
    for date in rp_port.index:
        rv = roll_vol.loc[date]
        scale = min(target_vol / rv, 1.0) if (not np.isnan(rv) and rv > 0) else 1.0
        scaled.append(rp_port.loc[date] * scale)

    port = pd.Series(scaled, index=rp_port.index)
    cum = cumulative_returns(port)
    dd = drawdown_series(cum)
    return port, cum, dd, _metrics_dict(port, cum)


def walk_forward_risk_parity_strategy(returns, lookback=252, target_vol=0.15, transaction_cost=0.001, max_leverage=1.0):
    """Walk-forward risk-parity with integrated vol targeting — the primary production strategy."""
    port, turnover = walk_forward_rp_vol_targeted(
        returns, lookback=lookback, target_vol=target_vol,
        transaction_cost_rate=transaction_cost, max_leverage=max_leverage,
    )
    cum = cumulative_returns(port)
    dd = drawdown_series(cum)
    metrics = _metrics_dict(port, cum)
    metrics["Total Turnover"] = turnover
    return port, cum, dd, metrics


# ------------------------------------------------------------------
# Research-only strategies
# ------------------------------------------------------------------

def regime_switching_strategy(returns, nifty_returns, rp_weights, transaction_cost_rate=0.001):
    """
    ML regime-based dynamic allocation (research mode only).
    Crash  → Risk Parity | Sideways → Blended | Bull → Equal Weight
    """
    n = returns.shape[1]
    equal_weights = np.ones(n) / n
    blended_weights = 0.5 * np.array(rp_weights) + 0.5 * equal_weights

    # --- KMeans regime detection on NIFTY ---
    nifty_roll_vol = rolling_volatility(nifty_returns)
    nifty_roll_ret = nifty_returns.rolling(window=30).mean()

    features = pd.DataFrame({
        "RollingVol": nifty_roll_vol,
        "RollingReturn": nifty_roll_ret,
    }).dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[["RollingVol", "RollingReturn"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled)
    features["Regime"] = kmeans.labels_
    features["Regime"] = features["Regime"].shift(1)

    cluster_stats = features.groupby("Regime")[["RollingVol", "RollingReturn"]].mean()
    crash_cluster = cluster_stats["RollingVol"].idxmax()
    remaining = cluster_stats.drop(crash_cluster)
    bull_cluster = remaining["RollingReturn"].idxmax()
    sideways_cluster = remaining.drop(bull_cluster).index[0]
    regime_labels = {crash_cluster: "Crash", bull_cluster: "Bull", sideways_cluster: "Sideways"}
    features["RegimeLabel"] = features["Regime"].map(regime_labels)

    # --- Dynamic allocation loop ---
    prev_w = np.array(equal_weights)
    total_turnover = 0
    min_hold_days = 20
    days_in_regime = 0
    current_regime = None
    actual_switches = 0
    dynamic_returns = []

    for date in returns.index:
        detected = (
            features.loc[date, "RegimeLabel"]
            if date in features.index and pd.notna(features.loc[date, "RegimeLabel"])
            else "Bull"
        )

        if current_regime is None:
            current_regime = detected
            days_in_regime = 0
        elif detected != current_regime and days_in_regime >= min_hold_days:
            current_regime = detected
            days_in_regime = 0
            actual_switches += 1
        days_in_regime += 1

        if current_regime == "Crash":
            w = np.array(rp_weights)
        elif current_regime == "Sideways":
            w = blended_weights
        else:
            w = np.array(equal_weights)

        turnover = np.sum(np.abs(w - prev_w))
        total_turnover += turnover
        daily_ret = returns.loc[date].dot(w) - transaction_cost_rate * turnover
        dynamic_returns.append(daily_ret)
        prev_w = w

    port = pd.Series(dynamic_returns, index=returns.index)
    cum = cumulative_returns(port)
    dd = drawdown_series(cum)

    regime_series = features["RegimeLabel"].reindex(returns.index)
    raw_switches = (regime_series != regime_series.shift(1)).sum()

    metrics = _metrics_dict(port, cum)
    metrics["Total Turnover"] = round(total_turnover, 4)
    metrics["Raw Regime Switches"] = raw_switches
    metrics["Actual Switches (20-day hold)"] = actual_switches
    metrics["features"] = features
    metrics["regime_labels"] = regime_labels
    metrics["cluster_stats"] = cluster_stats

    return port, cum, dd, metrics
