import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader import fetch_stock_data
from src.portfolio import calculate_daily_returns
from src.strategies import (
    equal_weight_strategy,
    static_risk_parity_strategy,
    vol_targeted_risk_parity_strategy,
    walk_forward_risk_parity_strategy,
)
from src.metrics import (
    cumulative_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    drawdown_series,
)
from src.factors import market_factor_analysis

#page config
st.set_page_config(
    page_title="Systematic Portfolio Engine",
    layout="wide",
)

st.title("Systematic Portfolio Engine")
st.caption("Risk Parity | Vol Targeting | Walk Forward Backtesting")

# Strategy guide
with st.expander("What do these strategies mean?", expanded=False):
    s1, s2 = st.columns(2)
    with s1:
        st.info(
            "**Equal Weight**\n\n"
            "Splits money equally across all stocks, rebalanced daily.\n\n"
            '_Think: "Put 25% in each of 4 stocks, reset every day."_'
        )
        st.success(
            "**Vol-Targeted RP**\n\n"
            "Same as Risk Parity, but scales down exposure when markets get volatile.\n\n"
            '_Think: "Risk Parity + a safety brake for rough markets."_'
        )
    with s2:
        st.warning(
            "**Static Risk Parity**\n\n"
            "Weights stocks so each contributes equal risk, not equal money. "
            "Safer stocks get more weight.\n\n"
            '_Think: "No single stock dominates your risk."_'
        )
        st.error(
            "**Integrated Walk-Forward RP**\n\n"
            "Recalculates weights monthly using only past data, "
            "with vol targeting and real trading costs.\n\n"
            '_Think: "The most realistic -- no future data, pays fees."_'
        )

#sidebar controls
st.sidebar.header("Configuration")

# Free ticker input
ticker_input = st.sidebar.text_input(
    "Enter NSE Tickers (comma separated)",
    value="RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS"
)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Date range (dynamic up to today)
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Strategy multi-selection
strategy_options = [
    "Equal Weight",
    "Static Risk Parity",
    "Vol-Targeted RP",
    "Integrated Walk-Forward RP"
]

selected_strategies = st.sidebar.multiselect(
    "Select Strategies to Compare",
    strategy_options,
    default=["Equal Weight"]
)

# Leverage slider
leverage = st.sidebar.slider(
    "Leverage (x)",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1
)

# Target volatility slider (only used for vol-targeted RP)
target_vol = st.sidebar.slider(
    "Target Volatility (Vol Targeted RP Only)",
    min_value=0.05,
    max_value=0.30,
    value=0.15,
    step=0.01
)

run_button = st.sidebar.button("Run Strategy")

# execution -- compute and cache in session_state
if run_button:
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
        st.stop()

    if not selected_strategies:
        st.error("Please select at least 1 strategy.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    try:
        prices = fetch_stock_data(tickers, str(start_date), str(end_date))
    except (ConnectionError, ValueError) as e:
        st.error(f"Data error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error fetching data: {e}")
        st.stop()

    returns = calculate_daily_returns(prices)
    returns = returns.dropna()  # Important safety line

    if returns.empty:
        st.error("No valid return data after cleaning. Try different tickers or a wider date range.")
        st.stop()

    # Safety checks
    if len(returns.columns) < 2:
        st.error("Need at least 2 assets with valid data. Some tickers may have returned no data.")
        st.stop()

    if len(returns) < 252:
        st.warning("Less than 1 year of data. Results may be unstable.")

    # Strategy dispatcher
    strategy_results = {}

    for strategy in selected_strategies:
        try:
            if strategy == "Equal Weight":
                port, cum, dd, metrics, weights = equal_weight_strategy(returns)

            elif strategy == "Static Risk Parity":
                port, cum, dd, metrics, weights = static_risk_parity_strategy(returns)

            elif strategy == "Vol-Targeted RP":
                port, cum, dd, metrics, weights = vol_targeted_risk_parity_strategy(
                    returns, target_vol=target_vol
                )

            elif strategy == "Integrated Walk-Forward RP":
                port, cum, dd, metrics, weights = walk_forward_risk_parity_strategy(
                    returns, max_leverage=leverage
                )

            # Apply leverage AFTER strategy logic
            port_leveraged = port * leverage
            cum_leveraged = cumulative_returns(port_leveraged)
            dd_leveraged = drawdown_series(cum_leveraged)

            metrics["Return"] = annualized_return(port_leveraged)
            metrics["Volatility"] = annualized_volatility(port_leveraged)
            metrics["Sharpe"] = sharpe_ratio(port_leveraged)
            metrics["Max Drawdown"] = max_drawdown(cum_leveraged)

            strategy_results[strategy] = {
                "returns": port_leveraged,
                "cumulative": cum_leveraged,
                "drawdown": dd_leveraged,
                "metrics": metrics,
                "weights": weights,
            }
        except Exception as e:
            st.error(f"{strategy} failed: {e}")
            continue

    if not strategy_results:
        st.error("All selected strategies failed. Check your inputs and try again.")
        st.stop()

    # Save results so they persist across widget interactions
    st.session_state["strategy_results"] = strategy_results

# ---------------------------------------------------------------
# Display results (reads from session_state, survives reruns)
# ---------------------------------------------------------------
if "strategy_results" in st.session_state:
    strategy_results = st.session_state["strategy_results"]

    # Comparison table
    st.subheader("Portfolio Comparison")

    with st.expander("What do these columns mean?", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Return** -- Annualized earnings per year.")
            st.markdown("**Volatility** -- How much returns bounce. Lower = smoother.")
            st.markdown("**Sharpe** -- Return per unit of risk. Above 1.0 is good.")
        with c2:
            st.markdown("**Max Drawdown** -- Worst peak-to-trough loss.")
            st.markdown("**Total Turnover** -- Total trading activity. Higher = more fees.")

    metrics_df = pd.DataFrame({
        name: result["metrics"]
        for name, result in strategy_results.items()
    }).T

    st.dataframe(metrics_df.style.format("{:.4f}"))

    # Growth chart
    st.subheader("Portfolio Growth")
    st.caption("Shows how 1 unit invested at the start would have grown over time for each strategy.")

    fig_growth = go.Figure()

    for name, result in strategy_results.items():
        fig_growth.add_trace(
            go.Scatter(
                x=result["cumulative"].index,
                y=result["cumulative"],
                mode="lines",
                name=name,
            )
        )

    fig_growth.update_layout(
        title="Portfolio Growth Comparison",
        xaxis_title="Date",
        yaxis_title="Growth of 1 unit",
        template="plotly_dark",
    )

    st.plotly_chart(fig_growth, use_container_width=True)

    # Drawdown chart
    st.subheader("Drawdown")
    st.caption("Shows how far each strategy fell from its peak at any point. Closer to 0 = less pain.")

    fig_dd = go.Figure()

    for name, result in strategy_results.items():
        fig_dd.add_trace(
            go.Scatter(
                x=result["drawdown"].index,
                y=result["drawdown"],
                mode="lines",
                name=name,
            )
        )

    fig_dd.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_dark",
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # ---------------------------------------------------------------
    # Factor Analysis (CAPM)
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("Factor Analysis (CAPM)")
    st.caption("Measures how much of each strategy's performance comes from the market vs. skill.")

    with st.expander("What do these columns mean?", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "**Alpha (Ann.)** -- Annualized excess return not explained by the market. "
                "Positive = the strategy adds value beyond just tracking the index."
            )
            st.markdown(
                "**Beta** -- Sensitivity to market moves. "
                "Beta of 1.0 means the strategy moves 1:1 with the market. "
                "Lower beta = less market risk."
            )
        with c2:
            st.markdown(
                "**R-squared** -- How much of the strategy's returns are explained by the market. "
                "0.90 means 90% is market-driven, only 10% is independent."
            )

    try:
        nifty_prices = fetch_stock_data(["^NSEI"], str(start_date), str(end_date))
        nifty_returns = calculate_daily_returns(nifty_prices).squeeze()

        factor_rows = {}
        for name, result in strategy_results.items():
            factor_rows[name] = market_factor_analysis(
                result["returns"], nifty_returns
            )

        factor_df = pd.DataFrame(factor_rows).T
        factor_df.columns = ["Alpha (Ann.)", "Beta", "R-squared"]
        st.dataframe(factor_df.style.format("{:.4f}"))
    except Exception as e:
        st.warning(f"Factor analysis unavailable: {e}")

    # ---------------------------------------------------------------
    # Export section
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("Download Research Data")

    # Let the user pick which strategy and data type to download
    export_strategy = st.selectbox(
        "Select strategy",
        list(strategy_results.keys()),
        key="export_strategy",
    )

    result = strategy_results[export_strategy]

    export_options = ["Daily Returns", "Cumulative Returns", "Drawdown", "Metrics"]
    if result.get("weights") is not None:
        export_options.append("Portfolio Weights")

    export_type = st.selectbox(
        "Select data to download",
        export_options,
        key="export_type",
    )

    if export_type == "Daily Returns":
        csv_data = result["returns"].to_csv().encode("utf-8")
        file_name = f"{export_strategy}_daily_returns.csv"
    elif export_type == "Cumulative Returns":
        csv_data = result["cumulative"].to_csv().encode("utf-8")
        file_name = f"{export_strategy}_cumulative.csv"
    elif export_type == "Drawdown":
        csv_data = result["drawdown"].to_csv().encode("utf-8")
        file_name = f"{export_strategy}_drawdown.csv"
    elif export_type == "Metrics":
        metrics_export = pd.DataFrame(result["metrics"], index=[0])
        csv_data = metrics_export.to_csv(index=False).encode("utf-8")
        file_name = f"{export_strategy}_metrics.csv"
    elif export_type == "Portfolio Weights":
        csv_data = result["weights"].to_csv().encode("utf-8")
        file_name = f"{export_strategy}_weights.csv"

    st.download_button(
        label=f"Download {export_type} CSV",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        key="export_download",
    )