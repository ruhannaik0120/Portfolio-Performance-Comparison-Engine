import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import fetch_stock_data
from src.portfolio import calculate_daily_returns
from src.strategies import(equal_weight_strategy,static_risk_parity_strategy,vol_targeted_risk_parity_strategy,walk_forward_risk_parity_strategy)

#page config
st.set_page_config(
    page_title="Systematic Portolio Engine",
    layout="wide",
)

st.title("Systematic Portfolio Engine")
st.caption("Risk Parity • Vol Targeting • Walk Forward Backtesting")

#sidebar controls
st.sidebar.header("Configuration")

tickers=st.sidebar.multiselect(
     "Select Assets",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS"],
    default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS"]
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

strategy_name = st.sidebar.selectbox(
    "Strategy",
    [
        "Equal Weight",
        "Static Risk Parity",
        "Vol Targeted Risk Parity"
        "Walk Forward Risk Parity"
    ]
)

target_vol = st.sidebar.slider(
    "Target Volatility (Vol Targeted RP Only)",
    0.05, 0.25, 0.15
)

run_button=st.sidebar.button("Run strategy")

#execution
if run_button:
    if len(tickers)<2:
        st.error("Please select atleast 2 assets")
        st.stop()

    prices=fetch_stock_data(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    returns=calculate_daily_returns(prices)

    if strategy_name == "Equal Weight":
        port, cum, dd, metrics = equal_weight_strategy(returns)

    elif strategy_name == "Static Risk Parity":
        port, cum, dd, metrics = static_risk_parity_strategy(returns)

    elif strategy_name == "Vol Targeted Risk Parity":
        port, cum, dd, metrics = vol_targeted_risk_parity_strategy(
            returns,
            target_vol=target_vol
        )

    elif strategy_name == "Walk Forward Risk Parity":
        port, cum, dd, metrics = walk_forward_risk_parity_strategy(returns)

    #metrics display
    col1,col2,col3,col4=st.columns(4)

    col1.metric("Annual Return",f"{metrics['Return']:.2%}")
    col2.metric("Volatility", f"{metrics['Volatility']:.2%}")
    col3.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
    col4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")

    #cumulative chart
    fig_growth = go.Figure()
    fig_growth.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum.values,
            mode="lines",
            name="Cumulative Return"
        )
    )

    fig_growth.update_layout(
        template="plotly_dark",
        title="Portfolio Growth",
        xaxis_title="Date",
        yaxis_title="Growth of ₹1"
    )

    st.plotly_chart(fig_growth,use_container_width=True)

    #drawdown chart
    fig_dd=go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Drawdown"
        )
    )

    fig_dd.update_layout(
        template="plotly_dark",
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown"
    )

    st.plotly_chart(fig_dd,use_container_width=True)

    #Diagnostics
    with st.expander("Diagnostics"):
        if "Total Turnover" in metrics:
            st.write("Total Turnover:", round(metrics["Total Turnover"], 4))

        st.write("Number of Trading Days:", len(port))