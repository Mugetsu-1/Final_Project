"""Simple Streamlit interface for the course final project."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st



st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; color: #154360; text-align: center; margin-bottom: 1rem; }
    .prediction-up { color: #0b8a2a; font-size: 1.8rem; font-weight: 700; }
    .prediction-down { color: #b22222; font-size: 1.8rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_price_chart(data: pd.DataFrame, ticker: str) -> None:
    """Show recent price movement with two moving averages."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MA_20"],
            mode="lines",
            name="MA 20",
            line=dict(color="orange", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MA_50"],
            mode="lines",
            name="MA 50",
            line=dict(color="steelblue", width=1),
        )
    )
    fig.update_layout(
        title=f"{ticker} price history",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_indicator_tabs(data: pd.DataFrame) -> None:
    """Show three simple indicator views."""
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])

    with tab1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], mode="lines", name="Signal"))
        fig_macd.add_trace(
            go.Bar(
                x=data["Date"],
                y=data["MACD"] - data["MACD_Signal"],
                name="Histogram",
                opacity=0.35,
            )
        )
        fig_macd.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_macd, use_container_width=True)

    with tab3:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
        fig_bb.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["BB_Upper"],
                mode="lines",
                name="Upper Band",
                line=dict(dash="dash"),
            )
        )
        fig_bb.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["BB_Lower"],
                mode="lines",
                name="Lower Band",
                line=dict(dash="dash"),
                fill="tonexty",
                fillcolor="rgba(0, 128, 128, 0.10)",
            )
        )
        fig_bb.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_bb, use_container_width=True)


def main() -> None:
    """Render the Streamlit interface."""
    st.markdown('<h1 class="main-header">Stock Market Price Movement Predictor</h1>', unsafe_allow_html=True)
    st.caption(
        "Final project app for a Data Science course. This Streamlit app is the required deployment, "
        "and a separate Flask API is included as an optional bonus."
    )

    try:
        model, scaler, feature_names, model_info = load_artifacts()
    except Exception as exc:
        st.error(f"Unable to load the saved model files: {exc}")
        st.info("Run the notebook up to Step 7 first so the trained artifacts are created.")
        return

    training_tickers = model_info.get("training_tickers", DEFAULT_TRAIN_TICKERS)

    st.sidebar.header("Project Summary")
    st.sidebar.write(f"Model: {model_info.get('model_name', 'Saved model')}")
    st.sidebar.write(f"Feature count: {model_info.get('feature_count', len(feature_names))}")
    st.sidebar.write(f"Accuracy: {model_info.get('accuracy', 0) * 100:.1f}%")
    st.sidebar.write(f"F1-score: {model_info.get('f1_score', 0) * 100:.1f}%")
    if model_info.get("roc_auc") is not None:
        st.sidebar.write(f"ROC-AUC: {model_info.get('roc_auc', 0) * 100:.1f}%")
    st.sidebar.write("Training tickers: " + ", ".join(training_tickers))
    st.sidebar.caption("Bonus API: run `flask --app flask_api run` in the terminal.")

    input_mode = st.sidebar.radio("Choose input type", ["Training Universe", "Custom Ticker"])
    if input_mode == "Training Universe":
        ticker = st.sidebar.selectbox("Select a stock", training_tickers)
    else:
        ticker = st.sidebar.text_input("Enter ticker symbol", "AAPL").strip().upper()

    col1, col2 = st.columns([2.2, 1.0])

    with col1:
        st.subheader(f"Prediction for {ticker}")

        if ticker not in training_tickers:
            st.info(
                "This ticker was not part of the training set. The app can still score it, "
                "but the result is less reliable."
            )

        if st.button("Predict Next Trading Day", type="primary"):
            with st.spinner(f"Preparing live features for {ticker}..."):
                result = run_prediction(ticker, model, scaler, feature_names)

            if result.get("error"):
                st.error(result["error"])
            else:
                probabilities = result["probabilities"]
                feature_data = result["feature_data"]
                latest = feature_data.iloc[-1]

                pred_col1, pred_col2, pred_col3 = st.columns(3)
                with pred_col1:
                    if result["prediction"] == 1:
                        st.markdown('<p class="prediction-up">UP</p>', unsafe_allow_html=True)
                        st.success("The model predicts an upward move on the next trading day.")
                    else:
                        st.markdown('<p class="prediction-down">DOWN</p>', unsafe_allow_html=True)
                        st.error("The model predicts a downward move on the next trading day.")
                with pred_col2:
                    st.metric("Probability of up move", f"{probabilities[1] * 100:.1f}%")
                with pred_col3:
                    st.metric("Probability of down move", f"{probabilities[0] * 100:.1f}%")

                st.markdown("---")
                st.subheader("Latest Market Snapshot")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Current price", f"${latest['Close']:.2f}")
                with m2:
                    st.metric("Daily return", f"{latest['Daily_Return']:.2f}%")
                with m3:
                    st.metric("RSI", f"{latest['RSI']:.1f}")
                with m4:
                    st.metric("Volume ratio", f"{latest['Volume_Ratio']:.2f}")

                st.caption(f"Latest available trading day: {pd.to_datetime(latest['Date']).date()}")

                st.markdown("---")
                st.subheader("Price Chart")
                render_price_chart(feature_data, ticker)

                st.subheader("Technical Indicators")
                render_indicator_tabs(feature_data)

                with col2:
                    st.subheader("Model Notes")
                    st.write("This app uses the saved model from the notebook and scores the latest available stock data.")
                    st.write("The notebook compares multiple models and keeps the best-performing one.")

                    st.subheader("Top Features")
                    importance_rows = model_info.get("top_feature_importance", [])
                    if importance_rows:
                        st.dataframe(pd.DataFrame(importance_rows).head(5), use_container_width=True, height=220)
                    else:
                        st.write("Feature importance information is not available.")
        else:
            with col2:
                st.subheader("Project Notes")
                st.write(
                    "This project predicts next-day stock direction using historical Yahoo Finance data, "
                    "technical indicators, and a saved classification model."
                )
                st.write("Use the button on the left to generate a prediction for the selected ticker.")
                st.write("The optional Flask bonus is available separately in `flask_api.py`.")

    st.markdown("---")
    st.caption(
        "Educational use only. Predictions are based on historical market data and should not be treated as financial advice."
    )


if __name__ == "__main__":
    main()
