from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stock_prediction_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "feature_names.txt"
WEEKLY_MODEL_PATH = BASE_DIR / "stock_prediction_model_weekly.pkl"
WEEKLY_SCALER_PATH = BASE_DIR / "scaler_weekly.pkl"
WEEKLY_MODEL_INFO_PATH = BASE_DIR / "model_info_weekly.pkl"
WEEKLY_FEATURE_NAMES_PATH = BASE_DIR / "feature_names_weekly.txt"
DEFAULT_TRAIN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).strip() for col in df.columns.to_flat_index()]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    return df.rename(columns={"Date_": "Date", "Adj Close": "Adj_Close"})


def load_artifacts(horizon: str):
    if horizon == "1d":
        model_path = MODEL_PATH
        scaler_path = SCALER_PATH
        model_info_path = MODEL_INFO_PATH
        feature_names_path = FEATURE_NAMES_PATH
    else:
        model_path = WEEKLY_MODEL_PATH
        scaler_path = WEEKLY_SCALER_PATH
        model_info_path = WEEKLY_MODEL_INFO_PATH
        feature_names_path = WEEKLY_FEATURE_NAMES_PATH

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_info = joblib.load(model_info_path) if model_info_path.exists() else {}
    if isinstance(model_info, dict) and model_info.get("features"):
        features = list(model_info["features"])
    else:
        features = [line.strip() for line in feature_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return model, scaler, features, model_info


def latest_feature_row(ticker: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = (pd.Timestamp.today() - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, progress=False, auto_adjust=False).reset_index()
    data = normalize_downloaded_columns(data)
    if data.empty:
        raise ValueError("No market data found for this ticker.")

    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["Gap_Open"] = ((data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)) * 100
    data["HL_Spread"] = ((data["High"] - data["Low"]) / data["Close"]) * 100
    data["Price_Range"] = ((data["High"] - data["Low"]) / data["Open"]) * 100
    data["Momentum_5"] = data["Close"] - data["Close"].shift(5)
    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["MA_50"] = data["Close"].rolling(50).mean()
    data["Volatility_10"] = data["Daily_Return"].rolling(10).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(20).std()
    data["Volume_MA_10"] = data["Volume"].rolling(10).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA_10"]

    delta = data["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    rs = gains.rolling(14).mean() / losses.rolling(14).mean().replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    mean_20 = data["Close"].rolling(20).mean()
    std_20 = data["Close"].rolling(20).std()
    data["BB_Upper"] = mean_20 + 2 * std_20
    data["BB_Lower"] = mean_20 - 2 * std_20
    data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]

    model_data = data.dropna().reset_index(drop=True)
    if model_data.empty:
        raise ValueError("Not enough rows to compute indicators. Try another ticker.")

    x = model_data.reindex(columns=features).ffill().bfill().fillna(0.0)
    return x.tail(1), model_data


def render_charts(data: pd.DataFrame, ticker: str) -> None:
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )
    price_fig.add_trace(go.Scatter(x=data["Date"], y=data["MA_20"], mode="lines", name="MA 20"))
    price_fig.add_trace(go.Scatter(x=data["Date"], y=data["MA_50"], mode="lines", name="MA 50"))
    price_fig.update_layout(title=f"{ticker} Price + Moving Averages", template="plotly_white", height=420)
    st.plotly_chart(price_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines", name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        rsi_fig.update_layout(title="RSI", template="plotly_white", height=300)
        st.plotly_chart(rsi_fig, use_container_width=True)

    with c2:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], mode="lines", name="MACD"))
        macd_fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], mode="lines", name="Signal"))
        macd_fig.update_layout(title="MACD", template="plotly_white", height=300)
        st.plotly_chart(macd_fig, use_container_width=True)


def main():
    st.title("Stock Direction Predictor")
    st.write("Simple localhost app for next-day or next-week UP/DOWN prediction with notebook-style analysis charts.")

    horizon_label = st.sidebar.radio("Prediction horizon", ["Next Day (1d)", "Next Week (5d)"])
    horizon = "1d" if horizon_label == "Next Day (1d)" else "5d"

    try:
        model, scaler, feature_names, model_info = load_artifacts(horizon)
    except Exception as exc:
        st.error(f"Model files not ready: {exc}")
        return

    training_tickers = model_info.get("training_tickers", DEFAULT_TRAIN_TICKERS) if isinstance(model_info, dict) else DEFAULT_TRAIN_TICKERS
    st.sidebar.header("Available Tickers")
    ticker = st.sidebar.selectbox("Choose ticker", training_tickers)
    custom_ticker = st.sidebar.text_input("Or type custom ticker", "").strip().upper()
    if custom_ticker:
        ticker = custom_ticker

    st.sidebar.markdown("---")
    st.sidebar.write(f"Model: {model_info.get('model_name', 'Saved model') if isinstance(model_info, dict) else 'Saved model'}")
    if isinstance(model_info, dict):
        st.sidebar.write(f"Accuracy: {model_info.get('accuracy', 0) * 100:.1f}%")
        st.sidebar.write(f"F1: {model_info.get('f1_score', 0) * 100:.1f}%")

    button_text = "Predict Next Day" if horizon == "1d" else "Predict Next Week"
    if st.button(button_text):
        try:
            x_live, chart_data = latest_feature_row(ticker, feature_names)
            x_scaled = scaler.transform(x_live.astype(float))
            threshold = float(model_info.get("decision_threshold", 0.5)) if isinstance(model_info, dict) else 0.5
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_scaled)[0]
                idx = {int(c): i for i, c in enumerate(model.classes_)}
                down_prob = float(probs[idx.get(0, 0)])
                up_prob = float(probs[idx.get(1, len(probs) - 1)])
                pred = 1 if up_prob >= threshold else 0
            else:
                pred = int(model.predict(x_scaled)[0])
                down_prob = 1.0 if pred == 0 else 0.0
                up_prob = 1.0 if pred == 1 else 0.0

            horizon_text = "next trading day" if horizon == "1d" else "next trading week (5 trading days)"
            st.success(f"Prediction for {horizon_text}: " + ("UP" if pred == 1 else "DOWN"))
            st.write(f"Probability of UP: {up_prob * 100:.2f}%")
            st.write(f"Probability of DOWN: {down_prob * 100:.2f}%")
            st.caption(f"Decision threshold: {threshold:.2f}")
            st.markdown("---")
            render_charts(chart_data, ticker)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
