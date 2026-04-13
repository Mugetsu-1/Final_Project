from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stock_prediction_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"
WEEKLY_MODEL_PATH = BASE_DIR / "stock_prediction_model_weekly.pkl"
WEEKLY_SCALER_PATH = BASE_DIR / "scaler_weekly.pkl"
WEEKLY_MODEL_INFO_PATH = BASE_DIR / "model_info_weekly.pkl"
DEFAULT_TRAIN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

ALGO_CLASSICAL = "Classical ML"
ALGO_ARIMA = "ARIMA Benchmark (Exploratory)"


def artifact_version(paths: list[Path]) -> tuple[int, ...]:
    values: list[int] = []
    for path in paths:
        if not path.exists():
            values.extend([-1, -1])
            continue
        stat = path.stat()
        values.extend([int(stat.st_mtime_ns), int(stat.st_size)])
    return tuple(values)


@st.cache_resource(show_spinner=False)
def load_classical_artifacts(horizon: str, version_key: tuple[int, ...]):
    # Keeps cache tied to artifact timestamps/sizes so retraining reloads without app restart.
    _ = version_key
    if horizon == "1d":
        model_path = MODEL_PATH
        scaler_path = SCALER_PATH
        model_info_path = MODEL_INFO_PATH
    else:
        model_path = WEEKLY_MODEL_PATH
        scaler_path = WEEKLY_SCALER_PATH
        model_info_path = WEEKLY_MODEL_INFO_PATH

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_info = joblib.load(model_info_path) if model_info_path.exists() else {}
    if not isinstance(model_info, dict) or not model_info.get("features"):
        raise ValueError("model_info artifact must include a non-empty 'features' list. Re-run notebook export.")
    features = list(model_info["features"])
    return model, scaler, features, model_info


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).strip() for col in df.columns.to_flat_index()]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    return df.rename(columns={"Date_": "Date", "Adj Close": "Adj_Close"})


def download_market_data(ticker: str, start: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, progress=False, auto_adjust=False).reset_index()
    data = normalize_downloaded_columns(data)
    if data.empty:
        raise ValueError(f"No market data found for ticker {ticker}.")
    return data


def direction_confidence_from_pct_change(pct_change: float) -> tuple[float, float]:
    # Heuristic score used only for exploratory ARIMA direction display.
    confidence = float(np.clip(0.5 + (abs(pct_change) * 4.0), 0.5, 0.99))
    if pct_change >= 0:
        return confidence, 1.0 - confidence
    return 1.0 - confidence, confidence


def arima_predict_direction(close_series: pd.Series, steps: int = 1, order: tuple[int, int, int] = (3, 1, 1)) -> dict:
    if len(close_series) < 80:
        raise ValueError("Not enough data for ARIMA. Need at least 80 rows of close prices.")

    model = ARIMA(close_series.astype(float), order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=steps)
    predicted_price = float(forecast.iloc[-1])
    last_price = float(close_series.iloc[-1])
    pct_change = ((predicted_price - last_price) / last_price) * 100.0
    up_prob, down_prob = direction_confidence_from_pct_change(pct_change)
    return {
        "predicted_price": predicted_price,
        "last_price": last_price,
        "pct_change": pct_change,
        "pred": 1 if pct_change >= 0 else 0,
        "up_prob": up_prob,
        "down_prob": down_prob,
    }


def build_indicator_data(ticker: str) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    data = download_market_data(ticker=ticker, start=start)

    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["Gap_Open"] = ((data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)) * 100
    data["HL_Spread"] = ((data["High"] - data["Low"]) / data["Close"]) * 100
    data["Price_Range"] = ((data["High"] - data["Low"]) / data["Open"]) * 100
    data["Momentum_5"] = data["Close"] - data["Close"].shift(5)
    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["Momentum_20"] = data["Close"] - data["Close"].shift(20)
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["MA_50"] = data["Close"].rolling(50).mean()
    data["MA_Ratio_5_20"] = data["MA_5"] / data["MA_20"]
    data["MA_Ratio_10_50"] = data["MA_10"] / data["MA_50"]
    data["Ret_1"] = data["Close"].pct_change(1) * 100
    data["Ret_2"] = data["Close"].pct_change(2) * 100
    data["Ret_3"] = data["Close"].pct_change(3) * 100
    data["Ret_5"] = data["Close"].pct_change(5) * 100
    data["Ret_10"] = data["Close"].pct_change(10) * 100
    data["Volatility_5"] = data["Daily_Return"].rolling(5).std()
    data["Volatility_10"] = data["Daily_Return"].rolling(10).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(20).std()
    data["Volume_MA_10"] = data["Volume"].rolling(10).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA_10"]
    data["Volume_Change"] = data["Volume"].pct_change() * 100

    delta = data["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    rs = gains.rolling(14).mean() / losses.rolling(14).mean().replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    mean_20 = data["Close"].rolling(20).mean()
    std_20 = data["Close"].rolling(20).std()
    data["BB_Upper"] = mean_20 + 2 * std_20
    data["BB_Lower"] = mean_20 - 2 * std_20
    data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]
    data["BB_PctB"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"])

    tr1 = (data["High"] - data["Low"]).abs()
    tr2 = (data["High"] - data["Close"].shift(1)).abs()
    tr3 = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR_14"] = true_range.rolling(14).mean()
    data["Day_Of_Week"] = data["Date"].dt.dayofweek.astype(float)

    model_data = data.dropna().reset_index(drop=True)
    if model_data.empty:
        raise ValueError("Not enough rows to compute indicators. Try another ticker.")
    return model_data


def latest_feature_row(ticker: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = build_indicator_data(ticker)

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
    st.write("Predict next-day or next-week stock direction with Classical ML and ARIMA time series.")

    horizon_label = st.sidebar.radio("Prediction horizon", ["Next Day (1d)", "Next Week (5d)"])
    horizon = "1d" if horizon_label == "Next Day (1d)" else "5d"
    algo = st.sidebar.radio("Algorithm", [ALGO_CLASSICAL, ALGO_ARIMA])

    model = None
    scaler = None
    feature_names = []
    model_info: dict = {}

    if algo == ALGO_CLASSICAL:
        try:
            if horizon == "1d":
                key = artifact_version([MODEL_PATH, SCALER_PATH, MODEL_INFO_PATH])
            else:
                key = artifact_version([WEEKLY_MODEL_PATH, WEEKLY_SCALER_PATH, WEEKLY_MODEL_INFO_PATH])
            model, scaler, feature_names, model_info = load_classical_artifacts(horizon, key)
        except Exception as exc:
            st.error(f"Classical model files not ready: {exc}")
            return

    if algo == ALGO_CLASSICAL:
        training_tickers = model_info.get("training_tickers", DEFAULT_TRAIN_TICKERS) if isinstance(model_info, dict) else DEFAULT_TRAIN_TICKERS
    else:
        training_tickers = DEFAULT_TRAIN_TICKERS

    st.sidebar.header("Available Tickers")
    ticker = st.sidebar.selectbox("Choose ticker", training_tickers)
    custom_ticker = st.sidebar.text_input("Or type custom ticker", "").strip().upper()
    if custom_ticker:
        ticker = custom_ticker

    st.sidebar.markdown("---")
    if algo == ALGO_CLASSICAL:
        st.sidebar.write(f"Model: {model_info.get('model_name', 'Saved model') if isinstance(model_info, dict) else 'Saved model'}")
        if isinstance(model_info, dict):
            st.sidebar.write(f"Accuracy: {model_info.get('accuracy', 0) * 100:.1f}%")
            st.sidebar.write(f"F1: {model_info.get('f1_score', 0) * 100:.1f}%")
    else:
        st.sidebar.write("Model: ARIMA(3,1,1) benchmark")
        st.sidebar.caption("Exploratory baseline only. Confidence values are heuristic, not calibrated probabilities.")

    button_text = "Predict Next Day" if horizon == "1d" else "Predict Next Week"
    if st.button(button_text):
        try:
            horizon_steps = 1 if horizon == "1d" else 5
            if algo == ALGO_CLASSICAL:
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
                predicted_price = None
                last_price = None
                pct_change = None
            elif algo == ALGO_ARIMA:
                chart_data = build_indicator_data(ticker)
                result = arima_predict_direction(chart_data["Close"], steps=horizon_steps, order=(3, 1, 1))
                pred = int(result["pred"])
                up_prob = float(result["up_prob"])
                down_prob = float(result["down_prob"])
                predicted_price = float(result["predicted_price"])
                last_price = float(result["last_price"])
                pct_change = float(result["pct_change"])

            horizon_text = "next trading day" if horizon == "1d" else "next trading week (5 trading days)"
            if pred == 1:
                st.success(f"Prediction for {horizon_text}: UP")
            else:
                st.error(f"Prediction for {horizon_text}: DOWN")
            if algo == ALGO_CLASSICAL:
                st.write(f"Probability of UP: {up_prob * 100:.2f}%")
                st.write(f"Probability of DOWN: {down_prob * 100:.2f}%")
                threshold = float(model_info.get("decision_threshold", 0.5)) if isinstance(model_info, dict) else 0.5
                st.caption(f"Decision threshold: {threshold:.2f}")
            else:
                st.write(f"Heuristic confidence of UP: {up_prob * 100:.2f}%")
                st.write(f"Heuristic confidence of DOWN: {down_prob * 100:.2f}%")
                st.caption("ARIMA confidence values are derived from forecast magnitude and are not calibrated probabilities.")
                st.write(f"Last close: {last_price:.2f}")
                st.write(f"Forecast close: {predicted_price:.2f}")
                st.write(f"Expected change: {pct_change:.2f}%")
            st.markdown("---")
            render_charts(chart_data, ticker)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
