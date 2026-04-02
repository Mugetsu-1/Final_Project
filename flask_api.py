"""Optional Flask REST API for the course project bonus requirement."""

from __future__ import annotations

from flask import Flask, jsonify, request

from project_core import DEFAULT_TRAIN_TICKERS, load_artifacts, run_prediction

app = Flask(__name__)

RUNTIME_ERROR: str | None = None
MODEL = None
SCALER = None
FEATURE_NAMES: list[str] = []
MODEL_INFO: dict[str, object] = {}


def refresh_runtime() -> None:
    """Load the saved artifacts once so the API can serve predictions."""
    global FEATURE_NAMES, MODEL, MODEL_INFO, RUNTIME_ERROR, SCALER

    try:
        MODEL, SCALER, FEATURE_NAMES, MODEL_INFO = load_artifacts()
        RUNTIME_ERROR = None
    except Exception as exc:
        MODEL = None
        SCALER = None
        FEATURE_NAMES = []
        MODEL_INFO = {}
        RUNTIME_ERROR = str(exc)


refresh_runtime()


@app.get("/")
def home() -> tuple[object, int]:
    """Return a simple description of the API."""
    status = "ready" if RUNTIME_ERROR is None else "error"
    return (
        jsonify(
            {
                "project": "Stock Market Price Movement Prediction",
                "status": status,
                "main_deployment": "Streamlit",
                "bonus_deployment": "Flask REST API",
                "training_tickers": MODEL_INFO.get("training_tickers", DEFAULT_TRAIN_TICKERS),
                "routes": {
                    "health": "/health",
                    "predict_get": "/predict?ticker=AAPL",
                    "predict_post": "/predict",
                },
            }
        ),
        200,
    )


@app.get("/health")
def health() -> tuple[object, int]:
    """Health check route for quick testing."""
    if RUNTIME_ERROR is not None:
        return jsonify({"status": "error", "message": RUNTIME_ERROR}), 500
    return jsonify({"status": "ok", "model_name": MODEL_INFO.get("model_name", "Saved model")}), 200


@app.route("/predict", methods=["GET", "POST"])
def predict() -> tuple[object, int]:
    """Return a prediction for the requested ticker."""
    if RUNTIME_ERROR is not None:
        return jsonify({"error": RUNTIME_ERROR}), 500

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        ticker = str(payload.get("ticker", "")).strip().upper()
    else:
        ticker = request.args.get("ticker", "").strip().upper()

    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol."}), 400

    result = run_prediction(ticker, MODEL, SCALER, FEATURE_NAMES)
    if result.get("error"):
        return jsonify({"error": result["error"]}), 400

    latest_row = result["feature_data"].iloc[-1]
    probabilities = result["probabilities"]

    return (
        jsonify(
            {
                "ticker": ticker,
                "prediction": int(result["prediction"]),
                "prediction_label": "UP" if int(result["prediction"]) == 1 else "DOWN",
                "probability_up": round(float(probabilities[1]), 4),
                "probability_down": round(float(probabilities[0]), 4),
                "latest_trading_day": str(latest_row["Date"].date()),
                "latest_close": round(float(latest_row["Close"]), 4),
                "latest_rsi": round(float(latest_row["RSI"]), 4),
                "model_name": MODEL_INFO.get("model_name", "Saved model"),
                "is_training_ticker": ticker in MODEL_INFO.get("training_tickers", DEFAULT_TRAIN_TICKERS),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
