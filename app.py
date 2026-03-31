"""
Streamlit app for stock market price movement prediction.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stock_prediction_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "feature_names.txt"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-up {
        color: #0b8a2a;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .prediction-down {
        color: #b22222;
        font-size: 1.9rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_TRAIN_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "V",
    "JNJ",
]
REQUIRED_PRICE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
NON_FEATURE_COLUMNS = {"Date", "Target", "Open", "High", "Low", "Close", "Adj_Close"}


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output into a predictable single-level schema."""
    cleaned = df.copy()

    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [str(col[0]).strip() for col in cleaned.columns.to_flat_index()]
    else:
        cleaned.columns = [str(col).strip() for col in cleaned.columns]

    return cleaned.rename(columns={"Adj Close": "Adj_Close", "Date_": "Date"})


def download_stock_data(ticker: str, lookback_days: int = 365 * 3) -> pd.DataFrame:
    """Download one ticker's historical data from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if stock_data.empty:
        raise ValueError(f"Could not fetch data for ticker '{ticker}'.")

    stock_data = normalize_downloaded_columns(stock_data.reset_index())
    missing_columns = [col for col in REQUIRED_PRICE_COLUMNS if col not in stock_data.columns]

    if missing_columns:
        raise ValueError(
            f"Downloaded data for '{ticker}' is missing expected columns: {', '.join(missing_columns)}"
        )

    stock_data["Ticker"] = ticker
    return stock_data


def build_stock_universe(tickers: Iterable[str], lookback_days: int = 365 * 3) -> pd.DataFrame:
    """Download and concatenate multiple stock histories."""
    frames = [download_stock_data(ticker, lookback_days=lookback_days) for ticker in tickers]
    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and fill price data."""
    data = normalize_downloaded_columns(df).copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"]).reset_index(drop=True)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[["Open", "High", "Low", "Close", "Volume"]] = (
        data.groupby("Ticker")[["Open", "High", "Low", "Close", "Volume"]].transform(lambda s: s.ffill().bfill())
    )
    return data


def _calculate_indicators_for_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("Date").copy()

    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    data.loc[(loss == 0) & (gain == 0), "RSI"] = 50

    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + (bb_std * 2)
    data["BB_Lower"] = data["BB_Middle"] - (bb_std * 2)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]

    data["Volume_MA_10"] = data["Volume"].rolling(window=10).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA_10"]
    data["Momentum_5"] = data["Close"] - data["Close"].shift(5)
    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["Volatility_10"] = data["Daily_Return"].rolling(window=10).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
    data["HL_Spread"] = (data["High"] - data["Low"]) / data["Close"] * 100
    data["Price_Range"] = (data["High"] - data["Low"]) / data["Open"] * 100
    data["Gap_Open"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1) * 100
    data["Day_Of_Week"] = data["Date"].dt.day_name()

    return data.replace([np.inf, -np.inf], np.nan)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical and calendar features to each ticker's price history."""
    cleaned = clean_stock_data(df)
    frames = [_calculate_indicators_for_ticker(frame) for _, frame in cleaned.groupby("Ticker", sort=False)]
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create the next-day movement target for each ticker."""
    data = df.copy()
    future_close = data.groupby("Ticker")["Close"].shift(-1)
    data["Target"] = (future_close > data["Close"]).astype(int)
    data = data.loc[future_close.notna()].copy()
    return data.reset_index(drop=True)


def detect_outliers_zscore(df: pd.DataFrame, numeric_cols: list[str], threshold: float = 3.0) -> pd.Series:
    """Count outliers per numeric column using z-scores."""
    if not numeric_cols:
        return pd.Series(dtype=int)

    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    z_scores = np.atleast_2d(z_scores)
    counts = np.atleast_1d((z_scores > threshold).sum(axis=0))
    return pd.Series(counts, index=numeric_cols, dtype=int).sort_values(ascending=False)


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap numeric columns using the IQR rule."""
    capped = df.copy()
    bounds = {}

    for col in numeric_cols:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        capped[col] = capped[col].clip(lower_bound, upper_bound)
        bounds[col] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    return capped, pd.DataFrame(bounds).T


def encode_categorical_features(df: pd.DataFrame, categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    categorical_cols = categorical_cols or ["Ticker", "Day_Of_Week"]
    present_cols = [col for col in categorical_cols if col in df.columns]
    if not present_cols:
        return df.copy()
    return pd.get_dummies(df, columns=present_cols, drop_first=False, dtype=int)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the candidate feature columns."""
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def select_features(
    df: pd.DataFrame,
    target_col: str = "Target",
    max_features: int = 18,
    importance_threshold: float = 0.03,
    corr_threshold: float = 0.90,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """Select relevant features using importance ranking and correlation pruning."""
    feature_cols = get_feature_columns(df)
    x = df[feature_cols]
    y = df[target_col]

    selector = RandomForestClassifier(n_estimators=300, random_state=42)
    selector.fit(x, y)

    importance_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": selector.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    selected = importance_df.loc[importance_df["Importance"] >= importance_threshold, "Feature"].tolist()
    if len(selected) < 8:
        selected = importance_df.head(min(max_features, len(importance_df)))["Feature"].tolist()

    corr_matrix = x[selected].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    importance_rank = importance_df.set_index("Feature")["Importance"]
    dropped = set()

    for col in upper.columns:
        correlated = [idx for idx, value in upper[col].items() if pd.notna(value) and value > corr_threshold]
        for row in correlated:
            weaker = row if importance_rank[row] <= importance_rank[col] else col
            dropped.add(weaker)

    final_features = [feature for feature in selected if feature not in dropped]
    if len(final_features) < 8:
        final_features = importance_df.head(min(max_features, len(importance_df)))["Feature"].tolist()
        dropped = set()

    return final_features, importance_df, sorted(dropped)


def split_train_test_by_date(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Target",
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split the dataset by date to avoid look-ahead leakage."""
    date_series = pd.to_datetime(df["Date"]).dt.normalize()
    unique_dates = pd.Index(sorted(date_series.unique()))
    split_idx = int(len(unique_dates) * (1 - test_size))

    if split_idx <= 0 or split_idx >= len(unique_dates):
        raise ValueError("Could not create a train/test split from the available dates.")

    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_df = df[date_series.isin(train_dates)].copy()
    test_df = df[date_series.isin(test_dates)].copy()

    x_train = train_df[feature_cols]
    x_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return x_train, x_test, y_train, y_test, train_df, test_df


def build_time_series_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create time-series splits using unique trading dates."""
    date_series = pd.to_datetime(df["Date"]).dt.normalize().reset_index(drop=True)
    unique_dates = pd.Index(sorted(date_series.unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_date_idx, test_date_idx in tscv.split(unique_dates):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        train_idx = date_series.index[date_series.isin(train_dates)].to_numpy()
        test_idx = date_series.index[date_series.isin(test_dates)].to_numpy()
        splits.append((train_idx, test_idx))

    return splits


def get_model_catalog() -> dict[str, object]:
    """Return the baseline models used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float | None]:
    """Calculate common classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": None,
    }

    if y_prob is not None and y_true.nunique() > 1:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)

    return metrics


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Train and compare multiple baseline models."""
    results = []
    trained_models: dict[str, dict[str, object]] = {}

    for name, estimator in get_model_catalog().items():
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = clone(estimator)
        model.fit(x_train_scaled, y_train)

        y_pred = model.predict(x_test_scaled)
        y_prob = model.predict_proba(x_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)

        trained_models[name] = {
            "model": model,
            "scaler": scaler,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    results_df = pd.DataFrame(results).sort_values(["F1-Score", "Accuracy"], ascending=False).reset_index(drop=True)
    return results_df, trained_models


def cross_validate_model(model_name: str, x: pd.DataFrame, y: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Run time-series cross-validation for one model."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", get_model_catalog()[model_name])])
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(pipeline, x, y, cv=splits, scoring=scoring, n_jobs=1)

    return pd.DataFrame(
        {
            "Fold": range(1, len(scores["test_accuracy"]) + 1),
            "Accuracy": scores["test_accuracy"],
            "Precision": scores["test_precision"],
            "Recall": scores["test_recall"],
            "F1-Score": scores["test_f1"],
            "ROC-AUC": scores["test_roc_auc"],
        }
    )


def get_param_grid(model_name: str) -> dict[str, list[object]]:
    """Return the GridSearchCV parameter grid for a model."""
    grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [4, 8, None],
            "model__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 150, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "K-Nearest Neighbors": {
            "model__n_neighbors": [5, 7, 9],
            "model__weights": ["uniform", "distance"],
        },
        "Naive Bayes": {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7],
        },
        "SVM": {
            "model__C": [0.5, 1.0, 2.0],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        },
    }
    return grids[model_name]


def tune_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV:
    """Tune the selected model using GridSearchCV."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", get_model_catalog()[model_name])])
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=get_param_grid(model_name),
        scoring="f1",
        cv=splits,
        n_jobs=1,
        refit=True,
    )
    grid_search.fit(x_train, y_train)
    return grid_search


def save_artifacts(
    model: object,
    scaler: StandardScaler,
    feature_names: list[str],
    model_info: dict[str, object],
) -> None:
    """Persist the model artifacts used by the Streamlit app."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    FEATURE_NAMES_PATH.write_text("\n".join(feature_names), encoding="utf-8")
    joblib.dump(model_info, MODEL_INFO_PATH)


def load_artifacts() -> tuple[object, StandardScaler, list[str], dict[str, object]]:
    """Load the saved model artifacts from disk."""
    missing_files = [path.name for path in [MODEL_PATH, SCALER_PATH] if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_info = joblib.load(MODEL_INFO_PATH) if MODEL_INFO_PATH.exists() else {}
    if not isinstance(model_info, dict):
        model_info = {}

    if model_info.get("features"):
        feature_names = list(model_info["features"])
    elif FEATURE_NAMES_PATH.exists():
        feature_names = [line.strip() for line in FEATURE_NAMES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        raise FileNotFoundError("Feature names are missing. Re-run training to regenerate metadata.")

    return model, scaler, feature_names, model_info


def prepare_inference_dataset(
    ticker: str,
    feature_names: list[str],
    lookback_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare live prediction features for one ticker."""
    raw = download_stock_data(ticker, lookback_days=lookback_days)
    features = calculate_technical_indicators(raw).dropna().reset_index(drop=True)
    if features.empty:
        raise ValueError("Not enough recent rows to calculate technical indicators.")

    encoded = encode_categorical_features(features)
    x_live = encoded.reindex(columns=feature_names).ffill().bfill().fillna(0.0)
    return features, x_live


def predict_with_confidence(model: object, scaler: StandardScaler, feature_frame: pd.DataFrame) -> tuple[int, np.ndarray]:
    """Return the predicted class and down/up probabilities."""
    scaled = scaler.transform(feature_frame.astype(float))
    prediction = int(model.predict(scaled)[0])

    if hasattr(model, "predict_proba"):
        raw_prob = model.predict_proba(scaled)[0]
        class_index = {int(label): idx for idx, label in enumerate(model.classes_)}
        probability_down = float(raw_prob[class_index.get(0, 0)])
        probability_up = float(raw_prob[class_index.get(1, len(raw_prob) - 1)])
        return prediction, np.array([probability_down, probability_up])

    return prediction, np.array([1.0, 0.0]) if prediction == 0 else np.array([0.0, 1.0])


def train_project_model(
    tickers: list[str] | None = None,
    lookback_days: int = 365 * 3,
    test_size: float = 0.20,
) -> dict[str, object]:
    """Train the full project model and save deployment artifacts."""
    tickers = tickers or DEFAULT_TRAIN_TICKERS

    raw_data = build_stock_universe(tickers, lookback_days=lookback_days)
    engineered = calculate_technical_indicators(raw_data)
    labeled = add_target_column(engineered).dropna().reset_index(drop=True)

    numeric_cols = labeled.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "Target"]
    capped, _ = cap_outliers_iqr(labeled, numeric_cols)
    encoded = encode_categorical_features(capped)

    final_features, feature_importance, dropped_correlated = select_features(encoded)
    x_train, x_test, y_train, y_test, train_df, test_df = split_train_test_by_date(
        encoded,
        feature_cols=final_features,
        test_size=test_size,
    )

    results_df, trained_models = compare_models(x_train, y_train, x_test, y_test)
    best_model_name = str(results_df.iloc[0]["Model"])
    best_base = trained_models[best_model_name]
    base_f1 = float(results_df.iloc[0]["F1-Score"])

    train_subset = train_df[["Date"] + final_features + ["Target"]].copy()
    cv_splits = build_time_series_splits(train_subset)
    cv_df = cross_validate_model(best_model_name, x_train, y_train, cv_splits)
    grid_search = tune_model(best_model_name, x_train, y_train, cv_splits)

    tuned_scaler = grid_search.best_estimator_.named_steps["scaler"]
    tuned_model = grid_search.best_estimator_.named_steps["model"]
    tuned_pred = tuned_model.predict(tuned_scaler.transform(x_test))
    tuned_prob = tuned_model.predict_proba(tuned_scaler.transform(x_test))[:, 1] if hasattr(tuned_model, "predict_proba") else None
    tuned_metrics = evaluate_predictions(y_test, tuned_pred, tuned_prob)

    if tuned_metrics["F1-Score"] >= base_f1:
        final_model = tuned_model
        final_scaler = tuned_scaler
        final_metrics = tuned_metrics
        final_model_name = f"Tuned {best_model_name}"
    else:
        final_model = best_base["model"]
        final_scaler = best_base["scaler"]
        final_metrics = results_df.iloc[0].to_dict()
        final_model_name = best_model_name

    model_info = {
        "project_title": "Stock Market Price Movement Prediction",
        "model_name": final_model_name,
        "training_tickers": tickers,
        "feature_count": len(final_features),
        "features": final_features,
        "accuracy": float(final_metrics["Accuracy"]),
        "precision": float(final_metrics["Precision"]),
        "recall": float(final_metrics["Recall"]),
        "f1_score": float(final_metrics["F1-Score"]),
        "roc_auc": None if final_metrics["ROC-AUC"] is None else float(final_metrics["ROC-AUC"]),
        "best_baseline_model": best_model_name,
        "test_rows": int(len(x_test)),
        "train_rows": int(len(x_train)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "top_feature_importance": feature_importance.head(10).to_dict(orient="records"),
        "dropped_correlated_features": dropped_correlated,
        "cross_validation": {
            "accuracy_mean": float(cv_df["Accuracy"].mean()),
            "precision_mean": float(cv_df["Precision"].mean()),
            "recall_mean": float(cv_df["Recall"].mean()),
            "f1_mean": float(cv_df["F1-Score"].mean()),
            "roc_auc_mean": float(cv_df["ROC-AUC"].mean()),
        },
        "grid_search_best_params": grid_search.best_params_,
    }

    save_artifacts(final_model, final_scaler, final_features, model_info)
    return model_info


def run_prediction(ticker: str, model: object, scaler: StandardScaler, feature_names: list[str]) -> dict[str, object]:
    """Prepare live features and score the latest row."""
    try:
        feature_data, model_frame = prepare_inference_dataset(ticker, feature_names, lookback_days=365)
        latest_features = model_frame.iloc[[-1]]
        prediction, probabilities = predict_with_confidence(model, scaler, latest_features)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "feature_data": feature_data,
            "model_frame": model_frame,
        }
    except Exception as exc:
        return {"error": str(exc)}


def render_price_chart(data: pd.DataFrame, ticker: str) -> None:
    """Render the candlestick chart and moving averages."""
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
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_indicator_tabs(data: pd.DataFrame) -> None:
    """Render RSI, MACD, and Bollinger Band charts."""
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])

    with tab1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=320, template="plotly_white")
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
        fig_macd.update_layout(height=320, template="plotly_white")
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
        fig_bb.update_layout(height=320, template="plotly_white")
        st.plotly_chart(fig_bb, use_container_width=True)


def main() -> None:
    """Render the Streamlit interface."""
    st.markdown('<h1 class="main-header">Stock Market Price Movement Predictor</h1>', unsafe_allow_html=True)

    st.sidebar.header("Model Controls")

    try:
        model, scaler, feature_names, model_info = load_artifacts()
    except FileNotFoundError:
        st.warning("Model files are missing. Train the project model to continue.")
        if st.button("Train Model Now"):
            with st.spinner("Training the full project pipeline. This can take a few minutes."):
                try:
                    train_project_model()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
                    return
            st.success("Model trained successfully. Reloading the app.")
            st.rerun()
        return
    except Exception as exc:
        st.error(f"Unable to load saved model artifacts: {exc}")
        return

    training_tickers = model_info.get("training_tickers", DEFAULT_TRAIN_TICKERS)

    st.sidebar.subheader("Training Universe")
    st.sidebar.write(", ".join(training_tickers))

    st.sidebar.subheader("Model Summary")
    st.sidebar.write(f"Model: {model_info.get('model_name', 'Unknown')}")
    st.sidebar.write(f"Feature count: {model_info.get('feature_count', len(feature_names))}")
    st.sidebar.write(f"Accuracy: {model_info.get('accuracy', 0) * 100:.1f}%")
    st.sidebar.write(f"F1-score: {model_info.get('f1_score', 0) * 100:.1f}%")
    if model_info.get("roc_auc") is not None:
        st.sidebar.write(f"ROC-AUC: {model_info.get('roc_auc', 0) * 100:.1f}%")

    input_mode = st.sidebar.radio("Ticker input", ["Training Universe", "Custom Ticker"])
    if input_mode == "Training Universe":
        ticker = st.sidebar.selectbox("Select a stock", training_tickers)
    else:
        ticker = st.sidebar.text_input("Enter ticker symbol", "AAPL").strip().upper()

    if st.sidebar.button("Retrain Artifacts"):
        with st.spinner("Retraining the full project pipeline."):
            try:
                train_project_model()
            except Exception as exc:
                st.sidebar.error(f"Retraining failed: {exc}")
                return
        st.sidebar.success("Artifacts updated.")
        st.rerun()

    col1, col2 = st.columns([2.4, 1.1])

    with col1:
        st.subheader(f"Prediction for {ticker}")

        if ticker not in training_tickers:
            st.info(
                "This ticker was not part of the training universe. The model can still score it, "
                "but the prediction is less reliable."
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
                    st.subheader("Selected Feature Snapshot")
                    latest_features = result["model_frame"].iloc[-1]
                    snapshot = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Value": [float(latest_features.get(feature, 0.0)) for feature in feature_names],
                        }
                    ).sort_values("Feature")
                    st.dataframe(snapshot, use_container_width=True, height=420)

                    st.subheader("Top Ranked Features")
                    importance_rows = model_info.get("top_feature_importance", [])
                    if importance_rows:
                        st.dataframe(pd.DataFrame(importance_rows), use_container_width=True, height=260)
                    else:
                        st.write("Feature importance metadata is not available.")

                    dropped = model_info.get("dropped_correlated_features", [])
                    if dropped:
                        st.caption("Dropped during correlation pruning: " + ", ".join(dropped))
        else:
            with col2:
                st.subheader("Project Notes")
                st.write(
                    "This model is trained on a basket of large-cap US stocks using technical indicators and "
                    "encoded categorical features."
                )
                st.write("Use the button on the left to score the latest available market data for the selected ticker.")

    st.markdown("---")
    st.caption(
        "Educational use only. Predictions are model outputs based on historical market data and should not be treated as financial advice."
    )


if __name__ == "__main__":
    main()
