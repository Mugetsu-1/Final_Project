# Stock Market Prediction - Architecture

**Version:** 2.0  
**Date:** April 2026

## 1. System Overview

```text
Yahoo Finance (yfinance)
        │
        ▼
Notebook: Stock_Market_Prediction.ipynb
  - Data collection
  - Cleaning
  - EDA
  - Feature engineering
  - Model training/evaluation
  - Artifact export
        │
        ▼
Saved Artifacts (.pkl/.csv)
        │
        ▼
Streamlit App (app.py)
  - Loads saved artifacts
  - Live data fetch for selected ticker
        - Classical ML prediction OR ARIMA benchmark forecast
  - Displays prediction and charts
```

## 2. Training Pipeline (Notebook)

1. Download multi-ticker OHLCV data.
2. Normalize columns and clean missing values.
3. Export `cleaned_stock_data.csv`.
4. Perform EDA (price trends, return distribution, correlations).
5. Build engineered technical indicators.
6. Create targets for 1d and 5d horizons.
7. Train multiple candidate models.
   - Includes optional XGBoost GPU candidate (CUDA when available).
8. Apply TimeSeriesSplit cross-validation with RandomizedSearchCV.
9. Tune decision thresholds on validation split only.
10. Evaluate selected models on holdout test split.
11. Save best model artifacts for both horizons.

## 3. Inference Pipeline (Streamlit)

1. User selects horizon and algorithm.
2. App loads the corresponding model/scaler/features.
3. App downloads latest market data for selected ticker.
4. App computes the same engineered features as notebook.
5. App predicts direction and outputs class probabilities (Classical ML) or heuristic confidence (ARIMA benchmark).
6. App renders prediction and technical charts.

## 4. Artifacts Contract

Daily (1d):
- `stock_prediction_model.pkl`
- `scaler.pkl`
- `model_info.pkl`

Weekly (5d):
- `stock_prediction_model_weekly.pkl`
- `scaler_weekly.pkl`
- `model_info_weekly.pkl`

Shared:
- `cleaned_stock_data.csv`

## 5. Design Principles

- Notebook is the single source of truth for training.
- App is kept simple and focused on inference/visualization.
- Feature generation must remain consistent between notebook and app.
- Time-aware splitting is used to reduce look-ahead leakage.
- Threshold tuning is performed on validation data, not on final test data.
- GPU acceleration is optional and limited to notebook training candidates.
