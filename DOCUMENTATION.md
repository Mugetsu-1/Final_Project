# Stock Market Prediction - Project Documentation

## Project Goal

Predict whether stock price direction will be **UP** or **DOWN** for:
- Next trading day (1d)
- Next trading week (5d)

This project follows the formal objective workflow and keeps implementation clean:
- Notebook for all data science work
- Streamlit app for inference UI only

## Notebook Responsibilities

`Stock_Market_Prediction.ipynb` handles:
1. Data loading from Yahoo Finance
2. Data cleaning and transformation
3. EDA and key visual analysis
4. Feature engineering and selection
5. Model development and comparison
6. Time-series cross-validation and randomized hyperparameter search
7. Validation-set threshold tuning and final holdout evaluation
8. Saving artifacts for deployment

### Optional GPU Mode

- The notebook includes an optional XGBoost candidate with GPU preference.
- Enable it via `ENABLE_GPU = True` in the setup section.
- If CUDA is unavailable, training continues with CPU-capable candidates.

## Streamlit Responsibilities

`app.py` handles:
- Loading saved artifacts
- Live ticker selection and prediction
- Classical ML inference and ARIMA benchmark mode
- Displaying:
  - UP (green)
  - DOWN (red)
  - model outputs and charts

## Main Dependencies

- pandas, numpy
- scikit-learn
- xgboost (optional GPU acceleration candidate)
- matplotlib, seaborn
- yfinance
- streamlit, plotly
- statsmodels
- joblib

## Output Files

- `cleaned_stock_data.csv`
- `stock_prediction_model.pkl`, `scaler.pkl`, `model_info.pkl`
- `stock_prediction_model_weekly.pkl`, `scaler_weekly.pkl`, `model_info_weekly.pkl`

## Quality and Readability

- The notebook is structured in clear numbered steps.
- Each stage has short, focused code blocks.
- Pipeline is reproducible by running cells top-to-bottom.
- Data cleaning explicitly includes missing-value handling and IQR-based outlier treatment.
- Feature preparation explicitly includes categorical encoding for model-ready inputs.
