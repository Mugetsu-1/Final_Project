# Stock Market Price Movement Prediction

This project is a complete data science workflow that predicts whether a stock will move up or down on the next trading day. It follows the structure requested in the certification PDF: data collection, cleaning, transformation, EDA, feature selection, model development, evaluation, hyperparameter tuning, and Streamlit deployment on localhost.

## Project Scope

- Dataset: 3 years of Yahoo Finance historical data for a basket of large-cap US stocks
- Problem type: Binary classification
- Target: `1` if the next trading day's close is higher than the current close, otherwise `0`
- Deployment: Streamlit application running on localhost

## Main Features

- Multi-stock training dataset built from Yahoo Finance
- Technical indicators such as RSI, MACD, Bollinger Bands, moving averages, momentum, and volatility
- Categorical feature encoding for `Ticker` and `Day_Of_Week`
- Feature selection using Random Forest importance and correlation pruning
- Multiple baseline models compared before tuning the best candidate
- Time-aware validation using date-based splitting and time-series cross-validation
- Streamlit app for live prediction on the latest downloaded market data

## Project Files

```text
DataScienceFinalPrj/
├── Stock_Market_Prediction.ipynb   # Main notebook following the assignment steps
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── Data Science Certification Task.pdf
├── stock_prediction_model.pkl      # Generated after training
├── scaler.pkl                      # Generated after training
├── feature_names.txt               # Generated after training
└── model_info.pkl                  # Generated after training
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook if you want the full analysis workflow:

```bash
jupyter notebook
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. If the model files do not exist yet, click `Train Model Now` in the app.

## Workflow Covered In The Notebook

1. Data Collection and Exploration
2. Data Cleaning and Transformation
3. Exploratory Data Analysis
4. Feature Selection
5. Model Development
6. Model Evaluation and Hyperparameter Tuning
7. Saving The Final Model
8. Streamlit Deployment Instructions

## Modeling Notes

- The training dataset is built from multiple tickers so the app can score more than one stock honestly.
- The notebook uses a time-based split by trading date to reduce look-ahead leakage.
- Cross-validation is also time-aware.
- The final saved artifacts are shared by the notebook and the Streamlit app.
- The submission is centered on the two required deliverables: one notebook and one Streamlit app.

## Important Note

This project is for educational purposes only. It demonstrates a full machine learning workflow on financial time-series data, but it is not investment advice.
