# Stock Market Price Movement Prediction

This project is a complete data science workflow that predicts whether a stock will move up or down on the next trading day. It follows the structure requested in the certification PDF: data collection, cleaning, transformation, EDA, feature selection, model development, evaluation, hyperparameter tuning, and Streamlit deployment on localhost.

This version also includes two bonus items from the PDF:
- additional model comparison in the notebook
- a simple Flask REST API for predictions

Docker is not included in this submission.

## Project Scope

- Dataset: 3 years of Yahoo Finance historical data for a basket of large-cap US stocks
- Problem type: Binary classification
- Target: `1` if the next trading day's close is higher than the current close, otherwise `0`
- Main deployment: Streamlit application running on localhost
- Bonus deployment: Flask REST API running on localhost

## Main Features

- Multi-stock training dataset built from Yahoo Finance
- Technical indicators such as RSI, MACD, Bollinger Bands, moving averages, momentum, and volatility
- Categorical feature encoding for `Ticker` and `Day_Of_Week`
- Feature selection using Random Forest importance and correlation pruning
- Multiple baseline models compared before tuning the best candidate
- Time-aware validation using date-based splitting and time-series cross-validation
- Streamlit app for live prediction on the latest downloaded market data
- Optional Flask API for REST-style prediction requests

## Project Files

```text
DataScienceFinalPrj/
├── Stock_Market_Prediction.ipynb   # Main notebook following the assignment steps
├── app.py                          # Streamlit application
├── flask_api.py                    # Bonus Flask REST API
├── project_core.py                 # Shared data preparation and prediction utilities
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

4. Optional bonus: start the Flask API:

```bash
flask --app flask_api run
```

5. If the model files do not exist yet, run the notebook through Step 7 first so the saved artifacts are created.

## Workflow Covered In The Notebook

1. Data Collection and Exploration
2. Data Cleaning and Transformation
3. Exploratory Data Analysis
4. Feature Selection
5. Model Development
6. Model Evaluation and Hyperparameter Tuning
7. Saving The Final Model
8. Streamlit Deployment Instructions
9. Bonus Notes For Additional Models And Flask API

## Assignment Checklist

- Data collection and initial exploration: covered in the notebook with dataset shape, data types, summary statistics, and missing-value checks.
- Data cleaning and transformation: covered with missing-value handling, outlier detection/treatment, encoding, scaling, and feature engineering.
- EDA: covered with Matplotlib and Seaborn visualizations plus short findings.
- Feature selection: covered with feature importance and correlation-based reduction.
- Model development: covered with train/test split, multiple algorithms, training, and metric comparison.
- Model evaluation and tuning: covered with time-series cross-validation, GridSearchCV, and final model selection.
- Deployment: covered with the Streamlit app in `app.py`.
- Documentation: covered by the notebook, this README, and comments/docstrings in the code.
- Version control: the project is organized as a GitHub repository with code, notebook, app, and documentation.

## Modeling Notes

- The training dataset is built from multiple tickers so the app can score more than one stock honestly.
- The notebook uses a time-based split by trading date to reduce look-ahead leakage.
- Cross-validation is also time-aware.
- The final saved artifacts are shared by the notebook and the Streamlit app.
- The submission is centered on the two required deliverables: one notebook and one Streamlit app.
- The optional Flask API is included as an extra bonus feature.
- Docker is intentionally excluded because this submission uses the Flask and additional-model bonus options instead.

## Important Note

This project is for educational purposes only. It demonstrates a full machine learning workflow on financial time-series data, but it is not investment advice.
