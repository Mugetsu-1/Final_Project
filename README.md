# Stock Market Price Movement Prediction

A clean, notebook-first data science project that predicts stock direction for:
- **Next Day (1d)**
- **Next Week (5 trading days / 5d)**

The notebook is the main workflow for loading data, EDA, feature engineering, model training, evaluation, and artifact export.  
`app.py` is only for Streamlit inference and visualization.

### Optional GPU Acceleration

- Notebook training includes an optional **XGBoost GPU candidate**.
- Set `ENABLE_GPU = True` in the notebook to prefer CUDA (if available).
- If GPU/CUDA is not available, the notebook continues with CPU-based models.

## Project Structure

```text
DataScienceFinalPrj/
├── Stock_Market_Prediction.ipynb
├── app.py
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
├── ARCHITECTURE.md
├── Objective.md
├── cleaned_stock_data.csv
├── stock_prediction_model.pkl
├── scaler.pkl
├── model_info.pkl
├── stock_prediction_model_weekly.pkl
├── scaler_weekly.pkl
├── model_info_weekly.pkl
└── (features are stored inside model_info artifacts)
```

## Formal Workflow (Objective-Aligned)

The notebook follows the assignment structure:
1. Data Collection and Initial Exploration
2. Data Cleaning and Transformation
3. Exploratory Data Analysis (EDA)
4. Feature Engineering and Feature Selection
5. Model Development (multiple candidates)
6. Model Evaluation, Cross-Validation, and Hyperparameter Tuning
7. Final Model Saving (artifacts for Streamlit)

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook and execute cells top-to-bottom:

```bash
jupyter notebook
```

3. Start Streamlit:

```bash
streamlit run app.py
```

## Notes

- The notebook saves model artifacts that are directly used by `app.py`.
- The Streamlit app supports:
  - **Classical ML inference** (using saved artifacts)
  - **ARIMA benchmark inference** (live fit on recent close prices; heuristic confidence only)
- Prediction UI shows:
  - **UP in green**
  - **DOWN in red**

- The classical model threshold is tuned on the validation split and evaluated on a separate holdout test split.

## Important

This project is for educational purposes only and is **not financial advice**.
