
# Airfare Price Prediction

This project predicts flight ticket prices from route, booking window, seasonality, demand, fuel cost, competitor price, and load factor signals. It includes a small ML pipeline for training and a Flask web app for interactive predictions.

## What is included

- Synthetic dataset generation so the project can run without external data
- Regression training pipeline using Linear Regression and XGBoost
- Flask prediction app with separate frontend folders for HTML and CSS
- EDA notebooks and report assets

## Project structure

```text
Airfare Price Prediction/
|-- data/
|   |-- airfare_live_dataset.csv
|   |-- airfare_live_dataset.zip
|   `-- airfare_sample.csv
|-- models/
|   |-- linear_regression.pkl
|   |-- metrics.json
|   `-- xgboost_regression.pkl
|-- reports/
|   `-- eda/
|       |-- EDA.ipynb
|       |-- categorical_boxplots.png
|       |-- correlation_heatmap.png
|       |-- eda_report.md
|       |-- fare_distribution.png
|       |-- feature_vs_fare_scatter.png
|       `-- missing_values.png
|-- scripts/
|   |-- create_eda_notebook.py
|   |-- eda.py
|   `-- generate_sample_data.py
|-- src/
|   |-- __init__.py
|   `-- airfare/
|       |-- __init__.py
|       `-- train.py
|-- webapp/
|   |-- app.py
|   `-- frontend/
|       |-- css/
|       |   `-- styles.css
|       `-- html/
|           `-- index.html
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Folder purpose

- `src/airfare/`: ML feature definitions and training pipeline
- `scripts/`: utility scripts for dataset generation and EDA
- `data/`: input datasets
- `models/`: trained model artifacts and metrics
- `webapp/frontend/html/`: HTML templates
- `webapp/frontend/css/`: CSS styles
- `reports/`: notebook outputs and charts

## Setup

```bash
pip install -r requirements.txt
```

## Generate sample data

```bash
python scripts/generate_sample_data.py --rows 6000 --output data/airfare_sample.csv
```

## Train the models

```bash
python -m src.airfare.train --data data/airfare_sample.csv --target fare_price --model-dir models
```

Generated artifacts:

- `models/linear_regression.pkl`
- `models/xgboost_regression.pkl`
- `models/metrics.json`

## Run the web app

```bash
python webapp/app.py --model-dir models --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in your browser.

## Model input features

- `origin`
- `destination`
- `cabin_class`
- `demand_index`
- `seasonality_index`
- `days_to_departure`
- `weekend_before_departure`
- `fuel_cost_index`
- `holiday_flag`
- `competitor_price`
- `load_factor`

## GitHub push steps

This folder is not a git repository yet. To publish it after reviewing the files:

```bash
git init
git add .
git commit -m "Initial commit for airfare price prediction project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Notes

- Replace the sample dataset with real airfare history for production use.
- Keep training feature names identical to the fields submitted by the web app.
=======
# airfare-price-analytics
End-to-end airfare price prediction and revenue optimization using Python, SQL, and statistical analysis.
