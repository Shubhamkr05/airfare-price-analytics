import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.airfare import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise RuntimeError(
        "xgboost is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(rmse, 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_models(data_path: Path, target: str, model_dir: Path, random_state: int = 42) -> dict:
    df = pd.read_csv(data_path)
    required_cols = set(FEATURE_COLUMNS + [target])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {sorted(missing)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    linear_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=450,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )

    linear_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    linear_pred = linear_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    metrics = {
        "linear_regression": evaluate(y_test.values, linear_pred),
        "xgboost_regression": evaluate(y_test.values, xgb_pred),
        "dataset": {
            "rows": int(df.shape[0]),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "features": FEATURE_COLUMNS,
            "target": target,
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(linear_model, model_dir / "linear_regression.pkl")
    joblib.dump(xgb_model, model_dir / "xgboost_regression.pkl")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train airfare price prediction models.")
    parser.add_argument("--data", required=True, type=str, help="CSV path for training data")
    parser.add_argument("--target", default="fare_price", type=str, help="Target column name")
    parser.add_argument("--model-dir", default="models", type=str, help="Output model directory")
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    metrics = train_models(
        data_path=data_path,
        target=args.target,
        model_dir=model_dir,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
