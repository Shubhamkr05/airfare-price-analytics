import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.airfare import FEATURE_COLUMNS

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "html"),
    static_folder=str(FRONTEND_DIR / "css"),
)
MODEL_DIR = Path("models")
MODELS = {}
METRICS = {}


def load_artifacts(model_dir: Path) -> None:
    global MODEL_DIR, MODELS, METRICS
    MODEL_DIR = model_dir
    MODELS = {}
    METRICS = {}

    linear_path = model_dir / "linear_regression.pkl"
    xgb_path = model_dir / "xgboost_regression.pkl"
    metrics_path = model_dir / "metrics.json"

    if linear_path.exists():
        MODELS["linear_regression"] = joblib.load(linear_path)
    if xgb_path.exists():
        MODELS["xgboost_regression"] = joblib.load(xgb_path)
    if metrics_path.exists():
        METRICS = json.loads(metrics_path.read_text(encoding="utf-8"))


def parse_form(form_data) -> dict:
    return {
        "origin": str(form_data.get("origin", "DEL")).strip().upper(),
        "destination": str(form_data.get("destination", "BOM")).strip().upper(),
        "cabin_class": str(form_data.get("cabin_class", "Economy")).strip(),
        "demand_index": float(form_data.get("demand_index", 1.0)),
        "seasonality_index": float(form_data.get("seasonality_index", 1.0)),
        "days_to_departure": int(form_data.get("days_to_departure", 30)),
        "weekend_before_departure": int(form_data.get("weekend_before_departure", 0)),
        "fuel_cost_index": float(form_data.get("fuel_cost_index", 1.0)),
        "holiday_flag": int(form_data.get("holiday_flag", 0)),
        "competitor_price": float(form_data.get("competitor_price", 5000)),
        "load_factor": float(form_data.get("load_factor", 0.8)),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = "xgboost_regression"
    error = None
    payload = None

    if request.method == "POST":
        try:
            payload = parse_form(request.form)
            selected_model = request.form.get("model_name", "xgboost_regression")
            model = MODELS.get(selected_model)
            if model is None:
                raise RuntimeError(
                    f"Model '{selected_model}' not available in {MODEL_DIR}. Train models first."
                )

            input_df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
            predicted_fare = float(model.predict(input_df)[0])
            prediction = max(round(predicted_fare, 2), 0.0)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        selected_model=selected_model,
        available_models=list(MODELS.keys()),
        metrics=METRICS,
        error=error,
        payload=payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run airfare prediction web app.")
    parser.add_argument("--model-dir", default="models", type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    load_artifacts(Path(args.model_dir))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
