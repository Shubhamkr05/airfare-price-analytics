import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROUTES = [
    ("DEL", "BOM", 1148),
    ("DEL", "BLR", 1740),
    ("DEL", "HYD", 1258),
    ("BOM", "BLR", 841),
    ("BOM", "HYD", 710),
    ("BLR", "HYD", 500),
    ("DEL", "CCU", 1305),
    ("BOM", "CCU", 1660),
]

CABIN_PREMIUM = {"Economy": 1.0, "Premium Economy": 1.35, "Business": 2.2}


def build_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    route_idx = rng.integers(0, len(ROUTES), size=rows)
    route_data = [ROUTES[i] for i in route_idx]

    origin = [r[0] for r in route_data]
    destination = [r[1] for r in route_data]
    distance_km = np.array([r[2] for r in route_data], dtype=float)

    cabin_choices = rng.choice(
        ["Economy", "Premium Economy", "Business"],
        size=rows,
        p=[0.72, 0.18, 0.10],
    )

    demand_index = np.clip(rng.normal(1.0, 0.25, size=rows), 0.4, 1.8)
    seasonality_index = np.clip(rng.normal(1.0, 0.20, size=rows), 0.5, 1.7)
    days_to_departure = rng.integers(1, 180, size=rows)
    weekend_before_departure = rng.binomial(1, 0.35, size=rows)
    fuel_cost_index = np.clip(rng.normal(1.0, 0.18, size=rows), 0.6, 1.5)
    holiday_flag = rng.binomial(1, 0.12, size=rows)
    competitor_price = np.clip(3500 + 4.2 * distance_km + rng.normal(0, 1200, size=rows), 2500, None)
    load_factor = np.clip(rng.normal(0.78, 0.12, size=rows), 0.35, 0.98)

    base = 900 + (distance_km * 2.5)
    booking_curve = 900 * np.exp(-days_to_departure / 28.0)
    cabin_mult = np.array([CABIN_PREMIUM[c] for c in cabin_choices])

    fare_price = (
        base
        + booking_curve
        + demand_index * 1900
        + seasonality_index * 1200
        + weekend_before_departure * 380
        + fuel_cost_index * 1350
        + holiday_flag * 900
        + competitor_price * 0.28
        + load_factor * 1400
    ) * cabin_mult + rng.normal(0, 350, size=rows)

    fare_price = np.round(np.clip(fare_price, 1800, None), 2)

    return pd.DataFrame(
        {
            "origin": origin,
            "destination": destination,
            "cabin_class": cabin_choices,
            "demand_index": demand_index.round(3),
            "seasonality_index": seasonality_index.round(3),
            "days_to_departure": days_to_departure,
            "weekend_before_departure": weekend_before_departure,
            "fuel_cost_index": fuel_cost_index.round(3),
            "holiday_flag": holiday_flag,
            "competitor_price": competitor_price.round(2),
            "load_factor": load_factor.round(3),
            "fare_price": fare_price,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic airfare pricing dataset.")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--output", type=str, default="data/airfare_sample.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_dataset(rows=args.rows, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
