FEATURE_COLUMNS = [
    "origin",
    "destination",
    "cabin_class",
    "demand_index",
    "seasonality_index",
    "days_to_departure",
    "weekend_before_departure",
    "fuel_cost_index",
    "holiday_flag",
    "competitor_price",
    "load_factor",
]

NUMERIC_COLUMNS = [
    "demand_index",
    "seasonality_index",
    "days_to_departure",
    "weekend_before_departure",
    "fuel_cost_index",
    "holiday_flag",
    "competitor_price",
    "load_factor",
]

CATEGORICAL_COLUMNS = ["origin", "destination", "cabin_class"]
