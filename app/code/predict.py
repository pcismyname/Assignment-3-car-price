"""Core prediction logic for the Assignment 3 car-price *classifier* app.

Loads the fitted bundle (preprocessor + from-scratch multinomial logistic
regression) and turns a dict of user inputs into a predicted **price class**
(0-3) plus its rupee range. Missing/skipped fields are passed as NaN so the
preprocessor's imputers fill them — the app never blocks on a missing value.
Kept UI-free so it can be unit-tested.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Raw feature columns the fitted preprocessor expects (selected by name).
NUMERIC_FIELDS = ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]
CATEGORICAL_FIELDS = ["fuel", "seller_type", "transmission", "brand"]
ALL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

# owner is ordinal: map the human label to the integer the model trained on.
OWNER_MAP = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth & Above Owner": 4}


def load_model(model_dir: str | Path = ".") -> dict:
    """Load the fitted Assignment 3 classifier bundle from ``model_dir``."""
    model_dir = Path(model_dir)
    return joblib.load(model_dir / "a3_car_price_classifier.pkl")


def build_input_row(user_inputs: dict) -> pd.DataFrame:
    """Assemble a one-row DataFrame with every expected column.

    A field absent from ``user_inputs`` (or None/"") becomes NaN so the
    pipeline imputes it. ``owner`` may arrive as its text label and is mapped
    to the training ordinal.
    """
    row = {}
    for column in ALL_FIELDS:
        value = user_inputs.get(column, None)
        row[column] = np.nan if (value is None or value == "") else value

    if isinstance(row["owner"], str):
        row["owner"] = OWNER_MAP.get(row["owner"], np.nan)

    frame = pd.DataFrame([row], columns=ALL_FIELDS)
    for column in NUMERIC_FIELDS:                       # NaN-friendly numeric dtype
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def predict_category(bundle: dict, row_df: pd.DataFrame) -> dict:
    """Predict the price class and return label, probabilities, and rupee range.

    Returns a dict with:
      * ``class_id``   — integer 0..3
      * ``label``      — human name (Budget / Mid-range / Premium / Luxury)
      * ``low``/``high`` — rupee bounds of that quartile bin (``high`` is None
        for the open-topped Luxury class)
      * ``confidence`` — softmax probability of the winning class
    """
    transformed = bundle["preprocessor"].transform(row_df)
    probabilities = bundle["model"].predict_proba(transformed)[0]
    class_id = int(bundle["model"].classes_[int(np.argmax(probabilities))])

    edges = np.asarray(bundle["bin_edges"], dtype=float)
    low = float(edges[class_id])
    # Top class is open-ended for display purposes.
    high = float(edges[class_id + 1]) if class_id + 1 < len(edges) - 1 else None
    return {
        "class_id": class_id,
        "label": bundle["class_labels"][class_id],
        "low": low,
        "high": high,
        "confidence": float(np.max(probabilities)),
        "probabilities": probabilities.tolist(),
    }
