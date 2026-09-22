"""Unit tests for the deployed model — the two required by A3 Task 3, plus a few.

The assignment (Objective 3, TO DO #1) asks for two unit tests on the model:
  1. the model takes the expected input, and
  2. the output of the model has the expected shape.
``test_model_takes_expected_input`` and ``test_model_output_has_expected_shape``
cover exactly those two requirements.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predict import ALL_FIELDS, build_input_row, load_model, predict_category

APP_CODE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bundle() -> dict:
    return load_model(APP_CODE_DIR)


SAMPLE_CAR = {
    "brand": "Maruti", "year": 2015, "fuel": "Petrol", "transmission": "Manual",
    "km_driven": 60000, "owner": "First Owner", "seller_type": "Individual",
    "mileage": 20.0, "engine": 1200.0, "max_power": 85.0, "seats": 5,
}


# ---- Required test 1: the model takes the expected input -------------------- #
def test_model_takes_expected_input(bundle: dict) -> None:
    """A well-formed one-row DataFrame with all expected columns is accepted."""
    row = build_input_row(SAMPLE_CAR)
    assert list(row.columns) == ALL_FIELDS      # every expected feature present
    assert row.shape == (1, len(ALL_FIELDS))

    # The preprocessor + model accept this input without raising.
    result = predict_category(bundle, row)
    assert result["class_id"] in {0, 1, 2, 3}
    assert result["label"] in {"Budget", "Mid-range", "Premium", "Luxury"}


# ---- Required test 2: the output has the expected shape --------------------- #
def test_model_output_has_expected_shape(bundle: dict) -> None:
    """predict_proba returns one probability row of length 4 that sums to 1."""
    row = build_input_row(SAMPLE_CAR)
    transformed = bundle["preprocessor"].transform(row)
    probabilities = bundle["model"].predict_proba(transformed)

    assert probabilities.shape == (1, 4)                 # (n_samples, n_classes)
    assert probabilities.sum(axis=1) == pytest.approx(1.0)
    assert bundle["model"].predict(transformed).shape == (1,)


# ---- A few extra guards ----------------------------------------------------- #
def test_missing_fields_are_imputed_not_rejected(bundle: dict) -> None:
    """Skipped fields arrive as NaN and the imputers fill them — no crash."""
    row = build_input_row({"brand": "Hyundai", "year": 2018})
    assert row.isna().any(axis=None)                     # some fields are NaN
    result = predict_category(bundle, row)
    assert 0.0 <= result["confidence"] <= 1.0


def test_batch_prediction_shape_scales_with_rows(bundle: dict) -> None:
    rows = pd.concat([build_input_row(SAMPLE_CAR) for _ in range(3)], ignore_index=True)
    transformed = bundle["preprocessor"].transform(rows)
    assert bundle["model"].predict_proba(transformed).shape == (3, 4)
    assert bundle["model"].predict(transformed).shape == (3,)
