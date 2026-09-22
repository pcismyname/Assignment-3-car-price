"""Data, bucketing, and MLflow helpers for Assignment 3 (classification).

The cleaning and preprocessing reuse the exact A1/A2 decisions so the input to
the model is the *preprocessed version of the dataset* required by Task 1. The
new piece is turning the continuous ``selling_price`` into four ordered price
classes with ``pd.qcut`` (equal-frequency quartiles), giving a balanced
4-class problem.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from logistic_regression import LogisticRegression

NUMERIC_FEATURES = ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "brand"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OWNER_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 5,
}

# Human-friendly names for the four price quartiles (class 0 = cheapest).
CLASS_LABELS = {0: "Budget", 1: "Mid-range", 2: "Premium", 3: "Luxury"}


def clean_car_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic A1/A2 cleaning decisions."""
    data = raw_data.copy()
    data["owner"] = data["owner"].map(OWNER_MAP)
    data = data[~data["fuel"].isin(["CNG", "LPG"])].copy()
    for column in ["mileage", "engine", "max_power"]:
        data[column] = pd.to_numeric(data[column].str.split().str[0], errors="coerce")
        data[column] = data[column].replace(0, np.nan)
    data["brand"] = data["name"].str.split().str[0]
    data = data.drop(columns=["name", "torque"])
    data = data[data["owner"] != 5].copy()          # drop test-drive cars
    data = data.dropna(subset=["selling_price"])
    return data.drop_duplicates().reset_index(drop=True)


def bucketize_price(prices: pd.Series, n_classes: int = 4) -> tuple[pd.Series, np.ndarray]:
    """Turn continuous prices into ``n_classes`` ordered classes via quartiles.

    ``pd.qcut`` gives equal-frequency bins, so each class holds ~25% of the
    cars — a balanced 4-class problem. Returns the integer labels (0..k-1) and
    the bin edges (rupee thresholds) so the app can report a price range.
    """
    classes, bin_edges = pd.qcut(
        prices, q=n_classes, labels=list(range(n_classes)), retbins=True
    )
    return classes.astype(int), bin_edges


def build_preprocessor() -> ColumnTransformer:
    """Leakage-safe numeric + categorical preprocessing (as in A2)."""
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def configuration_grid() -> list[dict[str, object]]:
    """All hyper-parameter combinations swept in the MLflow experiment.

    Sweeps gradient-descent method x penalty on/off x learning rate x lambda,
    which directly demonstrates the Task 2 ridge switch inside the experiment.
    """
    combinations = product(
        ["batch", "minibatch", "stochastic"],   # gradient-descent method
        [False, True],                          # use_penalty (Task 2)
        [0.1, 0.05, 0.01],                      # learning_rate
        [0.01, 0.1],                            # lambda_ (ignored when penalty off)
    )
    grid: list[dict[str, object]] = []
    for method, use_penalty, learning_rate, lambda_ in combinations:
        # When the penalty is off, lambda_ has no effect — keep only one copy to
        # avoid logging duplicate runs.
        if not use_penalty and lambda_ != 0.01:
            continue
        grid.append(
            {
                "method": method,
                "use_penalty": use_penalty,
                "learning_rate": learning_rate,
                "lambda_": lambda_,
            }
        )
    return grid


def _build_model(configuration: dict[str, object], max_iter: int, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        method=str(configuration["method"]),
        learning_rate=float(configuration["learning_rate"]),
        max_iter=max_iter,
        use_penalty=bool(configuration["use_penalty"]),
        lambda_=float(configuration["lambda_"]),
        batch_size=64,
        random_state=random_state,
    )


def cross_validate_configuration(
    features: pd.DataFrame,
    labels: pd.Series,
    configuration: dict[str, object],
    *,
    folds: int = 3,
    max_iter: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified CV of one configuration with fold-local preprocessing."""
    label_array = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(features, label_array)):
        preprocessor = build_preprocessor()
        transformed_train = preprocessor.fit_transform(features.iloc[train_idx])
        transformed_valid = preprocessor.transform(features.iloc[valid_idx])
        feature_names = list(preprocessor.get_feature_names_out())

        model = _build_model(configuration, max_iter, random_state + fold_number)
        model.fit(transformed_train, label_array[train_idx], feature_names=feature_names)
        prediction = model.predict(transformed_valid)
        truth = label_array[valid_idx]
        fold_metrics.append(
            {
                "cv_accuracy": model.accuracy(truth, prediction),
                "cv_macro_f1": model.macro_f1(truth, prediction),
                "cv_weighted_f1": model.weighted_f1(truth, prediction),
                "cv_macro_precision": model.macro_precision(truth, prediction),
                "cv_macro_recall": model.macro_recall(truth, prediction),
            }
        )

    return {metric: float(np.mean([f[metric] for f in fold_metrics])) for metric in fold_metrics[0]}


def fit_final_model(
    features: pd.DataFrame,
    labels: pd.Series,
    configuration: dict[str, object],
    bin_edges: np.ndarray,
    *,
    max_iter: int = 4000,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit preprocessing + the chosen classifier on all supplied rows."""
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)
    feature_names = list(preprocessor.get_feature_names_out())
    model = _build_model(configuration, max_iter, random_state)
    model.fit(transformed, np.asarray(labels), feature_names=feature_names)
    return {
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": feature_names,
        "configuration": dict(configuration),
        "bin_edges": np.asarray(bin_edges, dtype=float),
        "class_labels": CLASS_LABELS,
    }


def predict_class(bundle: dict[str, object], features: pd.DataFrame) -> np.ndarray:
    """Predict the price class (0..3) for raw feature rows using a fitted bundle."""
    transformed = bundle["preprocessor"].transform(features)
    return bundle["model"].predict(transformed)


def run_mlflow_experiment(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    tracking_uri: str,
    experiment_name: str,
    configurations: Sequence[dict[str, object]] | None = None,
    folds: int = 3,
    max_iter: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run every configuration, log params + metrics to MLflow, and rank runs.

    ``tracking_uri`` points to a local ``sqlite`` store while the CSIM server is
    down (per the TA notice). Switching it to
    ``http://mlflow.ml.brain.cs.ait.ac.th/`` is the only change needed once the
    server is restored — the assignment's Objective 1.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    selected = list(configurations or configuration_grid())
    records: list[dict[str, object]] = []

    for index, configuration in enumerate(selected, start=1):
        penalty_tag = "L2" if configuration["use_penalty"] else "none"
        run_name = f"run-{index:03d}-{configuration['method']}-{penalty_tag}"
        with mlflow.start_run(run_name=run_name) as run:
            metrics = cross_validate_configuration(
                features, labels, configuration,
                folds=folds, max_iter=max_iter, random_state=random_state,
            )
            mlflow.log_params({**configuration, "folds": folds, "max_iter": max_iter, "seed": random_state})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("assignment", "A3 Predicting Car Price - Classification")
            records.append({**configuration, **metrics, "run_id": run.info.run_id})

        if index == 1 or index % 6 == 0 or index == len(selected):
            print(f"Completed {index}/{len(selected)} MLflow runs")

    return (
        pd.DataFrame(records)
        .sort_values(["cv_accuracy", "cv_macro_f1"], ascending=[False, False])
        .reset_index(drop=True)
    )
