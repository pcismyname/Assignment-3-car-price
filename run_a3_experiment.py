"""Reproduce the full Assignment 3 experiment end-to-end.

Steps
-----
1. Load and clean ``Cars.csv`` (A1/A2 preprocessing).
2. Bucket ``selling_price`` into 4 quartile classes.
3. Run the MLflow sweep locally (sqlite store — CSIM server is down per the TA).
4. Refit the best configuration on the full training split.
5. Save the model bundle, CV table, summary JSON, and evaluation figures.

Run:  ``python run_a3_experiment.py``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from a3_experiment import (
    CLASS_LABELS,
    FEATURES,
    bucketize_price,
    build_preprocessor,
    clean_car_data,
    configuration_grid,
    fit_final_model,
    run_mlflow_experiment,
)
from logistic_regression import LogisticRegression

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"
APP_MODEL_DIR = ROOT / "app" / "code"
STUDENT_ID = "st127004"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = f"{STUDENT_ID}-a3"
RANDOM_STATE = 42


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1-2. Load, clean, bucket ------------------------------------------------
    raw = pd.read_csv(ROOT / "Cars.csv")
    clean = clean_car_data(raw)
    labels, bin_edges = bucketize_price(clean["selling_price"], n_classes=4)
    features = clean[FEATURES]
    print(f"Rows after cleaning: {len(clean)}")
    print("Class distribution:\n", labels.value_counts().sort_index())
    print("Price bin edges (INR):", np.round(bin_edges).astype(int).tolist())

    # Hold out a stratified test set the experiment never sees.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )

    # 3. MLflow sweep (local) -------------------------------------------------
    grid = configuration_grid()
    print(f"\nRunning {len(grid)} configurations through MLflow (local sqlite store)...")
    results = run_mlflow_experiment(
        x_train, y_train,
        tracking_uri=TRACKING_URI,
        experiment_name=EXPERIMENT_NAME,
        configurations=grid,
        folds=3,
        max_iter=1500,
        random_state=RANDOM_STATE,
    )
    results.to_csv(ARTIFACTS / "cv_results.csv", index=False)
    best = results.iloc[0].to_dict()
    best_config = {k: best[k] for k in ("method", "use_penalty", "learning_rate", "lambda_")}
    print("\nBest configuration:", best_config)
    print(f"CV accuracy={best['cv_accuracy']:.4f}  macro_f1={best['cv_macro_f1']:.4f}")

    # 4. Refit best config on the full training split -------------------------
    bundle = fit_final_model(x_train, y_train, best_config, bin_edges, max_iter=4000, random_state=RANDOM_STATE)

    # 5. Evaluate on the held-out test set ------------------------------------
    transformed_test = bundle["preprocessor"].transform(x_test)
    test_pred = bundle["model"].predict(transformed_test)
    model = bundle["model"]
    test_metrics = {
        "test_accuracy": model.accuracy(y_test, test_pred),
        "test_macro_f1": model.macro_f1(y_test, test_pred),
        "test_weighted_f1": model.weighted_f1(y_test, test_pred),
        "test_macro_precision": model.macro_precision(y_test, test_pred),
        "test_macro_recall": model.macro_recall(y_test, test_pred),
    }
    print("\nHeld-out test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\n" + LogisticRegression.classification_report(y_test, test_pred))

    # Save the served bundle to model/ and mirror it into the app.
    bundle_path = MODEL_DIR / "a3_car_price_classifier.pkl"
    joblib.dump(bundle, bundle_path)
    shutil.copy(bundle_path, APP_MODEL_DIR / "a3_car_price_classifier.pkl")
    print(f"\nSaved model bundle -> {bundle_path} (and mirrored into app/code/)")

    # Summary JSON + figures.
    summary = {
        "student_id": STUDENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "n_rows_clean": int(len(clean)),
        "bin_edges_inr": np.round(bin_edges).astype(int).tolist(),
        "class_labels": CLASS_LABELS,
        "best_configuration": best_config,
        "cv_accuracy": float(best["cv_accuracy"]),
        "cv_macro_f1": float(best["cv_macro_f1"]),
        **{k: float(v) for k, v in test_metrics.items()},
    }
    (ARTIFACTS / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_confusion(y_test, test_pred, ARTIFACTS / "confusion_matrix.png")
    _plot_loss(model.loss_history_, ARTIFACTS / "loss_curve.png")
    print("Wrote artifacts: cv_results.csv, experiment_summary.json, confusion_matrix.png, loss_curve.png")


def _plot_confusion(y_true, y_pred, path: Path) -> None:
    classes = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    index = {c: i for i, c in enumerate(classes)}
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        matrix[index[t], index[p]] += 1
    figure, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([f"{c}\n{CLASS_LABELS[c]}" for c in classes])
    ax.set_yticklabels([f"{c} {CLASS_LABELS[c]}" for c in classes])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (held-out test set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_loss(history, path: Path) -> None:
    figure, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, color="#00A6A6")
    ax.set_xlabel("Recorded iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss of the best classifier")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
