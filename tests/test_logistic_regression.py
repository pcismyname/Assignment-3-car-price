"""Tests for the from-scratch multinomial logistic regression and its metrics.

These verify Task 1 (metrics match scikit-learn) and Task 2 (the ridge penalty
actually shrinks the weights), and that training learns a separable problem.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression


@pytest.fixture(scope="module")
def trained_model():
    features, labels = make_classification(
        n_samples=600, n_features=8, n_informative=6, n_classes=4,
        n_clusters_per_class=1, random_state=0,
    )
    features = StandardScaler().fit_transform(features)
    model = LogisticRegression(method="batch", learning_rate=0.3, max_iter=3000, random_state=0)
    model.fit(features, labels)
    return model, features, labels


# ---- Task 1: metrics match scikit-learn ------------------------------------ #
def test_accuracy_matches_sklearn(trained_model):
    model, features, labels = trained_model
    predictions = model.predict(features)
    assert model.accuracy(labels, predictions) == pytest.approx(accuracy_score(labels, predictions))


def test_macro_and_weighted_match_sklearn(trained_model):
    model, features, labels = trained_model
    predictions = model.predict(features)
    assert model.macro_precision(labels, predictions) == pytest.approx(
        precision_score(labels, predictions, average="macro", zero_division=0))
    assert model.macro_recall(labels, predictions) == pytest.approx(
        recall_score(labels, predictions, average="macro", zero_division=0))
    assert model.macro_f1(labels, predictions) == pytest.approx(
        f1_score(labels, predictions, average="macro", zero_division=0))
    assert model.weighted_f1(labels, predictions) == pytest.approx(
        f1_score(labels, predictions, average="weighted", zero_division=0))


def test_per_class_precision_matches_sklearn_on_mockup():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3, 0, 0]
    y_pred = [0, 1, 2, 2, 0, 1, 1, 3, 0, 3]
    mine = LogisticRegression.precision(y_true, y_pred)
    reference = precision_score(y_true, y_pred, average=None, zero_division=0)
    for class_id, score in enumerate(reference):
        assert mine[class_id] == pytest.approx(score)


def test_support_equals_true_class_counts():
    y_true = [0, 0, 1, 1, 1, 2, 3, 3]
    counts = LogisticRegression._per_class_counts(y_true, y_true)
    # "support" is simply how many ground-truth rows belong to each class.
    assert counts[0]["support"] == 2
    assert counts[1]["support"] == 3
    assert counts[3]["support"] == 2


# ---- Task 2: ridge penalty shrinks weights --------------------------------- #
def test_ridge_penalty_shrinks_weight_magnitude():
    features, labels = make_classification(
        n_samples=400, n_features=10, n_informative=6, n_classes=4,
        n_clusters_per_class=1, random_state=1,
    )
    features = StandardScaler().fit_transform(features)
    common = dict(method="batch", learning_rate=0.2, max_iter=1500, random_state=1)
    plain = LogisticRegression(use_penalty=False, **common).fit(features, labels)
    ridge = LogisticRegression(use_penalty=True, lambda_=1.0, **common).fit(features, labels)
    # L2 regularisation pulls the slope weights toward zero (bias row excluded).
    assert np.sum(ridge.weights[1:] ** 2) < np.sum(plain.weights[1:] ** 2)


# ---- Training + input validation ------------------------------------------- #
def test_softmax_rows_are_probability_distributions():
    scores = np.array([[1.0, 2.0, 3.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    probabilities = LogisticRegression.softmax(scores)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)


def test_model_learns_separable_problem(trained_model):
    model, features, labels = trained_model
    assert model.accuracy(labels, model.predict(features)) > 0.6


@pytest.mark.parametrize("bad_method", ["online", "sgd", ""])
def test_invalid_method_is_rejected(bad_method):
    with pytest.raises(ValueError, match="method"):
        LogisticRegression(method=bad_method).fit(np.ones((4, 2)), np.array([0, 1, 0, 1]))
