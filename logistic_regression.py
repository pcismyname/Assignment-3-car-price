"""From-scratch multinomial logistic regression for AT82.03 Assignment 3.

This module extends the ``02 - Multinomial Logistic Regression`` notebook from
class in two ways required by the assignment:

* **Task 1** — every classification metric (accuracy, per-class precision /
  recall / f1, and their macro and weighted averages) is implemented by hand so
  we understand exactly what ``sklearn.metrics.classification_report`` reports.
* **Task 2** — an optional **Ridge (L2) penalty** on the loss, which the user
  turns on or off through the constructor.

Only array maths and plotting come from libraries; softmax, the cross-entropy
loss, the gradient, the three gradient-descent flavours, and all of the metrics
are written out here so the mechanics are visible.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class LogisticRegression:
    """Multinomial (softmax) logistic regression trained with gradient descent.

    Args:
        method: Gradient-descent flavour: ``batch``, ``minibatch`` or
            ``stochastic`` (mirrors the class notebook).
        learning_rate: Step size for the weight update ``W := W - lr * grad``.
        max_iter: Number of gradient-descent iterations.
        use_penalty: If ``True`` add the Ridge (L2) penalty to the loss and its
            gradient. This is the Task 2 switch — ``False`` reproduces plain
            multinomial logistic regression.
        lambda_: Ridge penalty strength ``λ`` (only used when ``use_penalty``).
        batch_size: Samples per update when ``method="minibatch"``.
        random_state: Seed for weight initialisation and batch sampling.
    """

    VALID_METHODS = {"batch", "minibatch", "stochastic"}

    def __init__(
        self,
        method: str = "batch",
        learning_rate: float = 0.01,
        max_iter: int = 5000,
        use_penalty: bool = False,
        lambda_: float = 0.1,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.use_penalty = use_penalty
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # scikit-learn compatibility (lets us reuse cross_val_score, GridSearch)
    # ------------------------------------------------------------------ #
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters (scikit-learn estimator protocol)."""
        del deep
        return {
            "method": self.method,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "use_penalty": self.use_penalty,
            "lambda_": self.lambda_,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "LogisticRegression":
        """Set constructor parameters (scikit-learn estimator protocol)."""
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"Unknown parameter: {name}")
            setattr(self, name, value)
        return self

    # ------------------------------------------------------------------ #
    # Core maths: softmax, one-hot, loss, gradient
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_intercept(features: np.ndarray) -> np.ndarray:
        """Prepend a column of ones so w[0] acts as the per-class bias term."""
        ones = np.ones((features.shape[0], 1), dtype=float)
        return np.concatenate([ones, features], axis=1)

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """Row-wise softmax turning class scores into a probability distribution.

        The max is subtracted before exponentiating for numerical stability —
        this cannot change the result because softmax is shift-invariant.
        """
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(labels: np.ndarray, number_of_classes: int) -> np.ndarray:
        """Encode integer labels as an (m, k) matrix with a single 1 per row."""
        encoded = np.zeros((labels.shape[0], number_of_classes), dtype=float)
        encoded[np.arange(labels.shape[0]), labels] = 1.0
        return encoded

    def _loss_and_gradient(
        self, features: np.ndarray, one_hot_target: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Cross-entropy loss (+ optional ridge) and its gradient w.r.t. W."""
        sample_count = features.shape[0]
        probabilities = self.softmax(features @ self.weights)
        # Cross-entropy: -sum(Y * log(h)) / m. The clip avoids log(0).
        log_likelihood = -np.sum(
            one_hot_target * np.log(np.clip(probabilities, 1e-15, 1.0))
        ) / sample_count
        gradient = features.T @ (probabilities - one_hot_target) / sample_count

        if self.use_penalty:
            # Ridge / L2: J += λ * Σ θ².  The bias row (index 0) is excluded so
            # we regularise slopes only, which is the standard convention.
            penalised = self.weights.copy()
            penalised[0, :] = 0.0
            log_likelihood += self.lambda_ * float(np.sum(penalised**2))
            gradient += 2.0 * self.lambda_ * penalised

        return float(log_likelihood), gradient

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _iter_batches(
        self, feature_count: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Return the row indices used for a single gradient-descent update."""
        if self.method == "batch":
            return np.arange(feature_count)
        if self.method == "stochastic":
            return generator.integers(0, feature_count, size=1)
        size = min(self.batch_size, feature_count)
        return generator.choice(feature_count, size=size, replace=False)

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "LogisticRegression":
        """Fit softmax weights on preprocessed features and integer labels."""
        if self.method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {sorted(self.VALID_METHODS)}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.use_penalty and self.lambda_ < 0:
            raise ValueError("lambda_ cannot be negative")

        feature_array = np.asarray(features, dtype=float)
        label_array = np.asarray(target).reshape(-1)
        if feature_array.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if len(feature_array) != len(label_array) or len(label_array) == 0:
            raise ValueError("features and target must share the same non-zero length")

        # Map raw labels to 0..k-1 so one-hot encoding and argmax line up even if
        # the incoming classes are, say, [1, 2, 3, 4] rather than [0, 1, 2, 3].
        self.classes_ = np.unique(label_array)
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        indexed_labels = np.array([class_to_index[label] for label in label_array])
        number_of_classes = len(self.classes_)

        design = self._add_intercept(feature_array)
        one_hot_target = self._one_hot(indexed_labels, number_of_classes)

        base_names = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(feature_array.shape[1])
        ]
        if len(base_names) != feature_array.shape[1]:
            raise ValueError("feature_names must match the input column count")
        self.feature_names_out_ = base_names

        generator = np.random.default_rng(self.random_state)
        # Small random weights break the symmetry between classes.
        self.weights = generator.normal(
            0.0, 0.01, size=(design.shape[1], number_of_classes)
        )
        self.loss_history_ = []
        history_interval = max(1, self.max_iter // 50)

        for iteration in range(self.max_iter):
            batch_indices = self._iter_batches(len(design), generator)
            batch_features = design[batch_indices]
            batch_target = one_hot_target[batch_indices]
            _, gradient = self._loss_and_gradient(batch_features, batch_target)
            self.weights -= self.learning_rate * gradient

            if iteration % history_interval == 0 or iteration == self.max_iter - 1:
                full_loss, _ = self._loss_and_gradient(design, one_hot_target)
                self.loss_history_.append(full_loss)

        self.n_features_in_ = feature_array.shape[1]
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the (m, k) class-probability matrix for new samples."""
        if not hasattr(self, "weights"):
            raise RuntimeError("fit must be called before predict_proba")
        feature_array = np.asarray(features, dtype=float)
        if feature_array.ndim != 2 or feature_array.shape[1] != self.n_features_in_:
            raise ValueError("features do not match the fitted input shape")
        return self.softmax(self._add_intercept(feature_array) @ self.weights)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the most probable class label for each sample."""
        winning_index = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[winning_index]

    # ------------------------------------------------------------------ #
    # Task 1 metrics — implemented from scratch
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unique_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.unique(np.concatenate([np.asarray(y_true).reshape(-1),
                                         np.asarray(y_pred).reshape(-1)]))

    @staticmethod
    def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
        """Correct predictions / all predictions."""
        true_array = np.asarray(y_true).reshape(-1)
        pred_array = np.asarray(y_pred).reshape(-1)
        if true_array.shape != pred_array.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return float(np.mean(true_array == pred_array))

    @classmethod
    def _per_class_counts(
        cls, y_true: Sequence, y_pred: Sequence
    ) -> dict[Any, dict[str, int]]:
        """Return TP, FP, FN and support (true count) for every class."""
        true_array = np.asarray(y_true).reshape(-1)
        pred_array = np.asarray(y_pred).reshape(-1)
        counts: dict[Any, dict[str, int]] = {}
        for label in cls._unique_labels(true_array, pred_array):
            true_positive = int(np.sum((pred_array == label) & (true_array == label)))
            false_positive = int(np.sum((pred_array == label) & (true_array != label)))
            false_negative = int(np.sum((pred_array != label) & (true_array == label)))
            support = int(np.sum(true_array == label))
            counts[label] = {
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "support": support,
            }
        return counts

    @classmethod
    def precision(cls, y_true: Sequence, y_pred: Sequence) -> dict[Any, float]:
        """precision_c = TP_c / (TP_c + FP_c) for each class c."""
        return {
            label: (c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0)
            for label, c in cls._per_class_counts(y_true, y_pred).items()
        }

    @classmethod
    def recall(cls, y_true: Sequence, y_pred: Sequence) -> dict[Any, float]:
        """recall_c = TP_c / (TP_c + FN_c) for each class c."""
        return {
            label: (c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0)
            for label, c in cls._per_class_counts(y_true, y_pred).items()
        }

    @classmethod
    def f1(cls, y_true: Sequence, y_pred: Sequence) -> dict[Any, float]:
        """f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)."""
        precision = cls.precision(y_true, y_pred)
        recall = cls.recall(y_true, y_pred)
        scores: dict[Any, float] = {}
        for label in precision:
            p, r = precision[label], recall[label]
            scores[label] = (2 * p * r / (p + r)) if (p + r) else 0.0
        return scores

    # -- macro averages: unweighted mean across classes ---------------- #
    @classmethod
    def macro_precision(cls, y_true: Sequence, y_pred: Sequence) -> float:
        values = list(cls.precision(y_true, y_pred).values())
        return float(np.mean(values)) if values else 0.0

    @classmethod
    def macro_recall(cls, y_true: Sequence, y_pred: Sequence) -> float:
        values = list(cls.recall(y_true, y_pred).values())
        return float(np.mean(values)) if values else 0.0

    @classmethod
    def macro_f1(cls, y_true: Sequence, y_pred: Sequence) -> float:
        values = list(cls.f1(y_true, y_pred).values())
        return float(np.mean(values)) if values else 0.0

    # -- weighted averages: weight each class by its support ----------- #
    @classmethod
    def _weighted(cls, per_class: dict[Any, float], y_true: Sequence) -> float:
        true_array = np.asarray(y_true).reshape(-1)
        total = true_array.shape[0]
        if total == 0:
            return 0.0
        weighted_sum = 0.0
        for label, score in per_class.items():
            support = float(np.sum(true_array == label))
            weighted_sum += (support / total) * score
        return float(weighted_sum)

    @classmethod
    def weighted_precision(cls, y_true: Sequence, y_pred: Sequence) -> float:
        return cls._weighted(cls.precision(y_true, y_pred), y_true)

    @classmethod
    def weighted_recall(cls, y_true: Sequence, y_pred: Sequence) -> float:
        return cls._weighted(cls.recall(y_true, y_pred), y_true)

    @classmethod
    def weighted_f1(cls, y_true: Sequence, y_pred: Sequence) -> float:
        return cls._weighted(cls.f1(y_true, y_pred), y_true)

    @classmethod
    def classification_report(
        cls, y_true: Sequence, y_pred: Sequence, digits: int = 4
    ) -> str:
        """Reproduce sklearn's classification_report layout from scratch.

        ``support`` is the number of ground-truth samples for each class — i.e.
        how many rows actually belong to that class in ``y_true``. It tells you
        how much each per-class score should be trusted and is exactly the
        weight used by the weighted averages.
        """
        precision = cls.precision(y_true, y_pred)
        recall = cls.recall(y_true, y_pred)
        f1 = cls.f1(y_true, y_pred)
        counts = cls._per_class_counts(y_true, y_pred)
        total_support = sum(c["support"] for c in counts.values())

        header = f"{'':>12}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}"
        lines = [header, ""]
        for label in sorted(precision):
            lines.append(
                f"{str(label):>12}{precision[label]:>12.{digits}f}"
                f"{recall[label]:>12.{digits}f}{f1[label]:>12.{digits}f}"
                f"{counts[label]['support']:>12d}"
            )
        lines.append("")
        lines.append(
            f"{'accuracy':>12}{'':>12}{'':>12}"
            f"{cls.accuracy(y_true, y_pred):>12.{digits}f}{total_support:>12d}"
        )
        lines.append(
            f"{'macro avg':>12}{cls.macro_precision(y_true, y_pred):>12.{digits}f}"
            f"{cls.macro_recall(y_true, y_pred):>12.{digits}f}"
            f"{cls.macro_f1(y_true, y_pred):>12.{digits}f}{total_support:>12d}"
        )
        lines.append(
            f"{'weighted avg':>12}{cls.weighted_precision(y_true, y_pred):>12.{digits}f}"
            f"{cls.weighted_recall(y_true, y_pred):>12.{digits}f}"
            f"{cls.weighted_f1(y_true, y_pred):>12.{digits}f}{total_support:>12d}"
        )
        return "\n".join(lines)
