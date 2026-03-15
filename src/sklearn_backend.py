from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_random_state

from src.metrics import binary_cross_entropy, compute_binary_metrics
from src.weights import clone_weight_bundle, weight_l2_norm


@dataclass
class SklearnTrainingResult:
    history: pd.DataFrame
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_loss: float
    val_loss: float
    test_loss: float
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    final_weights: list[np.ndarray]
    final_biases: list[np.ndarray]
    weight_norm: float


class SklearnMLPAdapter:
    def __init__(
        self,
        architecture: tuple[int, ...],
        *,
        hidden_activation: str,
        learning_rate: float,
        l2_lambda: float,
    ) -> None:
        activation = "logistic" if hidden_activation == "sigmoid" else hidden_activation
        self.architecture = architecture
        self.estimator = MLPClassifier(
            hidden_layer_sizes=architecture[1:-1],
            activation=activation,
            solver="sgd",
            alpha=l2_lambda,
            batch_size=1,
            learning_rate="constant",
            learning_rate_init=learning_rate,
            max_iter=1,
            shuffle=False,
            random_state=42,
            warm_start=True,
            momentum=0.0,
            nesterovs_momentum=False,
        )

    def initialize_with_weights(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        cloned_weights, cloned_biases = clone_weight_bundle(weights, biases)
        classes = np.array([0, 1], dtype=int)
        self.estimator.classes_ = classes
        self.estimator._label_binarizer = LabelBinarizer()
        self.estimator._label_binarizer.fit(classes)

        validated_X, validated_y = self.estimator._validate_input(
            X_train,
            y_train,
            incremental=True,
            reset=True,
        )
        if validated_y.ndim == 1:
            validated_y = validated_y.reshape(-1, 1)

        self.estimator._random_state = check_random_state(self.estimator.random_state)
        self.estimator.batch_size = len(X_train)
        layer_units = [validated_X.shape[1], *self.architecture[1:-1], validated_y.shape[1]]
        self.estimator._initialize(validated_y, layer_units, validated_X.dtype)
        self.estimator.coefs_ = cloned_weights
        self.estimator.intercepts_ = [bias.reshape(-1) for bias in cloned_biases]
        self.estimator._best_coefs = [array.copy() for array in cloned_weights]
        self.estimator._best_intercepts = [array.reshape(-1).copy() for array in cloned_biases]

    def fit(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        epochs: int,
    ) -> SklearnTrainingResult:
        self.initialize_with_weights(weights, biases, X_train, y_train)
        history_rows: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            self.estimator.partial_fit(X_train, y_train)
            train_prob = self.estimator.predict_proba(X_train)[:, 1]
            val_prob = self.estimator.predict_proba(X_val)[:, 1]
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": binary_cross_entropy(y_train, train_prob),
                    "val_loss": binary_cross_entropy(y_val, val_prob),
                    "train_accuracy": float(np.mean((train_prob >= 0.5).astype(int) == y_train)),
                    "val_accuracy": float(np.mean((val_prob >= 0.5).astype(int) == y_val)),
                }
            )

        history = pd.DataFrame(history_rows)
        train_prob = self.estimator.predict_proba(X_train)[:, 1]
        val_prob = self.estimator.predict_proba(X_val)[:, 1]
        test_prob = self.estimator.predict_proba(X_test)[:, 1]
        train_pred = self.estimator.predict(X_train)
        val_pred = self.estimator.predict(X_val)
        test_pred = self.estimator.predict(X_test)
        final_weights = [array.copy() for array in self.estimator.coefs_]
        final_biases = [array.reshape(1, -1).copy() for array in self.estimator.intercepts_]

        return SklearnTrainingResult(
            history=history,
            train_metrics=compute_binary_metrics(y_train, train_pred, train_prob),
            val_metrics=compute_binary_metrics(y_val, val_pred, val_prob),
            test_metrics=compute_binary_metrics(y_test, test_pred, test_prob),
            train_loss=binary_cross_entropy(y_train, train_prob),
            val_loss=binary_cross_entropy(y_val, val_prob),
            test_loss=binary_cross_entropy(y_test, test_prob),
            train_probabilities=train_prob,
            val_probabilities=val_prob,
            test_probabilities=test_prob,
            train_predictions=train_pred,
            val_predictions=val_pred,
            test_predictions=test_pred,
            final_weights=final_weights,
            final_biases=final_biases,
            weight_norm=weight_l2_norm(final_weights),
        )
