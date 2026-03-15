from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import binary_cross_entropy, compute_binary_metrics
from src.weights import clone_weight_bundle, weight_l2_norm


@dataclass
class NumpyTrainingResult:
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


class NumpyMLP:
    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        *,
        hidden_activation: str,
        learning_rate: float,
        l2_lambda: float,
    ) -> None:
        self.weights, self.biases = clone_weight_bundle(weights, biases)
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))

    def _hidden_forward(self, values: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "sigmoid":
            return self._sigmoid(values)
        if self.hidden_activation == "relu":
            return np.maximum(values, 0.0)
        raise ValueError(f"Unsupported activation: {self.hidden_activation}")

    def _hidden_backward(self, z_values: np.ndarray, activations: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "sigmoid":
            return activations * (1.0 - activations)
        return (z_values > 0.0).astype(np.float64)

    def _forward(self, features: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        linear_outputs: list[np.ndarray] = []
        current = features
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z_values = current @ weight + bias
            linear_outputs.append(z_values)
            if layer_index == len(self.weights) - 1:
                current = self._sigmoid(z_values)
            else:
                current = self._hidden_forward(z_values)
            activations.append(current)
        return activations, linear_outputs

    def _loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        penalty = 0.5 * self.l2_lambda * sum(np.sum(np.square(weight)) for weight in self.weights)
        penalty /= len(y_true)
        return binary_cross_entropy(y_true, y_prob) + float(penalty)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[0][-1].reshape(-1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= 0.5).astype(int)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        epochs: int,
    ) -> NumpyTrainingResult:
        y_train = y_train.astype(np.float64).reshape(-1, 1)
        y_val = y_val.astype(np.float64).reshape(-1, 1)
        y_test = y_test.astype(np.float64).reshape(-1, 1)

        history_rows: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            activations, linear_outputs = self._forward(X_train)
            predictions = activations[-1]
            gradient = predictions - y_train
            grad_weights: list[np.ndarray] = []
            grad_biases: list[np.ndarray] = []

            for layer_index in reversed(range(len(self.weights))):
                previous_activation = activations[layer_index]
                grad_weight = (previous_activation.T @ gradient) / len(X_train)
                grad_weight += (self.l2_lambda / len(X_train)) * self.weights[layer_index]
                grad_bias = gradient.mean(axis=0, keepdims=True)
                grad_weights.append(grad_weight)
                grad_biases.append(grad_bias)

                if layer_index > 0:
                    backprop_signal = gradient @ self.weights[layer_index].T
                    gradient = backprop_signal * self._hidden_backward(
                        linear_outputs[layer_index - 1],
                        activations[layer_index],
                    )

            for layer_index, (grad_weight, grad_bias) in enumerate(
                zip(reversed(grad_weights), reversed(grad_biases))
            ):
                self.weights[layer_index] -= self.learning_rate * grad_weight
                self.biases[layer_index] -= self.learning_rate * grad_bias

            train_prob = self.predict_proba(X_train)
            val_prob = self.predict_proba(X_val)
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": self._loss(y_train, train_prob),
                    "val_loss": self._loss(y_val, val_prob),
                    "train_accuracy": float(
                        np.mean((train_prob >= 0.5).astype(int) == y_train.reshape(-1))
                    ),
                    "val_accuracy": float(
                        np.mean((val_prob >= 0.5).astype(int) == y_val.reshape(-1))
                    ),
                }
            )

        history = pd.DataFrame(history_rows)
        train_prob = self.predict_proba(X_train)
        val_prob = self.predict_proba(X_val)
        test_prob = self.predict_proba(X_test)
        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)
        final_weights, final_biases = clone_weight_bundle(self.weights, self.biases)

        return NumpyTrainingResult(
            history=history,
            train_metrics=compute_binary_metrics(y_train.reshape(-1), train_pred, train_prob),
            val_metrics=compute_binary_metrics(y_val.reshape(-1), val_pred, val_prob),
            test_metrics=compute_binary_metrics(y_test.reshape(-1), test_pred, test_prob),
            train_loss=self._loss(y_train, train_prob),
            val_loss=self._loss(y_val, val_prob),
            test_loss=self._loss(y_test, test_prob),
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
