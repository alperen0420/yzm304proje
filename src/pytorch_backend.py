from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.metrics import binary_cross_entropy, compute_binary_metrics
from src.weights import clone_weight_bundle, weight_l2_norm


@dataclass
class TorchTrainingResult:
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


class TorchMLP(nn.Module):
    def __init__(self, architecture: tuple[int, ...], hidden_activation: str) -> None:
        super().__init__()
        self.hidden_activation = hidden_activation
        self.layers = nn.ModuleList(
            nn.Linear(fan_in, fan_out, bias=True)
            for fan_in, fan_out in zip(architecture[:-1], architecture[1:])
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        current = features
        for index, layer in enumerate(self.layers):
            current = layer(current)
            if index < len(self.layers) - 1:
                if self.hidden_activation == "sigmoid":
                    current = torch.sigmoid(current)
                elif self.hidden_activation == "relu":
                    current = torch.relu(current)
                else:
                    raise ValueError(f"Unsupported activation: {self.hidden_activation}")
            else:
                current = torch.sigmoid(current)
        return current


def fit_torch_model(
    architecture: tuple[int, ...],
    *,
    hidden_activation: str,
    learning_rate: float,
    l2_lambda: float,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
) -> TorchTrainingResult:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)

    model = TorchMLP(architecture, hidden_activation)
    cloned_weights, cloned_biases = clone_weight_bundle(weights, biases)
    with torch.no_grad():
        for layer, weight, bias in zip(model.layers, cloned_weights, cloned_biases):
            layer.weight.copy_(torch.from_numpy(weight.T))
            layer.bias.copy_(torch.from_numpy(bias.reshape(-1)))

    X_train_tensor = torch.from_numpy(X_train).to(dtype=torch.float64)
    y_train_tensor = torch.from_numpy(y_train.reshape(-1, 1)).to(dtype=torch.float64)
    X_val_tensor = torch.from_numpy(X_val).to(dtype=torch.float64)
    X_test_tensor = torch.from_numpy(X_test).to(dtype=torch.float64)

    criterion = nn.BCELoss()
    regularized_parameters = []
    unregularized_parameters = []
    for name, parameter in model.named_parameters():
        if name.endswith("weight"):
            regularized_parameters.append(parameter)
        else:
            unregularized_parameters.append(parameter)
    optimizer = torch.optim.SGD(
        [
            {"params": regularized_parameters, "weight_decay": l2_lambda},
            {"params": unregularized_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )

    history_rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        train_prob = model(X_train_tensor)
        train_loss = criterion(train_prob, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_prob_np = model(X_train_tensor).detach().cpu().numpy().reshape(-1)
            val_prob_np = model(X_val_tensor).detach().cpu().numpy().reshape(-1)
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": binary_cross_entropy(y_train, train_prob_np),
                    "val_loss": binary_cross_entropy(y_val, val_prob_np),
                    "train_accuracy": float(np.mean((train_prob_np >= 0.5).astype(int) == y_train)),
                    "val_accuracy": float(np.mean((val_prob_np >= 0.5).astype(int) == y_val)),
                }
            )

    with torch.no_grad():
        train_prob = model(X_train_tensor).detach().cpu().numpy().reshape(-1)
        val_prob = model(X_val_tensor).detach().cpu().numpy().reshape(-1)
        test_prob = model(X_test_tensor).detach().cpu().numpy().reshape(-1)

    train_pred = (train_prob >= 0.5).astype(int)
    val_pred = (val_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)
    final_weights = [layer.weight.detach().cpu().numpy().T.copy() for layer in model.layers]
    final_biases = [layer.bias.detach().cpu().numpy().reshape(1, -1).copy() for layer in model.layers]

    return TorchTrainingResult(
        history=pd.DataFrame(history_rows),
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
