import numpy as np

from src.metrics import binary_cross_entropy, compute_binary_metrics


def test_binary_cross_entropy_prefers_better_probabilities() -> None:
    y_true = np.array([0, 1, 1, 0])
    good = np.array([0.1, 0.9, 0.8, 0.2])
    bad = np.array([0.6, 0.4, 0.3, 0.8])
    assert binary_cross_entropy(y_true, good) < binary_cross_entropy(y_true, bad)


def test_compute_binary_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["roc_auc"] == 1.0
