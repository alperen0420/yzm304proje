import numpy as np

from src.sklearn_backend import SklearnMLPAdapter


def test_sklearn_adapter_uses_injected_weights() -> None:
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    weights = [np.full((2, 2), 0.1, dtype=np.float64), np.full((2, 1), -0.2, dtype=np.float64)]
    biases = [np.zeros((1, 2), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)]
    adapter = SklearnMLPAdapter(
        (2, 2, 1),
        hidden_activation="sigmoid",
        learning_rate=0.1,
        l2_lambda=0.0,
    )
    adapter.initialize_with_weights(weights, biases, X, y)
    assert np.allclose(adapter.estimator.coefs_[0], weights[0])
    assert np.allclose(adapter.estimator.coefs_[1], weights[1])
