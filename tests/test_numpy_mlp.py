import numpy as np

from src.numpy_mlp import NumpyMLP


def test_numpy_mlp_loss_decreases_on_simple_problem() -> None:
    weights = [np.array([[0.2], [0.2]], dtype=np.float64)]
    biases = [np.zeros((1, 1), dtype=np.float64)]
    model = NumpyMLP(
        weights,
        biases,
        hidden_activation="sigmoid",
        learning_rate=0.2,
        l2_lambda=0.0,
    )
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    result = model.fit(X, y, X, y, X, y, epochs=80)
    assert result.history["train_loss"].iloc[-1] < result.history["train_loss"].iloc[0]
