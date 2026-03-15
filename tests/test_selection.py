import pandas as pd

from src.run_all import build_model_selection


def test_model_selection_prefers_balanced_accuracy_then_steps_then_size() -> None:
    frame = pd.DataFrame(
        [
            {
                "run_name": "large_model",
                "val_balanced_accuracy": 0.95,
                "n_steps": 100,
                "parameter_count": 500,
                "val_roc_auc": 0.99,
            },
            {
                "run_name": "compact_model",
                "val_balanced_accuracy": 0.95,
                "n_steps": 100,
                "parameter_count": 200,
                "val_roc_auc": 0.98,
            },
            {
                "run_name": "faster_model",
                "val_balanced_accuracy": 0.95,
                "n_steps": 80,
                "parameter_count": 900,
                "val_roc_auc": 0.97,
            },
            {
                "run_name": "best_balanced_accuracy",
                "val_balanced_accuracy": 0.97,
                "n_steps": 500,
                "parameter_count": 900,
                "val_roc_auc": 0.90,
            },
        ]
    )

    selection = build_model_selection(frame)

    assert selection.iloc[0]["run_name"] == "best_balanced_accuracy"
    assert selection.iloc[1]["run_name"] == "faster_model"
    assert selection.iloc[2]["run_name"] == "compact_model"
