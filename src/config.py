from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
WEIGHT_DIR = DATA_DIR / "weights"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

GLOBAL_SEED = 42
TARGET_NAMES = ("malignant", "benign")
TRAIN_TEST_RANDOM_STATE = GLOBAL_SEED
TRAIN_VALID_RANDOM_STATE = GLOBAL_SEED
TRAIN_FRACTION_RANDOM_STATE = GLOBAL_SEED
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.25


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    architecture: tuple[int, ...]
    hidden_activation: str
    learning_rate: float
    l2_lambda: float
    use_scaler: bool
    train_fraction: float
    epochs: int
    notes: str

    @property
    def architecture_label(self) -> str:
        return "-".join(str(unit) for unit in self.architecture)

    @property
    def parameter_count(self) -> int:
        total = 0
        for fan_in, fan_out in zip(self.architecture[:-1], self.architecture[1:]):
            total += fan_in * fan_out + fan_out
        return total


NUMPY_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="baseline_raw",
        architecture=(30, 8, 1),
        hidden_activation="sigmoid",
        learning_rate=0.08,
        l2_lambda=0.0,
        use_scaler=False,
        train_fraction=1.0,
        epochs=1200,
        notes="Laboratuvar modelinin ham veri ile egitilen temel surumu.",
    ),
    ExperimentSpec(
        name="baseline_scaled",
        architecture=(30, 8, 1),
        hidden_activation="sigmoid",
        learning_rate=0.08,
        l2_lambda=0.0,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1200,
        notes="Ayni mimarinin standardizasyon ile iyilestirilmis surumu.",
    ),
    ExperimentSpec(
        name="wide_scaled",
        architecture=(30, 16, 1),
        hidden_activation="sigmoid",
        learning_rate=0.05,
        l2_lambda=0.0,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Gizli katmandaki noron sayisini artiran genis model.",
    ),
    ExperimentSpec(
        name="deep_scaled_no_l2",
        architecture=(30, 32, 16, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=0.0,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Cok katmanli modelin reglarizasyonsuz referans surumu.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2",
        architecture=(30, 32, 16, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Cok katmanli modelin L2 reglarizasyonlu temel surumu.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2_data50",
        architecture=(30, 32, 16, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=0.50,
        epochs=1400,
        notes="Egitim verisinin yuzde 50'si ile egitilen derin model.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2_data75",
        architecture=(30, 32, 16, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=0.75,
        epochs=1400,
        notes="Egitim verisinin yuzde 75'i ile egitilen derin model.",
    ),
    ExperimentSpec(
        name="deep_scaled_l2_data100",
        architecture=(30, 32, 16, 1),
        hidden_activation="relu",
        learning_rate=0.01,
        l2_lambda=1e-3,
        use_scaler=True,
        train_fraction=1.0,
        epochs=1400,
        notes="Egitim verisinin tamami ile egitilen derin model.",
    ),
)

BACKEND_COMPARISON_RUNS: tuple[ExperimentSpec, ...] = (
    NUMPY_EXPERIMENTS[1],
    NUMPY_EXPERIMENTS[2],
    NUMPY_EXPERIMENTS[7],
)

EXPECTED_ASSIGNMENT_OUTPUTS = (
    FIGURE_DIR / "class_distribution.png",
    FIGURE_DIR / "numpy_learning_curves.png",
    FIGURE_DIR / "data_fraction_comparison.png",
    FIGURE_DIR / "backend_test_accuracy.png",
    TABLE_DIR / "numpy_experiment_metrics.csv",
    TABLE_DIR / "backend_comparison_metrics.csv",
    TABLE_DIR / "model_selection.csv",
    REPORT_DIR / "selected_model_report.md",
    REPORT_DIR / "traceability_matrix.md",
)
