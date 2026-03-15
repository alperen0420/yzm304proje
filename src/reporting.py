from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import EXPECTED_ASSIGNMENT_OUTPUTS


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_class_distribution(dataframe: pd.DataFrame, output_path: Path) -> None:
    counts = dataframe["target_name"].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index.tolist(), counts.values, color=["#B03A2E", "#1F618D"])
    axis.set_title("Sinif Dagilimi")
    axis.set_ylabel("Ornek Sayisi")
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(counts.values):
        axis.text(index, value + 5, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_learning_curves(
    history_map: dict[str, pd.DataFrame],
    output_path: Path,
    selected_runs: list[str],
) -> None:
    fig, axes = plt.subplots(len(selected_runs), 2, figsize=(12, 3.5 * len(selected_runs)))
    if len(selected_runs) == 1:
        axes = np.array([axes])

    for row_index, run_name in enumerate(selected_runs):
        history = history_map[run_name]
        axes[row_index, 0].plot(history["epoch"], history["train_loss"], label="Train loss")
        axes[row_index, 0].plot(history["epoch"], history["val_loss"], label="Val loss")
        axes[row_index, 0].set_title(f"{run_name} loss")
        axes[row_index, 0].set_xlabel("Epoch")
        axes[row_index, 0].set_ylabel("BCE loss")
        axes[row_index, 0].legend()
        axes[row_index, 0].grid(alpha=0.25)

        axes[row_index, 1].plot(history["epoch"], history["train_accuracy"], label="Train acc")
        axes[row_index, 1].plot(history["epoch"], history["val_accuracy"], label="Val acc")
        axes[row_index, 1].set_title(f"{run_name} accuracy")
        axes[row_index, 1].set_xlabel("Epoch")
        axes[row_index, 1].set_ylabel("Accuracy")
        axes[row_index, 1].set_ylim(0.0, 1.05)
        axes[row_index, 1].legend()
        axes[row_index, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    confusion_values: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion_values, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(title)
    axis.set_xlabel("Tahmin")
    axis.set_ylabel("Gercek")
    for row_index in range(confusion_values.shape[0]):
        for column_index in range(confusion_values.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(int(confusion_values[row_index, column_index])),
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_data_fraction_comparison(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    fraction_rows = metrics_frame.loc[
        metrics_frame["run_name"].isin(
            ["deep_scaled_l2_data50", "deep_scaled_l2_data75", "deep_scaled_l2_data100"]
        )
    ].copy()
    fraction_rows["train_fraction_percent"] = (
        fraction_rows["train_fraction"].astype(float) * 100.0
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        fraction_rows["train_fraction_percent"],
        fraction_rows["test_accuracy"],
        marker="o",
        label="Test accuracy",
    )
    axis.plot(
        fraction_rows["train_fraction_percent"],
        fraction_rows["val_accuracy"],
        marker="o",
        label="Validation accuracy",
    )
    axis.set_title("Veri Miktari Etkisi")
    axis.set_xlabel("Egitimde Kullanilan Veri Yuzdesi")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_backend_accuracy(backend_frame: pd.DataFrame, output_path: Path) -> None:
    pivot = backend_frame.pivot(index="architecture", columns="backend", values="test_accuracy")
    fig, axis = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(pivot.index))
    width = 0.22

    for offset_index, backend in enumerate(pivot.columns):
        axis.bar(
            x_positions + (offset_index - 1) * width,
            pivot[backend].values,
            width=width,
            label=backend,
        )

    axis.set_xticks(x_positions, pivot.index.tolist())
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Ayni Mimari Icin Backend Karsilastirmasi")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_selected_model_report(
    selected_row: pd.Series,
    metrics_frame: pd.DataFrame,
    backend_frame: pd.DataFrame,
) -> str:
    selected_table = pd.DataFrame([selected_row]).to_markdown(index=False)
    top_rows = metrics_frame.sort_values(
        ["val_balanced_accuracy", "n_steps", "parameter_count", "val_roc_auc"],
        ascending=[False, True, True, False],
    ).head(5)
    top_table = top_rows.to_markdown(index=False)
    backend_table = backend_frame.to_markdown(index=False)

    return "\n".join(
        [
            "# Secilen Model Raporu",
            "",
            "## Ozet",
            (
                f"Veri seti dengesiz oldugu icin secim kurali validation balanced accuracy "
                f"> dusuk n_steps > dusuk parametre sayisi > yuksek validation ROC-AUC "
                f"olarak uygulandi. Secilen model `{selected_row['run_name']}` oldu. "
                f"Validation balanced accuracy `{selected_row['val_balanced_accuracy']:.4f}`, "
                f"raw validation accuracy `{selected_row['val_accuracy']:.4f}` ve toplam adim "
                f"sayisi `{int(selected_row['n_steps'])}`."
            ),
            "",
            "## Secilen Satir",
            selected_table,
            "",
            "## Ilk 5 NumPy Deneyi",
            top_table,
            "",
            "## Backend Karsilastirmasi",
            backend_table,
        ]
    )


def build_traceability_matrix() -> str:
    lines = [
        "# Izlenebilirlik Matrisi",
        "",
        "| PDF Gereksinimi | Kanit | Durum |",
        "| --- | --- | --- |",
        "| Ikili veya coklu siniflandirma verisi | `data/raw/breast_cancer_dataset.csv`, `README.md` | Tamam |",
        "| Veri analizi ve on isleme | `outputs/figures/class_distribution.png`, `README.md Methods` | Tamam |",
        "| Laboratuvar modelinin egitimi ve testi | `baseline_raw`, `baseline_scaled`, `outputs/tables/numpy_experiment_metrics.csv` | Tamam |",
        "| Overfitting / underfitting incelemesi | `outputs/figures/numpy_learning_curves.png`, `README.md Results` | Tamam |",
        "| Cok katmanli modeller, veri miktari, reglarizasyon | `deep_scaled_*`, `outputs/figures/data_fraction_comparison.png` | Tamam |",
        "| Tekrar eden fonksiyonlardan kacinma / class yapisi | `src/numpy_mlp.py`, `src/sklearn_backend.py`, `src/pytorch_backend.py` | Tamam |",
        "| Dengesiz sinif dagiliminda uygun secim kriteri | `outputs/tables/model_selection.csv`, `outputs/reports/selected_model_report.md` | Tamam |",
        "| sklearn ve PyTorch ile yeniden yazim | `outputs/tables/backend_comparison_metrics.csv`, `src/sklearn_backend.py`, `src/pytorch_backend.py` | Tamam |",
        "| Confusion matrix ve temel metrikler | `outputs/figures/confusion_matrix_*`, `outputs/tables/*.csv` | Tamam |",
        "| Ayni train/test, baslangic agirliklari, hiperparametreler, SGD | `data/splits/split_manifest.json`, `data/weights/*.npz`, `README.md Methods` | Tamam |",
        "| BCE veya uygun loss function | `src/numpy_mlp.py`, `src/pytorch_backend.py`, `README.md Methods` | Tamam |",
        "| Uygun dosya ve klasor hiyerarsisi | Repo koku, `src/`, `data/`, `outputs/`, `tests/` | Tamam |",
        "| README IMRAD formati | `README.md` | Tamam |",
    ]
    return "\n".join(lines)


def build_run_summary(
    metrics_frame: pd.DataFrame,
    backend_frame: pd.DataFrame,
    selected_row: pd.Series,
) -> dict[str, object]:
    return {
        "selected_model": selected_row.to_dict(),
        "numpy_runs": metrics_frame.to_dict(orient="records"),
        "backend_runs": backend_frame.to_dict(orient="records"),
        "expected_outputs": [str(path) for path in EXPECTED_ASSIGNMENT_OUTPUTS],
    }
