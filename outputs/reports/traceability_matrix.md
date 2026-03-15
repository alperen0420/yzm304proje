# Izlenebilirlik Matrisi

| PDF Gereksinimi | Kanit | Durum |
| --- | --- | --- |
| Ikili veya coklu siniflandirma verisi | `data/raw/breast_cancer_dataset.csv`, `README.md` | Tamam |
| Veri analizi ve on isleme | `outputs/figures/class_distribution.png`, `README.md Methods` | Tamam |
| Laboratuvar modelinin egitimi ve testi | `baseline_raw`, `baseline_scaled`, `outputs/tables/numpy_experiment_metrics.csv` | Tamam |
| Overfitting / underfitting incelemesi | `outputs/figures/numpy_learning_curves.png`, `README.md Results` | Tamam |
| Cok katmanli modeller, veri miktari, reglarizasyon | `deep_scaled_*`, `outputs/figures/data_fraction_comparison.png` | Tamam |
| Tekrar eden fonksiyonlardan kacinma / class yapisi | `src/numpy_mlp.py`, `src/sklearn_backend.py`, `src/pytorch_backend.py` | Tamam |
| Dengesiz sinif dagiliminda uygun secim kriteri | `outputs/tables/model_selection.csv`, `outputs/reports/selected_model_report.md` | Tamam |
| sklearn ve PyTorch ile yeniden yazim | `outputs/tables/backend_comparison_metrics.csv`, `src/sklearn_backend.py`, `src/pytorch_backend.py` | Tamam |
| Confusion matrix ve temel metrikler | `outputs/figures/confusion_matrix_*`, `outputs/tables/*.csv` | Tamam |
| Ayni train/test, baslangic agirliklari, hiperparametreler, SGD | `data/splits/split_manifest.json`, `data/weights/*.npz`, `README.md Methods` | Tamam |
| BCE veya uygun loss function | `src/numpy_mlp.py`, `src/pytorch_backend.py`, `README.md Methods` | Tamam |
| Uygun dosya ve klasor hiyerarsisi | Repo koku, `src/`, `data/`, `outputs/`, `tests/` | Tamam |
| README IMRAD formati | `README.md` | Tamam |