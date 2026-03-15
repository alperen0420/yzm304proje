from src.dataset import export_dataset_artifacts, prepare_features


def test_split_manifest_is_reproducible() -> None:
    bundle_one = export_dataset_artifacts()
    bundle_two = export_dataset_artifacts()
    assert bundle_one.manifest == bundle_two.manifest


def test_train_fraction_reduces_training_rows() -> None:
    bundle = export_dataset_artifacts()
    prepared_full = prepare_features(bundle, bundle.feature_names, use_scaler=True, train_fraction=1.0)
    prepared_half = prepare_features(bundle, bundle.feature_names, use_scaler=True, train_fraction=0.5)
    assert len(prepared_half.X_train) < len(prepared_full.X_train)
