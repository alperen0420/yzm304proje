from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    GLOBAL_SEED,
    RAW_DIR,
    SPLIT_DIR,
    TARGET_NAMES,
    TEST_SIZE,
    TRAIN_FRACTION_RANDOM_STATE,
    TRAIN_TEST_RANDOM_STATE,
    TRAIN_VALID_RANDOM_STATE,
    VALIDATION_SIZE_WITHIN_TRAIN,
)


@dataclass
class PreparedSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler | None
    train_indices: list[int]


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    feature_names: list[str]
    target_names: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: dict[str, object]


def _load_dataframe() -> tuple[pd.DataFrame, list[str], list[str]]:
    dataset = load_breast_cancer(as_frame=True)
    dataframe = dataset.frame.copy()
    dataframe["target_name"] = dataframe["target"].map(
        {0: TARGET_NAMES[0], 1: TARGET_NAMES[1]}
    )
    return dataframe, dataset.feature_names.tolist(), list(dataset.target_names)


def _build_split_manifest(dataframe: pd.DataFrame) -> dict[str, object]:
    feature_frame = dataframe.drop(columns=["target_name"])
    target = feature_frame["target"]

    train_full_idx, test_idx = train_test_split(
        feature_frame.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=TRAIN_TEST_RANDOM_STATE,
        stratify=target,
    )
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        random_state=TRAIN_VALID_RANDOM_STATE,
        stratify=target.loc[train_full_idx],
    )
    return {
        "seed": GLOBAL_SEED,
        "splits": {
            "train": sorted(int(index) for index in train_idx),
            "validation": sorted(int(index) for index in val_idx),
            "test": sorted(int(index) for index in test_idx),
        },
    }


def export_dataset_artifacts() -> DatasetBundle:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe, feature_names, target_names = _load_dataframe()
    manifest = _build_split_manifest(dataframe)

    dataframe.to_csv(RAW_DIR / "breast_cancer_dataset.csv", index_label="row_id")
    metadata = {
        "dataset_name": "sklearn.datasets.load_breast_cancer",
        "samples": int(len(dataframe)),
        "features": len(feature_names),
        "target_names": target_names,
        "feature_names": feature_names,
    }
    (RAW_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (SPLIT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, indices in manifest["splits"].items():
        split_frame = dataframe.loc[indices].copy().reset_index(names="row_id")
        split_frames[split_name] = split_frame
        split_frame.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False)

    return DatasetBundle(
        dataframe=dataframe.reset_index(names="row_id"),
        feature_names=feature_names,
        target_names=target_names,
        train_df=split_frames["train"],
        val_df=split_frames["validation"],
        test_df=split_frames["test"],
        manifest=manifest,
    )


def prepare_features(
    bundle: DatasetBundle,
    feature_names: list[str],
    *,
    use_scaler: bool,
    train_fraction: float,
) -> PreparedSplit:
    train_df = bundle.train_df.copy()
    if train_fraction < 1.0:
        train_df, _ = train_test_split(
            train_df,
            train_size=train_fraction,
            random_state=TRAIN_FRACTION_RANDOM_STATE,
            stratify=train_df["target"],
        )
        train_df = train_df.sort_values("row_id").reset_index(drop=True)

    val_df = bundle.val_df.copy()
    test_df = bundle.test_df.copy()

    X_train = train_df[feature_names]
    X_val = val_df[feature_names]
    X_test = test_df[feature_names]
    y_train = train_df["target"].astype(int)
    y_val = val_df["target"].astype(int)
    y_test = test_df["target"].astype(int)

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_names,
            index=train_df.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(X_val),
            columns=feature_names,
            index=val_df.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_names,
            index=test_df.index,
        )

    return PreparedSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        train_indices=train_df["row_id"].astype(int).tolist(),
    )
