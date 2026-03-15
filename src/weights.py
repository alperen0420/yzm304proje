from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import GLOBAL_SEED, WEIGHT_DIR


def _scale_for_layer(
    fan_in: int,
    fan_out: int,
    *,
    hidden_activation: str,
    is_output_layer: bool,
) -> float:
    if is_output_layer or hidden_activation == "sigmoid":
        return float(np.sqrt(2.0 / (fan_in + fan_out)))
    return float(np.sqrt(2.0 / fan_in))


def _weight_paths(architecture: tuple[int, ...], hidden_activation: str) -> tuple[Path, Path]:
    stem = f"{'-'.join(str(unit) for unit in architecture)}_{hidden_activation}"
    return WEIGHT_DIR / f"{stem}.npz", WEIGHT_DIR / f"{stem}.json"


def ensure_weight_bundle(
    architecture: tuple[int, ...],
    hidden_activation: str,
    *,
    seed: int = GLOBAL_SEED,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, metadata_path = _weight_paths(architecture, hidden_activation)
    if npz_path.exists():
        return load_weight_bundle(npz_path)

    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for layer_index, (fan_in, fan_out) in enumerate(zip(architecture[:-1], architecture[1:])):
        scale = _scale_for_layer(
            fan_in,
            fan_out,
            hidden_activation=hidden_activation,
            is_output_layer=layer_index == len(architecture) - 2,
        )
        weights.append(
            rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)).astype(np.float64)
        )
        biases.append(np.zeros((1, fan_out), dtype=np.float64))

    payload: dict[str, np.ndarray] = {}
    for index, array in enumerate(weights):
        payload[f"W{index}"] = array
    for index, array in enumerate(biases):
        payload[f"b{index}"] = array
    np.savez(npz_path, **payload)

    metadata_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "architecture": list(architecture),
                "hidden_activation": hidden_activation,
                "weight_file": npz_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return weights, biases


def load_weight_bundle(npz_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    bundle = np.load(npz_path)
    weight_keys = sorted(key for key in bundle.files if key.startswith("W"))
    bias_keys = sorted(key for key in bundle.files if key.startswith("b"))
    weights = [bundle[key].astype(np.float64) for key in weight_keys]
    biases = [bundle[key].astype(np.float64) for key in bias_keys]
    return weights, biases


def clone_weight_bundle(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return [array.copy() for array in weights], [array.copy() for array in biases]


def weight_l2_norm(weights: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(np.sum(np.square(weight)) for weight in weights)))
