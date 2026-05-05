from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GroundTruth:
    name: str
    active_variables: Tuple[int, ...]
    interactions: Tuple[Tuple[int, int], ...]
    formula: str


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _standardize_y(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std < 1e-12:
        std = 1.0
    return (y_train - mean) / std, (y_test - mean) / std, mean, std


def make_synthetic(
    function_name: str = "core_interaction",
    n_train: int = 1024,
    n_test: int = 2048,
    d: int = 20,
    noise: float = 0.0,
    seed: int = 0,
    input_low: float = -1.0,
    input_high: float = 1.0,
    standardize_target: bool = True,
) -> Dict:
    """
    Generate synthetic regression data with known active variables and interactions.

    All formulas use zero-based variable names:
      x0, x1, x2, ...

    Returns a dict containing numpy arrays and a GroundTruth object.
    """
    if d < 5:
        raise ValueError("Use d >= 5 so sparse active variables plus irrelevant variables exist.")

    gen = _rng(seed)
    n_total = n_train + n_test
    X = gen.uniform(input_low, input_high, size=(n_total, d)).astype(np.float32)

    y_clean, gt = evaluate_synthetic_function(function_name, X)
    if noise > 0:
        y = y_clean + gen.normal(0.0, noise, size=y_clean.shape).astype(np.float32)
    else:
        y = y_clean.astype(np.float32)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train].reshape(-1, 1)
    y_test = y[n_train:].reshape(-1, 1)
    y_clean_train = y_clean[:n_train].reshape(-1, 1)
    y_clean_test = y_clean[n_train:].reshape(-1, 1)

    target_mean = 0.0
    target_std = 1.0
    if standardize_target:
        y_train, y_test, target_mean, target_std = _standardize_y(y_train, y_test)
        y_clean_train = (y_clean_train - target_mean) / target_std
        y_clean_test = (y_clean_test - target_mean) / target_std

    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "y_clean_train": y_clean_train.astype(np.float32),
        "y_clean_test": y_clean_test.astype(np.float32),
        "ground_truth": gt,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def evaluate_synthetic_function(function_name: str, X: np.ndarray) -> Tuple[np.ndarray, GroundTruth]:
    x = X
    pi = np.pi

    if function_name == "core_interaction":
        # Main first experiment:
        # f = sin(2 pi x0) + x1^2 + x2*x3
        y = np.sin(2 * pi * x[:, 0]) + x[:, 1] ** 2 + x[:, 2] * x[:, 3]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3),
            interactions=((2, 3),),
            formula="sin(2*pi*x0) + x1^2 + x2*x3",
        )
    elif function_name == "additive_sparse":
        y = np.sin(2 * pi * x[:, 0]) + x[:, 1] ** 2 + np.exp(x[:, 2])
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=(),
            formula="sin(2*pi*x0) + x1^2 + exp(x2)",
        )
    elif function_name == "pairwise_interaction":
        y = x[:, 0] * x[:, 1] + np.sin(2 * pi * x[:, 2])
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="x0*x1 + sin(2*pi*x2)",
        )
    elif function_name == "compositional":
        y = np.sin(x[:, 0] * x[:, 1] + x[:, 2] ** 2)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="sin(x0*x1 + x2^2)",
        )
    elif function_name == "rational":
        y = (x[:, 0] * x[:, 1]) / (1.0 + x[:, 2] ** 2)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1), (0, 2), (1, 2)),
            formula="x0*x1/(1+x2^2)",
        )
    elif function_name == "discontinuous":
        y = (x[:, 0] > 0).astype(np.float32) + 0.5 * x[:, 1]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1),
            interactions=(),
            formula="1[x0>0] + 0.5*x1",
        )
    elif function_name == "dense_quadratic":
        y = np.zeros(x.shape[0], dtype=np.float32)
        pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                coef = ((i + 1) * (j + 2)) / 20.0
                y += coef * x[:, i] * x[:, j]
                pairs.append((i, j))
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3, 4),
            interactions=tuple(pairs),
            formula="dense pairwise quadratic over x0,...,x4",
        )
    else:
        raise ValueError(
            f"Unknown function_name={function_name!r}. "
            "Use one of: core_interaction, additive_sparse, pairwise_interaction, "
            "compositional, rational, discontinuous, dense_quadratic."
        )

    return y.astype(np.float32), gt


def load_uci_energy(test_size: float = 0.2, seed: int = 0, target: str = "heating") -> Dict:
    """
    Load UCI Energy Efficiency dataset.

    target:
      - "heating": Y1
      - "cooling": Y2

    The code downloads the Excel file directly when run locally.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url)

    # Common column names in this file:
    # X1 relative compactness, X2 surface area, ..., X8 glazing area distribution, Y1, Y2.
    df = df.dropna()
    feature_cols = [c for c in df.columns if str(c).startswith("X")]
    if target == "heating":
        target_col = "Y1"
    elif target == "cooling":
        target_col = "Y2"
    else:
        raise ValueError("target must be 'heating' or 'cooling'.")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test = x_scaler.transform(X_test).astype(np.float32)
    y_train = y_scaler.fit_transform(y_train).astype(np.float32)
    y_test = y_scaler.transform(y_test).astype(np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(map(str, feature_cols)),
        "target": target,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }
