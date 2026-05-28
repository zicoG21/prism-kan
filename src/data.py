from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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


def _standardize_y(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
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
    nuisance_correlation: float = 0.0,
    n_correlated_proxies: int = 0,
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

    # For correlated_proxy, keep x0--x3 as the true causal variables,
    # but add proxy variables x4 and x5 that are strongly correlated with
    # x0 and x1. This tests whether explanations confuse true variables
    # with correlated but non-causal features.
    if function_name == "correlated_proxy":
        proxy_noise = 0.05
        X[:, 4] = X[:, 0] + gen.normal(0.0, proxy_noise, size=n_total).astype(np.float32)
        if d > 5:
            X[:, 5] = X[:, 1] + gen.normal(0.0, proxy_noise, size=n_total).astype(np.float32)

    proxy_groups: Dict[int, int] = {}
    if nuisance_correlation > 0 and n_correlated_proxies > 0:
        if not 0 <= nuisance_correlation < 1:
            raise ValueError("nuisance_correlation must be in [0, 1).")
        max_proxies = max(0, d - 4)
        n_proxy = min(int(n_correlated_proxies), max_proxies)
        independent_scale = float(np.sqrt(max(1.0 - nuisance_correlation ** 2, 0.0)))
        for offset in range(n_proxy):
            proxy_idx = 4 + offset
            active_idx = offset % 4
            proxy_noise = gen.uniform(input_low, input_high, size=n_total).astype(np.float32)
            X[:, proxy_idx] = (
                nuisance_correlation * X[:, active_idx]
                + independent_scale * proxy_noise
            ).astype(np.float32)
            proxy_groups[int(proxy_idx)] = int(active_idx)

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
        "nuisance_correlation": float(nuisance_correlation),
        "proxy_groups": proxy_groups,
    }


def _core_interaction_with_coeff(
    function_name: str,
    X: np.ndarray,
    coeff: float,
) -> Tuple[np.ndarray, GroundTruth]:
    x = X
    y = np.sin(2 * np.pi * x[:, 0]) + x[:, 1] ** 2 + coeff * x[:, 2] * x[:, 3]

    if coeff == 1.0:
        coeff_text = ""
        formula = "sin(2*pi*x0) + x1^2 + x2*x3"
    else:
        coeff_text = f"{coeff}*"
        formula = f"sin(2*pi*x0) + x1^2 + {coeff}*x2*x3"

    gt = GroundTruth(
        name=function_name,
        active_variables=(0, 1, 2, 3),
        interactions=((2, 3),),
        formula=formula,
    )
    return y.astype(np.float32), gt



def _feynman_physics_function(
    function_name: str,
    X: np.ndarray,
) -> Tuple[np.ndarray, GroundTruth]:
    """Physics-inspired symbolic-regression formulas embedded in high dimension.

    These are Feynman-style known-formula benchmarks with explicit
    ground-truth active variables and interaction sets. The remaining
    coordinates are irrelevant features created by make_synthetic.
    """
    x = X
    pi = np.pi
    eps = 0.25

    if function_name == "feynman_energy":
        # Kinetic-energy style: E = 1/2 m v^2.
        # Shift mass positive but keep x0-x1 interaction.
        mass = x[:, 0] + 1.5
        velocity = x[:, 1]
        y = 0.5 * mass * velocity ** 2
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1),
            interactions=((0, 1),),
            formula="0.5*(x0+1.5)*x1^2",
        )

    elif function_name == "feynman_gravity":
        # Gravity-style: F = m1*m2/r^2, regularized to avoid r=0 blow-up.
        m1 = x[:, 0] + 1.5
        m2 = x[:, 1] + 1.5
        r = x[:, 2]
        y = (m1 * m2) / (r ** 2 + eps)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1), (0, 2), (1, 2)),
            formula="(x0+1.5)*(x1+1.5)/(x2^2+0.25)",
        )

    elif function_name == "feynman_coulomb":
        # Coulomb-style: F = q1*q2/r^2, regularized.
        q1 = x[:, 0]
        q2 = x[:, 1]
        r = x[:, 2]
        y = (q1 * q2) / (r ** 2 + eps)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1), (0, 2), (1, 2)),
            formula="x0*x1/(x2^2+0.25)",
        )

    elif function_name == "feynman_damped_wave":
        # Damped oscillator style: exp(-gamma*t)*sin(omega*t).
        t = 0.5 * (x[:, 0] + 1.0)     # [0, 1]
        gamma = x[:, 1] + 1.5         # [0.5, 2.5]
        omega = 2.0 + 2.0 * x[:, 2]   # [0, 4]
        y = np.exp(-gamma * t) * np.sin(2 * pi * omega * t)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1), (0, 2)),
            formula="exp(-(x1+1.5)*0.5*(x0+1))*sin(2*pi*(2+2*x2)*0.5*(x0+1))",
        )

    else:
        raise ValueError(f"Unknown Feynman-style function_name={function_name!r}")

    return y.astype(np.float32), gt


def _formula_suite_function(
    function_name: str,
    X: np.ndarray,
) -> Tuple[np.ndarray, GroundTruth]:
    """Small formula-fidelity benchmark suite.

    The formulas are intentionally modest and ground-truth labeled. They are
    meant to broaden the diagnostic benchmark beyond the core polynomial while
    keeping variables, endpoints, and pairwise dependencies auditable.
    """
    x = X
    pi = np.pi

    if function_name == "formula_poly_additive":
        y = np.sin(2 * pi * x[:, 0]) + x[:, 1] ** 2 + 0.5 * x[:, 2] ** 3
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=(),
            formula="sin(2*pi*x0) + x1^2 + 0.5*x2^3",
        )

    elif function_name == "formula_bilinear":
        y = x[:, 0] * x[:, 1] + 0.5 * x[:, 2]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="x0*x1 + 0.5*x2",
        )

    elif function_name == "formula_weak_centered":
        y = np.sin(2 * pi * x[:, 0]) + x[:, 1] ** 2 + 0.25 * x[:, 2] * x[:, 3]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3),
            interactions=((2, 3),),
            formula="sin(2*pi*x0) + x1^2 + 0.25*x2*x3",
        )

    elif function_name == "formula_trig_product":
        y = np.sin(pi * x[:, 0] * x[:, 1]) + 0.5 * x[:, 2]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="sin(pi*x0*x1) + 0.5*x2",
        )

    elif function_name == "formula_nested_trig":
        y = np.sin(2 * pi * (x[:, 0] + 0.5 * x[:, 1] * x[:, 2]))
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((1, 2),),
            formula="sin(2*pi*(x0 + 0.5*x1*x2))",
        )

    elif function_name == "formula_rational_product":
        y = (x[:, 0] * x[:, 1]) / (1.0 + x[:, 2] ** 2)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1), (0, 2), (1, 2)),
            formula="x0*x1/(1+x2^2)",
        )

    elif function_name == "formula_division_mixed":
        y = (x[:, 0] + 1.2) / (1.5 + x[:, 1] ** 2) + 0.3 * x[:, 2] * x[:, 3]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3),
            interactions=((0, 1), (2, 3)),
            formula="(x0+1.2)/(1.5+x1^2) + 0.3*x2*x3",
        )

    elif function_name == "formula_exp_product":
        y = np.exp(0.5 * x[:, 0] * x[:, 1]) + 0.2 * x[:, 2]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="exp(0.5*x0*x1) + 0.2*x2",
        )

    elif function_name == "formula_log_product":
        y = np.log(2.0 + x[:, 0] * x[:, 1]) + 0.25 * x[:, 2] ** 2
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="log(2+x0*x1) + 0.25*x2^2",
        )

    elif function_name == "formula_three_way_product":
        y = x[:, 0] * x[:, 1] * x[:, 2] + 0.5 * x[:, 3]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3),
            interactions=((0, 1), (0, 2), (1, 2)),
            formula="x0*x1*x2 + 0.5*x3",
        )

    elif function_name == "formula_mixed_sparse":
        y = np.sin(2 * pi * x[:, 0]) + (x[:, 1] * x[:, 2]) / (1.0 + x[:, 3] ** 2)
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3),
            interactions=((1, 2), (1, 3), (2, 3)),
            formula="sin(2*pi*x0) + x1*x2/(1+x3^2)",
        )

    elif function_name == "formula_sqrt_energy":
        y = np.sqrt(x[:, 0] + 1.2) * (x[:, 1] + 1.5) ** 2 + 0.25 * x[:, 2]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2),
            interactions=((0, 1),),
            formula="sqrt(x0+1.2)*(x1+1.5)^2 + 0.25*x2",
        )

    else:
        raise ValueError(f"Unknown formula-suite function_name={function_name!r}")

    return y.astype(np.float32), gt


def evaluate_synthetic_function(
    function_name: str,
    X: np.ndarray,
) -> Tuple[np.ndarray, GroundTruth]:
    x = X
    pi = np.pi

    if function_name == "core_interaction":
        y, gt = _core_interaction_with_coeff(function_name, X, coeff=1.0)

    elif function_name == "highdim_sparse":
        # Alias for core_interaction, useful when running the same sparse
        # ground-truth function in d=50 or d=100 dimensions.
        y, gt = _core_interaction_with_coeff(function_name, X, coeff=1.0)

    elif function_name in {
    "core_interaction_c01",
    "core_interaction_c025",
    "core_interaction_c05",
    "core_interaction_c1",
    "core_interaction_c2",
    "core_interaction_c5",
    }:
        coeff_map = {
        "core_interaction_c01": 0.1,
        "core_interaction_c025": 0.25,
        "core_interaction_c05": 0.5,
        "core_interaction_c1": 1.0,
        "core_interaction_c2": 2.0,
        "core_interaction_c5": 5.0,
        }
        y, gt = _core_interaction_with_coeff(
            function_name,
            X,
            coeff=coeff_map[function_name],
        )

    elif function_name == "correlated_proxy":
        # True variables are still x0, x1, x2, x3. x4 and x5 are correlated
        # proxies added in make_synthetic but are not part of the true formula.
        y = np.sin(2 * pi * x[:, 0]) + x[:, 1] ** 2 + x[:, 2] * x[:, 3]
        gt = GroundTruth(
            name=function_name,
            active_variables=(0, 1, 2, 3),
            interactions=((2, 3),),
            formula="sin(2*pi*x0) + x1^2 + x2*x3 with x4~x0 and x5~x1 proxies",
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

    elif function_name in {
        "feynman_energy",
        "feynman_gravity",
        "feynman_coulomb",
        "feynman_damped_wave",
    }:
        y, gt = _feynman_physics_function(function_name, X)

    elif function_name in {
        "formula_poly_additive",
        "formula_bilinear",
        "formula_weak_centered",
        "formula_trig_product",
        "formula_nested_trig",
        "formula_rational_product",
        "formula_division_mixed",
        "formula_exp_product",
        "formula_log_product",
        "formula_three_way_product",
        "formula_mixed_sparse",
        "formula_sqrt_energy",
    }:
        y, gt = _formula_suite_function(function_name, X)

    else:
        raise ValueError(
            f"Unknown function_name={function_name!r}. "
            "Use one of: core_interaction, highdim_sparse, correlated_proxy, "
            "core_interaction_c05, core_interaction_c1, core_interaction_c2, core_interaction_c5, "
            "additive_sparse, pairwise_interaction, compositional, rational, "
            "discontinuous, dense_quadratic, "
            "feynman_energy, feynman_gravity, feynman_coulomb, feynman_damped_wave, "
            "or formula_* mini-suite functions."
        )

    return y.astype(np.float32), gt


def load_uci_energy(
    test_size: float = 0.2,
    seed: int = 0,
    target: str = "heating",
) -> Dict:
    """
    Load UCI Energy Efficiency dataset.

    target:
      - "heating": Y1
      - "cooling": Y2

    The code downloads the Excel file directly when run locally.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url)

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
        X,
        y,
        test_size=test_size,
        random_state=seed,
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
