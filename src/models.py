from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .torch_utils import set_seed, torch_dataset


class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers = []
        last = d_in
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.SiLU()]
            last = hidden
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainResult:
    model: object
    train_mse: float
    test_mse: float
    extra: Dict


def _mse(model, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(X)
        return float(torch.mean((pred - y) ** 2).detach().cpu())


def train_mlp(
    data: Dict,
    seed: int = 0,
    hidden: int = 64,
    depth: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 1500,
    batch_size: int = 256,
    device: str = "cpu",
) -> TrainResult:
    set_seed(seed)
    ds = torch_dataset(data, device=device)
    d = ds["train_input"].shape[1]
    model = MLPRegressor(d, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X = ds["train_input"]
    y = ds["train_label"]
    n = X.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X[idx])
            loss = torch.mean((pred - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return TrainResult(
        model=model,
        train_mse=_mse(model, ds["train_input"], ds["train_label"]),
        test_mse=_mse(model, ds["test_input"], ds["test_label"]),
        extra={"epochs": epochs, "hidden": hidden, "depth": depth},
    )


def _call_kan_fit_or_train(model, dataset, **kwargs):
    """
    pykan has used both model.fit(...) and model.train(...) in examples/docs.
    This wrapper tries fit first, then the special KAN train method.
    """
    if hasattr(model, "fit"):
        return model.fit(dataset, **kwargs)
    return model.train(dataset, **kwargs)


def train_kan(
    data: Dict,
    seed: int = 0,
    width_hidden: int = 8,
    grid: int = 5,
    k: int = 3,
    steps: int = 50,
    lamb: float = 1e-2,
    lamb_entropy: float = 0.0,
    prune: bool = True,
    finetune_steps: int = 20,
    device: str = "cpu",
) -> TrainResult:
    """
    Train a pykan KAN.

    Requires:
      pip install pykan
    or:
      pip install git+https://github.com/KindXiaoming/pykan.git
    """
    try:
        from kan import KAN
    except Exception as e:
        raise ImportError(
            "Could not import pykan. Install with `pip install pykan` or "
            "`pip install git+https://github.com/KindXiaoming/pykan.git`."
        ) from e

    set_seed(seed)
    ds = torch_dataset(data, device=device)
    d = int(ds["train_input"].shape[1])

    model = KAN(width=[d, width_hidden, 1], grid=grid, k=k, seed=seed, device=device)

    # pykan README recommends speed() if not using symbolic branch during training.
    if hasattr(model, "speed"):
        try:
            model.speed()
        except Exception:
            pass

    fit_kwargs = {
        "opt": "LBFGS",
        "steps": steps,
        "lamb": lamb,
    }
    if lamb_entropy and lamb_entropy > 0:
        fit_kwargs["lamb_entropy"] = lamb_entropy

    _call_kan_fit_or_train(model, ds, **fit_kwargs)

    pruned = False
    if prune:
        try:
            # This returns a smaller pruned model in current pykan docs.
            model = model.prune()
            pruned = True
            if finetune_steps > 0:
                _call_kan_fit_or_train(model, ds, opt="LBFGS", steps=finetune_steps)
        except Exception as e:
            # Pruning sometimes fails or yields NaNs depending on pykan version/task.
            pruned = False

    return TrainResult(
        model=model,
        train_mse=_mse(model, ds["train_input"], ds["train_label"]),
        test_mse=_mse(model, ds["test_input"], ds["test_label"]),
        extra={
            "width_hidden": width_hidden,
            "grid": grid,
            "k": k,
            "steps": steps,
            "lamb": lamb,
            "pruned": pruned,
            "finetune_steps": finetune_steps,
        },
    )
