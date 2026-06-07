#!/usr/bin/env python3
"""Run an EQL-style neural-symbolic adapter on standard-formula cards.

This adapter is intentionally small and reproducible: a neural feature layer
applies a fixed symbolic operator bank to learned affine projections, then a
linear readout predicts the target.  It exposes prediction, gradient support,
Hessian pair scores, and an operator/complexity summary as normalized
ClaimTransfer adapter-output rows.  The official scorer recomputes all verdicts.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "run_eql_standard_formula_baseline.py requires torch. "
        f"Import failed with: {type(exc).__name__}: {exc}"
    )

from run_standard_formula_adapter_sweep import (
    ROOT,
    emit_common,
    load_cards,
    make_data,
    pair_claims,
    parse_seed_range,
    support_claim,
)


class EQLNet(nn.Module):
    """One hidden symbolic operator bank plus a linear readout."""

    OP_NAMES = ("linear", "square", "sin", "cos", "tanh", "exp")

    def __init__(self, d: int, units: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, units)
        self.readout = nn.Linear(units * len(self.OP_NAMES), 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        exp_z = torch.exp(torch.clamp(z, min=-4.0, max=4.0))
        feats = [z, z * z, torch.sin(z), torch.cos(z), torch.tanh(z), exp_z]
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x)).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    units: int,
    steps: int,
    batch_size: int,
    lr: float,
    l1: float,
    device: str,
) -> tuple[EQLNet, float]:
    set_seed(seed)
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = EQLNet(x_train.shape[1], units).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    xtr = torch.as_tensor(x_train, dtype=torch.float32, device=dev)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=dev)
    xte = torch.as_tensor(x_test, dtype=torch.float32, device=dev)
    yte = torch.as_tensor(y_test, dtype=torch.float32, device=dev)
    n = len(xtr)
    for _ in range(steps):
        if batch_size <= 0 or batch_size >= n:
            idx = torch.arange(n, device=dev)
        else:
            idx = torch.randint(0, n, (batch_size,), device=dev)
        pred = model(xtr[idx])
        mse = torch.mean((pred - ytr[idx]) ** 2)
        sparsity = sum(p.abs().mean() for p in model.parameters())
        loss = mse + l1 * sparsity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_mse = float(torch.mean((model(xte) - yte) ** 2).detach().cpu())
    return model, test_mse


def gradient_support_scores(model: EQLNet, x: np.ndarray, points: int) -> np.ndarray:
    dev = next(model.parameters()).device
    sample = torch.as_tensor(x[:points], dtype=torch.float32, device=dev).clone().requires_grad_(True)
    y = model(sample).sum()
    grad = torch.autograd.grad(y, sample, create_graph=False)[0]
    return grad.detach().abs().mean(dim=0).cpu().numpy()


def hessian_pair_scores(model: EQLNet, x: np.ndarray, points: int) -> dict[tuple[int, int], float]:
    dev = next(model.parameters()).device
    d = x.shape[1]
    scores = {pair: 0.0 for pair in combinations(range(d), 2)}
    n = min(points, len(x))
    for row in x[:n]:
        inp = torch.as_tensor(row[None, :], dtype=torch.float32, device=dev).clone().requires_grad_(True)
        out = model(inp).sum()
        grad = torch.autograd.grad(out, inp, create_graph=True)[0].squeeze(0)
        for i in range(d):
            hrow = torch.autograd.grad(grad[i], inp, retain_graph=True, create_graph=False)[0].squeeze(0)
            for j in range(i + 1, d):
                scores[(i, j)] += abs(float(hrow[j].detach().cpu()))
    if n:
        scores = {pair: val / n for pair, val in scores.items()}
    return scores


def observed_operators(model: EQLNet, threshold: float) -> tuple[str, float]:
    weights = model.readout.weight.detach().abs().cpu().numpy().reshape(len(EQLNet.OP_NAMES), -1)
    active = []
    for name, row in zip(EQLNet.OP_NAMES, weights):
        if float(row.max()) >= threshold:
            if name == "square":
                active.append("power")
            elif name == "tanh":
                active.append("tanh")
            elif name == "linear":
                active.append("plus")
            else:
                active.append(name)
    ops = sorted(set(active))
    complexity = float(len(ops) + int((weights >= threshold).sum()))
    return ",".join(ops), complexity


def run_card(card: dict, seed: int, args: argparse.Namespace, rows: list[dict]) -> None:
    x_train, y_train, x_test, y_test = make_data(card, seed)
    t0 = perf_counter()
    model, mse = train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        seed=seed,
        units=args.units,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        l1=args.l1,
        device=args.device,
    )
    runtime = perf_counter() - t0
    grad_scores = gradient_support_scores(model, x_test, args.explain_points)
    support = support_claim(card)
    selected = list(map(int, np.argsort(-grad_scores)[: max(1, len(support))]))
    pair_scores = hessian_pair_scores(model, x_test, args.hessian_points) if pair_claims(card) else None
    operators, complexity = observed_operators(model, args.operator_threshold)

    before = len(rows)
    emit_common(
        rows,
        card,
        adapter="eql_neural_symbolic",
        adapter_family="neural_symbolic",
        seed=seed,
        evidence_object="eql_operator_bank",
        mse=mse,
        support_selected=selected,
        pair_scores=pair_scores,
        operators=operators,
        complexity=complexity,
    )
    for row in rows[before:]:
        row["source_kind"] = "eql_standard_formula_baseline"
        row["source_file"] = "eql_standard_formula_adapter_outputs.csv"
        row["runtime_seconds"] = f"{runtime:.3f}"
        row["protocol"] = (
            f"eql units={args.units} steps={args.steps} l1={args.l1:g} "
            f"ops={operators or 'none'} grad_points={args.explain_points} "
            f"hessian_points={args.hessian_points}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-cards", default="task_cards/claimtransfer_v1_standard_formula_public.json")
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--out-dir", default="results/revision/eql_standard_formula_baseline")
    parser.add_argument("--units", type=int, default=48)
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--operator-threshold", type=float, default=0.03)
    parser.add_argument("--explain-points", type=int, default=128)
    parser.add_argument("--hessian-points", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cards = load_cards(ROOT / args.task_cards)
    seeds = parse_seed_range(args.seeds)
    rows: list[dict] = []
    for card in cards:
        for seed in seeds:
            run_card(card, seed, args, rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "eql_standard_formula_adapter_outputs.csv", index=False)
    summary = detail.groupby(["task_family", "claim_type"], dropna=False).size().reset_index(name="rows")
    summary.to_csv(out_dir / "eql_standard_formula_summary.csv", index=False)
    print(f"Wrote {out_dir / 'eql_standard_formula_adapter_outputs.csv'} ({len(detail)} rows)")
    print(f"Wrote {out_dir / 'eql_standard_formula_summary.csv'}")


if __name__ == "__main__":
    main()
