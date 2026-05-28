from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import make_synthetic
from src.metrics import precision_recall_f1


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int, depth: int):
        super().__init__()
        layers = []
        last = d
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.ReLU())
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def canonical_pairs(pairs):
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def train_model(data: dict, args: argparse.Namespace, seed: int, device: str) -> tuple[MLP, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(data["X_train"].shape[1], args.hidden, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    X = torch.tensor(data["X_train"], dtype=torch.float32, device=device)
    y = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32, device=device)
    yte = torch.tensor(data["y_test"], dtype=torch.float32, device=device)
    n = X.shape[0]
    best_state = None
    best_test = float("inf")
    patience_left = args.patience
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            loss = torch.mean((model(X[idx]) - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % args.eval_every == 0:
            with torch.no_grad():
                test_mse = float(torch.mean((model(Xte) - yte) ** 2).detach().cpu())
            if test_mse < best_test:
                best_test = test_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= args.eval_every
                if patience_left <= 0:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        train_mse = float(torch.mean((model(X) - y) ** 2).detach().cpu())
        test_mse = float(torch.mean((model(Xte) - yte) ** 2).detach().cpu())
    return model, train_mse, test_mse


def downstream_influence(model: MLP) -> np.ndarray:
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        raise ValueError("Expected at least input and output linear layers.")
    influence = torch.abs(linear_layers[-1].weight.detach().cpu()).reshape(-1)
    for layer in reversed(linear_layers[1:-1]):
        W = torch.abs(layer.weight.detach().cpu())
        influence = W.T @ influence
    return influence.numpy()


def nid_pair_scores(model: MLP) -> dict[tuple[int, int], float]:
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    W0 = torch.abs(linear_layers[0].weight.detach().cpu()).numpy()
    influence = downstream_influence(model)
    d = W0.shape[1]
    scores = {}
    for i, j in itertools.combinations(range(d), 2):
        hidden_joint = np.minimum(W0[:, i], W0[:, j])
        scores[(i, j)] = float(np.sum(hidden_joint * influence))
    return scores


def gradient_pair_scores(model: MLP, X_np: np.ndarray, device: str, points: int = 128) -> dict[tuple[int, int], float]:
    X = torch.tensor(X_np[:points], dtype=torch.float32, device=device)
    X.requires_grad_(True)
    y = model(X).sum()
    grad = torch.autograd.grad(y, X, create_graph=True)[0]
    d = X.shape[1]
    scores = {}
    for i, j in itertools.combinations(range(d), 2):
        gi = grad[:, i].sum()
        gij = torch.autograd.grad(gi, X, retain_graph=True, create_graph=False)[0][:, j]
        scores[(i, j)] = float(torch.mean(torch.abs(gij)).detach().cpu())
    return scores


def evaluate(scores: dict[tuple[int, int], float], true_pairs: set[tuple[int, int]]) -> dict:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    k = max(1, len(true_pairs))
    selected = {pair for pair, _ in ranked[:k]}
    p, r, f1 = precision_recall_f1(true_pairs, selected)
    ranks = {pair: idx + 1 for idx, (pair, _) in enumerate(ranked)}
    true_ranks = [ranks.get(pair, len(ranked) + 1) for pair in true_pairs]
    true_scores = [scores.get(pair, 0.0) for pair in true_pairs]
    false_scores = [score for pair, score in scores.items() if pair not in true_pairs]
    return {
        "selected_interactions": sorted(selected),
        "interaction_precision": p,
        "interaction_recall": r,
        "interaction_f1": f1,
        "true_interaction_rank_mean": float(np.mean(true_ranks)) if true_ranks else np.nan,
        "true_interaction_score_mean": float(np.mean(true_scores)) if true_scores else np.nan,
        "max_false_interaction_score": float(np.max(false_scores)) if false_scores else np.nan,
        "true_interaction_margin": float(np.mean(true_scores) - np.max(false_scores)) if false_scores else np.nan,
    }


def run_one(args: argparse.Namespace, seed: int, device: str) -> list[dict]:
    t0 = time.time()
    data = make_synthetic(
        function_name=args.function,
        n_train=args.samples,
        n_test=args.test_samples,
        d=args.dimension,
        noise=args.noise,
        seed=seed,
        standardize_target=True,
        nuisance_correlation=args.nuisance_correlation,
        n_correlated_proxies=args.n_correlated_proxies,
    )
    gt = data["ground_truth"]
    base = {
        "function": args.function,
        "samples": args.samples,
        "dimension": args.dimension,
        "noise": args.noise,
        "nuisance_correlation": args.nuisance_correlation,
        "n_correlated_proxies": args.n_correlated_proxies,
        "seed": seed,
        "hidden": args.hidden,
        "depth": args.depth,
        "epochs": args.epochs,
        "status": "ok",
        "error": "",
    }
    rows = []
    try:
        model, train_mse, test_mse = train_model(data, args, seed, device)
        train_time = time.time() - t0
        true_pairs = canonical_pairs(gt.interactions)
        for method in args.methods:
            score_t0 = time.time()
            if method == "nid":
                scores = nid_pair_scores(model)
            elif method == "hessian":
                scores = gradient_pair_scores(model, data["X_test"], device=device, points=args.hessian_points)
            else:
                raise ValueError(f"Unknown method={method}")
            row = dict(base)
            row.update(
                {
                    "method": method,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "train_runtime_sec": train_time,
                    "score_runtime_sec": time.time() - score_t0,
                }
            )
            row.update(evaluate(scores, true_pairs))
            rows.append(row)
    except Exception as exc:
        row = dict(base)
        row.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["function", "dimension", "samples", "noise", "nuisance_correlation", "method"]
    numeric = [
        "train_mse",
        "test_mse",
        "interaction_f1",
        "true_interaction_rank_mean",
        "true_interaction_margin",
        "train_runtime_sec",
        "score_runtime_sec",
    ]
    for col in numeric:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    out = ok.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(v) for v in c if v != "").rstrip("_") for c in out.columns]
    counts = ok.groupby(group_cols, dropna=False).size().reset_index(name="num_runs")
    return out.merge(counts, on=group_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="NID-style MLP interaction baseline.")
    parser.add_argument("--function", default="core_interaction_c025")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--test_samples", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--nuisance_correlation", type=float, default=0.0)
    parser.add_argument("--n_correlated_proxies", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=["nid", "hessian"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hessian_points", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out_dir", default="results/interaction_baselines/nid")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    detail_path = out_dir / "nid_interaction_detail.csv"
    for seed in args.seeds:
        print(f"[RUN] seed={seed} device={device}", flush=True)
        rows.extend(run_one(args, seed, device))
        pd.DataFrame(rows).to_csv(detail_path, index=False)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    summary_path = out_dir / "nid_interaction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False) if not summary.empty else detail.to_string(index=False))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
