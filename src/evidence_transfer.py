from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransferLink:
    """A directed handoff between two evidence-object success predicates."""

    name: str
    source_col: str
    target_col: str


DEFAULT_TRANSFER_LINKS: tuple[TransferLink, ...] = (
    TransferLink("prediction->support", "prediction_success", "support_success_all_true"),
    TransferLink("prediction->endpoint", "prediction_success", "endpoint_success"),
    TransferLink("prediction->pair", "prediction_success", "pair_success_all_true_at_budget"),
    TransferLink("support->pair", "support_success_all_true", "pair_success_all_true_at_budget"),
    TransferLink("endpoint->pair", "endpoint_success", "pair_success_all_true_at_budget"),
    TransferLink("pair->support", "pair_success_all_true_at_budget", "support_success_all_true"),
)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial count."""

    if n <= 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([math.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def summarize_transfer_link(
    df: pd.DataFrame,
    link: TransferLink,
) -> dict[str, float | int | str]:
    """Summarize unsupported transfer for one directed evidence-object link."""

    source = _bool_series(df, link.source_col)
    target = _bool_series(df, link.target_col)
    valid = source.notna() & target.notna()
    n = int(valid.sum())
    if n == 0:
        return {
            "transfer_link": link.name,
            "source_event": link.source_col,
            "target_event": link.target_col,
            "num_runs": 0,
            "source_success_count": 0,
            "target_success_count": 0,
            "transfer_failure_count": 0,
            "transfer_failure_rate": math.nan,
            "transfer_failure_ci_low": math.nan,
            "transfer_failure_ci_high": math.nan,
            "conditional_failure_rate": math.nan,
            "conditional_failure_ci_low": math.nan,
            "conditional_failure_ci_high": math.nan,
        }

    s = source[valid].astype(int)
    t = target[valid].astype(int)
    failure = (s.eq(1) & t.eq(0)).astype(int)
    source_count = int(s.sum())
    target_count = int(t.sum())
    failure_count = int(failure.sum())
    uncond = failure_count / n
    cond = failure_count / source_count if source_count > 0 else math.nan
    uncond_lo, uncond_hi = wilson_interval(failure_count, n)
    if source_count > 0:
        cond_lo, cond_hi = wilson_interval(failure_count, source_count)
    else:
        cond_lo, cond_hi = math.nan, math.nan
    return {
        "transfer_link": link.name,
        "source_event": link.source_col,
        "target_event": link.target_col,
        "num_runs": n,
        "source_success_count": source_count,
        "target_success_count": target_count,
        "transfer_failure_count": failure_count,
        "transfer_failure_rate": float(uncond),
        "transfer_failure_ci_low": float(uncond_lo),
        "transfer_failure_ci_high": float(uncond_hi),
        "conditional_failure_rate": float(cond) if math.isfinite(cond) else math.nan,
        "conditional_failure_ci_low": float(cond_lo) if math.isfinite(cond_lo) else math.nan,
        "conditional_failure_ci_high": float(cond_hi) if math.isfinite(cond_hi) else math.nan,
    }


def build_transfer_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    links: Iterable[TransferLink] = DEFAULT_TRANSFER_LINKS,
) -> pd.DataFrame:
    """Build a grouped unsupported-transfer table from evidence-object records."""

    group_cols = [c for c in group_cols if c in df.columns]
    rows: list[dict[str, object]] = []
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), df)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key))
        for link in links:
            rows.append({**base, **summarize_transfer_link(group, link)})
    return pd.DataFrame(rows)

