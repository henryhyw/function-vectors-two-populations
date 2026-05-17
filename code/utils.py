"""Statistical helpers: bootstrap, FDR, BCa CI, deterministic seeding.

All routines are deterministic for fixed seed.
"""
from __future__ import annotations
import numpy as np
import torch
import random
from typing import Sequence


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(
    diffs: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "bca",
) -> tuple[float, float, float]:
    """Paired-prompt bootstrap CI on the mean of ``diffs``.

    Returns ``(point, lo, hi)`` for the (1-alpha) confidence interval.
    BCa adjustment by default (more accurate than percentile near tails).
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot_means[b] = diffs[idx].mean()
    point = float(diffs.mean())
    if method == "percentile":
        lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
        return point, float(lo), float(hi)

    # BCa: bias-correction + acceleration
    z0 = _norm_ppf((boot_means < point).mean() or 0.5 / B)
    jack = np.empty(n)
    for i in range(n):
        jack[i] = np.delete(diffs, i).mean()
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5 + 1e-30)
    a = num / den
    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)
    p_lo = _norm_cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = _norm_cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo, hi = np.quantile(boot_means, [p_lo, p_hi])
    return point, float(lo), float(hi)


def _norm_ppf(p: float) -> float:
    """Inverse CDF of a standard normal (vectorised for callers)."""
    from scipy.stats import norm
    return float(norm.ppf(p))


def _norm_cdf(z: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# FDR / multiple comparisons
# ---------------------------------------------------------------------------
def benjamini_hochberg(p: Sequence[float], q: float = 0.10) -> np.ndarray:
    """Return boolean array of significant tests under BH-FDR at level ``q``."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    sig_at = (p[order] <= thresh)
    if not sig_at.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.where(sig_at)[0].max()
    out = np.zeros(n, dtype=bool)
    out[order[: cutoff + 1]] = True
    return out


def holm_bonferroni(p: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm--Bonferroni step-down family-wise correction."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    cutoffs = alpha / (n - np.arange(n))
    sig_sorted = p[order] <= cutoffs
    out = np.zeros(n, dtype=bool)
    if not sig_sorted.any():
        return out
    for i in range(n):
        if sig_sorted[i]:
            out[order[i]] = True
        else:
            break
    return out


# ---------------------------------------------------------------------------
# TOST equivalence (induction-overlap)
# ---------------------------------------------------------------------------
def tost_hypergeometric(
    observed: int, K: int, n: int, N: int, margin: float | None = None,
    alpha: float = 0.05,
) -> str:
    """TOST equivalence to chance under hypergeometric expectation.

    ``observed`` = |W∪C ∩ top-K|, ``K`` = top-K size, ``n`` = |W∪C|,
    ``N`` = total head population.

    Returns one of: ``"equivalent_to_chance"``, ``"no_overlap"``,
    ``"inconclusive"``.
    """
    from scipy.stats import hypergeom
    expected = K * n / N
    var = K * n * (N - K) * (N - n) / (N * N * (N - 1))
    margin = margin if margin is not None else float(np.sqrt(max(var, 0.0)))
    if observed == 0:
        return "no_overlap"
    p_lower = hypergeom.cdf(observed, N, K, n)                           # P(X ≤ obs)
    p_upper = 1.0 - hypergeom.cdf(observed - 1, N, K, n)                 # P(X ≥ obs)
    # Two one-sided tests centered at expected with `margin` width
    p_low = 1.0 - hypergeom.cdf(int(np.floor(expected + margin)) - 1, N, K, n)
    p_high = hypergeom.cdf(int(np.ceil(expected - margin)), N, K, n)
    if p_low < alpha and p_high < alpha:
        return "equivalent_to_chance"
    return "inconclusive"
