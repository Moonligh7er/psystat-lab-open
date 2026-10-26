"""Non-parametric depth — bootstrap CIs, permutation tests, robust statistics."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import itertools
import math

import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


class BootstrapCIRequest(BaseModel):
    data: list[float]
    statistic: str = "mean"  # "mean", "median", "std", "trimmed_mean"
    n_bootstrap: int = 10000
    confidence: float = 0.95
    trim_fraction: float = 0.1  # for trimmed mean


class PermutationTestRequest(BaseModel):
    group1: list[float]
    group2: list[float]
    statistic: str = "mean_diff"  # "mean_diff", "median_diff", "ks"
    n_permutations: int = 10000
    alternative: str = "two-sided"


class RobustStatsRequest(BaseModel):
    data: list[float]
    trim_fraction: float = 0.2


class TwoSampleBootstrapRequest(BaseModel):
    group1: list[float]
    group2: list[float]
    statistic: str = "mean_diff"  # "mean_diff", "ratio_means", "cohens_d"
    n_bootstrap: int = 10000
    confidence: float = 0.95


_STATS = {
    "mean": lambda a, t: a.mean(axis=-1),
    "median": lambda a, t: np.median(a, axis=-1),
    "std": lambda a, t: a.std(axis=-1, ddof=1),
    "trimmed_mean": lambda a, t: sp_stats.trim_mean(a, t, axis=-1),
}


def _check_resamples(n, name):
    if not 1000 <= n <= 50000:
        raise HTTPException(400, f"{name} must be between 1000 and 50,000.")


@router.post("/bootstrap-ci")
def bootstrap_confidence_interval(req: BootstrapCIRequest):
    """Percentile and BCa bootstrap confidence intervals (Efron, 1987) for a one-sample statistic.

    Resampling uses a fixed seed (42) so results are reproducible. The BCa acceleration is
    estimated by the jackknife.
    """
    data = np.asarray(req.data, dtype=float)
    if len(data) < 3:
        raise HTTPException(400, "Need at least 3 observations.")
    if not np.all(np.isfinite(data)):
        raise HTTPException(400, "Data must be finite numbers.")
    if req.statistic not in _STATS:
        raise HTTPException(400, f"Unknown statistic: {req.statistic}. Use mean, median, std or trimmed_mean.")
    if not 0.5 <= req.confidence < 1:
        raise HTTPException(400, "confidence must be between 0.5 and 1.")
    if not 0 <= req.trim_fraction < 0.5:
        raise HTTPException(400, "trim_fraction must be in [0, 0.5).")
    _check_resamples(req.n_bootstrap, "n_bootstrap")
    stat = _STATS[req.statistic]
    t = req.trim_fraction
    n = len(data)

    rng = np.random.default_rng(42)
    boot_stats = stat(data[rng.integers(0, n, size=(req.n_bootstrap, n))], t)
    observed = float(stat(data, t))

    alpha = 1 - req.confidence
    ci_lower, ci_upper = (float(v) for v in np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

    # BCa: bias correction z0 and jackknife acceleration a
    prop = np.clip(np.mean(boot_stats < observed), 1 / (2 * req.n_bootstrap), 1 - 1 / (2 * req.n_bootstrap))
    z0 = sp_stats.norm.ppf(prop)
    jack = stat(np.array([np.delete(data, j) for j in range(n)]), t)
    d = jack.mean() - jack
    denom = 6 * np.sum(d ** 2) ** 1.5
    a_hat = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0
    z_lo, z_hi = sp_stats.norm.ppf(alpha / 2), sp_stats.norm.ppf(1 - alpha / 2)
    adj_lo = sp_stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    adj_hi = sp_stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
    bca_lower, bca_upper = (float(v) for v in np.percentile(boot_stats, [100 * adj_lo, 100 * adj_hi]))

    return {
        "statistic": req.statistic,
        "observed_value": observed,
        "confidence": req.confidence,
        "n_bootstrap": req.n_bootstrap,
        "percentile_ci": {"lower": ci_lower, "upper": ci_upper},
        "bca_ci": {"lower": bca_lower, "upper": bca_upper},
        "bootstrap_mean": float(boot_stats.mean()),
        "bootstrap_std": float(boot_stats.std(ddof=1)),
        "bias": float(boot_stats.mean() - observed),
        "acceleration": a_hat,
        "bias_correction_z0": float(z0),
        "n": n,
        "seed": 42,
    }


def _perm_stat(kind, g1, g2):
    """Statistic for each row of g1/g2 arrays (rows = permutations)."""
    if kind == "mean_diff":
        return g1.mean(axis=-1) - g2.mean(axis=-1)
    if kind == "median_diff":
        return np.median(g1, axis=-1) - np.median(g2, axis=-1)
    # Kolmogorov-Smirnov D
    if g1.ndim == 1:
        return sp_stats.ks_2samp(g1, g2).statistic
    return np.array([sp_stats.ks_2samp(a, b).statistic for a, b in zip(g1, g2)])


@router.post("/permutation-test")
def permutation_test(req: PermutationTestRequest):
    """Two-sample permutation test.

    Uses the exact permutation distribution when the number of distinct splits is at most
    20,000; otherwise Monte Carlo with p = (b + 1) / (m + 1) (Phipson & Smyth, 2010),
    which is never exactly zero. Seeded for reproducibility.
    """
    g1 = np.asarray(req.group1, dtype=float)
    g2 = np.asarray(req.group2, dtype=float)
    if len(g1) < 2 or len(g2) < 2:
        raise HTTPException(400, "Each group must have at least 2 observations.")
    if not (np.all(np.isfinite(g1)) and np.all(np.isfinite(g2))):
        raise HTTPException(400, "Data must be finite numbers.")
    if req.statistic not in ("mean_diff", "median_diff", "ks"):
        raise HTTPException(400, f"Unknown statistic: {req.statistic}. Use mean_diff, median_diff or ks.")
    if req.alternative not in ("two-sided", "greater", "less"):
        raise HTTPException(400, "alternative must be two-sided, greater or less.")
    if req.statistic == "ks" and req.alternative != "two-sided":
        raise HTTPException(400, "The KS statistic is non-directional; use alternative='two-sided'.")
    _check_resamples(req.n_permutations, "n_permutations")

    combined = np.concatenate([g1, g2])
    n1, N = len(g1), len(g1) + len(g2)
    observed = float(_perm_stat(req.statistic, g1, g2))

    n_splits = math.comb(N, n1)
    if n_splits <= 20000:
        idx = np.array(list(itertools.combinations(range(N), n1)))
        mask = np.zeros((len(idx), N), dtype=bool)
        mask[np.arange(len(idx))[:, None], idx] = True
        perm_g1 = np.array([combined[m] for m in mask])
        perm_g2 = np.array([combined[~m] for m in mask])
        exact = True
    else:
        rng = np.random.default_rng(42)
        order = np.argsort(rng.random((req.n_permutations, N)), axis=1)
        shuffled = combined[order]
        perm_g1, perm_g2 = shuffled[:, :n1], shuffled[:, n1:]
        exact = False
    perm_stats = np.asarray(_perm_stat(req.statistic, perm_g1, perm_g2), dtype=float)

    eps = 1e-12 * max(1.0, abs(observed))  # tolerance so ties with the observed value count
    if req.alternative == "two-sided":
        b = int(np.sum(np.abs(perm_stats) >= abs(observed) - eps))
    elif req.alternative == "greater":
        b = int(np.sum(perm_stats >= observed - eps))
    else:
        b = int(np.sum(perm_stats <= observed + eps))
    m = len(perm_stats)
    p_value = b / m if exact else (b + 1) / (m + 1)

    return {
        "statistic": req.statistic,
        "observed_value": observed,
        "p_value": float(p_value),
        "method": "exact (all splits)" if exact else "Monte Carlo, (b + 1) / (m + 1)",
        "n_permutations": int(m),
        "alternative": req.alternative,
        "significant": bool(p_value < 0.05),
        "group1_n": len(g1),
        "group2_n": len(g2),
        "perm_mean": float(perm_stats.mean()),
        "perm_std": float(perm_stats.std(ddof=1)),
    }


@router.post("/robust-stats")
def robust_statistics(req: RobustStatsRequest):
    """Compute robust descriptive statistics resistant to outliers."""
    data = np.array(req.data)
    if len(data) < 3:
        raise HTTPException(400, "Need at least 3 observations.")

    n = len(data)
    sorted_data = np.sort(data)

    # Trimmed mean
    trimmed_mean = float(sp_stats.trim_mean(data, req.trim_fraction))

    # Winsorized mean and variance
    k = int(n * req.trim_fraction)
    winsorized = sorted_data.copy()
    if k > 0:
        winsorized[:k] = sorted_data[k]
        winsorized[-k:] = sorted_data[-k - 1]
    winsorized_mean = float(winsorized.mean())
    winsorized_var = float(winsorized.var(ddof=1))

    # MAD (Median Absolute Deviation)
    median_val = float(np.median(data))
    mad = float(np.median(np.abs(data - median_val)))
    mad_std = float(mad * 1.4826)  # scaled to match normal std

    # Interquartile mean
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqm_mask = (data >= q1) & (data <= q3)
    iqm = float(data[iqm_mask].mean()) if iqm_mask.sum() > 0 else float(median_val)

    # Huber M-estimator of location, k = 1.345, with the scale fixed at the normalised MAD
    # (residuals are standardised before the Huber loss is applied)
    scale = mad_std if mad_std > 0 else float(np.std(data, ddof=1)) or 1.0
    huber_location = median_val
    for _ in range(100):
        u = (data - huber_location) / scale
        w = np.where(np.abs(u) <= 1.345, 1.0, 1.345 / np.abs(np.where(u == 0, 1, u)))
        new_loc = float(np.sum(w * data) / np.sum(w))
        if abs(new_loc - huber_location) < 1e-12 * max(1.0, abs(new_loc)):
            huber_location = new_loc
            break
        huber_location = new_loc

    # Biweight midvariance (Tukey)
    u = (data - median_val) / (9 * mad + 1e-10)
    mask = np.abs(u) < 1
    if mask.sum() > 1:
        num = n * np.sum(((data[mask] - median_val) ** 2) * (1 - u[mask]**2)**4)
        denom = (np.sum((1 - u[mask]**2) * (1 - 5 * u[mask]**2)))**2
        biweight_var = float(num / (denom + 1e-10))
    else:
        biweight_var = float(data.var(ddof=1))

    return {
        "n": n,
        "mean": float(data.mean()),
        "trimmed_mean": trimmed_mean,
        "trim_fraction": req.trim_fraction,
        "winsorized_mean": winsorized_mean,
        "winsorized_variance": winsorized_var,
        "median": median_val,
        "mad": mad,
        "mad_std_estimate": mad_std,
        "iqm": iqm,
        "huber_location": huber_location,
        "huber_note": "Huber M-estimate (k = 1.345) with scale = 1.4826 x MAD",
        "biweight_midvariance": biweight_var,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


@router.post("/two-sample-bootstrap")
def two_sample_bootstrap(req: TwoSampleBootstrapRequest):
    """Bootstrap CI for two-sample comparisons."""
    g1 = np.array(req.group1)
    g2 = np.array(req.group2)
    if len(g1) < 2 or len(g2) < 2:
        raise HTTPException(400, "Each group must have at least 2 observations.")

    rng = np.random.default_rng(42)
    boot_stats = np.empty(req.n_bootstrap)

    for i in range(req.n_bootstrap):
        s1 = rng.choice(g1, size=len(g1), replace=True)
        s2 = rng.choice(g2, size=len(g2), replace=True)
        if req.statistic == "mean_diff":
            boot_stats[i] = s1.mean() - s2.mean()
        elif req.statistic == "ratio_means":
            boot_stats[i] = s1.mean() / s2.mean() if s2.mean() != 0 else float('nan')
        elif req.statistic == "cohens_d":
            pooled = np.sqrt(((len(s1)-1)*s1.var(ddof=1) + (len(s2)-1)*s2.var(ddof=1)) / (len(s1)+len(s2)-2))
            boot_stats[i] = (s1.mean() - s2.mean()) / pooled if pooled > 0 else 0

    boot_stats = boot_stats[~np.isnan(boot_stats)]
    alpha = 1 - req.confidence

    if req.statistic == "mean_diff":
        observed = float(g1.mean() - g2.mean())
    elif req.statistic == "ratio_means":
        observed = float(g1.mean() / g2.mean()) if g2.mean() != 0 else None
    else:
        pooled = np.sqrt(((len(g1)-1)*g1.var(ddof=1) + (len(g2)-1)*g2.var(ddof=1)) / (len(g1)+len(g2)-2))
        observed = float((g1.mean() - g2.mean()) / pooled) if pooled > 0 else 0.0

    return {
        "statistic": req.statistic,
        "observed_value": observed,
        "confidence": req.confidence,
        "n_bootstrap": req.n_bootstrap,
        "ci_lower": float(np.percentile(boot_stats, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        "bootstrap_mean": float(boot_stats.mean()),
        "bootstrap_std": float(boot_stats.std()),
        "group1_n": len(g1),
        "group2_n": len(g2),
    }
