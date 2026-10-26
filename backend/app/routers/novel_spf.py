"""
Correlation Stability (formerly "Scale-Persistent Features")
===========================================================
For every pair of variables: the Pearson correlation, a Holm-adjusted
p-value across all pairs, a percentile bootstrap CI, and a case-dropping
stability coefficient.

Case-dropping stability (adapted from the CS-coefficient of Epskamp,
Borsboom & Fried, 2018): repeatedly drop a share of cases at random and
recompute r. The coefficient is the largest share that can be dropped while
at least 95% of the subsamples keep the sign of the full-sample r.
Epskamp et al. recommend at least 0.25 and preferably 0.5.

What this is and is not: every subsample is drawn from your one sample, so
these numbers describe how precisely r is estimated — they behave like the
p-value and CI do. They cannot show that a relationship will replicate in a
new sample, and they are not topological data analysis.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as scipy_stats

router = APIRouter()

DROP_SHARES = [round(x, 2) for x in np.arange(0.05, 0.751, 0.05)]
N_DROP_REPS = 500
SIGN_KEEP = 0.95
STABLE, ACCEPTABLE = 0.5, 0.25  # Epskamp et al. (2018) guidance


# ── Request / Response ────────────────────────────────────────────────

class SPFRequest(BaseModel):
    data: Dict[str, List[float]]  # variable_name → values
    n_bootstrap: int = 2000
    alpha: float = 0.05


class PairwiseResult(BaseModel):
    var1: str
    var2: str
    n: int
    full_correlation: float
    full_p_value: float
    holm_p_value: float  # adjusted across all pairs

    # Case-dropping stability
    subsample_correlations: dict  # drop share → median r of the subsamples
    sign_kept_by_drop: dict  # drop share → share of subsamples keeping the full-sample sign
    direction_stability: float  # share keeping the sign when 50% of cases are dropped

    # Bootstrap
    bootstrap_median: float
    bootstrap_sd: float
    bootstrap_ci_lower: float  # 95% percentile interval
    bootstrap_ci_upper: float
    bootstrap_includes_zero: bool

    # Stability coefficient (0–0.75) and label
    persistence_score: float  # the case-dropping stability coefficient
    persistence_label: str  # "Stable" (≥ 0.5), "Acceptable" (≥ 0.25), "Unstable"

    interpretation: str


class SPFResponse(BaseModel):
    n_observations: int
    n_variables: int
    n_pairs: int
    features: List[PairwiseResult]
    persistent_features: List[str]  # "Stable" pairs
    moderate_features: List[str]  # "Acceptable" pairs
    fragile_features: List[str]  # "Unstable" pairs
    summary: str
    interpretation: str


# ── Helpers ───────────────────────────────────────────────────────────

def _pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r for matrices of resampled x and y (NaN where undefined)."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (xc * yc).sum(axis=1) / den, np.nan)


CHUNK_CELLS = 2_000_000  # resample in chunks so memory stays bounded for large n


def _chunked(reps: int, n: int, draw) -> np.ndarray:
    """Concatenate draw(k) over chunks of at most CHUNK_CELLS / n replicates."""
    size = max(1, CHUNK_CELLS // n)
    return np.concatenate([draw(min(size, reps - s)) for s in range(0, reps, size)])


def _case_dropping(x: np.ndarray, y: np.ndarray, full_r: float, rng) -> tuple[dict, dict, float]:
    n = len(x)
    sign = np.sign(full_r)
    medians, kept = {}, {}
    cs = 0.0
    for share in DROP_SHARES:
        m = n - int(round(share * n))
        if m < 4:
            break

        def draw(k, m=m):
            idx = np.argsort(rng.random((k, n)), axis=1)[:, :m]  # k subsamples without replacement
            return _pearson_rows(x[idx], y[idx])

        rs = _chunked(N_DROP_REPS, n, draw)
        rs = rs[~np.isnan(rs)]
        if len(rs) == 0:
            break
        keep = float(np.mean(np.sign(rs) == sign)) if sign != 0 else 0.0
        medians[str(share)] = round(float(np.median(rs)), 4)
        kept[str(share)] = round(keep, 4)
        if keep >= SIGN_KEEP:
            cs = share
    return medians, kept, cs


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/analyze")
async def scale_persistent_features(req: SPFRequest):
    names = list(req.data.keys())
    if len(names) < 2:
        raise HTTPException(422, "Need at least 2 variables.")
    lengths = {k: len(v) for k, v in req.data.items()}
    if len(set(lengths.values())) > 1:
        raise HTTPException(422, f"All variables must have the same length. Got: {lengths}")
    n_total = next(iter(lengths.values()))
    if n_total < 10:
        raise HTTPException(422, "Need at least 10 observations.")
    if not 200 <= req.n_bootstrap <= 20000:
        raise HTTPException(422, "n_bootstrap must be between 200 and 20000.")

    rng = np.random.default_rng(20240101)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v1, v2 = names[i], names[j]
            x = np.asarray(req.data[v1], dtype=float)
            y = np.asarray(req.data[v2], dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
                continue
            r, p = scipy_stats.pearsonr(x, y)
            r, p = float(r), float(p)

            medians, kept, cs = _case_dropping(x, y, r, rng)
            def boot_draw(k, x=x, y=y):
                idx = rng.integers(0, len(x), size=(k, len(x)))
                return _pearson_rows(x[idx], y[idx])

            boot = _chunked(req.n_bootstrap, len(x), boot_draw)
            boot = boot[~np.isnan(boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(v1=v1, v2=v2, n=len(x), r=r, p=p, medians=medians, kept=kept, cs=cs,
                             b_med=float(np.median(boot)), b_sd=float(np.std(boot, ddof=1)),
                             lo=float(lo), hi=float(hi)))

    if not rows:
        raise HTTPException(422, "No variable pair had at least 10 complete, non-constant observations.")

    holm = _holm([row["p"] for row in rows])
    features, stable, acceptable, unstable = [], [], [], []
    for row, hp in zip(rows, holm):
        cs = row["cs"]
        label = "Stable" if cs >= STABLE else "Acceptable" if cs >= ACCEPTABLE else "Unstable"
        pair = f"{row['v1']} ↔ {row['v2']}"
        {"Stable": stable, "Acceptable": acceptable, "Unstable": unstable}[label].append(pair)
        sig = hp < req.alpha
        interp = (
            f"r = {row['r']:.3f} (n = {row['n']}), 95% bootstrap CI [{row['lo']:.3f}, {row['hi']:.3f}], "
            f"Holm-adjusted p = {hp:.4g}. "
            + (f"The sign held in at least 95% of subsamples with up to {cs:.0%} of cases dropped "
               f"(stability coefficient {cs:.2f}, {label.lower()} by Epskamp et al.'s guidance). "
               if cs > 0 else
               "The sign did not hold in 95% of subsamples even with 5% of cases dropped (unstable). ")
            + ("The correlation is significant after adjusting for the number of pairs." if sig else
               "The correlation is not significant after adjusting for the number of pairs.")
        )
        features.append(PairwiseResult(
            var1=row["v1"], var2=row["v2"], n=row["n"],
            full_correlation=round(row["r"], 6), full_p_value=row["p"], holm_p_value=hp,
            subsample_correlations=row["medians"], sign_kept_by_drop=row["kept"],
            direction_stability=row["kept"].get("0.5", 0.0),
            bootstrap_median=round(row["b_med"], 6), bootstrap_sd=round(row["b_sd"], 6),
            bootstrap_ci_lower=round(row["lo"], 6), bootstrap_ci_upper=round(row["hi"], 6),
            bootstrap_includes_zero=bool(row["lo"] <= 0 <= row["hi"]),
            persistence_score=cs, persistence_label=label, interpretation=interp,
        ))

    features.sort(key=lambda f: (f.persistence_score, abs(f.full_correlation)), reverse=True)
    total = len(features)
    summary = (f"{total} variable pairs ({len(names)} variables, n = {n_total}): "
               f"{len(stable)} stable, {len(acceptable)} acceptable, {len(unstable)} unstable.")
    interpretation = (
        "Stability here means the sign of r is precisely estimated in this sample: it behaves like "
        "a small p-value and a CI that excludes zero, and it reflects the same sample size and effect "
        "size. It does not show that a correlation will replicate in new data, and it does not test "
        "for non-linear relationships. Holm-adjusted p-values account for testing many pairs."
    )
    return SPFResponse(
        n_observations=n_total, n_variables=len(names), n_pairs=total, features=features,
        persistent_features=stable, moderate_features=acceptable, fragile_features=unstable,
        summary=summary, interpretation=interpretation,
    )
