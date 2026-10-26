"""What Would It Take — sample-size and sensitivity calculations for a result.

Given a test, an effect size and a sample size, reports:
  * the total N needed for 80% / 90% power to detect a given effect size
    (the observed one unless a target is supplied);
  * the smallest effect the current N detects with 80% power (sensitivity
    power analysis; Lakens, 2022);
  * the p-value (the result is significant for any alpha above it);
  * holding the effect size fixed, the smallest total N at which it would be
    significant.

Power uses the exact noncentral distributions G*Power uses (Faul et al., 2007):
noncentral t for t-tests, the exact sampling distribution of r for correlations
(Hotelling, 1953), and noncentral F with lambda = f²·N for ANOVA and regression.

Caveat: observed effect sizes are noisy and, for significant results, tend to be
overestimates, so "N needed for the observed effect" is optimistic. A smallest
effect of interest chosen in advance is the better target.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import integrate, special
from scipy import stats as sp_stats
from scipy.optimize import brentq

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────

class SensitivityRequest(BaseModel):
    # "ttest" (two independent groups, equal n), "paired", "one_sample",
    # "correlation", "anova", "regression"
    test_type: str
    # Effect size: Cohen's d (t-tests), r (correlation), Cohen's f (ANOVA), R² (regression)
    observed_es: float
    n: int  # total sample size
    observed_p: Optional[float] = None
    observed_stat: Optional[float] = None
    target_es: Optional[float] = None  # effect size to plan for; defaults to the observed one
    alpha: float = 0.05
    groups: int = 2  # ANOVA
    predictors: int = 1  # regression


class SensitivityResponse(BaseModel):
    effect_size_label: str
    effect_size_used: float  # the effect size the required-N figures are for
    planning_on_observed: bool
    current_power: Optional[float] = None  # power at the current N for effect_size_used
    required_n_80_power: Optional[int] = None  # total N
    required_n_90_power: Optional[int] = None
    min_detectable_es: Optional[float] = None  # 80% power at the current N
    alpha_threshold: Optional[float] = None  # = two-sided p-value of the observed effect
    n_for_significance: Optional[int] = None  # smallest total N at which the observed effect is significant
    practical_interpretation: str
    caveat: str


CAVEAT = ("Observed effect sizes are noisy, and significant results tend to overestimate them, so the N needed "
          "for the observed effect is usually optimistic. Plan on the smallest effect you would care about "
          "where possible. The 'N for significance' figure assumes the effect size would stay exactly the same, "
          "which real data will not do.")


# ── Power functions (two-sided, total N) ──────────────────────────────────

def _t_power(ncp: float, df: float, alpha: float) -> float:
    crit = sp_stats.t.ppf(1 - alpha / 2, df)
    return float(sp_stats.nct.sf(crit, df, ncp) + sp_stats.nct.cdf(-crit, df, ncp))


def power_two_sample(d: float, n: float, alpha: float) -> float:
    if n < 4:
        return 0.0
    return _t_power(abs(d) * math.sqrt(n / 4), n - 2, alpha)  # equal groups of n/2


def power_one_sample(d: float, n: float, alpha: float) -> float:
    if n < 2:
        return 0.0
    return _t_power(abs(d) * math.sqrt(n), n - 1, alpha)


def _log_r_density(r: np.ndarray, rho: float, n: int) -> np.ndarray:
    """Exact log density of the sample correlation r given rho (Hotelling, 1953)."""
    return (math.log(n - 2) + special.gammaln(n - 1) - 0.5 * math.log(2 * math.pi) - special.gammaln(n - 0.5)
            + (n - 1) / 2 * math.log1p(-rho * rho) + (n - 4) / 2 * np.log1p(-r * r)
            - (n - 1.5) * np.log1p(-rho * r)
            + np.log(special.hyp2f1(0.5, 0.5, n - 0.5, (1 + rho * r) / 2)))


def power_correlation(rho: float, n: float, alpha: float) -> float:
    n = int(round(n))
    if n < 4:
        return 0.0
    rho = min(abs(rho), 0.999999)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n - 2)
    r_crit = t_crit / math.sqrt(t_crit ** 2 + n - 2)
    f = lambda r: math.exp(float(_log_r_density(np.array([r]), rho, n)[0]))
    upper = integrate.quad(f, r_crit, 1, limit=200, points=[rho] if r_crit < rho < 1 else None)[0]
    lower = integrate.quad(f, -1, -r_crit, limit=200)[0]
    return float(min(1.0, max(0.0, upper + lower)))


def _f_power(f2: float, n: float, df1: int, df2: float, alpha: float) -> float:
    if df2 < 1:
        return 0.0
    crit = sp_stats.f.ppf(1 - alpha, df1, df2)
    return float(sp_stats.ncf.sf(crit, df1, df2, f2 * n))


def power_anova(f: float, n: float, alpha: float, k: int) -> float:
    return _f_power(f * f, n, k - 1, n - k, alpha)


def power_regression(r2: float, n: float, alpha: float, p: int) -> float:
    r2 = min(abs(r2), 0.999999)
    return _f_power(r2 / (1 - r2), n, p, n - p - 1, alpha)


def _required_n(power: Callable[[float, float], float], es: float, target: float,
                n_min: int, step: int, n_max: int = 10_000_000) -> Optional[int]:
    """Smallest N (a multiple of `step`, ≥ n_min) with power ≥ target."""
    if power(es, n_max) < target:
        return None
    lo, hi = n_min, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if power(es, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return int(math.ceil(lo / step) * step)


def _min_detectable(power: Callable[[float, float], float], n: int, es_max: float, target: float = 0.8) -> Optional[float]:
    g = lambda es: power(es, n) - target
    if g(es_max) < 0:
        return None
    return float(brentq(g, 1e-6, es_max, xtol=1e-6))


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=SensitivityResponse)
async def what_would_it_take(req: SensitivityRequest):
    tt = req.test_type.lower().replace("-", "_").replace(" ", "_")
    alias = {"t_test": "ttest", "t": "ttest", "independent": "ttest", "paired_t": "paired",
             "pearson": "correlation", "one_way_anova": "anova", "f_test": "anova"}
    tt = alias.get(tt, tt)
    if tt not in ("ttest", "paired", "one_sample", "correlation", "anova", "regression"):
        raise HTTPException(422, "test_type must be ttest, paired, one_sample, correlation, anova or regression.")
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    n, alpha, k, p = req.n, req.alpha, req.groups, req.predictors
    es_obs = abs(req.observed_es)
    es_plan = abs(req.target_es) if req.target_es is not None else es_obs

    if tt == "ttest":
        label, es_max, n_min, step = "Cohen's d", 10.0, 4, 2
        power = lambda es, m: power_two_sample(es, m, alpha)
        df_of = lambda m: m - 2
        t_of = lambda es, m: es * math.sqrt(m / 4)
    elif tt in ("paired", "one_sample"):
        label, es_max, n_min, step = "Cohen's d", 10.0, 2, 1
        power = lambda es, m: power_one_sample(es, m, alpha)
        df_of = lambda m: m - 1
        t_of = lambda es, m: es * math.sqrt(m)
    elif tt == "correlation":
        if es_obs >= 1 or es_plan >= 1:
            raise HTTPException(422, "A correlation must be strictly between −1 and 1.")
        label, es_max, n_min, step = "r", 0.999, 4, 1
        power = lambda es, m: power_correlation(es, m, alpha)
        df_of = lambda m: m - 2
        t_of = lambda es, m: es * math.sqrt((m - 2) / (1 - es * es))
    elif tt == "anova":
        if k < 2:
            raise HTTPException(422, "groups must be at least 2.")
        label, es_max, n_min, step = "Cohen's f", 10.0, k + 1, k
        power = lambda es, m: power_anova(es, m, alpha, k)
    else:
        if p < 1:
            raise HTTPException(422, "predictors must be at least 1.")
        if es_obs >= 1 or es_plan >= 1:
            raise HTTPException(422, "R² must be between 0 and 1.")
        label, es_max, n_min, step = "R²", 0.999, p + 2, 1
        power = lambda es, m: power_regression(es, m, alpha, p)

    if n < n_min:
        raise HTTPException(422, f"n must be at least {n_min} for this test.")

    req80 = _required_n(power, es_plan, 0.80, n_min, step) if es_plan > 0 else None
    req90 = _required_n(power, es_plan, 0.90, n_min, step) if es_plan > 0 else None
    current = power(es_plan, n) if es_plan > 0 else None
    mdes = _min_detectable(power, n, es_max)

    # p-value of the observed effect at the current N, and the N at which it would cross alpha
    def p_at(m: int) -> float:
        if tt in ("anova", "regression"):
            f2 = es_obs ** 2 if tt == "anova" else es_obs / (1 - es_obs)
            df1, df2 = (k - 1, m - k) if tt == "anova" else (p, m - p - 1)
            if df2 < 1:
                return 1.0
            return float(sp_stats.f.sf(f2 * df2 / df1, df1, df2))  # F = f²·df2/df1 for these designs
        df = df_of(m)
        return float(2 * sp_stats.t.sf(abs(t_of(es_obs, m)), df)) if df >= 1 else 1.0

    # Prefer the caller's actual p-value (it reflects the real design, e.g. unequal groups);
    # otherwise recompute it from the effect size and N.
    p_now = req.observed_p if req.observed_p is not None else (p_at(n) if es_obs > 0 else 1.0)
    n_sig = None
    if es_obs > 0:
        hi = n_min
        while p_at(hi) >= alpha and hi < 10_000_000:
            hi *= 2
        if p_at(hi) < alpha:
            lo = n_min
            while lo < hi:
                mid = (lo + hi) // 2
                if p_at(mid) < alpha:
                    hi = mid
                else:
                    lo = mid + 1
            n_sig = lo

    es_txt = f"{label} = {es_plan:.3g}"
    parts = []
    sig = p_now < alpha
    parts.append(f"With N = {n}, {label} = {es_obs:.3g} gives p = {p_now:.4g} "
                 f"({'significant' if sig else 'not significant'} at α = {alpha}).")
    if current is not None:
        parts.append(f"Power to detect {es_txt} at this N is {current:.0%}.")
    if req80 is not None:
        parts.append(f"Detecting {es_txt} with 80% power needs N = {req80:,}"
                     + (f" ({req80 - n:,} more)." if req80 > n else " (you already have that)."))
    if mdes is not None:
        parts.append(f"With N = {n}, the smallest effect detectable with 80% power is {label} = {mdes:.3g}.")
    if n_sig is not None:
        parts.append(f"If the effect size stayed at {es_obs:.3g} (and, for t-tests, groups were equal), "
                     f"the result would be significant from N = {n_sig:,}"
                     + (" — so it would stay significant with up to "
                        f"{n - n_sig:,} fewer participants." if sig and n_sig <= n else "."))

    return SensitivityResponse(
        effect_size_label=label, effect_size_used=round(es_plan, 6),
        planning_on_observed=req.target_es is None,
        current_power=round(current, 4) if current is not None else None,
        required_n_80_power=req80, required_n_90_power=req90,
        min_detectable_es=round(mdes, 4) if mdes is not None else None,
        alpha_threshold=p_now, n_for_significance=n_sig,
        practical_interpretation=" ".join(parts), caveat=CAVEAT,
    )
