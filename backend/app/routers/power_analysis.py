"""Power Analysis — G*Power-equivalent statistical power computations.

Provides endpoints for computing power, sample size, effect size, or alpha
for a wide range of statistical tests using scipy non-central distributions
and Brent's root-finding method.
"""

from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────


def _safe_float(v) -> float:
    """Convert numpy scalars / inf / nan to JSON-safe Python floats."""
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return None  # type: ignore[return-value]
    return v


def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.20:
        return "negligible"
    if d < 0.50:
        return "small"
    if d < 0.80:
        return "medium"
    return "large"


def _interpret_cohens_f(f: float) -> str:
    f = abs(f)
    if f < 0.10:
        return "negligible"
    if f < 0.25:
        return "small"
    if f < 0.40:
        return "medium"
    return "large"


def _interpret_cohens_w(w: float) -> str:
    w = abs(w)
    if w < 0.10:
        return "negligible"
    if w < 0.30:
        return "small"
    if w < 0.50:
        return "medium"
    return "large"


def _interpret_r(r: float) -> str:
    r = abs(r)
    if r < 0.10:
        return "negligible"
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "medium"
    return "large"


def _interpret_f_squared(f2: float) -> str:
    if f2 < 0.02:
        return "negligible"
    if f2 < 0.15:
        return "small"
    if f2 < 0.35:
        return "medium"
    return "large"


# ── T-test power core ────────────────────────────────────────────────────


def _ttest_power(
    d: float,
    n1: float,
    alpha: float,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    test_type: str = "independent",
) -> float:
    """Compute power for a t-test using the non-central t distribution.

    Parameters
    ----------
    d : Cohen's d effect size
    n1 : sample size (group 1 for independent, total n for paired/one-sample)
    alpha : significance level
    alternative : "two-sided" or "one-sided"
    ratio : n2/n1 allocation ratio (independent only)
    test_type : "independent", "paired", or "one-sample"
    """
    if test_type == "independent":
        n2 = n1 * ratio
        df = n1 + n2 - 2
        # non-centrality parameter
        ncp = d * math.sqrt((n1 * n2) / (n1 + n2))
    elif test_type == "paired":
        df = n1 - 1
        ncp = d * math.sqrt(n1)
    else:  # one-sample
        df = n1 - 1
        ncp = d * math.sqrt(n1)

    if df < 1:
        return 0.0

    nct = sp_stats.nct(df, ncp)

    if alternative == "two-sided":
        crit_upper = sp_stats.t.ppf(1 - alpha / 2, df)
        crit_lower = -crit_upper
        power = 1 - nct.cdf(crit_upper) + nct.cdf(crit_lower)
    else:
        crit = sp_stats.t.ppf(1 - alpha, df)
        power = 1 - nct.cdf(crit)

    return float(np.clip(power, 0, 1))


def _power_curve_vs_n(
    compute_power_fn, n_range: list[int], label: str = "power",
) -> list[dict]:
    return [{"n": n, label: _safe_float(compute_power_fn(n))} for n in n_range]


def _power_curve_vs_es(
    compute_power_fn, es_range: list[float], label: str = "power",
) -> list[dict]:
    return [
        {"effect_size": round(es, 4), label: _safe_float(compute_power_fn(es))}
        for es in es_range
    ]


# ── 1. T-Test Power ──────────────────────────────────────────────────────


class TTestPowerRequest(BaseModel):
    effect_size: Optional[float] = Field(None, description="Cohen's d")
    alpha: Optional[float] = Field(None, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n: Optional[int] = Field(None, ge=2, description="Per-group sample size")
    test_type: str = Field(
        "independent", description="independent | paired | one-sample"
    )
    alternative: str = Field("two-sided", description="two-sided | one-sided")
    allocation_ratio: float = Field(
        1.0, gt=0, description="n2/n1 ratio (independent only)"
    )


@router.post("/t-test")
def power_ttest(req: TTestPowerRequest):
    """Power analysis for t-tests (independent, paired, one-sample).

    Supply any 3 of {effect_size, alpha, power, n} to solve for the 4th.
    """
    given = {
        "effect_size": req.effect_size,
        "alpha": req.alpha,
        "power": req.power,
        "n": req.n,
    }
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        raise HTTPException(
            400, "Provide exactly 3 of {effect_size, alpha, power, n}."
        )
    solve_for = missing[0]

    if req.test_type not in ("independent", "paired", "one-sample"):
        raise HTTPException(400, "test_type must be independent, paired, or one-sample.")
    if req.alternative not in ("two-sided", "one-sided"):
        raise HTTPException(400, "alternative must be two-sided or one-sided.")

    def _pw(d=None, a=None, p=None, n=None):
        _d = d if d is not None else req.effect_size
        _a = a if a is not None else req.alpha
        _n = n if n is not None else req.n
        return _ttest_power(
            _d, _n, _a, req.alternative, req.allocation_ratio, req.test_type,
        )

    try:
        if solve_for == "power":
            result_value = _pw()
        elif solve_for == "n":
            target_power = req.power

            def f(n):
                return _pw(n=n) - target_power

            result_value = int(math.ceil(brentq(f, 2, 1_000_000, xtol=0.5)))
        elif solve_for == "effect_size":
            target_power = req.power

            def f(d):
                return _pw(d=d) - target_power

            result_value = round(float(brentq(f, 0.001, 10.0)), 6)
        else:  # alpha
            target_power = req.power

            def f(a):
                return _pw(a=a) - target_power

            result_value = round(float(brentq(f, 0.0001, 0.5)), 6)
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    # Build power curves
    final_d = result_value if solve_for == "effect_size" else req.effect_size
    final_n = result_value if solve_for == "n" else req.n
    final_a = result_value if solve_for == "alpha" else req.alpha

    n_range = list(range(5, min(max(int(final_n * 3), 100), 1001), max(int(final_n * 3 // 30), 2)))
    es_range = [round(x * 0.05, 2) for x in range(1, 41)]

    power_vs_n = _power_curve_vs_n(lambda n: _pw(d=final_d, a=final_a, n=n), n_range)
    power_vs_es = _power_curve_vs_es(lambda d: _pw(d=d, a=final_a, n=final_n), es_range)

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "effect_size": final_d,
            "alpha": final_a,
            "power": result_value if solve_for == "power" else req.power,
            "n": final_n,
            "test_type": req.test_type,
            "alternative": req.alternative,
            "allocation_ratio": req.allocation_ratio,
        },
        "effect_size_interpretation": _interpret_cohens_d(final_d),
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_effect_size": power_vs_es,
    }


# ── 2. One-Way ANOVA Power ───────────────────────────────────────────────


class AnovaPowerRequest(BaseModel):
    effect_size_f: Optional[float] = Field(None, description="Cohen's f")
    eta_squared: Optional[float] = Field(
        None, ge=0, lt=1, description="Eta-squared (converted to f internally)"
    )
    alpha: Optional[float] = Field(None, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n_per_group: Optional[int] = Field(None, ge=2)
    k_groups: int = Field(..., ge=2, description="Number of groups")


def _anova_power(f: float, alpha: float, n_per_group: float, k: int) -> float:
    """Power for one-way ANOVA using non-central F."""
    df1 = k - 1
    N = n_per_group * k
    df2 = N - k
    if df2 < 1:
        return 0.0
    ncp = f ** 2 * N
    crit = sp_stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - sp_stats.ncf.cdf(crit, df1, df2, ncp)
    return float(np.clip(power, 0, 1))


@router.post("/anova")
def power_anova(req: AnovaPowerRequest):
    """Power analysis for one-way ANOVA.

    Supply any 3 of {effect_size_f (or eta_squared), alpha, power, n_per_group}
    to solve for the 4th.  k_groups is always required.
    """
    # Convert eta-squared to f if provided
    if req.eta_squared is not None and req.effect_size_f is None:
        req.effect_size_f = math.sqrt(req.eta_squared / (1 - req.eta_squared))

    given = {
        "effect_size_f": req.effect_size_f,
        "alpha": req.alpha,
        "power": req.power,
        "n_per_group": req.n_per_group,
    }
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        raise HTTPException(
            400,
            "Provide exactly 3 of {effect_size_f/eta_squared, alpha, power, n_per_group}.",
        )
    solve_for = missing[0]
    k = req.k_groups

    try:
        if solve_for == "power":
            result_value = _anova_power(req.effect_size_f, req.alpha, req.n_per_group, k)
        elif solve_for == "n_per_group":
            target = req.power

            def f(n):
                return _anova_power(req.effect_size_f, req.alpha, n, k) - target

            result_value = int(math.ceil(brentq(f, 2, 1_000_000, xtol=0.5)))
        elif solve_for == "effect_size_f":
            target = req.power

            def f(ef):
                return _anova_power(ef, req.alpha, req.n_per_group, k) - target

            result_value = round(float(brentq(f, 0.001, 10.0)), 6)
        else:  # alpha
            target = req.power

            def f(a):
                return _anova_power(req.effect_size_f, a, req.n_per_group, k) - target

            result_value = round(float(brentq(f, 0.0001, 0.5)), 6)
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    final_f = result_value if solve_for == "effect_size_f" else req.effect_size_f
    final_n = result_value if solve_for == "n_per_group" else req.n_per_group
    final_a = result_value if solve_for == "alpha" else req.alpha

    n_range = list(range(5, min(max(int(final_n * 3), 100), 1001), max(int(final_n * 3 // 30), 2)))
    es_range = [round(x * 0.05, 2) for x in range(1, 41)]

    power_vs_n = _power_curve_vs_n(
        lambda n: _anova_power(final_f, final_a, n, k), n_range
    )
    power_vs_es = _power_curve_vs_es(
        lambda ef: _anova_power(ef, final_a, final_n, k), es_range
    )

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "effect_size_f": final_f,
            "eta_squared": round(final_f ** 2 / (1 + final_f ** 2), 6),
            "alpha": final_a,
            "power": result_value if solve_for == "power" else req.power,
            "n_per_group": final_n,
            "total_n": final_n * k if isinstance(final_n, int) else int(final_n) * k,
            "k_groups": k,
        },
        "effect_size_interpretation": _interpret_cohens_f(final_f),
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_effect_size": power_vs_es,
    }


# ── 3. Correlation Power ─────────────────────────────────────────────────


class CorrelationPowerRequest(BaseModel):
    r: Optional[float] = Field(None, ge=-0.999, le=0.999, description="Pearson r")
    alpha: Optional[float] = Field(None, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n: Optional[int] = Field(None, ge=4)
    alternative: str = Field("two-sided", description="two-sided | one-sided")


def _correlation_power(r: float, alpha: float, n: float, alternative: str) -> float:
    """Power for testing H0: rho=0 using Fisher z-transform."""
    z_r = np.arctanh(r)
    se = 1 / math.sqrt(n - 3) if n > 3 else 1e10

    if alternative == "two-sided":
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        power = (
            sp_stats.norm.cdf(-z_crit + z_r / se)
            + sp_stats.norm.cdf(-z_crit - z_r / se)
        )
        # Correct formulation: P(|Z| > z_crit) under H1
        power = (
            sp_stats.norm.sf(z_crit - z_r / se)
            + sp_stats.norm.cdf(-z_crit - z_r / se)
        )
    else:
        z_crit = sp_stats.norm.ppf(1 - alpha)
        power = sp_stats.norm.sf(z_crit - abs(z_r) / se)

    return float(np.clip(power, 0, 1))


@router.post("/correlation")
def power_correlation(req: CorrelationPowerRequest):
    """Power analysis for Pearson correlation.

    Supply any 3 of {r, alpha, power, n} to solve for the 4th.
    """
    given = {"r": req.r, "alpha": req.alpha, "power": req.power, "n": req.n}
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        raise HTTPException(400, "Provide exactly 3 of {r, alpha, power, n}.")
    solve_for = missing[0]

    if req.alternative not in ("two-sided", "one-sided"):
        raise HTTPException(400, "alternative must be two-sided or one-sided.")

    try:
        if solve_for == "power":
            result_value = _correlation_power(req.r, req.alpha, req.n, req.alternative)
        elif solve_for == "n":
            target = req.power

            def f(n):
                return _correlation_power(req.r, req.alpha, n, req.alternative) - target

            result_value = int(math.ceil(brentq(f, 4, 1_000_000, xtol=0.5)))
        elif solve_for == "r":
            target = req.power

            def f(r):
                return _correlation_power(r, req.alpha, req.n, req.alternative) - target

            result_value = round(float(brentq(f, 0.001, 0.999)), 6)
        else:  # alpha
            target = req.power

            def f(a):
                return _correlation_power(req.r, a, req.n, req.alternative) - target

            result_value = round(float(brentq(f, 0.0001, 0.5)), 6)
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    final_r = result_value if solve_for == "r" else req.r
    final_n = result_value if solve_for == "n" else req.n
    final_a = result_value if solve_for == "alpha" else req.alpha

    n_range = list(range(10, min(max(int(final_n * 3), 200), 1001), max(int(final_n * 3 // 30), 3)))
    es_range = [round(x * 0.02, 2) for x in range(1, 50)]

    power_vs_n = _power_curve_vs_n(
        lambda n: _correlation_power(final_r, final_a, n, req.alternative), n_range
    )
    power_vs_es = _power_curve_vs_es(
        lambda r: _correlation_power(r, final_a, final_n, req.alternative), es_range
    )

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "r": final_r,
            "alpha": final_a,
            "power": result_value if solve_for == "power" else req.power,
            "n": final_n,
            "alternative": req.alternative,
        },
        "effect_size_interpretation": _interpret_r(final_r),
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_effect_size": power_vs_es,
    }


# ── 4. Multiple Regression Power ─────────────────────────────────────────


class RegressionPowerRequest(BaseModel):
    f_squared: Optional[float] = Field(None, ge=0, description="Cohen's f-squared")
    r_squared: Optional[float] = Field(
        None, ge=0, lt=1, description="R-squared (converted to f-squared internally)"
    )
    alpha: Optional[float] = Field(None, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n: Optional[int] = Field(None, ge=5)
    n_predictors: int = Field(..., ge=1, description="Total number of predictors")
    n_tested: Optional[int] = Field(
        None,
        ge=1,
        description="Number of tested predictors (for partial f-squared). Defaults to n_predictors.",
    )


def _regression_power(
    f2: float, alpha: float, n: float, n_predictors: int, n_tested: int
) -> float:
    """Power for multiple regression (overall or partial) using non-central F."""
    df1 = n_tested
    df2 = n - n_predictors - 1
    if df2 < 1:
        return 0.0
    ncp = f2 * n
    crit = sp_stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - sp_stats.ncf.cdf(crit, df1, df2, ncp)
    return float(np.clip(power, 0, 1))


@router.post("/regression")
def power_regression(req: RegressionPowerRequest):
    """Power analysis for multiple regression.

    Supply any 3 of {f_squared (or r_squared), alpha, power, n} to solve for
    the 4th.  n_predictors is always required.
    """
    if req.r_squared is not None and req.f_squared is None:
        req.f_squared = req.r_squared / (1 - req.r_squared)

    n_tested = req.n_tested if req.n_tested is not None else req.n_predictors

    given = {
        "f_squared": req.f_squared,
        "alpha": req.alpha,
        "power": req.power,
        "n": req.n,
    }
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        raise HTTPException(
            400,
            "Provide exactly 3 of {f_squared/r_squared, alpha, power, n}.",
        )
    solve_for = missing[0]

    p = req.n_predictors

    try:
        if solve_for == "power":
            result_value = _regression_power(
                req.f_squared, req.alpha, req.n, p, n_tested
            )
        elif solve_for == "n":
            target = req.power

            def f(n):
                return _regression_power(req.f_squared, req.alpha, n, p, n_tested) - target

            # Minimum n must be > n_predictors + 1
            min_n = p + 2
            result_value = int(
                math.ceil(brentq(f, min_n, 1_000_000, xtol=0.5))
            )
        elif solve_for == "f_squared":
            target = req.power

            def f(f2):
                return _regression_power(f2, req.alpha, req.n, p, n_tested) - target

            result_value = round(float(brentq(f, 0.0001, 10.0)), 6)
        else:  # alpha
            target = req.power

            def f(a):
                return _regression_power(req.f_squared, a, req.n, p, n_tested) - target

            result_value = round(float(brentq(f, 0.0001, 0.5)), 6)
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    final_f2 = result_value if solve_for == "f_squared" else req.f_squared
    final_n = result_value if solve_for == "n" else req.n
    final_a = result_value if solve_for == "alpha" else req.alpha

    n_range = list(range(
        p + 2,
        min(max(int(final_n * 3), 200), 1001),
        max(int(final_n * 3 // 30), 3),
    ))
    es_range = [round(x * 0.02, 3) for x in range(1, 51)]

    power_vs_n = _power_curve_vs_n(
        lambda n: _regression_power(final_f2, final_a, n, p, n_tested), n_range
    )
    power_vs_es = _power_curve_vs_es(
        lambda f2: _regression_power(f2, final_a, final_n, p, n_tested), es_range
    )

    final_r2 = final_f2 / (1 + final_f2)

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "f_squared": final_f2,
            "r_squared": round(final_r2, 6),
            "alpha": final_a,
            "power": result_value if solve_for == "power" else req.power,
            "n": final_n,
            "n_predictors": p,
            "n_tested": n_tested,
        },
        "effect_size_interpretation": _interpret_f_squared(final_f2),
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_effect_size": power_vs_es,
    }


# ── 5. Chi-Square Power ──────────────────────────────────────────────────


class ChiSquarePowerRequest(BaseModel):
    w: Optional[float] = Field(None, ge=0, description="Cohen's w effect size")
    alpha: Optional[float] = Field(None, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n: Optional[int] = Field(None, ge=5)
    df: int = Field(..., ge=1, description="Degrees of freedom = (rows-1)*(cols-1)")


def _chisquare_power(w: float, alpha: float, n: float, df: int) -> float:
    """Power for chi-square test of independence using non-central chi-square."""
    ncp = w ** 2 * n
    crit = sp_stats.chi2.ppf(1 - alpha, df)
    power = 1 - sp_stats.ncx2.cdf(crit, df, ncp)
    return float(np.clip(power, 0, 1))


@router.post("/chi-square")
def power_chisquare(req: ChiSquarePowerRequest):
    """Power analysis for chi-square test of independence.

    Supply any 3 of {w, alpha, power, n} to solve for the 4th. df is required.
    """
    given = {"w": req.w, "alpha": req.alpha, "power": req.power, "n": req.n}
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        raise HTTPException(400, "Provide exactly 3 of {w, alpha, power, n}.")
    solve_for = missing[0]

    try:
        if solve_for == "power":
            result_value = _chisquare_power(req.w, req.alpha, req.n, req.df)
        elif solve_for == "n":
            target = req.power

            def f(n):
                return _chisquare_power(req.w, req.alpha, n, req.df) - target

            result_value = int(math.ceil(brentq(f, 5, 1_000_000, xtol=0.5)))
        elif solve_for == "w":
            target = req.power

            def f(w):
                return _chisquare_power(w, req.alpha, req.n, req.df) - target

            result_value = round(float(brentq(f, 0.001, 10.0)), 6)
        else:  # alpha
            target = req.power

            def f(a):
                return _chisquare_power(req.w, a, req.n, req.df) - target

            result_value = round(float(brentq(f, 0.0001, 0.5)), 6)
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    final_w = result_value if solve_for == "w" else req.w
    final_n = result_value if solve_for == "n" else req.n
    final_a = result_value if solve_for == "alpha" else req.alpha

    n_range = list(range(10, min(max(int(final_n * 3), 300), 2001), max(int(final_n * 3 // 30), 5)))
    es_range = [round(x * 0.02, 2) for x in range(1, 51)]

    power_vs_n = _power_curve_vs_n(
        lambda n: _chisquare_power(final_w, final_a, n, req.df), n_range
    )
    power_vs_es = _power_curve_vs_es(
        lambda w: _chisquare_power(w, final_a, final_n, req.df), es_range
    )

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "w": final_w,
            "alpha": final_a,
            "power": result_value if solve_for == "power" else req.power,
            "n": final_n,
            "df": req.df,
        },
        "effect_size_interpretation": _interpret_cohens_w(final_w),
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_effect_size": power_vs_es,
    }


# ── 6. Mixed / Repeated-Measures ANOVA Power ─────────────────────────────


class MixedAnovaPowerRequest(BaseModel):
    effect_size_f: float = Field(..., ge=0, description="Cohen's f")
    alpha: float = Field(0.05, ge=0.001, le=0.5)
    n_per_group: int = Field(..., ge=2)
    k_groups: int = Field(1, ge=1, description="Number of between-subject groups (1 = RM-only)")
    n_measurements: int = Field(..., ge=2, description="Number of repeated measurements")
    correlation_among_measures: float = Field(
        0.5, ge=0, lt=1, description="Average correlation among repeated measures"
    )
    epsilon: float = Field(
        1.0, ge=0.01, le=1.0,
        description="Sphericity correction (Greenhouse-Geisser epsilon). 1 = sphericity met.",
    )


@router.post("/mixed-anova")
def power_mixed_anova(req: MixedAnovaPowerRequest):
    """Power for repeated-measures and mixed ANOVA.

    Uses sphericity-corrected non-central F distribution.
    """
    f = req.effect_size_f
    k = req.k_groups
    m = req.n_measurements
    n = req.n_per_group
    rho = req.correlation_among_measures
    eps = req.epsilon
    alpha = req.alpha

    N = n * k

    # For the within-subjects effect (the repeated measure factor):
    # Corrected df
    df1 = (m - 1) * eps
    df2 = (N - k) * (m - 1) * eps

    if df2 < 1:
        raise HTTPException(400, "Denominator df < 1. Increase sample size or reduce groups.")

    # Non-centrality parameter for within-subjects effect
    # Accounts for correlation: the effective n is multiplied by
    # m * (1 - rho) because repeated measures reduce error variance
    ncp = (f ** 2) * N * m / (1 - rho) if rho < 1 else float("inf")

    crit = sp_stats.f.ppf(1 - alpha, df1, df2)
    power = float(np.clip(1 - sp_stats.ncf.cdf(crit, df1, df2, ncp), 0, 1))

    # Also compute between-subjects power if k > 1
    between_power = None
    if k > 1:
        df1_b = k - 1
        df2_b = N - k
        ncp_b = f ** 2 * N
        if df2_b >= 1:
            crit_b = sp_stats.f.ppf(1 - alpha, df1_b, df2_b)
            between_power = float(
                np.clip(1 - sp_stats.ncf.cdf(crit_b, df1_b, df2_b, ncp_b), 0, 1)
            )

    # Interaction power (if mixed design)
    interaction_power = None
    if k > 1:
        df1_int = (k - 1) * (m - 1) * eps
        df2_int = (N - k) * (m - 1) * eps
        ncp_int = (f ** 2) * N * m / (1 - rho) if rho < 1 else float("inf")
        if df2_int >= 1:
            crit_int = sp_stats.f.ppf(1 - alpha, df1_int, df2_int)
            interaction_power = float(
                np.clip(1 - sp_stats.ncf.cdf(crit_int, df1_int, df2_int, ncp_int), 0, 1)
            )

    # Power curve: vary n_per_group
    n_range = list(range(5, min(max(n * 5, 100), 501), max(n * 5 // 25, 2)))

    def _pw_for_n(n_val):
        N_val = n_val * k
        df2_val = (N_val - k) * (m - 1) * eps
        if df2_val < 1:
            return 0.0
        ncp_val = (f ** 2) * N_val * m / (1 - rho) if rho < 1 else float("inf")
        crit_val = sp_stats.f.ppf(1 - alpha, df1, df2_val)
        return float(np.clip(1 - sp_stats.ncf.cdf(crit_val, df1, df2_val, ncp_val), 0, 1))

    power_vs_n = _power_curve_vs_n(_pw_for_n, n_range)

    return {
        "within_subjects_power": power,
        "between_subjects_power": between_power,
        "interaction_power": interaction_power,
        "parameters": {
            "effect_size_f": f,
            "alpha": alpha,
            "n_per_group": n,
            "total_n": N,
            "k_groups": k,
            "n_measurements": m,
            "correlation_among_measures": rho,
            "epsilon": eps,
            "corrected_df1": round(df1, 2),
            "corrected_df2": round(df2, 2),
        },
        "effect_size_interpretation": _interpret_cohens_f(f),
        "power_curve_vs_n": power_vs_n,
    }


# ── 7. Logistic Regression Power ─────────────────────────────────────────


class LogisticRegressionPowerRequest(BaseModel):
    odds_ratio: float = Field(..., gt=0, description="Odds ratio of interest")
    p_exposure: float = Field(
        ..., gt=0, lt=1,
        description="Proportion of subjects exposed (or P(X=1) for binary predictor)",
    )
    alpha: Optional[float] = Field(0.05, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n: Optional[int] = Field(None, ge=10)
    p_outcome: float = Field(
        0.5, gt=0, lt=1,
        description="Overall probability of the outcome under H0",
    )


def _logistic_power_hsieh(
    odds_ratio: float,
    p_exposure: float,
    p_outcome: float,
    alpha: float,
    n: float,
) -> float:
    """Power for logistic regression — Hsieh (1989) / Demidenko (2007) formula.

    Uses the variance of log(OR) for a binary predictor.
    """
    log_or = math.log(odds_ratio)
    # P(Y=1|X=0) solve from overall: p_outcome = p_exp * p1 + (1-p_exp) * p0
    # with OR = (p1/(1-p1)) / (p0/(1-p0))
    # Approximate: use p_outcome as baseline p0
    p0 = p_outcome
    odds0 = p0 / (1 - p0)
    p1 = (odds_ratio * odds0) / (1 + odds_ratio * odds0)

    # Variance of the Wald statistic
    var_beta = 1 / (
        n * p_exposure * (1 - p_exposure) * p0 * (1 - p0)
    )

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_stat = abs(log_or) / math.sqrt(var_beta)
    power = sp_stats.norm.sf(z_alpha - z_stat) + sp_stats.norm.cdf(-z_alpha - z_stat)
    return float(np.clip(power, 0, 1))


@router.post("/logistic-regression")
def power_logistic_regression(req: LogisticRegressionPowerRequest):
    """Power analysis for logistic regression (binary predictor).

    Supply either power (to solve for n) or n (to compute power).
    odds_ratio, p_exposure, and alpha are required.
    """
    if req.power is None and req.n is None:
        raise HTTPException(400, "Provide either 'power' or 'n'.")
    if req.power is not None and req.n is not None:
        raise HTTPException(400, "Provide only one of 'power' or 'n' (solve for the other).")

    try:
        if req.n is not None:
            # Solve for power
            result_value = _logistic_power_hsieh(
                req.odds_ratio, req.p_exposure, req.p_outcome, req.alpha, req.n
            )
            solve_for = "power"
        else:
            # Solve for n
            target = req.power

            def f(n):
                return (
                    _logistic_power_hsieh(
                        req.odds_ratio, req.p_exposure, req.p_outcome, req.alpha, n
                    )
                    - target
                )

            result_value = int(math.ceil(brentq(f, 10, 1_000_000, xtol=0.5)))
            solve_for = "n"
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    final_n = result_value if solve_for == "n" else req.n

    # Power curve vs n
    n_range = list(range(20, min(max(int(final_n * 3), 500), 5001), max(int(final_n * 3 // 30), 10)))
    power_vs_n = _power_curve_vs_n(
        lambda n: _logistic_power_hsieh(
            req.odds_ratio, req.p_exposure, req.p_outcome, req.alpha, n
        ),
        n_range,
    )

    # Power curve vs odds ratio
    or_range = [round(0.2 + i * 0.1, 1) for i in range(37)]  # 0.2 to 4.0
    or_range = [v for v in or_range if v != 1.0]  # exclude OR=1 (null)
    power_vs_or = [
        {
            "odds_ratio": or_val,
            "power": _safe_float(
                _logistic_power_hsieh(
                    or_val, req.p_exposure, req.p_outcome, req.alpha, final_n
                )
            ),
        }
        for or_val in or_range
    ]

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "odds_ratio": req.odds_ratio,
            "log_odds_ratio": round(math.log(req.odds_ratio), 6),
            "p_exposure": req.p_exposure,
            "p_outcome": req.p_outcome,
            "alpha": req.alpha,
            "power": result_value if solve_for == "power" else req.power,
            "n": final_n,
        },
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_odds_ratio": power_vs_or,
    }


# ── 8. Equivalence (TOST) Power ──────────────────────────────────────────


class EquivalencePowerRequest(BaseModel):
    equivalence_bound: float = Field(
        ..., gt=0, description="Symmetric equivalence bound (absolute units)"
    )
    true_difference: float = Field(
        0.0, description="Assumed true difference between means"
    )
    sd: float = Field(..., gt=0, description="Standard deviation")
    alpha: Optional[float] = Field(0.05, ge=0.001, le=0.5)
    power: Optional[float] = Field(None, ge=0.01, le=0.9999)
    n: Optional[int] = Field(None, ge=4, description="Per-group sample size")
    n_groups: int = Field(
        2, ge=1, le=2,
        description="1 for one-sample TOST, 2 for two-sample TOST",
    )


def _tost_power(
    bound: float,
    true_diff: float,
    sd: float,
    alpha: float,
    n: float,
    n_groups: int,
) -> float:
    """Power for TOST equivalence test.

    Power = P(reject both one-sided tests).
    """
    if n_groups == 2:
        se = sd * math.sqrt(2 / n)
        df = 2 * (n - 1)
    else:
        se = sd / math.sqrt(n)
        df = n - 1

    if df < 1 or se <= 0:
        return 0.0

    t_crit = sp_stats.t.ppf(1 - alpha, df)

    # Non-centrality parameters for the two one-sided tests
    ncp_lower = (true_diff - (-bound)) / se  # test H0: mu_diff <= -bound
    ncp_upper = (true_diff - bound) / se  # test H0: mu_diff >= bound

    # Power = P(T1 > crit) * P(T2 < -crit)  approximately
    # More precisely: P(T1 > t_crit AND T2 < -t_crit)
    # Since they are highly correlated, approximate as min of the two
    power_lower = 1 - sp_stats.nct.cdf(t_crit, df, ncp_lower)
    power_upper = sp_stats.nct.cdf(-t_crit, df, ncp_upper)

    # Joint power: P(both reject) = P(lower rejects) + P(upper rejects) - 1
    # when both are near 1, or more accurately:
    # Power = P(lower rejects) - P(lower rejects AND upper does not reject)
    # Good approximation since the two tests share the same t-statistic:
    power = power_lower + power_upper - 1
    # But can also be computed exactly since both conditions use same T:
    # P(-t_crit + ncp_upper < T < t_crit + ncp_lower) ... simplify:
    # Actually for TOST the exact power using the same t-statistic is:
    # P(t > t_crit | ncp_lower) and P(t < -t_crit | ncp_upper)
    # which for same T means power = max(0, power_lower + power_upper - 1)
    power = max(0.0, power)

    return float(np.clip(power, 0, 1))


@router.post("/equivalence")
def power_equivalence(req: EquivalencePowerRequest):
    """Power analysis for TOST equivalence tests.

    Supply either power (to solve for n) or n (to compute power).
    """
    if req.power is None and req.n is None:
        raise HTTPException(400, "Provide either 'power' or 'n'.")
    if req.power is not None and req.n is not None:
        raise HTTPException(400, "Provide only one of 'power' or 'n'.")

    try:
        if req.n is not None:
            result_value = _tost_power(
                req.equivalence_bound,
                req.true_difference,
                req.sd,
                req.alpha,
                req.n,
                req.n_groups,
            )
            solve_for = "power"
        else:
            target = req.power

            def f(n):
                return (
                    _tost_power(
                        req.equivalence_bound,
                        req.true_difference,
                        req.sd,
                        req.alpha,
                        n,
                        req.n_groups,
                    )
                    - target
                )

            result_value = int(math.ceil(brentq(f, 4, 1_000_000, xtol=0.5)))
            solve_for = "n"
    except ValueError as exc:
        raise HTTPException(
            422, f"Could not solve for {solve_for}: {exc}"
        ) from exc

    final_n = result_value if solve_for == "n" else req.n

    n_range = list(range(5, min(max(int(final_n * 3), 200), 2001), max(int(final_n * 3 // 30), 5)))
    power_vs_n = _power_curve_vs_n(
        lambda n: _tost_power(
            req.equivalence_bound,
            req.true_difference,
            req.sd,
            req.alpha,
            n,
            req.n_groups,
        ),
        n_range,
    )

    # Power vs true difference
    bound = req.equivalence_bound
    diff_range = [round(-bound + i * bound / 10, 4) for i in range(21)]
    power_vs_diff = [
        {
            "true_difference": d,
            "power": _safe_float(
                _tost_power(bound, d, req.sd, req.alpha, final_n, req.n_groups)
            ),
        }
        for d in diff_range
    ]

    return {
        "solved_for": solve_for,
        "result": result_value,
        "parameters": {
            "equivalence_bound": req.equivalence_bound,
            "true_difference": req.true_difference,
            "sd": req.sd,
            "alpha": req.alpha,
            "power": result_value if solve_for == "power" else req.power,
            "n_per_group": final_n,
            "total_n": final_n * req.n_groups,
            "n_groups": req.n_groups,
        },
        "power_curve_vs_n": power_vs_n,
        "power_curve_vs_true_difference": power_vs_diff,
    }


# ── 9. Sample Size Table ─────────────────────────────────────────────────


class SampleSizeTableRequest(BaseModel):
    analysis_type: str = Field(
        ...,
        description=(
            "One of: t-test, anova, correlation, regression, chi-square"
        ),
    )
    effect_sizes: list[float] = Field(
        ..., min_length=1, max_length=50, description="Range of effect sizes"
    )
    sample_sizes: list[int] = Field(
        ..., min_length=1, max_length=50, description="Range of sample sizes"
    )
    alpha: float = Field(0.05, ge=0.001, le=0.5)
    # Extra params for specific analyses
    k_groups: Optional[int] = Field(None, ge=2, description="For ANOVA")
    n_predictors: Optional[int] = Field(None, ge=1, description="For regression")
    df: Optional[int] = Field(None, ge=1, description="For chi-square")
    alternative: str = Field("two-sided")
    test_type: str = Field("independent", description="For t-test: independent|paired|one-sample")


@router.post("/sample-size-table")
def sample_size_table(req: SampleSizeTableRequest):
    """Generate a power table (matrix) for a given analysis type.

    Returns power values for every combination of effect size x sample size.
    """
    analysis = req.analysis_type.lower().replace(" ", "-").replace("_", "-")

    def _compute_cell(es: float, n: int) -> Optional[float]:
        try:
            if analysis == "t-test":
                return _ttest_power(es, n, req.alpha, req.alternative, 1.0, req.test_type)
            elif analysis == "anova":
                k = req.k_groups or 3
                return _anova_power(es, req.alpha, n, k)
            elif analysis == "correlation":
                if n < 4:
                    return None
                return _correlation_power(es, req.alpha, n, req.alternative)
            elif analysis == "regression":
                p = req.n_predictors or 2
                if n <= p + 1:
                    return None
                return _regression_power(es, req.alpha, n, p, p)
            elif analysis == "chi-square":
                d = req.df or 1
                return _chisquare_power(es, req.alpha, n, d)
            else:
                return None
        except Exception:
            return None

    matrix = []
    for es in req.effect_sizes:
        row = []
        for n in req.sample_sizes:
            pw = _compute_cell(es, n)
            row.append(_safe_float(pw) if pw is not None else None)
        matrix.append(row)

    # Find minimum n for 80% power at each effect size
    recommended_n = []
    for i, es in enumerate(req.effect_sizes):
        found = None
        for j, n in enumerate(req.sample_sizes):
            if matrix[i][j] is not None and matrix[i][j] >= 0.80:
                found = n
                break
        recommended_n.append(found)

    return {
        "analysis_type": req.analysis_type,
        "alpha": req.alpha,
        "effect_sizes": req.effect_sizes,
        "sample_sizes": req.sample_sizes,
        "power_matrix": matrix,
        "recommended_n_for_80_power": recommended_n,
        "note": "power_matrix[i][j] = power for effect_sizes[i] and sample_sizes[j]",
    }
