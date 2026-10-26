"""Bayesian Lab — Bayes Factor computation for common tests."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.special import gammaln

router = APIRouter()


class BayesFactorTTestRequest(BaseModel):
    group1: list[float]
    group2: list[float]
    prior_scale: float = 0.707  # Cauchy prior scale (JZS default)


class BayesFactorCorrelationRequest(BaseModel):
    x: list[float]
    y: list[float]


class BayesFactorProportionRequest(BaseModel):
    successes1: int
    trials1: int
    successes2: int
    trials2: int


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/bayes-factor/t-test")
def bayes_factor_t_test(req: BayesFactorTTestRequest):
    """JZS Bayes Factor for independent samples t-test."""
    g1, g2 = np.array(req.group1), np.array(req.group2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise HTTPException(400, "Each group needs at least 2 observations.")

    # Frequentist t-test for reference
    t_stat, p_value = sp_stats.ttest_ind(g1, g2)
    df = n1 + n2 - 2
    n_eff = (n1 * n2) / (n1 + n2)

    # JZS Bayes Factor (Rouder et al., 2009) — numerical integration
    bf10 = _jzs_bf(t_stat, n_eff, df, req.prior_scale)

    return {
        "bf10": float(bf10),
        "bf01": float(1 / bf10) if bf10 > 0 else float("inf"),
        "log_bf10": float(np.log(bf10)) if bf10 > 0 else float("-inf"),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": df,
        "interpretation": _interpret_bf(bf10),
        "prior_scale": req.prior_scale,
        "evidence_for": "H1 (effect exists)" if bf10 > 1 else "H0 (no effect)",
    }


@router.post("/bayes-factor/correlation")
def bayes_factor_correlation(req: BayesFactorCorrelationRequest):
    """Bayes Factor for Pearson correlation (Ly et al., 2016; uniform prior on rho)."""
    x, y = np.array(req.x), np.array(req.y)
    if len(x) != len(y):
        raise HTTPException(400, "x and y must have the same length.")
    n = len(x)
    if n < 3:
        raise HTTPException(400, "Need at least 3 data points.")

    r, p_value = sp_stats.pearsonr(x, y)

    bf10 = _bf_correlation(r, n)

    return {
        "bf10": float(bf10),
        "bf01": float(1 / bf10) if bf10 > 0 else float("inf"),
        "r": float(r),
        "p_value": float(p_value),
        "n": n,
        "interpretation": _interpret_bf(bf10),
        "evidence_for": "H1 (correlation exists)" if bf10 > 1 else "H0 (no correlation)",
    }


@router.post("/bayes-factor/proportion")
def bayes_factor_proportion(req: BayesFactorProportionRequest):
    """Bayes Factor for comparing two proportions (beta-binomial model)."""
    s1, t1 = req.successes1, req.trials1
    s2, t2 = req.successes2, req.trials2

    if t1 < 1 or t2 < 1:
        raise HTTPException(400, "Each group must have at least 1 trial.")
    if s1 > t1 or s2 > t2 or s1 < 0 or s2 < 0:
        raise HTTPException(400, "Successes must be between 0 and trials.")

    # Log marginal likelihood under H0 (common proportion)
    log_m_h0 = (gammaln(2) - gammaln(t1 + t2 + 2)
                + gammaln(s1 + s2 + 1) + gammaln(t1 + t2 - s1 - s2 + 1))

    # Log marginal likelihood under H1 (separate proportions)
    log_m_h1 = (gammaln(2) - gammaln(t1 + 2) + gammaln(s1 + 1) + gammaln(t1 - s1 + 1)
                + gammaln(2) - gammaln(t2 + 2) + gammaln(s2 + 1) + gammaln(t2 - s2 + 1))

    bf10 = np.exp(log_m_h1 - log_m_h0)

    p1, p2 = s1 / t1, s2 / t2

    return {
        "bf10": float(bf10),
        "bf01": float(1 / bf10) if bf10 > 0 else float("inf"),
        "proportion1": float(p1),
        "proportion2": float(p2),
        "difference": float(p1 - p2),
        "interpretation": _interpret_bf(bf10),
        "evidence_for": "H1 (proportions differ)" if bf10 > 1 else "H0 (proportions equal)",
    }


def _jzs_bf(t: float, n_eff: float, df: int, r: float = 0.707) -> float:
    """JZS Bayes Factor (Rouder et al., 2009). Shares the verified implementation
    in novel_dialectical (checked against pingouin to ~1e-9)."""
    from app.routers.novel_dialectical import jzs_bf10_t
    return jzs_bf10_t(float(t), float(n_eff), float(df), float(r))


def _bf_correlation(r: float, n: int) -> float:
    """Default correlation Bayes factor of Ly, Verhagen & Wagenmakers (2016),
    uniform prior on rho (kappa = 1)."""
    if abs(r) >= 1:
        return 1e300
    from app.routers.novel_dialectical import ly_bf10_r
    return ly_bf10_r(float(r), int(n), 1.0)


def _interpret_bf(bf: float) -> str:
    """Interpret Bayes Factor using Jeffreys' scale."""
    if bf > 100:
        return "Extreme evidence for H1"
    elif bf > 30:
        return "Very strong evidence for H1"
    elif bf > 10:
        return "Strong evidence for H1"
    elif bf > 3:
        return "Moderate evidence for H1"
    elif bf > 1:
        return "Anecdotal evidence for H1"
    elif bf == 1:
        return "No evidence either way"
    elif bf > 1/3:
        return "Anecdotal evidence for H0"
    elif bf > 1/10:
        return "Moderate evidence for H0"
    elif bf > 1/30:
        return "Strong evidence for H0"
    elif bf > 1/100:
        return "Very strong evidence for H0"
    else:
        return "Extreme evidence for H0"
