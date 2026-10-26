"""Method Triangulation — run three standard methods on the same data and report
whether they agree on direction and on whether there is an effect.

  * Two groups: Student t-test (Cohen's d, exact CI), Mann–Whitney U
    (rank-biserial correlation), and a JZS Bayes factor (Rouder et al., 2009).
  * Correlation: Pearson (Fisher-z CI), Spearman, and a percentile bootstrap of r.
  * Simple regression: OLS slope, Huber robust regression (statsmodels RLM),
    and a percentile bootstrap of the slope.

Agreement among methods that share most assumptions is limited evidence of
robustness; disagreement is a prompt to look at outliers, skew and ties.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats

from app.routers.novel_dialectical import _d_ci, jzs_bf10_t

router = APIRouter()
N_BOOT = 5000


# ── Models ────────────────────────────────────────────────────────────────

class TriangulationRequest(BaseModel):
    data_type: str  # "group_comparison", "correlation", "regression"
    group1: Optional[List[float]] = None
    group2: Optional[List[float]] = None
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    alpha: float = 0.05


class MethodResult(BaseModel):
    method: str
    statistic: Optional[float] = None
    statistic_label: Optional[str] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_label: Optional[str] = None  # effect sizes are only comparable within a label
    ci: Optional[List[float]] = None  # 95%
    significant: bool = False  # p < alpha, CI excludes 0, or BF10 > 3 for the Bayes factor
    criterion: str = ""


class Convergence(BaseModel):
    direction_agree: bool
    significance_agree: bool
    magnitude_range: List[float]  # range over methods sharing the first method's effect label
    magnitude_label: str


class TriangulationResponse(BaseModel):
    methods: List[MethodResult]
    convergence: Convergence
    verdict: str  # "converge", "partial", "diverge"
    interpretation: str


# ── Helpers ───────────────────────────────────────────────────────────────

def _f(v) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def _bootstrap(x: np.ndarray, y: np.ndarray, func):
    """Percentile bootstrap: 95% CI and the two-sided percentile p-value 2·min(P(θ* ≤ 0), P(θ* ≥ 0))."""
    rng = np.random.default_rng(42)
    n = len(x)
    stats = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        stats[i] = func(x[idx], y[idx])
    stats = stats[np.isfinite(stats)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = max(2 * min(np.mean(stats <= 0), np.mean(stats >= 0)), 1 / len(stats))
    return [float(lo), float(hi)], float(min(1.0, p))


def _group(g1, g2, alpha) -> List[MethodResult]:
    n1, n2 = len(g1), len(g2)
    t, p = sp_stats.ttest_ind(g1, g2)
    n_eff, df = n1 * n2 / (n1 + n2), n1 + n2 - 2
    d = float(t) / math.sqrt(n_eff)
    out = [MethodResult(method="Student t-test", statistic=_f(t), statistic_label="t", p_value=_f(p),
                        effect_size=_f(d), effect_label="Cohen's d", ci=list(_d_ci(float(t), df, n_eff)),
                        significant=p < alpha, criterion=f"p < {alpha}")]
    u, p_mw = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    r_rb = 2 * float(u) / (n1 * n2) - 1  # positive when group 1 tends to be larger
    out.append(MethodResult(method="Mann–Whitney U", statistic=_f(u), statistic_label="U (group 1)", p_value=_f(p_mw),
                            effect_size=_f(r_rb), effect_label="rank-biserial r",
                            significant=p_mw < alpha, criterion=f"p < {alpha}"))
    bf = jzs_bf10_t(float(t), n_eff, df, 0.707)
    out.append(MethodResult(method="Bayesian t-test (JZS, r = 0.707)", statistic=_f(bf), statistic_label="BF₁₀",
                            effect_size=_f(d), effect_label="Cohen's d",
                            significant=bf > 3, criterion="BF₁₀ > 3"))
    return out


def _correlation(x, y, alpha) -> List[MethodResult]:
    n = len(x)
    r, p = sp_stats.pearsonr(x, y)
    z, se = math.atanh(min(max(float(r), -0.999999), 0.999999)), 1 / math.sqrt(n - 3)
    out = [MethodResult(method="Pearson correlation", statistic=_f(r), statistic_label="r", p_value=_f(p),
                        effect_size=_f(r), effect_label="r", ci=[math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)],
                        significant=p < alpha, criterion=f"p < {alpha}")]
    rho, p_s = sp_stats.spearmanr(x, y)
    out.append(MethodResult(method="Spearman correlation", statistic=_f(rho), statistic_label="ρ", p_value=_f(p_s),
                            effect_size=_f(rho), effect_label="Spearman ρ",
                            significant=p_s < alpha, criterion=f"p < {alpha}"))
    ci, p_b = _bootstrap(x, y, lambda a, b: np.corrcoef(a, b)[0, 1])
    out.append(MethodResult(method=f"Bootstrap correlation ({N_BOOT} resamples)", statistic=_f(r), statistic_label="r",
                            p_value=p_b, effect_size=_f(r), effect_label="r", ci=ci,
                            significant=not (ci[0] <= 0 <= ci[1]), criterion="95% CI excludes 0"))
    return out


def _regression(x, y, alpha) -> List[MethodResult]:
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = math.sqrt(float(resid @ resid) / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
    t = beta[1] / se if se > 0 else 0.0
    p = float(2 * sp_stats.t.sf(abs(t), n - 2))
    crit = sp_stats.t.ppf(0.975, n - 2)
    out = [MethodResult(method="OLS regression", statistic=_f(t), statistic_label="t", p_value=p,
                        effect_size=_f(beta[1]), effect_label="slope", ci=[beta[1] - crit * se, beta[1] + crit * se],
                        significant=p < alpha, criterion=f"p < {alpha}")]
    try:
        import statsmodels.api as sm
        fit = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        b, s = float(fit.params[1]), float(fit.bse[1])
        z = b / s if s > 0 else 0.0
        p_r = float(2 * sp_stats.norm.sf(abs(z)))  # RLM standard errors are asymptotic
        out.append(MethodResult(method="Robust regression (Huber M-estimator)", statistic=_f(z), statistic_label="z",
                                p_value=p_r, effect_size=b, effect_label="slope", ci=[b - 1.96 * s, b + 1.96 * s],
                                significant=p_r < alpha, criterion=f"p < {alpha}"))
    except Exception as e:  # statsmodels missing or fit failure
        out.append(MethodResult(method=f"Robust regression (unavailable: {e})", criterion="—"))

    def slope(a, b):
        return float(np.polyfit(a, b, 1)[0]) if np.std(a) > 0 else float("nan")
    ci, p_b = _bootstrap(x, y, slope)
    out.append(MethodResult(method=f"Bootstrap regression ({N_BOOT} resamples)", statistic=_f(beta[1]),
                            statistic_label="slope", p_value=p_b, effect_size=_f(beta[1]), effect_label="slope",
                            ci=ci, significant=not (ci[0] <= 0 <= ci[1]), criterion="95% CI excludes 0"))
    return out


def _convergence(methods: List[MethodResult]):
    valid = [m for m in methods if m.effect_size is not None]
    signs = {np.sign(m.effect_size) for m in valid if m.effect_size != 0}
    direction = len(signs) <= 1
    significance = len({m.significant for m in methods if m.criterion != "—"}) == 1
    label = valid[0].effect_label if valid else ""
    same = [m.effect_size for m in valid if m.effect_label == label]
    conv = Convergence(direction_agree=direction, significance_agree=significance,
                       magnitude_range=[min(same), max(same)] if same else [0.0, 0.0], magnitude_label=label or "")
    verdict = "converge" if direction and significance else "partial" if direction or significance else "diverge"
    return conv, verdict


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=TriangulationResponse)
async def triangulate(req: TriangulationRequest):
    dt = req.data_type.lower().replace("-", "_").replace(" ", "_")
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    if dt == "group_comparison":
        if not req.group1 or not req.group2 or len(req.group1) < 3 or len(req.group2) < 3:
            raise HTTPException(422, "group1 and group2 each need at least 3 observations.")
        g1, g2 = np.asarray(req.group1, float), np.asarray(req.group2, float)
        if np.var(g1, ddof=1) + np.var(g2, ddof=1) == 0:
            raise HTTPException(422, "Both groups have zero variance.")
        methods = _group(g1, g2, req.alpha)
    elif dt in ("correlation", "regression"):
        if not req.x or not req.y or len(req.x) != len(req.y) or len(req.x) < 5:
            raise HTTPException(422, "x and y must have the same length, with at least 5 pairs.")
        x, y = np.asarray(req.x, float), np.asarray(req.y, float)
        if np.std(x) == 0 or np.std(y) == 0:
            raise HTTPException(422, "x and y must both vary.")
        methods = _correlation(x, y, req.alpha) if dt == "correlation" else _regression(x, y, req.alpha)
    else:
        raise HTTPException(422, "data_type must be group_comparison, correlation or regression.")

    conv, verdict = _convergence(methods)
    if verdict == "converge":
        text = ("All three methods agree on the direction and on whether there is an effect. Because they share some "
                "assumptions, this shows the conclusion does not hinge on the choice among these methods; it is not "
                "proof the result is correct.")
    elif verdict == "partial":
        what = "whether there is an effect" if conv.direction_agree else "the direction of the effect"
        text = (f"The methods agree on {'the direction' if conv.direction_agree else 'whether there is an effect'} "
                f"but not on {what}. The conclusion depends on the method; check outliers, skew and ties.")
    else:
        text = ("The methods disagree on both direction and whether there is an effect. Do not draw a conclusion "
                "before investigating outliers, skew and assumption violations.")
    return TriangulationResponse(methods=methods, convergence=conv, verdict=verdict, interpretation=text)
