"""Equivalence Testing (TOST) — two one-sided tests (Schuirmann, 1987; Lakens, 2017).

Tests whether an effect is smaller than the smallest effect you care about
(the equivalence bounds), alongside the usual test against zero.

  * Two independent groups: Welch t-tests on the raw mean difference.
    Bounds may be given in raw units or as Cohen's d; a d bound is converted
    to raw units with the observed standardiser sqrt((s1² + s2²) / 2), which
    treats the bound as fixed (as in Lakens, 2017 / TOSTER).
  * Paired / one-sample: t-tests on the mean (difference); d bounds use the SD
    of the differences.
  * Correlation: z-tests on Fisher's z (Goertzen & Cribbie, 2010).
  * Two proportions: Wald z-tests on the difference in proportions.

Equivalence is declared when both one-sided tests reject at alpha, which is the
same as the 100(1 − 2α)% confidence interval lying entirely inside the bounds.
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats

router = APIRouter()


# ── Models ───────────────────────────────────────────────────────────────

class EquivalenceRequest(BaseModel):
    test_type: str  # "ttest" (two independent groups), "paired", "one_sample", "correlation", "proportion"
    group1: Optional[List[float]] = None
    group2: Optional[List[float]] = None
    mu0: float = 0.0  # one-sample reference value
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    successes1: Optional[int] = None
    n1: Optional[int] = None
    successes2: Optional[int] = None
    n2: Optional[int] = None
    equivalence_bound: Optional[float] = None  # symmetric bound ±value; default depends on test
    # t-tests only: whether equivalence_bound is a raw difference or Cohen's d
    bound_type: Literal["d", "raw"] = "d"
    alpha: float = 0.05


class EquivalenceResponse(BaseModel):
    method: str
    tost_p1: float  # H0: effect ≤ lower bound
    tost_p2: float  # H0: effect ≥ upper bound
    tost_p: float  # max(p1, p2)
    tost_significant: bool
    nhst_p: float  # two-sided test against zero
    nhst_significant: bool
    effect_label: str  # what observed_effect, the bounds and the CI are measured in
    equivalence_bounds: List[float]  # [lower, upper], in effect_label units
    observed_effect: float
    standardized_effect: Optional[float] = None  # Cohen's d for t-tests
    standardized_bounds: Optional[List[float]] = None  # bounds in d units for t-tests
    confidence_interval: List[float]  # 100(1 − 2α)% CI in effect_label units
    ci_level: float
    classification: str  # "equivalence", "trivial", "effect", "inconclusive"
    classification_label: str
    interpretation: str


DEFAULT_BOUNDS = {"ttest": 0.5, "paired": 0.5, "one_sample": 0.5, "correlation": 0.1, "proportion": 0.1}


# ── Helpers ──────────────────────────────────────────────────────────────

def _t_tost(diff: float, se: float, df: float, lo: float, hi: float, alpha: float):
    p1 = float(sp_stats.t.sf((diff - lo) / se, df))
    p2 = float(sp_stats.t.cdf((diff - hi) / se, df))
    nhst = float(2 * sp_stats.t.sf(abs(diff) / se, df))
    crit = float(sp_stats.t.ppf(1 - alpha, df))
    return p1, p2, nhst, [diff - crit * se, diff + crit * se]


def _z_tost(est: float, se: float, lo: float, hi: float, alpha: float):
    p1 = float(sp_stats.norm.sf((est - lo) / se))
    p2 = float(sp_stats.norm.cdf((est - hi) / se))
    crit = float(sp_stats.norm.ppf(1 - alpha))
    return p1, p2, [est - crit * se, est + crit * se]


def _tost_two_groups(g1, g2, bound, bound_type, alpha):
    a1, a2 = np.asarray(g1, float), np.asarray(g2, float)
    if len(a1) < 2 or len(a2) < 2:
        raise HTTPException(422, "Each group needs at least 2 observations.")
    n1, n2 = len(a1), len(a2)
    v1, v2 = float(np.var(a1, ddof=1)), float(np.var(a2, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        raise HTTPException(422, "Both groups have zero variance.")
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    sd_std = math.sqrt((v1 + v2) / 2)  # standardiser for Welch (Lakens, 2017)
    diff = float(np.mean(a1) - np.mean(a2))
    raw = bound * sd_std if bound_type == "d" else bound
    p1, p2, nhst, ci = _t_tost(diff, se, df, -raw, raw, alpha)
    return dict(method="Welch two one-sided t-tests (independent groups)", p1=p1, p2=p2, nhst=nhst,
                label="mean difference (group 1 − group 2)", bounds=[-raw, raw], est=diff, ci=ci,
                d=diff / sd_std if sd_std > 0 else None,
                d_bounds=[-raw / sd_std, raw / sd_std] if sd_std > 0 else None)


def _tost_one_sample(values, mu0, bound, bound_type, alpha, method, label):
    a = np.asarray(values, float)
    if len(a) < 2:
        raise HTTPException(422, "Need at least 2 observations.")
    sd = float(np.std(a, ddof=1))
    if sd == 0:
        raise HTTPException(422, "The values have zero variance.")
    n = len(a)
    se = sd / math.sqrt(n)
    diff = float(np.mean(a) - mu0)
    raw = bound * sd if bound_type == "d" else bound
    p1, p2, nhst, ci = _t_tost(diff, se, n - 1, -raw, raw, alpha)
    return dict(method=method, p1=p1, p2=p2, nhst=nhst, label=label, bounds=[-raw, raw], est=diff, ci=ci,
                d=diff / sd, d_bounds=[-raw / sd, raw / sd])


def _tost_correlation(x, y, bound, alpha):
    ax, ay = np.asarray(x, float), np.asarray(y, float)
    if len(ax) != len(ay):
        raise HTTPException(422, "x and y must have the same length.")
    if len(ax) < 4:
        raise HTTPException(422, "Need at least 4 pairs.")
    if not 0 < bound < 1:
        raise HTTPException(422, "A correlation bound must be between 0 and 1.")
    n = len(ax)
    r, nhst = sp_stats.pearsonr(ax, ay)
    r = float(r)
    if abs(r) >= 1:
        raise HTTPException(422, "The correlation is exactly ±1; Fisher's z is undefined.")
    se = 1 / math.sqrt(n - 3)
    p1, p2, ci_z = _z_tost(math.atanh(r), se, math.atanh(-bound), math.atanh(bound), alpha)
    return dict(method="Two one-sided z-tests on Fisher's z (correlation)", p1=p1, p2=p2, nhst=float(nhst),
                label="Pearson r", bounds=[-bound, bound], est=r, ci=[math.tanh(c) for c in ci_z],
                d=None, d_bounds=None)


def _tost_proportions(s1, n1, s2, n2, bound, alpha):
    if n1 < 1 or n2 < 1 or not (0 <= s1 <= n1) or not (0 <= s2 <= n2):
        raise HTTPException(422, "Counts must satisfy 0 ≤ successes ≤ n, with n ≥ 1.")
    if not 0 < bound < 1:
        raise HTTPException(422, "A proportion-difference bound must be between 0 and 1.")
    p1h, p2h = s1 / n1, s2 / n2
    se = math.sqrt(p1h * (1 - p1h) / n1 + p2h * (1 - p2h) / n2)
    if se == 0:
        raise HTTPException(422, "Every observation in both groups has the same outcome, so the Wald "
                                 "standard error is zero and the test cannot be computed.")
    diff = p1h - p2h
    p1, p2, ci = _z_tost(diff, se, -bound, bound, alpha)
    pooled = (s1 + s2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    nhst = float(2 * sp_stats.norm.sf(abs(diff) / se0)) if se0 > 0 else 1.0
    return dict(method="Two one-sided Wald z-tests (difference in proportions)", p1=p1, p2=p2, nhst=nhst,
                label="difference in proportions (group 1 − group 2)", bounds=[-bound, bound], est=diff,
                ci=ci, d=None, d_bounds=None)


def _classify(tost_sig, nhst_sig, ci, bounds):
    lo, hi = bounds
    ci_txt = f"[{ci[0]:.3g}, {ci[1]:.3g}]"
    b_txt = f"[{lo:.3g}, {hi:.3g}]"
    if tost_sig and not nhst_sig:
        return ("equivalence", "Equivalent",
                f"The confidence interval {ci_txt} lies entirely inside the bounds {b_txt}, and the effect is not "
                f"significantly different from zero. The effect is smaller than the smallest effect of interest.")
    if tost_sig and nhst_sig:
        return ("trivial", "Significant but smaller than the bounds",
                f"The effect differs significantly from zero, but the confidence interval {ci_txt} lies inside the "
                f"bounds {b_txt}: it is smaller than the smallest effect of interest.")
    if nhst_sig:
        outside = ci[0] > hi or ci[1] < lo
        return ("effect", "Different from zero; equivalence not shown",
                f"The effect differs significantly from zero and equivalence was not shown. "
                + (f"The whole confidence interval {ci_txt} lies beyond the bounds {b_txt}, so the effect is "
                   f"larger than the smallest effect of interest." if outside else
                   f"The confidence interval {ci_txt} overlaps the bounds {b_txt}, so the effect may or may not "
                   f"exceed the smallest effect of interest."))
    return ("inconclusive", "Inconclusive",
            f"The effect is neither significantly different from zero nor shown to be inside the bounds {b_txt} "
            f"(confidence interval {ci_txt}). The data cannot distinguish these possibilities; more data are needed.")


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=EquivalenceResponse)
async def analyze_equivalence(req: EquivalenceRequest):
    test = req.test_type.lower().strip()
    if test not in DEFAULT_BOUNDS:
        raise HTTPException(422, "test_type must be ttest, paired, one_sample, correlation or proportion.")
    if not 0 < req.alpha < 0.5:
        raise HTTPException(422, "alpha must be between 0 and 0.5.")
    bound = req.equivalence_bound if req.equivalence_bound is not None else DEFAULT_BOUNDS[test]
    if not bound > 0:
        raise HTTPException(422, "equivalence_bound must be positive.")

    if test == "ttest":
        if not req.group1 or not req.group2:
            raise HTTPException(422, "group1 and group2 are required.")
        r = _tost_two_groups(req.group1, req.group2, bound, req.bound_type, req.alpha)
    elif test == "paired":
        if not req.group1 or not req.group2 or len(req.group1) != len(req.group2):
            raise HTTPException(422, "Paired data need group1 and group2 of equal length.")
        diffs = np.asarray(req.group1, float) - np.asarray(req.group2, float)
        r = _tost_one_sample(diffs, 0.0, bound, req.bound_type, req.alpha,
                             "Two one-sided paired t-tests", "mean of the paired differences (1 − 2)")
    elif test == "one_sample":
        if not req.group1:
            raise HTTPException(422, "group1 is required.")
        r = _tost_one_sample(req.group1, req.mu0, bound, req.bound_type, req.alpha,
                             "Two one-sided one-sample t-tests", f"mean − {req.mu0:g}")
    elif test == "correlation":
        if not req.x or not req.y:
            raise HTTPException(422, "x and y are required.")
        r = _tost_correlation(req.x, req.y, bound, req.alpha)
    else:
        if None in (req.successes1, req.n1, req.successes2, req.n2):
            raise HTTPException(422, "successes1, n1, successes2 and n2 are required.")
        r = _tost_proportions(req.successes1, req.n1, req.successes2, req.n2, bound, req.alpha)

    tost_p = max(r["p1"], r["p2"])
    tost_sig, nhst_sig = tost_p < req.alpha, r["nhst"] < req.alpha
    code, label, text = _classify(tost_sig, nhst_sig, r["ci"], r["bounds"])
    return EquivalenceResponse(
        method=r["method"], tost_p1=r["p1"], tost_p2=r["p2"], tost_p=tost_p, tost_significant=tost_sig,
        nhst_p=r["nhst"], nhst_significant=nhst_sig, effect_label=r["label"],
        equivalence_bounds=[round(b, 6) for b in r["bounds"]], observed_effect=round(r["est"], 6),
        standardized_effect=round(r["d"], 6) if r["d"] is not None else None,
        standardized_bounds=[round(b, 6) for b in r["d_bounds"]] if r["d_bounds"] else None,
        confidence_interval=[round(c, 6) for c in r["ci"]], ci_level=round(1 - 2 * req.alpha, 4),
        classification=code, classification_label=label, interpretation=text,
    )
