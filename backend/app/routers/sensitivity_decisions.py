"""Analytic Decision Sensitivity — re-runs a comparison under every combination of
common analytic choices (outlier handling × transformation × test) and reports
how often the conclusion holds. This is a small multiverse analysis
(Steegen et al., 2016).

Group comparisons:
  parametric    Welch t-test; Cohen's d (pooled SD)
  nonparametric Mann–Whitney U; rank-biserial correlation (positive = group 1 larger)
  robust        Yuen's test on 20% trimmed means (scipy ttest_ind with trim=0.2);
                robust d of Algina, Keselman & Penfield (2005)
Correlations: Pearson r, Spearman ρ, Kendall τ.

Transformations are applied to the pooled data with one shared shift, so the
two groups stay on the same scale. Note that log/sqrt/rank transforms change
the hypothesis being tested (e.g. a difference in means of logs).

The specifications reuse the same data, so the share that are significant is a
description of sensitivity, not independent evidence.
"""
from __future__ import annotations

import math
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats
from scipy.stats import mstats

router = APIRouter()


class SensitivityRequest(BaseModel):
    data_type: str  # "group_comparison" or "correlation"
    group1: Optional[List[float]] = None
    group2: Optional[List[float]] = None
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    decisions: Optional[Dict[str, List[str]]] = None  # override the default decision space
    alpha: float = 0.05


class SpecResult(BaseModel):
    outlier_method: str
    transform: str
    test_variant: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_label: Optional[str] = None
    significant: bool
    n_used: int
    error: Optional[str] = None


class SensitivityResponse(BaseModel):
    specifications: List[SpecResult]
    total_specifications: int
    significant_count: int
    robustness_score: float  # share of valid specifications agreeing with the majority conclusion
    verdict: str  # "robust" (≥ .9), "somewhat_robust" (≥ .7), "fragile"
    most_sensitive_decision: str
    summary: Dict[str, Any]


DEFAULT_DECISIONS = {
    "outlier_method": ["none", "trim_2sd", "winsorize_5pct"],
    "transform": ["raw", "log", "sqrt", "rank"],
    "test_variant": ["parametric", "nonparametric", "robust"],
}
GROUP_LABELS = {"parametric": "Cohen's d", "nonparametric": "rank-biserial r", "robust": "robust d (AKP)"}
CORR_LABELS = {"parametric": "Pearson r", "nonparametric": "Spearman ρ", "robust": "Kendall τ"}


def _f(v) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else round(v, 6)


# ── Outliers and transforms ──────────────────────────────────────────────

def _trim_2sd(a: np.ndarray) -> np.ndarray:
    m, s = a.mean(), a.std(ddof=1)
    return a if s == 0 else a[np.abs(a - m) <= 2 * s]


def _winsorize_5pct(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [5, 95])
    return np.clip(a, lo, hi)


def _outliers(a: np.ndarray, method: str) -> np.ndarray:
    if method == "trim_2sd":
        return _trim_2sd(a)
    if method == "winsorize_5pct":
        return _winsorize_5pct(a)
    if method == "none":
        return a.copy()
    raise ValueError(f"Unknown outlier method: {method}")


def _transform(parts: List[np.ndarray], how: str) -> List[np.ndarray]:
    """Transform several arrays with one shared shift / one joint ranking."""
    if how == "raw":
        return [p.copy() for p in parts]
    pooled = np.concatenate(parts)
    if how == "log":
        shift = 1 - pooled.min() if pooled.min() <= 0 else 0.0
        return [np.log(p + shift) for p in parts]
    if how == "sqrt":
        shift = -pooled.min() if pooled.min() < 0 else 0.0
        return [np.sqrt(p + shift) for p in parts]
    if how == "rank":
        ranks = sp_stats.rankdata(pooled)
        out, i = [], 0
        for p in parts:
            out.append(ranks[i:i + len(p)].astype(float))
            i += len(p)
        return out
    raise ValueError(f"Unknown transform: {how}")


# ── Effect sizes ─────────────────────────────────────────────────────────

def _cohens_d(a, b) -> float:
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def _akp_d(a, b, trim=0.2) -> float:
    """Algina, Keselman & Penfield (2005) robust d: 0.642 × trimmed-mean difference / pooled winsorized SD
    (0.642 rescales so it equals Cohen's d under normality for 20% trimming)."""
    wa = np.asarray(mstats.winsorize(a, limits=(trim, trim)))
    wb = np.asarray(mstats.winsorize(b, limits=(trim, trim)))
    n1, n2 = len(a), len(b)
    sw = math.sqrt(((n1 - 1) * wa.var(ddof=1) + (n2 - 1) * wb.var(ddof=1)) / (n1 + n2 - 2))
    diff = sp_stats.trim_mean(a, trim) - sp_stats.trim_mean(b, trim)
    return float(0.642 * diff / sw) if sw > 0 else float("nan")


# ── Specifications ───────────────────────────────────────────────────────

def _group_spec(g1, g2, outlier, transform, variant, alpha) -> SpecResult:
    base = dict(outlier_method=outlier, transform=transform, test_variant=variant,
                effect_label=GROUP_LABELS.get(variant))
    try:
        a, b = _outliers(g1, outlier), _outliers(g2, outlier)
        if len(a) < 3 or len(b) < 3:
            return SpecResult(**base, significant=False, n_used=len(a) + len(b),
                              error="Fewer than 3 observations left in a group")
        a, b = _transform([a, b], transform)
        if variant == "parametric":
            stat, p = sp_stats.ttest_ind(a, b, equal_var=False)
            es = _cohens_d(a, b)
        elif variant == "nonparametric":
            stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            es = 2 * float(stat) / (len(a) * len(b)) - 1
        elif variant == "robust":
            stat, p = sp_stats.ttest_ind(a, b, equal_var=False, trim=0.2)  # Yuen (1974)
            es = _akp_d(a, b)
        else:
            raise ValueError(f"Unknown test variant: {variant}")
        if not np.isfinite(p):
            raise ValueError("test undefined (no variation)")
        return SpecResult(**base, statistic=_f(stat), p_value=_f(p), effect_size=_f(es),
                          significant=float(p) < alpha, n_used=len(a) + len(b))
    except Exception as e:
        return SpecResult(**base, significant=False, n_used=0, error=str(e))


def _corr_spec(x, y, outlier, transform, variant, alpha) -> SpecResult:
    base = dict(outlier_method=outlier, transform=transform, test_variant=variant,
                effect_label=CORR_LABELS.get(variant))
    try:
        if outlier == "trim_2sd":
            keep = np.ones(len(x), bool)
            for arr in (x, y):
                s = arr.std(ddof=1)
                if s > 0:
                    keep &= np.abs(arr - arr.mean()) <= 2 * s
            xs, ys = x[keep], y[keep]
        elif outlier == "winsorize_5pct":
            xs, ys = _winsorize_5pct(x), _winsorize_5pct(y)
        elif outlier == "none":
            xs, ys = x.copy(), y.copy()
        else:
            raise ValueError(f"Unknown outlier method: {outlier}")
        if len(xs) < 4:
            return SpecResult(**base, significant=False, n_used=len(xs), error="Fewer than 4 pairs left")
        (xs,), (ys,) = _transform([xs], transform), _transform([ys], transform)
        if variant == "parametric":
            stat, p = sp_stats.pearsonr(xs, ys)
        elif variant == "nonparametric":
            stat, p = sp_stats.spearmanr(xs, ys)
        elif variant == "robust":
            stat, p = sp_stats.kendalltau(xs, ys)
        else:
            raise ValueError(f"Unknown test variant: {variant}")
        if not np.isfinite(p):
            raise ValueError("test undefined (no variation)")
        return SpecResult(**base, statistic=_f(stat), p_value=_f(p), effect_size=_f(stat),
                          significant=float(p) < alpha, n_used=len(xs))
    except Exception as e:
        return SpecResult(**base, significant=False, n_used=0, error=str(e))


def _most_sensitive(specs: List[SpecResult]) -> str:
    """The decision whose choices produce the most different significance rates."""
    best, best_var = "none", -1.0
    for dim in ("outlier_method", "transform", "test_variant"):
        groups: Dict[str, List[bool]] = {}
        for s in specs:
            if s.error is None:
                groups.setdefault(getattr(s, dim), []).append(s.significant)
        rates = [sum(v) / len(v) for v in groups.values() if v]
        if len(rates) >= 2 and float(np.var(rates)) > best_var:
            best, best_var = dim, float(np.var(rates))
    return best if best_var > 0 else "none"


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=SensitivityResponse)
async def analyze_sensitivity(req: SensitivityRequest):
    dtype = req.data_type.lower().strip()
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    dec = {**DEFAULT_DECISIONS, **(req.decisions or {})}
    for key, allowed in DEFAULT_DECISIONS.items():
        bad = [v for v in dec[key] if v not in allowed]
        if bad or not dec[key]:
            raise HTTPException(422, f"{key} must be a non-empty subset of {allowed}.")
    combos = list(product(dec["outlier_method"], dec["transform"], dec["test_variant"]))

    if dtype == "group_comparison":
        if not req.group1 or not req.group2 or len(req.group1) < 3 or len(req.group2) < 3:
            raise HTTPException(422, "group1 and group2 each need at least 3 observations.")
        g1, g2 = np.asarray(req.group1, float), np.asarray(req.group2, float)
        specs = [_group_spec(g1, g2, o, t, v, req.alpha) for o, t, v in combos]
        labels = GROUP_LABELS
    elif dtype == "correlation":
        if not req.x or not req.y or len(req.x) != len(req.y) or len(req.x) < 4:
            raise HTTPException(422, "x and y must have the same length, with at least 4 pairs.")
        x, y = np.asarray(req.x, float), np.asarray(req.y, float)
        specs = [_corr_spec(x, y, o, t, v, req.alpha) for o, t, v in combos]
        labels = CORR_LABELS
    else:
        raise HTTPException(422, "data_type must be group_comparison or correlation.")

    valid = [s for s in specs if s.error is None]
    sig = sum(s.significant for s in valid)
    if valid:
        majority = sig > len(valid) / 2
        robustness = sum(s.significant == majority for s in valid) / len(valid)
    else:
        robustness = 0.0
    verdict = "robust" if robustness >= 0.9 else "somewhat_robust" if robustness >= 0.7 else "fragile"

    by_metric = {}
    for variant, label in labels.items():
        vals = [s.effect_size for s in valid if s.test_variant == variant and s.effect_size is not None]
        if vals:
            by_metric[label] = {"median": round(float(np.median(vals)), 4),
                                "range": [round(min(vals), 4), round(max(vals), 4)], "n_specs": len(vals)}
    pvals = [s.p_value for s in valid if s.p_value is not None]
    summary = {
        "valid_specifications": len(valid),
        "errored_specifications": len(specs) - len(valid),
        "significant_percentage": round(100 * sig / len(valid), 1) if valid else 0.0,
        "effect_sizes_by_metric": by_metric,  # effect sizes are only comparable within a metric
        "median_p_value": round(float(np.median(pvals)), 4) if pvals else None,
        "note": ("Specifications reuse the same data, so these figures describe how sensitive the conclusion is "
                 "to analytic choices; they are not independent replications. Transformations change the "
                 "hypothesis being tested."),
    }
    return SensitivityResponse(
        specifications=specs, total_specifications=len(specs), significant_count=sig,
        robustness_score=round(robustness, 4), verdict=verdict,
        most_sensitive_decision=_most_sensitive(specs), summary=summary,
    )
