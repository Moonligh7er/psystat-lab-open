"""Reviewer 2 — a rule-based checklist that flags common reporting and design issues.

It is a heuristic screen, not a substitute for expert review: every rule below
is a simple threshold on the numbers you supply, and the A–F grade is just a
count of the flags.

Checks:
  1. Sensitivity: the smallest effect this design detects with 80% power
     (Lakens, 2022). Observed ("post-hoc") power is deliberately not used — it is
     a function of the p-value (Hoenig & Heisey, 2001). If a significant effect is
     smaller than that, the design is underpowered for it and significant
     estimates from such designs tend to be exaggerated (Gelman & Carlin, 2014).
  2. Multiple testing: family-wise error rate if the tests were independent.
  3. Effect size below the conventional "small" benchmark for its metric (Cohen, 1988).
  4. Very small samples.
  5. Confidence interval spanning zero, or very wide relative to the estimate.
  6. p just below .05 (weak evidence: roughly BF ≈ 2.5–3.4 under default priors;
     Benjamin et al., 2018) or just above it (should not be called a "trend").
  7. No assumption checks supplied.
  8. Conclusion changing when outliers are removed (if you supply that p-value).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.routers.what_would_it_take import (
    _min_detectable, power_anova, power_correlation, power_one_sample, power_regression, power_two_sample,
)

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    test_type: str  # "ttest", "paired", "anova", "correlation", "regression", "chi_square"
    # p_value, effect_size; optionally effect_size_type, ci_lower, ci_upper, assumptions,
    # outlier_sensitivity: {p_without_outliers}
    results: Dict[str, Any]
    sample_size: int  # total N
    num_tests: int = 1
    alpha: float = 0.05
    design_info: Optional[Dict[str, Any]] = None  # groups, predictors


class Critique(BaseModel):
    severity: str  # "critical", "warning", "note"
    category: str
    message: str
    recommendation: str


class ReviewResponse(BaseModel):
    critiques: List[Critique]
    overall_grade: str  # A–F from the count of flags
    summary: str


# Conventional "small" benchmarks (Cohen, 1988) by effect-size metric
SMALL = {"d": 0.2, "r": 0.1, "f": 0.1, "eta_squared": 0.01, "partial_eta_squared": 0.01,
         "r_squared": 0.02, "f_squared": 0.02, "w": 0.1, "cramers_v": 0.1}
DEFAULT_TYPE = {"ttest": "d", "paired": "d", "one_sample": "d", "correlation": "r",
                "anova": "eta_squared", "regression": "r_squared", "chi_square": "w"}
NAMES = {"d": "d", "r": "r", "f": "f", "eta_squared": "η²", "partial_eta_squared": "partial η²",
         "r_squared": "R²", "f_squared": "f²", "w": "w", "cramers_v": "Cramér's V"}


def _grade(critiques: List[Critique]) -> str:
    crit = sum(c.severity == "critical" for c in critiques)
    warn = sum(c.severity == "warning" for c in critiques)
    if crit >= 3: return "F"
    if crit == 2: return "D"
    if crit == 1: return "C" if warn <= 1 else "D"
    if warn >= 3: return "C"
    if warn >= 1: return "B"
    return "A"


def _sensitivity(tt: str, es_type: str, n: int, alpha: float, k: int, p: int):
    """(minimum detectable effect at 80% power, in the same metric as es_type) or None."""
    try:
        if tt == "ttest" and es_type == "d":
            return _min_detectable(lambda e, m: power_two_sample(e, m, alpha), n, 10.0)
        if tt in ("paired", "one_sample") and es_type == "d":
            return _min_detectable(lambda e, m: power_one_sample(e, m, alpha), n, 10.0)
        if tt == "correlation" and es_type == "r":
            return _min_detectable(lambda e, m: power_correlation(e, m, alpha), n, 0.999)
        if tt == "anova" and es_type in ("f", "eta_squared"):
            f = _min_detectable(lambda e, m: power_anova(e, m, alpha, k), n, 10.0)
            return None if f is None else (f if es_type == "f" else f * f / (1 + f * f))
        if tt == "regression" and es_type in ("r_squared", "f_squared"):
            r2 = _min_detectable(lambda e, m: power_regression(e, m, alpha, p), n, 0.999)
            return None if r2 is None else (r2 if es_type == "r_squared" else r2 / (1 - r2))
    except Exception:
        return None
    return None


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=ReviewResponse)
async def adversarial_review(req: ReviewRequest):
    tt = req.test_type.lower().replace("-", "_")
    tt = {"t_test": "ttest", "t": "ttest", "paired_t": "paired", "pearson": "correlation",
          "spearman": "correlation", "one_way_anova": "anova", "chi2": "chi_square"}.get(tt, tt)
    n, alpha = req.sample_size, req.alpha
    if n < 3:
        raise HTTPException(422, "sample_size must be at least 3.")
    if not 0 < alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    design = req.design_info or {}
    k = int(design.get("groups", 2))
    n_pred = int(design.get("predictors", 1))

    res = req.results
    p = res.get("p_value")
    es = res.get("effect_size")
    es_type = str(res.get("effect_size_type") or DEFAULT_TYPE.get(tt, "")).lower().replace("²", "_squared")
    ci_lo, ci_hi = res.get("ci_lower"), res.get("ci_upper")
    critiques: List[Critique] = []
    p_val = float(p) if p is not None else None
    sig = p_val is not None and p_val < alpha

    # 1. Sensitivity
    mdes = _sensitivity(tt, es_type, n, alpha, k, n_pred)
    if mdes is not None:
        name = NAMES.get(es_type, es_type)
        if es is not None and sig and abs(float(es)) < mdes:
            critiques.append(Critique(
                severity="warning", category="Sensitivity",
                message=(f"With N = {n}, this design detects {name} = {mdes:.3g} with 80% power, but the significant "
                         f"estimate is only {name} = {abs(float(es)):.3g}. Significant results from designs that are "
                         f"underpowered for the effect tend to overestimate it."),
                recommendation="Treat the estimate as likely inflated and replicate with a larger sample planned "
                               "for the smallest effect you care about."))
        else:
            critiques.append(Critique(
                severity="note", category="Sensitivity",
                message=f"With N = {n}, the smallest effect this design detects with 80% power is {name} = {mdes:.3g}.",
                recommendation="Compare this with the smallest effect that would matter in your field. Do not use "
                               "power computed from the observed effect; it adds nothing to the p-value."))

    # 2. Multiple testing
    if req.num_tests > 1:
        fwer = 1 - (1 - alpha) ** req.num_tests
        critiques.append(Critique(
            severity="critical" if req.num_tests >= 10 else "warning", category="Multiple Comparisons",
            message=(f"With {req.num_tests} tests at α = {alpha}, the chance of at least one false positive is "
                     f"{fwer:.2f} if the tests are independent (and up to {min(1.0, alpha * req.num_tests):.2f} "
                     f"in general)."),
            recommendation=f"Use Holm or Bonferroni (α/m = {alpha / req.num_tests:.4g}) to control the family-wise "
                           f"error rate, or Benjamini–Hochberg to control the false discovery rate."))

    # 3. Effect size below the small benchmark
    if es is not None and es_type in SMALL:
        if abs(float(es)) < SMALL[es_type]:
            name = NAMES[es_type]
            critiques.append(Critique(
                severity="warning", category="Effect Size",
                message=f"The effect ({name} = {abs(float(es)):.3g}) is below Cohen's conventional small benchmark "
                        f"({name} = {SMALL[es_type]}).",
                recommendation="Benchmarks are rough; judge practical importance in context, and report the effect "
                               "size with its confidence interval."))
    elif es is not None:
        critiques.append(Critique(
            severity="note", category="Effect Size",
            message="The effect-size metric was not recognised, so it was not compared with a benchmark.",
            recommendation="Pass effect_size_type (e.g. d, r, eta_squared, r_squared, w)."))

    # 4. Very small samples
    per_group = n // max(k, 1) if tt in ("ttest", "anova") else n
    if tt in ("ttest", "paired", "one_sample", "anova", "regression", "correlation") and per_group < 10:
        critiques.append(Critique(
            severity="warning", category="Sample Size",
            message=(f"Very small sample ({per_group} {'per group' if tt in ('ttest', 'anova') else 'in total'}). "
                     f"Estimates will be imprecise and assumption checks have little power to detect violations."),
            recommendation="Report confidence intervals, consider a robust or nonparametric check, and treat the "
                           "result as preliminary."))

    # 5. Confidence interval
    if ci_lo is not None and ci_hi is not None:
        lo, hi = float(ci_lo), float(ci_hi)
        if lo <= 0 <= hi and tt not in ("anova", "regression", "chi_square"):
            critiques.append(Critique(
                severity="warning", category="Precision",
                message=f"The confidence interval [{lo:.3g}, {hi:.3g}] includes zero, so the data are compatible with "
                        f"no effect and with effects in either direction.",
                recommendation="Do not claim a direction; a larger sample would narrow the interval."))
        if es is not None and float(es) != 0 and (hi - lo) / abs(float(es)) > 4:
            critiques.append(Critique(
                severity="note", category="Precision",
                message=f"The confidence interval is more than four times as wide as the estimate "
                        f"(width {hi - lo:.3g}, estimate {abs(float(es)):.3g}).",
                recommendation="The estimate is imprecise; report the interval prominently."))

    # 6. p near .05
    if p_val is not None and alpha == 0.05:
        if 0.01 < p_val < 0.05:
            critiques.append(Critique(
                severity="warning", category="Strength of Evidence",
                message=f"p = {p_val:.4g} is between .01 and .05. Such p-values are relatively weak evidence: "
                        f"p ≈ .05 corresponds to Bayes factors of only about 2.5–3.4 under common default priors.",
                recommendation="Report the effect size and interval, avoid strong claims, and consider a "
                               "Bayes factor or replication."))
        elif 0.05 <= p_val < 0.10:
            critiques.append(Critique(
                severity="note", category="P-Value",
                message=f"p = {p_val:.4g} is not significant. Avoid describing it as a 'trend' or 'marginally significant'.",
                recommendation="Report it as non-significant with the effect size and confidence interval."))

    # 7. Assumption checks
    if res.get("assumptions") is None and tt in ("ttest", "paired", "one_sample", "anova", "regression", "correlation"):
        critiques.append(Critique(
            severity="note", category="Assumption Checking",
            message="No assumption checks were supplied with this result.",
            recommendation="Check normality of residuals, equal variances where relevant, and outliers, and "
                           "report what you checked."))

    # 8. Outlier sensitivity
    outl = res.get("outlier_sensitivity") or {}
    p_trim = outl.get("p_without_outliers")
    if p_val is not None and p_trim is not None and (p_val < alpha) != (float(p_trim) < alpha):
        critiques.append(Critique(
            severity="critical", category="Outlier Sensitivity",
            message=f"The conclusion changes when outliers are removed (p = {p_val:.4g} → {float(p_trim):.4g}).",
            recommendation="The result depends on a few observations. Report both analyses, justify the choice, "
                           "and consider a robust method."))

    grade = _grade(critiques)
    counts = {s: sum(c.severity == s for c in critiques) for s in ("critical", "warning", "note")}
    if not any(counts.values()):
        summary = "No flags raised by this checklist. It only checks the numbers supplied, not the design or data."
    else:
        bits = [f"{v} {k}{'s' if v != 1 else ''}" for k, v in counts.items() if v]
        summary = (f"Checklist found {', '.join(bits)} (grade {grade}, a count of flags, not a quality score). "
                   f"It only checks the numbers supplied, not the design or data.")
    return ReviewResponse(critiques=critiques, overall_grade=grade, summary=summary)
