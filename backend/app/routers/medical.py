"""Medical Stats — survival analysis, clinical trial metrics, risk calculations."""

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator
import numpy as np
from scipy import stats as sp_stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import restricted_mean_survival_time
import pandas as pd

router = APIRouter()

Z95 = float(sp_stats.norm.ppf(0.975))


class KaplanMeierRequest(BaseModel):
    durations: list[float]
    events: list[int]  # 1 = event occurred, 0 = censored
    group_labels: list[str] | None = None  # optional grouping

    @model_validator(mode="after")
    def _check(self):
        n = len(self.durations)
        if len(self.events) != n:
            raise ValueError(f"{n} durations but {len(self.events)} event flags")
        if self.group_labels is not None and len(self.group_labels) != n:
            raise ValueError(f"{n} durations but {len(self.group_labels)} group labels")
        if any(e not in (0, 1) for e in self.events):
            raise ValueError("events must be 1 (event) or 0 (censored)")
        if any(d < 0 for d in self.durations):
            raise ValueError("durations must be non-negative")
        return self


class CoxRegressionRequest(BaseModel):
    durations: list[float]
    events: list[int]
    covariates: list[list[float]]  # rows = subjects, cols = covariates
    covariate_names: list[str] | None = None


class _TwoGroupCounts(BaseModel):
    def _check_counts(self, pairs):
        for events, total, label in pairs:
            if total <= 0:
                raise ValueError(f"{label}: group size must be > 0")
            if not 0 <= events <= total:
                raise ValueError(f"{label}: events must be between 0 and the group size")
        return self


class NNTRequest(_TwoGroupCounts):
    events_treatment: int
    n_treatment: int
    events_control: int
    n_control: int

    @model_validator(mode="after")
    def _v(self):
        return self._check_counts([(self.events_treatment, self.n_treatment, "treatment"),
                                   (self.events_control, self.n_control, "control")])


class RiskRatioRequest(_TwoGroupCounts):
    exposed_events: int
    exposed_total: int
    unexposed_events: int
    unexposed_total: int

    @model_validator(mode="after")
    def _v(self):
        return self._check_counts([(self.exposed_events, self.exposed_total, "exposed"),
                                   (self.unexposed_events, self.unexposed_total, "unexposed")])


class DiagnosticAccuracyRequest(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @model_validator(mode="after")
    def _check(self):
        if min(self.true_positives, self.false_positives, self.true_negatives, self.false_negatives) < 0:
            raise ValueError("Counts must be non-negative")
        return self


# ── Helpers ────────────────────────────────────────────────────────────

def _f(v):
    return float(v) if v is not None and math.isfinite(v) else None


def _wilson(k, n):
    """Wilson score 95% interval for a proportion k/n (None when n = 0)."""
    if n == 0:
        return None
    p = k / n
    den = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) / den
    return [centre - half, centre + half]


def _km_curve(kmf):
    return {
        "timeline": kmf.timeline.tolist(),
        "survival_probability": kmf.survival_function_.iloc[:, 0].tolist(),
        "confidence_lower": kmf.confidence_interval_.iloc[:, 0].tolist(),
        "confidence_upper": kmf.confidence_interval_.iloc[:, 1].tolist(),
    }


def _median(kmf):
    m = kmf.median_survival_time_
    return None if np.isinf(m) else float(m)


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/kaplan-meier")
def kaplan_meier(req: KaplanMeierRequest):
    """Kaplan-Meier survival analysis with optional group comparison (log-rank test).

    Confidence bands are lifelines' default (exponential Greenwood, log(-log) scale).
    The restricted mean survival time (RMST) is the area under the step curve up to
    the last observed time; it equals the mean only if the longest time is an event.
    """
    durations = np.array(req.durations, dtype=float)
    events = np.array(req.events, dtype=int)
    if len(durations) < 3:
        raise HTTPException(400, "Need at least 3 observations.")

    results = {}
    tau = float(durations.max())

    if req.group_labels is None:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events)
        results["survival_table"] = _km_curve(kmf)
        results["median_survival"] = _median(kmf)
        results["restricted_mean_survival"] = float(restricted_mean_survival_time(kmf, t=tau))
        results["rmst_horizon"] = tau
    else:
        labels = np.array(req.group_labels)
        groups = sorted(set(req.group_labels))
        if len(groups) < 2:
            raise HTTPException(400, "Group labels contain only one group; omit them for a single curve.")
        group_results = {}
        for group in groups:
            mask = labels == group
            kmf = KaplanMeierFitter()
            kmf.fit(durations[mask], event_observed=events[mask], label=group)
            group_results[group] = {
                **_km_curve(kmf),
                "median_survival": _median(kmf),
                "restricted_mean_survival": float(restricted_mean_survival_time(kmf, t=tau)),
                "n": int(mask.sum()),
                "events": int(events[mask].sum()),
            }
        results["groups"] = group_results
        results["rmst_horizon"] = tau

        if len(groups) == 2:
            a, b = labels == groups[0], labels == groups[1]
            lr = logrank_test(durations[a], durations[b], event_observed_A=events[a], event_observed_B=events[b])
            df = 1
        else:
            lr = multivariate_logrank_test(durations, labels, events)
            df = len(groups) - 1
        results["log_rank_test"] = {
            "test_statistic": float(lr.test_statistic),
            "df": df,
            "p_value": float(lr.p_value),
            "significant": bool(lr.p_value < 0.05),
        }

    results["n_total"] = int(len(durations))
    results["n_events"] = int(events.sum())
    results["n_censored"] = int((1 - events).sum())
    return results


@router.post("/cox-regression")
def cox_regression(req: CoxRegressionRequest):
    """Cox proportional hazards regression (Efron ties, lifelines)."""
    n = len(req.durations)
    if len(req.events) != n:
        raise HTTPException(400, f"{n} durations but {len(req.events)} event flags.")
    if any(e not in (0, 1) for e in req.events):
        raise HTTPException(400, "events must be 1 (event) or 0 (censored).")
    if n < 10:
        raise HTTPException(400, "Need at least 10 subjects for Cox regression.")
    covariates = np.array(req.covariates, dtype=float)
    if covariates.ndim != 2 or covariates.shape[0] != n:
        raise HTTPException(400, "covariates must have one row per subject.")
    n_covs = covariates.shape[1]
    cov_names = req.covariate_names or [f"x{i + 1}" for i in range(n_covs)]
    if len(cov_names) != n_covs or len(set(cov_names)) != n_covs:
        raise HTTPException(400, f"Need {n_covs} distinct covariate names.")
    if sum(req.events) == 0:
        raise HTTPException(400, "No events observed; hazard ratios cannot be estimated.")

    df = pd.DataFrame(covariates, columns=cov_names)
    df["duration"] = req.durations
    df["event"] = req.events

    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="duration", event_col="event")
    except Exception as e:  # lifelines raises on separation / singular designs
        raise HTTPException(400, f"Cox model could not be fitted: {e}")

    summary = cph.summary
    covariate_results = []
    for name in cov_names:
        row = summary.loc[name]
        covariate_results.append({
            "name": name,
            "coefficient": float(row["coef"]),
            "hazard_ratio": float(row["exp(coef)"]),
            "se": float(row["se(coef)"]),
            "z": float(row["z"]),
            "p_value": float(row["p"]),
            "ci_lower": float(row["exp(coef) lower 95%"]),
            "ci_upper": float(row["exp(coef) upper 95%"]),
            "significant": bool(row["p"] < 0.05),
        })

    return {
        "covariates": covariate_results,
        "concordance_index": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "aic": float(cph.AIC_partial_),
        "n_subjects": n,
        "n_events": int(sum(req.events)),
    }


@router.post("/nnt")
def number_needed_to_treat(req: NNTRequest):
    """NNT from the absolute risk reduction, with the Altman (1998) confidence interval.

    ARR = control risk − treatment risk (positive = treatment prevents events).
    The 95% CI for ARR is Wald. When it excludes 0 the NNT CI is its reciprocal;
    when it includes 0 the NNT CI runs from NNT(benefit) through infinity to NNT(harm).
    NNTs are rounded up to whole patients, as is conventional.
    """
    p_t = req.events_treatment / req.n_treatment
    p_c = req.events_control / req.n_control
    arr = p_c - p_t
    se_arr = math.sqrt(p_t * (1 - p_t) / req.n_treatment + p_c * (1 - p_c) / req.n_control)
    lo, hi = arr - Z95 * se_arr, arr + Z95 * se_arr

    nnt = 1 / arr if arr != 0 else None
    if lo > 0 or hi < 0:
        nnt_ci = sorted([1 / lo, 1 / hi])
        nnt_ci_text = (f"NNTB {math.ceil(abs(nnt_ci[0]))} to {math.ceil(abs(nnt_ci[1]))}" if arr > 0
                       else f"NNTH {math.ceil(abs(nnt_ci[1]))} to {math.ceil(abs(nnt_ci[0]))}")
    else:
        nnt_ci = None
        benefit = f"NNTB {math.ceil(1 / hi)}" if hi > 0 else None
        harm = f"NNTH {math.ceil(1 / abs(lo))}" if lo < 0 else None
        nnt_ci_text = f"{benefit or 'NNTB ∞'} to ∞ to {harm or 'NNTH ∞'} (the CI for ARR includes 0)"

    if nnt is None:
        interpretation = "No difference in event rates between groups"
    elif nnt > 0:
        interpretation = f"Treat {math.ceil(nnt)} patients to prevent one additional event (NNT to benefit)"
    else:
        interpretation = f"Treating {math.ceil(abs(nnt))} patients causes one additional event (NNT to harm)"

    return {
        "nnt": _f(nnt),
        "nnt_rounded": math.ceil(abs(nnt)) if nnt is not None else None,
        "nnt_ci_95": nnt_ci,
        "nnt_ci_text": nnt_ci_text,
        "absolute_risk_reduction": arr,
        "arr_ci_95": [lo, hi],
        "relative_risk_reduction": arr / p_c if p_c > 0 else None,
        "relative_risk": p_t / p_c if p_c > 0 else None,
        "event_rate_treatment": p_t,
        "event_rate_control": p_c,
        "interpretation": interpretation,
    }


@router.post("/risk-ratio")
def risk_ratio(req: RiskRatioRequest):
    """Risk ratio and odds ratio with log-scale Wald 95% CIs (Katz; Woolf).

    If any cell of the 2x2 table is zero, 0.5 is added to every cell (Haldane–Anscombe)
    for the ratio estimates and their CIs, and the response says so.
    """
    a, n1 = req.exposed_events, req.exposed_total
    c, n2 = req.unexposed_events, req.unexposed_total
    b, d = n1 - a, n2 - c
    p_exp, p_unexp = a / n1, c / n2

    corrected = min(a, b, c, d) == 0
    if corrected:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d
    n1_, n2_ = a_ + b_, c_ + d_

    rr = (a_ / n1_) / (c_ / n2_)
    se_log_rr = math.sqrt(1 / a_ - 1 / n1_ + 1 / c_ - 1 / n2_)
    odds_ratio = (a_ * d_) / (b_ * c_)
    se_log_or = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)

    rd = p_exp - p_unexp
    se_rd = math.sqrt(p_exp * (1 - p_exp) / n1 + p_unexp * (1 - p_unexp) / n2)

    return {
        "risk_ratio": rr,
        "rr_ci_95": [rr * math.exp(-Z95 * se_log_rr), rr * math.exp(Z95 * se_log_rr)],
        "odds_ratio": odds_ratio,
        "or_ci_95": [odds_ratio * math.exp(-Z95 * se_log_or), odds_ratio * math.exp(Z95 * se_log_or)],
        "risk_difference": rd,
        "rd_ci_95": [rd - Z95 * se_rd, rd + Z95 * se_rd],
        "exposed_risk": p_exp,
        "unexposed_risk": p_unexp,
        "zero_cell_correction": corrected,
        "note": "A zero cell was found, so 0.5 was added to every cell for the ratios and their CIs." if corrected else None,
    }


@router.post("/diagnostic-accuracy")
def diagnostic_accuracy(req: DiagnosticAccuracyRequest):
    """Sensitivity, specificity, predictive values and likelihood ratios with Wilson 95% CIs.

    Quantities with a zero denominator are returned as null rather than 0.
    PPV and NPV depend on the prevalence in this sample.
    """
    tp, fp, tn, fn = req.true_positives, req.false_positives, req.true_negatives, req.false_negatives
    total = tp + fp + tn + fn
    if total == 0:
        raise HTTPException(400, "All counts are zero.")

    def prop(k, n):
        return k / n if n > 0 else None

    sens, spec = prop(tp, tp + fn), prop(tn, tn + fp)
    lr_pos = sens / (1 - spec) if sens is not None and spec is not None and spec < 1 else None
    lr_neg = (1 - sens) / spec if sens is not None and spec is not None and spec > 0 else None

    return {
        "sensitivity": sens,
        "sensitivity_ci_95": _wilson(tp, tp + fn),
        "specificity": spec,
        "specificity_ci_95": _wilson(tn, tn + fp),
        "ppv": prop(tp, tp + fp),
        "ppv_ci_95": _wilson(tp, tp + fp),
        "npv": prop(tn, tn + fn),
        "npv_ci_95": _wilson(tn, tn + fn),
        "accuracy": (tp + tn) / total,
        "accuracy_ci_95": _wilson(tp + tn, total),
        "prevalence": (tp + fn) / total,
        "lr_positive": lr_pos,
        "lr_negative": lr_neg,
        "youdens_j": sens + spec - 1 if sens is not None and spec is not None else None,
        "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "note": "PPV and NPV reflect the prevalence in these data; they change with prevalence.",
    }
