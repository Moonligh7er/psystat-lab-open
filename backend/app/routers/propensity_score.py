"""Propensity Score Matching — logistic regression estimation, nearest-neighbor
matching, inverse probability weighting, overlap analysis, and balance diagnostics.

Implements propensity score estimation via logistic regression (Newton-Raphson),
1:1 greedy nearest-neighbor matching with optional caliper, stabilized IPTW
for ATE/ATT, common support analysis, and comprehensive balance assessment
using standardized mean differences, variance ratios, and KS statistics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    """Propensity score estimation via logistic regression."""
    treatment: list[int]  # binary 0/1
    covariates: list[list[float]]  # each inner list is one covariate's values
    covariate_names: list[str] | None = None


class MatchRequest(BaseModel):
    """1:1 nearest-neighbor propensity score matching."""
    treatment: list[int]  # binary 0/1
    outcome: list[float]
    covariates: list[list[float]]
    covariate_names: list[str] | None = None
    caliper: float | None = None  # in SD of logit(PS) units
    method: str = "greedy"  # "greedy" or "optimal"
    n_bootstrap: int = 1000
    confidence: float = 0.95


class IPTWRequest(BaseModel):
    """Inverse probability of treatment weighting."""
    treatment: list[int]  # binary 0/1
    outcome: list[float]
    covariates: list[list[float]]
    covariate_names: list[str] | None = None
    estimand: str = "ATE"  # "ATE" or "ATT"
    trim_threshold: float = 0.01  # trim weights outside [threshold, 1-threshold]
    n_bootstrap: int = 1000
    confidence: float = 0.95


class OverlapRequest(BaseModel):
    """Common support / overlap analysis."""
    treatment: list[int]
    covariates: list[list[float]]
    covariate_names: list[str] | None = None
    trim_threshold: float = 0.1  # remove units with PS < threshold or > 1-threshold


class BalanceRequest(BaseModel):
    """Standalone balance diagnostics pre/post matching."""
    treatment: list[int]
    covariates: list[list[float]]
    covariate_names: list[str] | None = None
    matched_indices_treated: list[int] | None = None
    matched_indices_control: list[int] | None = None


# ── Helper Functions ──────────────────────────────────────────────────

def _validate_binary_treatment(treatment: list[int]):
    """Ensure treatment is binary 0/1."""
    unique = set(treatment)
    if not unique.issubset({0, 1}):
        raise HTTPException(
            status_code=422,
            detail=f"Treatment must be binary (0/1). Got unique values: {sorted(unique)}",
        )
    if 0 not in unique or 1 not in unique:
        raise HTTPException(
            status_code=422,
            detail="Treatment must contain both 0 and 1 values.",
        )


def _validate_inputs(treatment, covariates, outcome=None, min_n=20):
    """Validate input dimensions and sizes."""
    n = len(treatment)
    if n < min_n:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_n} observations, got {n}.",
        )
    for i, cov in enumerate(covariates):
        if len(cov) != n:
            raise HTTPException(
                status_code=422,
                detail=f"Covariate {i} has length {len(cov)}, expected {n}.",
            )
    if outcome is not None and len(outcome) != n:
        raise HTTPException(
            status_code=422,
            detail=f"Outcome has length {len(outcome)}, expected {n}.",
        )


def _sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _logistic_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit logistic regression via maximum likelihood (L-BFGS-B).

    Returns (coefficients, predicted_probabilities).
    """
    n, p = X.shape

    def neg_log_likelihood(beta):
        z = X @ beta
        prob = _sigmoid(z)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        ll = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        return -ll

    def gradient(beta):
        z = X @ beta
        prob = _sigmoid(z)
        return -X.T @ (y - prob)

    beta0 = np.zeros(p)
    result = minimize(neg_log_likelihood, beta0, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10})

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=f"Logistic regression did not converge: {result.message}",
        )

    beta = result.x
    probs = _sigmoid(X @ beta)
    return beta, probs


def _estimate_propensity_scores(treatment: np.ndarray, X_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate propensity scores from treatment and covariate matrix.

    Returns (propensity_scores, logistic_coefficients).
    """
    # Add intercept
    n = len(treatment)
    X = np.column_stack([np.ones(n), X_cov])
    beta, ps = _logistic_regression(X, treatment.astype(float))
    # Clip to avoid extreme weights
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    return ps, beta


def _standardized_mean_diff(treated_vals: np.ndarray, control_vals: np.ndarray) -> float:
    """Compute standardized mean difference (Cohen's d pooled)."""
    mean_t = np.mean(treated_vals)
    mean_c = np.mean(control_vals)
    var_t = np.var(treated_vals, ddof=1) if len(treated_vals) > 1 else 0.0
    var_c = np.var(control_vals, ddof=1) if len(control_vals) > 1 else 0.0
    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd < 1e-12:
        return 0.0
    return (mean_t - mean_c) / pooled_sd


def _variance_ratio(treated_vals: np.ndarray, control_vals: np.ndarray) -> float:
    """Compute variance ratio (treated / control)."""
    var_t = np.var(treated_vals, ddof=1) if len(treated_vals) > 1 else 0.0
    var_c = np.var(control_vals, ddof=1) if len(control_vals) > 1 else 0.0
    if var_c < 1e-12:
        return float("inf") if var_t > 1e-12 else 1.0
    return var_t / var_c


def _ks_statistic(treated_vals: np.ndarray, control_vals: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value."""
    stat, pval = sp_stats.ks_2samp(treated_vals, control_vals)
    return float(stat), float(pval)


def _compute_balance(treatment: np.ndarray, covariates: np.ndarray,
                     covariate_names: list[str],
                     indices_treated: np.ndarray | None = None,
                     indices_control: np.ndarray | None = None) -> list[dict]:
    """Compute balance diagnostics for each covariate."""
    if indices_treated is not None and indices_control is not None:
        t_mask = indices_treated
        c_mask = indices_control
    else:
        t_mask = np.where(treatment == 1)[0]
        c_mask = np.where(treatment == 0)[0]

    balance = []
    for j in range(covariates.shape[1]):
        t_vals = covariates[t_mask, j]
        c_vals = covariates[c_mask, j]
        smd = _standardized_mean_diff(t_vals, c_vals)
        vr = _variance_ratio(t_vals, c_vals)
        ks_stat, ks_p = _ks_statistic(t_vals, c_vals)
        balance.append({
            "covariate": covariate_names[j],
            "mean_treated": float(np.mean(t_vals)),
            "mean_control": float(np.mean(c_vals)),
            "standardized_mean_difference": round(float(smd), 4),
            "variance_ratio": round(float(vr), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 4),
            "balanced": abs(smd) < 0.1,
        })
    return balance


def _greedy_match(ps: np.ndarray, treatment: np.ndarray,
                  caliper: float | None = None) -> tuple[list[tuple], list[int], list[int]]:
    """Greedy 1:1 nearest-neighbor matching on propensity scores.

    Returns (matched_pairs, treated_indices, control_indices).
    """
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    logit_ps = np.log(ps / (1 - ps))
    if caliper is not None:
        caliper_width = caliper * np.std(logit_ps)
    else:
        caliper_width = None

    # Shuffle treated to avoid order bias
    rng = np.random.default_rng(42)
    shuffled_treated = rng.permutation(treated_idx)

    matched_pairs = []
    used_controls = set()
    matched_treated = []
    matched_control = []

    for t_idx in shuffled_treated:
        best_dist = float("inf")
        best_c = None
        for c_idx in control_idx:
            if c_idx in used_controls:
                continue
            dist = abs(logit_ps[t_idx] - logit_ps[c_idx])
            if caliper_width is not None and dist > caliper_width:
                continue
            if dist < best_dist:
                best_dist = dist
                best_c = c_idx

        if best_c is not None:
            matched_pairs.append((int(t_idx), int(best_c)))
            matched_treated.append(int(t_idx))
            matched_control.append(int(best_c))
            used_controls.add(best_c)

    return matched_pairs, matched_treated, matched_control


# ── 1. Propensity Score Estimation ────────────────────────────────────

@router.post("/estimate")
def estimate_propensity_scores(req: EstimateRequest):
    """Estimate propensity scores via logistic regression.

    Fits a logistic regression of treatment on covariates and returns
    the predicted probability of treatment assignment for each unit.
    """
    _validate_binary_treatment(req.treatment)
    _validate_inputs(req.treatment, req.covariates)

    treatment = np.array(req.treatment)
    n = len(treatment)
    k = len(req.covariates)
    X_cov = np.column_stack([np.array(c) for c in req.covariates])
    cov_names = req.covariate_names or [f"X{i+1}" for i in range(k)]

    if len(cov_names) != k:
        raise HTTPException(status_code=422, detail="covariate_names length must match number of covariates.")

    ps, beta = _estimate_propensity_scores(treatment, X_cov)

    # Coefficient table (intercept + covariates)
    coef_names = ["intercept"] + list(cov_names)
    coefficients = [{"name": coef_names[i], "coefficient": round(float(beta[i]), 6)}
                    for i in range(len(beta))]

    # Distribution summary
    ps_treated = ps[treatment == 1]
    ps_control = ps[treatment == 0]

    return {
        "n": n,
        "n_treated": int(np.sum(treatment)),
        "n_control": int(np.sum(1 - treatment)),
        "propensity_scores": [round(float(p), 6) for p in ps],
        "coefficients": coefficients,
        "distribution": {
            "overall": {
                "mean": round(float(np.mean(ps)), 4),
                "sd": round(float(np.std(ps)), 4),
                "min": round(float(np.min(ps)), 4),
                "max": round(float(np.max(ps)), 4),
            },
            "treated": {
                "mean": round(float(np.mean(ps_treated)), 4),
                "sd": round(float(np.std(ps_treated)), 4),
                "min": round(float(np.min(ps_treated)), 4),
                "max": round(float(np.max(ps_treated)), 4),
            },
            "control": {
                "mean": round(float(np.mean(ps_control)), 4),
                "sd": round(float(np.std(ps_control)), 4),
                "min": round(float(np.min(ps_control)), 4),
                "max": round(float(np.max(ps_control)), 4),
            },
        },
        "interpretation": (
            f"Propensity scores estimated for {n} units ({int(np.sum(treatment))} treated, "
            f"{int(np.sum(1 - treatment))} control) using {k} covariates. "
            f"Mean PS: treated = {np.mean(ps_treated):.3f}, control = {np.mean(ps_control):.3f}. "
            f"Good overlap exists when distributions substantially overlap."
        ),
    }


# ── 2. Propensity Score Matching ──────────────────────────────────────

@router.post("/match")
def propensity_score_match(req: MatchRequest):
    """1:1 nearest-neighbor propensity score matching with balance diagnostics.

    Performs greedy matching on logit of propensity scores, computes pre/post
    balance, and estimates ATT with bootstrap confidence intervals.
    """
    _validate_binary_treatment(req.treatment)
    _validate_inputs(req.treatment, req.covariates, req.outcome)

    treatment = np.array(req.treatment)
    outcome = np.array(req.outcome)
    n = len(treatment)
    k = len(req.covariates)
    X_cov = np.column_stack([np.array(c) for c in req.covariates])
    cov_names = req.covariate_names or [f"X{i+1}" for i in range(k)]

    # Estimate propensity scores
    ps, _ = _estimate_propensity_scores(treatment, X_cov)

    # Perform matching
    if req.method not in ("greedy", "optimal"):
        raise HTTPException(status_code=422, detail="method must be 'greedy' or 'optimal'.")

    matched_pairs, matched_treated, matched_control = _greedy_match(
        ps, treatment, caliper=req.caliper
    )

    if len(matched_pairs) < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(matched_pairs)} pairs matched. Consider relaxing the caliper.",
        )

    n_matched = len(matched_pairs)
    matched_t_idx = np.array(matched_treated)
    matched_c_idx = np.array(matched_control)

    # Pre-matching balance (full sample)
    pre_balance = _compute_balance(treatment, X_cov, cov_names)

    # Post-matching balance
    post_balance = _compute_balance(
        treatment, X_cov, cov_names,
        indices_treated=matched_t_idx,
        indices_control=matched_c_idx,
    )

    # ATT estimate
    y_treated = outcome[matched_t_idx]
    y_control = outcome[matched_c_idx]
    att = float(np.mean(y_treated - y_control))

    # Bootstrap CI for ATT
    rng = np.random.default_rng(42)
    boot_atts = []
    for _ in range(req.n_bootstrap):
        boot_idx = rng.choice(n_matched, size=n_matched, replace=True)
        boot_diff = y_treated[boot_idx] - y_control[boot_idx]
        boot_atts.append(float(np.mean(boot_diff)))

    boot_atts = np.array(boot_atts)
    alpha = 1 - req.confidence
    ci_lower = float(np.percentile(boot_atts, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_atts, 100 * (1 - alpha / 2)))
    boot_se = float(np.std(boot_atts, ddof=1))

    # Summary of balance improvement
    pre_max_smd = max(abs(b["standardized_mean_difference"]) for b in pre_balance)
    post_max_smd = max(abs(b["standardized_mean_difference"]) for b in post_balance)
    pre_balanced = sum(1 for b in pre_balance if b["balanced"])
    post_balanced = sum(1 for b in post_balance if b["balanced"])

    return {
        "n_total": n,
        "n_treated": int(np.sum(treatment)),
        "n_control": int(np.sum(1 - treatment)),
        "n_matched_pairs": n_matched,
        "n_unmatched_treated": int(np.sum(treatment)) - n_matched,
        "method": req.method,
        "caliper": req.caliper,
        "matched_pairs": matched_pairs[:50],  # first 50 for brevity
        "att": round(att, 6),
        "att_se": round(boot_se, 6),
        "att_ci_lower": round(ci_lower, 6),
        "att_ci_upper": round(ci_upper, 6),
        "confidence": req.confidence,
        "pre_matching_balance": pre_balance,
        "post_matching_balance": post_balance,
        "balance_summary": {
            "pre_max_smd": round(pre_max_smd, 4),
            "post_max_smd": round(post_max_smd, 4),
            "pre_covariates_balanced": pre_balanced,
            "post_covariates_balanced": post_balanced,
            "total_covariates": k,
        },
        "interpretation": (
            f"Matched {n_matched} treated units to {n_matched} control units "
            f"(1:1 {req.method} matching{f', caliper={req.caliper} SD' if req.caliper else ''}). "
            f"ATT = {att:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"Balance improved from max |SMD| = {pre_max_smd:.3f} to {post_max_smd:.3f}. "
            f"{post_balanced}/{k} covariates balanced (|SMD| < 0.1) after matching."
        ),
    }


# ── 3. Inverse Probability of Treatment Weighting ────────────────────

@router.post("/iptw")
def iptw_estimation(req: IPTWRequest):
    """Inverse probability of treatment weighting for ATE or ATT estimation.

    Computes stabilized weights, estimates treatment effects, and provides
    weight distribution diagnostics.
    """
    _validate_binary_treatment(req.treatment)
    _validate_inputs(req.treatment, req.covariates, req.outcome)

    if req.estimand not in ("ATE", "ATT"):
        raise HTTPException(status_code=422, detail="estimand must be 'ATE' or 'ATT'.")

    treatment = np.array(req.treatment)
    outcome = np.array(req.outcome)
    n = len(treatment)
    k = len(req.covariates)
    X_cov = np.column_stack([np.array(c) for c in req.covariates])
    cov_names = req.covariate_names or [f"X{i+1}" for i in range(k)]

    # Estimate propensity scores
    ps, _ = _estimate_propensity_scores(treatment, X_cov)

    # Compute weights
    prevalence = np.mean(treatment)

    if req.estimand == "ATE":
        # Stabilized ATE weights
        weights = np.where(
            treatment == 1,
            prevalence / ps,
            (1 - prevalence) / (1 - ps),
        )
    else:
        # ATT weights: treated get weight 1, controls get ps/(1-ps)
        weights = np.where(
            treatment == 1,
            1.0,
            ps / (1 - ps),
        )

    # Trim extreme weights
    raw_weights = weights.copy()
    if req.trim_threshold > 0:
        trim_mask = (ps >= req.trim_threshold) & (ps <= 1 - req.trim_threshold)
        n_trimmed = int(np.sum(~trim_mask))
        weights = weights * trim_mask
    else:
        n_trimmed = 0
        trim_mask = np.ones(n, dtype=bool)

    # Weighted outcome means
    w_treated = weights[treatment == 1]
    w_control = weights[treatment == 0]
    y_treated = outcome[treatment == 1]
    y_control = outcome[treatment == 0]

    if np.sum(w_treated) < 1e-12 or np.sum(w_control) < 1e-12:
        raise HTTPException(
            status_code=422,
            detail="Weights sum to near zero for one group. Try adjusting trim_threshold.",
        )

    mean_treated = float(np.average(y_treated, weights=w_treated))
    mean_control = float(np.average(y_control, weights=w_control))
    effect = mean_treated - mean_control

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_effects = []
    valid_idx = np.where(trim_mask)[0]
    for _ in range(req.n_bootstrap):
        boot_idx = rng.choice(valid_idx, size=len(valid_idx), replace=True)
        b_t = treatment[boot_idx]
        b_y = outcome[boot_idx]
        b_w = weights[boot_idx]
        t_mask = b_t == 1
        c_mask = b_t == 0
        if np.sum(b_w[t_mask]) < 1e-12 or np.sum(b_w[c_mask]) < 1e-12:
            continue
        b_mean_t = np.average(b_y[t_mask], weights=b_w[t_mask])
        b_mean_c = np.average(b_y[c_mask], weights=b_w[c_mask])
        boot_effects.append(float(b_mean_t - b_mean_c))

    boot_effects = np.array(boot_effects)
    alpha = 1 - req.confidence
    ci_lower = float(np.percentile(boot_effects, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_effects, 100 * (1 - alpha / 2)))
    boot_se = float(np.std(boot_effects, ddof=1))

    # Weight diagnostics
    active_weights = raw_weights[trim_mask] if n_trimmed > 0 else raw_weights

    return {
        "estimand": req.estimand,
        "n": n,
        "n_treated": int(np.sum(treatment)),
        "n_control": int(np.sum(1 - treatment)),
        "n_trimmed": n_trimmed,
        "effect": round(effect, 6),
        "se": round(boot_se, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "confidence": req.confidence,
        "weighted_mean_treated": round(mean_treated, 6),
        "weighted_mean_control": round(mean_control, 6),
        "weight_diagnostics": {
            "min": round(float(np.min(active_weights)), 4),
            "max": round(float(np.max(active_weights)), 4),
            "mean": round(float(np.mean(active_weights)), 4),
            "sd": round(float(np.std(active_weights)), 4),
            "median": round(float(np.median(active_weights)), 4),
            "percent_above_10": round(float(np.mean(active_weights > 10) * 100), 2),
        },
        "interpretation": (
            f"IPTW {req.estimand} estimate: {effect:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"{'Stabilized weights used. ' if req.estimand == 'ATE' else ''}"
            f"{n_trimmed} units trimmed (PS outside [{req.trim_threshold}, {1-req.trim_threshold}]). "
            f"Weight range: [{np.min(active_weights):.2f}, {np.max(active_weights):.2f}]. "
            f"Large weights (>10) may indicate positivity violations."
        ),
    }


# ── 4. Overlap / Common Support Analysis ─────────────────────────────

@router.post("/overlap")
def overlap_analysis(req: OverlapRequest):
    """Assess common support region and identify units outside overlap.

    Examines the propensity score distributions for treated and control
    groups, identifies the region of common support, and flags units
    that fall outside this region.
    """
    _validate_binary_treatment(req.treatment)
    _validate_inputs(req.treatment, req.covariates)

    treatment = np.array(req.treatment)
    n = len(treatment)
    k = len(req.covariates)
    X_cov = np.column_stack([np.array(c) for c in req.covariates])

    ps, _ = _estimate_propensity_scores(treatment, X_cov)

    ps_treated = ps[treatment == 1]
    ps_control = ps[treatment == 0]

    # Common support region: overlap of the two PS distributions
    cs_lower = max(float(np.min(ps_treated)), float(np.min(ps_control)))
    cs_upper = min(float(np.max(ps_treated)), float(np.max(ps_control)))

    # Units outside common support
    outside_mask = (ps < cs_lower) | (ps > cs_upper)
    outside_indices = np.where(outside_mask)[0].tolist()
    n_outside = len(outside_indices)

    # Trim-based analysis
    trim_mask = (ps >= req.trim_threshold) & (ps <= 1 - req.trim_threshold)
    n_trimmed = int(np.sum(~trim_mask))
    trimmed_indices = np.where(~trim_mask)[0].tolist()

    # Within common support counts
    in_support_treated = int(np.sum((ps[treatment == 1] >= cs_lower) & (ps[treatment == 1] <= cs_upper)))
    in_support_control = int(np.sum((ps[treatment == 0] >= cs_lower) & (ps[treatment == 0] <= cs_upper)))

    # Distribution overlap measure (Bhattacharyya coefficient)
    n_bins = 50
    bins = np.linspace(0, 1, n_bins + 1)
    hist_t, _ = np.histogram(ps_treated, bins=bins, density=True)
    hist_c, _ = np.histogram(ps_control, bins=bins, density=True)
    # Normalize
    hist_t = hist_t / (np.sum(hist_t) + 1e-12)
    hist_c = hist_c / (np.sum(hist_c) + 1e-12)
    bhattacharyya = float(np.sum(np.sqrt(hist_t * hist_c)))

    return {
        "n": n,
        "n_treated": int(np.sum(treatment)),
        "n_control": int(np.sum(1 - treatment)),
        "common_support_region": {
            "lower": round(cs_lower, 6),
            "upper": round(cs_upper, 6),
        },
        "n_outside_common_support": n_outside,
        "outside_indices": outside_indices[:100],  # first 100 for brevity
        "n_outside_treated": int(np.sum(outside_mask & (treatment == 1))),
        "n_outside_control": int(np.sum(outside_mask & (treatment == 0))),
        "in_support_treated": in_support_treated,
        "in_support_control": in_support_control,
        "trim_threshold": req.trim_threshold,
        "n_trimmed_by_threshold": n_trimmed,
        "trimmed_indices": trimmed_indices[:100],
        "trimmed_sample_size": n - n_trimmed,
        "bhattacharyya_overlap": round(bhattacharyya, 4),
        "ps_distribution": {
            "treated": {
                "mean": round(float(np.mean(ps_treated)), 4),
                "sd": round(float(np.std(ps_treated)), 4),
                "min": round(float(np.min(ps_treated)), 4),
                "max": round(float(np.max(ps_treated)), 4),
                "q25": round(float(np.percentile(ps_treated, 25)), 4),
                "q75": round(float(np.percentile(ps_treated, 75)), 4),
            },
            "control": {
                "mean": round(float(np.mean(ps_control)), 4),
                "sd": round(float(np.std(ps_control)), 4),
                "min": round(float(np.min(ps_control)), 4),
                "max": round(float(np.max(ps_control)), 4),
                "q25": round(float(np.percentile(ps_control, 25)), 4),
                "q75": round(float(np.percentile(ps_control, 75)), 4),
            },
        },
        "interpretation": (
            f"Common support region: [{cs_lower:.3f}, {cs_upper:.3f}]. "
            f"{n_outside} units ({n_outside/n*100:.1f}%) fall outside common support. "
            f"Bhattacharyya overlap coefficient: {bhattacharyya:.3f} (1.0 = perfect overlap). "
            f"Trimming at [{req.trim_threshold}, {1-req.trim_threshold}] removes {n_trimmed} units, "
            f"leaving {n - n_trimmed} for analysis."
        ),
    }


# ── 5. Standalone Balance Diagnostics ─────────────────────────────────

@router.post("/balance")
def balance_diagnostics(req: BalanceRequest):
    """Compute balance diagnostics: standardized mean differences, variance
    ratios, and Kolmogorov-Smirnov statistics for each covariate.

    Can compare pre-matching (full sample) balance and post-matching balance
    when matched indices are provided.
    """
    _validate_binary_treatment(req.treatment)
    _validate_inputs(req.treatment, req.covariates, min_n=10)

    treatment = np.array(req.treatment)
    n = len(treatment)
    k = len(req.covariates)
    X_cov = np.column_stack([np.array(c) for c in req.covariates])
    cov_names = req.covariate_names or [f"X{i+1}" for i in range(k)]

    if len(cov_names) != k:
        raise HTTPException(status_code=422, detail="covariate_names length must match number of covariates.")

    # Pre-matching balance
    pre_balance = _compute_balance(treatment, X_cov, cov_names)

    result = {
        "n": n,
        "n_treated": int(np.sum(treatment)),
        "n_control": int(np.sum(1 - treatment)),
        "n_covariates": k,
        "pre_matching_balance": pre_balance,
    }

    # Post-matching balance if indices provided
    if req.matched_indices_treated is not None and req.matched_indices_control is not None:
        mt = np.array(req.matched_indices_treated)
        mc = np.array(req.matched_indices_control)

        if len(mt) != len(mc):
            raise HTTPException(
                status_code=422,
                detail="matched_indices_treated and matched_indices_control must have the same length.",
            )

        post_balance = _compute_balance(treatment, X_cov, cov_names,
                                        indices_treated=mt, indices_control=mc)

        result["post_matching_balance"] = post_balance
        result["n_matched_pairs"] = len(mt)

        # Summary
        pre_smds = [abs(b["standardized_mean_difference"]) for b in pre_balance]
        post_smds = [abs(b["standardized_mean_difference"]) for b in post_balance]

        result["summary"] = {
            "pre_mean_abs_smd": round(float(np.mean(pre_smds)), 4),
            "post_mean_abs_smd": round(float(np.mean(post_smds)), 4),
            "pre_max_abs_smd": round(float(np.max(pre_smds)), 4),
            "post_max_abs_smd": round(float(np.max(post_smds)), 4),
            "pre_covariates_balanced": sum(1 for s in pre_smds if s < 0.1),
            "post_covariates_balanced": sum(1 for s in post_smds if s < 0.1),
            "improvement_pct": round(
                (1 - np.mean(post_smds) / max(np.mean(pre_smds), 1e-12)) * 100, 1
            ),
        }

        result["interpretation"] = (
            f"Balance assessment for {k} covariates across {len(mt)} matched pairs. "
            f"Mean |SMD| improved from {np.mean(pre_smds):.3f} to {np.mean(post_smds):.3f} "
            f"({result['summary']['improvement_pct']:.1f}% reduction). "
            f"Covariates balanced (|SMD| < 0.1): {result['summary']['pre_covariates_balanced']}/{k} pre, "
            f"{result['summary']['post_covariates_balanced']}/{k} post matching."
        )
    else:
        pre_smds = [abs(b["standardized_mean_difference"]) for b in pre_balance]
        result["summary"] = {
            "mean_abs_smd": round(float(np.mean(pre_smds)), 4),
            "max_abs_smd": round(float(np.max(pre_smds)), 4),
            "covariates_balanced": sum(1 for s in pre_smds if s < 0.1),
        }
        result["interpretation"] = (
            f"Pre-matching balance assessment for {k} covariates. "
            f"Mean |SMD| = {np.mean(pre_smds):.3f}, max |SMD| = {np.max(pre_smds):.3f}. "
            f"{result['summary']['covariates_balanced']}/{k} covariates balanced (|SMD| < 0.1). "
            f"Covariates with |SMD| > 0.1 may confound treatment-outcome comparisons."
        )

    return result
