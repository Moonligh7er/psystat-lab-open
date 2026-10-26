"""Political Science — Statistical methods for policy evaluation, legislative analysis, and panel data."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.integrate import quad
import statsmodels.api as sm

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════
# Request Models
# ══════════════════════════════════════════════════════════════════════

class RDDRequest(BaseModel):
    running_variable: list[float]
    outcome: list[float]
    cutoff: float = 0
    bandwidth: float | None = None
    kernel: str = "triangular"
    polynomial_order: int = 1
    covariates: list[list[float]] | None = None


class IdealPointRequest(BaseModel):
    vote_matrix: list[list[int]]
    legislator_labels: list[str] | None = None
    vote_labels: list[str] | None = None
    dimensions: int = 1


class DiffInDiffRequest(BaseModel):
    outcome: list[float]
    treated: list[int]
    post: list[int]
    covariates: list[list[float]] | None = None
    covariate_names: list[str] | None = None
    cluster: list[int | str] | None = None


class TSCSRequest(BaseModel):
    dependent: list[float]
    predictors: list[list[float]]
    units: list[int | str]
    time: list[int | float]
    predictor_names: list[str] | None = None
    model: str = "fixed"


class VoteScalingRequest(BaseModel):
    responses: list[list[int]]
    item_labels: list[str] | None = None
    n_dimensions: int = 2


# ══════════════════════════════════════════════════════════════════════
# 1. Regression Discontinuity Design
# ══════════════════════════════════════════════════════════════════════

@router.post("/regression-discontinuity")
def regression_discontinuity(req: RDDRequest):
    """Regression Discontinuity Design (RDD) with local polynomial estimation."""
    X = np.array(req.running_variable, dtype=float)
    Y = np.array(req.outcome, dtype=float)
    c = req.cutoff

    if len(X) != len(Y):
        raise HTTPException(400, "running_variable and outcome must have the same length.")
    if len(X) < 10:
        raise HTTPException(400, "Need at least 10 observations for RDD.")
    if req.kernel not in ("triangular", "uniform", "epanechnikov"):
        raise HTTPException(400, "kernel must be 'triangular', 'uniform', or 'epanechnikov'.")
    if req.polynomial_order < 1 or req.polynomial_order > 5:
        raise HTTPException(400, "polynomial_order must be between 1 and 5.")

    # Centre running variable at cutoff
    R = X - c
    left_mask = R < 0
    right_mask = R >= 0
    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))

    if n_left < 3 or n_right < 3:
        raise HTTPException(400, "Need at least 3 observations on each side of cutoff.")

    # ── Bandwidth selection (IK method approximation) ─────────────
    if req.bandwidth is None:
        h = _ik_bandwidth(R, Y)
    else:
        h = req.bandwidth
        if h <= 0:
            raise HTTPException(400, "bandwidth must be positive.")

    # ── Kernel weights ────────────────────────────────────────────
    weights = _kernel_weights(R, h, req.kernel)

    # ── Local polynomial regression on each side ──────────────────
    p = req.polynomial_order
    covs = np.array(req.covariates).T if req.covariates else None

    left_coef, left_cov = _local_poly(R, Y, weights, left_mask, p, covs)
    right_coef, right_cov = _local_poly(R, Y, weights, right_mask, p, covs)

    if left_coef is None or right_coef is None:
        raise HTTPException(400, "Not enough observations within the bandwidth on one side.")

    # LATE = intercept_right - intercept_left
    tau = float(right_coef[0] - left_coef[0])
    se_tau = float(np.sqrt(right_cov[0, 0] + left_cov[0, 0]))
    t_stat = tau / se_tau if se_tau > 0 else 0.0
    df_tau = n_left + n_right - 2 * (p + 1)
    df_tau = max(df_tau, 1)
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df_tau))
    ci_low = float(tau - sp_stats.t.ppf(0.975, df_tau) * se_tau)
    ci_high = float(tau + sp_stats.t.ppf(0.975, df_tau) * se_tau)

    # ── McCrary density test ──────────────────────────────────────
    mccrary = _mccrary_test(R)

    # ── Placebo tests at false cutoffs ────────────────────────────
    placebo_results = _placebo_tests(R, Y, h, req.kernel, p)

    # ── Visualisation data ────────────────────────────────────────
    viz = _rdd_viz_data(R, Y, left_coef, right_coef, p, c, h)

    return {
        "late": {
            "estimate": tau,
            "se": se_tau,
            "t": float(t_stat),
            "p": p_value,
            "ci": [ci_low, ci_high],
        },
        "bandwidth": float(h),
        "n_left": n_left,
        "n_right": n_right,
        "mccrary_test": mccrary,
        "placebo_tests": placebo_results,
        "coefficients_left": [float(c) for c in left_coef],
        "coefficients_right": [float(c) for c in right_coef],
        "visualization": viz,
    }


# ── RDD helper functions ─────────────────────────────────────────────

def _kernel_weights(R: np.ndarray, h: float, kernel: str) -> np.ndarray:
    u = R / h
    if kernel == "triangular":
        w = np.maximum(1 - np.abs(u), 0)
    elif kernel == "uniform":
        w = (np.abs(u) <= 1).astype(float)
    else:  # epanechnikov
        w = np.maximum(0.75 * (1 - u ** 2), 0)
    return w


def _ik_bandwidth(R: np.ndarray, Y: np.ndarray) -> float:
    """Imbens-Kalyanaraman optimal bandwidth (simplified)."""
    n = len(R)
    sd_R = np.std(R)
    if sd_R == 0:
        return 1.0

    # Pilot bandwidth: Silverman rule-of-thumb
    h_pilot = 1.06 * sd_R * n ** (-0.2)

    # Estimate second derivative of conditional mean using pilot
    mask_pilot = np.abs(R) <= h_pilot
    if np.sum(mask_pilot) < 10:
        return h_pilot

    try:
        R_p = R[mask_pilot]
        Y_p = Y[mask_pilot]
        X_p = np.column_stack([np.ones(len(R_p)), R_p, R_p ** 2])
        beta = np.linalg.lstsq(X_p, Y_p, rcond=None)[0]
        m2 = beta[2]  # second derivative estimate
    except Exception:
        return h_pilot

    if abs(m2) < 1e-12:
        return h_pilot

    # IK bandwidth formula (simplified)
    C_k = 3.4375  # for triangular kernel
    sigma2 = np.var(Y)
    h_opt = C_k * (sigma2 / (m2 ** 2 * n)) ** (1 / 5)
    h_opt = max(h_opt, sd_R * 0.1)  # floor
    h_opt = min(h_opt, sd_R * 2.0)  # cap
    return float(h_opt)


def _local_poly(
    R: np.ndarray, Y: np.ndarray, weights: np.ndarray,
    side_mask: np.ndarray, p: int, covs: np.ndarray | None
) -> tuple:
    """Weighted local polynomial regression on one side of cutoff."""
    mask = side_mask & (weights > 0)
    if np.sum(mask) < p + 2:
        return None, None

    R_s = R[mask]
    Y_s = Y[mask]
    W_s = weights[mask]

    # Build polynomial design matrix
    cols = [np.ones(len(R_s))]
    for j in range(1, p + 1):
        cols.append(R_s ** j)
    if covs is not None:
        for col_idx in range(covs.shape[1]):
            cov_col = covs[mask, col_idx] if covs.shape[0] > np.sum(mask) else covs[:, col_idx]
            if len(cov_col) == len(R_s):
                cols.append(cov_col)
    Xmat = np.column_stack(cols)

    # Weighted least squares
    W_diag = np.diag(W_s)
    try:
        XtWX = Xmat.T @ W_diag @ Xmat
        XtWY = Xmat.T @ W_diag @ Y_s
        beta = np.linalg.solve(XtWX, XtWY)
        resid = Y_s - Xmat @ beta
        sigma2 = np.sum(W_s * resid ** 2) / max(np.sum(mask) - len(beta), 1)
        cov_beta = sigma2 * np.linalg.inv(XtWX)
        return beta, cov_beta
    except np.linalg.LinAlgError:
        return None, None


def _mccrary_test(R: np.ndarray) -> dict:
    """McCrary (2008) density test for manipulation at the cutoff."""
    n = len(R)
    # Histogram-based density estimation on each side
    n_bins = max(int(np.sqrt(n)), 10)
    left_data = R[R < 0]
    right_data = R[R >= 0]

    if len(left_data) < 5 or len(right_data) < 5:
        return {"test_statistic": None, "p_value": None, "note": "Insufficient data"}

    # Estimate densities near cutoff using narrow bins
    bw = (np.max(R) - np.min(R)) / n_bins
    if bw <= 0:
        return {"test_statistic": None, "p_value": None, "note": "Zero bandwidth"}

    # Density just left of cutoff
    near_left = left_data[left_data > -bw * 3]
    near_right = right_data[right_data < bw * 3]
    f_left = len(near_left) / (n * bw * 3) if bw > 0 else 0
    f_right = len(near_right) / (n * bw * 3) if bw > 0 else 0

    # Log difference test
    if f_left > 0 and f_right > 0:
        theta = np.log(f_right) - np.log(f_left)
        se = np.sqrt(1 / max(len(near_right), 1) + 1 / max(len(near_left), 1))
        z = theta / se if se > 0 else 0.0
        p_val = float(2 * sp_stats.norm.sf(abs(z)))
    else:
        z = 0.0
        p_val = 1.0

    return {
        "test_statistic": float(z),
        "p_value": p_val,
        "density_left": float(f_left),
        "density_right": float(f_right),
        "interpretation": "No evidence of manipulation" if p_val > 0.05
                          else "Possible manipulation at cutoff",
    }


def _placebo_tests(
    R: np.ndarray, Y: np.ndarray, h: float, kernel: str, p: int
) -> list[dict]:
    """Run RDD at false cutoffs (median of each side)."""
    results = []
    for side_name, mask in [("left", R < 0), ("right", R >= 0)]:
        R_side = R[mask]
        Y_side = Y[mask]
        if len(R_side) < 10:
            continue
        false_cutoff = float(np.median(R_side))
        R_shifted = R_side - false_cutoff
        w = _kernel_weights(R_shifted, h, kernel)
        left_m = R_shifted < 0
        right_m = R_shifted >= 0
        lc, lcov = _local_poly(R_shifted, Y_side, w, left_m, p, None)
        rc, rcov = _local_poly(R_shifted, Y_side, w, right_m, p, None)
        if lc is not None and rc is not None:
            tau_p = float(rc[0] - lc[0])
            se_p = float(np.sqrt(rcov[0, 0] + lcov[0, 0]))
            t_p = tau_p / se_p if se_p > 0 else 0.0
            results.append({
                "false_cutoff": false_cutoff,
                "side": side_name,
                "estimate": tau_p,
                "se": se_p,
                "t": float(t_p),
                "p": float(2 * sp_stats.norm.sf(abs(t_p))),
            })
    return results


def _rdd_viz_data(
    R: np.ndarray, Y: np.ndarray,
    left_coef: np.ndarray, right_coef: np.ndarray,
    p: int, cutoff: float, h: float
) -> dict:
    """Binned scatter + fitted curves for RDD plot."""
    n_bins = min(40, max(10, len(R) // 5))
    left_mask = R < 0
    right_mask = R >= 0

    def _bin_means(r, y, n_b):
        if len(r) < 2:
            return [], []
        bins_e = np.linspace(r.min(), r.max(), n_b + 1)
        centres, means = [], []
        for i in range(n_b):
            in_bin = (r >= bins_e[i]) & (r < bins_e[i + 1])
            if np.any(in_bin):
                centres.append(float(np.mean(r[in_bin]) + cutoff))
                means.append(float(np.mean(y[in_bin])))
        return centres, means

    bins_l, means_l = _bin_means(R[left_mask], Y[left_mask], n_bins // 2)
    bins_r, means_r = _bin_means(R[right_mask], Y[right_mask], n_bins // 2)

    # Fitted curves
    def _fitted(coef, r_range):
        r_grid = np.linspace(r_range[0], r_range[1], 100)
        y_hat = np.zeros_like(r_grid)
        for j in range(len(coef)):
            if j <= p:
                y_hat += coef[j] * r_grid ** j
        return (r_grid + cutoff).tolist(), y_hat.tolist()

    fit_l_x, fit_l_y = _fitted(left_coef, (max(R[left_mask].min(), -h), 0))
    fit_r_x, fit_r_y = _fitted(right_coef, (0, min(R[right_mask].max(), h)))

    return {
        "running_var_bins_left": bins_l,
        "outcome_means_left": means_l,
        "running_var_bins_right": bins_r,
        "outcome_means_right": means_r,
        "fitted_left_x": fit_l_x,
        "fitted_left_y": fit_l_y,
        "fitted_right_x": fit_r_x,
        "fitted_right_y": fit_r_y,
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Ideal Point Estimation
# ══════════════════════════════════════════════════════════════════════

@router.post("/ideal-point")
def ideal_point(req: IdealPointRequest):
    """Ideal point estimation from roll-call / survey vote matrix using IRT-like decomposition."""
    V = np.array(req.vote_matrix, dtype=float)
    n_legs, n_votes = V.shape

    if n_legs < 3:
        raise HTTPException(400, "Need at least 3 legislators (rows).")
    if n_votes < 3:
        raise HTTPException(400, "Need at least 3 votes (columns).")
    if req.dimensions < 1 or req.dimensions > min(n_legs, n_votes, 5):
        raise HTTPException(400, f"dimensions must be between 1 and {min(n_legs, n_votes, 5)}.")

    d = req.dimensions
    leg_labels = req.legislator_labels or [f"Legislator_{i}" for i in range(n_legs)]
    vote_lbls = req.vote_labels or [f"Vote_{j}" for j in range(n_votes)]

    if len(leg_labels) != n_legs:
        raise HTTPException(400, "legislator_labels length must match number of rows in vote_matrix.")
    if len(vote_lbls) != n_votes:
        raise HTTPException(400, "vote_labels length must match number of columns in vote_matrix.")

    # Replace -1 (abstain/missing) with NaN for masking
    mask_valid = V >= 0
    V_work = np.where(mask_valid, V, np.nan)

    # ── Agreement matrix & double-centering ───────────────────────
    # Compute pairwise agreement (proportion of shared votes where both valid)
    agree = np.zeros((n_legs, n_legs))
    for i in range(n_legs):
        for j in range(i, n_legs):
            both = mask_valid[i] & mask_valid[j]
            if np.sum(both) > 0:
                a = np.sum(V_work[i, both] == V_work[j, both]) / np.sum(both)
            else:
                a = 0.5
            agree[i, j] = a
            agree[j, i] = a

    # Double-center the agreement matrix
    row_means = np.nanmean(agree, axis=1, keepdims=True)
    col_means = np.nanmean(agree, axis=0, keepdims=True)
    grand_mean = np.nanmean(agree)
    B = agree - row_means - col_means + grand_mean

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take top d dimensions
    pos_eigvals = np.maximum(eigvals[:d], 0)
    ideal_pts = eigvecs[:, :d] * np.sqrt(pos_eigvals)[np.newaxis, :]

    # ── Discrimination & difficulty via 2PL-like estimation ───────
    # Simple approach: regress each vote on ideal points
    discrimination = []
    difficulty = []
    predicted_correct = 0
    total_valid = 0

    for j in range(n_votes):
        valid_j = mask_valid[:, j]
        if np.sum(valid_j) < 3:
            discrimination.append([0.0] * d)
            difficulty.append(0.0)
            continue

        y_j = V_work[valid_j, j]
        X_j = ideal_pts[valid_j]
        X_j_c = sm.add_constant(X_j)

        try:
            # Logistic regression for 2PL
            model = sm.GLM(y_j, X_j_c, family=sm.families.Binomial())
            result = model.fit(disp=False)
            coefs = result.params
            difficulty.append(float(-coefs[0]))
            discrimination.append([float(c) for c in coefs[1:]])

            # Classification accuracy
            pred = (result.predict(X_j_c) >= 0.5).astype(float)
            predicted_correct += np.sum(pred == y_j)
            total_valid += len(y_j)
        except Exception:
            difficulty.append(0.0)
            discrimination.append([0.0] * d)
            total_valid += int(np.sum(valid_j))

    accuracy = float(predicted_correct / total_valid) if total_valid > 0 else 0.0

    # ── Polarization index ────────────────────────────────────────
    # Bimodality coefficient on first dimension
    x1 = ideal_pts[:, 0]
    n_ = len(x1)
    skew = float(sp_stats.skew(x1))
    kurt = float(sp_stats.kurtosis(x1, fisher=True))  # excess kurtosis
    # Sarle's bimodality coefficient
    bc = (skew ** 2 + 1) / (kurt + 3 * (n_ - 1) ** 2 / ((n_ - 2) * (n_ - 3)))
    polarization_index = float(np.clip(bc, 0, 1))

    # ── Build response ────────────────────────────────────────────
    legislator_positions = []
    for i in range(n_legs):
        pos = {"label": leg_labels[i]}
        if d == 1:
            pos["position"] = float(ideal_pts[i, 0])
        else:
            pos["position"] = [float(ideal_pts[i, k]) for k in range(d)]
        legislator_positions.append(pos)

    vote_params = []
    for j in range(n_votes):
        vote_params.append({
            "label": vote_lbls[j],
            "discrimination": discrimination[j],
            "difficulty": difficulty[j],
        })

    variance_explained = []
    total_var = np.sum(np.maximum(eigvals, 0))
    for k in range(d):
        ve = float(pos_eigvals[k] / total_var) if total_var > 0 else 0.0
        variance_explained.append(ve)

    return {
        "legislator_positions": legislator_positions,
        "vote_parameters": vote_params,
        "eigenvalues": [float(e) for e in eigvals[:min(d + 3, len(eigvals))]],
        "variance_explained": variance_explained,
        "classification_accuracy": accuracy,
        "polarization_index": polarization_index,
        "n_legislators": n_legs,
        "n_votes": n_votes,
        "dimensions": d,
        "visualization": {
            "legislator_positions": legislator_positions,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# 3. Difference-in-Differences
# ══════════════════════════════════════════════════════════════════════

@router.post("/diff-in-diff")
def diff_in_diff(req: DiffInDiffRequest):
    """Difference-in-Differences estimation with optional clustered standard errors."""
    Y = np.array(req.outcome, dtype=float)
    D = np.array(req.treated, dtype=float)
    P = np.array(req.post, dtype=float)
    n = len(Y)

    if not (len(D) == len(P) == n):
        raise HTTPException(400, "outcome, treated, and post must all have the same length.")
    if n < 8:
        raise HTTPException(400, "Need at least 8 observations.")
    if not set(np.unique(D)).issubset({0, 1}):
        raise HTTPException(400, "treated must contain only 0 and 1.")
    if not set(np.unique(P)).issubset({0, 1}):
        raise HTTPException(400, "post must contain only 0 and 1.")

    # Check we have all 4 cells
    for d_val in [0, 1]:
        for p_val in [0, 1]:
            cell = (D == d_val) & (P == p_val)
            if np.sum(cell) < 1:
                raise HTTPException(400, f"No observations in cell treated={d_val}, post={p_val}.")

    # ── Build design matrix ───────────────────────────────────────
    interaction = D * P
    X_cols = [np.ones(n), D, P, interaction]
    col_names = ["const", "treated", "post", "treated_x_post"]

    if req.covariates:
        covs = np.array(req.covariates, dtype=float)
        if covs.ndim == 1:
            covs = covs.reshape(-1, 1)
        if covs.shape[0] != n:
            # Might be transposed
            if covs.shape[1] == n:
                covs = covs.T
            else:
                raise HTTPException(400, "Covariates dimensions don't match outcome length.")
        cov_names = req.covariate_names or [f"cov_{i}" for i in range(covs.shape[1])]
        if len(cov_names) != covs.shape[1]:
            raise HTTPException(400, "covariate_names length must match number of covariates.")
        for k in range(covs.shape[1]):
            X_cols.append(covs[:, k])
            col_names.append(cov_names[k])

    X = np.column_stack(X_cols)

    # ── OLS estimation ────────────────────────────────────────────
    try:
        model = sm.OLS(Y, X)

        if req.cluster is not None:
            clusters = np.array(req.cluster)
            if len(clusters) != n:
                raise HTTPException(400, "cluster must have the same length as outcome.")
            result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters}, disp=False)
            se_type = "cluster-robust"
        else:
            result = model.fit(cov_type="HC1", disp=False)
            se_type = "HC1-robust"
    except Exception as e:
        raise HTTPException(400, f"Estimation failed: {str(e)}")

    # DiD coefficient is the interaction term (index 3)
    did_idx = 3
    did_est = float(result.params[did_idx])
    did_se = float(result.bse[did_idx])
    did_t = float(result.tvalues[did_idx])
    did_p = float(result.pvalues[did_idx])
    ci = result.conf_int(alpha=0.05)
    did_ci = [float(ci[did_idx, 0]), float(ci[did_idx, 1])]

    # ── Group means ───────────────────────────────────────────────
    mean_ctrl_pre = float(np.mean(Y[(D == 0) & (P == 0)]))
    mean_ctrl_post = float(np.mean(Y[(D == 0) & (P == 1)]))
    mean_treat_pre = float(np.mean(Y[(D == 1) & (P == 0)]))
    mean_treat_post = float(np.mean(Y[(D == 1) & (P == 1)]))

    # ── Parallel trends assessment ────────────────────────────────
    # Simple check: compare pre-period trends if multiple pre-periods exist
    # With only two periods, report the pre-treatment difference
    pre_diff = mean_treat_pre - mean_ctrl_pre
    post_diff = mean_treat_post - mean_ctrl_post
    parallel_trends = {
        "pre_treatment_diff": float(pre_diff),
        "post_treatment_diff": float(post_diff),
        "did_estimate_manual": float(post_diff - pre_diff),
        "note": "With only pre/post data, parallel trends is assumed, not testable. "
                "Use multiple pre-periods for a formal test.",
    }

    # ── All coefficients ──────────────────────────────────────────
    coefficients = {}
    for i, name in enumerate(col_names):
        coefficients[name] = {
            "estimate": float(result.params[i]),
            "se": float(result.bse[i]),
            "t": float(result.tvalues[i]),
            "p": float(result.pvalues[i]),
            "ci": [float(ci[i, 0]), float(ci[i, 1])],
        }

    return {
        "did_estimate": {
            "estimate": did_est,
            "se": did_se,
            "t": did_t,
            "p": did_p,
            "ci": did_ci,
        },
        "pre_treatment_means": {
            "treatment": mean_treat_pre,
            "control": mean_ctrl_pre,
        },
        "post_treatment_means": {
            "treatment": mean_treat_post,
            "control": mean_ctrl_post,
        },
        "parallel_trends": parallel_trends,
        "r_squared": float(result.rsquared),
        "r_squared_adj": float(result.rsquared_adj),
        "se_type": se_type,
        "coefficients": coefficients,
        "n": n,
        "n_clusters": int(len(np.unique(req.cluster))) if req.cluster else None,
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Time-Series Cross-Sectional (Panel Data)
# ══════════════════════════════════════════════════════════════════════

@router.post("/tscs")
def tscs(req: TSCSRequest):
    """Time-Series Cross-Sectional panel data models (Fixed / Random effects + Hausman)."""
    Y = np.array(req.dependent, dtype=float)
    n = len(Y)
    X_raw = np.array(req.predictors, dtype=float)

    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(-1, 1)
    # If predictors come as columns (k x n), transpose to (n x k)
    if X_raw.shape[0] != n and X_raw.shape[1] == n:
        X_raw = X_raw.T
    if X_raw.shape[0] != n:
        raise HTTPException(400, "predictors dimensions don't match dependent length.")

    units = np.array(req.units)
    time = np.array(req.time, dtype=float)
    if len(units) != n or len(time) != n:
        raise HTTPException(400, "units and time must have the same length as dependent.")

    if req.model not in ("fixed", "random", "both"):
        raise HTTPException(400, "model must be 'fixed', 'random', or 'both'.")

    k = X_raw.shape[1]
    pred_names = req.predictor_names or [f"x{i}" for i in range(k)]
    if len(pred_names) != k:
        raise HTTPException(400, "predictor_names length must match number of predictors.")

    unique_units = np.unique(units)
    n_units = len(unique_units)
    unique_times = np.unique(time)
    n_times = len(unique_times)

    if n_units < 2:
        raise HTTPException(400, "Need at least 2 panel units.")
    if n_times < 2:
        raise HTTPException(400, "Need at least 2 time periods.")

    results = {}

    # ── Fixed Effects (within estimator) ──────────────────────────
    if req.model in ("fixed", "both"):
        fe_result = _fixed_effects(Y, X_raw, units, unique_units, pred_names)
        results["fixed_effects"] = fe_result

    # ── Random Effects (GLS) ──────────────────────────────────────
    if req.model in ("random", "both"):
        re_result = _random_effects(Y, X_raw, units, unique_units, pred_names)
        results["random_effects"] = re_result

    # ── Hausman Test ──────────────────────────────────────────────
    if req.model == "both" and results.get("fixed_effects") and results.get("random_effects"):
        fe_b = np.array(results["fixed_effects"]["_beta"])
        re_b = np.array(results["random_effects"]["_beta"])
        fe_V = np.array(results["fixed_effects"]["_vcov"])
        re_V = np.array(results["random_effects"]["_vcov"])

        diff = fe_b - re_b
        try:
            V_diff = fe_V - re_V
            chi2 = float(diff @ np.linalg.inv(V_diff) @ diff)
            chi2 = max(chi2, 0)
        except np.linalg.LinAlgError:
            chi2 = float(np.sum(diff ** 2))

        hausman_df = k
        hausman_p = float(sp_stats.chi2.sf(chi2, hausman_df))
        results["hausman_test"] = {
            "chi_square": chi2,
            "df": hausman_df,
            "p_value": hausman_p,
            "recommendation": "Fixed effects preferred (reject RE consistency)"
                              if hausman_p < 0.05
                              else "Random effects is consistent and efficient",
        }

        # Clean up internal keys
        for key in ("fixed_effects", "random_effects"):
            results[key].pop("_beta", None)
            results[key].pop("_vcov", None)
    else:
        # Clean internal keys from whichever model was run
        for key in ("fixed_effects", "random_effects"):
            if key in results:
                results[key].pop("_beta", None)
                results[key].pop("_vcov", None)

    # ── Serial correlation test (Wooldridge) ──────────────────────
    wooldridge = _wooldridge_test(Y, X_raw, units, time, unique_units)
    results["serial_correlation_test"] = wooldridge

    # ── Heteroskedasticity test (modified Wald) ───────────────────
    hetero = _panel_hetero_test(Y, X_raw, units, unique_units)
    results["heteroskedasticity_test"] = hetero

    results["n_observations"] = n
    results["n_units"] = n_units
    results["n_periods"] = n_times

    return results


# ── Panel helpers ─────────────────────────────────────────────────────

def _fixed_effects(
    Y: np.ndarray, X: np.ndarray, units: np.ndarray,
    unique_units: np.ndarray, pred_names: list[str]
) -> dict:
    """Within (fixed-effects) estimator."""
    n = len(Y)
    k = X.shape[1]

    # Demean by unit
    Y_dm = Y.copy()
    X_dm = X.copy()
    unit_effects = {}
    for u in unique_units:
        mask = units == u
        Y_dm[mask] -= np.mean(Y[mask])
        X_dm[mask] -= np.mean(X[mask], axis=0)
        unit_effects[str(u)] = float(np.mean(Y[mask]) - np.mean(X[mask], axis=0) @ np.zeros(k))

    # OLS on demeaned data
    try:
        XtX = X_dm.T @ X_dm
        XtY = X_dm.T @ Y_dm
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Fixed effects estimation failed: singular matrix.")

    resid = Y_dm - X_dm @ beta
    dof = n - len(unique_units) - k
    dof = max(dof, 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    vcov = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(vcov))

    # Recompute unit effects with estimated beta
    for u in unique_units:
        mask = units == u
        unit_effects[str(u)] = float(np.mean(Y[mask] - X[mask] @ beta))

    # R-squared (within)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((Y_dm - np.mean(Y_dm)) ** 2)
    r2_within = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    coefficients = {}
    ci_mult = sp_stats.t.ppf(0.975, dof)
    for i, name in enumerate(pred_names):
        t_val = float(beta[i] / se[i]) if se[i] > 0 else 0.0
        p_val = float(2 * sp_stats.t.sf(abs(t_val), dof))
        coefficients[name] = {
            "estimate": float(beta[i]),
            "se": float(se[i]),
            "t": t_val,
            "p": p_val,
            "ci": [float(beta[i] - ci_mult * se[i]), float(beta[i] + ci_mult * se[i])],
        }

    return {
        "coefficients": coefficients,
        "r_squared_within": r2_within,
        "sigma2": sigma2,
        "unit_effects": unit_effects,
        "_beta": beta.tolist(),
        "_vcov": vcov.tolist(),
    }


def _random_effects(
    Y: np.ndarray, X: np.ndarray, units: np.ndarray,
    unique_units: np.ndarray, pred_names: list[str]
) -> dict:
    """GLS random-effects estimator."""
    n = len(Y)
    k = X.shape[1]

    # Step 1: Pooled OLS for initial residuals
    X_c = sm.add_constant(X)
    try:
        pooled = sm.OLS(Y, X_c).fit(disp=False)
    except Exception:
        raise HTTPException(400, "Random effects estimation failed at pooled OLS step.")

    resid_pooled = pooled.resid

    # Step 2: Estimate variance components
    # sigma2_e from within residuals
    Y_dm = Y.copy()
    X_dm = X.copy()
    for u in unique_units:
        mask = units == u
        Y_dm[mask] -= np.mean(Y[mask])
        X_dm[mask] -= np.mean(X[mask], axis=0)

    try:
        beta_fe = np.linalg.solve(X_dm.T @ X_dm, X_dm.T @ Y_dm)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Random effects estimation failed.")

    resid_fe = Y_dm - X_dm @ beta_fe
    n_units = len(unique_units)
    sigma2_e = float(np.sum(resid_fe ** 2) / max(n - n_units - k, 1))

    # sigma2_u from between variation
    T_counts = np.array([np.sum(units == u) for u in unique_units])
    T_bar = np.mean(T_counts)

    unit_means_Y = np.array([np.mean(Y[units == u]) for u in unique_units])
    unit_means_X = np.array([np.mean(X[units == u], axis=0) for u in unique_units])
    X_b = sm.add_constant(unit_means_X)
    try:
        between = sm.OLS(unit_means_Y, X_b).fit(disp=False)
        sigma2_b = float(np.sum(between.resid ** 2) / max(n_units - k - 1, 1))
    except Exception:
        sigma2_b = sigma2_e

    sigma2_u = max(sigma2_b - sigma2_e / T_bar, 0)

    # Step 3: GLS transformation (quasi-demeaning)
    theta = {}
    for u in unique_units:
        T_i = np.sum(units == u)
        lam = 1 - np.sqrt(sigma2_e / (sigma2_e + T_i * sigma2_u)) if sigma2_u > 0 else 0
        theta[str(u)] = lam

    Y_re = Y.copy()
    X_re = X.copy()
    ones_re = np.ones(n)
    for u in unique_units:
        mask = units == u
        lam = theta[str(u)]
        Y_re[mask] = Y[mask] - lam * np.mean(Y[mask])
        X_re[mask] = X[mask] - lam * np.mean(X[mask], axis=0)
        ones_re[mask] = 1 - lam

    X_re_c = np.column_stack([ones_re, X_re])
    try:
        XtX = X_re_c.T @ X_re_c
        XtY = X_re_c.T @ Y_re
        beta_re = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Random effects GLS estimation failed.")

    resid_re = Y_re - X_re_c @ beta_re
    dof = n - k - 1
    dof = max(dof, 1)
    sigma2_re = float(np.sum(resid_re ** 2) / dof)
    vcov = sigma2_re * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(vcov))

    # R-squared (overall)
    Y_hat = X_c @ np.concatenate([[beta_re[0]], beta_re[1:]])
    ss_res = np.sum((Y - Y_hat) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2_overall = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Coefficients (skip constant at index 0)
    coefficients = {}
    ci_mult = sp_stats.norm.ppf(0.975)
    for i, name in enumerate(pred_names):
        idx = i + 1  # skip constant
        z_val = float(beta_re[idx] / se[idx]) if se[idx] > 0 else 0.0
        p_val = float(2 * sp_stats.norm.sf(abs(z_val)))
        coefficients[name] = {
            "estimate": float(beta_re[idx]),
            "se": float(se[idx]),
            "z": z_val,
            "p": p_val,
            "ci": [float(beta_re[idx] - ci_mult * se[idx]),
                   float(beta_re[idx] + ci_mult * se[idx])],
        }

    return {
        "coefficients": coefficients,
        "intercept": float(beta_re[0]),
        "r_squared_overall": r2_overall,
        "sigma2_e": sigma2_e,
        "sigma2_u": sigma2_u,
        "_beta": beta_re[1:].tolist(),
        "_vcov": vcov[1:, 1:].tolist(),
    }


def _wooldridge_test(
    Y: np.ndarray, X: np.ndarray, units: np.ndarray,
    time: np.ndarray, unique_units: np.ndarray
) -> dict:
    """Wooldridge test for serial correlation in panel data (first-difference residuals)."""
    # First-difference the data within each unit
    fd_resids = []
    for u in unique_units:
        mask = units == u
        t_u = time[mask]
        y_u = Y[mask]
        x_u = X[mask]
        order = np.argsort(t_u)
        y_u = y_u[order]
        x_u = x_u[order]
        if len(y_u) < 3:
            continue
        dy = np.diff(y_u)
        dx = np.diff(x_u, axis=0)
        if dx.shape[0] < 2:
            continue
        # Residuals from FD regression
        dx_c = sm.add_constant(dx)
        try:
            res = sm.OLS(dy, dx_c).fit(disp=False)
            fd_resids.append(res.resid)
        except Exception:
            continue

    if len(fd_resids) < 2:
        return {"test_statistic": None, "p_value": None, "note": "Insufficient data"}

    # Test: regress residual_t on residual_{t-1}
    all_r1 = []
    all_r0 = []
    for r in fd_resids:
        if len(r) >= 2:
            all_r1.extend(r[1:].tolist())
            all_r0.extend(r[:-1].tolist())

    if len(all_r1) < 5:
        return {"test_statistic": None, "p_value": None, "note": "Insufficient data"}

    r1 = np.array(all_r1)
    r0 = np.array(all_r0)
    try:
        rho_model = sm.OLS(r1, sm.add_constant(r0)).fit(disp=False)
        f_stat = float(rho_model.tvalues[1] ** 2)
        p_val = float(2 * sp_stats.t.sf(abs(rho_model.tvalues[1]), rho_model.df_resid))
    except Exception:
        return {"test_statistic": None, "p_value": None, "note": "Test computation failed"}

    return {
        "test_statistic": f_stat,
        "p_value": p_val,
        "interpretation": "No serial correlation" if p_val > 0.05
                          else "Serial correlation detected",
    }


def _panel_hetero_test(
    Y: np.ndarray, X: np.ndarray, units: np.ndarray,
    unique_units: np.ndarray
) -> dict:
    """Modified Wald test for groupwise heteroskedasticity in FE model."""
    k = X.shape[1]
    n = len(Y)

    # Get FE residuals
    Y_dm = Y.copy()
    X_dm = X.copy()
    for u in unique_units:
        mask = units == u
        Y_dm[mask] -= np.mean(Y[mask])
        X_dm[mask] -= np.mean(X[mask], axis=0)

    try:
        beta = np.linalg.solve(X_dm.T @ X_dm, X_dm.T @ Y_dm)
    except np.linalg.LinAlgError:
        return {"test_statistic": None, "p_value": None, "note": "Estimation failed"}

    resid = Y_dm - X_dm @ beta
    sigma2_overall = np.sum(resid ** 2) / max(n - len(unique_units) - k, 1)

    # Unit-specific variances
    chi2 = 0.0
    n_units = len(unique_units)
    for u in unique_units:
        mask = units == u
        T_i = np.sum(mask)
        if T_i < 2:
            continue
        sigma2_i = np.sum(resid[mask] ** 2) / max(T_i - 1, 1)
        chi2 += (T_i - 1) * (np.log(sigma2_overall) - np.log(max(sigma2_i, 1e-12)))

    chi2 = abs(float(chi2))
    p_val = float(sp_stats.chi2.sf(chi2, n_units - 1))

    return {
        "test_statistic": chi2,
        "df": n_units - 1,
        "p_value": p_val,
        "interpretation": "Homoskedasticity" if p_val > 0.05
                          else "Heteroskedasticity detected — consider robust SEs",
    }


# ══════════════════════════════════════════════════════════════════════
# 5. Vote / Survey Scaling
# ══════════════════════════════════════════════════════════════════════

@router.post("/vote-scaling")
def vote_scaling(req: VoteScalingRequest):
    """Vote/survey scaling via tetrachoric correlation + PCA/factor analysis."""
    R = np.array(req.responses, dtype=float)
    n_resp, n_items = R.shape

    if n_resp < 5:
        raise HTTPException(400, "Need at least 5 respondents.")
    if n_items < 2:
        raise HTTPException(400, "Need at least 2 items.")
    if req.n_dimensions < 1 or req.n_dimensions > min(n_resp, n_items):
        raise HTTPException(400, f"n_dimensions must be between 1 and {min(n_resp, n_items)}.")

    d = req.n_dimensions
    item_labels = req.item_labels or [f"Item_{j}" for j in range(n_items)]
    if len(item_labels) != n_items:
        raise HTTPException(400, "item_labels length must match number of columns.")

    # ── Tetrachoric / polychoric correlation matrix (approximation) ─
    # For binary data, use Pearson phi + correction; for ordinal, use Pearson as proxy
    is_binary = np.all(np.isin(R, [0, 1]))

    if is_binary:
        corr_matrix = _tetrachoric_correlation(R)
    else:
        # Polychoric approximation: Pearson on standardised ranks
        from scipy.stats import rankdata
        R_ranked = np.apply_along_axis(lambda x: rankdata(x, method="average"), axis=0, arr=R)
        R_norm = sp_stats.norm.ppf(R_ranked / (n_resp + 1))
        corr_matrix = np.corrcoef(R_norm, rowvar=False)

    # Ensure positive semi-definite
    eigvals_check, eigvecs_check = np.linalg.eigh(corr_matrix)
    eigvals_check = np.maximum(eigvals_check, 1e-8)
    corr_matrix = eigvecs_check @ np.diag(eigvals_check) @ eigvecs_check.T
    # Re-normalise to correlation
    diag_inv_sqrt = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
    corr_matrix = diag_inv_sqrt @ corr_matrix @ diag_inv_sqrt

    # ── Eigendecomposition (PCA) ──────────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Loadings (top d)
    loadings = eigvecs[:, :d] * np.sqrt(np.maximum(eigvals[:d], 0))[np.newaxis, :]

    # Scores: project standardised responses onto components
    R_std = (R - np.mean(R, axis=0)) / np.maximum(np.std(R, axis=0, ddof=1), 1e-12)
    scores = R_std @ eigvecs[:, :d]

    # Variance explained
    total_var = np.sum(np.maximum(eigvals, 0))
    var_explained = [float(eigvals[k] / total_var) if total_var > 0 else 0.0 for k in range(d)]
    cumulative_var = [float(np.sum(eigvals[:k + 1]) / total_var) if total_var > 0 else 0.0
                      for k in range(d)]

    # ── Item fit (communalities) ──────────────────────────────────
    communalities = np.sum(loadings ** 2, axis=1)

    item_fit = []
    for j in range(n_items):
        item_fit.append({
            "label": item_labels[j],
            "communality": float(communalities[j]),
            "uniqueness": float(1 - communalities[j]),
            "loadings": [float(loadings[j, k]) for k in range(d)],
        })

    # ── Polarisation metrics ──────────────────────────────────────
    dim1 = scores[:, 0]
    skew = float(sp_stats.skew(dim1))
    kurt = float(sp_stats.kurtosis(dim1, fisher=True))
    n_ = len(dim1)
    bc = (skew ** 2 + 1) / (kurt + 3 * (n_ - 1) ** 2 / ((n_ - 2) * (n_ - 3))) if n_ > 3 else 0
    # Dip test approximation: variance ratio of two halves
    median_val = np.median(dim1)
    lower = dim1[dim1 <= median_val]
    upper = dim1[dim1 > median_val]
    var_ratio = (np.var(lower) + np.var(upper)) / (2 * np.var(dim1)) if np.var(dim1) > 0 else 1

    polarization = {
        "bimodality_coefficient": float(np.clip(bc, 0, 1)),
        "variance_ratio": float(var_ratio),
        "skewness": skew,
        "kurtosis": kurt,
        "interpretation": "Polarised" if bc > 0.555 else "Unimodal / moderate",
    }

    # ── Respondent scores ─────────────────────────────────────────
    respondent_scores = []
    for i in range(n_resp):
        if d == 1:
            respondent_scores.append(float(scores[i, 0]))
        else:
            respondent_scores.append([float(scores[i, k]) for k in range(d)])

    return {
        "dimension_scores": respondent_scores,
        "item_loadings": item_fit,
        "eigenvalues": [float(e) for e in eigvals[:min(d + 3, len(eigvals))]],
        "variance_explained": var_explained,
        "cumulative_variance_explained": cumulative_var,
        "polarization": polarization,
        "n_respondents": n_resp,
        "n_items": n_items,
        "dimensions": d,
        "correlation_type": "tetrachoric" if is_binary else "polychoric_approx",
    }


def _tetrachoric_correlation(R: np.ndarray) -> np.ndarray:
    """Approximate tetrachoric correlation for binary data using Digby (1983)."""
    n_items = R.shape[1]
    n = R.shape[0]
    corr = np.eye(n_items)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            # 2x2 table
            a = np.sum((R[:, i] == 1) & (R[:, j] == 1))  # (1,1)
            b = np.sum((R[:, i] == 1) & (R[:, j] == 0))  # (1,0)
            c = np.sum((R[:, i] == 0) & (R[:, j] == 1))  # (0,1)
            d = np.sum((R[:, i] == 0) & (R[:, j] == 0))  # (0,0)

            # Add 0.5 continuity correction
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

            # Digby's approximation
            ad = a * d
            bc = b * c
            if ad + bc == 0:
                r_tet = 0.0
            else:
                odds = ad / bc if bc > 0 else 1e6
                # cos(pi / (1 + odds^(pi/4)))  approximation
                r_tet = np.cos(np.pi / (1 + odds ** (np.pi / 4)))

            corr[i, j] = r_tet
            corr[j, i] = r_tet

    return corr
