"""Mediation & Moderation — PROCESS-style analysis (Models 1, 4, 6, 7).

Full bootstrap-based mediation, parallel mediation, serial mediation,
moderation with Johnson-Neyman, and moderated mediation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class SimpleMediationRequest(BaseModel):
    """PROCESS Model 4 — simple mediation: X -> M -> Y."""
    x: list[float]
    m: list[float]
    y: list[float]
    covariates: list[list[float]] | None = None
    covariate_names: list[str] | None = None
    n_bootstrap: int = 5000
    confidence: float = 0.95
    seed: int | None = 20240101  # fixed by default so results are reproducible


class ParallelMediationRequest(BaseModel):
    """PROCESS Model 4 (multiple) — parallel mediators: X -> M1,M2,...Mk -> Y."""
    x: list[float]
    mediators: list[list[float]]
    y: list[float]
    mediator_names: list[str] | None = None
    covariates: list[list[float]] | None = None
    n_bootstrap: int = 5000
    confidence: float = 0.95
    seed: int | None = 20240101  # fixed by default so results are reproducible


class SerialMediationRequest(BaseModel):
    """PROCESS Model 6 — serial mediation: X -> M1 -> M2 -> Y."""
    x: list[float]
    m1: list[float]
    m2: list[float]
    y: list[float]
    n_bootstrap: int = 5000
    confidence: float = 0.95
    seed: int | None = 20240101  # fixed by default so results are reproducible


class ModerationRequest(BaseModel):
    """PROCESS Model 1 — simple moderation: X*W interaction on Y."""
    x: list[float]
    w: list[float]
    y: list[float]
    covariates: list[list[float]] | None = None
    probe_values: list[float] | None = None
    johnson_neyman: bool = True
    n_bootstrap: int = 5000
    seed: int | None = 20240101  # fixed by default so results are reproducible


class ModeratedMediationRequest(BaseModel):
    """PROCESS Model 7 — moderated mediation (first-stage): W moderates X->M."""
    x: list[float]
    m: list[float]
    w: list[float]
    y: list[float]
    probe_values: list[float] | None = None
    n_bootstrap: int = 5000
    confidence: float = 0.95
    seed: int | None = 20240101  # fixed by default so results are reproducible


# ── Helpers ───────────────────────────────────────────────────────────

def _validate_lengths(*arrays, min_n: int = 10):
    """Validate that all arrays are the same length and meet minimum n."""
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"All variables must have the same length. Got lengths: {lengths}",
        )
    if lengths[0] < min_n:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_n} observations, got {lengths[0]}.",
        )


def _ols_fit(y: np.ndarray, X: np.ndarray):
    """Fit OLS with constant already included in X. Returns results object."""
    try:
        model = sm.OLS(y, X).fit()
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OLS fitting failed: {str(e)}")


def _build_design(x: np.ndarray, *extra_cols) -> np.ndarray:
    """Stack columns and add intercept.  x is always first predictor."""
    cols = [x]
    for c in extra_cols:
        if c is not None:
            if c.ndim == 1:
                cols.append(c)
            else:
                for j in range(c.shape[1]):
                    cols.append(c[:, j])
    X = np.column_stack(cols)
    return sm.add_constant(X)


def _path_summary(model, param_idx: int) -> dict:
    """Extract coeff, SE, t, p for a single parameter index."""
    return {
        "coeff": float(model.params[param_idx]),
        "se": float(model.bse[param_idx]),
        "t": float(model.tvalues[param_idx]),
        "p": float(model.pvalues[param_idx]),
    }


def _percentile_ci(boot: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1 - confidence
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _bias_corrected_ci(boot: np.ndarray, point_est: float, confidence: float) -> tuple[float, float]:
    """Bias-corrected (BC) bootstrap confidence interval."""
    z0 = sp_stats.norm.ppf(np.mean(boot < point_est))
    z_alpha = sp_stats.norm.ppf((1 - confidence) / 2)
    lo_p = sp_stats.norm.cdf(2 * z0 + z_alpha) * 100
    hi_p = sp_stats.norm.cdf(2 * z0 - z_alpha) * 100
    lo_p = np.clip(lo_p, 0.0, 100.0)
    hi_p = np.clip(hi_p, 0.0, 100.0)
    return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))


def _batched_coef(Y: np.ndarray, X: np.ndarray, j: int) -> np.ndarray:
    """OLS coefficient j for B resamples at once: Y (B, n), X (B, n, k). NaN where singular."""
    XtX = np.einsum("bni,bnj->bij", X, X)
    Xty = np.einsum("bni,bn->bi", X, Y)
    out = np.full(Y.shape[0], np.nan)
    ok = np.abs(np.linalg.det(XtX)) > 1e-12
    if ok.any():
        out[ok] = np.linalg.solve(XtX[ok], Xty[ok][..., None])[:, j, 0]
    return out


def _sobel_test(a: float, b: float, se_a: float, se_b: float) -> dict:
    """Sobel test for indirect effect significance."""
    se_ab = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    z = (a * b) / se_ab if se_ab > 0 else 0.0
    p = 2 * (sp_stats.norm.sf(abs(z)))
    return {"z": float(z), "p": float(p), "se": float(se_ab)}


def _prepare_covariates(covariates: list[list[float]] | None, n: int) -> np.ndarray | None:
    """Convert covariate lists to numpy matrix, or None."""
    if covariates is None or len(covariates) == 0:
        return None
    cov = np.array(covariates, dtype=float).T  # shape (n, k)
    if cov.shape[0] != n:
        raise HTTPException(
            status_code=422,
            detail=f"Covariates have {cov.shape[0]} rows but expected {n}.",
        )
    return cov


# ── 1. Simple Mediation (PROCESS Model 4) ────────────────────────────

@router.post("/simple-mediation")
def simple_mediation(req: SimpleMediationRequest):
    """Full mediation analysis: X -> M -> Y with bootstrap CIs."""
    x = np.array(req.x, dtype=float)
    m = np.array(req.m, dtype=float)
    y = np.array(req.y, dtype=float)
    _validate_lengths(x, m, y)
    n = len(x)
    cov = _prepare_covariates(req.covariates, n)
    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_boot = max(1000, min(req.n_bootstrap, 50000))

    # --- OLS regressions ---
    # Path a: M = i + a*X (+ covariates)
    Xa = _build_design(x, cov)
    model_m = _ols_fit(m, Xa)
    a = model_m.params[1]
    se_a = model_m.bse[1]

    # Path b & c': Y = i + c'*X + b*M (+ covariates)
    Xbc = _build_design(x, m, cov)
    model_y = _ols_fit(y, Xbc)
    c_prime = model_y.params[1]
    b = model_y.params[2]
    se_b = model_y.bse[2]

    # Path c (total): Y = i + c*X (+ covariates)
    Xc = _build_design(x, cov)
    model_total = _ols_fit(y, Xc)

    indirect = a * b

    # --- Bootstrap ---
    rng = np.random.default_rng(req.seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    a_boot = _batched_coef(m[idx], Xa[idx], 1)
    b_boot = _batched_coef(y[idx], Xbc[idx], 2)
    boot_indirect = a_boot * b_boot

    boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
    if len(boot_indirect) < n_boot * 0.5:
        raise HTTPException(status_code=500, detail="Too many bootstrap samples failed.")

    boot_se = float(np.std(boot_indirect, ddof=1))
    pct_ci = _percentile_ci(boot_indirect, confidence)
    bc_ci = _bias_corrected_ci(boot_indirect, indirect, confidence)

    # Sobel
    sobel = _sobel_test(a, b, se_a, se_b)

    # Proportion mediated
    total_c = model_total.params[1]
    prop_mediated = float(indirect / total_c) if abs(total_c) > 1e-10 else None

    # Covariate labels
    cov_names = req.covariate_names
    if cov is not None and cov_names is None:
        cov_names = [f"Cov{j+1}" for j in range(cov.shape[1])]

    return {
        "model": "simple_mediation_process_model_4",
        "n": n,
        "path_a": _path_summary(model_m, 1),
        "path_b": _path_summary(model_y, 2),
        "path_c_total": _path_summary(model_total, 1),
        "path_c_prime_direct": _path_summary(model_y, 1),
        "indirect_effect": {
            "point_estimate": float(indirect),
            "bootstrap_se": boot_se,
            "percentile_ci": list(pct_ci),
            "bias_corrected_ci": list(bc_ci),
            "significant": bool(
                (pct_ci[0] > 0 and pct_ci[1] > 0) or (pct_ci[0] < 0 and pct_ci[1] < 0)
            ),
        },
        "proportion_mediated": prop_mediated,
        "sobel_test": sobel,
        "r_squared_m_model": float(model_m.rsquared),
        "r_squared_y_model": float(model_y.rsquared),
        "r_squared_total_model": float(model_total.rsquared),
        "n_bootstrap": len(boot_indirect),
        "confidence_level": confidence,
        "covariate_names": cov_names,
    }


# ── 2. Parallel Mediation (PROCESS Model 4, multiple mediators) ──────

@router.post("/parallel-mediation")
def parallel_mediation(req: ParallelMediationRequest):
    """Multiple parallel mediators: X -> M1, M2, ... Mk -> Y."""
    x = np.array(req.x, dtype=float)
    y = np.array(req.y, dtype=float)
    mediators_raw = [np.array(mi, dtype=float) for mi in req.mediators]
    k = len(mediators_raw)
    if k < 2:
        raise HTTPException(status_code=422, detail="Parallel mediation requires at least 2 mediators.")

    all_arrays = [x, y] + mediators_raw
    _validate_lengths(*all_arrays)
    n = len(x)
    cov = _prepare_covariates(req.covariates, n)
    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_boot = max(1000, min(req.n_bootstrap, 50000))

    med_names = req.mediator_names or [f"M{j+1}" for j in range(k)]
    if len(med_names) != k:
        raise HTTPException(status_code=422, detail="mediator_names length must match number of mediators.")

    M = np.column_stack(mediators_raw)  # (n, k)

    # --- OLS: a-paths (X -> each M_j) ---
    a_models = []
    for j in range(k):
        Xa = _build_design(x, cov)
        a_models.append(_ols_fit(M[:, j], Xa))

    # --- OLS: b-paths + direct (Y = c' X + b1 M1 + ... + bk Mk + covariates) ---
    Xfull = _build_design(x, M, cov)
    model_y = _ols_fit(y, Xfull)

    # --- OLS: total effect (Y = c X + covariates) ---
    Xc = _build_design(x, cov)
    model_total = _ols_fit(y, Xc)

    # Point estimates
    a_coefs = [float(a_models[j].params[1]) for j in range(k)]
    b_coefs = [float(model_y.params[2 + j]) for j in range(k)]
    specific_indirect = [a_coefs[j] * b_coefs[j] for j in range(k)]
    total_indirect = sum(specific_indirect)
    direct_effect = float(model_y.params[1])
    total_effect = float(model_total.params[1])

    # --- Bootstrap ---
    rng = np.random.default_rng(req.seed)
    boot_specific = np.empty((n_boot, k))
    boot_total_indirect = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb, Mb = x[idx], y[idx], M[idx]
        covb = cov[idx] if cov is not None else None
        try:
            a_b = np.empty(k)
            for j in range(k):
                Xa_b = _build_design(xb, covb)
                a_b[j] = sm.OLS(Mb[:, j], Xa_b).fit().params[1]
            Xf_b = _build_design(xb, Mb, covb)
            ym = sm.OLS(yb, Xf_b).fit()
            b_b = np.array([ym.params[2 + j] for j in range(k)])
            boot_specific[i] = a_b * b_b
            boot_total_indirect[i] = np.sum(boot_specific[i])
        except Exception:
            boot_specific[i] = np.nan
            boot_total_indirect[i] = np.nan

    # Drop failed rows
    valid = ~np.isnan(boot_total_indirect)
    if np.sum(valid) < n_boot * 0.5:
        raise HTTPException(status_code=500, detail="Too many bootstrap samples failed.")
    boot_specific = boot_specific[valid]
    boot_total_indirect = boot_total_indirect[valid]

    # Specific indirect effects
    specific_results = []
    for j in range(k):
        pct = _percentile_ci(boot_specific[:, j], confidence)
        bc = _bias_corrected_ci(boot_specific[:, j], specific_indirect[j], confidence)
        specific_results.append({
            "mediator": med_names[j],
            "a_path": _path_summary(a_models[j], 1),
            "b_path": _path_summary(model_y, 2 + j),
            "indirect_effect": float(specific_indirect[j]),
            "bootstrap_se": float(np.std(boot_specific[:, j], ddof=1)),
            "percentile_ci": list(pct),
            "bias_corrected_ci": list(bc),
            "significant": bool((pct[0] > 0 and pct[1] > 0) or (pct[0] < 0 and pct[1] < 0)),
        })

    # Pairwise contrasts
    contrasts = []
    for j1 in range(k):
        for j2 in range(j1 + 1, k):
            diff = specific_indirect[j1] - specific_indirect[j2]
            boot_diff = boot_specific[:, j1] - boot_specific[:, j2]
            pct = _percentile_ci(boot_diff, confidence)
            contrasts.append({
                "contrast": f"{med_names[j1]} vs {med_names[j2]}",
                "difference": float(diff),
                "bootstrap_se": float(np.std(boot_diff, ddof=1)),
                "percentile_ci": list(pct),
                "significant": bool((pct[0] > 0 and pct[1] > 0) or (pct[0] < 0 and pct[1] < 0)),
            })

    total_indirect_pct = _percentile_ci(boot_total_indirect, confidence)

    return {
        "model": "parallel_mediation_process_model_4",
        "n": n,
        "n_mediators": k,
        "specific_indirect_effects": specific_results,
        "total_indirect_effect": {
            "point_estimate": float(total_indirect),
            "bootstrap_se": float(np.std(boot_total_indirect, ddof=1)),
            "percentile_ci": list(total_indirect_pct),
            "significant": bool(
                (total_indirect_pct[0] > 0 and total_indirect_pct[1] > 0)
                or (total_indirect_pct[0] < 0 and total_indirect_pct[1] < 0)
            ),
        },
        "direct_effect": _path_summary(model_y, 1),
        "total_effect": _path_summary(model_total, 1),
        "contrasts": contrasts,
        "r_squared_y_model": float(model_y.rsquared),
        "r_squared_total_model": float(model_total.rsquared),
        "n_bootstrap_valid": int(np.sum(valid)),
        "confidence_level": float(confidence),
    }


# ── 3. Serial Mediation (PROCESS Model 6) ────────────────────────────

@router.post("/serial-mediation")
def serial_mediation(req: SerialMediationRequest):
    """Serial mediation: X -> M1 -> M2 -> Y (three indirect paths)."""
    x = np.array(req.x, dtype=float)
    m1 = np.array(req.m1, dtype=float)
    m2 = np.array(req.m2, dtype=float)
    y = np.array(req.y, dtype=float)
    _validate_lengths(x, m1, m2, y)
    n = len(x)
    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_boot = max(1000, min(req.n_bootstrap, 50000))

    # --- OLS models ---
    # Model M1: M1 = i + a1*X
    X_m1 = sm.add_constant(x)
    model_m1 = _ols_fit(m1, X_m1)
    a1 = model_m1.params[1]

    # Model M2: M2 = i + a2*X + d21*M1
    X_m2 = sm.add_constant(np.column_stack([x, m1]))
    model_m2 = _ols_fit(m2, X_m2)
    a2 = model_m2.params[1]
    d21 = model_m2.params[2]

    # Model Y: Y = i + c'*X + b1*M1 + b2*M2
    X_y = sm.add_constant(np.column_stack([x, m1, m2]))
    model_y = _ols_fit(y, X_y)
    c_prime = model_y.params[1]
    b1 = model_y.params[2]
    b2 = model_y.params[3]

    # Total effect: Y = i + c*X
    X_c = sm.add_constant(x)
    model_total = _ols_fit(y, X_c)

    # Indirect effects (point estimates)
    ind1 = a1 * b1          # X -> M1 -> Y
    ind2 = a2 * b2          # X -> M2 -> Y
    ind3 = a1 * d21 * b2    # X -> M1 -> M2 -> Y
    total_indirect = ind1 + ind2 + ind3

    # --- Bootstrap ---
    rng = np.random.default_rng(req.seed)
    boot = np.empty((n_boot, 3))  # ind1, ind2, ind3

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, m1b, m2b, yb = x[idx], m1[idx], m2[idx], y[idx]
        try:
            a1_b = sm.OLS(m1b, sm.add_constant(xb)).fit().params[1]
            m2_fit = sm.OLS(m2b, sm.add_constant(np.column_stack([xb, m1b]))).fit()
            a2_b, d21_b = m2_fit.params[1], m2_fit.params[2]
            y_fit = sm.OLS(yb, sm.add_constant(np.column_stack([xb, m1b, m2b]))).fit()
            b1_b, b2_b = y_fit.params[2], y_fit.params[3]
            boot[i, 0] = a1_b * b1_b
            boot[i, 1] = a2_b * b2_b
            boot[i, 2] = a1_b * d21_b * b2_b
        except Exception:
            boot[i] = np.nan

    valid = ~np.isnan(boot[:, 0])
    if np.sum(valid) < n_boot * 0.5:
        raise HTTPException(status_code=500, detail="Too many bootstrap samples failed.")
    boot = boot[valid]
    boot_total = boot.sum(axis=1)

    def _effect_result(name, point_est, boot_col):
        pct = _percentile_ci(boot_col, confidence)
        bc = _bias_corrected_ci(boot_col, point_est, confidence)
        return {
            "path": name,
            "point_estimate": float(point_est),
            "bootstrap_se": float(np.std(boot_col, ddof=1)),
            "percentile_ci": list(pct),
            "bias_corrected_ci": list(bc),
            "significant": bool((pct[0] > 0 and pct[1] > 0) or (pct[0] < 0 and pct[1] < 0)),
        }

    return {
        "model": "serial_mediation_process_model_6",
        "n": n,
        "model_m1": {
            "a1_x_to_m1": _path_summary(model_m1, 1),
            "r_squared": float(model_m1.rsquared),
        },
        "model_m2": {
            "a2_x_to_m2": _path_summary(model_m2, 1),
            "d21_m1_to_m2": _path_summary(model_m2, 2),
            "r_squared": float(model_m2.rsquared),
        },
        "model_y": {
            "c_prime_direct": _path_summary(model_y, 1),
            "b1_m1_to_y": _path_summary(model_y, 2),
            "b2_m2_to_y": _path_summary(model_y, 3),
            "r_squared": float(model_y.rsquared),
        },
        "total_effect": _path_summary(model_total, 1),
        "indirect_effects": [
            _effect_result("X -> M1 -> Y", ind1, boot[:, 0]),
            _effect_result("X -> M2 -> Y", ind2, boot[:, 1]),
            _effect_result("X -> M1 -> M2 -> Y", ind3, boot[:, 2]),
        ],
        "total_indirect_effect": _effect_result("total indirect", total_indirect, boot_total),
        "direct_effect": _path_summary(model_y, 1),
        "n_bootstrap_valid": int(np.sum(valid)),
        "confidence_level": float(confidence),
    }


# ── 4. Moderation (PROCESS Model 1) ──────────────────────────────────

@router.post("/moderation")
def moderation_analysis(req: ModerationRequest):
    """Simple moderation: X*W interaction predicting Y, with probing."""
    x = np.array(req.x, dtype=float)
    w = np.array(req.w, dtype=float)
    y = np.array(req.y, dtype=float)
    _validate_lengths(x, w, y)
    n = len(x)
    cov = _prepare_covariates(req.covariates, n)
    n_boot = max(1000, min(req.n_bootstrap, 50000))

    # Mean-center X and W for interpretability
    x_mean, w_mean = float(np.mean(x)), float(np.mean(w))
    x_c = x - x_mean
    w_c = w - w_mean
    xw = x_c * w_c

    # --- Main model: Y = b0 + b1*X + b2*W + b3*X*W (+ covariates) ---
    cols = [x_c, w_c, xw]
    if cov is not None:
        for j in range(cov.shape[1]):
            cols.append(cov[:, j])
    X_full = sm.add_constant(np.column_stack(cols))
    model_full = _ols_fit(y, X_full)

    # Model without interaction for R² change
    cols_no_int = [x_c, w_c]
    if cov is not None:
        for j in range(cov.shape[1]):
            cols_no_int.append(cov[:, j])
    X_no_int = sm.add_constant(np.column_stack(cols_no_int))
    model_no_int = _ols_fit(y, X_no_int)

    r2_change = float(model_full.rsquared - model_no_int.rsquared)

    # F-test for R² change
    df_num = 1  # one interaction term added
    df_den = model_full.df_resid
    if r2_change > 0 and (1 - model_full.rsquared) > 0:
        f_change = (r2_change / df_num) / ((1 - model_full.rsquared) / df_den)
        p_change = float(sp_stats.f.sf(f_change, df_num, df_den))
    else:
        f_change = 0.0
        p_change = 1.0

    # --- Simple slopes probing ---
    w_sd = float(np.std(w, ddof=1))
    if req.probe_values is not None:
        probe_raw = req.probe_values
    else:
        probe_raw = [w_mean - w_sd, w_mean, w_mean + w_sd]

    b1 = model_full.params[1]  # X
    b3 = model_full.params[3]  # X*W
    # Variance-covariance for simple slope SE
    vcov = model_full.cov_params()
    V11, V33, V13 = vcov[1, 1], vcov[3, 3], vcov[1, 3]

    simple_slopes = []
    for w_val in probe_raw:
        w_centered = w_val - w_mean
        slope = b1 + b3 * w_centered
        # SE of simple slope = sqrt(Var(b1) + w_c^2 * Var(b3) + 2*w_c*Cov(b1,b3))
        se_slope = np.sqrt(
            vcov[1, 1] + w_centered**2 * vcov[3, 3] + 2 * w_centered * vcov[1, 3]
        )
        t_val = slope / se_slope if se_slope > 0 else 0.0
        p_val = float(2 * sp_stats.t.sf(abs(t_val), model_full.df_resid))
        alpha = 0.05
        t_crit = sp_stats.t.ppf(1 - alpha / 2, model_full.df_resid)
        simple_slopes.append({
            "w_value": float(w_val),
            "w_centered": float(w_centered),
            "slope": float(slope),
            "se": float(se_slope),
            "t": float(t_val),
            "p": float(p_val),
            "ci_lower": float(slope - t_crit * se_slope),
            "ci_upper": float(slope + t_crit * se_slope),
        })

    # --- Johnson-Neyman ---
    jn_result = None
    if req.johnson_neyman:
        # Simple slope = b1 + b3 * w_c = 0 at significance boundary
        # We solve: t^2 = (b1 + b3*w_c)^2 / (V11 + w_c^2*V33 + 2*w_c*V13) = t_crit^2
        # Rearranging: (b1 + b3*w_c)^2 = t_crit^2 * (V11 + 2*V13*w_c + V33*w_c^2)
        # => (b3^2 - t_crit^2*V33)*w_c^2 + (2*b1*b3 - 2*t_crit^2*V13)*w_c + (b1^2 - t_crit^2*V11) = 0
        t_crit_jn = sp_stats.t.ppf(0.975, model_full.df_resid)

        A = b3**2 - t_crit_jn**2 * V33
        B = 2 * b1 * b3 - 2 * t_crit_jn**2 * V13
        C = b1**2 - t_crit_jn**2 * V11

        discriminant = B**2 - 4 * A * C
        jn_points = []
        if discriminant >= 0 and abs(A) > 1e-12:
            sqrt_disc = np.sqrt(discriminant)
            wc1 = (-B + sqrt_disc) / (2 * A)
            wc2 = (-B - sqrt_disc) / (2 * A)
            # Convert back to raw W scale
            for wc in sorted([wc1, wc2]):
                w_raw = wc + w_mean
                # Only report if within observed range (with small buffer)
                w_min, w_max = float(np.min(w)), float(np.max(w))
                buffer = 0.1 * (w_max - w_min)
                if w_min - buffer <= w_raw <= w_max + buffer:
                    slope_at = b1 + b3 * wc
                    jn_points.append({
                        "w_value": float(w_raw),
                        "w_centered": float(wc),
                        "slope_at_point": float(slope_at),
                    })

        jn_result = {
            "transition_points": jn_points,
            "n_points_found": len(jn_points),
            "interpretation": (
                f"The effect of X on Y transitions significance at W = "
                + ", ".join(f"{p['w_value']:.4f}" for p in jn_points)
                if jn_points
                else "No Johnson-Neyman transition point within the observed range of W."
            ),
        }

    # --- Visualization data: slopes across range of W ---
    w_range = np.linspace(float(np.min(w)), float(np.max(w)), 100)
    viz_slopes = []
    viz_lo = []
    viz_hi = []
    t_crit_viz = sp_stats.t.ppf(0.975, model_full.df_resid)
    for wv in w_range:
        wc = wv - w_mean
        slope = float(b1 + b3 * wc)
        se = float(np.sqrt(V11 + wc**2 * V33 + 2 * wc * V13))
        viz_slopes.append(slope)
        viz_lo.append(slope - t_crit_viz * se)
        viz_hi.append(slope + t_crit_viz * se)

    return {
        "model": "moderation_process_model_1",
        "n": n,
        "mean_centered": True,
        "coefficients": {
            "intercept": _path_summary(model_full, 0),
            "x": _path_summary(model_full, 1),
            "w": _path_summary(model_full, 2),
            "x_w_interaction": _path_summary(model_full, 3),
        },
        "r_squared": float(model_full.rsquared),
        "r_squared_change_interaction": r2_change,
        "f_change": float(f_change),
        "p_change": float(p_change),
        "simple_slopes": simple_slopes,
        "johnson_neyman": jn_result,
        "visualization": {
            "w_values": [float(v) for v in w_range],
            "slopes": viz_slopes,
            "lower_ci": viz_lo,
            "upper_ci": viz_hi,
        },
    }


# ── 5. Moderated Mediation (PROCESS Model 7) ─────────────────────────

@router.post("/moderated-mediation")
def moderated_mediation(req: ModeratedMediationRequest):
    """Moderated mediation (first stage): W moderates X -> M path."""
    x = np.array(req.x, dtype=float)
    m = np.array(req.m, dtype=float)
    w = np.array(req.w, dtype=float)
    y = np.array(req.y, dtype=float)
    _validate_lengths(x, m, w, y)
    n = len(x)
    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_boot = max(1000, min(req.n_bootstrap, 50000))

    # Mean-center W
    w_mean = float(np.mean(w))
    w_sd = float(np.std(w, ddof=1))
    w_c = w - w_mean

    # Probe values
    if req.probe_values is not None:
        probe_raw = req.probe_values
    else:
        probe_raw = [w_mean - w_sd, w_mean, w_mean + w_sd]

    # --- Stage 1: M = i + a1*X + a2*W + a3*X*W ---
    xw = x * w_c
    X_m = sm.add_constant(np.column_stack([x, w_c, xw]))
    model_m = _ols_fit(m, X_m)
    a1 = model_m.params[1]  # X
    a2 = model_m.params[2]  # W
    a3 = model_m.params[3]  # X*W

    # --- Stage 2: Y = i + c'*X + b*M ---
    X_y = sm.add_constant(np.column_stack([x, m]))
    model_y = _ols_fit(y, X_y)
    c_prime = model_y.params[1]
    b = model_y.params[2]

    # --- Total effect: Y = i + c*X ---
    X_c = sm.add_constant(x)
    model_total = _ols_fit(y, X_c)

    # Conditional indirect effects at each probe value
    # Indirect = (a1 + a3 * w_centered) * b
    conditional_indirect = []
    for wv in probe_raw:
        wc = wv - w_mean
        a_cond = a1 + a3 * wc
        ind = a_cond * b
        conditional_indirect.append({
            "w_value": float(wv),
            "a_conditional": float(a_cond),
            "indirect_effect": float(ind),
        })

    # Index of moderated mediation = a3 * b
    index_mm = a3 * b

    # --- Bootstrap ---
    rng = np.random.default_rng(req.seed)
    n_probes = len(probe_raw)
    boot_conditional = np.empty((n_boot, n_probes))
    boot_index = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, mb, wb, yb = x[idx], m[idx], w[idx], y[idx]
        wc_b = wb - w_mean
        xw_b = xb * wc_b
        try:
            m_fit = sm.OLS(mb, sm.add_constant(np.column_stack([xb, wc_b, xw_b]))).fit()
            a1_b, a3_b = m_fit.params[1], m_fit.params[3]
            y_fit = sm.OLS(yb, sm.add_constant(np.column_stack([xb, mb]))).fit()
            b_b = y_fit.params[2]
            boot_index[i] = a3_b * b_b
            for j, wv in enumerate(probe_raw):
                wc = wv - w_mean
                boot_conditional[i, j] = (a1_b + a3_b * wc) * b_b
        except Exception:
            boot_index[i] = np.nan
            boot_conditional[i] = np.nan

    valid = ~np.isnan(boot_index)
    if np.sum(valid) < n_boot * 0.5:
        raise HTTPException(status_code=500, detail="Too many bootstrap samples failed.")
    boot_index = boot_index[valid]
    boot_conditional = boot_conditional[valid]

    # Index of moderated mediation CI
    idx_pct = _percentile_ci(boot_index, confidence)
    idx_bc = _bias_corrected_ci(boot_index, index_mm, confidence)

    # Conditional indirect CIs
    for j, entry in enumerate(conditional_indirect):
        pct = _percentile_ci(boot_conditional[:, j], confidence)
        bc = _bias_corrected_ci(boot_conditional[:, j], entry["indirect_effect"], confidence)
        entry["bootstrap_se"] = float(np.std(boot_conditional[:, j], ddof=1))
        entry["percentile_ci"] = list(pct)
        entry["bias_corrected_ci"] = list(bc)
        entry["significant"] = bool(
            (pct[0] > 0 and pct[1] > 0) or (pct[0] < 0 and pct[1] < 0)
        )

    return {
        "model": "moderated_mediation_process_model_7",
        "n": n,
        "stage1_m_model": {
            "x": _path_summary(model_m, 1),
            "w": _path_summary(model_m, 2),
            "x_w_interaction": _path_summary(model_m, 3),
            "r_squared": float(model_m.rsquared),
        },
        "stage2_y_model": {
            "x_direct": _path_summary(model_y, 1),
            "m_to_y": _path_summary(model_y, 2),
            "r_squared": float(model_y.rsquared),
        },
        "total_effect": _path_summary(model_total, 1),
        "direct_effect": _path_summary(model_y, 1),
        "conditional_indirect_effects": conditional_indirect,
        "index_of_moderated_mediation": {
            "point_estimate": float(index_mm),
            "bootstrap_se": float(np.std(boot_index, ddof=1)),
            "percentile_ci": list(idx_pct),
            "bias_corrected_ci": list(idx_bc),
            "significant": bool(
                (idx_pct[0] > 0 and idx_pct[1] > 0) or (idx_pct[0] < 0 and idx_pct[1] < 0)
            ),
        },
        "n_bootstrap_valid": int(np.sum(valid)),
        "confidence_level": float(confidence),
    }
