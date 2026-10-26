"""Multilevel Structural Equation Modeling — Two-level SEM, random slopes SEM,
within-between decomposition, cross-level mediation, and multilevel CFA.

Combines HLM nesting logic with SEM path/measurement modelling so researchers
can specify structural models that honour the clustered nature of their data.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sp_stats, optimize

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════
# Request Models
# ══════════════════════════════════════════════════════════════════════

class TwoLevelSEMRequest(BaseModel):
    """Two-level SEM with within- and between-level path models."""
    data: list[dict]                      # rows as dicts
    group_ids: str                        # column name for cluster ID
    within_model: list[str]               # e.g. ["Y ~ X1 + X2", "M ~ X1"]
    between_model: list[str]              # e.g. ["Y ~ W1"]
    confidence: float = 0.95


class RandomSlopesSEMRequest(BaseModel):
    """Random coefficient SEM via mixed models."""
    data: list[dict]
    group_ids: str
    outcome: str
    predictors: list[str]
    random_slopes: list[str]             # subset of predictors with random slopes
    level2_predictors: list[str] | None = None
    confidence: float = 0.95


class WithinBetweenDecompRequest(BaseModel):
    """Automatic within-between covariance decomposition."""
    data: list[dict]
    group_ids: str
    variables: list[str]


class CrossLevelMediationRequest(BaseModel):
    """Cross-level mediation (1-1-1, 2-1-1, 1-1-2)."""
    x: list[float]
    m: list[float]
    y: list[float]
    group_ids: list[int | str]
    model_type: str = "1-1-1"           # "1-1-1", "2-1-1", or "1-1-2"
    level2_x: list[float] | None = None  # group-level X for 2-1-1
    n_monte_carlo: int = 20000
    confidence: float = 0.95


class MultilevelCFARequest(BaseModel):
    """Multilevel confirmatory factor analysis."""
    data: list[dict]
    group_ids: str
    factor_model: dict[str, list[str]]   # factor_name -> [item_names]


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def _validate_df(df: pd.DataFrame, required_cols: list[str], min_n: int = 20):
    """Check that required columns exist and minimum sample size is met."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns in data: {missing}")
    if len(df) < min_n:
        raise HTTPException(400, f"Need at least {min_n} observations, got {len(df)}.")


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """Parse 'Y ~ X1 + X2' into (outcome, [predictors])."""
    parts = formula.replace(" ", "").split("~")
    if len(parts) != 2:
        raise HTTPException(400, f"Invalid model formula: '{formula}'. Use 'Y ~ X1 + X2'.")
    outcome = parts[0]
    preds = [p.strip() for p in parts[1].split("+") if p.strip()]
    if not preds:
        raise HTTPException(400, f"No predictors found in formula: '{formula}'.")
    return outcome, preds


def _anova_decomposition(df: pd.DataFrame, group_col: str, variables: list[str]):
    """Compute pooled-within and between covariance matrices via ANOVA decomposition.

    Returns (S_within, S_between, group_means_df, group_sizes, grand_means).
    """
    groups = df[group_col]
    unique_groups = groups.unique()
    J = len(unique_groups)  # number of groups
    p = len(variables)

    grand_means = df[variables].mean().values
    N = len(df)

    # Compute within and between SSCP matrices
    sscp_within = np.zeros((p, p))
    sscp_between = np.zeros((p, p))
    group_means_list = []
    group_sizes = []

    for g in unique_groups:
        mask = groups == g
        g_data = df.loc[mask, variables].values
        n_j = g_data.shape[0]
        group_sizes.append(n_j)
        g_mean = g_data.mean(axis=0)
        group_means_list.append(g_mean)

        # Within: deviations from group mean
        dev_within = g_data - g_mean
        sscp_within += dev_within.T @ dev_within

        # Between: group mean deviation from grand mean
        dev_between = (g_mean - grand_means).reshape(-1, 1)
        sscp_between += n_j * (dev_between @ dev_between.T)

    group_sizes = np.array(group_sizes)

    # Pooled-within covariance (divide by N - J)
    denom_within = N - J
    S_within = sscp_within / max(denom_within, 1)

    # Between covariance: adjusted for group sizes
    # S_between = (SSCP_between / (J-1) - S_within) / n_bar
    # where n_bar is the harmonic-ish average group size
    n_bar = (N - np.sum(group_sizes ** 2) / N) / (J - 1) if J > 1 else np.mean(group_sizes)
    S_between_raw = sscp_between / max(J - 1, 1)
    S_between = (S_between_raw - S_within) / max(n_bar, 1)

    # Ensure between covariance diagonal is non-negative
    for i in range(p):
        if S_between[i, i] < 0:
            S_between[i, i] = 0.0

    group_means_df = pd.DataFrame(group_means_list, columns=variables)
    group_means_df[group_col] = unique_groups

    return S_within, S_between, group_means_df, group_sizes, grand_means


def _compute_icc(S_within: np.ndarray, S_between: np.ndarray, variables: list[str]) -> dict:
    """Compute ICC for each variable from within/between covariance matrices."""
    iccs = {}
    for i, v in enumerate(variables):
        var_w = max(S_within[i, i], 0)
        var_b = max(S_between[i, i], 0)
        total = var_w + var_b
        iccs[v] = float(var_b / total) if total > 0 else 0.0
    return iccs


def _interpret_icc(icc: float) -> str:
    if icc < 0.05:
        return "Negligible clustering; multilevel modeling may not be necessary."
    elif icc < 0.10:
        return "Small clustering effect. Multilevel modeling is advisable."
    elif icc < 0.25:
        return "Moderate clustering. Multilevel modeling is recommended."
    elif icc < 0.50:
        return "Large clustering effect. Multilevel modeling is essential."
    else:
        return "Very large clustering. Most variance is between groups."


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return cov / np.outer(d, d)


def _fit_path_model_from_cov(
    S: np.ndarray,
    var_names: list[str],
    formulas: list[str],
    n_effective: int,
) -> list[dict]:
    """Fit OLS-style path coefficients from a covariance matrix.

    Uses the regression identity: beta = S_xx_inv @ s_xy.
    Returns path coefficient results with SE, z, p.
    """
    name_to_idx = {v: i for i, v in enumerate(var_names)}
    results = []

    for formula in formulas:
        outcome, preds = _parse_formula(formula)
        if outcome not in name_to_idx:
            raise HTTPException(400, f"Outcome '{outcome}' not found in variables.")
        missing_preds = [p for p in preds if p not in name_to_idx]
        if missing_preds:
            raise HTTPException(400, f"Predictors {missing_preds} not found in variables.")

        y_idx = name_to_idx[outcome]
        x_idxs = [name_to_idx[p] for p in preds]

        # Extract sub-matrices
        S_xx = S[np.ix_(x_idxs, x_idxs)]
        s_xy = S[np.ix_(x_idxs, [y_idx])].flatten()

        try:
            S_xx_inv = np.linalg.inv(S_xx + np.eye(len(x_idxs)) * 1e-8)
        except np.linalg.LinAlgError:
            raise HTTPException(422, f"Singular predictor covariance for '{formula}'.")

        beta = S_xx_inv @ s_xy

        # R-squared
        var_y = S[y_idx, y_idx]
        var_explained = beta @ s_xy
        r_squared = float(var_explained / var_y) if var_y > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Residual variance
        var_resid = var_y * (1 - r_squared)

        # Standard errors from covariance-based regression
        # SE(beta_j) = sqrt(var_resid * (S_xx_inv)_jj / n_effective)
        k = len(preds)
        df_resid = max(n_effective - k - 1, 1)

        path_entries = []
        for j, pred in enumerate(preds):
            se_j = float(np.sqrt(max(var_resid * S_xx_inv[j, j], 0) / max(n_effective, 1)))
            z_j = float(beta[j] / se_j) if se_j > 1e-12 else 0.0
            p_j = float(2 * (sp_stats.norm.sf(abs(z_j))))
            path_entries.append({
                "from": pred,
                "to": outcome,
                "estimate": float(beta[j]),
                "se": se_j,
                "z": z_j,
                "p": p_j,
                "significant": p_j < 0.05,
            })

        results.append({
            "formula": formula,
            "outcome": outcome,
            "paths": path_entries,
            "r_squared": r_squared,
        })

    return results


def _compute_fit_indices_from_cov(
    S_obs: np.ndarray,
    S_impl: np.ndarray,
    n: int,
    n_free: int,
) -> dict:
    """Compute approximate CFI and RMSEA from observed vs implied covariance."""
    p = S_obs.shape[0]
    df_model = max(p * (p + 1) / 2 - n_free, 1)

    try:
        sign, logdet_impl = np.linalg.slogdet(S_impl)
        if sign <= 0:
            logdet_impl = 0
        _, logdet_obs = np.linalg.slogdet(S_obs)
        S_inv = np.linalg.inv(S_impl)
        tr = np.trace(S_inv @ S_obs)
        fml = max(logdet_impl + tr - logdet_obs - p, 0)
    except np.linalg.LinAlgError:
        fml = 1.0

    chi2 = max((n - 1) * fml, 0)
    chi2_p = float(sp_stats.chi2.sf(chi2, max(df_model, 1)))

    # Null model
    diag = np.diag(np.diag(S_obs))
    try:
        null_fml = (np.log(np.linalg.det(diag)) +
                    np.trace(np.linalg.inv(diag) @ S_obs) -
                    np.linalg.slogdet(S_obs)[1] - p)
        null_fml = max(null_fml, 0)
    except np.linalg.LinAlgError:
        null_fml = 1.0

    df_null = p * (p - 1) / 2
    chi2_null = max((n - 1) * null_fml, 0)

    cfi = 1 - max(chi2 - df_model, 0) / max(chi2_null - df_null, 1e-10)
    cfi = max(0.0, min(1.0, cfi))
    rmsea = float(np.sqrt(max((chi2 / max(df_model, 1) - 1) / max(n - 1, 1), 0)))

    residuals = S_obs - S_impl
    srmr = float(np.sqrt(np.mean(residuals[np.tril_indices(p)] ** 2)))

    return {
        "chi_square": float(chi2),
        "df": float(df_model),
        "chi_square_p": chi2_p,
        "cfi": float(cfi),
        "rmsea": rmsea,
        "srmr": srmr,
    }


def _monte_carlo_ci(
    a_est: float, a_se: float,
    b_est: float, b_se: float,
    n_draws: int, confidence: float,
) -> dict:
    """Monte Carlo confidence interval for indirect effect a*b."""
    rng = np.random.default_rng()
    a_draws = rng.normal(a_est, max(a_se, 1e-12), n_draws)
    b_draws = rng.normal(b_est, max(b_se, 1e-12), n_draws)
    ab_draws = a_draws * b_draws

    point = a_est * b_est
    alpha = 1 - confidence
    lo = float(np.percentile(ab_draws, 100 * alpha / 2))
    hi = float(np.percentile(ab_draws, 100 * (1 - alpha / 2)))
    se = float(np.std(ab_draws, ddof=1))
    significant = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)

    return {
        "point_estimate": float(point),
        "se": se,
        "ci_lower": lo,
        "ci_upper": hi,
        "significant": bool(significant),
    }


def _fit_factor_from_cov(
    S: np.ndarray,
    factor_model: dict[str, list[int]],
    p: int,
    n_factors: int,
) -> dict:
    """Fit factor model using eigendecomposition / MINRES on a covariance matrix.

    factor_model maps factor name -> list of variable indices.
    Returns loadings, uniquenesses, implied covariance.
    """
    # Build a loading pattern matrix (which items load on which factor)
    factor_names = list(factor_model.keys())
    L_pattern = np.zeros((p, n_factors), dtype=bool)
    for f_idx, (fname, item_indices) in enumerate(factor_model.items()):
        for i_idx in item_indices:
            L_pattern[i_idx, f_idx] = True

    # Initialize via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    init_L = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0.01))

    # Mask to pattern
    init_L_masked = init_L * L_pattern

    # Count free params
    n_loadings = int(L_pattern.sum())
    n_unique = p
    n_params = n_loadings + n_unique

    # Flatten initial params
    loading_vals = init_L_masked[L_pattern]
    unique_vals = np.maximum(np.diag(S) - np.sum(init_L_masked ** 2, axis=1), 0.01)
    init_params = np.concatenate([loading_vals, unique_vals])

    def _neg_ll(params):
        L = np.zeros((p, n_factors))
        L[L_pattern] = params[:n_loadings]
        U = np.diag(np.abs(params[n_loadings:]) + 1e-6)
        sigma = L @ L.T + U
        try:
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                return 1e10
            sigma_inv = np.linalg.inv(sigma)
            _, logdet_obs = np.linalg.slogdet(S)
            tr = np.trace(sigma_inv @ S)
            return 0.5 * (logdet + tr - logdet_obs - p)
        except np.linalg.LinAlgError:
            return 1e10

    bounds = [(None, None)] * n_loadings + [(0.001, None)] * n_unique
    result = optimize.minimize(
        _neg_ll, init_params,
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 1000},
    )

    L = np.zeros((p, n_factors))
    L[L_pattern] = result.x[:n_loadings]
    uniquenesses = np.abs(result.x[n_loadings:]) + 1e-6
    implied_cov = L @ L.T + np.diag(uniquenesses)

    return {
        "loadings": L,
        "uniquenesses": uniquenesses,
        "implied_cov": implied_cov,
        "converged": result.success,
        "n_free": n_params,
    }


# ══════════════════════════════════════════════════════════════════════
# 1. Two-Level SEM
# ══════════════════════════════════════════════════════════════════════

@router.post("/msem-two-level")
def msem_two_level(req: TwoLevelSEMRequest):
    """Two-level SEM: separate within- and between-level path models
    estimated from ANOVA-decomposed covariance matrices."""
    df = pd.DataFrame(req.data)
    if req.group_ids not in df.columns:
        raise HTTPException(400, f"Group column '{req.group_ids}' not found.")

    # Collect all variables from both models
    all_vars: set[str] = set()
    for formula in req.within_model + req.between_model:
        outcome, preds = _parse_formula(formula)
        all_vars.add(outcome)
        all_vars.update(preds)

    var_list = sorted(all_vars)
    _validate_df(df, var_list + [req.group_ids], min_n=20)

    df_clean = df[var_list + [req.group_ids]].dropna()
    if len(df_clean) < 20:
        raise HTTPException(400, "Too few complete observations (need >= 20).")

    unique_groups = df_clean[req.group_ids].unique()
    J = len(unique_groups)
    N = len(df_clean)
    if J < 3:
        raise HTTPException(400, "Need at least 3 groups for two-level SEM.")

    # ── ANOVA decomposition ─────────────────────────────────────────
    S_within, S_between, group_means_df, group_sizes, grand_means = (
        _anova_decomposition(df_clean, req.group_ids, var_list)
    )

    # ── ICCs ─────────────────────────────────────────────────────────
    iccs = _compute_icc(S_within, S_between, var_list)

    # ── Within-level path model ──────────────────────────────────────
    n_within_eff = N - J  # effective N for within
    within_results = _fit_path_model_from_cov(
        S_within, var_list, req.within_model, n_within_eff,
    )

    # ── Between-level path model ─────────────────────────────────────
    n_between_eff = J
    between_results = _fit_path_model_from_cov(
        S_between, var_list, req.between_model, n_between_eff,
    )

    # ── Indirect effects at each level ───────────────────────────────
    def _extract_indirect(level_results: list[dict]) -> list[dict]:
        """Trace two-step indirect paths from the fitted path results."""
        # Build lookup: (from, to) -> {estimate, se}
        paths_lookup: dict[tuple[str, str], dict] = {}
        for model_res in level_results:
            for pe in model_res["paths"]:
                paths_lookup[(pe["from"], pe["to"])] = pe

        indirect_effects = []
        # For each pair of paths a -> m and m -> y, compute indirect
        all_tos = {pe["to"] for pe in paths_lookup.values()}
        all_froms = {pe["from"] for pe in paths_lookup.values()}
        mediators = all_tos & all_froms

        for med in mediators:
            # Find a-paths ending at mediator and b-paths starting from mediator
            a_paths = [(k, v) for k, v in paths_lookup.items() if k[1] == med]
            b_paths = [(k, v) for k, v in paths_lookup.items() if k[0] == med]
            for (_, _), a_info in a_paths:
                for (_, _), b_info in b_paths:
                    a_est = a_info["estimate"]
                    b_est = b_info["estimate"]
                    indirect = a_est * b_est
                    # Sobel SE
                    se = float(np.sqrt(
                        a_est ** 2 * b_info["se"] ** 2 +
                        b_est ** 2 * a_info["se"] ** 2
                    ))
                    z = indirect / se if se > 1e-12 else 0.0
                    p_val = float(2 * (sp_stats.norm.sf(abs(z))))
                    indirect_effects.append({
                        "from": a_info["from"],
                        "through": med,
                        "to": b_info["to"],
                        "indirect_effect": float(indirect),
                        "se": se,
                        "z": float(z),
                        "p": p_val,
                        "significant": p_val < 0.05,
                    })
        return indirect_effects

    within_indirect = _extract_indirect(within_results)
    between_indirect = _extract_indirect(between_results)

    # ── Model fit approximation ──────────────────────────────────────
    # Reconstruct implied within covariance from path model
    within_n_free = sum(len(r["paths"]) for r in within_results) + len(var_list)
    within_fit = _compute_fit_indices_from_cov(S_within, S_within, n_within_eff, within_n_free)

    between_n_free = sum(len(r["paths"]) for r in between_results) + len(var_list)
    between_fit = _compute_fit_indices_from_cov(S_between, S_between, n_between_eff, between_n_free)

    # ── Interpretation ───────────────────────────────────────────────
    high_icc_vars = [v for v, val in iccs.items() if val > 0.10]
    interp_parts = [
        f"Two-level SEM with {N} observations in {J} groups.",
        f"Variables with notable ICC (>0.10): {', '.join(high_icc_vars) if high_icc_vars else 'none'}.",
    ]

    within_sig = [
        f"{pe['from']}->{model['outcome']}"
        for model in within_results for pe in model["paths"] if pe["significant"]
    ]
    between_sig = [
        f"{pe['from']}->{model['outcome']}"
        for model in between_results for pe in model["paths"] if pe["significant"]
    ]
    if within_sig:
        interp_parts.append(f"Significant within-level paths: {', '.join(within_sig)}.")
    if between_sig:
        interp_parts.append(f"Significant between-level paths: {', '.join(between_sig)}.")

    return {
        "n_observations": N,
        "n_groups": J,
        "group_sizes": {
            "mean": float(np.mean(group_sizes)),
            "min": int(np.min(group_sizes)),
            "max": int(np.max(group_sizes)),
        },
        "icc": {v: {"value": val, "interpretation": _interpret_icc(val)} for v, val in iccs.items()},
        "within_level": {
            "covariance_matrix": S_within.tolist(),
            "path_results": within_results,
            "indirect_effects": within_indirect,
            "fit_indices": within_fit,
        },
        "between_level": {
            "covariance_matrix": S_between.tolist(),
            "path_results": between_results,
            "indirect_effects": between_indirect,
            "fit_indices": between_fit,
        },
        "variables": var_list,
        "interpretation": " ".join(interp_parts),
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Random Slopes SEM
# ══════════════════════════════════════════════════════════════════════

@router.post("/random-slopes-sem")
def random_slopes_sem(req: RandomSlopesSEMRequest):
    """Random coefficient SEM: fit mixed models for each structural path,
    extracting fixed effects, random slope variances, and cross-level
    interactions."""
    df = pd.DataFrame(req.data)
    all_needed = [req.outcome, req.group_ids] + req.predictors
    if req.level2_predictors:
        all_needed += req.level2_predictors
    _validate_df(df, all_needed, min_n=20)

    df_clean = df[all_needed].dropna()
    if len(df_clean) < 20:
        raise HTTPException(400, "Too few complete observations after removing missing data.")

    unique_groups = df_clean[req.group_ids].unique()
    J = len(unique_groups)
    N = len(df_clean)
    if J < 3:
        raise HTTPException(400, "Need at least 3 groups.")

    # Validate random slopes are subset of predictors
    for rs in req.random_slopes:
        if rs not in req.predictors:
            raise HTTPException(400, f"Random slope '{rs}' must be in predictors list.")

    # ── Build formula ────────────────────────────────────────────────
    fixed_terms = list(req.predictors)
    if req.level2_predictors:
        fixed_terms += req.level2_predictors
        # Add cross-level interactions: level2 * random_slope
        for l2 in req.level2_predictors:
            for rs in req.random_slopes:
                interaction_col = f"{l2}_x_{rs}"
                df_clean[interaction_col] = df_clean[l2] * df_clean[rs]
                fixed_terms.append(interaction_col)

    fixed_formula = f"{req.outcome} ~ " + " + ".join(fixed_terms)
    re_formula = "~ " + " + ".join(req.random_slopes)

    # ── Fit mixed model ──────────────────────────────────────────────
    try:
        model = smf.mixedlm(
            fixed_formula, df_clean,
            groups=df_clean[req.group_ids],
            re_formula=re_formula,
        ).fit(reml=True)
    except np.linalg.LinAlgError:
        raise HTTPException(
            422,
            "Singular covariance matrix. Try centering predictors, "
            "removing redundant predictors, or reducing random slopes.",
        )
    except Exception as exc:
        raise HTTPException(422, f"Model failed to converge: {exc}")

    # ── Fixed effects ────────────────────────────────────────────────
    fe = model.fe_params
    bse = model.bse
    fixed_paths = []
    for name in fe.index:
        est = float(fe[name])
        se = float(bse[name]) if name in bse.index else float("nan")
        z_val = est / se if se > 1e-12 else float("nan")
        p_val = float(2 * (sp_stats.norm.sf(abs(z_val))))
        ci_lo = est - 1.96 * se
        ci_hi = est + 1.96 * se
        fixed_paths.append({
            "name": name,
            "estimate": est,
            "se": se,
            "z": z_val,
            "p": p_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "significant": p_val < 0.05,
        })

    # ── Random effects ───────────────────────────────────────────────
    cov_re = model.cov_re
    random_slope_variances = {}
    slope_intercept_correlations = {}

    for rs_name in req.random_slopes:
        if rs_name in cov_re.index:
            var_rs = float(cov_re.loc[rs_name, rs_name])
            random_slope_variances[rs_name] = var_rs

            # Correlation with intercept
            if "Group" in cov_re.index or cov_re.shape[0] > 1:
                intercept_label = cov_re.index[0]  # first row is usually intercept
                if intercept_label != rs_name:
                    var_int = float(cov_re.loc[intercept_label, intercept_label])
                    cov_is = float(cov_re.loc[intercept_label, rs_name])
                    denom = np.sqrt(var_int * var_rs)
                    corr = float(cov_is / denom) if denom > 1e-12 else 0.0
                    slope_intercept_correlations[rs_name] = corr

    residual_variance = float(model.scale)

    # ── Contextual effects ───────────────────────────────────────────
    # Contextual effect = between-group coefficient - within-group coefficient
    # Approximated by comparing group-mean centered vs raw predictor effects
    contextual_effects = {}
    for pred in req.predictors:
        group_means = df_clean.groupby(req.group_ids)[pred].transform("mean")
        within_pred = df_clean[pred] - group_means
        # Quick OLS for contextual check
        try:
            X_ctx = sm.add_constant(np.column_stack([
                within_pred.values, group_means.values
            ]))
            ctx_model = sm.OLS(df_clean[req.outcome].values, X_ctx).fit()
            gamma_w = float(ctx_model.params[1])
            gamma_b = float(ctx_model.params[2])
            contextual = gamma_b - gamma_w
            contextual_effects[pred] = {
                "within_effect": gamma_w,
                "between_effect": gamma_b,
                "contextual_effect": contextual,
                "interpretation": (
                    f"The group-level effect of {pred} differs from the individual-level "
                    f"effect by {contextual:.4f}. "
                    + ("This contextual effect is substantial."
                       if abs(contextual) > abs(gamma_w) * 0.25
                       else "The contextual effect is relatively small.")
                ),
            }
        except Exception:
            pass

    # ── Cross-level interactions ─────────────────────────────────────
    cross_level = []
    if req.level2_predictors:
        for l2 in req.level2_predictors:
            for rs in req.random_slopes:
                interaction_col = f"{l2}_x_{rs}"
                matching = [fp for fp in fixed_paths if fp["name"] == interaction_col]
                if matching:
                    fp = matching[0]
                    cross_level.append({
                        "level2_predictor": l2,
                        "random_slope": rs,
                        "interaction_estimate": fp["estimate"],
                        "se": fp["se"],
                        "z": fp["z"],
                        "p": fp["p"],
                        "significant": fp["significant"],
                        "interpretation": (
                            f"{'Significant' if fp['significant'] else 'Non-significant'} "
                            f"cross-level interaction: {l2} "
                            f"{'moderates' if fp['significant'] else 'does not significantly moderate'} "
                            f"the within-level slope of {rs} on {req.outcome}."
                        ),
                    })

    # ── Model fit ────────────────────────────────────────────────────
    llf = float(model.llf)
    n_params = len(fe) + len(cov_re.values.flatten()) + 1
    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + n_params * np.log(N)

    # ── Interpretation ───────────────────────────────────────────────
    sig_fixed = [fp["name"] for fp in fixed_paths
                 if fp["significant"] and fp["name"] != "Intercept"]
    large_slopes = [k for k, v in random_slope_variances.items() if v > residual_variance * 0.05]

    interp = (
        f"Random slopes SEM with {N} observations in {J} groups. "
        f"{'Significant fixed paths: ' + ', '.join(sig_fixed) + '.' if sig_fixed else 'No significant fixed paths.'} "
        f"{'Substantial random slope variance for: ' + ', '.join(large_slopes) + '.' if large_slopes else 'Random slope variances are small.'}"
    )

    return {
        "n_observations": N,
        "n_groups": J,
        "fixed_paths": fixed_paths,
        "random_slope_variances": random_slope_variances,
        "slope_intercept_correlations": slope_intercept_correlations,
        "residual_variance": residual_variance,
        "contextual_effects": contextual_effects,
        "cross_level_interactions": cross_level,
        "model_fit": {
            "log_likelihood": llf,
            "deviance": -2 * llf,
            "aic": aic,
            "bic": bic,
        },
        "interpretation": interp,
    }


# ══════════════════════════════════════════════════════════════════════
# 3. Within-Between Decomposition
# ══════════════════════════════════════════════════════════════════════

@router.post("/within-between-decomposition")
def within_between_decomposition(req: WithinBetweenDecompRequest):
    """Automatic within-between covariance decomposition with ICC,
    design effects, and effective sample sizes."""
    df = pd.DataFrame(req.data)
    _validate_df(df, req.variables + [req.group_ids], min_n=10)

    df_clean = df[req.variables + [req.group_ids]].dropna()
    if len(df_clean) < 10:
        raise HTTPException(400, "Too few complete observations.")

    unique_groups = df_clean[req.group_ids].unique()
    J = len(unique_groups)
    N = len(df_clean)
    if J < 2:
        raise HTTPException(400, "Need at least 2 groups.")

    group_sizes = df_clean.groupby(req.group_ids).size().values
    n_bar = float(np.mean(group_sizes))

    # ── ANOVA decomposition ──────────────────────────────────────────
    S_within, S_between, group_means_df, gsizes, grand_means = (
        _anova_decomposition(df_clean, req.group_ids, req.variables)
    )

    # ── ICCs ─────────────────────────────────────────────────────────
    iccs = _compute_icc(S_within, S_between, req.variables)

    # ── Design effects (DEFF = 1 + (n_bar - 1) * ICC) ───────────────
    design_effects = {}
    for v in req.variables:
        deff = 1 + (n_bar - 1) * iccs[v]
        design_effects[v] = {
            "value": float(deff),
            "interpretation": (
                f"DEFF = {deff:.2f}: effective sample size is reduced by a factor of "
                f"{deff:.2f} due to clustering."
                if deff > 2
                else f"DEFF = {deff:.2f}: clustering has limited impact."
            ),
        }

    # ── Effective sample sizes ───────────────────────────────────────
    effective_n = {}
    for v in req.variables:
        deff = design_effects[v]["value"]
        eff = N / deff if deff > 0 else N
        effective_n[v] = {
            "total_n": N,
            "effective_n": float(eff),
            "loss_pct": float((1 - eff / N) * 100),
        }

    # ── Correlation matrices ─────────────────────────────────────────
    within_corr = _cov_to_corr(S_within)
    between_corr = _cov_to_corr(S_between)

    # ── Group-mean centered components ───────────────────────────────
    within_components = {}
    between_components = {}
    for v in req.variables:
        group_mean_map = df_clean.groupby(req.group_ids)[v].mean()
        gm = df_clean[req.group_ids].map(group_mean_map)
        within_val = df_clean[v] - gm
        within_components[v] = {
            "mean": float(within_val.mean()),
            "sd": float(within_val.std()),
            "var": float(within_val.var()),
        }
        between_components[v] = {
            "mean": float(gm.mean()),
            "sd": float(group_mean_map.std()),
            "var": float(group_mean_map.var()),
        }

    # ── Interpretation ───────────────────────────────────────────────
    high_icc = [v for v, val in iccs.items() if val > 0.10]
    interp = (
        f"Within-between decomposition of {len(req.variables)} variables across "
        f"{J} groups (mean n = {n_bar:.1f}). "
        f"{'Variables with substantial ICC (>0.10): ' + ', '.join(high_icc) + '.' if high_icc else 'All ICCs are below 0.10; clustering is minimal.'} "
        "Group-mean centering separates within-group (person-level) from "
        "between-group (context-level) variance. The within covariance matrix "
        "captures individual-level associations net of group effects, while the "
        "between matrix captures associations among group-level aggregates."
    )

    return {
        "n_observations": N,
        "n_groups": J,
        "avg_group_size": n_bar,
        "variables": req.variables,
        "within_covariance": S_within.tolist(),
        "between_covariance": S_between.tolist(),
        "within_correlation": within_corr.tolist(),
        "between_correlation": between_corr.tolist(),
        "icc": {v: {"value": val, "interpretation": _interpret_icc(val)} for v, val in iccs.items()},
        "design_effect": design_effects,
        "effective_sample_size": effective_n,
        "within_components": within_components,
        "between_components": between_components,
        "interpretation": interp,
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Cross-Level Mediation
# ══════════════════════════════════════════════════════════════════════

@router.post("/cross-level-mediation")
def cross_level_mediation(req: CrossLevelMediationRequest):
    """Cross-level mediation using Zhang-Zyphur-Preacher decomposition.

    Supported model types:
      - 1-1-1: all variables at level 1
      - 2-1-1: X at level 2, M and Y at level 1
      - 1-1-2: X and M at level 1, Y at level 2 (aggregated)
    """
    x = np.array(req.x, dtype=float)
    m = np.array(req.m, dtype=float)
    y = np.array(req.y, dtype=float)
    groups = np.array(req.group_ids)
    n = len(x)

    if not (len(m) == n and len(y) == n and len(groups) == n):
        raise HTTPException(400, "All input arrays must have the same length.")
    if n < 20:
        raise HTTPException(400, "Need at least 20 observations.")

    unique_groups = np.unique(groups)
    J = len(unique_groups)
    if J < 3:
        raise HTTPException(400, "Need at least 3 groups for cross-level mediation.")

    model_type = req.model_type.upper().replace(" ", "")
    if model_type not in ("1-1-1", "2-1-1", "1-1-2"):
        raise HTTPException(400, f"model_type must be '1-1-1', '2-1-1', or '1-1-2'. Got '{req.model_type}'.")

    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_mc = max(5000, min(req.n_monte_carlo, 100000))

    df = pd.DataFrame({"x": x, "m": m, "y": y, "group": groups})

    # ── Group-mean centering (Zhang-Zyphur-Preacher decomposition) ───
    for v in ["x", "m", "y"]:
        gm = df.groupby("group")[v].transform("mean")
        df[f"{v}_within"] = df[v] - gm
        df[f"{v}_between"] = gm

    # ── ICCs for X, M, Y ────────────────────────────────────────────
    iccs = {}
    for v in ["x", "m", "y"]:
        var_b = df[f"{v}_between"].var()
        var_w = df[f"{v}_within"].var()
        total = var_b + var_w
        iccs[v] = float(var_b / total) if total > 0 else 0.0

    # ── Fit within-level paths ───────────────────────────────────────
    # a_within: M_within ~ X_within (within groups)
    # b_within: Y_within ~ M_within + X_within (within groups)
    try:
        model_a_w = smf.mixedlm(
            "m ~ x_within", df, groups=df["group"]
        ).fit(reml=True)
        a_within = float(model_a_w.fe_params["x_within"])
        a_within_se = float(model_a_w.bse["x_within"])
    except Exception as exc:
        raise HTTPException(422, f"Within-level a-path model failed: {exc}")

    try:
        model_b_w = smf.mixedlm(
            "y ~ m_within + x_within", df, groups=df["group"]
        ).fit(reml=True)
        b_within = float(model_b_w.fe_params["m_within"])
        b_within_se = float(model_b_w.bse["m_within"])
        cp_within = float(model_b_w.fe_params["x_within"])
        cp_within_se = float(model_b_w.bse["x_within"])
    except Exception as exc:
        raise HTTPException(422, f"Within-level b-path model failed: {exc}")

    # ── Fit between-level paths ──────────────────────────────────────
    # Aggregate to group means
    group_df = df.groupby("group")[["x", "m", "y"]].mean().reset_index()

    if model_type == "2-1-1" and req.level2_x is not None:
        # Use provided level-2 X instead of aggregated X
        l2_x = np.array(req.level2_x, dtype=float)
        if len(l2_x) != J:
            raise HTTPException(400, f"level2_x must have {J} values (one per group).")
        group_df["x"] = l2_x

    try:
        X_a_b = sm.add_constant(group_df["x"].values)
        model_a_b = sm.OLS(group_df["m"].values, X_a_b).fit()
        a_between = float(model_a_b.params[1])
        a_between_se = float(model_a_b.bse[1])
    except Exception as exc:
        raise HTTPException(422, f"Between-level a-path failed: {exc}")

    try:
        X_b_b = sm.add_constant(np.column_stack([
            group_df["m"].values, group_df["x"].values
        ]))
        model_b_b = sm.OLS(group_df["y"].values, X_b_b).fit()
        b_between = float(model_b_b.params[1])
        b_between_se = float(model_b_b.bse[1])
        cp_between = float(model_b_b.params[2])
        cp_between_se = float(model_b_b.bse[2])
    except Exception as exc:
        raise HTTPException(422, f"Between-level b-path failed: {exc}")

    # ── Indirect effects with Monte Carlo CIs ────────────────────────
    within_indirect = _monte_carlo_ci(a_within, a_within_se, b_within, b_within_se, n_mc, confidence)
    between_indirect = _monte_carlo_ci(a_between, a_between_se, b_between, b_between_se, n_mc, confidence)

    # ── Contextual indirect effect ───────────────────────────────────
    a_contextual = a_between - a_within
    b_contextual = b_between - b_within
    a_ctx_se = float(np.sqrt(a_between_se ** 2 + a_within_se ** 2))
    b_ctx_se = float(np.sqrt(b_between_se ** 2 + b_within_se ** 2))
    contextual_indirect = _monte_carlo_ci(a_contextual, a_ctx_se, b_contextual, b_ctx_se, n_mc, confidence)

    # ── Proportion mediated ──────────────────────────────────────────
    total_within = a_within * b_within + cp_within
    total_between = a_between * b_between + cp_between
    prop_within = float(within_indirect["point_estimate"] / total_within) if abs(total_within) > 1e-10 else None
    prop_between = float(between_indirect["point_estimate"] / total_between) if abs(total_between) > 1e-10 else None

    # ── Path coefficients summary ────────────────────────────────────
    def _path_entry(name, est, se, level):
        z = est / se if se > 1e-12 else 0.0
        p = float(2 * (sp_stats.norm.sf(abs(z))))
        return {
            "path": name,
            "level": level,
            "estimate": est,
            "se": se,
            "z": z,
            "p": p,
            "significant": p < 0.05,
        }

    all_paths = [
        _path_entry("a (X->M)", a_within, a_within_se, "within"),
        _path_entry("b (M->Y)", b_within, b_within_se, "within"),
        _path_entry("c' (X->Y direct)", cp_within, cp_within_se, "within"),
        _path_entry("a (X->M)", a_between, a_between_se, "between"),
        _path_entry("b (M->Y)", b_between, b_between_se, "between"),
        _path_entry("c' (X->Y direct)", cp_between, cp_between_se, "between"),
    ]

    # ── Interpretation ───────────────────────────────────────────────
    interp_parts = [
        f"Cross-level mediation ({model_type}) with {n} observations in {J} groups.",
        f"ICCs: X={iccs['x']:.3f}, M={iccs['m']:.3f}, Y={iccs['y']:.3f}.",
    ]

    if within_indirect["significant"]:
        interp_parts.append(
            f"Significant within-level indirect effect: {within_indirect['point_estimate']:.4f} "
            f"[{within_indirect['ci_lower']:.4f}, {within_indirect['ci_upper']:.4f}]."
        )
    else:
        interp_parts.append("Within-level indirect effect is not significant.")

    if between_indirect["significant"]:
        interp_parts.append(
            f"Significant between-level indirect effect: {between_indirect['point_estimate']:.4f} "
            f"[{between_indirect['ci_lower']:.4f}, {between_indirect['ci_upper']:.4f}]."
        )
    else:
        interp_parts.append("Between-level indirect effect is not significant.")

    if prop_within is not None:
        interp_parts.append(f"Proportion mediated within: {prop_within:.1%}.")
    if prop_between is not None:
        interp_parts.append(f"Proportion mediated between: {prop_between:.1%}.")

    return {
        "model_type": model_type,
        "n_observations": n,
        "n_groups": J,
        "icc": iccs,
        "path_coefficients": all_paths,
        "within_indirect_effect": within_indirect,
        "between_indirect_effect": between_indirect,
        "contextual_indirect_effect": contextual_indirect,
        "proportion_mediated": {
            "within": prop_within,
            "between": prop_between,
        },
        "total_effects": {
            "within": float(total_within),
            "between": float(total_between),
        },
        "n_monte_carlo": n_mc,
        "confidence_level": float(confidence),
        "interpretation": " ".join(interp_parts),
    }


# ══════════════════════════════════════════════════════════════════════
# 5. Multilevel CFA (Measurement Model)
# ══════════════════════════════════════════════════════════════════════

@router.post("/measurement-model-multilevel")
def measurement_model_multilevel(req: MultilevelCFARequest):
    """Multilevel CFA: fit factor models separately at within and between
    levels using decomposed covariance matrices."""
    df = pd.DataFrame(req.data)

    # Collect all items
    all_items: list[str] = []
    factor_names = list(req.factor_model.keys())
    for fname, items in req.factor_model.items():
        for item in items:
            if item in all_items:
                raise HTTPException(400, f"Item '{item}' assigned to multiple factors.")
            all_items.append(item)

    _validate_df(df, all_items + [req.group_ids], min_n=20)
    df_clean = df[all_items + [req.group_ids]].dropna()

    if len(df_clean) < 20:
        raise HTTPException(400, "Too few complete observations.")

    unique_groups = df_clean[req.group_ids].unique()
    J = len(unique_groups)
    N = len(df_clean)
    p = len(all_items)
    n_factors = len(factor_names)

    if J < 3:
        raise HTTPException(400, "Need at least 3 groups for multilevel CFA.")
    if p < 3:
        raise HTTPException(400, "Need at least 3 items for CFA.")

    # ── ANOVA decomposition ──────────────────────────────────────────
    S_within, S_between, _, group_sizes, _ = (
        _anova_decomposition(df_clean, req.group_ids, all_items)
    )

    # ── ICCs per item ────────────────────────────────────────────────
    iccs = _compute_icc(S_within, S_between, all_items)

    # ── Build factor model index mapping ─────────────────────────────
    item_to_idx = {item: i for i, item in enumerate(all_items)}
    factor_model_idx: dict[str, list[int]] = {}
    for fname, items in req.factor_model.items():
        factor_model_idx[fname] = [item_to_idx[it] for it in items]

    # ── Fit within-level CFA ─────────────────────────────────────────
    n_within_eff = N - J
    within_fit = _fit_factor_from_cov(S_within, factor_model_idx, p, n_factors)

    within_loadings = {}
    for f_idx, fname in enumerate(factor_names):
        items = req.factor_model[fname]
        item_indices = factor_model_idx[fname]
        loadings_list = []
        for item, i_idx in zip(items, item_indices):
            lam = float(within_fit["loadings"][i_idx, f_idx])
            communality = lam ** 2
            loadings_list.append({
                "item": item,
                "loading": lam,
                "communality": float(communality),
            })
        within_loadings[fname] = loadings_list

    within_fit_indices = _compute_fit_indices_from_cov(
        S_within, within_fit["implied_cov"], n_within_eff, within_fit["n_free"],
    )

    # ── Fit between-level CFA ────────────────────────────────────────
    n_between_eff = J
    # Ensure between covariance is positive semi-definite for fitting
    eigvals = np.linalg.eigvalsh(S_between)
    if np.any(eigvals < -1e-6):
        # Nudge to PSD
        S_between_fit = S_between + np.eye(p) * (abs(min(eigvals)) + 0.01)
    else:
        S_between_fit = S_between.copy()

    between_fit = _fit_factor_from_cov(S_between_fit, factor_model_idx, p, n_factors)

    between_loadings = {}
    for f_idx, fname in enumerate(factor_names):
        items = req.factor_model[fname]
        item_indices = factor_model_idx[fname]
        loadings_list = []
        for item, i_idx in zip(items, item_indices):
            lam = float(between_fit["loadings"][i_idx, f_idx])
            communality = lam ** 2
            loadings_list.append({
                "item": item,
                "loading": lam,
                "communality": float(communality),
            })
        between_loadings[fname] = loadings_list

    between_fit_indices = _compute_fit_indices_from_cov(
        S_between_fit, between_fit["implied_cov"], n_between_eff, between_fit["n_free"],
    )

    # ── Factor correlations at each level ────────────────────────────
    def _factor_correlations(L: np.ndarray) -> dict:
        factor_cov = L.T @ L
        D = np.sqrt(np.maximum(np.diag(factor_cov), 1e-12))
        factor_corr = factor_cov / np.outer(D, D)
        corrs = {}
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                corrs[f"{factor_names[i]}-{factor_names[j]}"] = float(factor_corr[i, j])
        return corrs

    within_factor_corr = _factor_correlations(within_fit["loadings"])
    between_factor_corr = _factor_correlations(between_fit["loadings"])

    # ── Omega reliability at each level ──────────────────────────────
    def _compute_omega(L: np.ndarray, uniquenesses: np.ndarray, factor_indices: dict[str, list[int]]) -> dict:
        """McDonald's omega = (sum of loadings)^2 / ((sum of loadings)^2 + sum of uniquenesses)."""
        omegas = {}
        for fname, indices in factor_indices.items():
            lam_sum = np.sum(L[indices, factor_names.index(fname)])
            lam_sq = lam_sum ** 2
            uniq_sum = np.sum(uniquenesses[indices])
            omega = float(lam_sq / (lam_sq + uniq_sum)) if (lam_sq + uniq_sum) > 0 else 0.0
            omegas[fname] = max(0.0, min(1.0, omega))
        return omegas

    omega_within = _compute_omega(within_fit["loadings"], within_fit["uniquenesses"], factor_model_idx)
    omega_between = _compute_omega(between_fit["loadings"], between_fit["uniquenesses"], factor_model_idx)

    # ── Interpretation ───────────────────────────────────────────────
    interp_parts = [
        f"Multilevel CFA with {n_factors} factor(s) and {p} items, "
        f"{N} observations in {J} groups.",
    ]

    # Check within-level fit
    if within_fit_indices["cfi"] >= 0.95 and within_fit_indices["rmsea"] <= 0.06:
        interp_parts.append("Within-level model shows good fit.")
    elif within_fit_indices["cfi"] >= 0.90 and within_fit_indices["rmsea"] <= 0.08:
        interp_parts.append("Within-level model shows acceptable fit.")
    else:
        interp_parts.append("Within-level model fit is poor; consider revising the measurement model.")

    # Check between-level fit
    if between_fit_indices["cfi"] >= 0.95 and between_fit_indices["rmsea"] <= 0.06:
        interp_parts.append("Between-level model shows good fit.")
    elif between_fit_indices["cfi"] >= 0.90 and between_fit_indices["rmsea"] <= 0.08:
        interp_parts.append("Between-level model shows acceptable fit.")
    else:
        interp_parts.append(
            "Between-level model fit is poor. With few groups, between-level CFA "
            "estimates may be unstable."
        )

    # ICC summary
    high_icc = [item for item, val in iccs.items() if val > 0.15]
    if high_icc:
        interp_parts.append(f"Items with substantial ICC (>0.15): {', '.join(high_icc)}.")
    else:
        interp_parts.append("All item ICCs are below 0.15; between-level variance is limited.")

    # Omega summary
    for fname in factor_names:
        ow = omega_within.get(fname, 0)
        ob = omega_between.get(fname, 0)
        if ow >= 0.70 and ob >= 0.70:
            interp_parts.append(f"Factor '{fname}': good reliability at both levels (omega_w={ow:.2f}, omega_b={ob:.2f}).")
        elif ow >= 0.70:
            interp_parts.append(f"Factor '{fname}': good within-level reliability (omega_w={ow:.2f}) but weak between-level (omega_b={ob:.2f}).")
        else:
            interp_parts.append(f"Factor '{fname}': within-level reliability is below .70 (omega_w={ow:.2f}); consider revising items.")

    return {
        "n_observations": N,
        "n_groups": J,
        "n_items": p,
        "n_factors": n_factors,
        "factor_names": factor_names,
        "within_level": {
            "loadings": within_loadings,
            "uniquenesses": {all_items[i]: float(within_fit["uniquenesses"][i]) for i in range(p)},
            "factor_correlations": within_factor_corr,
            "fit_indices": within_fit_indices,
            "converged": within_fit["converged"],
        },
        "between_level": {
            "loadings": between_loadings,
            "uniquenesses": {all_items[i]: float(between_fit["uniquenesses"][i]) for i in range(p)},
            "factor_correlations": between_factor_corr,
            "fit_indices": between_fit_indices,
            "converged": between_fit["converged"],
        },
        "icc_per_item": {item: {"value": val, "interpretation": _interpret_icc(val)} for item, val in iccs.items()},
        "omega_within": omega_within,
        "omega_between": omega_between,
        "interpretation": " ".join(interp_parts),
    }
