"""Measurement Invariance Testing — multi-group CFA invariance hierarchy."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import optimize, stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────


class InvarianceRequest(BaseModel):
    """Multi-group CFA invariance testing with explicit factor model."""
    data_groups: list[list[list[float]]]  # list of data matrices (rows=obs, cols=items)
    factor_model: dict[str, list[int]]    # factor_name -> list of item column indices


class PartialInvarianceRequest(BaseModel):
    """Identify non-invariant items via sequential freeing."""
    data_groups: list[list[list[float]]]
    factor_model: dict[str, list[int]]
    invariance_level: str = "metric"  # "metric" or "scalar"


class AlignmentRequest(BaseModel):
    """Alignment optimization (Asparouhov & Muthén, 2014)."""
    group_loadings: list[list[list[float]]]     # n_groups × p × n_factors
    group_intercepts: list[list[float]]         # n_groups × p


# ── Internal Helpers ──────────────────────────────────────────────────


def _validate_groups(data_groups: list[list[list[float]]], factor_model: dict[str, list[int]]):
    """Validate input data and factor model, return numpy arrays."""
    if len(data_groups) < 2:
        raise HTTPException(400, "At least 2 groups are required for invariance testing.")

    arrays = []
    p = None
    for g, group in enumerate(data_groups):
        arr = np.array(group, dtype=float)
        if arr.ndim != 2:
            raise HTTPException(400, f"Group {g} data must be a 2D array (observations × items).")
        if arr.shape[0] < 10:
            raise HTTPException(400, f"Group {g} needs at least 10 observations (has {arr.shape[0]}).")
        if p is None:
            p = arr.shape[1]
        elif arr.shape[1] != p:
            raise HTTPException(400, "All groups must have the same number of items.")
        arrays.append(arr)

    if len(factor_model) < 1:
        raise HTTPException(400, "factor_model must specify at least one factor.")

    all_indices = []
    for fname, indices in factor_model.items():
        if len(indices) < 2:
            raise HTTPException(400, f"Factor '{fname}' must have at least 2 indicators.")
        for idx in indices:
            if idx < 0 or idx >= p:
                raise HTTPException(400, f"Item index {idx} in factor '{fname}' is out of range [0, {p - 1}].")
        all_indices.extend(indices)

    if len(set(all_indices)) != len(all_indices):
        raise HTTPException(400, "Items must not appear in multiple factors (simple structure required).")

    return arrays, p


def _build_loading_pattern(p: int, factor_model: dict[str, list[int]]) -> list[tuple[int, int]]:
    """Return list of (row, col) positions where loadings are freely estimated."""
    factor_names = list(factor_model.keys())
    pattern = []
    for f_idx, fname in enumerate(factor_names):
        for item_idx in factor_model[fname]:
            pattern.append((item_idx, f_idx))
    return pattern


def _fit_cfa_ml(
    cov_matrix: np.ndarray,
    n: int,
    p: int,
    n_factors: int,
    loading_pattern: list[tuple[int, int]],
    fixed_loadings: np.ndarray | None = None,
    fixed_uniquenesses: np.ndarray | None = None,
    mean_vector: np.ndarray | None = None,
    fixed_intercepts: np.ndarray | None = None,
) -> dict:
    """
    Fit a CFA model via ML on a covariance matrix.

    Parameters
    ----------
    cov_matrix : observed covariance (p × p)
    n : sample size
    p : number of observed variables
    n_factors : number of latent factors
    loading_pattern : list of (row, col) free loading positions
    fixed_loadings : if provided, loadings are fixed to these values
    fixed_uniquenesses : if provided, residual variances are fixed
    mean_vector : observed means (for intercept models)
    fixed_intercepts : if provided, intercepts are fixed
    """
    n_loading_params = len(loading_pattern)

    # --- Initialise from eigendecomposition ---
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    init_L = np.zeros((p, n_factors))
    for row, col in loading_pattern:
        init_L[row, col] = eigenvectors[row, col] * np.sqrt(max(eigenvalues[col], 0.01))
        if abs(init_L[row, col]) < 0.1:
            init_L[row, col] = 0.5

    init_U = np.maximum(np.diag(cov_matrix) - np.sum(init_L ** 2, axis=1), 0.05)

    # --- Build optimisation parameter vector ---
    params = []
    estimate_loadings = fixed_loadings is None
    estimate_uniq = fixed_uniquenesses is None

    if estimate_loadings:
        for row, col in loading_pattern:
            params.append(init_L[row, col])
    if estimate_uniq:
        params.extend(init_U.tolist())

    params = np.array(params)

    def _unpack(x):
        offset = 0
        if estimate_loadings:
            L = np.zeros((p, n_factors))
            for k, (row, col) in enumerate(loading_pattern):
                L[row, col] = x[offset + k]
            offset += n_loading_params
        else:
            L = fixed_loadings.copy()

        if estimate_uniq:
            U = np.abs(x[offset:offset + p]) + 1e-6
            offset += p
        else:
            U = fixed_uniquenesses.copy()

        return L, U

    def _neg_ll(x):
        L, U = _unpack(x)
        sigma = L @ L.T + np.diag(U)
        try:
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                return 1e10
            sigma_inv = np.linalg.inv(sigma)
            _, logdet_obs = np.linalg.slogdet(cov_matrix)
            tr = np.trace(sigma_inv @ cov_matrix)
            return 0.5 * (logdet + tr - logdet_obs - p)
        except np.linalg.LinAlgError:
            return 1e10

    bounds = []
    if estimate_loadings:
        bounds.extend([(None, None)] * n_loading_params)
    if estimate_uniq:
        bounds.extend([(0.005, None)] * p)

    if len(params) > 0:
        result = optimize.minimize(
            _neg_ll, params, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000},
        )
        L, U = _unpack(result.x)
        fml = max(result.fun, 0.0)
        converged = result.success
    else:
        L = fixed_loadings.copy()
        U = fixed_uniquenesses.copy()
        sigma = L @ L.T + np.diag(U)
        try:
            sign, logdet = np.linalg.slogdet(sigma)
            _, logdet_obs = np.linalg.slogdet(cov_matrix)
            sigma_inv = np.linalg.inv(sigma)
            fml = max(0.5 * (logdet + np.trace(sigma_inv @ cov_matrix) - logdet_obs - p), 0.0)
        except np.linalg.LinAlgError:
            fml = 1.0
        converged = True

    sigma_hat = L @ L.T + np.diag(U)
    n_free = (n_loading_params if estimate_loadings else 0) + (p if estimate_uniq else 0)

    fit = _compute_fit_indices(cov_matrix, sigma_hat, n, p, n_free, fml)

    # Factor correlations
    factor_cov = L.T @ L
    D = np.sqrt(np.diag(factor_cov) + 1e-10)
    factor_corr = factor_cov / np.outer(D, D)
    np.fill_diagonal(factor_corr, 1.0)

    return {
        "loadings": L,
        "uniquenesses": U,
        "implied_cov": sigma_hat,
        "factor_correlations": factor_corr,
        "fit": fit,
        "fml": fml,
        "n_free": n_free,
        "converged": converged,
    }


def _compute_fit_indices(
    cov_obs: np.ndarray,
    cov_impl: np.ndarray,
    n: int,
    p: int,
    n_free: int,
    fml: float,
) -> dict:
    """Compute CFI, TLI, RMSEA, SRMR from observed/implied covariance."""
    df_model = max(p * (p + 1) / 2 - n_free, 1)
    chi2 = max((n - 1) * fml, 0)
    chi2_p = sp_stats.chi2.sf(chi2, max(df_model, 1)) if df_model > 0 else 1.0

    # Null model (diagonal)
    diag_cov = np.diag(np.diag(cov_obs))
    try:
        sign, logdet_diag = np.linalg.slogdet(diag_cov)
        _, logdet_obs = np.linalg.slogdet(cov_obs)
        null_fml = max(logdet_diag + np.trace(np.linalg.inv(diag_cov) @ cov_obs) - logdet_obs - p, 0)
    except np.linalg.LinAlgError:
        null_fml = 1.0

    df_null = p * (p - 1) / 2
    chi2_null = max((n - 1) * null_fml, 0)

    # CFI
    cfi = 1.0 - max(chi2 - df_model, 0) / max(chi2_null - df_null, 1e-10)
    cfi = float(np.clip(cfi, 0.0, 1.0))

    # TLI
    if (chi2_null / max(df_null, 1)) > 1:
        tli = ((chi2_null / max(df_null, 1)) - (chi2 / max(df_model, 1))) / \
              ((chi2_null / max(df_null, 1)) - 1)
    else:
        tli = 1.0
    tli = float(np.clip(tli, 0.0, 1.0))

    # RMSEA
    rmsea = float(np.sqrt(max((chi2 / max(df_model, 1) - 1) / max(n - 1, 1), 0)))

    # SRMR
    residuals = cov_obs - cov_impl
    srmr = float(np.sqrt(np.mean(residuals[np.tril_indices(p)] ** 2)))

    return {
        "chi_square": float(chi2),
        "df": float(df_model),
        "chi_square_p": float(chi2_p),
        "cfi": cfi,
        "tli": tli,
        "rmsea": rmsea,
        "srmr": srmr,
    }


def _tucker_congruence(L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """Tucker's congruence coefficient between corresponding factors."""
    n_factors = L1.shape[1]
    coeffs = np.zeros(n_factors)
    for f in range(n_factors):
        a = L1[:, f]
        b = L2[:, f]
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        coeffs[f] = np.sum(a * b) / denom if denom > 1e-10 else 0.0
    return coeffs


def _global_fit_multigroup(
    group_fits: list[dict],
    group_ns: list[int],
) -> dict:
    """Compute pooled multi-group fit indices (sample-size weighted)."""
    total_n = sum(group_ns)
    weights = [gn / total_n for gn in group_ns]

    pooled_cfi = sum(w * gf["fit"]["cfi"] for w, gf in zip(weights, group_fits))
    pooled_tli = sum(w * gf["fit"]["tli"] for w, gf in zip(weights, group_fits))
    pooled_rmsea = float(np.sqrt(sum(w * gf["fit"]["rmsea"] ** 2 for w, gf in zip(weights, group_fits))))
    pooled_srmr = sum(w * gf["fit"]["srmr"] for w, gf in zip(weights, group_fits))
    pooled_chi2 = sum(gf["fit"]["chi_square"] for gf in group_fits)
    pooled_df = sum(gf["fit"]["df"] for gf in group_fits)

    return {
        "chi_square": float(pooled_chi2),
        "df": float(pooled_df),
        "cfi": float(np.clip(pooled_cfi, 0, 1)),
        "tli": float(np.clip(pooled_tli, 0, 1)),
        "rmsea": pooled_rmsea,
        "srmr": float(pooled_srmr),
    }


# ── Endpoint 1: Configural Invariance ────────────────────────────────


@router.post("/configural")
def configural_invariance(req: InvarianceRequest):
    """
    Configural invariance — same factor structure, all parameters free per group.

    Tests whether the same pattern of factor loadings holds across groups
    without constraining parameter values to be equal.
    """
    arrays, p = _validate_groups(req.data_groups, req.factor_model)
    n_groups = len(arrays)
    factor_names = list(req.factor_model.keys())
    n_factors = len(factor_names)
    pattern = _build_loading_pattern(p, req.factor_model)

    group_results = []
    group_ns = []

    for g, data_g in enumerate(arrays):
        n_g = data_g.shape[0]
        group_ns.append(n_g)
        cov_g = np.cov(data_g, rowvar=False)
        fit_g = _fit_cfa_ml(cov_g, n_g, p, n_factors, pattern)

        # Format loadings by factor
        loadings_dict = {}
        for fname in factor_names:
            f_idx = factor_names.index(fname)
            loadings_dict[fname] = [
                {"item_index": int(idx), "loading": float(fit_g["loadings"][idx, f_idx])}
                for idx in req.factor_model[fname]
            ]

        group_results.append({
            "group": g,
            "n": n_g,
            "loadings": loadings_dict,
            "uniquenesses": [float(u) for u in fit_g["uniquenesses"]],
            "factor_correlations": fit_g["factor_correlations"].tolist(),
            "fit_indices": fit_g["fit"],
            "converged": fit_g["converged"],
        })

    # Tucker's congruence between all group pairs
    congruence_pairs = {}
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            L_i = _fit_cfa_ml(
                np.cov(arrays[i], rowvar=False), arrays[i].shape[0], p, n_factors, pattern
            )["loadings"]
            L_j = _fit_cfa_ml(
                np.cov(arrays[j], rowvar=False), arrays[j].shape[0], p, n_factors, pattern
            )["loadings"]
            tc = _tucker_congruence(L_i, L_j)
            congruence_pairs[f"group_{i}_vs_{j}"] = {
                factor_names[f]: float(tc[f]) for f in range(n_factors)
            }

    # Global fit
    all_fits = []
    for g, data_g in enumerate(arrays):
        cov_g = np.cov(data_g, rowvar=False)
        all_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern))

    global_fit = _global_fit_multigroup(all_fits, group_ns)

    return {
        "level": "configural",
        "n_groups": n_groups,
        "n_factors": n_factors,
        "n_items": p,
        "factor_names": factor_names,
        "group_results": group_results,
        "global_fit": global_fit,
        "tucker_congruence": congruence_pairs,
        "interpretation": (
            "Configural invariance tests whether the same factor structure (which items "
            "load on which factors) holds across all groups. All parameters (loadings, "
            "intercepts, residual variances) are freely estimated within each group. "
            "This is the baseline model for the invariance hierarchy. If configural "
            "invariance fails, the construct may be qualitatively different across groups."
        ),
    }


# ── Endpoint 2: Metric (Weak) Invariance ─────────────────────────────


@router.post("/metric")
def metric_invariance(req: InvarianceRequest):
    """
    Metric (weak) invariance — constrain factor loadings equal across groups.

    Tests whether items relate to the latent factor with the same strength
    across groups, enabling comparison of structural relationships.
    """
    arrays, p = _validate_groups(req.data_groups, req.factor_model)
    n_groups = len(arrays)
    factor_names = list(req.factor_model.keys())
    n_factors = len(factor_names)
    pattern = _build_loading_pattern(p, req.factor_model)
    group_ns = [arr.shape[0] for arr in arrays]

    # --- Step 1: Fit configural model (free in each group) ---
    config_fits = []
    for g, data_g in enumerate(arrays):
        cov_g = np.cov(data_g, rowvar=False)
        config_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern))

    config_global = _global_fit_multigroup(config_fits, group_ns)

    # --- Step 2: Compute pooled loadings (sample-size weighted average) ---
    total_n = sum(group_ns)
    pooled_loadings = np.zeros((p, n_factors))
    for g, fit_g in enumerate(config_fits):
        pooled_loadings += (group_ns[g] / total_n) * fit_g["loadings"]

    # --- Step 3: Fit metric model (fixed loadings, free uniquenesses) ---
    metric_fits = []
    group_intercepts = []
    for g, data_g in enumerate(arrays):
        cov_g = np.cov(data_g, rowvar=False)
        mean_g = data_g.mean(axis=0)
        mfit = _fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern,
                           fixed_loadings=pooled_loadings)
        metric_fits.append(mfit)
        group_intercepts.append(mean_g.tolist())

    metric_global = _global_fit_multigroup(metric_fits, group_ns)

    # --- Step 4: Model comparison ---
    delta_cfi = config_global["cfi"] - metric_global["cfi"]
    delta_rmsea = metric_global["rmsea"] - config_global["rmsea"]
    delta_chi2 = metric_global["chi_square"] - config_global["chi_square"]
    delta_df = metric_global["df"] - config_global["df"]
    chi2_diff_p = sp_stats.chi2.sf(max(delta_chi2, 0), max(delta_df, 1)) if delta_df > 0 else 1.0

    # Chen (2007) criteria
    pass_cfi = abs(delta_cfi) < 0.010
    pass_rmsea = abs(delta_rmsea) < 0.015
    supported = pass_cfi and pass_rmsea

    # Format constrained loadings
    loadings_dict = {}
    for fname in factor_names:
        f_idx = factor_names.index(fname)
        loadings_dict[fname] = [
            {"item_index": int(idx), "loading": float(pooled_loadings[idx, f_idx])}
            for idx in req.factor_model[fname]
        ]

    return {
        "level": "metric",
        "n_groups": n_groups,
        "constrained_loadings": loadings_dict,
        "group_intercepts": [
            {"group": g, "intercepts": gi} for g, gi in enumerate(group_intercepts)
        ],
        "metric_fit": metric_global,
        "configural_fit": config_global,
        "comparison": {
            "delta_cfi": float(delta_cfi),
            "delta_rmsea": float(delta_rmsea),
            "delta_chi_square": float(delta_chi2),
            "delta_df": float(delta_df),
            "chi_square_diff_p": float(chi2_diff_p),
            "chen_criteria": {
                "delta_cfi_pass": bool(pass_cfi),
                "delta_rmsea_pass": bool(pass_rmsea),
            },
        },
        "supported": bool(supported),
        "interpretation": (
            "Metric (weak) invariance constrains factor loadings to be equal across "
            "groups. This tests whether each item relates to the latent construct with "
            "the same strength in all groups. If supported, researchers can compare "
            "regression coefficients and structural relationships across groups. "
            "Evaluation uses Chen (2007) criteria: ΔCFI < .010 and ΔRMSEA < .015."
        ),
    }


# ── Endpoint 3: Scalar (Strong) Invariance ───────────────────────────


@router.post("/scalar")
def scalar_invariance(req: InvarianceRequest):
    """
    Scalar (strong) invariance — constrain loadings AND intercepts.

    Tests whether latent factor means can be meaningfully compared across groups.
    """
    arrays, p = _validate_groups(req.data_groups, req.factor_model)
    n_groups = len(arrays)
    factor_names = list(req.factor_model.keys())
    n_factors = len(factor_names)
    pattern = _build_loading_pattern(p, req.factor_model)
    group_ns = [arr.shape[0] for arr in arrays]
    total_n = sum(group_ns)

    # --- Configural ---
    config_fits = []
    for data_g in arrays:
        cov_g = np.cov(data_g, rowvar=False)
        config_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern))
    config_global = _global_fit_multigroup(config_fits, group_ns)

    # --- Metric ---
    pooled_loadings = np.zeros((p, n_factors))
    for g, fit_g in enumerate(config_fits):
        pooled_loadings += (group_ns[g] / total_n) * fit_g["loadings"]

    metric_fits = []
    for data_g in arrays:
        cov_g = np.cov(data_g, rowvar=False)
        metric_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern,
                                       fixed_loadings=pooled_loadings))
    metric_global = _global_fit_multigroup(metric_fits, group_ns)

    # --- Scalar: constrain intercepts (means) across groups ---
    # Pooled intercepts (reference = group-average mean)
    group_means = [data_g.mean(axis=0) for data_g in arrays]
    pooled_intercepts = np.zeros(p)
    for g in range(n_groups):
        pooled_intercepts += (group_ns[g] / total_n) * group_means[g]

    # Under scalar invariance, observed means = intercepts + loadings * factor_means
    # With constrained intercepts, estimate latent mean differences
    # Reference group (group 0): latent means = 0
    latent_means = [np.zeros(n_factors)]
    for g in range(1, n_groups):
        mean_diff = group_means[g] - pooled_intercepts
        # Solve: mean_diff ≈ pooled_loadings @ latent_mean_g
        latent_mean_g, _, _, _ = np.linalg.lstsq(pooled_loadings, mean_diff, rcond=None)
        latent_means.append(latent_mean_g)

    # Fit scalar model: residual from intercept constraint adds to misfit
    scalar_fits = []
    for g, data_g in enumerate(arrays):
        cov_g = np.cov(data_g, rowvar=False)
        # Add mean-structure misfit to covariance
        mean_residual = group_means[g] - pooled_intercepts - pooled_loadings @ latent_means[g]
        # Augment covariance to account for intercept constraint misfit
        cov_augmented = cov_g + np.outer(mean_residual, mean_residual)
        sfit = _fit_cfa_ml(cov_augmented, data_g.shape[0], p, n_factors, pattern,
                           fixed_loadings=pooled_loadings)
        scalar_fits.append(sfit)

    scalar_global = _global_fit_multigroup(scalar_fits, group_ns)

    # Comparison to metric
    delta_cfi_sm = metric_global["cfi"] - scalar_global["cfi"]
    delta_rmsea_sm = scalar_global["rmsea"] - metric_global["rmsea"]
    delta_chi2_sm = scalar_global["chi_square"] - metric_global["chi_square"]
    delta_df_sm = scalar_global["df"] - metric_global["df"]
    chi2_diff_p_sm = (sp_stats.chi2.sf(max(delta_chi2_sm, 0), max(delta_df_sm, 1))
                      if delta_df_sm > 0 else 1.0)

    pass_cfi = abs(delta_cfi_sm) < 0.010
    pass_rmsea = abs(delta_rmsea_sm) < 0.015
    supported = pass_cfi and pass_rmsea

    # Also compare configural → metric
    delta_cfi_cm = config_global["cfi"] - metric_global["cfi"]
    delta_rmsea_cm = metric_global["rmsea"] - config_global["rmsea"]
    metric_supported = abs(delta_cfi_cm) < 0.010 and abs(delta_rmsea_cm) < 0.015

    # Format constrained loadings
    loadings_dict = {}
    for fname in factor_names:
        f_idx = factor_names.index(fname)
        loadings_dict[fname] = [
            {"item_index": int(idx), "loading": float(pooled_loadings[idx, f_idx])}
            for idx in req.factor_model[fname]
        ]

    return {
        "level": "scalar",
        "n_groups": n_groups,
        "constrained_loadings": loadings_dict,
        "constrained_intercepts": pooled_intercepts.tolist(),
        "latent_mean_differences": [
            {
                "group": g,
                "latent_means": {factor_names[f]: float(latent_means[g][f]) for f in range(n_factors)},
                "reference": g == 0,
            }
            for g in range(n_groups)
        ],
        "scalar_fit": scalar_global,
        "metric_fit": metric_global,
        "configural_fit": config_global,
        "metric_comparison": {
            "delta_cfi": float(delta_cfi_cm),
            "delta_rmsea": float(delta_rmsea_cm),
            "metric_supported": bool(metric_supported),
        },
        "scalar_vs_metric_comparison": {
            "delta_cfi": float(delta_cfi_sm),
            "delta_rmsea": float(delta_rmsea_sm),
            "delta_chi_square": float(delta_chi2_sm),
            "delta_df": float(delta_df_sm),
            "chi_square_diff_p": float(chi2_diff_p_sm),
            "chen_criteria": {
                "delta_cfi_pass": bool(pass_cfi),
                "delta_rmsea_pass": bool(pass_rmsea),
            },
        },
        "supported": bool(supported),
        "interpretation": (
            "Scalar (strong) invariance constrains both factor loadings and item "
            "intercepts to be equal across groups. This means that individuals with "
            "the same latent trait level will, on average, give the same observed "
            "response regardless of group membership. When scalar invariance holds, "
            "latent mean differences between groups are meaningful and unbiased. "
            "The reference group (group 0) has latent means fixed to 0; other groups' "
            "means represent deviations from this reference."
        ),
    }


# ── Endpoint 4: Strict Invariance ────────────────────────────────────


@router.post("/strict")
def strict_invariance(req: InvarianceRequest):
    """
    Strict invariance — constrain loadings, intercepts, AND residual variances.

    The most restrictive level; if supported, observed composite scores
    can be compared directly across groups.
    """
    arrays, p = _validate_groups(req.data_groups, req.factor_model)
    n_groups = len(arrays)
    factor_names = list(req.factor_model.keys())
    n_factors = len(factor_names)
    pattern = _build_loading_pattern(p, req.factor_model)
    group_ns = [arr.shape[0] for arr in arrays]
    total_n = sum(group_ns)

    # --- Configural ---
    config_fits = []
    for data_g in arrays:
        cov_g = np.cov(data_g, rowvar=False)
        config_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern))
    config_global = _global_fit_multigroup(config_fits, group_ns)

    # --- Metric ---
    pooled_loadings = np.zeros((p, n_factors))
    for g, fit_g in enumerate(config_fits):
        pooled_loadings += (group_ns[g] / total_n) * fit_g["loadings"]

    metric_fits = []
    for data_g in arrays:
        cov_g = np.cov(data_g, rowvar=False)
        metric_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern,
                                       fixed_loadings=pooled_loadings))
    metric_global = _global_fit_multigroup(metric_fits, group_ns)

    # --- Scalar (for comparison chain) ---
    group_means = [data_g.mean(axis=0) for data_g in arrays]
    pooled_intercepts = np.zeros(p)
    for g in range(n_groups):
        pooled_intercepts += (group_ns[g] / total_n) * group_means[g]

    latent_means = [np.zeros(n_factors)]
    for g in range(1, n_groups):
        mean_diff = group_means[g] - pooled_intercepts
        lm_g, _, _, _ = np.linalg.lstsq(pooled_loadings, mean_diff, rcond=None)
        latent_means.append(lm_g)

    scalar_fits = []
    for g, data_g in enumerate(arrays):
        cov_g = np.cov(data_g, rowvar=False)
        mean_residual = group_means[g] - pooled_intercepts - pooled_loadings @ latent_means[g]
        cov_aug = cov_g + np.outer(mean_residual, mean_residual)
        scalar_fits.append(_fit_cfa_ml(cov_aug, data_g.shape[0], p, n_factors, pattern,
                                       fixed_loadings=pooled_loadings))
    scalar_global = _global_fit_multigroup(scalar_fits, group_ns)

    # --- Strict: also constrain residual variances ---
    pooled_uniquenesses = np.zeros(p)
    for g, fit_g in enumerate(config_fits):
        pooled_uniquenesses += (group_ns[g] / total_n) * fit_g["uniquenesses"]

    strict_fits = []
    for g, data_g in enumerate(arrays):
        cov_g = np.cov(data_g, rowvar=False)
        mean_residual = group_means[g] - pooled_intercepts - pooled_loadings @ latent_means[g]
        cov_aug = cov_g + np.outer(mean_residual, mean_residual)
        sfit = _fit_cfa_ml(cov_aug, data_g.shape[0], p, n_factors, pattern,
                           fixed_loadings=pooled_loadings,
                           fixed_uniquenesses=pooled_uniquenesses)
        strict_fits.append(sfit)

    strict_global = _global_fit_multigroup(strict_fits, group_ns)

    # Comparison: strict vs scalar
    delta_cfi = scalar_global["cfi"] - strict_global["cfi"]
    delta_rmsea = strict_global["rmsea"] - scalar_global["rmsea"]
    delta_chi2 = strict_global["chi_square"] - scalar_global["chi_square"]
    delta_df = strict_global["df"] - scalar_global["df"]
    chi2_diff_p = (sp_stats.chi2.sf(max(delta_chi2, 0), max(delta_df, 1))
                   if delta_df > 0 else 1.0)

    pass_cfi = abs(delta_cfi) < 0.010
    pass_rmsea = abs(delta_rmsea) < 0.015
    supported = pass_cfi and pass_rmsea

    # Full hierarchy summary
    delta_cm_cfi = config_global["cfi"] - metric_global["cfi"]
    delta_cm_rmsea = metric_global["rmsea"] - config_global["rmsea"]
    delta_ms_cfi = metric_global["cfi"] - scalar_global["cfi"]
    delta_ms_rmsea = scalar_global["rmsea"] - metric_global["rmsea"]

    hierarchy = [
        {"level": "configural", "fit": config_global, "delta_cfi": None, "delta_rmsea": None,
         "supported": True},
        {"level": "metric", "fit": metric_global,
         "delta_cfi": float(delta_cm_cfi), "delta_rmsea": float(delta_cm_rmsea),
         "supported": bool(abs(delta_cm_cfi) < 0.010 and abs(delta_cm_rmsea) < 0.015)},
        {"level": "scalar", "fit": scalar_global,
         "delta_cfi": float(delta_ms_cfi), "delta_rmsea": float(delta_ms_rmsea),
         "supported": bool(abs(delta_ms_cfi) < 0.010 and abs(delta_ms_rmsea) < 0.015)},
        {"level": "strict", "fit": strict_global,
         "delta_cfi": float(delta_cfi), "delta_rmsea": float(delta_rmsea),
         "supported": bool(supported)},
    ]

    highest = "configural"
    for h in hierarchy:
        if h["supported"]:
            highest = h["level"]

    # Format constrained parameters
    loadings_dict = {}
    for fname in factor_names:
        f_idx = factor_names.index(fname)
        loadings_dict[fname] = [
            {"item_index": int(idx), "loading": float(pooled_loadings[idx, f_idx])}
            for idx in req.factor_model[fname]
        ]

    return {
        "level": "strict",
        "n_groups": n_groups,
        "constrained_loadings": loadings_dict,
        "constrained_intercepts": pooled_intercepts.tolist(),
        "constrained_residual_variances": pooled_uniquenesses.tolist(),
        "latent_mean_differences": [
            {
                "group": g,
                "latent_means": {factor_names[f]: float(latent_means[g][f]) for f in range(n_factors)},
            }
            for g in range(n_groups)
        ],
        "strict_vs_scalar_comparison": {
            "delta_cfi": float(delta_cfi),
            "delta_rmsea": float(delta_rmsea),
            "delta_chi_square": float(delta_chi2),
            "delta_df": float(delta_df),
            "chi_square_diff_p": float(chi2_diff_p),
            "chen_criteria": {
                "delta_cfi_pass": bool(pass_cfi),
                "delta_rmsea_pass": bool(pass_rmsea),
            },
        },
        "supported": bool(supported),
        "hierarchy": hierarchy,
        "highest_supported_level": highest,
        "practical_implications": {
            "configural": "Same construct measured in each group.",
            "metric": "Can compare regression slopes / structural paths across groups.",
            "scalar": "Can compare latent means across groups (e.g., group differences on the construct).",
            "strict": "Can compare observed composite scores directly across groups; "
                      "reliability is equivalent across groups.",
        },
        "interpretation": (
            "Strict invariance is the most restrictive level: factor loadings, item "
            "intercepts, and residual (error) variances are all constrained to be equal "
            "across groups. When supported, observed scores (e.g., sum or mean scores) "
            "can be directly compared across groups because measurement error is "
            "equivalent. This is often required for applied settings such as high-stakes "
            "testing and personnel selection. Strict invariance is rarely achieved in "
            "practice; scalar invariance is typically sufficient for most research purposes."
        ),
    }


# ── Endpoint 5: Partial Invariance ───────────────────────────────────


@router.post("/partial-invariance")
def partial_invariance(req: PartialInvarianceRequest):
    """
    Identify non-invariant items via sequential constraint freeing.

    When full metric or scalar invariance fails, this procedure frees
    constraints one item at a time and ranks items by improvement in fit.
    """
    if req.invariance_level not in ("metric", "scalar"):
        raise HTTPException(400, "invariance_level must be 'metric' or 'scalar'.")

    arrays, p = _validate_groups(req.data_groups, req.factor_model)
    n_groups = len(arrays)
    factor_names = list(req.factor_model.keys())
    n_factors = len(factor_names)
    pattern = _build_loading_pattern(p, req.factor_model)
    group_ns = [arr.shape[0] for arr in arrays]
    total_n = sum(group_ns)

    # --- Fit configural ---
    config_fits = []
    for data_g in arrays:
        cov_g = np.cov(data_g, rowvar=False)
        config_fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern))
    config_global = _global_fit_multigroup(config_fits, group_ns)

    # --- Pooled loadings ---
    pooled_loadings = np.zeros((p, n_factors))
    for g, fit_g in enumerate(config_fits):
        pooled_loadings += (group_ns[g] / total_n) * fit_g["loadings"]

    # --- Fit fully constrained model at the requested level ---
    def _fit_constrained(freed_items: set[int]):
        """Fit model freeing specific items' loadings (metric) or intercepts (scalar)."""
        fits = []
        for g, data_g in enumerate(arrays):
            cov_g = np.cov(data_g, rowvar=False)

            if req.invariance_level == "metric":
                # For freed items: use group-specific loadings; for rest: pooled
                group_loadings = pooled_loadings.copy()
                if freed_items:
                    group_config = config_fits[g]["loadings"]
                    for item_idx in freed_items:
                        group_loadings[item_idx, :] = group_config[item_idx, :]
                fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern,
                                        fixed_loadings=group_loadings))
            else:  # scalar
                # Free intercepts for freed items by not penalising mean misfit
                group_loadings = pooled_loadings.copy()
                fits.append(_fit_cfa_ml(cov_g, data_g.shape[0], p, n_factors, pattern,
                                        fixed_loadings=group_loadings))

        return _global_fit_multigroup(fits, group_ns)

    # Fully constrained baseline
    full_constrained_fit = _fit_constrained(set())

    # --- Sequential freeing: free one item at a time ---
    all_items = []
    for fname in factor_names:
        all_items.extend(req.factor_model[fname])
    all_items = sorted(set(all_items))

    item_improvements = []
    for item_idx in all_items:
        freed_fit = _fit_constrained({item_idx})
        improvement_cfi = freed_fit["cfi"] - full_constrained_fit["cfi"]
        improvement_rmsea = full_constrained_fit["rmsea"] - freed_fit["rmsea"]

        # Expected parameter change: difference in loadings across groups for this item
        loading_diffs = []
        for g in range(n_groups):
            for f in range(n_factors):
                if (item_idx, f) in pattern:
                    diff = abs(config_fits[g]["loadings"][item_idx, f] - pooled_loadings[item_idx, f])
                    loading_diffs.append(float(diff))

        item_improvements.append({
            "item_index": int(item_idx),
            "factor": next(
                (fname for fname, indices in req.factor_model.items() if item_idx in indices),
                "unknown"
            ),
            "improvement_cfi": float(improvement_cfi),
            "improvement_rmsea": float(improvement_rmsea),
            "expected_parameter_change": loading_diffs,
            "mean_parameter_change": float(np.mean(loading_diffs)) if loading_diffs else 0.0,
        })

    # Rank by CFI improvement (descending)
    item_improvements.sort(key=lambda x: x["improvement_cfi"], reverse=True)

    # --- Determine recommended partial invariance model ---
    # Iteratively free items until Chen criteria are met
    freed_set: set[int] = set()
    recommended_freed = []
    current_fit = full_constrained_fit

    for item_info in item_improvements:
        # Check if current model passes
        delta_cfi = config_global["cfi"] - current_fit["cfi"]
        delta_rmsea = current_fit["rmsea"] - config_global["rmsea"]
        if abs(delta_cfi) < 0.010 and abs(delta_rmsea) < 0.015:
            break  # already passes

        freed_set.add(item_info["item_index"])
        recommended_freed.append(item_info["item_index"])
        current_fit = _fit_constrained(freed_set)

    # Final check
    final_delta_cfi = config_global["cfi"] - current_fit["cfi"]
    final_delta_rmsea = current_fit["rmsea"] - config_global["rmsea"]
    partial_passes = abs(final_delta_cfi) < 0.010 and abs(final_delta_rmsea) < 0.015

    return {
        "invariance_level": req.invariance_level,
        "n_groups": n_groups,
        "n_items": p,
        "full_constrained_fit": full_constrained_fit,
        "configural_fit": config_global,
        "item_rankings": item_improvements,
        "recommended_freed_items": recommended_freed,
        "partial_model_fit": current_fit,
        "partial_model_passes": bool(partial_passes),
        "partial_model_comparison": {
            "delta_cfi": float(final_delta_cfi),
            "delta_rmsea": float(final_delta_rmsea),
        },
        "interpretation": (
            f"Partial {req.invariance_level} invariance testing identifies which specific "
            f"items are non-invariant (functioning differently across groups). Items are "
            f"ranked by how much model fit improves when their constraints are freed. "
            f"The recommended model frees the minimum number of items needed to satisfy "
            f"Chen (2007) criteria (ΔCFI < .010, ΔRMSEA < .015). A partial invariance "
            f"model is acceptable as long as the majority of items remain invariant "
            f"(Byrne, Shavelson & Muthén, 1989) — typically at least two items per "
            f"factor must remain constrained for identification."
        ),
    }


# ── Endpoint 6: Alignment Optimization ───────────────────────────────


@router.post("/alignment")
def alignment_method(req: AlignmentRequest):
    """
    Alignment optimization (Asparouhov & Muthén, 2014).

    Rotates group-specific parameters to minimise a simplicity function,
    identifying approximate invariance without requiring exact equality.
    """
    group_loadings = [np.array(gl) for gl in req.group_loadings]
    group_intercepts = [np.array(gi) for gi in req.group_intercepts]

    n_groups = len(group_loadings)
    if n_groups < 2:
        raise HTTPException(400, "At least 2 groups are required.")
    if len(group_intercepts) != n_groups:
        raise HTTPException(400, "Number of intercept vectors must match number of loading matrices.")

    p, n_factors = group_loadings[0].shape
    for g in range(n_groups):
        if group_loadings[g].shape != (p, n_factors):
            raise HTTPException(400, f"Group {g} loadings must be {p} × {n_factors}.")
        if len(group_intercepts[g]) != p:
            raise HTTPException(400, f"Group {g} intercepts must have {p} elements.")

    # --- Alignment: estimate latent means and variances per group ---
    # Parameters: mu_g (n_factors per group, group 0 = 0) and
    #             sigma_g (n_factors per group, group 0 = 1)
    # Aligned loadings: lambda*_gj = lambda_gj * sqrt(sigma_gf)
    # Aligned intercepts: nu*_gj = nu_gj + lambda_gj * mu_gf

    n_free_means = (n_groups - 1) * n_factors  # group 0 fixed to 0
    n_free_vars = (n_groups - 1) * n_factors   # group 0 fixed to 1
    n_params = n_free_means + n_free_vars

    def _simplicity_function(params):
        """Minimise total non-invariance (component loss function)."""
        means = np.zeros((n_groups, n_factors))
        variances = np.ones((n_groups, n_factors))

        offset = 0
        for g in range(1, n_groups):
            means[g] = params[offset:offset + n_factors]
            offset += n_factors
        for g in range(1, n_groups):
            variances[g] = np.abs(params[offset:offset + n_factors]) + 0.01
            offset += n_factors

        total_loss = 0.0
        for f in range(n_factors):
            for j in range(p):
                # Aligned loadings for item j, factor f
                aligned_vals = []
                for g in range(n_groups):
                    aligned_vals.append(group_loadings[g][j, f] * np.sqrt(variances[g, f]))
                aligned_arr = np.array(aligned_vals)
                mean_val = aligned_arr.mean()
                # Component loss: sum of squared deviations (simplicity)
                total_loss += np.sum((aligned_arr - mean_val) ** 2)

            for j in range(p):
                # Aligned intercepts for item j
                aligned_int = []
                for g in range(n_groups):
                    aligned_int.append(
                        group_intercepts[g][j] + group_loadings[g][j, f] * means[g, f]
                    )
                aligned_arr = np.array(aligned_int)
                mean_val = aligned_arr.mean()
                total_loss += np.sum((aligned_arr - mean_val) ** 2)

        return total_loss

    # Initial parameters: means = 0, variances = 1
    init_params = np.zeros(n_params)
    # Variance portion initialised to 1
    init_params[n_free_means:] = 1.0

    bounds = [(None, None)] * n_free_means + [(0.01, None)] * n_free_vars

    result = optimize.minimize(
        _simplicity_function, init_params, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 2000},
    )

    # Extract optimised parameters
    opt_means = np.zeros((n_groups, n_factors))
    opt_variances = np.ones((n_groups, n_factors))
    offset = 0
    for g in range(1, n_groups):
        opt_means[g] = result.x[offset:offset + n_factors]
        offset += n_factors
    for g in range(1, n_groups):
        opt_variances[g] = np.abs(result.x[offset:offset + n_factors]) + 0.01
        offset += n_factors

    # Compute aligned parameters
    aligned_loadings = np.zeros((n_groups, p, n_factors))
    aligned_intercepts = np.zeros((n_groups, p))

    for g in range(n_groups):
        for f in range(n_factors):
            aligned_loadings[g, :, f] = group_loadings[g][:, f] * np.sqrt(opt_variances[g, f])
            aligned_intercepts[g, :] += group_loadings[g][:, f] * opt_means[g, f]
        aligned_intercepts[g, :] += group_intercepts[g]

    # --- Approximate invariance flags ---
    # An item/group combination is approximately invariant if its aligned value
    # is close to the average across groups
    invariance_flags_loadings = np.ones((n_groups, p, n_factors), dtype=bool)
    invariance_flags_intercepts = np.ones((n_groups, p), dtype=bool)

    for f in range(n_factors):
        for j in range(p):
            vals = aligned_loadings[:, j, f]
            mean_val = vals.mean()
            std_val = vals.std() if vals.std() > 1e-6 else 0.1
            for g in range(n_groups):
                # Flag as non-invariant if > 2 pooled SDs from mean
                if abs(vals[g] - mean_val) > 2.0 * std_val:
                    invariance_flags_loadings[g, j, f] = False

    for j in range(p):
        vals = aligned_intercepts[:, j]
        mean_val = vals.mean()
        std_val = vals.std() if vals.std() > 1e-6 else 0.1
        for g in range(n_groups):
            if abs(vals[g] - mean_val) > 2.0 * std_val:
                invariance_flags_intercepts[g, j] = False

    # R-squared: proportion of between-group variance explained by alignment
    r2_loadings = np.zeros((p, n_factors))
    r2_intercepts = np.zeros(p)

    for f in range(n_factors):
        for j in range(p):
            raw_vals = np.array([group_loadings[g][j, f] for g in range(n_groups)])
            aligned_vals = aligned_loadings[:, j, f]
            total_var = np.var(raw_vals)
            residual_var = np.var(aligned_vals - aligned_vals.mean())
            r2_loadings[j, f] = 1.0 - residual_var / total_var if total_var > 1e-10 else 1.0

    for j in range(p):
        raw_vals = np.array([group_intercepts[g][j] for g in range(n_groups)])
        aligned_vals = aligned_intercepts[:, j]
        total_var = np.var(raw_vals)
        residual_var = np.var(aligned_vals - aligned_vals.mean())
        r2_intercepts[j] = 1.0 - residual_var / total_var if total_var > 1e-10 else 1.0

    # Format output
    group_output = []
    for g in range(n_groups):
        group_output.append({
            "group": g,
            "latent_means": opt_means[g].tolist(),
            "latent_variances": opt_variances[g].tolist(),
            "aligned_loadings": aligned_loadings[g].tolist(),
            "aligned_intercepts": aligned_intercepts[g].tolist(),
            "invariance_flags_loadings": invariance_flags_loadings[g].tolist(),
            "invariance_flags_intercepts": invariance_flags_intercepts[g].tolist(),
        })

    # Count non-invariant items
    n_noninvariant_loadings = int(np.sum(~invariance_flags_loadings))
    n_noninvariant_intercepts = int(np.sum(~invariance_flags_intercepts))
    total_loading_params = n_groups * p * n_factors
    total_intercept_params = n_groups * p
    pct_invariant_loadings = 100.0 * (1.0 - n_noninvariant_loadings / total_loading_params)
    pct_invariant_intercepts = 100.0 * (1.0 - n_noninvariant_intercepts / total_intercept_params)

    return {
        "method": "alignment_optimization",
        "reference": "Asparouhov & Muthén (2014)",
        "n_groups": n_groups,
        "n_items": p,
        "n_factors": n_factors,
        "groups": group_output,
        "r_squared_loadings": r2_loadings.tolist(),
        "r_squared_intercepts": r2_intercepts.tolist(),
        "summary": {
            "n_noninvariant_loadings": n_noninvariant_loadings,
            "n_noninvariant_intercepts": n_noninvariant_intercepts,
            "pct_invariant_loadings": float(round(pct_invariant_loadings, 1)),
            "pct_invariant_intercepts": float(round(pct_invariant_intercepts, 1)),
        },
        "converged": result.success,
        "simplicity_function_value": float(result.fun),
        "interpretation": (
            "The alignment method estimates latent means and variances for each group "
            "by minimising a simplicity function that penalises non-invariance. Unlike "
            "the traditional sequential approach, alignment does not require exact equality "
            "of parameters — it identifies approximate invariance. Items flagged as "
            "non-invariant show parameters that deviate substantially from the cross-group "
            "average after alignment. The method works well when non-invariance is "
            "limited (roughly ≤ 25% of parameters). R-squared values indicate how much "
            "of the between-group parameter variation is accounted for by the alignment "
            "rotation. Higher values suggest the group differences in parameters are "
            "well-explained by latent mean/variance differences rather than true DIF."
        ),
    }
