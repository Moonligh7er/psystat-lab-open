"""Structural Equation Modeling — CFA, Path Analysis, and full SEM."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy import optimize, stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class CFARequest(BaseModel):
    """Confirmatory Factor Analysis."""
    data: list[dict]  # rows as dicts
    factors: dict[str, list[str]]  # factor_name -> [indicator_names]


class PathAnalysisRequest(BaseModel):
    """Path Analysis — specify directed paths between observed variables."""
    data: list[dict]
    paths: list[dict]  # [{"from": "x1", "to": "y"}, ...]
    mediators: list[str] | None = None


class SEMRequest(BaseModel):
    """Full SEM: measurement model + structural model."""
    data: list[dict]
    measurement: dict[str, list[str]]  # latent -> [indicators]
    structural: list[dict]  # [{"from": "latent1", "to": "latent2"}, ...]


# ── CFA Implementation ───────────────────────────────────────────────

def _compute_cov_matrix(df: pd.DataFrame, vars_list: list[str]) -> np.ndarray:
    return df[vars_list].cov().values


def _cfa_fit(cov_matrix: np.ndarray, n_indicators: list[int], n: int) -> dict:
    """Fit CFA using maximum likelihood estimation."""
    p = cov_matrix.shape[0]
    n_factors = len(n_indicators)

    # Initialize loadings
    loadings_flat = np.random.uniform(0.4, 0.8, sum(n_indicators))
    uniquenesses = np.full(p, 0.5)
    init_params = np.concatenate([loadings_flat, uniquenesses])

    def _build_loading_matrix(params):
        L = np.zeros((p, n_factors))
        idx = 0
        for f, ni in enumerate(n_indicators):
            start_row = sum(n_indicators[:f])
            for i in range(ni):
                L[start_row + i, f] = params[idx]
                idx += 1
        return L

    def _neg_log_likelihood(params):
        n_loadings = sum(n_indicators)
        L = _build_loading_matrix(params[:n_loadings])
        U = np.diag(np.abs(params[n_loadings:]) + 1e-6)
        sigma = L @ L.T + U

        try:
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                return 1e10
            sigma_inv = np.linalg.inv(sigma)
            tr = np.trace(sigma_inv @ cov_matrix)
            return 0.5 * (logdet + tr - np.linalg.slogdet(cov_matrix)[1] - p)
        except np.linalg.LinAlgError:
            return 1e10

    result = optimize.minimize(
        _neg_log_likelihood, init_params,
        method='L-BFGS-B',
        bounds=[(None, None)] * sum(n_indicators) + [(0.01, None)] * p,
        options={'maxiter': 1000},
    )

    n_loadings = sum(n_indicators)
    L = _build_loading_matrix(result.x[:n_loadings])
    U = np.diag(np.abs(result.x[n_loadings:]) + 1e-6)
    sigma_hat = L @ L.T + U

    # Fit indices
    fml = result.fun  # minimum of fitting function
    df_model = p * (p + 1) / 2 - len(result.x)

    # Chi-square
    chi2 = max((n - 1) * fml, 0)
    chi2_p = sp_stats.chi2.sf(chi2, max(df_model, 1)) if df_model > 0 else 1.0

    # CFI
    # Null model: diagonal covariance
    null_fml = 0.5 * (np.log(np.linalg.det(np.diag(np.diag(cov_matrix)))) +
                       np.trace(np.linalg.inv(np.diag(np.diag(cov_matrix))) @ cov_matrix) -
                       np.linalg.slogdet(cov_matrix)[1] - p)
    df_null = p * (p - 1) / 2
    chi2_null = max((n - 1) * null_fml, 0)

    cfi = 1 - max(chi2 - df_model, 0) / max(chi2_null - df_null, 0) if (chi2_null - df_null) > 0 else 1.0
    cfi = max(0, min(1, cfi))

    # RMSEA
    rmsea_val = np.sqrt(max((chi2 / max(df_model, 1) - 1) / max(n - 1, 1), 0)) if df_model > 0 else 0

    # SRMR
    residuals = cov_matrix - sigma_hat
    srmr = np.sqrt(np.mean(residuals[np.tril_indices(p)]**2))

    # TLI (Tucker-Lewis Index)
    tli = ((chi2_null / max(df_null, 1)) - (chi2 / max(df_model, 1))) / ((chi2_null / max(df_null, 1)) - 1) if (chi2_null / max(df_null, 1)) > 1 else 1.0

    return {
        "loadings": L.tolist(),
        "uniquenesses": np.abs(result.x[n_loadings:]).tolist(),
        "model_covariance": sigma_hat.tolist(),
        "residual_covariance": residuals.tolist(),
        "fit_indices": {
            "chi_square": float(chi2),
            "chi_square_df": float(df_model),
            "chi_square_p": float(chi2_p),
            "cfi": float(cfi),
            "tli": float(tli),
            "rmsea": float(rmsea_val),
            "srmr": float(srmr),
        },
        "converged": result.success,
        "n_iterations": result.nit,
    }


def _interpret_fit(fit: dict) -> dict:
    """Interpret fit indices against conventional cutoffs."""
    fi = fit["fit_indices"]
    assessments = {}

    assessments["chi_square"] = {
        "value": f"χ²({fi['chi_square_df']:.0f}) = {fi['chi_square']:.2f}, p = {fi['chi_square_p']:.3f}",
        "good": fi["chi_square_p"] > 0.05,
        "note": "Non-significant = good fit (sensitive to sample size)",
    }
    assessments["cfi"] = {
        "value": fi["cfi"],
        "good": fi["cfi"] >= 0.95,
        "acceptable": fi["cfi"] >= 0.90,
        "cutoff": "≥ .95 (good), ≥ .90 (acceptable)",
    }
    assessments["tli"] = {
        "value": fi["tli"],
        "good": fi["tli"] >= 0.95,
        "acceptable": fi["tli"] >= 0.90,
        "cutoff": "≥ .95 (good), ≥ .90 (acceptable)",
    }
    assessments["rmsea"] = {
        "value": fi["rmsea"],
        "good": fi["rmsea"] <= 0.06,
        "acceptable": fi["rmsea"] <= 0.08,
        "cutoff": "≤ .06 (good), ≤ .08 (acceptable)",
    }
    assessments["srmr"] = {
        "value": fi["srmr"],
        "good": fi["srmr"] <= 0.08,
        "cutoff": "≤ .08 (good)",
    }

    overall = "good" if all(a.get("good", False) for a in assessments.values()) else \
              "acceptable" if all(a.get("acceptable", a.get("good", False)) for a in assessments.values()) else \
              "poor"

    return {"assessments": assessments, "overall": overall}


@router.post("/cfa")
def confirmatory_factor_analysis(req: CFARequest):
    """Run Confirmatory Factor Analysis."""
    df = pd.DataFrame(req.data)
    if len(df) < 10:
        raise HTTPException(400, "CFA requires at least 10 observations.")

    all_indicators = []
    n_indicators = []
    factor_names = list(req.factors.keys())

    for factor_name, indicators in req.factors.items():
        for ind in indicators:
            if ind not in df.columns:
                raise HTTPException(400, f"Indicator '{ind}' not found in data.")
        all_indicators.extend(indicators)
        n_indicators.append(len(indicators))

    if len(all_indicators) < 3:
        raise HTTPException(400, "CFA requires at least 3 indicators.")

    # Drop rows with missing data
    df_clean = df[all_indicators].dropna()
    if len(df_clean) < 10:
        raise HTTPException(400, "Too few complete observations after removing missing data.")

    cov_matrix = _compute_cov_matrix(df_clean, all_indicators)
    fit_result = _cfa_fit(cov_matrix, n_indicators, len(df_clean))

    # Format loadings by factor
    loadings_by_factor = {}
    idx = 0
    for f, factor_name in enumerate(factor_names):
        indicators = req.factors[factor_name]
        loadings_by_factor[factor_name] = [
            {"indicator": indicators[i], "loading": fit_result["loadings"][idx + i][f],
             "communality": fit_result["loadings"][idx + i][f]**2}
            for i in range(len(indicators))
        ]
        idx += len(indicators)

    # Factor correlations (from loading matrix)
    L = np.array(fit_result["loadings"])
    factor_cov = L.T @ L
    D = np.sqrt(np.diag(factor_cov))
    factor_corr = factor_cov / np.outer(D, D + 1e-10)

    interpretation = _interpret_fit(fit_result)

    return {
        "n_observations": len(df_clean),
        "n_factors": len(factor_names),
        "factor_names": factor_names,
        "loadings": loadings_by_factor,
        "uniquenesses": {all_indicators[i]: fit_result["uniquenesses"][i] for i in range(len(all_indicators))},
        "factor_correlations": {
            f"{factor_names[i]}-{factor_names[j]}": float(factor_corr[i][j])
            for i in range(len(factor_names))
            for j in range(i + 1, len(factor_names))
        },
        "fit_indices": fit_result["fit_indices"],
        "fit_interpretation": interpretation,
        "converged": fit_result["converged"],
    }


@router.post("/path-analysis")
def path_analysis(req: PathAnalysisRequest):
    """Run path analysis with direct and indirect effects."""
    df = pd.DataFrame(req.data)
    if len(df) < 10:
        raise HTTPException(400, "Path analysis requires at least 10 observations.")

    # Collect all variables
    all_vars = set()
    for path in req.paths:
        all_vars.add(path["from"])
        all_vars.add(path["to"])

    for v in all_vars:
        if v not in df.columns:
            raise HTTPException(400, f"Variable '{v}' not found in data.")

    df_clean = df[list(all_vars)].dropna()

    # Estimate each path via OLS
    import statsmodels.api as sm

    path_results = []
    for path in req.paths:
        y = df_clean[path["to"]].values
        X = sm.add_constant(df_clean[path["from"]].values)
        model = sm.OLS(y, X).fit()

        path_results.append({
            "from": path["from"],
            "to": path["to"],
            "coefficient": float(model.params[1]),
            "std_error": float(model.bse[1]),
            "t_value": float(model.tvalues[1]),
            "p_value": float(model.pvalues[1]),
            "significant": bool(model.pvalues[1] < 0.05),
            "r_squared": float(model.rsquared),
        })

    # Compute indirect effects through mediators
    indirect_effects = []
    if req.mediators:
        for mediator in req.mediators:
            # Find paths: X -> M and M -> Y
            a_paths = [p for p in path_results if p["to"] == mediator]
            b_paths = [p for p in path_results if p["from"] == mediator]

            for a in a_paths:
                for b in b_paths:
                    indirect = a["coefficient"] * b["coefficient"]
                    # Sobel SE
                    se = np.sqrt(a["coefficient"]**2 * b["std_error"]**2 + b["coefficient"]**2 * a["std_error"]**2)
                    z = indirect / se if se > 0 else 0
                    p = 2 * (sp_stats.norm.sf(abs(z)))

                    indirect_effects.append({
                        "from": a["from"],
                        "through": mediator,
                        "to": b["to"],
                        "indirect_effect": float(indirect),
                        "sobel_se": float(se),
                        "sobel_z": float(z),
                        "p_value": float(p),
                        "significant": bool(p < 0.05),
                    })

    # Total effects
    total_effects = {}
    for p in path_results:
        key = f"{p['from']} -> {p['to']}"
        total_effects[key] = {"direct": p["coefficient"]}

    for ie in indirect_effects:
        key = f"{ie['from']} -> {ie['to']}"
        if key not in total_effects:
            total_effects[key] = {"direct": 0}
        total_effects[key]["indirect"] = total_effects[key].get("indirect", 0) + ie["indirect_effect"]
        total_effects[key]["total"] = total_effects[key]["direct"] + total_effects[key].get("indirect", 0)

    return {
        "n_observations": len(df_clean),
        "n_paths": len(req.paths),
        "direct_effects": path_results,
        "indirect_effects": indirect_effects,
        "total_effects": total_effects,
    }


@router.post("/sem")
def structural_equation_model(req: SEMRequest):
    """Full SEM: measurement model (CFA) + structural paths between latents."""
    df = pd.DataFrame(req.data)

    # Step 1: Run CFA for measurement model
    cfa_req = CFARequest(data=req.data, factors=req.measurement)
    cfa_result = confirmatory_factor_analysis(cfa_req)

    # Step 2: Create factor scores (regression method)
    all_indicators = []
    for indicators in req.measurement.values():
        all_indicators.extend(indicators)

    df_clean = df[all_indicators].dropna()
    factor_names = list(req.measurement.keys())

    # Simple factor scores: weighted sum of standardized indicators
    factor_scores = pd.DataFrame(index=df_clean.index)
    for factor_name, indicators in req.measurement.items():
        z_scores = df_clean[indicators].apply(sp_stats.zscore)
        loadings = [l["loading"] for l in cfa_result["loadings"][factor_name]]
        weights = np.array(loadings) / sum(np.abs(loadings))
        factor_scores[factor_name] = z_scores.values @ weights

    # Step 3: Estimate structural paths between factor scores
    import statsmodels.api as sm
    structural_results = []
    for path in req.structural:
        f_from = path["from"]
        f_to = path["to"]
        y = factor_scores[f_to].values
        X = sm.add_constant(factor_scores[f_from].values)
        model = sm.OLS(y, X).fit()

        structural_results.append({
            "from": f_from,
            "to": f_to,
            "coefficient": float(model.params[1]),
            "std_error": float(model.bse[1]),
            "t_value": float(model.tvalues[1]),
            "p_value": float(model.pvalues[1]),
            "significant": bool(model.pvalues[1] < 0.05),
            "r_squared": float(model.rsquared),
        })

    return {
        "n_observations": cfa_result["n_observations"],
        "measurement_model": {
            "loadings": cfa_result["loadings"],
            "fit_indices": cfa_result["fit_indices"],
            "fit_interpretation": cfa_result["fit_interpretation"],
        },
        "structural_model": {
            "paths": structural_results,
        },
        "factor_correlations": cfa_result.get("factor_correlations", {}),
    }


# ── Measurement Invariance ────────────────────────────────────────────


class MeasurementInvarianceRequest(BaseModel):
    """Multi-group CFA invariance testing."""
    data_group1: list[list[float]]  # rows = participants, cols = indicators
    data_group2: list[list[float]]
    n_factors: int = 1


class LatentGrowthRequest(BaseModel):
    """Latent growth curve model."""
    data: list[list[float]]  # rows = persons, cols = timepoints
    time_points: list[float]  # e.g. [0, 1, 2, 3]


def _compute_fit_indices_from_cov(
    cov_observed: np.ndarray,
    cov_implied: np.ndarray,
    n: int,
    n_free_params: int,
) -> dict:
    """Compute CFA-style fit indices from observed and implied covariance matrices."""
    p = cov_observed.shape[0]
    df_model = max(p * (p + 1) / 2 - n_free_params, 1)

    # ML fitting function
    try:
        sign, logdet_implied = np.linalg.slogdet(cov_implied)
        if sign <= 0:
            logdet_implied = 0
        sign_obs, logdet_obs = np.linalg.slogdet(cov_observed)
        if sign_obs <= 0:
            logdet_obs = 0
        cov_inv = np.linalg.inv(cov_implied)
        tr = np.trace(cov_inv @ cov_observed)
        fml = logdet_implied + tr - logdet_obs - p
        fml = max(fml, 0)
    except np.linalg.LinAlgError:
        fml = 1.0

    chi2 = max((n - 1) * fml, 0)
    chi2_p = sp_stats.chi2.sf(chi2, max(df_model, 1)) if df_model > 0 else 1.0

    # Null model (diagonal)
    diag_cov = np.diag(np.diag(cov_observed))
    try:
        null_fml = (np.log(np.linalg.det(diag_cov)) +
                    np.trace(np.linalg.inv(diag_cov) @ cov_observed) -
                    logdet_obs - p)
        null_fml = max(null_fml, 0)
    except np.linalg.LinAlgError:
        null_fml = 1.0

    df_null = p * (p - 1) / 2
    chi2_null = max((n - 1) * null_fml, 0)

    cfi = 1 - max(chi2 - df_model, 0) / max(chi2_null - df_null, 1e-10)
    cfi = max(0.0, min(1.0, cfi))

    rmsea = np.sqrt(max((chi2 / max(df_model, 1) - 1) / max(n - 1, 1), 0))

    residuals = cov_observed - cov_implied
    srmr = np.sqrt(np.mean(residuals[np.tril_indices(p)]**2))

    return {
        "chi_square": float(chi2),
        "df": float(df_model),
        "chi_square_p": float(chi2_p),
        "cfi": float(cfi),
        "rmsea": float(rmsea),
        "srmr": float(srmr),
    }


def _fit_cfa_single_group(
    cov_matrix: np.ndarray,
    n: int,
    n_factors: int,
    fixed_loadings: np.ndarray | None = None,
    fixed_intercepts: np.ndarray | None = None,
    fixed_residuals: np.ndarray | None = None,
) -> dict:
    """
    Fit a CFA model to a covariance matrix using ML.
    Optionally fix loadings, intercepts, or residual variances (for invariance testing).
    Returns loadings, uniquenesses, implied covariance, and fit indices.
    """
    p = cov_matrix.shape[0]

    # Use eigendecomposition to get initial loadings
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    init_loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0.01))

    if fixed_loadings is not None:
        loadings = fixed_loadings.copy()
    else:
        loadings = init_loadings.copy()

    if fixed_residuals is not None:
        uniquenesses = fixed_residuals.copy()
    else:
        uniquenesses = np.maximum(np.diag(cov_matrix) - np.sum(init_loadings**2, axis=1), 0.01)

    # If not fixed, optimize loadings and/or uniquenesses
    params_to_optimize = []
    n_loading_params = 0
    n_unique_params = 0

    if fixed_loadings is None:
        n_loading_params = p * n_factors
        params_to_optimize.extend(loadings.flatten().tolist())
    if fixed_residuals is None:
        n_unique_params = p
        params_to_optimize.extend(uniquenesses.tolist())

    if len(params_to_optimize) > 0:
        def _neg_ll(params):
            idx = 0
            if fixed_loadings is None:
                L = params[idx:idx + n_loading_params].reshape(p, n_factors)
                idx += n_loading_params
            else:
                L = fixed_loadings

            if fixed_residuals is None:
                U = np.diag(np.abs(params[idx:idx + n_unique_params]) + 1e-6)
            else:
                U = np.diag(fixed_residuals)

            sigma = L @ L.T + U
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
        if fixed_loadings is None:
            bounds.extend([(None, None)] * n_loading_params)
        if fixed_residuals is None:
            bounds.extend([(0.001, None)] * n_unique_params)

        result = optimize.minimize(
            _neg_ll, params_to_optimize,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500},
        )

        idx = 0
        if fixed_loadings is None:
            loadings = result.x[idx:idx + n_loading_params].reshape(p, n_factors)
            idx += n_loading_params
        if fixed_residuals is None:
            uniquenesses = np.abs(result.x[idx:idx + n_unique_params]) + 1e-6

    implied_cov = loadings @ loadings.T + np.diag(uniquenesses)
    n_free = (0 if fixed_loadings is not None else p * n_factors) + \
             (0 if fixed_residuals is not None else p)

    fit = _compute_fit_indices_from_cov(cov_matrix, implied_cov, n, n_free)

    return {
        "loadings": loadings,
        "uniquenesses": uniquenesses,
        "implied_cov": implied_cov,
        "fit": fit,
    }


@router.post("/measurement-invariance")
def measurement_invariance(req: MeasurementInvarianceRequest):
    """Multi-group CFA measurement invariance testing (configural, metric, scalar, strict)."""
    data1 = np.array(req.data_group1)
    data2 = np.array(req.data_group2)

    if data1.ndim != 2 or data2.ndim != 2:
        raise HTTPException(400, "Both groups must provide 2D data arrays.")
    if data1.shape[1] != data2.shape[1]:
        raise HTTPException(400, "Both groups must have the same number of indicators.")

    n1, p = data1.shape
    n2 = data2.shape[0]

    if n1 < 10 or n2 < 10:
        raise HTTPException(400, "Each group needs at least 10 observations.")
    if req.n_factors < 1:
        raise HTTPException(400, "n_factors must be at least 1.")
    if req.n_factors >= p:
        raise HTTPException(400, "n_factors must be less than the number of indicators.")

    cov1 = np.cov(data1, rowvar=False)
    cov2 = np.cov(data2, rowvar=False)
    mean1 = data1.mean(axis=0)
    mean2 = data2.mean(axis=0)

    levels = []

    # ── Level 1: Configural invariance (same structure, free params) ──
    fit1_config = _fit_cfa_single_group(cov1, n1, req.n_factors)
    fit2_config = _fit_cfa_single_group(cov2, n2, req.n_factors)

    # Combined fit: weighted average of fit indices
    w1, w2 = n1 / (n1 + n2), n2 / (n1 + n2)
    config_cfi = w1 * fit1_config["fit"]["cfi"] + w2 * fit2_config["fit"]["cfi"]
    config_rmsea = np.sqrt(w1 * fit1_config["fit"]["rmsea"]**2 + w2 * fit2_config["fit"]["rmsea"]**2)
    config_srmr = w1 * fit1_config["fit"]["srmr"] + w2 * fit2_config["fit"]["srmr"]

    levels.append({
        "level": "configural",
        "cfi": float(config_cfi),
        "rmsea": float(config_rmsea),
        "srmr": float(config_srmr),
        "delta_cfi": None,
        "delta_rmsea": None,
        "supported": True,  # configural is the baseline
    })

    # ── Level 2: Metric invariance (constrain loadings equal) ──
    # Use group 1 loadings as the common loadings
    common_loadings = fit1_config["loadings"]
    fit1_metric = _fit_cfa_single_group(cov1, n1, req.n_factors, fixed_loadings=common_loadings)
    fit2_metric = _fit_cfa_single_group(cov2, n2, req.n_factors, fixed_loadings=common_loadings)

    metric_cfi = w1 * fit1_metric["fit"]["cfi"] + w2 * fit2_metric["fit"]["cfi"]
    metric_rmsea = np.sqrt(w1 * fit1_metric["fit"]["rmsea"]**2 + w2 * fit2_metric["fit"]["rmsea"]**2)
    metric_srmr = w1 * fit1_metric["fit"]["srmr"] + w2 * fit2_metric["fit"]["srmr"]

    delta_cfi_metric = config_cfi - metric_cfi
    delta_rmsea_metric = metric_rmsea - config_rmsea
    metric_supported = abs(delta_cfi_metric) < 0.01 and abs(delta_rmsea_metric) < 0.015

    levels.append({
        "level": "metric",
        "cfi": float(metric_cfi),
        "rmsea": float(metric_rmsea),
        "srmr": float(metric_srmr),
        "delta_cfi": float(delta_cfi_metric),
        "delta_rmsea": float(delta_rmsea_metric),
        "supported": bool(metric_supported),
    })

    # ── Level 3: Scalar invariance (constrain loadings + intercepts) ──
    # For covariance-based approach, scalar invariance involves constraining
    # indicator means. We test whether mean differences are explained by factor means.
    # Approximate: compare mean structures
    mean_diff = mean1 - mean2
    predicted_mean_diff = common_loadings @ np.linalg.lstsq(common_loadings, mean_diff, rcond=None)[0]
    residual_mean_diff = mean_diff - predicted_mean_diff
    intercept_misfit = np.sqrt(np.mean(residual_mean_diff**2))

    # Scalar invariance uses same fit as metric (covariance-based) with penalty for intercept misfit
    scalar_cfi = metric_cfi - intercept_misfit * 0.02  # small degradation from intercept constraints
    scalar_cfi = max(0.0, min(1.0, scalar_cfi))
    scalar_rmsea = np.sqrt(metric_rmsea**2 + intercept_misfit**2 * 0.001)
    scalar_srmr = metric_srmr + intercept_misfit * 0.01

    delta_cfi_scalar = metric_cfi - scalar_cfi
    delta_rmsea_scalar = scalar_rmsea - metric_rmsea
    scalar_supported = metric_supported and abs(delta_cfi_scalar) < 0.01 and abs(delta_rmsea_scalar) < 0.015

    levels.append({
        "level": "scalar",
        "cfi": float(scalar_cfi),
        "rmsea": float(scalar_rmsea),
        "srmr": float(scalar_srmr),
        "delta_cfi": float(delta_cfi_scalar),
        "delta_rmsea": float(delta_rmsea_scalar),
        "supported": bool(scalar_supported),
    })

    # ── Level 4: Strict invariance (constrain loadings + intercepts + residuals) ──
    common_residuals = (fit1_config["uniquenesses"] + fit2_config["uniquenesses"]) / 2
    fit1_strict = _fit_cfa_single_group(
        cov1, n1, req.n_factors,
        fixed_loadings=common_loadings, fixed_residuals=common_residuals,
    )
    fit2_strict = _fit_cfa_single_group(
        cov2, n2, req.n_factors,
        fixed_loadings=common_loadings, fixed_residuals=common_residuals,
    )

    strict_cfi = w1 * fit1_strict["fit"]["cfi"] + w2 * fit2_strict["fit"]["cfi"]
    strict_rmsea = np.sqrt(w1 * fit1_strict["fit"]["rmsea"]**2 + w2 * fit2_strict["fit"]["rmsea"]**2)
    strict_srmr = w1 * fit1_strict["fit"]["srmr"] + w2 * fit2_strict["fit"]["srmr"]

    delta_cfi_strict = scalar_cfi - strict_cfi
    delta_rmsea_strict = strict_rmsea - scalar_rmsea
    strict_supported = scalar_supported and abs(delta_cfi_strict) < 0.01 and abs(delta_rmsea_strict) < 0.015

    levels.append({
        "level": "strict",
        "cfi": float(strict_cfi),
        "rmsea": float(strict_rmsea),
        "srmr": float(strict_srmr),
        "delta_cfi": float(delta_cfi_strict),
        "delta_rmsea": float(delta_rmsea_strict),
        "supported": bool(strict_supported),
    })

    # Build conclusion
    highest_supported = "configural"
    for level in levels:
        if level["supported"]:
            highest_supported = level["level"]

    conclusion_map = {
        "configural": "Only configural invariance is supported. The same factor structure holds across groups, but loadings differ — factor means cannot be meaningfully compared.",
        "metric": "Metric invariance is supported. Factor loadings are equivalent across groups, allowing comparison of structural relationships but not latent means.",
        "scalar": "Scalar invariance is supported. Both loadings and intercepts are equivalent, allowing meaningful comparison of latent factor means across groups.",
        "strict": "Full strict invariance is supported. Loadings, intercepts, and residual variances are equivalent across groups — observed score comparisons are valid.",
    }

    return {
        "levels": levels,
        "highest_supported_level": highest_supported,
        "conclusion": conclusion_map[highest_supported],
        "n_group1": n1,
        "n_group2": n2,
        "n_indicators": p,
        "n_factors": req.n_factors,
    }


# ── Latent Growth Curve Model ─────────────────────────────────────────


@router.post("/latent-growth")
def latent_growth(req: LatentGrowthRequest):
    """Latent growth curve model — individual growth trajectories with intercept and slope."""
    data = np.array(req.data)
    time_points = np.array(req.time_points, dtype=float)

    if data.ndim != 2:
        raise HTTPException(400, "Data must be a 2D array (persons × timepoints).")

    n_persons, n_timepoints = data.shape

    if n_persons < 5:
        raise HTTPException(400, "Need at least 5 persons.")
    if n_timepoints < 3:
        raise HTTPException(400, "Need at least 3 timepoints for a growth model.")
    if len(time_points) != n_timepoints:
        raise HTTPException(400, "Length of time_points must match number of columns in data.")

    # ── Fit individual OLS regressions ──
    intercepts = np.zeros(n_persons)
    slopes = np.zeros(n_persons)
    r_squared_per_person = np.zeros(n_persons)
    residuals_all = np.zeros_like(data)

    for j in range(n_persons):
        y = data[j, :]
        # Check for missing values
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            intercepts[j] = np.nan
            slopes[j] = np.nan
            r_squared_per_person[j] = np.nan
            continue

        t_valid = time_points[valid]
        y_valid = y[valid]

        # OLS: y = intercept + slope * t
        X = np.column_stack([np.ones(len(t_valid)), t_valid])
        beta = np.linalg.lstsq(X, y_valid, rcond=None)[0]
        intercepts[j] = beta[0]
        slopes[j] = beta[1]

        y_hat = X @ beta
        ss_res = np.sum((y_valid - y_hat)**2)
        ss_tot = np.sum((y_valid - y_valid.mean())**2)
        r_squared_per_person[j] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Full residuals (including positions with data)
        y_hat_full = intercepts[j] + slopes[j] * time_points
        residuals_all[j, :] = y - y_hat_full

    # Remove persons with NaN trajectories
    valid_persons = ~(np.isnan(intercepts) | np.isnan(slopes))
    intercepts_valid = intercepts[valid_persons]
    slopes_valid = slopes[valid_persons]
    r_sq_valid = r_squared_per_person[valid_persons]
    n_valid = int(valid_persons.sum())

    if n_valid < 3:
        raise HTTPException(400, "Too few persons with sufficient data for growth model.")

    # ── Summary statistics ──
    intercept_mean = float(np.mean(intercepts_valid))
    intercept_var = float(np.var(intercepts_valid, ddof=1))
    intercept_se = float(np.std(intercepts_valid, ddof=1) / np.sqrt(n_valid))

    slope_mean = float(np.mean(slopes_valid))
    slope_var = float(np.var(slopes_valid, ddof=1))
    slope_se = float(np.std(slopes_valid, ddof=1) / np.sqrt(n_valid))

    # Intercept-slope correlation
    if intercept_var > 1e-10 and slope_var > 1e-10:
        corr_is = float(np.corrcoef(intercepts_valid, slopes_valid)[0, 1])
        cov_is = float(np.cov(intercepts_valid, slopes_valid)[0, 1])
    else:
        corr_is = 0.0
        cov_is = 0.0

    # ── Model fit ──
    mean_r_squared = float(np.mean(r_sq_valid))
    residual_variance = float(np.var(residuals_all[valid_persons], ddof=1))

    # ICC: proportion of total variance explained by between-person differences
    # ICC = between-person variance / (between-person variance + within-person variance)
    grand_mean = np.nanmean(data[valid_persons])
    person_means = np.nanmean(data[valid_persons], axis=1)
    between_var = np.var(person_means, ddof=1)
    within_vars = [np.nanvar(data[valid_persons][j, :], ddof=1) for j in range(n_valid)]
    within_var = np.mean(within_vars)
    icc = float(between_var / (between_var + within_var)) if (between_var + within_var) > 0 else 0.0

    # ── Individual trajectories ──
    individual_trajectories = []
    person_idx = 0
    for j in range(n_persons):
        if valid_persons[j]:
            individual_trajectories.append({
                "person": int(j),
                "intercept": float(intercepts[j]),
                "slope": float(slopes[j]),
                "r_squared": float(r_squared_per_person[j]),
            })

    return {
        "intercept": {
            "mean": intercept_mean,
            "variance": intercept_var,
            "se": intercept_se,
        },
        "slope": {
            "mean": slope_mean,
            "variance": slope_var,
            "se": slope_se,
        },
        "correlation": corr_is,
        "covariance": cov_is,
        "individual_trajectories": individual_trajectories,
        "model_fit": {
            "r_squared": mean_r_squared,
            "residual_variance": residual_variance,
            "icc": icc,
        },
        "n_persons": n_valid,
        "n_timepoints": n_timepoints,
        "time_points": time_points.tolist(),
    }
