"""Multi-Trait Multi-Method Matrix Analysis — construct validity assessment.

Implements Campbell & Fiske (1959) criteria evaluation, convergent and
discriminant validity analysis, and CFA-based MTMM decomposition of
variance into trait, method, and error components.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class MTMMAnalyzeRequest(BaseModel):
    """Traditional MTMM matrix analysis with Campbell & Fiske criteria."""
    correlation_matrix: list[list[float]]
    variable_names: list[str]
    trait_labels: list[str]   # e.g. ["anxiety", "depression", "stress"]
    method_labels: list[str]  # e.g. ["self_report", "observer", "physiological"]
    variable_trait_map: dict[str, str]   # variable_name -> trait_label
    variable_method_map: dict[str, str]  # variable_name -> method_label


class MTMMCFARequest(BaseModel):
    """CFA-based MTMM analysis — correlated traits, correlated methods model."""
    correlation_matrix: list[list[float]]
    variable_names: list[str]
    trait_labels: list[str]
    method_labels: list[str]
    variable_trait_map: dict[str, str]
    variable_method_map: dict[str, str]
    sample_size: int


# ── Helpers ───────────────────────────────────────────────────────────

def _validate_matrix(corr: np.ndarray, n_vars: int):
    """Validate correlation matrix properties."""
    if corr.shape != (n_vars, n_vars):
        raise HTTPException(
            status_code=422,
            detail=f"Correlation matrix should be {n_vars}x{n_vars}, got {corr.shape}",
        )
    if not np.allclose(corr, corr.T, atol=0.01):
        raise HTTPException(status_code=422, detail="Correlation matrix must be symmetric")
    diag = np.diag(corr)
    if not np.allclose(diag, 1.0, atol=0.05):
        raise HTTPException(status_code=422, detail="Diagonal of correlation matrix should be ~1.0")


def _get_pair_type(v1: str, v2: str, trait_map: dict, method_map: dict) -> str:
    """Classify a variable pair as monotrait-heteromethod, heterotrait-monomethod,
    heterotrait-heteromethod, or monotrait-monomethod (reliability diagonal).
    """
    same_trait = trait_map[v1] == trait_map[v2]
    same_method = method_map[v1] == method_map[v2]

    if same_trait and same_method:
        return "reliability"  # same variable = diagonal
    elif same_trait and not same_method:
        return "monotrait_heteromethod"  # convergent validity
    elif not same_trait and same_method:
        return "heterotrait_monomethod"  # method effects / discriminant
    else:
        return "heterotrait_heteromethod"  # baseline


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/analyze")
def mtmm_analyze(req: MTMMAnalyzeRequest):
    """Evaluate Campbell & Fiske (1959) MTMM criteria for construct validity.

    Returns convergent validity coefficients, discriminant validity analysis,
    and pass/fail evaluation of all four criteria.
    """
    n_vars = len(req.variable_names)
    corr = np.array(req.correlation_matrix, dtype=float)
    _validate_matrix(corr, n_vars)

    var_names = req.variable_names
    trait_map = req.variable_trait_map
    method_map = req.variable_method_map

    # Validate mappings
    for v in var_names:
        if v not in trait_map:
            raise HTTPException(status_code=422, detail=f"Variable '{v}' missing from trait map")
        if v not in method_map:
            raise HTTPException(status_code=422, detail=f"Variable '{v}' missing from method map")
        if trait_map[v] not in req.trait_labels:
            raise HTTPException(status_code=422, detail=f"Trait '{trait_map[v]}' not in trait_labels")
        if method_map[v] not in req.method_labels:
            raise HTTPException(status_code=422, detail=f"Method '{method_map[v]}' not in method_labels")

    # Build correlation pair classifications
    var_idx = {v: i for i, v in enumerate(var_names)}

    pair_types: dict[str, list] = {
        "monotrait_heteromethod": [],
        "heterotrait_monomethod": [],
        "heterotrait_heteromethod": [],
    }

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            v1, v2 = var_names[i], var_names[j]
            ptype = _get_pair_type(v1, v2, trait_map, method_map)
            r = float(corr[i, j])
            if ptype != "reliability":
                pair_types[ptype].append({
                    "var1": v1, "var2": v2,
                    "trait1": trait_map[v1], "trait2": trait_map[v2],
                    "method1": method_map[v1], "method2": method_map[v2],
                    "r": round(r, 4),
                })

    # Summary statistics for each type
    mt_hm_rs = [p["r"] for p in pair_types["monotrait_heteromethod"]]
    ht_mm_rs = [p["r"] for p in pair_types["heterotrait_monomethod"]]
    ht_hm_rs = [p["r"] for p in pair_types["heterotrait_heteromethod"]]

    convergent_coefficients = pair_types["monotrait_heteromethod"]

    summary = {
        "monotrait_heteromethod": {
            "label": "Convergent Validity (same trait, different methods)",
            "n": len(mt_hm_rs),
            "mean_r": round(float(np.mean(mt_hm_rs)), 4) if mt_hm_rs else None,
            "median_r": round(float(np.median(mt_hm_rs)), 4) if mt_hm_rs else None,
            "range": [round(float(min(mt_hm_rs)), 4), round(float(max(mt_hm_rs)), 4)] if mt_hm_rs else None,
        },
        "heterotrait_monomethod": {
            "label": "Method Effects (different traits, same method)",
            "n": len(ht_mm_rs),
            "mean_r": round(float(np.mean(ht_mm_rs)), 4) if ht_mm_rs else None,
            "median_r": round(float(np.median(ht_mm_rs)), 4) if ht_mm_rs else None,
            "range": [round(float(min(ht_mm_rs)), 4), round(float(max(ht_mm_rs)), 4)] if ht_mm_rs else None,
        },
        "heterotrait_heteromethod": {
            "label": "Baseline (different traits, different methods)",
            "n": len(ht_hm_rs),
            "mean_r": round(float(np.mean(ht_hm_rs)), 4) if ht_hm_rs else None,
            "median_r": round(float(np.median(ht_hm_rs)), 4) if ht_hm_rs else None,
            "range": [round(float(min(ht_hm_rs)), 4), round(float(max(ht_hm_rs)), 4)] if ht_hm_rs else None,
        },
    }

    # ── Campbell & Fiske (1959) Four Criteria ──
    r_of = lambda u, v: float(corr[var_idx[u], var_idx[v]])

    # Criterion 1: validity values (monotrait-heteromethod) should be large enough to
    # warrant further examination. Significance would need the sample size, which this
    # endpoint does not take, so the common rule of thumb r > .30 is applied instead.
    if mt_hm_rs:
        crit1_pass = all(r > 0.30 for r in mt_hm_rs)
        crit1_detail = f"{sum(r > 0.30 for r in mt_hm_rs)}/{len(mt_hm_rs)} validity values exceed .30 (range {min(mt_hm_rs):.3f} to {max(mt_hm_rs):.3f})"
    else:
        crit1_pass = False
        crit1_detail = "No monotrait-heteromethod pairs found"

    # Criterion 2: each validity value r(T_a M_i, T_a M_j) should exceed the
    # heterotrait-heteromethod values in its own row and column of the same
    # heteromethod block (M_i, M_j): r(T_a M_i, T_b M_j) and r(T_b M_i, T_a M_j).
    # Criterion 3: it should also exceed the heterotrait-monomethod values involving
    # either of its two variables: r(T_a M_i, T_b M_i) and r(T_a M_j, T_b M_j).
    crit2_violations = crit2_total = 0
    crit3_violations = crit3_total = 0
    for conv in pair_types["monotrait_heteromethod"]:
        v1, v2 = conv["var1"], conv["var2"]
        m1, m2 = method_map[v1], method_map[v2]
        for other in var_names:
            if trait_map[other] == trait_map[v1]:
                continue
            for anchor, partner_method, same_method in ((v1, m2, m1), (v2, m1, m2)):
                if method_map[other] == partner_method:
                    crit2_total += 1
                    crit2_violations += conv["r"] <= r_of(anchor, other)
                if method_map[other] == same_method:
                    crit3_total += 1
                    crit3_violations += conv["r"] <= r_of(anchor, other)

    crit2_pass = crit2_violations == 0 if crit2_total > 0 else False
    crit2_detail = f"{crit2_violations}/{crit2_total} comparisons violated (validity value ≤ heterotrait-heteromethod value in the same row/column)"
    crit3_pass = crit3_violations == 0 if crit3_total > 0 else False
    crit3_detail = f"{crit3_violations}/{crit3_total} comparisons violated (validity value ≤ heterotrait-monomethod value)"

    # Criterion 4: the same pattern of trait interrelationships in every heterotrait
    # triangle, monomethod and heteromethod alike. Each triangle gives one correlation
    # per trait pair (heteromethod blocks average their two off-diagonal values); the
    # pattern is the rank order of those correlations.
    traits = list(req.trait_labels)
    trait_pairs = [(traits[i], traits[j]) for i in range(len(traits)) for j in range(i + 1, len(traits))]
    cell = {(trait_map[v], method_map[v]): v for v in var_names}
    methods = list(req.method_labels)
    triangles = {}
    for mi in methods:
        vals = [r_of(cell[(a_, mi)], cell[(b_, mi)]) if (a_, mi) in cell and (b_, mi) in cell else None for a_, b_ in trait_pairs]
        if None not in vals:
            triangles[f"{mi} (monomethod)"] = vals
    for i, mi in enumerate(methods):
        for mj in methods[i + 1:]:
            vals = []
            for a_, b_ in trait_pairs:
                keys = [(cell.get((a_, mi)), cell.get((b_, mj))), (cell.get((b_, mi)), cell.get((a_, mj)))]
                rs = [r_of(u, v) for u, v in keys if u and v]
                vals.append(float(np.mean(rs)) if rs else None)
            if None not in vals:
                triangles[f"{mi} × {mj} (heteromethod)"] = vals
    pattern_rho = None
    if len(trait_pairs) < 3:
        crit4_pass = None
        crit4_detail = "Not testable: a pattern needs at least 3 trait pairs (3+ traits)"
    elif len(triangles) < 2:
        crit4_pass = False
        crit4_detail = "Insufficient heterotrait triangles for comparison"
    else:
        ranks = {k: list(np.argsort(np.argsort(-np.array(v)))) for k, v in triangles.items()}
        first = next(iter(ranks.values()))
        crit4_pass = all(r_ == first for r_ in ranks.values())
        names_ = list(triangles)
        rhos = [sp_stats.spearmanr(triangles[names_[i]], triangles[names_[j]])[0]
                for i in range(len(names_)) for j in range(i + 1, len(names_))]
        pattern_rho = round(float(np.nanmean(rhos)), 4) if rhos else None
        crit4_detail = (f"Rank order of trait-pair correlations {'identical' if crit4_pass else 'differs'} across "
                        f"{len(triangles)} heterotrait triangles (mean Spearman ρ between triangles = {pattern_rho})")

    criteria = {
        "criterion_1": {
            "description": "Validity values (same trait, different methods) are sufficiently large (rule of thumb: all > .30)",
            "pass": crit1_pass,
            "detail": crit1_detail,
        },
        "criterion_2": {
            "description": "Each validity value exceeds the heterotrait-heteromethod values in its row and column",
            "pass": crit2_pass,
            "detail": crit2_detail,
        },
        "criterion_3": {
            "description": "Each validity value exceeds the heterotrait-monomethod values for its variables",
            "pass": crit3_pass,
            "detail": crit3_detail,
        },
        "criterion_4": {
            "description": "Same pattern of trait intercorrelations in all heterotrait triangles",
            "pass": crit4_pass,
            "detail": crit4_detail,
        },
        "overall_pass": bool(crit1_pass and crit2_pass and crit3_pass and crit4_pass is not False),
        "criteria_met": sum(bool(c) for c in (crit1_pass, crit2_pass, crit3_pass, crit4_pass)),
        "criteria_testable": sum(c is not None for c in (crit1_pass, crit2_pass, crit3_pass, crit4_pass)),
        "pattern_mean_spearman": pattern_rho,
    }

    # Discriminant validity ratios
    discriminant_ratios = []
    if mt_hm_rs and ht_mm_rs:
        avg_conv = float(np.mean(mt_hm_rs))
        avg_method = float(np.mean(ht_mm_rs))
        ratio = avg_conv / avg_method if avg_method != 0 else float("inf")
        discriminant_ratios.append({
            "comparison": "convergent_vs_method_effects",
            "avg_convergent_r": round(avg_conv, 4),
            "avg_method_effect_r": round(avg_method, 4),
            "ratio": round(float(ratio), 4),
            "adequate": ratio > 1.0,
        })

    return {
        "analysis": "mtmm_campbell_fiske",
        "n_variables": n_vars,
        "n_traits": len(req.trait_labels),
        "n_methods": len(req.method_labels),
        "summary": summary,
        "convergent_validity_coefficients": convergent_coefficients,
        "campbell_fiske_criteria": criteria,
        "discriminant_validity_ratios": discriminant_ratios,
        "all_pairs": pair_types,
    }


@router.post("/cfa")
def mtmm_cfa(req: MTMMCFARequest):
    """CFA-based MTMM: fit a correlated traits, correlated methods model.

    Decomposes variance into trait, method, and error components.
    Returns factor loadings and model fit indices.
    """
    n_vars = len(req.variable_names)
    R = np.array(req.correlation_matrix, dtype=float)
    _validate_matrix(R, n_vars)

    var_names = req.variable_names
    trait_map = req.variable_trait_map
    method_map = req.variable_method_map
    traits = req.trait_labels
    methods = req.method_labels
    N = req.sample_size

    n_traits = len(traits)
    n_methods = len(methods)

    # Validate
    for v in var_names:
        if v not in trait_map or v not in method_map:
            raise HTTPException(status_code=422, detail=f"Variable '{v}' missing from trait/method maps")

    # Build indicator matrices
    trait_idx = {t: i for i, t in enumerate(traits)}
    method_idx = {m: i for i, m in enumerate(methods)}

    # Lambda_trait: (n_vars, n_traits) — which trait each variable loads on
    Lambda_t_pattern = np.zeros((n_vars, n_traits))
    Lambda_m_pattern = np.zeros((n_vars, n_methods))
    for i, v in enumerate(var_names):
        Lambda_t_pattern[i, trait_idx[trait_map[v]]] = 1.0
        Lambda_m_pattern[i, method_idx[method_map[v]]] = 1.0

    # Parameters to estimate:
    # - trait loadings (one per variable)
    # - method loadings (one per variable)
    # - trait correlations (n_traits * (n_traits - 1) / 2)
    # - method correlations (n_methods * (n_methods - 1) / 2)
    # - uniquenesses (n_vars)

    n_trait_corr = n_traits * (n_traits - 1) // 2
    n_method_corr = n_methods * (n_methods - 1) // 2
    n_params = n_vars + n_vars + n_trait_corr + n_method_corr + n_vars

    def _unpack(params):
        idx = 0
        trait_loadings = params[idx:idx + n_vars]
        idx += n_vars
        method_loadings = params[idx:idx + n_vars]
        idx += n_vars
        trait_corr_vals = params[idx:idx + n_trait_corr]
        idx += n_trait_corr
        method_corr_vals = params[idx:idx + n_method_corr]
        idx += n_method_corr
        uniquenesses = params[idx:idx + n_vars]
        return trait_loadings, method_loadings, trait_corr_vals, method_corr_vals, uniquenesses

    def _build_sigma(params):
        tl, ml, tc_vals, mc_vals, uniq = _unpack(params)

        # Build trait factor loading matrix
        Lt = np.zeros((n_vars, n_traits))
        for i in range(n_vars):
            t_col = np.argmax(Lambda_t_pattern[i])
            Lt[i, t_col] = tl[i]

        # Build method factor loading matrix
        Lm = np.zeros((n_vars, n_methods))
        for i in range(n_vars):
            m_col = np.argmax(Lambda_m_pattern[i])
            Lm[i, m_col] = ml[i]

        # Build trait correlation matrix
        Phi_t = np.eye(n_traits)
        k = 0
        for i in range(n_traits):
            for j in range(i + 1, n_traits):
                Phi_t[i, j] = Phi_t[j, i] = np.tanh(tc_vals[k])  # constrain to (-1, 1)
                k += 1

        # Build method correlation matrix
        Phi_m = np.eye(n_methods)
        k = 0
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                Phi_m[i, j] = Phi_m[j, i] = np.tanh(mc_vals[k])
                k += 1

        # Uniqueness matrix
        Theta = np.diag(uniq ** 2)  # squared to ensure positive

        # Model-implied correlation matrix
        # Sigma = Lt @ Phi_t @ Lt.T + Lm @ Phi_m @ Lm.T + Theta
        Sigma = Lt @ Phi_t @ Lt.T + Lm @ Phi_m @ Lm.T + Theta

        return Sigma, Lt, Lm, Phi_t, Phi_m, Theta

    def _ml_objective(params):
        """Maximum likelihood discrepancy function."""
        try:
            Sigma, *_ = _build_sigma(params)
            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(Sigma)
            if np.any(eigvals <= 0):
                return 1e10

            Sigma_inv = np.linalg.inv(Sigma)
            log_det_sigma = np.log(np.linalg.det(Sigma))
            log_det_R = np.log(max(np.linalg.det(R), 1e-10))

            F = log_det_sigma + np.trace(R @ Sigma_inv) - log_det_R - n_vars
            return float(F) if np.isfinite(F) else 1e10
        except Exception:
            return 1e10

    # Initialize
    best_result = None
    best_fval = np.inf

    for start in range(5):
        rng = np.random.default_rng(seed=start * 31 + 13)
        x0 = np.concatenate([
            rng.uniform(0.3, 0.8, n_vars),   # trait loadings
            rng.uniform(0.1, 0.5, n_vars),   # method loadings
            rng.uniform(-0.2, 0.2, n_trait_corr),   # trait correlations (atanh space)
            rng.uniform(-0.2, 0.2, n_method_corr),  # method correlations (atanh space)
            rng.uniform(0.3, 0.6, n_vars),   # sqrt(uniquenesses)
        ])

        result = minimize(_ml_objective, x0, method="Nelder-Mead",
                          options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
        if result.fun < best_fval:
            best_fval = result.fun
            best_result = result

    if best_result is None or best_fval > 1e5:
        raise HTTPException(status_code=500, detail="CFA optimization failed to converge")

    # Extract final solution
    Sigma, Lt, Lm, Phi_t, Phi_m, Theta = _build_sigma(best_result.x)
    tl, ml, _, _, uniq = _unpack(best_result.x)

    # Variance decomposition per variable
    variance_decomposition = []
    for i, v in enumerate(var_names):
        t_col = np.argmax(Lambda_t_pattern[i])
        m_col = np.argmax(Lambda_m_pattern[i])
        trait_var = float(tl[i] ** 2)
        method_var = float(ml[i] ** 2)
        error_var = float(uniq[i] ** 2)
        total = trait_var + method_var + error_var
        variance_decomposition.append({
            "variable": v,
            "trait": trait_map[v],
            "method": method_map[v],
            "trait_loading": round(float(tl[i]), 4),
            "method_loading": round(float(ml[i]), 4),
            "trait_variance": round(trait_var, 4),
            "method_variance": round(method_var, 4),
            "error_variance": round(error_var, 4),
            "trait_proportion": round(trait_var / total, 4) if total > 0 else 0.0,
            "method_proportion": round(method_var / total, 4) if total > 0 else 0.0,
            "error_proportion": round(error_var / total, 4) if total > 0 else 0.0,
        })

    # Model fit indices
    F_ml = best_fval
    df = n_vars * (n_vars + 1) // 2 - n_params
    chi2 = max(0, (N - 1) * F_ml)

    # CFI
    # Null model: diagonal Sigma (independence model)
    F_null = float(np.log(np.linalg.det(np.eye(n_vars))) +
                   np.trace(R @ np.eye(n_vars)) -
                   np.log(max(np.linalg.det(R), 1e-10)) - n_vars)
    chi2_null = max(0, (N - 1) * F_null)
    df_null = n_vars * (n_vars - 1) // 2

    cfi = 1 - max(0, chi2 - df) / max(1, chi2_null - df_null)
    cfi = round(float(np.clip(cfi, 0, 1)), 4)

    # RMSEA
    if df > 0:
        rmsea = float(np.sqrt(max(0, (chi2 / df - 1) / (N - 1))))
    else:
        rmsea = 0.0
    rmsea = round(rmsea, 4)

    # SRMR
    residual = R - Sigma
    srmr_sum = 0.0
    count = 0
    for i in range(n_vars):
        for j in range(i + 1):
            srmr_sum += (residual[i, j] / max(abs(R[i, i] * R[j, j]) ** 0.5, 1e-10)) ** 2
            count += 1
    srmr = round(float(np.sqrt(srmr_sum / count)), 4)

    # Trait correlations
    trait_correlations = {}
    for i in range(n_traits):
        for j in range(i + 1, n_traits):
            trait_correlations[f"{traits[i]}_x_{traits[j]}"] = round(float(Phi_t[i, j]), 4)

    # Method correlations
    method_correlations = {}
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method_correlations[f"{methods[i]}_x_{methods[j]}"] = round(float(Phi_m[i, j]), 4)

    fit_indices = {
        "chi_square": round(float(chi2), 4),
        "df": max(0, df),
        "p_value": round(float(sp_stats.chi2.sf(chi2, max(1, df))), 4) if df > 0 else None,
        "cfi": cfi,
        "rmsea": rmsea,
        "srmr": srmr,
        "cfi_acceptable": cfi >= 0.90,
        "cfi_good": cfi >= 0.95,
        "rmsea_acceptable": rmsea <= 0.08,
        "rmsea_good": rmsea <= 0.06,
        "srmr_acceptable": srmr <= 0.08,
    }

    # Average variance proportions
    avg_trait_prop = float(np.mean([v["trait_proportion"] for v in variance_decomposition]))
    avg_method_prop = float(np.mean([v["method_proportion"] for v in variance_decomposition]))
    avg_error_prop = float(np.mean([v["error_proportion"] for v in variance_decomposition]))

    return {
        "analysis": "mtmm_cfa",
        "model": "correlated_traits_correlated_methods",
        "n_variables": n_vars,
        "n_traits": n_traits,
        "n_methods": n_methods,
        "sample_size": N,
        "variance_decomposition": variance_decomposition,
        "trait_correlations": trait_correlations,
        "method_correlations": method_correlations,
        "fit_indices": fit_indices,
        "average_variance_proportions": {
            "trait": round(avg_trait_prop, 4),
            "method": round(avg_method_prop, 4),
            "error": round(avg_error_prop, 4),
        },
        "interpretation": {
            "trait_variance": "Proportion of variance due to the construct being measured (higher = better construct validity)",
            "method_variance": "Proportion due to measurement method (lower = less method bias)",
            "error_variance": "Proportion due to random error (lower = more reliable)",
        },
    }
