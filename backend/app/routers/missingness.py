"""Missing data diagnostics and imputation comparison."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DiagnoseRequest(BaseModel):
    data: dict[str, list]  # column_name -> list of values (None = missing)


class ImputationComparisonRequest(BaseModel):
    data: list  # list of float | None
    methods: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float_or_nan(val) -> float:
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _safe_skewness(arr: np.ndarray) -> float:
    clean = arr[~np.isnan(arr)]
    if len(clean) < 3:
        return 0.0
    return float(sp_stats.skew(clean, nan_policy="omit"))


# ---------------------------------------------------------------------------
# POST /diagnose
# ---------------------------------------------------------------------------

@router.post("/diagnose")
def diagnose_missingness(req: DiagnoseRequest):
    """Diagnose missing data mechanism and patterns."""
    if not req.data:
        raise HTTPException(400, "data must contain at least one column.")

    col_names = list(req.data.keys())
    n_cols = len(col_names)

    # Build numeric matrix (rows = observations, cols = variables)
    n_rows = max(len(v) for v in req.data.values()) if req.data else 0
    if n_rows == 0:
        raise HTTPException(400, "All columns are empty.")

    # Pad shorter columns with None
    matrix = np.full((n_rows, n_cols), np.nan)
    for j, col in enumerate(col_names):
        for i, val in enumerate(req.data[col]):
            matrix[i, j] = _to_float_or_nan(val)

    # --- Per-column summary ---
    column_summary = []
    for j, col in enumerate(col_names):
        n_miss = int(np.sum(np.isnan(matrix[:, j])))
        pct = round(n_miss / n_rows * 100, 2) if n_rows > 0 else 0.0
        column_summary.append({
            "column": col,
            "n_missing": n_miss,
            "pct_missing": pct,
        })

    # --- Missingness indicator matrix ---
    miss_ind = np.isnan(matrix).astype(float)  # 1 = missing, 0 = observed

    # --- Little's MCAR test (approximation) ---
    # Strategy: For each column with missing data, test if its missingness indicator
    # is related to the observed values of other columns via chi-square.
    mcar_chi2 = 0.0
    mcar_df = 0

    for j in range(n_cols):
        if np.all(miss_ind[:, j] == 0) or np.all(miss_ind[:, j] == 1):
            continue  # no variation in missingness for this column

        indicator = miss_ind[:, j]
        for k in range(n_cols):
            if k == j:
                continue
            observed_col = matrix[:, k]
            # Only use rows where column k is observed
            valid_mask = ~np.isnan(observed_col)
            if np.sum(valid_mask) < 4:
                continue

            obs_vals = observed_col[valid_mask]
            ind_vals = indicator[valid_mask]

            # Split observed values by missingness indicator of column j
            group0 = obs_vals[ind_vals == 0]
            group1 = obs_vals[ind_vals == 1]

            if len(group0) < 2 or len(group1) < 2:
                continue

            # Use t-test statistic squared as chi-square contribution (1 df)
            try:
                t_stat, _ = sp_stats.ttest_ind(group0, group1, equal_var=False)
                if np.isfinite(t_stat):
                    mcar_chi2 += t_stat ** 2
                    mcar_df += 1
            except Exception:
                continue

    if mcar_df > 0:
        mcar_p = float(sp_stats.chi2.sf(mcar_chi2, mcar_df))
        mcar_conclusion = "MCAR supported" if mcar_p >= 0.05 else "MCAR not supported"
    else:
        mcar_p = None
        mcar_conclusion = "Insufficient data to test MCAR"

    mcar_test = {
        "chi_square": round(float(mcar_chi2), 4) if mcar_df > 0 else None,
        "df": mcar_df if mcar_df > 0 else None,
        "p_value": round(mcar_p, 4) if mcar_p is not None else None,
        "conclusion": mcar_conclusion,
    }

    # --- Pairwise correlations of missingness indicators with observed variables ---
    missingness_correlations: dict[str, dict[str, float | None]] = {}
    for j in range(n_cols):
        if np.all(miss_ind[:, j] == 0) or np.all(miss_ind[:, j] == 1):
            continue
        corr_dict: dict[str, float | None] = {}
        for k in range(n_cols):
            if k == j:
                continue
            valid_mask = ~np.isnan(matrix[:, k])
            if np.sum(valid_mask) < 3:
                corr_dict[col_names[k]] = None
                continue
            try:
                r, _ = sp_stats.pointbiserialr(miss_ind[valid_mask, j], matrix[valid_mask, k])
                corr_dict[col_names[k]] = round(float(r), 4) if np.isfinite(r) else None
            except Exception:
                corr_dict[col_names[k]] = None
        if corr_dict:
            missingness_correlations[f"missing_{col_names[j]}"] = corr_dict

    # --- Missing data patterns ---
    # Each pattern is a tuple of which columns are missing
    pattern_counts: dict[tuple, int] = {}
    for i in range(n_rows):
        pattern = tuple(int(np.isnan(matrix[i, j])) for j in range(n_cols))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Sort by frequency, return top patterns
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
    patterns = []
    for pat, count in sorted_patterns[:20]:
        missing_cols = [col_names[j] for j in range(n_cols) if pat[j] == 1]
        patterns.append({
            "missing_columns": missing_cols,
            "count": count,
            "pct": round(count / n_rows * 100, 2),
        })

    # --- Recommendation ---
    total_missing_pct = sum(cs["pct_missing"] for cs in column_summary) / n_cols if n_cols > 0 else 0
    has_high_corr = any(
        abs(v) > 0.3
        for corrs in missingness_correlations.values()
        for v in corrs.values()
        if v is not None
    )

    if total_missing_pct < 5 and mcar_conclusion == "MCAR supported":
        recommendation = (
            "Low missingness (<5%) and MCAR supported. Listwise deletion is acceptable. "
            "Mean/median imputation is also safe."
        )
    elif mcar_conclusion == "MCAR supported":
        recommendation = (
            "MCAR supported but notable missingness. Use multiple imputation (MI) or "
            "mean/median imputation for simple analyses."
        )
    elif has_high_corr:
        recommendation = (
            "Missingness appears related to observed values (MAR likely). "
            "Use multiple imputation (MICE) or maximum likelihood estimation. "
            "Avoid listwise deletion — it may introduce bias."
        )
    else:
        recommendation = (
            "MCAR not supported. The mechanism may be MAR or MNAR. "
            "Use multiple imputation (MICE). If MNAR is suspected, consider "
            "pattern-mixture models or sensitivity analyses."
        )

    return {
        "column_summary": column_summary,
        "mcar_test": mcar_test,
        "missingness_correlations": missingness_correlations,
        "patterns": patterns,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# POST /imputation-comparison
# ---------------------------------------------------------------------------

@router.post("/imputation-comparison")
def imputation_comparison(req: ImputationComparisonRequest):
    """Compare imputation methods on a single variable."""
    if not req.data:
        raise HTTPException(400, "data must not be empty.")

    methods = req.methods or ["listwise", "mean", "median", "regression"]

    # Parse data
    raw = np.array([_to_float_or_nan(v) for v in req.data])
    observed_mask = ~np.isnan(raw)
    observed = raw[observed_mask]
    n_missing = int(np.sum(~observed_mask))

    if len(observed) < 2:
        raise HTTPException(400, "Need at least 2 observed values.")

    if n_missing == 0:
        return {
            "comparisons": [{
                "method": "none_needed",
                "mean": float(np.mean(observed)),
                "sd": float(np.std(observed, ddof=1)),
                "skewness": _safe_skewness(observed),
                "n_complete": len(observed),
            }],
            "recommendation": "No missing values — no imputation needed.",
        }

    # Reference stats from observed data
    obs_mean = float(np.mean(observed))
    obs_sd = float(np.std(observed, ddof=1))
    obs_skew = _safe_skewness(observed)

    comparisons = []

    for method in methods:
        m = method.lower().strip()
        imputed = raw.copy()

        if m == "listwise":
            completed = observed.copy()
        elif m == "mean":
            imputed[~observed_mask] = obs_mean
            completed = imputed
        elif m == "median":
            med = float(np.median(observed))
            imputed[~observed_mask] = med
            completed = imputed
        elif m == "regression":
            # Simple regression imputation: use index as predictor (time-based proxy)
            indices = np.arange(len(raw), dtype=float)
            obs_indices = indices[observed_mask]
            miss_indices = indices[~observed_mask]

            if len(obs_indices) >= 2:
                slope, intercept, _, _, _ = sp_stats.linregress(obs_indices, observed)
                predicted = intercept + slope * miss_indices
                imputed[~observed_mask] = predicted
                completed = imputed
            else:
                # Fall back to mean
                imputed[~observed_mask] = obs_mean
                completed = imputed
        elif m == "hot_deck":
            # Random hot deck: replace each missing with a random observed value
            rng = np.random.default_rng(42)
            replacements = rng.choice(observed, size=n_missing, replace=True)
            imputed[~observed_mask] = replacements
            completed = imputed
        else:
            continue  # skip unknown methods

        clean = completed[~np.isnan(completed)] if isinstance(completed, np.ndarray) else completed
        if len(clean) == 0:
            continue

        comp_mean = float(np.mean(clean))
        comp_sd = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
        comp_skew = _safe_skewness(clean)

        comparisons.append({
            "method": method,
            "mean": round(comp_mean, 4),
            "sd": round(comp_sd, 4),
            "skewness": round(comp_skew, 4),
            "n_complete": len(clean),
        })

    # Recommendation: find method that least distorts distribution
    if comparisons:
        best = None
        best_score = float("inf")
        for comp in comparisons:
            # Score = weighted distance from observed stats
            mean_diff = abs(comp["mean"] - obs_mean) / (obs_sd if obs_sd > 0 else 1)
            sd_diff = abs(comp["sd"] - obs_sd) / (obs_sd if obs_sd > 0 else 1)
            skew_diff = abs(comp["skewness"] - obs_skew)
            score = mean_diff + sd_diff + skew_diff
            if score < best_score:
                best_score = score
                best = comp["method"]
        rec = f"'{best}' least distorts the observed distribution (mean={obs_mean:.4f}, SD={obs_sd:.4f}, skewness={obs_skew:.4f})."
    else:
        rec = "No valid imputation methods were computed."

    return {
        "comparisons": comparisons,
        "recommendation": rec,
    }
