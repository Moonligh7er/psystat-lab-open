"""Smart test recommendation and pre-flight data validation."""

from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    data: dict[str, list]  # column_name -> list of values
    research_question: str  # group_comparison, relationship, prediction, proportion, repeated_measures, survival
    groups: Optional[int] = None
    paired: Optional[bool] = False


class ValidateRequest(BaseModel):
    data: list  # list of floats or list of lists
    test_type: str
    data2: Optional[list] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_QUESTIONS = {
    "group_comparison",
    "relationship",
    "prediction",
    "proportion",
    "repeated_measures",
    "survival",
}


def _detect_variable_type(values: list) -> str:
    """Classify a column as continuous, ordinal, or categorical."""
    clean = [v for v in values if v is not None]
    if len(clean) == 0:
        return "empty"

    # Check if numeric
    numeric_vals = []
    for v in clean:
        try:
            numeric_vals.append(float(v))
        except (ValueError, TypeError):
            pass

    ratio_numeric = len(numeric_vals) / len(clean) if clean else 0

    if ratio_numeric < 0.8:
        return "categorical"

    n_unique = len(set(numeric_vals))
    # If numeric but very few unique values relative to N, treat as ordinal
    if n_unique <= 7 and all(float(v) == int(float(v)) for v in numeric_vals):
        return "ordinal"

    return "continuous"


def _to_numeric(values: list) -> np.ndarray:
    """Convert list to numeric numpy array, dropping non-numeric."""
    nums = []
    for v in values:
        if v is None:
            continue
        try:
            nums.append(float(v))
        except (ValueError, TypeError):
            continue
    return np.array(nums, dtype=float)


def _normality_ok(arr: np.ndarray, alpha: float = 0.05) -> tuple[bool, float]:
    """Return (is_normal, p_value) via Shapiro-Wilk."""
    if len(arr) < 3:
        return True, 1.0
    if len(arr) > 5000:
        arr = np.random.default_rng(42).choice(arr, 5000, replace=False)
    stat, p = sp_stats.shapiro(arr)
    return bool(p >= alpha), float(p)


def _levene_ok(groups: list[np.ndarray], alpha: float = 0.05) -> tuple[bool, float]:
    """Return (equal_var, p_value) via Levene's test."""
    if len(groups) < 2:
        return True, 1.0
    stat, p = sp_stats.levene(*groups)
    return bool(p >= alpha), float(p)


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    pooled_sd = math.sqrt(((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float(abs(np.mean(g1) - np.mean(g2)) / pooled_sd)


def _required_n_ttest(d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Approximate per-group N for a two-sample t-test using the formula
    n = ((z_alpha + z_beta) / d)^2  per group."""
    if d <= 0:
        return 999
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    n = ((z_a + z_b) / d) ** 2
    return max(int(math.ceil(n)), 3)


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

@router.post("/recommend")
def recommend_test(req: RecommendRequest):
    """Recommend a statistical test based on data characteristics and research question."""
    if req.research_question not in VALID_QUESTIONS:
        raise HTTPException(400, f"research_question must be one of {sorted(VALID_QUESTIONS)}")

    if not req.data:
        raise HTTPException(400, "data must contain at least one column.")

    # --- Detect variable types ---
    var_types: dict[str, str] = {}
    for col, vals in req.data.items():
        var_types[col] = _detect_variable_type(vals)

    continuous_cols = {c: _to_numeric(v) for c, v in req.data.items() if var_types[c] == "continuous"}
    categorical_cols = {c: v for c, v in req.data.items() if var_types[c] == "categorical"}

    # Determine groups from data
    n_groups = req.groups
    col_names = list(req.data.keys())
    all_numeric = [_to_numeric(v) for v in req.data.values() if _detect_variable_type(v) == "continuous"]

    # --- Assumption checks ---
    assumptions_met: dict[str, bool] = {}
    assumption_details: dict[str, str] = {}
    warnings: list[str] = []

    # Normality
    for col, arr in continuous_cols.items():
        if len(arr) >= 3:
            ok, p = _normality_ok(arr)
            assumptions_met[f"normality_{col}"] = ok
            assumption_details[f"normality_{col}"] = (
                f"Shapiro-Wilk p = {p:.4f}. {'Normal' if ok else 'Non-normal'} distribution."
            )
            if not ok:
                warnings.append(f"Column '{col}' is not normally distributed (p = {p:.4f}).")

    # Homogeneity of variance
    if len(all_numeric) >= 2 and req.research_question in ("group_comparison", "repeated_measures"):
        valid_groups = [g for g in all_numeric if len(g) >= 2]
        if len(valid_groups) >= 2:
            eq_var, lev_p = _levene_ok(valid_groups)
            assumptions_met["homogeneity_of_variance"] = eq_var
            assumption_details["homogeneity_of_variance"] = (
                f"Levene's test p = {lev_p:.4f}. {'Equal' if eq_var else 'Unequal'} variances."
            )
            if not eq_var:
                warnings.append(f"Unequal variances detected (Levene's p = {lev_p:.4f}).")

    # Sample sizes
    ns = [len(arr) for arr in all_numeric]
    min_n = min(ns) if ns else 0
    assumptions_met["adequate_sample_size"] = min_n >= 20
    assumption_details["adequate_sample_size"] = (
        f"Minimum group N = {min_n}. {'Adequate.' if min_n >= 20 else 'Small sample — assumptions matter more.'}"
    )
    if min_n < 20:
        warnings.append(f"Small sample size (min N = {min_n}). Assumptions are more critical.")

    # --- Determine number of groups ---
    if n_groups is None:
        n_groups = len(all_numeric) if req.research_question == "group_comparison" else len(col_names)

    paired = req.paired or False
    normality_all = all(v for k, v in assumptions_met.items() if k.startswith("normality_"))
    equal_var = assumptions_met.get("homogeneity_of_variance", True)

    # --- Recommendation logic ---
    recommended_test = ""
    alternative_test = ""
    reason = ""
    effect_size_to_report = ""
    power_note = ""

    rq = req.research_question

    if rq == "group_comparison":
        if n_groups == 2:
            if paired:
                if normality_all:
                    recommended_test = "Paired t-test"
                    alternative_test = "Wilcoxon signed-rank test"
                    reason = "Two related groups with normally distributed differences."
                    effect_size_to_report = "Cohen's d (paired)"
                else:
                    recommended_test = "Wilcoxon signed-rank test"
                    alternative_test = "Paired t-test (if N > 30, CLT may apply)"
                    reason = "Two related groups but differences are not normally distributed."
                    effect_size_to_report = "Rank-biserial correlation"
            else:
                if normality_all and equal_var:
                    recommended_test = "Independent t-test"
                    alternative_test = "Mann-Whitney U test"
                    reason = "Two independent groups, normal distributions, equal variances."
                    effect_size_to_report = "Cohen's d"
                elif normality_all and not equal_var:
                    recommended_test = "Welch's t-test"
                    alternative_test = "Mann-Whitney U test"
                    reason = "Two independent groups, normal distributions, but unequal variances."
                    effect_size_to_report = "Cohen's d (with Welch correction)"
                else:
                    recommended_test = "Mann-Whitney U test"
                    alternative_test = "Welch's t-test (robust to mild non-normality with N > 30)"
                    reason = "Two independent groups with non-normal data."
                    effect_size_to_report = "Rank-biserial correlation"
        elif n_groups > 2:
            if normality_all and equal_var:
                recommended_test = "One-way ANOVA"
                alternative_test = "Kruskal-Wallis test"
                reason = "Multiple independent groups, normal distributions, equal variances."
                effect_size_to_report = "Eta-squared (eta2)"
            elif normality_all and not equal_var:
                recommended_test = "Welch's ANOVA"
                alternative_test = "Kruskal-Wallis test"
                reason = "Multiple independent groups, normal but unequal variances."
                effect_size_to_report = "Omega-squared"
            else:
                recommended_test = "Kruskal-Wallis test"
                alternative_test = "Welch's ANOVA (if mild non-normality and N > 30 per group)"
                reason = "Multiple independent groups with non-normal data."
                effect_size_to_report = "Epsilon-squared"
        else:
            recommended_test = "One-sample t-test"
            alternative_test = "Wilcoxon signed-rank test (one-sample)"
            reason = "Single group comparison to a hypothesized value."
            effect_size_to_report = "Cohen's d (one-sample)"

        # Power note
        if len(all_numeric) >= 2:
            d = _cohens_d(all_numeric[0], all_numeric[1]) if len(all_numeric) >= 2 else 0.5
            req_n = _required_n_ttest(d if d > 0 else 0.5)
            power_note = (
                f"Detected effect size d = {d:.2f}. "
                f"Required N per group for 80% power: {req_n}. "
                f"{'Current N is sufficient.' if min_n >= req_n else f'Current N ({min_n}) may be underpowered.'}"
            )
        else:
            power_note = "Cannot compute power — need at least two groups."

    elif rq == "relationship":
        if normality_all:
            recommended_test = "Pearson correlation"
            alternative_test = "Spearman rank correlation"
            reason = "Testing bivariate relationship between two continuous, normally distributed variables."
            effect_size_to_report = "Pearson r"
        else:
            recommended_test = "Spearman rank correlation"
            alternative_test = "Kendall's tau (for small N or many ties)"
            reason = "Testing bivariate relationship with non-normal data."
            effect_size_to_report = "Spearman rho"
        power_note = f"Minimum N for detecting medium correlation (r=.3) at 80% power: ~85."

    elif rq == "prediction":
        n_predictors = len(col_names) - 1  # last assumed outcome
        if n_predictors < 1:
            raise HTTPException(400, "Prediction requires at least one predictor and one outcome column.")
        recommended_test = "Multiple linear regression" if n_predictors > 1 else "Simple linear regression"
        alternative_test = "Robust regression (if heteroscedasticity or outliers)"
        reason = f"{n_predictors} predictor(s) predicting a continuous outcome."
        effect_size_to_report = "R-squared / Adjusted R-squared"
        min_needed = max(20, 10 * n_predictors + 10)
        total_n = min_n
        power_note = (
            f"Rule of thumb: need ~{min_needed} observations for {n_predictors} predictor(s). "
            f"{'Current N is sufficient.' if total_n >= min_needed else f'Current N ({total_n}) may be insufficient.'}"
        )

    elif rq == "proportion":
        recommended_test = "Chi-square test of independence"
        alternative_test = "Fisher's exact test (if any expected cell < 5)"
        reason = "Testing association between categorical variables."
        effect_size_to_report = "Cramer's V"
        power_note = "For medium effect (w=.3), need ~88 total observations for 80% power."

    elif rq == "repeated_measures":
        if n_groups == 2:
            if normality_all:
                recommended_test = "Paired t-test"
                alternative_test = "Wilcoxon signed-rank test"
                reason = "Two repeated measurements, normally distributed differences."
                effect_size_to_report = "Cohen's d (paired)"
            else:
                recommended_test = "Wilcoxon signed-rank test"
                alternative_test = "Paired t-test (if N > 30)"
                reason = "Two repeated measurements with non-normal differences."
                effect_size_to_report = "Rank-biserial correlation"
        else:
            if normality_all:
                recommended_test = "Repeated measures ANOVA"
                alternative_test = "Friedman test"
                reason = "Multiple repeated measurements with normally distributed data."
                effect_size_to_report = "Partial eta-squared"
            else:
                recommended_test = "Friedman test"
                alternative_test = "Repeated measures ANOVA (with N > 30 per condition)"
                reason = "Multiple repeated measurements with non-normal data."
                effect_size_to_report = "Kendall's W"
        power_note = "Repeated measures designs typically require fewer participants due to within-subject comparisons."

    elif rq == "survival":
        recommended_test = "Kaplan-Meier with Log-rank test"
        alternative_test = "Cox proportional hazards regression (for multiple predictors)"
        reason = "Time-to-event data with censoring."
        effect_size_to_report = "Hazard ratio"
        power_note = "Power depends on number of events, not total N. Aim for >= 10 events per predictor."

    return {
        "recommended_test": recommended_test,
        "alternative_test": alternative_test,
        "reason": reason,
        "assumptions_met": assumptions_met,
        "assumption_details": assumption_details,
        "effect_size_to_report": effect_size_to_report,
        "power_note": power_note,
        "warnings": warnings,
        "variable_types": var_types,
    }


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------

@router.post("/validate")
def validate_data(req: ValidateRequest):
    """Pre-flight data validation before running a statistical test."""
    tt = req.test_type.lower().replace("-", "").replace("_", "").replace(" ", "")
    warnings_list: list[dict] = []

    # Normalize data to list of arrays
    raw = req.data
    if not raw:
        raise HTTPException(400, "data must not be empty.")

    # Determine if data is flat list or list of lists
    is_nested = isinstance(raw[0], (list, tuple)) if raw else False
    if is_nested:
        arrays = [np.array([x for x in sub if x is not None], dtype=float) for sub in raw]
        flat = np.concatenate(arrays) if arrays else np.array([])
    else:
        clean = [x for x in raw if x is not None]
        try:
            flat = np.array(clean, dtype=float)
        except (ValueError, TypeError):
            raise HTTPException(400, "data values must be numeric.")
        arrays = [flat]

    data2_arr = None
    if req.data2 is not None:
        clean2 = [x for x in req.data2 if x is not None]
        try:
            data2_arr = np.array(clean2, dtype=float)
        except (ValueError, TypeError):
            raise HTTPException(400, "data2 values must be numeric.")

    n_total = len(flat)
    n_raw = len(raw) if not is_nested else sum(len(sub) for sub in raw)
    n_missing = n_raw - n_total
    missing_pct = (n_missing / n_raw * 100) if n_raw > 0 else 0.0

    valid = True

    # --- Missing values ---
    if missing_pct > 0:
        level = "error" if missing_pct > 50 else ("warning" if missing_pct > 10 else "info")
        warnings_list.append({
            "level": level,
            "message": f"{n_missing} missing value(s) ({missing_pct:.1f}%). {'Too many missing values.' if level == 'error' else ''}"
        })
        if level == "error":
            valid = False

    # --- Infinities ---
    n_inf = int(np.sum(~np.isfinite(flat))) if len(flat) > 0 else 0
    if n_inf > 0:
        warnings_list.append({"level": "error", "message": f"{n_inf} infinite value(s) detected. Remove or replace them."})
        valid = False
        flat = flat[np.isfinite(flat)]

    # --- Minimum N ---
    min_n_map = {
        "regression": 20, "linearregression": 20, "multipleregression": 20,
        "chisquare": 5, "chi2": 5,
    }
    min_required = min_n_map.get(tt, 3)
    if n_total < min_required:
        warnings_list.append({
            "level": "error",
            "message": f"Insufficient sample size: n = {n_total}, minimum required for {req.test_type} is {min_required}."
        })
        valid = False

    # --- Constant columns (zero variance) ---
    for i, arr in enumerate(arrays):
        if len(arr) > 0 and np.var(arr) == 0:
            warnings_list.append({
                "level": "error",
                "message": f"{'Data' if len(arrays) == 1 else f'Group {i+1}'} has zero variance (all values identical)."
            })
            valid = False

    # --- Paired tests: equal length ---
    paired_tests = {"pairedttest", "paired", "wilcoxon", "wilcoxonsignedrank"}
    if tt in paired_tests:
        if data2_arr is not None and len(flat) != len(data2_arr):
            warnings_list.append({
                "level": "error",
                "message": f"Paired test requires equal-length arrays: data has {len(flat)}, data2 has {len(data2_arr)}."
            })
            valid = False
        if is_nested and len(arrays) >= 2 and len(arrays[0]) != len(arrays[1]):
            warnings_list.append({
                "level": "error",
                "message": f"Paired test requires equal-length groups: group 1 has {len(arrays[0])}, group 2 has {len(arrays[1])}."
            })
            valid = False

    # --- Chi-square: expected cell counts >= 5 ---
    if tt in ("chisquare", "chi2"):
        total = n_total
        if is_nested and len(arrays) >= 2:
            # Rough check: minimum cell count
            min_cell = min(len(a) for a in arrays)
            if min_cell < 5:
                warnings_list.append({
                    "level": "warning",
                    "message": f"Some expected cell counts may be < 5 (min observed = {min_cell}). Consider Fisher's exact test."
                })

    # --- Regression: more obs than predictors ---
    if tt in ("regression", "linearregression", "multipleregression"):
        if is_nested:
            n_predictors = len(arrays) - 1
            n_obs = len(arrays[0]) if arrays else 0
            if n_obs <= n_predictors:
                warnings_list.append({
                    "level": "error",
                    "message": f"Regression requires more observations ({n_obs}) than predictors ({n_predictors})."
                })
                valid = False

    # --- Outliers ---
    outlier_count = 0
    if len(flat) >= 3:
        z = np.abs(sp_stats.zscore(flat))
        outlier_count = int(np.sum(z > 3))
        if outlier_count > 0:
            warnings_list.append({
                "level": "warning",
                "message": f"{outlier_count} potential outlier(s) detected (|z| > 3). Investigate before analysis."
            })

    # --- Data summary ---
    data_summary = {
        "n": n_total,
        "missing_count": n_missing,
        "missing_pct": round(missing_pct, 2),
        "outlier_count": outlier_count,
    }
    if len(flat) > 0:
        data_summary.update({
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "sd": float(np.std(flat, ddof=1)) if len(flat) > 1 else 0.0,
        })
    else:
        data_summary.update({"min": None, "max": None, "mean": None, "sd": None})

    return {
        "valid": valid,
        "warnings": warnings_list,
        "data_summary": data_summary,
    }
