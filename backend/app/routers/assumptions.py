"""Assumption checking — automatic pre-test diagnostics for statistical tests."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


class AssumptionCheckRequest(BaseModel):
    test_type: str  # ttest, anova, regression, paired, correlation
    groups: list[list[float]] | None = None  # for t-test, ANOVA
    x: list[list[float]] | None = None  # for regression predictors
    y: list[float] | None = None  # for regression outcome
    data: list[float] | None = None  # single sample
    alpha: float = 0.05


def _normality_check(data: np.ndarray, label: str, alpha: float) -> dict:
    """Run normality tests on a single array."""
    result = {"group": label, "n": len(data)}

    if len(data) < 3:
        result["verdict"] = "insufficient_data"
        result["message"] = f"{label}: Too few observations (n={len(data)}) for normality testing."
        return result

    # Shapiro-Wilk (best for n < 5000)
    if len(data) <= 5000:
        sw_stat, sw_p = sp_stats.shapiro(data)
        result["shapiro_wilk"] = {"statistic": float(sw_stat), "p_value": float(sw_p)}

    # D'Agostino-Pearson (needs n >= 8)
    if len(data) >= 8:
        da_stat, da_p = sp_stats.normaltest(data)
        result["dagostino"] = {"statistic": float(da_stat), "p_value": float(da_p)}

    # Skewness and kurtosis
    skew = float(sp_stats.skew(data))
    kurt = float(sp_stats.kurtosis(data))
    result["skewness"] = skew
    result["kurtosis"] = kurt

    # Verdict
    primary_p = result.get("shapiro_wilk", result.get("dagostino", {})).get("p_value", 1)
    if primary_p < alpha:
        result["passed"] = False
        result["message"] = f"{label}: Distribution departs significantly from normality ({_p(primary_p)}). Skewness={skew:.2f}, Kurtosis={kurt:.2f}."
        result["recommendation"] = "Consider non-parametric alternatives or data transformation."
    else:
        result["passed"] = True
        result["message"] = f"{label}: No significant departure from normality ({_p(primary_p)}). Skewness={skew:.2f}, Kurtosis={kurt:.2f}."

    return result


def _p(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"


@router.post("/check")
def check_assumptions(req: AssumptionCheckRequest):
    """Run all relevant assumption checks for the specified test type."""
    tt = req.test_type.lower().replace("-", "").replace("_", "").replace(" ", "")
    checks = []
    warnings = []
    passed_all = True

    if tt in ("ttest", "independentttest"):
        if not req.groups or len(req.groups) < 2:
            raise HTTPException(400, "t-test requires 'groups' with at least 2 arrays.")

        g1, g2 = np.array(req.groups[0]), np.array(req.groups[1])

        # 1. Normality
        n1 = _normality_check(g1, "Group 1", req.alpha)
        n2 = _normality_check(g2, "Group 2", req.alpha)
        checks.append({"test": "Normality (Group 1)", **n1})
        checks.append({"test": "Normality (Group 2)", **n2})
        if not n1.get("passed", True) or not n2.get("passed", True):
            warnings.append("Normality assumption violated. Consider Mann-Whitney U test.")
            passed_all = False

        # 2. Homogeneity of variance (Levene's)
        lev_stat, lev_p = sp_stats.levene(g1, g2)
        hov_passed = bool(lev_p >= req.alpha)
        checks.append({
            "test": "Homogeneity of Variance (Levene's)",
            "statistic": float(lev_stat),
            "p_value": float(lev_p),
            "passed": hov_passed,
            "message": f"Levene's F = {lev_stat:.2f}, {_p(lev_p)}. {'Variances are equal.' if hov_passed else 'Unequal variances detected.'}",
            "recommendation": None if hov_passed else "Use Welch's t-test (does not assume equal variances)."
        })
        if not hov_passed:
            warnings.append("Unequal variances. Use Welch's t-test.")
            passed_all = False

        # 3. Sample size
        min_n = min(len(g1), len(g2))
        size_ok = min_n >= 20
        checks.append({
            "test": "Sample Size",
            "n1": len(g1), "n2": len(g2),
            "passed": size_ok,
            "message": f"n1={len(g1)}, n2={len(g2)}. {'Adequate for CLT.' if size_ok else 'Small sample — normality assumption is more critical.'}",
            "recommendation": None if size_ok else "With small samples, verify normality carefully or use non-parametric test."
        })
        if not size_ok:
            passed_all = False

    elif tt in ("pairedttest", "paired"):
        if not req.groups or len(req.groups) < 2:
            raise HTTPException(400, "Paired t-test requires 'groups' with 2 arrays.")

        diff = np.array(req.groups[0]) - np.array(req.groups[1])

        # 1. Normality of differences
        nd = _normality_check(diff, "Differences", req.alpha)
        checks.append({"test": "Normality of Differences", **nd})
        if not nd.get("passed", True):
            warnings.append("Difference scores are not normally distributed. Consider Wilcoxon signed-rank test.")
            passed_all = False

        # 2. Outliers in differences
        z = np.abs(sp_stats.zscore(diff))
        n_outliers = int(np.sum(z > 3))
        checks.append({
            "test": "Outlier Detection (differences)",
            "n_outliers": n_outliers,
            "passed": n_outliers == 0,
            "message": f"{'No extreme outliers' if n_outliers == 0 else f'{n_outliers} extreme outlier(s)'} detected in difference scores (|z| > 3).",
            "recommendation": None if n_outliers == 0 else "Investigate outliers — they can distort paired t-test results."
        })
        if n_outliers > 0:
            passed_all = False

    elif tt in ("anova", "onewayanova"):
        if not req.groups or len(req.groups) < 2:
            raise HTTPException(400, "ANOVA requires 'groups' with at least 2 arrays.")

        groups = [np.array(g) for g in req.groups]

        # 1. Normality per group
        for i, g in enumerate(groups):
            nc = _normality_check(g, f"Group {i+1}", req.alpha)
            checks.append({"test": f"Normality (Group {i+1})", **nc})
            if not nc.get("passed", True):
                passed_all = False

        non_normal = sum(1 for c in checks if c["test"].startswith("Normality") and not c.get("passed", True))
        if non_normal > 0:
            warnings.append(f"{non_normal} group(s) violated normality. Consider Kruskal-Wallis test.")

        # 2. Homogeneity of variance (Levene's)
        lev_stat, lev_p = sp_stats.levene(*groups)
        hov_passed = bool(lev_p >= req.alpha)
        checks.append({
            "test": "Homogeneity of Variance (Levene's)",
            "statistic": float(lev_stat),
            "p_value": float(lev_p),
            "passed": hov_passed,
            "message": f"Levene's F = {lev_stat:.2f}, {_p(lev_p)}.",
            "recommendation": None if hov_passed else "Use Welch's ANOVA or Games-Howell post-hoc."
        })
        if not hov_passed:
            warnings.append("Unequal variances. Use Welch's ANOVA.")
            passed_all = False

        # 3. Balanced design check
        sizes = [len(g) for g in groups]
        balanced = max(sizes) / min(sizes) <= 1.5 if min(sizes) > 0 else False
        checks.append({
            "test": "Balance",
            "group_sizes": sizes,
            "passed": balanced,
            "message": f"Group sizes: {sizes}. {'Reasonably balanced.' if balanced else 'Unbalanced design — ANOVA is less robust.'}",
        })
        if not balanced:
            passed_all = False

    elif tt in ("regression", "linearregression", "multipleregression"):
        if not req.x or not req.y:
            raise HTTPException(400, "Regression requires 'x' and 'y'.")

        X = np.array(req.x)
        y = np.array(req.y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Fit simple OLS for residuals
        import statsmodels.api as sm
        X_c = sm.add_constant(X)
        model = sm.OLS(y, X_c).fit()
        residuals = model.resid

        # 1. Normality of residuals
        nr = _normality_check(residuals, "Residuals", req.alpha)
        checks.append({"test": "Normality of Residuals", **nr})
        if not nr.get("passed", True):
            warnings.append("Residuals are not normally distributed.")
            passed_all = False

        # 2. Homoscedasticity (Breusch-Pagan)
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_c)
        homo_passed = bool(bp_p >= req.alpha)
        checks.append({
            "test": "Homoscedasticity (Breusch-Pagan)",
            "statistic": float(bp_stat),
            "p_value": float(bp_p),
            "passed": homo_passed,
            "message": f"Breusch-Pagan LM = {bp_stat:.2f}, {_p(bp_p)}. {'Constant variance.' if homo_passed else 'Heteroscedasticity detected.'}",
            "recommendation": None if homo_passed else "Use robust standard errors (HC3) or transform the outcome."
        })
        if not homo_passed:
            warnings.append("Heteroscedasticity detected. Use robust standard errors.")
            passed_all = False

        # 3. Independence (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw = float(durbin_watson(residuals))
        indep_passed = 1.5 <= dw <= 2.5
        checks.append({
            "test": "Independence (Durbin-Watson)",
            "statistic": dw,
            "passed": indep_passed,
            "message": f"Durbin-Watson = {dw:.2f}. {'No autocorrelation.' if indep_passed else 'Possible autocorrelation.'}",
            "recommendation": None if indep_passed else "Check for temporal/spatial autocorrelation in data."
        })
        if not indep_passed:
            warnings.append("Possible autocorrelation in residuals.")
            passed_all = False

        # 4. Multicollinearity (VIF) — only for multiple regression
        if X.shape[1] > 1:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vifs = []
            for i in range(X_c.shape[1]):
                try:
                    v = variance_inflation_factor(X_c, i)
                    if i > 0:  # skip constant
                        vifs.append({"predictor": i, "vif": float(v)})
                except Exception:
                    pass
            max_vif = max((v["vif"] for v in vifs), default=0)
            mc_passed = max_vif < 10
            checks.append({
                "test": "Multicollinearity (VIF)",
                "vifs": vifs,
                "max_vif": max_vif,
                "passed": mc_passed,
                "message": f"Max VIF = {max_vif:.1f}. {'No concerning multicollinearity.' if mc_passed else 'High multicollinearity detected.'}",
                "recommendation": None if mc_passed else "Remove or combine correlated predictors (VIF > 10)."
            })
            if not mc_passed:
                warnings.append("High multicollinearity — consider removing correlated predictors.")
                passed_all = False

    elif tt in ("correlation",):
        if not req.groups or len(req.groups) < 2:
            raise HTTPException(400, "Correlation requires 'groups' with 2 arrays (x, y).")
        x, y_data = np.array(req.groups[0]), np.array(req.groups[1])

        # Normality (for Pearson)
        nx = _normality_check(x, "Variable X", req.alpha)
        ny = _normality_check(y_data, "Variable Y", req.alpha)
        checks.append({"test": "Normality (X)", **nx})
        checks.append({"test": "Normality (Y)", **ny})
        if not nx.get("passed", True) or not ny.get("passed", True):
            warnings.append("Non-normal data. Consider Spearman rank correlation.")
            passed_all = False

        # Linearity check (simple: compare Pearson vs Spearman)
        r_pearson, _ = sp_stats.pearsonr(x, y_data)
        r_spearman, _ = sp_stats.spearmanr(x, y_data)
        diff = abs(r_pearson - r_spearman)
        linear_ok = diff < 0.1
        checks.append({
            "test": "Linearity (Pearson vs Spearman)",
            "pearson_r": float(r_pearson),
            "spearman_r": float(r_spearman),
            "difference": float(diff),
            "passed": linear_ok,
            "message": f"Pearson r = {r_pearson:.3f}, Spearman ρ = {r_spearman:.3f}. {'Relationship appears linear.' if linear_ok else 'Possible non-linear relationship.'}",
            "recommendation": None if linear_ok else "Inspect scatterplot. Consider Spearman or polynomial regression."
        })
        if not linear_ok:
            warnings.append("Possible non-linearity. Use Spearman correlation.")
            passed_all = False

    else:
        return {
            "test_type": req.test_type,
            "checks": [],
            "warnings": [f"Assumption checking for '{req.test_type}' is not yet implemented."],
            "passed_all": True,
            "recommendation": "Proceed with caution.",
        }

    # Overall recommendation
    if passed_all:
        recommendation = f"All assumptions met. Safe to proceed with {req.test_type}."
    else:
        recommendation = f"Some assumptions violated ({len(warnings)} warning(s)). Review warnings and consider alternatives."

    return {
        "test_type": req.test_type,
        "checks": checks,
        "warnings": warnings,
        "passed_all": passed_all,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c.get("passed", True)),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Auto-check-and-route
# ---------------------------------------------------------------------------

class AutoCheckRequest(BaseModel):
    test_type: str  # e.g. "ttest", "paired_ttest", "anova", "correlation", "regression"
    data1: list[float]
    data2: Optional[list[float]] = None
    groups: Optional[list[list[float]]] = None


# Maps original test -> (alternative, reason)
_ALTERNATIVES = {
    "ttest": {
        "normality": ("Mann-Whitney U test", "Data are not normally distributed."),
        "homogeneity": ("Welch's t-test", "Variances are unequal across groups."),
        "both": ("Mann-Whitney U test", "Normality and equal-variance assumptions are both violated."),
    },
    "paired_ttest": {
        "normality": ("Wilcoxon signed-rank test", "Difference scores are not normally distributed."),
    },
    "anova": {
        "normality": ("Kruskal-Wallis test", "One or more groups are not normally distributed."),
        "homogeneity": ("Welch's ANOVA", "Variances are unequal across groups."),
        "both": ("Kruskal-Wallis test", "Normality and equal-variance assumptions are both violated."),
    },
    "correlation": {
        "normality": ("Spearman rank correlation", "One or both variables are not normally distributed."),
        "linearity": ("Spearman rank correlation", "Relationship appears non-linear."),
    },
    "regression": {
        "normality": ("Robust regression / bootstrap CIs", "Residuals are not normally distributed."),
        "homoscedasticity": ("Regression with HC3 robust SEs", "Heteroscedasticity detected in residuals."),
        "multicollinearity": ("Ridge / LASSO regression", "High multicollinearity among predictors."),
    },
}


@router.post("/auto-check-and-route")
def auto_check_and_route(req: AutoCheckRequest):
    """Check assumptions for a test and recommend an alternative if violated."""
    tt = req.test_type.lower().replace("-", "_").replace(" ", "_")
    assumptions: dict[str, dict] = {}
    violations: list[str] = []
    alpha = 0.05

    arr1 = np.array(req.data1, dtype=float)
    arr2 = np.array(req.data2, dtype=float) if req.data2 else None
    group_arrays = [np.array(g, dtype=float) for g in req.groups] if req.groups else None

    # ---- t-test (independent) ----
    if tt in ("ttest", "independent_ttest", "t_test"):
        if arr2 is None and group_arrays is None:
            raise HTTPException(400, "Independent t-test requires data2 or groups.")
        g1 = arr1
        g2 = arr2 if arr2 is not None else (group_arrays[1] if group_arrays and len(group_arrays) > 1 else None)
        if g2 is None:
            raise HTTPException(400, "Independent t-test requires two groups.")

        # Normality
        for label, arr in [("group_1", g1), ("group_2", g2)]:
            if len(arr) >= 3:
                stat, p = sp_stats.shapiro(arr)
                passed = bool(p >= alpha)
                assumptions[f"normality_{label}"] = {
                    "met": passed, "statistic": float(stat), "p_value": float(p),
                    "interpretation": f"{'Normal' if passed else 'Non-normal'} distribution (Shapiro-Wilk p={p:.4f}).",
                }
                if not passed:
                    violations.append("normality")

        # Homogeneity
        if len(g1) >= 2 and len(g2) >= 2:
            lev_stat, lev_p = sp_stats.levene(g1, g2)
            hov = bool(lev_p >= alpha)
            assumptions["homogeneity_of_variance"] = {
                "met": hov, "statistic": float(lev_stat), "p_value": float(lev_p),
                "interpretation": f"{'Equal' if hov else 'Unequal'} variances (Levene's p={lev_p:.4f}).",
            }
            if not hov:
                violations.append("homogeneity")

        alt_key = "ttest"

    # ---- Paired t-test ----
    elif tt in ("paired_ttest", "paired_t_test", "paired"):
        if arr2 is None:
            raise HTTPException(400, "Paired t-test requires data2.")
        if len(arr1) != len(arr2):
            raise HTTPException(400, "Paired t-test requires equal-length arrays.")
        diff = arr1 - arr2

        if len(diff) >= 3:
            stat, p = sp_stats.shapiro(diff)
            passed = bool(p >= alpha)
            assumptions["normality_differences"] = {
                "met": passed, "statistic": float(stat), "p_value": float(p),
                "interpretation": f"Difference scores are {'normally' if passed else 'not normally'} distributed (p={p:.4f}).",
            }
            if not passed:
                violations.append("normality")

        # Outliers in differences
        if len(diff) >= 3:
            z = np.abs(sp_stats.zscore(diff))
            n_out = int(np.sum(z > 3))
            assumptions["outliers_differences"] = {
                "met": n_out == 0, "statistic": n_out, "p_value": None,
                "interpretation": f"{'No extreme' if n_out == 0 else f'{n_out} extreme'} outlier(s) in differences.",
            }

        alt_key = "paired_ttest"

    # ---- ANOVA ----
    elif tt in ("anova", "one_way_anova", "oneway_anova"):
        grps = group_arrays if group_arrays else ([arr1, arr2] if arr2 is not None else None)
        if not grps or len(grps) < 2:
            raise HTTPException(400, "ANOVA requires groups (at least 2).")

        for i, g in enumerate(grps):
            if len(g) >= 3:
                stat, p = sp_stats.shapiro(g)
                passed = bool(p >= alpha)
                assumptions[f"normality_group_{i+1}"] = {
                    "met": passed, "statistic": float(stat), "p_value": float(p),
                    "interpretation": f"Group {i+1}: {'Normal' if passed else 'Non-normal'} (p={p:.4f}).",
                }
                if not passed:
                    violations.append("normality")

        # Levene
        lev_stat, lev_p = sp_stats.levene(*grps)
        hov = bool(lev_p >= alpha)
        assumptions["homogeneity_of_variance"] = {
            "met": hov, "statistic": float(lev_stat), "p_value": float(lev_p),
            "interpretation": f"{'Equal' if hov else 'Unequal'} variances (Levene's p={lev_p:.4f}).",
        }
        if not hov:
            violations.append("homogeneity")

        alt_key = "anova"

    # ---- Correlation ----
    elif tt in ("correlation", "pearson"):
        if arr2 is None:
            raise HTTPException(400, "Correlation requires data2.")

        for label, arr in [("x", arr1), ("y", arr2)]:
            if len(arr) >= 3:
                stat, p = sp_stats.shapiro(arr)
                passed = bool(p >= alpha)
                assumptions[f"normality_{label}"] = {
                    "met": passed, "statistic": float(stat), "p_value": float(p),
                    "interpretation": f"Variable {label}: {'Normal' if passed else 'Non-normal'} (p={p:.4f}).",
                }
                if not passed:
                    violations.append("normality")

        # Linearity
        if len(arr1) >= 3 and len(arr2) >= 3:
            r_p, _ = sp_stats.pearsonr(arr1, arr2)
            r_s, _ = sp_stats.spearmanr(arr1, arr2)
            diff = abs(r_p - r_s)
            linear = diff < 0.1
            assumptions["linearity"] = {
                "met": linear, "statistic": float(diff), "p_value": None,
                "interpretation": f"Pearson-Spearman difference = {diff:.3f}. {'Linear' if linear else 'Possibly non-linear'}.",
            }
            if not linear:
                violations.append("linearity")

        alt_key = "correlation"

    # ---- Regression ----
    elif tt in ("regression", "linear_regression", "multiple_regression"):
        if arr2 is None and group_arrays is None:
            raise HTTPException(400, "Regression requires data2 (outcome) or groups (predictors).")

        import statsmodels.api as sm

        y = arr2 if arr2 is not None else arr1
        if group_arrays:
            X = np.column_stack(group_arrays)
        else:
            X = arr1.reshape(-1, 1)

        X_c = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_c).fit()
        except Exception as e:
            raise HTTPException(400, f"Regression failed: {e}")
        resid = model.resid

        # Normality of residuals
        if len(resid) >= 3:
            stat, p = sp_stats.shapiro(resid[:5000])
            passed = bool(p >= alpha)
            assumptions["normality_residuals"] = {
                "met": passed, "statistic": float(stat), "p_value": float(p),
                "interpretation": f"Residuals {'are' if passed else 'are not'} normally distributed (p={p:.4f}).",
            }
            if not passed:
                violations.append("normality")

        # Homoscedasticity
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_c)
        homo = bool(bp_p >= alpha)
        assumptions["homoscedasticity"] = {
            "met": homo, "statistic": float(bp_stat), "p_value": float(bp_p),
            "interpretation": f"{'Constant' if homo else 'Non-constant'} variance (BP p={bp_p:.4f}).",
        }
        if not homo:
            violations.append("homoscedasticity")

        # Multicollinearity
        if X.shape[1] > 1:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            max_vif = 0
            for i in range(X_c.shape[1]):
                if i == 0:
                    continue
                try:
                    v = variance_inflation_factor(X_c, i)
                    max_vif = max(max_vif, v)
                except Exception:
                    pass
            mc = max_vif < 10
            assumptions["multicollinearity"] = {
                "met": mc, "statistic": float(max_vif), "p_value": None,
                "interpretation": f"Max VIF = {max_vif:.1f}. {'Acceptable' if mc else 'High multicollinearity'}.",
            }
            if not mc:
                violations.append("multicollinearity")

        alt_key = "regression"

    else:
        return {
            "assumptions": {},
            "all_met": True,
            "recommended_test": req.test_type,
            "alternative_reason": None,
            "severity": "proceed",
        }

    # --- Determine recommendation ---
    all_met = len(violations) == 0
    unique_violations = list(dict.fromkeys(violations))  # dedupe preserving order

    alt_map = _ALTERNATIVES.get(alt_key, {})
    if all_met:
        recommended_test = req.test_type
        alternative_reason = None
        severity = "proceed"
    else:
        # Pick the most relevant alternative
        if "both" in alt_map and "normality" in unique_violations and "homogeneity" in unique_violations:
            recommended_test, alternative_reason = alt_map["both"]
            severity = "use_alternative"
        else:
            # Use the first violated assumption that has an alternative
            recommended_test = req.test_type
            alternative_reason = None
            severity = "caution"
            for v in unique_violations:
                if v in alt_map:
                    recommended_test, alternative_reason = alt_map[v]
                    severity = "use_alternative"
                    break

        # If only minor violations (e.g., one group borderline), downgrade to caution
        borderline_count = sum(
            1 for a in assumptions.values()
            if not a["met"] and a.get("p_value") is not None and a["p_value"] >= 0.01
        )
        if borderline_count == len([v for v in assumptions.values() if not v["met"]]) and borderline_count > 0:
            severity = "caution"

    return {
        "assumptions": assumptions,
        "all_met": all_met,
        "recommended_test": recommended_test,
        "alternative_reason": alternative_reason,
        "severity": severity,
    }
