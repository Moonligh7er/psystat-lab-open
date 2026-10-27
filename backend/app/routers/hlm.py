"""Hierarchical Linear Modeling (Multilevel Modeling) — the workhorse of
nested-data analysis in psychology, education, and sociology."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from app.mixed import fit_mixedlm, fixed_effect_se
from scipy import stats as sp_stats

router = APIRouter()


# ── Request schemas ───────────────────────────────────────────────────

class TwoLevelRequest(BaseModel):
    dependent: list[float]
    level1_predictors: list[list[float]]
    groups: list[int | str]
    level2_predictors: list[list[float]] | None = None
    predictor_names: list[str] | None = None
    random_slopes: bool = False
    reml: bool = True


class ThreeLevelRequest(BaseModel):
    dependent: list[float]
    level1_predictors: list[list[float]]
    level2_groups: list[int | str]
    level3_groups: list[int | str]
    predictor_names: list[str] | None = None
    reml: bool = True


class GrowthCurveRequest(BaseModel):
    dependent: list[float]
    time: list[float]
    subjects: list[int | str]
    covariates: list[list[float]] | None = None
    covariate_names: list[str] | None = None
    quadratic: bool = False


class ICCRequest(BaseModel):
    values: list[float]
    groups: list[int | str]


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/two-level")
def two_level_hlm(req: TwoLevelRequest):
    """Two-level HLM: observations nested in groups."""
    n = len(req.dependent)
    n_pred = len(req.level1_predictors)

    # ── Validation ────────────────────────────────────────────────────
    if n < 4:
        raise HTTPException(400, "Need at least 4 observations.")
    if any(len(p) != n for p in req.level1_predictors):
        raise HTTPException(400, "Every level-1 predictor must have the same length as dependent.")
    if len(req.groups) != n:
        raise HTTPException(400, "groups must have the same length as dependent.")

    unique_groups = sorted(set(req.groups), key=str)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise HTTPException(400, "Need at least 2 groups for multilevel modeling.")

    # Build dataframe
    pred_names = req.predictor_names or [f"x{i+1}" for i in range(n_pred)]
    if len(pred_names) != n_pred:
        raise HTTPException(400, "predictor_names length must match number of level-1 predictors.")

    df = pd.DataFrame({"y": req.dependent, "group": req.groups})
    for i, name in enumerate(pred_names):
        df[name] = req.level1_predictors[i]

    # Add level-2 predictors (group-level)
    l2_names: list[str] = []
    if req.level2_predictors is not None:
        for j, col in enumerate(req.level2_predictors):
            if len(col) != n:
                raise HTTPException(
                    400,
                    "Each level-2 predictor must be broadcast to observation length "
                    "(same value repeated for every observation in a group).",
                )
            cname = f"w{j+1}"
            l2_names.append(cname)
            df[cname] = col

    # ── Null model (intercept-only) for ICC ───────────────────────────
    try:
        null_model = fit_mixedlm(smf.mixedlm("y ~ 1", df, groups=df["group"]), reml=req.reml)
    except Exception as exc:
        raise HTTPException(422, f"Null model failed to converge: {exc}")

    var_between = float(null_model.cov_re.iloc[0, 0])
    var_within = float(null_model.scale)
    icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0

    # ── Full model ────────────────────────────────────────────────────
    fixed_formula = "y ~ " + " + ".join(pred_names + l2_names)

    re_formula = None
    if req.random_slopes and n_pred > 0:
        re_formula = "~ " + " + ".join(pred_names)

    try:
        model = fit_mixedlm(smf.mixedlm(
            fixed_formula, df, groups=df["group"],
            re_formula=re_formula,
        ), reml=req.reml)
    except np.linalg.LinAlgError:
        raise HTTPException(
            422,
            "Model has a singular covariance matrix. Try centering predictors, "
            "removing redundant predictors, or setting random_slopes=false.",
        )
    except Exception as exc:
        raise HTTPException(422, f"Model failed to converge: {exc}")

    # ── Fixed effects ─────────────────────────────────────────────────
    # Wald z tests (statsmodels). Residual-df t tests would ignore the clustering; with few
    # groups even Wald tests are optimistic (Kenward-Roger / Satterthwaite would be preferable).
    fe_params = model.fe_params
    z95 = sp_stats.norm.ppf(0.975)
    fixed_effects = []
    for name in fe_params.index:
        est = float(fe_params[name])
        se = float(fixed_effect_se(model)[0][name])
        z_val = est / se if se > 0 else float("nan")
        p_val = float(2 * sp_stats.norm.sf(abs(z_val)))
        fixed_effects.append({
            "name": name,
            "estimate": est,
            "se": se,
            "z": z_val,
            "p": p_val,
            "ci_lower": est - z95 * se,
            "ci_upper": est + z95 * se,
            "significant": p_val < 0.05,
        })

    # ── Random effects & variance components ──────────────────────────
    cov_re = model.cov_re
    var_components = {}
    for idx in cov_re.index:
        for col in cov_re.columns:
            label = f"Var({idx})" if idx == col else f"Cov({idx},{col})"
            var_components[label] = float(cov_re.loc[idx, col])
    var_components["Residual"] = float(model.scale)

    # Group-level random effects (BLUPs)
    re_raw = model.random_effects
    group_summaries = []
    for grp in unique_groups:
        re_vals = re_raw.get(grp, re_raw.get(str(grp), {}))
        intercept = float(re_vals.iloc[0]) if hasattr(re_vals, "iloc") else float(list(re_vals.values())[0])
        group_summaries.append({
            "group": grp,
            "random_intercept": intercept,
            "conditional_intercept": float(fe_params["Intercept"]) + intercept,
        })

    # ── R-squared (Nakagawa & Schielzeth 2013; Johnson 2014 for random slopes) ──
    X_fixed = model.model.exog
    var_fixed = float(np.var(X_fixed @ fe_params.values, ddof=1))
    Z = model.model.exog_re  # random-effects design (intercept, plus slopes if any)
    var_re_total = float(np.mean(np.einsum("ij,jk,ik->i", Z, cov_re.values, Z)))
    var_resid = float(model.scale)
    total_var = var_fixed + var_re_total + var_resid
    r2_marginal = var_fixed / total_var if total_var > 0 else 0.0
    r2_conditional = (var_fixed + var_re_total) / total_var if total_var > 0 else 0.0

    # ── Model fit ─────────────────────────────────────────────────────
    llf = float(model.llf)
    q = cov_re.shape[0]
    n_params = len(fe_params) + q * (q + 1) // 2 + 1  # fixed effects, (co)variances, residual
    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + n_params * np.log(n)
    deviance = -2 * llf

    return {
        "fixed_effects": fixed_effects,
        "random_effects": {
            "variance_components": var_components,
            "group_summaries": group_summaries,
        },
        "icc": {
            "value": icc,
            "interpretation": _interpret_icc(icc),
        },
        "r_squared": {
            "marginal": r2_marginal,
            "conditional": r2_conditional,
            "interpretation": (
                f"Fixed effects explain {r2_marginal:.1%} of variance; "
                f"fixed + random effects explain {r2_conditional:.1%}."
            ),
        },
        "model_fit": {
            "log_likelihood": llf,
            "deviance": deviance,
            "aic": aic,
            "bic": bic,
        },
        "n_observations": n,
        "n_groups": n_groups,
        "estimation": "REML" if req.reml else "ML",
        "tests_note": "Fixed-effect tests are Wald z; with few groups they are optimistic. AIC/BIC from a REML fit compare only models with the same fixed effects.",
        "interpretation": _interpret_two_level(icc, fixed_effects, r2_marginal, r2_conditional),
    }


@router.post("/three-level")
def three_level_hlm(req: ThreeLevelRequest):
    """Three-level HLM: e.g., students in classrooms in schools."""
    n = len(req.dependent)
    n_pred = len(req.level1_predictors)

    if n < 6:
        raise HTTPException(400, "Need at least 6 observations for a three-level model.")
    if any(len(p) != n for p in req.level1_predictors):
        raise HTTPException(400, "Every level-1 predictor must have the same length as dependent.")
    if len(req.level2_groups) != n or len(req.level3_groups) != n:
        raise HTTPException(400, "level2_groups and level3_groups must match dependent length.")

    unique_l2 = set(req.level2_groups)
    unique_l3 = set(req.level3_groups)
    if len(unique_l3) < 2:
        raise HTTPException(400, "Need at least 2 level-3 groups.")
    if len(unique_l2) < len(unique_l3) + 1:
        raise HTTPException(400, "Need more level-2 groups than level-3 groups.")

    pred_names = req.predictor_names or [f"x{i+1}" for i in range(n_pred)]
    if len(pred_names) != n_pred:
        raise HTTPException(400, "predictor_names length must match number of predictors.")

    df = pd.DataFrame({
        "y": req.dependent,
        "l2": [str(g) for g in req.level2_groups],
        "l3": [str(g) for g in req.level3_groups],
    })
    for i, name in enumerate(pred_names):
        df[name] = req.level1_predictors[i]

    # Create a combined nesting variable for level-2 within level-3
    df["l2_nested"] = df["l3"] + ":" + df["l2"]

    # ── Null model for ICCs ───────────────────────────────────────────
    try:
        null_l3 = fit_mixedlm(smf.mixedlm("y ~ 1", df, groups=df["l3"]), reml=req.reml)
    except Exception as exc:
        raise HTTPException(422, f"Level-3 null model failed: {exc}")

    try:
        null_full = fit_mixedlm(smf.mixedlm(
            "y ~ 1", df, groups=df["l3"],
            re_formula="1", vc_formula={"l2_nested": "0 + C(l2_nested)"},
        ), reml=req.reml)
    except Exception:
        # Fallback: fit level-2 grouping separately
        try:
            null_l2 = fit_mixedlm(smf.mixedlm("y ~ 1", df, groups=df["l2_nested"]), reml=req.reml)
            var_l2 = float(null_l2.cov_re.iloc[0, 0])
        except Exception as exc:
            raise HTTPException(422, f"Three-level null model failed: {exc}")

        var_l3 = float(null_l3.cov_re.iloc[0, 0])
        var_resid = float(null_l3.scale)
        null_full = None
    else:
        var_l3 = float(null_full.cov_re.iloc[0, 0])
        vc = null_full.vcomp
        var_l2 = float(vc[0]) if len(vc) > 0 else 0.0
        var_resid = float(null_full.scale)

    total_var = var_l3 + var_l2 + var_resid
    icc_l3 = var_l3 / total_var if total_var > 0 else 0.0
    icc_l2 = (var_l3 + var_l2) / total_var if total_var > 0 else 0.0

    # ── Full model ────────────────────────────────────────────────────
    fixed_formula = "y ~ " + " + ".join(pred_names) if pred_names else "y ~ 1"
    try:
        model = fit_mixedlm(smf.mixedlm(
            fixed_formula, df, groups=df["l3"],
            re_formula="1",
            vc_formula={"l2_nested": "0 + C(l2_nested)"},
        ), reml=req.reml)
    except Exception:
        # Simpler fallback: treat l2_nested as the grouping variable
        try:
            model = fit_mixedlm(smf.mixedlm(
                fixed_formula, df, groups=df["l2_nested"],
            ), reml=req.reml)
        except Exception as exc:
            raise HTTPException(422, f"Three-level model failed to converge: {exc}")

    # ── Fixed effects ─────────────────────────────────────────────────
    fe = model.fe_params
    bse = fixed_effect_se(model)[0]
    fixed_effects = []
    for name in fe.index:
        est = float(fe[name])
        se = float(bse[name]) if name in bse.index else float("nan")
        t_val = est / se if se > 0 else float("nan")
        p_val = float(2 * sp_stats.t.sf(abs(t_val), df=n - len(fe)))
        fixed_effects.append({
            "name": name,
            "estimate": est,
            "se": se,
            "t": t_val,
            "p": p_val,
            "ci_lower": est - 1.96 * se,
            "ci_upper": est + 1.96 * se,
            "significant": p_val < 0.05,
        })

    # ── Model fit ─────────────────────────────────────────────────────
    llf = float(model.llf)
    n_params = len(fe) + 3  # fixed + 3 variance components
    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + n_params * np.log(n)

    return {
        "fixed_effects": fixed_effects,
        "variance_components": {
            "level3_intercept": var_l3,
            "level2_intercept": var_l2,
            "residual": var_resid,
        },
        "icc": {
            "level3": {
                "value": icc_l3,
                "interpretation": (
                    f"{icc_l3:.1%} of total variance is between level-3 units."
                ),
            },
            "level2_conditional": {
                "value": icc_l2,
                "interpretation": (
                    f"{icc_l2:.1%} of total variance is between level-2 units "
                    "(including level-3 contribution)."
                ),
            },
        },
        "model_fit": {
            "log_likelihood": llf,
            "deviance": -2 * llf,
            "aic": aic,
            "bic": bic,
        },
        "n_observations": n,
        "n_level2_groups": len(unique_l2),
        "n_level3_groups": len(unique_l3),
        "interpretation": (
            f"Three-level model with {len(unique_l2)} level-2 and "
            f"{len(unique_l3)} level-3 units. "
            f"Level-3 ICC = {icc_l3:.3f}, Level-2 conditional ICC = {icc_l2:.3f}. "
            + (
                "Substantial clustering exists at both levels."
                if icc_l3 > 0.05 and (icc_l2 - icc_l3) > 0.05
                else "Clustering is modest; a simpler model may suffice."
            )
        ),
    }


@router.post("/growth-curve")
def growth_curve(req: GrowthCurveRequest):
    """Growth curve modeling via HLM (repeated measures over time)."""
    n = len(req.dependent)

    if n < 4:
        raise HTTPException(400, "Need at least 4 observations.")
    if len(req.time) != n or len(req.subjects) != n:
        raise HTTPException(400, "dependent, time, and subjects must all have the same length.")

    unique_subjects = sorted(set(req.subjects), key=str)
    n_subjects = len(unique_subjects)
    if n_subjects < 2:
        raise HTTPException(400, "Need at least 2 subjects.")

    df = pd.DataFrame({
        "y": req.dependent,
        "time": req.time,
        "subject": req.subjects,
    })

    # Covariates (time-invariant)
    cov_names: list[str] = []
    if req.covariates is not None:
        cov_names = req.covariate_names or [f"cov{j+1}" for j in range(len(req.covariates))]
        if len(cov_names) != len(req.covariates):
            raise HTTPException(400, "covariate_names length must match number of covariates.")
        for j, col in enumerate(req.covariates):
            if len(col) != n:
                raise HTTPException(400, "Each covariate must have the same length as dependent.")
            df[cov_names[j]] = col

    # ── Build formula ─────────────────────────────────────────────────
    fixed_terms = ["time"]
    if req.quadratic:
        df["time_sq"] = df["time"] ** 2
        fixed_terms.append("time_sq")
    fixed_terms.extend(cov_names)
    fixed_formula = "y ~ " + " + ".join(fixed_terms)

    # Random slopes for time (and quadratic if requested)
    re_terms = ["time"]
    if req.quadratic:
        re_terms.append("time_sq")
    re_formula = "~ " + " + ".join(re_terms)

    # ── Fit model ─────────────────────────────────────────────────────
    try:
        model = fit_mixedlm(smf.mixedlm(
            fixed_formula, df, groups=df["subject"],
            re_formula=re_formula,
        ), reml=True)
    except (np.linalg.LinAlgError, Exception):
        # Fallback: random intercept + slope without correlation
        try:
            model = fit_mixedlm(smf.mixedlm(
                fixed_formula, df, groups=df["subject"],
                re_formula="~ time",
            ), reml=True)
        except Exception as exc:
            raise HTTPException(422, f"Growth curve model failed to converge: {exc}")

    # ── Fixed effects ─────────────────────────────────────────────────
    fe = model.fe_params
    bse = fixed_effect_se(model)[0]
    fixed_effects = []
    for name in fe.index:
        est = float(fe[name])
        se = float(bse[name]) if name in bse.index else float("nan")
        t_val = est / se if se > 0 else float("nan")
        p_val = float(2 * sp_stats.t.sf(abs(t_val), df=n - len(fe)))
        fixed_effects.append({
            "name": name,
            "estimate": est,
            "se": se,
            "t": t_val,
            "p": p_val,
            "ci_lower": est - 1.96 * se,
            "ci_upper": est + 1.96 * se,
            "significant": p_val < 0.05,
        })

    # ── Random effects ────────────────────────────────────────────────
    cov_re = model.cov_re
    var_components = {}
    for idx in cov_re.index:
        for col in cov_re.columns:
            label = f"Var({idx})" if idx == col else f"Cov({idx},{col})"
            var_components[label] = float(cov_re.loc[idx, col])
    var_components["Residual"] = float(model.scale)

    # Correlation between random intercept and slope
    re_corr = None
    if cov_re.shape[0] >= 2:
        var_int = cov_re.iloc[0, 0]
        var_slope = cov_re.iloc[1, 1]
        cov_is = cov_re.iloc[0, 1]
        denom = np.sqrt(var_int * var_slope)
        if denom > 0:
            re_corr = float(cov_is / denom)

    # ── Individual trajectories ───────────────────────────────────────
    re_raw = model.random_effects
    avg_intercept = float(fe["Intercept"])
    avg_slope = float(fe["time"])
    avg_quad = float(fe.get("time_sq", 0))

    individual_trajectories = []
    for subj in unique_subjects:
        re_vals = re_raw.get(subj, re_raw.get(str(subj), {}))
        ri = float(re_vals.iloc[0]) if hasattr(re_vals, "iloc") else 0.0
        rs = float(re_vals.iloc[1]) if hasattr(re_vals, "iloc") and len(re_vals) > 1 else 0.0
        individual_trajectories.append({
            "subject": subj,
            "intercept": avg_intercept + ri,
            "slope": avg_slope + rs,
            "random_intercept": ri,
            "random_slope": rs,
        })

    # ── Model fit ─────────────────────────────────────────────────────
    llf = float(model.llf)
    n_params = len(fe) + len(cov_re.values.flatten()) + 1
    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + n_params * np.log(n)

    return {
        "fixed_effects": fixed_effects,
        "random_effects": {
            "variance_components": var_components,
            "intercept_slope_correlation": re_corr,
        },
        "average_trajectory": {
            "intercept": avg_intercept,
            "slope": avg_slope,
            **({"quadratic": avg_quad} if req.quadratic else {}),
            "interpretation": (
                f"Average starting value = {avg_intercept:.3f}, "
                f"average rate of change = {avg_slope:.3f} per unit time"
                + (f", quadratic = {avg_quad:.3f}" if req.quadratic else "")
                + "."
            ),
        },
        "individual_trajectories": individual_trajectories,
        "model_fit": {
            "log_likelihood": llf,
            "deviance": -2 * llf,
            "aic": aic,
            "bic": bic,
        },
        "n_observations": n,
        "n_subjects": n_subjects,
        "interpretation": _interpret_growth(avg_slope, re_corr, fixed_effects),
    }


@router.post("/icc")
def compute_icc(req: ICCRequest):
    """Compute intraclass correlation coefficients for a grouping variable."""
    n = len(req.values)
    if n < 3:
        raise HTTPException(400, "Need at least 3 observations.")
    if len(req.groups) != n:
        raise HTTPException(400, "values and groups must have the same length.")

    values = np.array(req.values, dtype=float)
    groups = np.array(req.groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    if k < 2:
        raise HTTPException(400, "Need at least 2 groups.")
    if k >= n:
        raise HTTPException(400, "Number of groups must be less than number of observations.")

    # Group means and sizes
    group_means = []
    group_sizes = []
    group_data = []
    for g in unique_groups:
        mask = groups == g
        gvals = values[mask]
        group_data.append(gvals)
        group_means.append(np.mean(gvals))
        group_sizes.append(len(gvals))

    group_means = np.array(group_means)
    group_sizes = np.array(group_sizes)
    grand_mean = np.mean(values)
    n_total = n

    # One-way ANOVA decomposition
    ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
    ss_within = sum(np.sum((gvals - gmean) ** 2) for gvals, gmean in zip(group_data, group_means))

    df_between = k - 1
    df_within = n_total - k

    if df_between == 0 or df_within == 0:
        raise HTTPException(400, "Insufficient degrees of freedom.")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-test
    f_stat = ms_between / ms_within if ms_within > 0 else float("inf")
    f_p = float(sp_stats.f.sf(f_stat, df_between, df_within))

    # Average group size (harmonic mean for unbalanced designs)
    n_0 = (n_total - np.sum(group_sizes ** 2) / n_total) / (k - 1) if k > 1 else np.mean(group_sizes)

    # Variance components
    var_between = max((ms_between - ms_within) / n_0, 0.0)
    var_within = ms_within

    # ICC(1): proportion of variance due to group membership
    icc1 = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0

    # ICC(2): reliability of group means
    icc2 = (ms_between - ms_within) / ms_between if ms_between > 0 else 0.0
    icc2 = max(icc2, 0.0)

    # Design effect (Kish 1965)
    avg_n = np.mean(group_sizes)
    design_effect = 1 + (avg_n - 1) * icc1

    return {
        "icc1": {
            "value": float(icc1),
            "interpretation": _interpret_icc(icc1),
            "description": "Proportion of total variance attributable to group membership.",
        },
        "icc2": {
            "value": float(icc2),
            "interpretation": (
                "Group means are reliable."
                if icc2 > 0.70
                else "Group means have low reliability; consider increasing within-group n."
            ),
            "description": "Reliability of group means.",
        },
        "f_test": {
            "f_statistic": float(f_stat),
            "df_between": df_between,
            "df_within": df_within,
            "p_value": f_p,
            "significant": f_p < 0.05,
            "interpretation": (
                "Significant group effect: groups differ meaningfully."
                if f_p < 0.05
                else "Non-significant group effect: limited between-group variation."
            ),
        },
        "variance_components": {
            "between_group": float(var_between),
            "within_group": float(var_within),
        },
        "design_effect": {
            "value": float(design_effect),
            "interpretation": (
                f"Effective sample size is reduced by a factor of {design_effect:.2f} "
                f"due to clustering. Ignoring nesting would underestimate standard errors."
                if design_effect > 2
                else f"Design effect of {design_effect:.2f} is small; "
                     "clustering has limited impact on standard errors."
            ),
        },
        "n_observations": n_total,
        "n_groups": int(k),
        "avg_group_size": float(avg_n),
        "interpretation": (
            f"ICC(1) = {icc1:.3f}: "
            + _interpret_icc(icc1)
            + f" The F-test {'is' if f_p < 0.05 else 'is not'} significant "
            f"(F({df_between},{df_within}) = {f_stat:.2f}, p = {f_p:.4f}). "
            f"Design effect = {design_effect:.2f}."
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _interpret_icc(icc: float) -> str:
    """Interpret ICC magnitude using conventional thresholds."""
    if icc < 0.05:
        return "Negligible clustering; multilevel modeling may not be necessary."
    elif icc < 0.10:
        return "Small clustering effect. Multilevel modeling is advisable to correct standard errors."
    elif icc < 0.25:
        return "Moderate clustering. Multilevel modeling is recommended."
    elif icc < 0.50:
        return "Large clustering effect. Multilevel modeling is essential."
    else:
        return "Very large clustering. Most variance is between groups."


def _interpret_two_level(
    icc: float,
    fixed_effects: list[dict],
    r2m: float,
    r2c: float,
) -> str:
    """Generate a plain-language interpretation of two-level HLM results."""
    sig_predictors = [
        fe["name"] for fe in fixed_effects
        if fe["significant"] and fe["name"] != "Intercept"
    ]
    parts = [
        f"ICC = {icc:.3f}: " + _interpret_icc(icc),
    ]
    if sig_predictors:
        parts.append(
            f"Significant fixed effects: {', '.join(sig_predictors)}."
        )
    else:
        parts.append("No level-1 predictors reached significance.")
    parts.append(
        f"Marginal R-squared = {r2m:.3f} (fixed effects only), "
        f"Conditional R-squared = {r2c:.3f} (fixed + random)."
    )
    if r2c - r2m > 0.10:
        parts.append("Random effects explain substantial additional variance beyond fixed effects.")
    return " ".join(parts)


def _interpret_growth(
    avg_slope: float,
    re_corr: float | None,
    fixed_effects: list[dict],
) -> str:
    """Generate interpretation for growth curve model."""
    slope_sig = any(fe["name"] == "time" and fe["significant"] for fe in fixed_effects)
    quad_sig = any(fe["name"] == "time_sq" and fe["significant"] for fe in fixed_effects)

    parts = []
    if slope_sig:
        direction = "increasing" if avg_slope > 0 else "decreasing"
        parts.append(f"Significant linear trend: outcome is {direction} over time (slope = {avg_slope:.3f}).")
    else:
        parts.append("No significant linear trend over time.")

    if quad_sig:
        parts.append("Significant quadratic component: the rate of change itself changes over time.")

    if re_corr is not None:
        if re_corr < -0.3:
            parts.append(
                f"Negative intercept-slope correlation ({re_corr:.2f}): "
                "those starting higher tend to change less (fan-in pattern)."
            )
        elif re_corr > 0.3:
            parts.append(
                f"Positive intercept-slope correlation ({re_corr:.2f}): "
                "those starting higher tend to change more (fan-out pattern)."
            )

    return " ".join(parts) if parts else "Growth curve model fitted successfully."
