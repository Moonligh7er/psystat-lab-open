"""Advanced Stats — mixed models, meta-analysis, time series, non-parametric tests,
correlation matrices, multiple regression, mediation and repeated measures.

These endpoints back the advanced tools in Stats Studio. Reference values for the
regression tests (tests/test_advanced_stats.py) come from R 4.5.2: stats, lme4,
afex and metafor.
"""

import math
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from app.mixed import fit_mixedlm, fixed_effect_se
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

from app.routers import advanced_anova as _anova
from app.routers import meta_analysis as _meta
from app.routers import time_series as _ts

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class MixedModelRequest(BaseModel):
    """Linear mixed-effects model (random intercepts/slopes)."""
    dependent: list[float]
    fixed_effects: list[list[float]]  # columns of fixed predictors
    groups: list[str | int]  # grouping variable for random effects
    random_slopes: bool = False  # include random slopes for first fixed effect
    fixed_labels: list[str] | None = None


class MetaAnalysisRequest(BaseModel):
    """Fixed/random-effects meta-analysis."""
    effect_sizes: list[float]  # e.g., Cohen's d, log OR
    standard_errors: list[float]
    study_labels: list[str] | None = None
    method: str = "random"  # "fixed" or "random" (DerSimonian-Laird)


class TimeSeriesRequest(BaseModel):
    """Time series decomposition and ARIMA forecasting."""
    values: list[float]
    period: int | None = None  # seasonal period (auto-detected if None)
    forecast_steps: int = 10
    arima_order: list[int] | None = None  # [p, d, q] — auto-fit if None


class MultilevelRequest(BaseModel):
    """Two-level hierarchical/multilevel model."""
    outcome: list[float]
    level1_predictors: list[list[float]]  # individual-level predictors
    level2_predictors: list[list[float]] | None = None  # group-level predictors
    groups: list[str | int]
    predictor_labels: list[str] | None = None


class NonParametricRequest(BaseModel):
    """Non-parametric test suite."""
    test: str  # "mann-whitney", "wilcoxon", "kruskal", "friedman", "spearman", "kendall"
    group1: list[float] | None = None
    group2: list[float] | None = None
    groups: list[list[float]] | None = None  # for kruskal/friedman
    alternative: str = "two-sided"


class CorrelationMatrixRequest(BaseModel):
    """Correlation matrix with multiple methods and p-values."""
    variables: list[list[float]]  # each inner list is a variable
    labels: list[str] | None = None
    method: str = "pearson"  # "pearson", "spearman", "kendall"


class MultipleRegressionRequest(BaseModel):
    """Multiple regression with diagnostics."""
    predictors: list[list[float]]  # each inner list is one predictor (column)
    outcome: list[float]
    predictor_labels: list[str] | None = None


class MediationtRequest(BaseModel):
    """Simple mediation X -> M -> Y."""
    x: list[float]  # independent variable
    m: list[float]  # mediator
    y: list[float]  # dependent variable
    n_bootstrap: int = 5000
    seed: int = 20240101


class RepeatedMeasuresRequest(BaseModel):
    """Repeated measures / within-subjects design."""
    measurements: list[list[float]]  # each list = one condition/timepoint
    condition_labels: list[str] | None = None


# ── Helpers ───────────────────────────────────────────────────────────

def _labels(labels, n, prefix):
    if labels is None:
        return [f"{prefix}{i + 1}" for i in range(n)]
    if len(labels) != n:
        raise HTTPException(400, f"Expected {n} labels, got {len(labels)}.")
    if len(set(labels)) != n:
        raise HTTPException(400, "Labels must be unique.")
    return list(labels)


def _finite(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(np.asarray(a, dtype=float))):
            raise HTTPException(400, "All values must be finite numbers.")


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/mixed-model")
def mixed_effects_model(req: MixedModelRequest):
    """Linear mixed-effects model fitted by REML (statsmodels MixedLM).

    Random intercepts for each group, plus an optional random slope for the first
    fixed effect (with an intercept–slope covariance). Fixed-effect tests are Wald z.
    AIC/BIC are computed from the REML log-likelihood and count fixed effects,
    variance–covariance parameters and the residual variance; REML criteria are only
    comparable between models with the same fixed effects.
    """
    n = len(req.dependent)
    if n < 10:
        raise HTTPException(400, "Need at least 10 observations for mixed models.")
    n_fixed = len(req.fixed_effects)
    if n_fixed == 0:
        raise HTTPException(400, "Provide at least one fixed-effect predictor.")
    if any(len(col) != n for col in req.fixed_effects):
        raise HTTPException(400, "All fixed effect columns must match dependent variable length.")
    if len(req.groups) != n:
        raise HTTPException(400, "Groups array must match dependent variable length.")
    _finite(req.dependent, *req.fixed_effects)
    labels = _labels(req.fixed_labels, n_fixed, "X")
    n_groups = len(set(map(str, req.groups)))
    if n_groups < 2:
        raise HTTPException(400, "Need at least 2 groups for random effects.")

    # Internal column names keep the formula valid whatever the user's labels are
    safe = [f"x{i}" for i in range(n_fixed)]
    df = pd.DataFrame({"y": req.dependent, "group": [str(g) for g in req.groups]})
    for i, col in enumerate(req.fixed_effects):
        df[safe[i]] = col
    formula = "y ~ " + " + ".join(safe)
    name_of = {"Intercept": "Intercept", **dict(zip(safe, labels))}

    try:
        if req.random_slopes:
            model = mixedlm(formula, df, groups=df["group"], re_formula=f"~{safe[0]}")
        else:
            model = mixedlm(formula, df, groups=df["group"])
        result = fit_mixedlm(model, reml=True)
    except Exception as e:
        raise HTTPException(400, f"Model fitting failed: {e}")

    fixed_effects = {}
    for name in result.fe_params.index:
        fixed_effects[name_of.get(name, name)] = {
            "coefficient": float(result.fe_params[name]),
            "std_error": float(fixed_effect_se(result)[0][name]),
            "z_value": float(result.tvalues[name]),
            "p_value": float(result.pvalues[name]),
        }

    cov = result.cov_re
    re_names = ["Intercept" if c == "Group" else name_of.get(c, c) for c in cov.columns]
    variance = {}
    for i, a in enumerate(re_names):
        variance[f"{a} (group)"] = float(cov.iloc[i, i])
    if len(re_names) == 2:
        variance[f"Cov({re_names[0]}, {re_names[1]})"] = float(cov.iloc[0, 1])
    variance["Residual"] = float(result.scale)

    icc = None
    if not req.random_slopes:
        tau00 = float(cov.iloc[0, 0])
        icc = tau00 / (tau00 + float(result.scale))

    k_params = len(result.fe_params) + (3 if req.random_slopes else 1) + 1
    ll = float(result.llf)
    return {
        "fixed_effects": fixed_effects,
        "random_effects_variance": variance,
        "icc": icc,
        "estimation": "REML",
        "log_likelihood": ll,
        "aic": -2 * ll + 2 * k_params,
        "bic": -2 * ll + math.log(n) * k_params,
        "n_parameters": k_params,
        "n_observations": n,
        "n_groups": n_groups,
        "converged": bool(result.converged),
        "note": "Tests are Wald z. AIC/BIC use the REML likelihood: compare only models with identical fixed effects.",
    }


@router.post("/meta-analysis")
def meta_analysis(req: MetaAnalysisRequest):
    """Fixed or DerSimonian–Laird random-effects meta-analysis (shares the Meta-Analysis module's code)."""
    if req.method not in ("fixed", "random"):
        raise HTTPException(400, "method must be 'fixed' or 'random'.")
    es, se, k = _meta._arrays(req.effect_sizes, req.standard_errors)
    labels = _meta._labels(req.study_labels, k)
    tau2 = _meta._tau2_dl(es, se) if req.method == "random" else 0.0
    fit = _meta._pool_re(es, se, tau2)
    het = _meta._heterogeneity(es, se)
    w = fit["w"]
    z = sp_stats.norm.ppf(0.975)
    studies = [{"label": labels[i], "effect_size": float(es[i]), "se": float(se[i]),
                "ci_lower": float(es[i] - z * se[i]), "ci_upper": float(es[i] + z * se[i]),
                "weight": float(w[i] / np.sum(w) * 100)} for i in range(k)]
    return {
        "pooled_effect": fit["mu"],
        "pooled_se": fit["se"],
        "ci_lower": fit["ci"][0],
        "ci_upper": fit["ci"][1],
        "z_value": fit["stat"],
        "p_value": fit["p"],
        "method": req.method,
        "k_studies": k,
        "Q_statistic": het["Q"],
        "Q_df": het["Q_df"],
        "Q_p_value": het["Q_p"],
        "I_squared": het["I_squared"],
        "tau_squared": tau2,
        "prediction_interval": _meta._prediction_interval(fit["mu"], tau2, fit["se"], k) if req.method == "random" else None,
        "studies": studies,
        "significant": bool(fit["p"] < 0.05),
    }


@router.post("/time-series")
def time_series_analysis(req: TimeSeriesRequest):
    """Summary, ADF test, ACF/PACF, classical decomposition and an ARIMA forecast.

    ARIMA uses the Time Series module's fitting (d by KPSS, then p and q by AICc).
    """
    values = np.asarray(req.values, dtype=float)
    n = len(values)
    if n < 10:
        raise HTTPException(400, "Need at least 10 observations for time series analysis.")
    _finite(values)
    if not 1 <= req.forecast_steps <= 200:
        raise HTTPException(400, "forecast_steps must be between 1 and 200.")

    period, period_source = req.period, "user"
    if period is not None and period < 2:
        raise HTTPException(400, "period must be at least 2.")
    if period is None:
        period_source = "first ACF peak"
        if n > 20:
            acf_vals = acf(values, nlags=min(n // 2, 50), fft=True)
            for i in range(2, len(acf_vals) - 1):
                if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1] and acf_vals[i] > 0:
                    period = i
                    break
        if period is None:
            period, period_source = None, "none detected"

    result = {}
    if period is not None and n >= 2 * period:
        decomp = seasonal_decompose(values, model="additive", period=period)
        result["decomposition"] = {
            "trend": [None if np.isnan(v) else float(v) for v in decomp.trend],
            "seasonal": [float(v) for v in decomp.seasonal],
            "residual": [None if np.isnan(v) else float(v) for v in decomp.resid],
            "period": period,
            "period_source": period_source,
        }
    else:
        result["decomposition"] = None

    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(values, autolag="AIC")
    result["stationarity"] = {
        "adf_statistic": float(adf_stat), "p_value": float(adf_p), "n_lags": int(adf_lags),
        "is_stationary": bool(adf_p < 0.05),
        "critical_values": {k: float(v) for k, v in adf_crit.items()},
    }

    n_lags = min(n // 2 - 1, 40)
    if n_lags > 1:
        result["acf"] = [float(v) for v in acf(values, nlags=n_lags, fft=True)]
        result["pacf"] = [float(v) for v in pacf(values, nlags=n_lags)]

    try:
        arima = _ts.arima_model(_ts.ARIMARequest(values=values.tolist(), order=req.arima_order, n_forecast=req.forecast_steps))
        o = arima["order"]
        result["arima"] = {
            "order": [o["p"], o["d"], o["q"]],
            "order_selection": arima["order_selection"],
            "aic": arima["aic"],
            "bic": arima["bic"],
            "forecast": arima["forecast"]["mean"],
            "forecast_ci_lower": arima["forecast"]["lower_ci"],
            "forecast_ci_upper": arima["forecast"]["upper_ci"],
            "ljung_box_p": arima["residual_diagnostics"]["ljung_box_p"],
        }
    except HTTPException as e:
        result["arima"] = {"error": e.detail}

    result["summary"] = {
        "n": n, "mean": float(values.mean()), "std": float(values.std(ddof=1)),
        "min": float(values.min()), "max": float(values.max()),
        "trend_direction": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "flat",
    }
    return result


@router.post("/non-parametric")
def non_parametric_tests(req: NonParametricRequest):
    """Non-parametric tests with effect sizes (scipy; p-values as R's defaults noted per test)."""
    test = req.test.lower().replace(" ", "-")
    if req.alternative not in ("two-sided", "less", "greater"):
        raise HTTPException(400, "alternative must be two-sided, less or greater.")

    if test == "mann-whitney":
        if not req.group1 or not req.group2:
            raise HTTPException(400, "Mann-Whitney requires group1 and group2.")
        g1, g2 = np.asarray(req.group1, float), np.asarray(req.group2, float)
        _finite(g1, g2)
        res = sp_stats.mannwhitneyu(g1, g2, alternative=req.alternative)
        n1, n2 = len(g1), len(g2)
        return {
            "test": "Mann-Whitney U",
            "U_statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            # rank-biserial r = 2U1/(n1 n2) - 1: positive when group 1 tends to be larger
            "rank_biserial_r": float(2 * res.statistic / (n1 * n2) - 1),
            "group1_median": float(np.median(g1)),
            "group2_median": float(np.median(g2)),
            "group1_n": n1,
            "group2_n": n2,
            "significant": bool(res.pvalue < 0.05),
        }

    if test == "wilcoxon":
        if not req.group1 or not req.group2:
            raise HTTPException(400, "Wilcoxon requires group1 and group2 (paired).")
        g1, g2 = np.asarray(req.group1, float), np.asarray(req.group2, float)
        if len(g1) != len(g2):
            raise HTTPException(400, "Paired samples must have equal length.")
        _finite(g1, g2)
        diff = g1 - g2
        nz = diff[diff != 0]
        if len(nz) == 0:
            raise HTTPException(400, "All paired differences are zero; the test is undefined.")
        res = sp_stats.wilcoxon(g1, g2, alternative=req.alternative)
        ranks = sp_stats.rankdata(np.abs(nz))
        w_plus, w_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        total = len(nz) * (len(nz) + 1) / 2
        return {
            "test": "Wilcoxon Signed-Rank",
            "W_statistic": w_plus,  # V = sum of positive ranks, as reported by R's wilcox.test
            "W_plus": w_plus,
            "W_minus": w_minus,
            "p_value": float(res.pvalue),
            # matched-pairs rank-biserial correlation (Kerby, 2014)
            "rank_biserial_r": (w_plus - w_minus) / total,
            "median_difference": float(np.median(diff)),
            "n_pairs": len(g1),
            "n_nonzero": int(len(nz)),
            "significant": bool(res.pvalue < 0.05),
        }

    if test == "kruskal":
        if not req.groups or len(req.groups) < 2:
            raise HTTPException(400, "Kruskal-Wallis requires at least 2 groups.")
        groups = [np.asarray(g, float) for g in req.groups]
        if any(len(g) == 0 for g in groups):
            raise HTTPException(400, "Every group needs at least one value.")
        _finite(*groups)
        H, p = sp_stats.kruskal(*groups)
        N, k = sum(len(g) for g in groups), len(groups)
        return {
            "test": "Kruskal-Wallis H",
            "H_statistic": float(H),
            "p_value": float(p),
            "df": k - 1,
            "epsilon_squared": float(H / (N - 1)),               # Tomczak & Tomczak (2014)
            "eta_squared_H": float(max(0.0, (H - k + 1) / (N - k))),
            "group_medians": [float(np.median(g)) for g in groups],
            "group_ns": [len(g) for g in groups],
            "significant": bool(p < 0.05),
        }

    if test == "friedman":
        if not req.groups or len(req.groups) < 3:
            raise HTTPException(400, "Friedman test requires at least 3 conditions.")
        groups = [np.asarray(g, float) for g in req.groups]
        if len(set(len(g) for g in groups)) != 1:
            raise HTTPException(400, "All conditions must have the same number of observations.")
        _finite(*groups)
        chi2, p = sp_stats.friedmanchisquare(*groups)
        k, n = len(groups), len(groups[0])
        return {
            "test": "Friedman",
            "chi2_statistic": float(chi2),
            "p_value": float(p),
            "kendalls_W": float(chi2 / (n * (k - 1))),
            "df": k - 1,
            "condition_medians": [float(np.median(g)) for g in groups],
            "n_subjects": n,
            "n_conditions": k,
            "significant": bool(p < 0.05),
        }

    if test in ("spearman", "kendall"):
        if not req.group1 or not req.group2:
            raise HTTPException(400, f"{test.title()} requires group1 and group2.")
        g1, g2 = np.asarray(req.group1, float), np.asarray(req.group2, float)
        if len(g1) != len(g2):
            raise HTTPException(400, "Variables must have equal length.")
        if len(g1) < 3:
            raise HTTPException(400, "Need at least 3 pairs.")
        _finite(g1, g2)
        if test == "spearman":
            res = sp_stats.spearmanr(g1, g2, alternative=req.alternative)
            name = "Spearman's rho"
        else:
            res = sp_stats.kendalltau(g1, g2, alternative=req.alternative)
            name = "Kendall's tau-b"
        return {"test": name, "statistic": float(res.statistic), "p_value": float(res.pvalue),
                "n": len(g1), "significant": bool(res.pvalue < 0.05)}

    raise HTTPException(400, f"Unknown test: {req.test}. Options: mann-whitney, wilcoxon, kruskal, friedman, spearman, kendall")


@router.post("/correlation-matrix")
def correlation_matrix(req: CorrelationMatrixRequest):
    """Correlation matrix with pairwise p-values (no multiplicity adjustment)."""
    if req.method not in ("pearson", "spearman", "kendall"):
        raise HTTPException(400, "method must be pearson, spearman or kendall.")
    n_vars = len(req.variables)
    if n_vars < 2:
        raise HTTPException(400, "Need at least 2 variables.")
    n_obs = len(req.variables[0])
    if any(len(v) != n_obs for v in req.variables):
        raise HTTPException(400, "All variables must have equal length.")
    if n_obs < 3:
        raise HTTPException(400, "Need at least 3 observations.")
    _finite(*req.variables)
    labels = _labels(req.labels, n_vars, "V")
    data = np.array(req.variables, dtype=float).T
    fn = {"pearson": sp_stats.pearsonr, "spearman": sp_stats.spearmanr, "kendall": sp_stats.kendalltau}[req.method]

    corr = np.eye(n_vars)
    pmat = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if np.ptp(data[:, i]) == 0 or np.ptp(data[:, j]) == 0:
                r, p = float("nan"), float("nan")
            else:
                r, p = fn(data[:, i], data[:, j])
            corr[i, j] = corr[j, i] = r
            pmat[i, j] = pmat[j, i] = p

    return {
        "method": req.method,
        "labels": labels,
        "correlation_matrix": corr.tolist(),
        "p_value_matrix": pmat.tolist(),
        "n_observations": n_obs,
        "n_variables": n_vars,
        "note": "p-values are unadjusted; with many pairs consider a correction (e.g. Holm).",
    }


@router.post("/multiple-regression")
def multiple_regression_diagnostics(req: MultipleRegressionRequest):
    """OLS multiple regression with VIF, Cook's distance and assumption checks."""
    if not req.predictors:
        raise HTTPException(400, "Provide at least one predictor.")
    n = len(req.outcome)
    if any(len(col) != n for col in req.predictors):
        raise HTTPException(400, "Every predictor must have one value per observation of the outcome.")
    _finite(req.outcome, *req.predictors)
    X = np.array(req.predictors, dtype=float).T
    y = np.asarray(req.outcome, dtype=float)
    p = X.shape[1]
    labels = _labels(req.predictor_labels, p, "X")
    if n < p + 2:
        raise HTTPException(400, f"Need at least {p + 2} observations for {p} predictors.")
    X_const = sm.add_constant(X, has_constant="add")
    if np.linalg.matrix_rank(X_const) < p + 1:
        raise HTTPException(400, "Predictors are perfectly collinear (or constant).")
    model = sm.OLS(y, X_const).fit()

    vifs = []
    for i in range(p):
        if p == 1:
            vifs.append(1.0)
            continue
        others = sm.add_constant(np.delete(X, i, axis=1), has_constant="add")
        r2_i = sm.OLS(X[:, i], others).fit().rsquared
        vifs.append(1 / (1 - r2_i) if r2_i < 1 else float("inf"))

    cooks_d = model.get_influence().cooks_distance[0]
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X_const)
    sw_p = float(sp_stats.shapiro(model.resid)[1]) if 3 <= n <= 5000 else None
    ci = model.conf_int()

    return {
        "coefficients": [
            {"name": name, "coefficient": float(model.params[i]), "std_error": float(model.bse[i]),
             "t_value": float(model.tvalues[i]), "p_value": float(model.pvalues[i]),
             "ci_lower": float(ci[i][0]), "ci_upper": float(ci[i][1])}
            for i, name in enumerate(["Intercept"] + labels)
        ],
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_df": [int(model.df_model), int(model.df_resid)],
        "f_p_value": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n": n,
        "p_predictors": p,
        "vif": {labels[i]: float(vifs[i]) for i in range(p)},
        "multicollinearity_warning": any(v > 10 for v in vifs),
        "durbin_watson": float(durbin_watson(model.resid)),
        "breusch_pagan_statistic": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "breusch_pagan_variant": "Koenker (studentised)",
        "heteroscedasticity_warning": bool(bp_p < 0.05),
        "residuals_normality_p": sw_p,
        "cooks_distance": [float(d) for d in cooks_d],
        "influential_points": [int(i) for i, d in enumerate(cooks_d) if d > 4 / n],
        "influence_rule": "Cook's D > 4/n (observation indices are 0-based)",
    }


@router.post("/mediation")
def mediation_analysis(req: MediationtRequest):
    """Simple mediation X → M → Y: OLS paths, Sobel test and a percentile bootstrap CI for a×b.

    The bootstrap CI (Preacher & Hayes, 2008) is the recommended test of the indirect
    effect; the Sobel test assumes a×b is normally distributed and is shown for reference.
    "Full" vs "partial" mediation labels follow Baron & Kenny and depend on power.
    """
    x, m, y = (np.asarray(v, dtype=float) for v in (req.x, req.m, req.y))
    n = len(x)
    if n != len(m) or n != len(y):
        raise HTTPException(400, "x, m, y must have equal length.")
    if n < 10:
        raise HTTPException(400, "Need at least 10 observations.")
    _finite(x, m, y)
    if not 1000 <= req.n_bootstrap <= 20000:
        raise HTTPException(400, "n_bootstrap must be between 1000 and 20000.")

    X = sm.add_constant(x)
    model_c = sm.OLS(y, X).fit()
    model_a = sm.OLS(m, X).fit()
    model_bc = sm.OLS(y, sm.add_constant(np.column_stack([x, m]))).fit()
    c, c_p = model_c.params[1], model_c.pvalues[1]
    a, a_se, a_p = model_a.params[1], model_a.bse[1], model_a.pvalues[1]
    c_prime, c_prime_p = model_bc.params[1], model_bc.pvalues[1]
    b, b_se, b_p = model_bc.params[2], model_bc.bse[2], model_bc.pvalues[2]
    indirect = a * b

    sobel_se = math.sqrt(a ** 2 * b_se ** 2 + b ** 2 * a_se ** 2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else float("nan")
    sobel_p = float(2 * sp_stats.norm.sf(abs(sobel_z)))

    # Percentile bootstrap of a×b (vectorised OLS on resampled rows)
    rng = np.random.default_rng(req.seed)
    idx = rng.integers(0, n, size=(req.n_bootstrap, n))
    xb, mb, yb = x[idx], m[idx], y[idx]
    xc = xb - xb.mean(axis=1, keepdims=True)
    mc = mb - mb.mean(axis=1, keepdims=True)
    yc = yb - yb.mean(axis=1, keepdims=True)
    sxx, sxm = (xc * xc).sum(1), (xc * mc).sum(1)
    smm, sxy, smy = (mc * mc).sum(1), (xc * yc).sum(1), (mc * yc).sum(1)
    a_b = sxm / sxx
    det = sxx * smm - sxm ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        b_b = (sxx * smy - sxm * sxy) / det
    ab = a_b * b_b
    ab = ab[np.isfinite(ab)]
    ci_lo, ci_hi = (float(v) for v in np.percentile(ab, [2.5, 97.5]))
    boot_sig = not (ci_lo <= 0 <= ci_hi)

    return {
        "total_effect_c": float(c), "total_effect_p": float(c_p),
        "path_a": float(a), "path_a_p": float(a_p),
        "path_b": float(b), "path_b_p": float(b_p),
        "direct_effect_c_prime": float(c_prime), "direct_effect_p": float(c_prime_p),
        "indirect_effect_ab": float(indirect),
        "indirect_ci_lower": ci_lo, "indirect_ci_upper": ci_hi,
        "bootstrap": {"method": "percentile", "n_resamples": int(len(ab)), "seed": req.seed},
        "sobel_z": float(sobel_z), "sobel_p": sobel_p,
        "proportion_mediated": float(indirect / c) if abs(c) > 1e-10 else None,
        "mediation_significant": bool(boot_sig),
        "full_mediation": bool(boot_sig and c_prime_p >= 0.05),
        "partial_mediation": bool(boot_sig and c_prime_p < 0.05),
        "n": n,
        "note": "Indirect effect judged by the 95% bootstrap CI excluding 0. Proportion mediated is unstable when c is small.",
    }


@router.post("/repeated-measures")
def repeated_measures_anova(req: RepeatedMeasuresRequest):
    """Within-subjects comparison: one-way repeated-measures ANOVA (with sphericity
    corrections) and the Friedman test, plus Bonferroni-adjusted pairwise tests.
    """
    k = len(req.measurements)
    if k < 2:
        raise HTTPException(400, "Need at least 2 conditions/timepoints.")
    n = len(req.measurements[0])
    if any(len(m) != n for m in req.measurements):
        raise HTTPException(400, "All conditions must have the same number of observations (one per subject).")
    if n < 3:
        raise HTTPException(400, "Need at least 3 subjects.")
    _finite(*req.measurements)
    labels = _labels(req.condition_labels, k, "T")
    data = np.array(req.measurements, dtype=float).T  # subjects x conditions

    rm = _anova.repeated_measures_anova(_anova.RepeatedMeasuresRequest(data=data.tolist(), condition_labels=labels))
    an, eps = rm["anova"], rm["epsilon"]
    gg_p = float(sp_stats.f.sf(an["F"], an["df_conditions"] * eps["greenhouse_geisser"], an["df_error"] * eps["greenhouse_geisser"])) if k >= 3 else an["p"]

    if k >= 3:
        chi2, fp = sp_stats.friedmanchisquare(*[data[:, j] for j in range(k)])
        nonpar = {"test": "Friedman", "statistic": float(chi2), "df": k - 1, "p_value": float(fp),
                  "kendalls_W": float(chi2 / (n * (k - 1)))}
    else:
        w = sp_stats.wilcoxon(data[:, 0], data[:, 1])
        nonpar = {"test": "Wilcoxon signed-rank", "statistic": float(w.statistic), "df": None, "p_value": float(w.pvalue), "kendalls_W": None}

    n_comp = k * (k - 1) // 2
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            t = sp_stats.ttest_rel(data[:, i], data[:, j])
            pairwise.append({"condition_i": labels[i], "condition_j": labels[j],
                             "mean_difference": float(np.mean(data[:, i] - data[:, j])),
                             "t": float(t.statistic), "df": n - 1, "p_value": float(t.pvalue),
                             "p_adjusted": min(float(t.pvalue) * n_comp, 1.0),
                             "significant": bool(min(float(t.pvalue) * n_comp, 1.0) < 0.05)})

    return {
        "test": "Repeated-measures ANOVA",
        "statistic": an["F"],
        "df": [an["df_conditions"], an["df_error"]],
        "p_value": an["p"],
        "p_greenhouse_geisser": gg_p,
        "epsilon_greenhouse_geisser": eps["greenhouse_geisser"] if k >= 3 else 1.0,
        "epsilon_huynh_feldt": eps["huynh_feldt"] if k >= 3 else 1.0,
        "mauchly": rm["sphericity_test"] if k >= 3 else None,
        "partial_eta_squared": an["partial_eta_squared"],
        "nonparametric": nonpar,
        "kendalls_W": nonpar["kendalls_W"],
        "n_subjects": n,
        "n_conditions": k,
        "condition_means": [float(v) for v in data.mean(axis=0)],
        "condition_medians": [float(v) for v in np.median(data, axis=0)],
        "condition_stds": [float(v) for v in data.std(axis=0, ddof=1)],
        "condition_labels": labels,
        "pairwise_comparisons": pairwise,
        "pairwise_method": "paired t-tests, Bonferroni-adjusted",
        "significant": bool((gg_p if k >= 3 else an["p"]) < 0.05),
    }
