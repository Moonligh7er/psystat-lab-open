"""Stats Studio — real statistical tests using scipy and statsmodels."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import t as t_dist
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

router = APIRouter()


# ── Request / Response Models ──────────────────────────────────────────

class TTestRequest(BaseModel):
    group1: list[float]
    group2: list[float]
    alternative: str = "two-sided"  # "two-sided", "less", "greater"


class PairedTTestRequest(BaseModel):
    before: list[float]
    after: list[float]
    alternative: str = "two-sided"


class AnovaRequest(BaseModel):
    groups: list[list[float]]


class ChiSquareRequest(BaseModel):
    observed: list[list[int]]


class RegressionRequest(BaseModel):
    x: list[list[float]]  # Can be multiple predictors
    y: list[float]


class PowerAnalysisRequest(BaseModel):
    effect_size: float
    alpha: float = 0.05
    power: float = 0.80
    test_type: str = "t-test"  # "t-test" or "anova"
    n_groups: int = 2  # for ANOVA


class DescriptiveRequest(BaseModel):
    data: list[float]


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/t-test")
def independent_t_test(req: TTestRequest):
    g1, g2 = np.array(req.group1), np.array(req.group2)
    if len(g1) < 2 or len(g2) < 2:
        raise HTTPException(400, "Each group must have at least 2 observations.")

    t_stat, p_value = sp_stats.ttest_ind(g1, g2, alternative=req.alternative)
    # Cohen's d
    pooled_std = np.sqrt(((len(g1) - 1) * g1.std(ddof=1)**2 + (len(g2) - 1) * g2.std(ddof=1)**2) / (len(g1) + len(g2) - 2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0
    # Levene's test for equality of variances
    lev_stat, lev_p = sp_stats.levene(g1, g2)

    # 95% CI for mean difference
    df = len(g1) + len(g2) - 2
    mean_diff = float(g1.mean() - g2.mean())
    se_diff = float(pooled_std * np.sqrt(1 / len(g1) + 1 / len(g2)))
    ci_low, ci_high = t_dist.interval(0.95, df, loc=mean_diff, scale=se_diff)

    # Glass's delta (using control group = group2 SD)
    g2_std = float(g2.std(ddof=1))
    glass_delta = mean_diff / g2_std if g2_std > 0 else 0.0

    # Hedges' g (bias-corrected Cohen's d)
    correction = 1 - (3 / (4 * df - 1))
    hedges_g = float(cohens_d) * correction

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "df": df,
        "group1_mean": float(g1.mean()),
        "group2_mean": float(g2.mean()),
        "group1_std": float(g1.std(ddof=1)),
        "group2_std": g2_std,
        "group1_n": len(g1),
        "group2_n": len(g2),
        "levene_statistic": float(lev_stat),
        "levene_p": float(lev_p),
        "significant": bool(p_value < 0.05),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "glass_delta": float(glass_delta),
        "hedges_g": float(hedges_g),
    }


@router.post("/paired-t-test")
def paired_t_test(req: PairedTTestRequest):
    before, after = np.array(req.before), np.array(req.after)
    if len(before) != len(after):
        raise HTTPException(400, "Before and after arrays must be the same length.")
    if len(before) < 2:
        raise HTTPException(400, "Need at least 2 paired observations.")

    # Everything (t, mean difference, CI, d) describes the change after − before, so their
    # signs agree. A one-sided `alternative` therefore also refers to after − before.
    t_stat, p_value = sp_stats.ttest_rel(after, before, alternative=req.alternative)
    diff = after - before
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

    # 95% CI for mean difference
    df = len(before) - 1
    mean_diff = float(diff.mean())
    se_diff = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    if se_diff > 0:
        ci_low, ci_high = t_dist.interval(0.95, df, loc=mean_diff, scale=se_diff)
    else:
        ci_low, ci_high = mean_diff, mean_diff

    # Hedges' g (bias-corrected Cohen's d)
    correction = 1 - (3 / (4 * df - 1))
    hedges_g = float(cohens_d) * correction

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "df": df,
        "mean_difference": mean_diff,
        "std_difference": float(diff.std(ddof=1)),
        "n_pairs": len(before),
        "significant": bool(p_value < 0.05),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "hedges_g": float(hedges_g),
    }


@router.post("/anova")
def one_way_anova(req: AnovaRequest):
    if len(req.groups) < 2:
        raise HTTPException(400, "Need at least 2 groups for ANOVA.")
    groups = [np.array(g) for g in req.groups]
    if any(len(g) < 2 for g in groups):
        raise HTTPException(400, "Each group must have at least 2 observations.")

    f_stat, p_value = sp_stats.f_oneway(*groups)

    # Eta-squared
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum(((g - grand_mean)**2).sum() for g in groups)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # Levene's test
    lev_stat, lev_p = sp_stats.levene(*groups)

    # Post-hoc Tukey HSD
    all_data = np.concatenate(groups)
    group_labels = np.concatenate([[i] * len(g) for i, g in enumerate(groups)])
    tukey = sp_stats.tukey_hsd(*groups)

    pairwise = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pairwise.append({
                "group_i": i,
                "group_j": j,
                "p_value": float(tukey.pvalue[i][j]),
                "significant": bool(tukey.pvalue[i][j] < 0.05),
            })

    # Omega-squared and partial eta-squared
    df_between = len(groups) - 1
    df_within = len(all_data) - len(groups)
    ss_within = float(ss_total - ss_between)
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    omega_sq = float((ss_between - df_between * ms_within) / (ss_total + ms_within)) if (ss_total + ms_within) > 0 else 0.0
    partial_eta_sq = float(ss_between / (ss_between + ss_within)) if (ss_between + ss_within) > 0 else 0.0

    # 95% CI for eta-squared using non-central F distribution
    try:
        f_val = float(f_stat)
        if f_val > 0 and df_between > 0 and df_within > 0:
            # Non-centrality parameter
            nc_lambda = f_val * df_between
            # Lower bound: find non-centrality giving upper p = 0.025
            ci_eta_lower = float(max(0.0, (f_val * df_between - df_between) / (f_val * df_between + df_within)))
            # Use non-central F to get CI bounds for eta-squared
            from scipy.stats import ncf
            # Lower CI: non-centrality parameter at alpha/2
            nc_low = 0.0
            try:
                # Invert to find non-centrality bounds
                nc_low = ncf.ppf(0.025, df_between, df_within, nc_lambda)
                nc_high = ncf.ppf(0.975, df_between, df_within, nc_lambda)
                eta_ci_lower = float(max(0.0, 1 - (1 / (1 + nc_low * df_between / (df_within * df_between)))))
                eta_ci_upper = float(min(1.0, 1 - (1 / (1 + nc_high * df_between / (df_within * df_between)))))
            except Exception:
                eta_ci_lower = float(max(0.0, eta_squared - 1.96 * np.sqrt(2 * df_between * (df_within + ss_total) ** 2 / (df_within ** 2 * (df_within + 2))) / ss_total)) if ss_total > 0 else 0.0
                eta_ci_upper = float(min(1.0, eta_squared + 1.96 * np.sqrt(2 * df_between * (df_within + ss_total) ** 2 / (df_within ** 2 * (df_within + 2))) / ss_total)) if ss_total > 0 else 0.0
        else:
            eta_ci_lower = 0.0
            eta_ci_upper = 0.0
    except Exception:
        eta_ci_lower = 0.0
        eta_ci_upper = 0.0

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "df_between": df_between,
        "df_within": df_within,
        "group_means": [float(g.mean()) for g in groups],
        "group_stds": [float(g.std(ddof=1)) for g in groups],
        "group_ns": [len(g) for g in groups],
        "levene_statistic": float(lev_stat),
        "levene_p": float(lev_p),
        "tukey_pairwise": pairwise,
        "significant": bool(p_value < 0.05),
        "omega_squared": float(omega_sq),
        "partial_eta_squared": float(partial_eta_sq),
        "eta_squared_ci_lower": float(eta_ci_lower),
        "eta_squared_ci_upper": float(eta_ci_upper),
    }


@router.post("/chi-square")
def chi_square_test(req: ChiSquareRequest):
    observed = np.array(req.observed)
    if observed.ndim != 2:
        raise HTTPException(400, "Observed must be a 2D array (contingency table).")
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        raise HTTPException(400, "Contingency table must be at least 2x2.")

    chi2, p_value, dof, expected = sp_stats.chi2_contingency(observed)
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0.0

    # Phi coefficient (for 2x2 tables: sqrt(chi2/n))
    phi = float(np.sqrt(chi2 / n)) if n > 0 else 0.0

    # Odds ratio with 95% CI (only for 2x2 tables)
    is_2x2 = bool(observed.shape == (2, 2))
    if is_2x2:
        a, b = float(observed[0][0]), float(observed[0][1])
        c, d = float(observed[1][0]), float(observed[1][1])
        if b > 0 and c > 0 and d > 0 and a > 0:
            odds_ratio = float((a * d) / (b * c))
            log_or = float(np.log(odds_ratio))
            se_log_or = float(np.sqrt(1/a + 1/b + 1/c + 1/d))
            or_ci_lower = float(np.exp(log_or - 1.96 * se_log_or))
            or_ci_upper = float(np.exp(log_or + 1.96 * se_log_or))
        else:
            odds_ratio = None
            or_ci_lower = None
            or_ci_upper = None
    else:
        odds_ratio = None
        or_ci_lower = None
        or_ci_upper = None

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "df": int(dof),
        "cramers_v": float(cramers_v),
        "expected": expected.tolist(),
        "observed": observed.tolist(),
        "significant": bool(p_value < 0.05),
        "phi": phi,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_lower": or_ci_lower,
        "odds_ratio_ci_upper": or_ci_upper,
    }


@router.post("/regression")
def linear_regression(req: RegressionRequest):
    X = np.array(req.x)
    y = np.array(req.y)
    if len(X) != len(y):
        raise HTTPException(400, "X and y must have the same number of observations.")
    if len(y) < 3:
        raise HTTPException(400, "Need at least 3 observations.")

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "coefficients": [float(c) for c in model.params],
        "std_errors": [float(s) for s in model.bse],
        "t_values": [float(t) for t in model.tvalues],
        "p_values": [float(p) for p in model.pvalues],
        "confidence_intervals": model.conf_int().tolist(),
        "residual_std_error": float(np.sqrt(model.mse_resid)),
        "n_observations": int(model.nobs),
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }


@router.post("/power-analysis")
def power_analysis(req: PowerAnalysisRequest):
    if req.test_type == "t-test":
        analysis = TTestIndPower()
        n = analysis.solve_power(
            effect_size=req.effect_size,
            alpha=req.alpha,
            power=req.power,
            alternative="two-sided",
        )
        return {
            "required_n_per_group": int(np.ceil(n)),
            "total_n": int(np.ceil(n)) * 2,
            "effect_size": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
            "test_type": "independent t-test",
        }
    elif req.test_type == "anova":
        analysis = FTestAnovaPower()
        n = analysis.solve_power(
            effect_size=req.effect_size,
            alpha=req.alpha,
            power=req.power,
            k_groups=req.n_groups,
        )
        # statsmodels' FTestAnovaPower solves for the TOTAL sample size
        per_group = int(np.ceil(n / req.n_groups))
        return {
            "required_n_per_group": per_group,
            "total_n": per_group * req.n_groups,
            "effect_size": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
            "test_type": f"one-way ANOVA ({req.n_groups} groups)",
        }
    else:
        raise HTTPException(400, f"Unknown test_type: {req.test_type}")


@router.post("/descriptive")
def descriptive_stats(req: DescriptiveRequest):
    data = np.array(req.data)
    if len(data) == 0:
        raise HTTPException(400, "Data array cannot be empty.")

    _, normality_p = sp_stats.shapiro(data) if len(data) >= 3 else (0, None)

    # 95% CI for mean
    if len(data) > 1:
        sem = float(sp_stats.sem(data))
        ci_low, ci_high = t_dist.interval(0.95, len(data) - 1, loc=float(data.mean()), scale=sem)
        ci_lower = float(ci_low)
        ci_upper = float(ci_high)
    else:
        sem = 0.0
        ci_lower = float(data.mean())
        ci_upper = float(data.mean())

    return {
        "n": len(data),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "variance": float(data.var(ddof=1)) if len(data) > 1 else 0.0,
        "min": float(data.min()),
        "max": float(data.max()),
        "range": float(data.max() - data.min()),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        "skewness": float(sp_stats.skew(data)),
        "kurtosis": float(sp_stats.kurtosis(data)),
        "sem": sem,
        "shapiro_p": float(normality_p) if normality_p is not None else None,
        "normal_distribution": bool(normality_p > 0.05) if normality_p is not None else None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
