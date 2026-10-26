"""Extended Stats — logistic regression, ROC-AUC, Bland-Altman, effect-size
benchmarks, dual frequentist/Bayesian framework, and sensitivity analysis."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.special import gammaln
from scipy.integrate import quad
from typing import Optional

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class LogisticRegressionRequest(BaseModel):
    dependent: list[int]
    predictors: list[list[float]]
    predictor_names: Optional[list[str]] = None


class RocAucRequest(BaseModel):
    true_labels: list[int]
    scores: list[float]
    thresholds_count: int = 100


class BlandAltmanRequest(BaseModel):
    method1: list[float]
    method2: list[float]


class EffectSizeBenchmarkRequest(BaseModel):
    effect_size: float
    effect_type: str  # "cohens_d", "eta_squared", "r", "odds_ratio"
    domain: str  # "clinical_psychology", "educational", "social_psychology",
    # "neuroscience", "organizational", "medical", "general"


class DualFrameworkRequest(BaseModel):
    test_type: str  # "t-test" or "correlation"
    data1: list[float]
    data2: Optional[list[float]] = None


class SensitivityAnalysisRequest(BaseModel):
    data1: list[float]
    data2: list[float]
    test_type: str = "two-sample"


# ── 1. Logistic Regression ───────────────────────────────────────────

@router.post("/logistic-regression")
def logistic_regression(req: LogisticRegressionRequest):
    """Binary logistic regression via statsmodels Logit."""
    import statsmodels.api as sm

    y = np.array(req.dependent, dtype=float)
    X = np.array(req.predictors, dtype=float)

    if len(y) < 2:
        raise HTTPException(400, "Need at least 2 observations.")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != len(y):
        raise HTTPException(400, "Number of rows in predictors must match length of dependent.")
    if not set(np.unique(y)) <= {0.0, 1.0} or len(np.unique(y)) < 2:
        raise HTTPException(400, "Dependent variable must be coded 0/1 and contain both values.")
    if not np.all(np.isfinite(X)):
        raise HTTPException(400, "Predictors must be finite numbers.")

    n_predictors = X.shape[1]
    names = req.predictor_names or [f"X{i+1}" for i in range(n_predictors)]
    if len(names) != n_predictors:
        raise HTTPException(400, "Length of predictor_names must match number of predictor columns.")

    # Add constant (intercept)
    X_const = sm.add_constant(X, has_constant="add")

    try:
        model = sm.Logit(y, X_const)
        result = model.fit(disp=0, maxiter=100)
    except np.linalg.LinAlgError:
        raise HTTPException(422, "Singular matrix — predictors may be perfectly collinear.")
    except Exception as e:
        raise HTTPException(422, f"Model fitting failed: {str(e)}")

    coefficients = result.params.tolist()
    std_errors = result.bse.tolist()
    z_values = result.tvalues.tolist()
    p_values = result.pvalues.tolist()
    conf_int = result.conf_int(alpha=0.05)

    # Odds ratios and their CIs
    odds_ratios = np.exp(result.params).tolist()
    or_ci_lower = np.exp(conf_int[:, 0]).tolist()
    or_ci_upper = np.exp(conf_int[:, 1]).tolist()

    # Pseudo R-squared (McFadden)
    pseudo_r2 = float(result.prsquared)

    # AIC / BIC
    aic = float(result.aic)
    bic = float(result.bic)

    # Classification accuracy and confusion matrix
    predicted_probs = result.predict(X_const)
    predicted_classes = (predicted_probs >= 0.5).astype(int)
    accuracy = float(np.mean(predicted_classes == y))

    tp = int(np.sum((predicted_classes == 1) & (y == 1)))
    tn = int(np.sum((predicted_classes == 0) & (y == 0)))
    fp = int(np.sum((predicted_classes == 1) & (y == 0)))
    fn = int(np.sum((predicted_classes == 0) & (y == 1)))

    param_names = ["Intercept"] + list(names)
    lr_p = float(result.llr_pvalue)

    return {
        "coefficient_table": [
            {"name": param_names[i], "coefficient": coefficients[i], "std_error": std_errors[i], "z_value": z_values[i],
             "p_value": p_values[i], "odds_ratio": odds_ratios[i], "or_ci_lower": or_ci_lower[i], "or_ci_upper": or_ci_upper[i]}
            for i in range(len(param_names))
        ],
        "likelihood_ratio_test": {"chi_square": float(result.llr), "df": int(result.df_model), "p": lr_p},
        "log_likelihood": float(result.llf),
        "classification_note": "In-sample classification at a 0.5 probability cut-off (optimistic).",
        "coefficients": {param_names[i]: coefficients[i] for i in range(len(param_names))},
        "std_errors": {param_names[i]: std_errors[i] for i in range(len(param_names))},
        "z_values": {param_names[i]: z_values[i] for i in range(len(param_names))},
        "p_values": {param_names[i]: p_values[i] for i in range(len(param_names))},
        "odds_ratios": {param_names[i]: odds_ratios[i] for i in range(len(param_names))},
        "or_ci_lower": {param_names[i]: or_ci_lower[i] for i in range(len(param_names))},
        "or_ci_upper": {param_names[i]: or_ci_upper[i] for i in range(len(param_names))},
        "pseudo_r2_mcfadden": pseudo_r2,
        "aic": aic,
        "bic": bic,
        "classification_accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": int(len(y)),
    }


# ── 2. ROC-AUC ───────────────────────────────────────────────────────

@router.post("/roc-auc")
def roc_auc(req: RocAucRequest):
    """Empirical ROC curve, exact AUC (Mann–Whitney) with DeLong 95% CI, and Youden-optimal threshold."""
    y_true = np.array(req.true_labels, dtype=int)
    scores = np.array(req.scores, dtype=float)

    if len(y_true) != len(scores):
        raise HTTPException(400, "true_labels and scores must have the same length.")
    if len(y_true) < 2:
        raise HTTPException(400, "Need at least 2 observations.")
    if len(np.unique(y_true)) < 2:
        raise HTTPException(400, "true_labels must contain both 0 and 1.")

    if not set(np.unique(y_true)) <= {0, 1}:
        raise HTTPException(400, "true_labels must be coded 0/1.")
    if not np.all(np.isfinite(scores)):
        raise HTTPException(400, "Scores must be finite numbers.")
    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    pos, neg = scores[y_true == 1], scores[y_true == 0]

    # Exact empirical ROC: one point per distinct score (higher score = predicted positive)
    cuts = np.unique(scores)[::-1]
    roc_points = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    for c in cuts:
        roc_points.append({"fpr": float(np.mean(neg >= c)), "tpr": float(np.mean(pos >= c)), "threshold": float(c)})

    # AUC = P(score_pos > score_neg) + 0.5 P(tie)  (Mann-Whitney), equal to the trapezoid under the exact ROC
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = float(psi.mean())

    # DeLong et al. (1988) standard error (as pROC's default ci.auc)
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    s10 = np.var(v10, ddof=1) if n_pos > 1 else 0.0
    s01 = np.var(v01, ddof=1) if n_neg > 1 else 0.0
    se_auc = float(np.sqrt(s10 / n_pos + s01 / n_neg))
    z = sp_stats.norm.ppf(0.975)
    auc_ci_lower, auc_ci_upper = max(0.0, auc - z * se_auc), min(1.0, auc + z * se_auc)

    # Youden's J over the exact thresholds
    best = max(roc_points[1:], key=lambda pt: pt["tpr"] - pt["fpr"])
    optimal_threshold = best["threshold"]
    sensitivity_at_optimal = best["tpr"]
    specificity_at_optimal = 1.0 - best["fpr"]

    return {
        "auc": auc,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "auc_se": se_auc,
        "auc_ci_method": "DeLong (1988)",
        "auc_note": "AUC below 0.5 means higher scores go with the 0 class; it is reported as is, not flipped." if auc < 0.5 else None,
        "youden_j": float(best["tpr"] - best["fpr"]),
        "optimal_threshold": optimal_threshold,
        "sensitivity_at_optimal": sensitivity_at_optimal,
        "specificity_at_optimal": specificity_at_optimal,
        "roc_points": roc_points,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


# ── 3. Bland-Altman ──────────────────────────────────────────────────

@router.post("/bland-altman")
def bland_altman(req: BlandAltmanRequest):
    """Bland-Altman method comparison analysis."""
    import statsmodels.api as sm

    m1 = np.array(req.method1, dtype=float)
    m2 = np.array(req.method2, dtype=float)

    if len(m1) != len(m2):
        raise HTTPException(400, "method1 and method2 must have the same length.")
    n = len(m1)
    if n < 3:
        raise HTTPException(400, "Need at least 3 paired observations.")

    diffs = m1 - m2
    means = (m1 + m2) / 2.0

    bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    # 95% limits of agreement
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff

    # 95% CI for bias (t-based)
    se_bias = sd_diff / np.sqrt(n)
    t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
    bias_ci_lower = bias - t_crit * se_bias
    bias_ci_upper = bias + t_crit * se_bias

    # 95% CI for limits of agreement
    # SE of LOA = sqrt(3 * s^2 / n)  (Bland & Altman, 1999)
    se_loa = np.sqrt(3.0 * sd_diff**2 / n)
    loa_upper_ci_lower = loa_upper - t_crit * se_loa
    loa_upper_ci_upper = loa_upper + t_crit * se_loa
    loa_lower_ci_lower = loa_lower - t_crit * se_loa
    loa_lower_ci_upper = loa_lower + t_crit * se_loa

    # Proportional bias test: regress difference on mean
    X_mean = sm.add_constant(means)
    try:
        ols_result = sm.OLS(diffs, X_mean).fit()
        prop_bias_slope = float(ols_result.params[1])
        prop_bias_p = float(ols_result.pvalues[1])
    except Exception:
        prop_bias_slope = None
        prop_bias_p = None

    points = [{"mean": float(means[i]), "diff": float(diffs[i])} for i in range(n)]

    return {
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_upper": float(loa_upper),
        "loa_lower": float(loa_lower),
        "bias_ci_lower": float(bias_ci_lower),
        "bias_ci_upper": float(bias_ci_upper),
        "loa_upper_ci": {"lower": float(loa_upper_ci_lower), "upper": float(loa_upper_ci_upper)},
        "loa_lower_ci": {"lower": float(loa_lower_ci_lower), "upper": float(loa_lower_ci_upper)},
        "proportional_bias_slope": prop_bias_slope,
        "proportional_bias_p": prop_bias_p,
        "n": n,
        "points": points,
    }


# ── 4. Effect-Size Benchmarks ────────────────────────────────────────

# Published benchmark tables per domain
_BENCHMARKS = {
    "cohens_d": {
        "general":            {"small": 0.2, "medium": 0.5, "large": 0.8},
        "clinical_psychology": {"small": 0.2, "medium": 0.5, "large": 0.8},
        "educational":        {"small": 0.1, "medium": 0.3, "large": 0.5},
        "social_psychology":  {"small": 0.1, "medium": 0.2, "large": 0.4},
        "neuroscience":       {"small": 0.1, "medium": 0.2, "large": 0.3},
        "organizational":     {"small": 0.2, "medium": 0.5, "large": 0.8},
        "medical":            {"small": 0.1, "medium": 0.3, "large": 0.5},
    },
    "eta_squared": {
        "general":            {"small": 0.01, "medium": 0.06, "large": 0.14},
        "clinical_psychology": {"small": 0.01, "medium": 0.06, "large": 0.14},
        "educational":        {"small": 0.005, "medium": 0.03, "large": 0.08},
        "social_psychology":  {"small": 0.005, "medium": 0.02, "large": 0.06},
        "neuroscience":       {"small": 0.005, "medium": 0.02, "large": 0.05},
        "organizational":     {"small": 0.01, "medium": 0.06, "large": 0.14},
        "medical":            {"small": 0.005, "medium": 0.03, "large": 0.08},
    },
    "r": {
        "general":            {"small": 0.10, "medium": 0.30, "large": 0.50},
        "clinical_psychology": {"small": 0.10, "medium": 0.30, "large": 0.50},
        "educational":        {"small": 0.05, "medium": 0.15, "large": 0.30},
        "social_psychology":  {"small": 0.05, "medium": 0.10, "large": 0.25},
        "neuroscience":       {"small": 0.05, "medium": 0.10, "large": 0.20},
        "organizational":     {"small": 0.10, "medium": 0.25, "large": 0.45},
        "medical":            {"small": 0.05, "medium": 0.15, "large": 0.30},
    },
    "odds_ratio": {
        "general":            {"small": 1.5, "medium": 2.5, "large": 4.3},
        "clinical_psychology": {"small": 1.5, "medium": 2.5, "large": 4.3},
        "educational":        {"small": 1.2, "medium": 1.8, "large": 2.8},
        "social_psychology":  {"small": 1.2, "medium": 1.5, "large": 2.2},
        "neuroscience":       {"small": 1.2, "medium": 1.5, "large": 2.0},
        "organizational":     {"small": 1.5, "medium": 2.5, "large": 4.3},
        "medical":            {"small": 1.2, "medium": 1.8, "large": 3.0},
    },
}

_DOMAIN_DESCRIPTIONS = {
    "clinical_psychology": "clinical psychology (therapy/intervention meta-analyses)",
    "educational": "educational research",
    "social_psychology": "social psychology (including replication studies)",
    "neuroscience": "neuroscience/neuroimaging",
    "organizational": "organizational/industrial psychology",
    "medical": "medical/epidemiological research",
    "general": "general behavioral science",
}


@router.post("/effect-size-benchmarks")
def effect_size_benchmarks(req: EffectSizeBenchmarkRequest):
    """Domain-specific effect-size interpretation with published benchmarks."""
    et = req.effect_type.lower()
    domain = req.domain.lower()

    if et not in _BENCHMARKS:
        raise HTTPException(400, f"Unknown effect_type '{et}'. Use: {list(_BENCHMARKS.keys())}")
    if domain not in _BENCHMARKS[et]:
        raise HTTPException(400, f"Unknown domain '{domain}'. Use: {list(_BENCHMARKS[et].keys())}")

    bench = _BENCHMARKS[et][domain]
    d = req.effect_size
    abs_d = abs(d)

    # Generic Cohen label (always based on general thresholds)
    gen = _BENCHMARKS[et]["general"]
    if abs_d < gen["small"]:
        generic_label = "negligible"
    elif abs_d < gen["medium"]:
        generic_label = "small"
    elif abs_d < gen["large"]:
        generic_label = "medium"
    else:
        generic_label = "large"

    # Domain-specific label
    if et == "odds_ratio":
        cmp = d if d >= 1 else 1 / d  # always compare magnitude away from 1
        if cmp < bench["small"]:
            domain_label = "negligible"
        elif cmp < bench["medium"]:
            domain_label = "small"
        elif cmp < bench["large"]:
            domain_label = "medium"
        else:
            domain_label = "large"
    else:
        if abs_d < bench["small"]:
            domain_label = "negligible"
        elif abs_d < bench["medium"]:
            domain_label = "small"
        elif abs_d < bench["large"]:
            domain_label = "medium"
        else:
            domain_label = "large"

    domain_desc = _DOMAIN_DESCRIPTIONS.get(domain, domain)

    # Domain interpretation sentence
    if et == "odds_ratio":
        domain_interpretation = (
            f"An odds ratio of {d:.2f} is considered {domain_label} in {domain_desc}. "
            f"Domain benchmarks: small={bench['small']}, medium={bench['medium']}, large={bench['large']}."
        )
    else:
        domain_interpretation = (
            f"A {et.replace('_', ' ')} of {d:.3f} is considered {domain_label} in {domain_desc}. "
            f"Domain benchmarks: small={bench['small']}, medium={bench['medium']}, large={bench['large']}."
        )

    # NNT-equivalent framing for medical domain
    if domain == "medical" and et == "cohens_d" and abs_d > 0:
        # Approximate NNT from Cohen's d using Furukawa & Leucht (2011) formula
        cer = 0.2  # assumed control event rate
        z_d = abs_d / np.sqrt(2)
        nnt = max(1, round(1 / (sp_stats.norm.cdf(abs_d - sp_stats.norm.ppf(1 - cer)) - cer)))
        domain_interpretation += f" Approximate NNT (assuming 20% base rate): {nnt}."

    # Percentile rank: approximate using a half-normal CDF assumption
    # (published effects roughly follow a half-normal with median ~0.4 for d)
    if et == "odds_ratio":
        # Convert OR to d-scale approximation: d = ln(OR) * sqrt(3) / pi
        d_equiv = abs(np.log(max(d, 0.01))) * np.sqrt(3) / np.pi
    elif et == "eta_squared":
        d_equiv = 2 * np.sqrt(abs_d / (1 - min(abs_d, 0.99)))
    elif et == "r":
        d_equiv = 2 * abs_d / np.sqrt(max(1 - abs_d**2, 0.01))
    else:
        d_equiv = abs_d

    # Half-normal with sigma calibrated so median ~ 0.4
    sigma = 0.4 / 0.6745  # ~0.593
    percentile_rank = float(sp_stats.halfnorm.cdf(d_equiv, scale=sigma) * 100)

    # CLES (Common Language Effect Size) — P(X_treatment > X_control)
    # Only meaningful for d-like metrics
    cles = None
    u3 = None
    if et == "cohens_d":
        cles = float(sp_stats.norm.cdf(d / np.sqrt(2)) * 100)
        u3 = float(sp_stats.norm.cdf(d) * 100)
    elif et == "r":
        # Convert r to d, then compute CLES
        d_from_r = 2 * d / np.sqrt(max(1 - d**2, 0.01))
        cles = float(sp_stats.norm.cdf(d_from_r / np.sqrt(2)) * 100)
        u3 = float(sp_stats.norm.cdf(d_from_r) * 100)
    elif et == "eta_squared":
        if abs_d > 0 and abs_d < 1:
            d_from_eta = 2 * np.sqrt(abs_d / (1 - abs_d))
            cles = float(sp_stats.norm.cdf(d_from_eta / np.sqrt(2)) * 100)
            u3 = float(sp_stats.norm.cdf(d_from_eta) * 100)
    elif et == "odds_ratio":
        d_from_or = np.log(max(d, 0.01)) * np.sqrt(3) / np.pi
        cles = float(sp_stats.norm.cdf(d_from_or / np.sqrt(2)) * 100)
        u3 = float(sp_stats.norm.cdf(d_from_or) * 100)

    return {
        "effect_size": d,
        "effect_type": et,
        "domain": domain,
        "generic_label": generic_label,
        "domain_label": domain_label,
        "domain_interpretation": domain_interpretation,
        "percentile_rank": round(percentile_rank, 1),
        "cles": round(cles, 1) if cles is not None else None,
        "u3": round(u3, 1) if u3 is not None else None,
    }


# ── 5. Dual Framework (Frequentist + Bayesian) ───────────────────────

@router.post("/dual-framework")
def dual_framework(req: DualFrameworkRequest):
    """Run both frequentist and Bayesian analysis on the same data."""
    test = req.test_type.lower()

    if test == "t-test":
        return _dual_ttest(req)
    elif test == "correlation":
        return _dual_correlation(req)
    else:
        raise HTTPException(400, "test_type must be 't-test' or 'correlation'.")


def _dual_ttest(req: DualFrameworkRequest):
    d1 = np.array(req.data1, dtype=float)
    d2 = np.array(req.data2, dtype=float) if req.data2 is not None else None
    if d2 is None or len(d2) < 2:
        raise HTTPException(400, "data2 is required with at least 2 values for a t-test.")
    if len(d1) < 2:
        raise HTTPException(400, "data1 must have at least 2 values.")

    # ── Frequentist ──
    t_stat, p_value = sp_stats.ttest_ind(d1, d2)
    n1, n2 = len(d1), len(d2)
    df = n1 + n2 - 2
    pooled_std = np.sqrt(((n1 - 1) * d1.std(ddof=1)**2 + (n2 - 1) * d2.std(ddof=1)**2) / df)
    cohens_d = float((d1.mean() - d2.mean()) / pooled_std) if pooled_std > 0 else 0.0
    se_diff = float(pooled_std * np.sqrt(1/n1 + 1/n2))
    mean_diff = float(d1.mean() - d2.mean())
    ci_low, ci_high = sp_stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)

    freq_result = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "mean_difference": mean_diff,
        "ci_95": [float(ci_low), float(ci_high)],
        "df": df,
        "significant": bool(p_value < 0.05),
    }

    # ── Bayesian (JZS BF) ──
    n_eff = (n1 * n2) / (n1 + n2)
    bf10 = _jzs_bf(float(t_stat), n_eff, df)

    bayes_result = {
        "bf10": float(bf10),
        "bf01": float(1 / bf10) if bf10 > 0 else float("inf"),
        "interpretation": _interpret_bf(bf10),
        "evidence_for": "H1 (effect exists)" if bf10 > 1 else "H0 (no effect)",
    }

    agreement = _assess_agreement(p_value < 0.05, bf10)

    return {
        "test_type": "t-test",
        "frequentist": freq_result,
        "bayesian": bayes_result,
        "agreement": agreement,
    }


def _dual_correlation(req: DualFrameworkRequest):
    d1 = np.array(req.data1, dtype=float)
    d2 = np.array(req.data2, dtype=float) if req.data2 is not None else None
    if d2 is None:
        raise HTTPException(400, "data2 is required for correlation.")
    if len(d1) != len(d2):
        raise HTTPException(400, "data1 and data2 must have the same length.")
    n = len(d1)
    if n < 3:
        raise HTTPException(400, "Need at least 3 data points.")

    # ── Frequentist ──
    r, p_value = sp_stats.pearsonr(d1, d2)
    # Fisher z CI
    z_r = np.arctanh(r)
    se_z = 1 / np.sqrt(n - 3)
    z_low = z_r - 1.96 * se_z
    z_high = z_r + 1.96 * se_z
    ci_low = float(np.tanh(z_low))
    ci_high = float(np.tanh(z_high))

    freq_result = {
        "r": float(r),
        "p_value": float(p_value),
        "ci_95": [ci_low, ci_high],
        "n": n,
        "significant": bool(p_value < 0.05),
    }

    # ── Bayesian correlation BF ──
    bf10 = _bf_correlation(float(r), n)

    bayes_result = {
        "bf10": float(bf10),
        "bf01": float(1 / bf10) if bf10 > 0 else float("inf"),
        "interpretation": _interpret_bf(bf10),
        "evidence_for": "H1 (correlation exists)" if bf10 > 1 else "H0 (no correlation)",
    }

    agreement = _assess_agreement(p_value < 0.05, bf10)

    return {
        "test_type": "correlation",
        "frequentist": freq_result,
        "bayesian": bayes_result,
        "agreement": agreement,
    }


def _assess_agreement(freq_sig: bool, bf10: float) -> str:
    """Characterize agreement between frequentist and Bayesian results."""
    bayes_supports_h1 = bf10 > 3
    bayes_supports_h0 = bf10 < 1 / 3
    bayes_inconclusive = not bayes_supports_h1 and not bayes_supports_h0

    if freq_sig and bayes_supports_h1:
        return "Both support an effect (p < .05 and BF10 > 3)"
    elif not freq_sig and bayes_supports_h0:
        return "Both point away from an effect (p ≥ .05 and BF10 < 1/3)"
    elif freq_sig and bayes_supports_h0:
        return "Divergent — frequentist significant but Bayesian evidence favors H0"
    elif not freq_sig and bayes_supports_h1:
        return "Divergent — frequentist non-significant but Bayesian evidence favors H1"
    elif freq_sig and bayes_inconclusive:
        return "Divergent — frequentist significant but weak Bayesian evidence"
    elif not freq_sig and bayes_inconclusive:
        return "Neither is decisive — frequentist non-significant and Bayesian evidence inconclusive"
    else:
        return "Inconclusive"


# ── 6. Sensitivity Analysis ──────────────────────────────────────────

@router.post("/sensitivity-analysis")
def sensitivity_analysis(req: SensitivityAnalysisRequest):
    """Run the same comparison via 4 methods to check robustness."""
    d1 = np.array(req.data1, dtype=float)
    d2 = np.array(req.data2, dtype=float)

    if len(d1) < 2 or len(d2) < 2:
        raise HTTPException(400, "Each group must have at least 2 observations.")

    alpha = 0.05
    results = {}

    # 1. Parametric independent t-test
    t_stat, t_p = sp_stats.ttest_ind(d1, d2)
    results["parametric_ttest"] = {
        "statistic": float(t_stat),
        "p_value": float(t_p),
        "significant": bool(t_p < alpha),
        "direction": "group1 > group2" if t_stat > 0 else "group1 < group2",
    }

    # 2. Mann-Whitney U
    u_stat, u_p = sp_stats.mannwhitneyu(d1, d2, alternative="two-sided")
    # Direction from rank-biserial
    n1, n2 = len(d1), len(d2)
    rank_biserial = 2 * float(u_stat) / (n1 * n2) - 1  # positive when group 1 tends to be larger
    results["mann_whitney_u"] = {
        "statistic": float(u_stat),
        "p_value": float(u_p),
        "significant": bool(u_p < alpha),
        "rank_biserial": float(rank_biserial),
        "direction": "group1 > group2" if rank_biserial > 0 else "group1 < group2",
    }

    # 3. Permutation test (10000 permutations)
    rng = np.random.default_rng(42)
    combined = np.concatenate([d1, d2])
    obs_diff = float(d1.mean() - d2.mean())
    n_perms = 10000
    perm_diffs = np.empty(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(combined)
        perm_diffs[i] = perm[:n1].mean() - perm[n1:].mean()
    # (count + 1) / (B + 1): a valid p-value that is never exactly 0 (Phipson & Smyth, 2010)
    perm_p = float((np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (n_perms + 1))
    results["permutation_test"] = {
        "observed_diff": obs_diff,
        "p_value": perm_p,
        "significant": bool(perm_p < alpha),
        "n_permutations": n_perms,
        "direction": "group1 > group2" if obs_diff > 0 else "group1 < group2",
    }

    # 4. Bootstrap CI for mean difference (10000 resamples)
    n_boot = 10000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        b1 = rng.choice(d1, size=n1, replace=True)
        b2 = rng.choice(d2, size=n2, replace=True)
        boot_diffs[i] = b1.mean() - b2.mean()
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    # Significant if CI doesn't contain 0
    boot_sig = bool(ci_low > 0 or ci_high < 0)
    results["bootstrap_ci"] = {
        "mean_difference": obs_diff,
        "ci_95": [ci_low, ci_high],
        "significant": boot_sig,
        "n_resamples": n_boot,
        "direction": "group1 > group2" if obs_diff > 0 else "group1 < group2",
    }

    # Aggregate: all agree?
    sig_list = [
        results["parametric_ttest"]["significant"],
        results["mann_whitney_u"]["significant"],
        results["permutation_test"]["significant"],
        results["bootstrap_ci"]["significant"],
    ]
    dir_list = [
        results["parametric_ttest"]["direction"],
        results["mann_whitney_u"]["direction"],
        results["permutation_test"]["direction"],
        results["bootstrap_ci"]["direction"],
    ]

    all_agree_sig = all(sig_list) or not any(sig_list)
    all_agree_dir = len(set(dir_list)) == 1
    robust = bool(all_agree_sig and all_agree_dir)

    n_sig = sum(sig_list)
    if robust and all(sig_list):
        summary = (f"All 4 methods agree: the difference is statistically significant "
                   f"({dir_list[0]}). The conclusion does not depend on which of these methods is used.")
    elif robust and not any(sig_list):
        summary = ("All 4 methods agree: the difference is not statistically significant. This is not evidence "
                   "that there is no difference; an equivalence test or Bayes factor can address that.")
    else:
        methods = ["parametric t-test", "Mann-Whitney U", "permutation test", "bootstrap CI"]
        sig_methods = [m for m, s in zip(methods, sig_list) if s]
        nonsig_methods = [m for m, s in zip(methods, sig_list) if not s]
        summary = (f"{n_sig}/4 methods found significance ({', '.join(sig_methods) if sig_methods else 'none'}). "
                   f"Non-significant: {', '.join(nonsig_methods) if nonsig_methods else 'none'}. "
                   f"The conclusion depends on the method; check outliers and skew before relying on it.")

    return {
        "results": results,
        "robust": robust,
        "summary": summary,
    }


# ── Shared helpers (JZS BF, correlation BF, BF interpretation) ───────

def _jzs_bf(t: float, n_eff: float, df: int, r: float = 0.707) -> float:
    """JZS Bayes Factor (Rouder et al., 2009). Shares the verified implementation
    in novel_dialectical (checked against pingouin to ~1e-9)."""
    from app.routers.novel_dialectical import jzs_bf10_t
    return jzs_bf10_t(float(t), float(n_eff), float(df), float(r))


def _bf_correlation(r: float, n: int) -> float:
    """Default correlation Bayes factor of Ly, Verhagen & Wagenmakers (2016), uniform prior on rho."""
    if abs(r) >= 1:
        return 1e300
    from app.routers.novel_dialectical import ly_bf10_r
    return ly_bf10_r(float(r), int(n), 1.0)


def _interpret_bf(bf: float) -> str:
    """Interpret Bayes Factor using Jeffreys' scale."""
    if bf > 100:
        return "Extreme evidence for H1"
    elif bf > 30:
        return "Very strong evidence for H1"
    elif bf > 10:
        return "Strong evidence for H1"
    elif bf > 3:
        return "Moderate evidence for H1"
    elif bf > 1:
        return "Anecdotal evidence for H1"
    elif bf == 1:
        return "No evidence either way"
    elif bf > 1/3:
        return "Anecdotal evidence for H0"
    elif bf > 1/10:
        return "Moderate evidence for H0"
    elif bf > 1/30:
        return "Strong evidence for H0"
    elif bf > 1/100:
        return "Very strong evidence for H0"
    else:
        return "Extreme evidence for H0"
