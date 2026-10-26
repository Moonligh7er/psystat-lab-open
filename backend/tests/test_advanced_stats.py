"""
Advanced Stats (/api/advanced/*, the advanced tools in Stats Studio) against R 4.5.2.

R code for the references:
    # mixed models (lme4 2.0.6, REML) on mixed_data() below written to CSV
    lmer(y ~ day + (1|subject), d); lmer(y ~ day + (day|subject), d)
    wilcox.test(g1, g2, exact=FALSE, correct=TRUE); wilcox.test(g1, g2, paired=TRUE)
    kruskal.test(list(g1, g2, g3)); friedman.test(cbind(g1, g2, g3))
    cor.test(g1, g3, method="spearman", exact=FALSE); cor.test(g1, g3, method="kendall", exact=FALSE)
    fit <- lm(y ~ x1 + x2); car::vif(fit); cooks.distance(fit)
    afex::aov_ez("id", "y", d, within="cond")   # RM-ANOVA, GG/HF, Mauchly
"""

import numpy as np
import pytest
import statsmodels.api as sm

from app.routers import advanced_stats as A


def mixed_data():
    def lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) % 2 ** 31
            yield x / 2 ** 31 * 2 - 1
    g = lcg(7)
    subj, day, y = [], [], []
    for s in range(12):
        u, v = 25 * next(g), 4 * next(g)
        for d in range(8):
            subj.append(f"s{s + 1}")
            day.append(d)
            y.append(round(250 + 10 * d + u + v * d + 20 * next(g), 3))
    return subj, day, y


G1 = [12, 15, 11, 19, 14, 16, 13, 18, 17, 10]
G2 = [9, 8, 12, 7, 11, 10, 6, 13, 9, 8]
G3 = [14, 18, 16, 20, 15, 19, 17, 21, 16, 18]
RREF = "R 4.5.2"


def check(record, method, endpoint, stat, observed, expected, rel=None, abs_=None, dataset=""):
    if rel is not None:
        assert observed == pytest.approx(expected, rel=rel), f"{method} {stat}"
    else:
        assert observed == pytest.approx(expected, abs=abs_), f"{method} {stat}"
    record(method=method, endpoint=endpoint, statistic=stat, observed=float(observed), expected=float(expected),
           reference=RREF, reference_kind="r", dataset=dataset, tolerance_rel=rel, tolerance_abs=abs_)


@pytest.mark.validation
def test_mixed_model_matches_lme4(record_validation):
    subj, day, y = mixed_data()
    e, ds = "/api/advanced/mixed-model", "deterministic 12 subjects x 8 days (LCG)"
    ri = A.mixed_effects_model(A.MixedModelRequest(dependent=y, fixed_effects=[day], groups=subj, fixed_labels=["day"]))
    check(record_validation, "Linear mixed model, random intercept (REML)", e, "slope", ri["fixed_effects"]["day"]["coefficient"], 9.95268948, rel=1e-6, dataset=ds)
    check(record_validation, "Linear mixed model, random intercept (REML)", e, "slope SE", ri["fixed_effects"]["day"]["std_error"], 0.58690718, rel=1e-4, dataset=ds)
    check(record_validation, "Linear mixed model, random intercept (REML)", e, "intercept variance", ri["random_effects_variance"]["Intercept (group)"], 70.352547, rel=1e-3, dataset=ds)
    check(record_validation, "Linear mixed model, random intercept (REML)", e, "REML log-likelihood", ri["log_likelihood"], -389.090810, abs_=1e-3, dataset=ds)
    assert ri["icc"] == pytest.approx(70.352547 / (70.352547 + 173.607856), rel=1e-3)
    rs = A.mixed_effects_model(A.MixedModelRequest(dependent=y, fixed_effects=[day], groups=subj, fixed_labels=["day"], random_slopes=True))
    check(record_validation, "Linear mixed model, random slope (REML)", e, "slope SE", rs["fixed_effects"]["day"]["std_error"], 0.73931970, rel=1e-3, dataset=ds)
    check(record_validation, "Linear mixed model, random slope (REML)", e, "REML log-likelihood", rs["log_likelihood"], -386.951256, abs_=1e-3, dataset=ds)
    assert rs["random_effects_variance"]["day (group)"] == pytest.approx(2.796175, rel=1e-2)
    assert rs["n_parameters"] == 6  # 2 fixed + 3 (co)variances + residual


@pytest.mark.validation
def test_nonparametric_tests_match_r(record_validation):
    NP = lambda **k: A.non_parametric_tests(A.NonParametricRequest(**k))
    e, ds = "/api/advanced/non-parametric", "two 10-value groups (G1, G2, G3)"
    mw = NP(test="mann-whitney", group1=G1, group2=G2)
    check(record_validation, "Mann-Whitney U (Stats Studio)", e, "p", mw["p_value"], 0.0016653854, rel=1e-6, dataset=ds)
    assert mw["U_statistic"] == 92 and mw["rank_biserial_r"] == pytest.approx(2 * 92 / 100 - 1)
    w = NP(test="wilcoxon", group1=G1, group2=G2)
    assert w["W_statistic"] == 54  # R's V
    assert w["p_value"] == pytest.approx(2 / 512, rel=1e-9)  # exact: only W+ in {54, 55} or {0, 1}
    assert w["rank_biserial_r"] == pytest.approx((54 - 1) / 55)
    kw = NP(test="kruskal", groups=[G1, G2, G3])
    check(record_validation, "Kruskal-Wallis H", e, "H", kw["H_statistic"], 18.9377708287, rel=1e-9, dataset=ds)
    assert kw["epsilon_squared"] == pytest.approx(18.9377708287 / 29)
    fr = NP(test="friedman", groups=[G1, G2, G3])
    check(record_validation, "Friedman test", e, "chi-square", fr["chi2_statistic"], 16.2, rel=1e-9, dataset=ds)
    sp = NP(test="spearman", group1=G1, group2=G3)
    check(record_validation, "Spearman rank correlation (Stats Studio)", e, "p", sp["p_value"], 0.0832476271, rel=1e-6, dataset=ds)
    kt = NP(test="kendall", group1=G1, group2=G3)
    check(record_validation, "Kendall tau-b", e, "tau", kt["statistic"], 0.4319297483, rel=1e-9, dataset=ds)


@pytest.mark.validation
def test_multiple_regression_matches_r(record_validation):
    x1 = list(range(1, 13))
    x2 = [3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 6]
    y = [2.1, 4.0, 5.8, 8.2, 9.8, 12.1, 14.0, 15.9, 17.5, 20.3, 21.8, 24.4]
    r = A.multiple_regression_diagnostics(A.MultipleRegressionRequest(predictors=[x1, x2], outcome=y))
    e, ds = "/api/advanced/multiple-regression", "12 observations, 2 predictors"
    check(record_validation, "Multiple regression with diagnostics", e, "b1", r["coefficients"][1]["coefficient"], 2.0627619048, rel=1e-9, dataset=ds)
    check(record_validation, "Multiple regression with diagnostics", e, "VIF", r["vif"]["X1"], 2.5603809524, rel=1e-9, dataset=ds)
    check(record_validation, "Multiple regression with diagnostics", e, "Breusch-Pagan (Koenker)", r["breusch_pagan_statistic"], 0.7579401810, rel=1e-8, dataset=ds)
    assert r["cooks_distance"][0] == pytest.approx(0.43069949, rel=1e-6)
    assert r["f_df"] == [2, 9]


@pytest.mark.validation
def test_repeated_measures_matches_afex(record_validation):
    Y = np.array([[45, 50, 55, 70], [42, 42, 45, 60], [36, 41, 43, 62], [39, 35, 40, 53],
                  [51, 55, 59, 74], [44, 49, 56, 69], [40, 42, 46, 59], [43, 47, 52, 66]], float)
    r = A.repeated_measures_anova(A.RepeatedMeasuresRequest(measurements=Y.T.tolist()))
    e, ds = "/api/advanced/repeated-measures", "8 subjects x 4 conditions"
    check(record_validation, "Repeated-measures ANOVA", e, "F", r["statistic"], 182.19359534, rel=1e-8, dataset=ds)
    check(record_validation, "Repeated-measures ANOVA", e, "Greenhouse-Geisser p", r["p_greenhouse_geisser"], 3.772106e-08, rel=1e-5, dataset=ds)
    check(record_validation, "Repeated-measures ANOVA", e, "Huynh-Feldt epsilon", r["epsilon_huynh_feldt"], 0.558162, abs_=1e-6, dataset=ds)
    check(record_validation, "Mauchly sphericity test", e, "p", r["mauchly"]["p"], 0.0134, abs_=5e-5, dataset=ds)
    assert r["nonparametric"]["statistic"] == pytest.approx(22.82278, abs=1e-5)


def test_mediation_bootstrap_matches_loop():
    rng = np.random.default_rng(3)
    x = rng.normal(size=60); m = 0.5 * x + rng.normal(size=60); y = 0.4 * m + 0.2 * x + rng.normal(size=60)
    r = A.mediation_analysis(A.MediationtRequest(x=x.tolist(), m=m.tolist(), y=y.tolist(), n_bootstrap=2000, seed=11))
    a = sm.OLS(m, sm.add_constant(x)).fit().params[1]
    b = sm.OLS(y, sm.add_constant(np.column_stack([x, m]))).fit().params[2]
    assert r["indirect_effect_ab"] == pytest.approx(a * b)
    # Recompute the bootstrap with explicit OLS fits on the same resamples
    idx = np.random.default_rng(11).integers(0, 60, size=(2000, 60))
    ab = []
    for row in idx:
        aa = sm.OLS(m[row], sm.add_constant(x[row])).fit().params[1]
        bb = sm.OLS(y[row], sm.add_constant(np.column_stack([x[row], m[row]]))).fit().params[2]
        ab.append(aa * bb)
    lo, hi = np.percentile(ab, [2.5, 97.5])
    assert r["indirect_ci_lower"] == pytest.approx(lo, rel=1e-8) and r["indirect_ci_upper"] == pytest.approx(hi, rel=1e-8)


def test_time_series_forecast_now_works():
    vals = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140]
    r = A.time_series_analysis(A.TimeSeriesRequest(values=vals, forecast_steps=5))
    assert "error" not in r["arima"] and len(r["arima"]["forecast"]) == 5
    assert len(r["arima"]["forecast_ci_lower"]) == 5


def test_validation_errors(client):
    assert client.post("/api/advanced/meta-analysis", json={"effect_sizes": [0.1, 0.2], "standard_errors": [0.1, 0.1], "study_labels": ["a"]}).status_code == 422
    assert client.post("/api/advanced/repeated-measures", json={"measurements": [[1, 2, 3], [2, 3, 4]], "condition_labels": ["a"]}).status_code == 400
    assert client.post("/api/advanced/multiple-regression", json={"predictors": [[1, 2, 3, 4, 5]], "outcome": [1, 2, 3]}).status_code == 400
    assert client.post("/api/advanced/non-parametric", json={"test": "wilcoxon", "group1": [1, 2, 3], "group2": [1, 2, 3]}).status_code == 400
    assert client.post("/api/advanced/correlation-matrix", json={"variables": [[1, 2, 3], [3, 2, 1]], "method": "pearsn"}).status_code == 400
