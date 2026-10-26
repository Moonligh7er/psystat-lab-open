"""
Regression tests for the Dual Report module (frequentist + Bayesian side by side).
Reference Bayes factors were computed with pingouin 0.7 (bayesfactor_ttest,
bayesfactor_pearson); the reference CIs are standard closed forms.
"""

import asyncio
import math

import numpy as np
import pytest
from scipy import stats

from app.routers import dual_report as R

run = asyncio.run


def _dual(test_type, data, **kw):
    return run(R.dual_test(R.DualTestRequest(test_type=test_type, data=data, **kw)))


X = [5.1, 4.8, 6.2, 5.9, 6.5, 5.3, 6.0, 5.7, 6.8, 5.5]
Y = [4.2, 4.9, 5.0, 4.4, 5.6, 4.1, 4.7, 5.2, 4.6, 4.3, 5.1, 4.8]


@pytest.mark.validation
@pytest.mark.parametrize("r", [0.5, 0.707, 1.414])
def test_two_sample_bf_uses_n_eff_and_matches_pingouin(r):
    out = _dual("t-test", {"a": X, "b": Y}, prior_scale=r)
    t = out.frequentist.statistic
    n1, n2 = len(X), len(Y)
    ref = R.D.jzs_bf10_t(t, n1 * n2 / (n1 + n2), n1 + n2 - 2, r)
    assert out.bayesian.bf10 == pytest.approx(ref, rel=1e-5)
    assert out.bayesian.prior_scale == r and "Cauchy" in out.bayesian.prior_description


@pytest.mark.validation
def test_two_sample_bf_pingouin_reference():
    # pingouin.bayesfactor_ttest(2.1, 12, 18, r=0.707) = 1.72415 (n_eff = 7.2, df = 28)
    assert R.D.jzs_bf10_t(2.1, 7.2, 28, 0.707) == pytest.approx(1.72415, rel=1e-4)


@pytest.mark.validation
def test_paired_bf_matches_one_sample_reference():
    # One-sample t = 2.1, n = 20, r = 0.707 -> pingouin 1.4047 (old code gave 1.218).
    rng = np.random.default_rng(3)
    d = rng.normal(size=20)
    d = (d - d.mean()) / d.std(ddof=1)
    d = d + 2.1 / math.sqrt(20)  # mean/sd*sqrt(n) = 2.1 exactly
    out = _dual("paired-t", {"a": d.tolist(), "b": [0.0] * 20}, prior_scale=0.707)
    assert out.frequentist.statistic == pytest.approx(2.1, abs=1e-6)
    assert out.bayesian.bf10 == pytest.approx(1.4047, rel=1e-4)


@pytest.mark.validation
@pytest.mark.parametrize("kappa", [1.0, 0.5])
def test_correlation_bf_uses_ly_and_kappa(kappa):
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    y = 0.3 * x + rng.normal(size=30)
    out = _dual("correlation", {"x": x.tolist(), "y": y.tolist()}, params={"kappa": kappa},
                prior_scale=99.0)  # prior_scale must not affect the correlation BF
    r = stats.pearsonr(x, y)[0]
    assert out.bayesian.bf10 == pytest.approx(R.D.ly_bf10_r(r, 30, kappa), rel=1e-5)
    assert out.bayesian.prior_scale == kappa
    assert "stretched Beta" in out.bayesian.prior_description
    assert "stretched Beta" in out.comparison.apa_text
    lo, hi = out.bayesian.credible_interval
    assert -1 < lo < out.bayesian.posterior_mean < hi < 1


@pytest.mark.validation
def test_correlation_reference_value():
    assert R.D.ly_bf10_r(-0.3, 30, 1) == pytest.approx(0.783243, rel=1e-4)


@pytest.mark.validation
def test_t_ci_uses_pooled_se_and_t_critical():
    out = _dual("t-test", {"a": X, "b": Y})
    res = stats.ttest_ind(X, Y).confidence_interval(0.95)
    assert out.frequentist.ci_lower == pytest.approx(res.low, abs=1e-6)
    assert out.frequentist.ci_upper == pytest.approx(res.high, abs=1e-6)


@pytest.mark.validation
def test_paired_ci_uses_t_critical():
    out = _dual("paired-t", {"a": X, "b": X[::-1]})
    res = stats.ttest_rel(X, X[::-1]).confidence_interval(0.95)
    assert out.frequentist.ci_lower == pytest.approx(res.low, abs=1e-6)
    assert out.frequentist.ci_upper == pytest.approx(res.high, abs=1e-6)


@pytest.mark.validation
def test_d_ci_is_noncentral_t_inversion():
    out = _dual("t-test", {"a": X, "b": Y})
    lo, hi = out.frequentist.effect_ci
    t, df, n_eff = out.frequentist.statistic, len(X) + len(Y) - 2, len(X) * len(Y) / (len(X) + len(Y))
    # At the bounds, the observed t sits at the 97.5th / 2.5th percentile of the noncentral t.
    assert stats.nct.cdf(t, df, lo * math.sqrt(n_eff)) == pytest.approx(0.975, abs=1e-4)
    assert stats.nct.cdf(t, df, hi * math.sqrt(n_eff)) == pytest.approx(0.025, abs=1e-4)


@pytest.mark.validation
def test_t_posterior_is_exact_cauchy_posterior():
    out = _dual("t-test", {"a": X, "b": Y}, prior_scale=0.707)
    n_eff, df = len(X) * len(Y) / (len(X) + len(Y)), len(X) + len(Y) - 2
    mean, sd, lo, hi = R.D.posterior_delta(out.frequentist.statistic, n_eff, df, 0.707)
    assert out.bayesian.posterior_mean == pytest.approx(mean, abs=1e-5)
    assert out.bayesian.credible_interval == pytest.approx([lo, hi], abs=1e-5)
    # Equal-tailed interval, not mean +/- 1.96 sd
    assert out.bayesian.credible_interval[0] != pytest.approx(mean - 1.96 * sd, abs=1e-4)


@pytest.mark.validation
def test_anova_eta2_ci_is_noncentral_f_and_bic_labelled():
    groups = {"a": [4.1, 5.2, 4.8, 5.0, 4.6], "b": [5.9, 6.1, 5.4, 6.3, 5.8], "c": [5.0, 5.5, 4.9, 5.7, 5.2]}
    out = _dual("anova", groups)
    f = out.frequentist
    lo, hi = f.effect_ci
    N, df1, df2 = 15, 2, 12
    lam_lo, lam_hi = lo * N / (1 - lo), hi * N / (1 - hi)
    assert stats.ncf.cdf(f.statistic, df1, df2, lam_lo) == pytest.approx(0.975, abs=1e-4)
    assert stats.ncf.cdf(f.statistic, df1, df2, lam_hi) == pytest.approx(0.025, abs=1e-4)
    assert hi - lo != pytest.approx(0.10, abs=1e-3)
    b = out.bayesian
    assert b.posterior_mean is None and b.credible_interval is None and b.prior_scale is None
    assert "BIC approximation" in b.prior_description and "JZS" not in out.comparison.apa_text


@pytest.mark.smoke
def test_anova_null_effect_ci_has_zero_lower_bound():
    groups = {"a": [1.0, 2.0, 3.0], "b": [1.1, 2.0, 2.9], "c": [1.0, 2.1, 3.0]}
    out = _dual("anova", groups)
    assert out.frequentist.effect_ci[0] == 0.0


@pytest.mark.validation
def test_regression_r2_ci_and_bic_label():
    rng = np.random.default_rng(5)
    x1, x2 = rng.normal(size=50), rng.normal(size=50)
    y = 0.8 * x1 + rng.normal(size=50)
    out = _dual("regression", {"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}, params={"dv": "y"})
    lo, hi = out.frequentist.effect_ci
    assert 0 < lo < out.frequentist.effect_size < hi
    lam_lo = lo * 50 / (1 - lo)
    assert stats.ncf.cdf(out.frequentist.statistic, 2, 47, lam_lo) == pytest.approx(0.975, abs=1e-4)
    assert out.bayesian.bf_method == "BIC approximation" and out.bayesian.posterior_sd is None


@pytest.mark.smoke
@pytest.mark.parametrize("test_type", ["t-test", "anova"])
def test_apa_report_handles_missing_posterior(test_type):
    data = {"a": X, "b": Y} if test_type == "t-test" else {"a": X, "b": Y, "c": [v + 0.3 for v in Y]}
    out = _dual(test_type, data)
    para = run(R.dual_report_apa(R.APAReportRequest(dual_result=out))).apa_paragraph
    assert ("CrI" in para) == (test_type == "t-test")
    assert "BIC" in para or "Cauchy" in para
