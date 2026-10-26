"""
Regression tests for the robustness-check modules (formerly "Novel Methods").
Reference Bayes factors were computed with pingouin 0.5 (bayesfactor_ttest,
bayesfactor_pearson) and agree with our implementation to ~1e-9.
"""

import asyncio
import math

import numpy as np
import pytest

from app.routers import novel_cca as A
from app.routers import novel_complexity as C
from app.routers import novel_dialectical as D
from app.routers import novel_spf as S

run = asyncio.run


@pytest.mark.validation
@pytest.mark.parametrize("t,n_eff,df,r,ref", [
    (-6.676, 7.5, 28, 0.707, 34793.5),
    (2.1, 20, 19, 0.707, 1.4047),
    (2.1, 20, 19, 1.414, 0.900307),
    (2.1, 7.2, 28, 0.707, 1.72415),
    (12.0, 100, 398, 0.707, 1.6772e25),
])
def test_jzs_bf_matches_reference(t, n_eff, df, r, ref):
    assert D.jzs_bf10_t(t, n_eff, df, r) == pytest.approx(ref, rel=1e-4)


@pytest.mark.validation
@pytest.mark.parametrize("r,n,kappa,ref", [
    (-0.3, 30, 1, 0.783243), (0.5, 100, 1, 125565), (0.3, 30, 0.5, 1.05746), (0.3, 30, 2, 0.527814),
])
def test_correlation_bf_matches_reference(r, n, kappa, ref):
    assert D.ly_bf10_r(r, n, kappa) == pytest.approx(ref, rel=1e-4)


@pytest.mark.smoke
def test_d_ci_contains_estimate_and_posterior_is_finite():
    lo, hi = D._d_ci(60, 14, 15)
    assert lo < 60 / math.sqrt(15) < hi
    mean, sd, clo, chi = D.posterior_delta(60, 15, 14, 0.707)
    assert all(map(math.isfinite, (mean, sd, clo, chi))) and clo < mean < chi


@pytest.mark.smoke
def test_dialectical_correlation_reports_r_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = 0.5 * x + rng.normal(size=40)
    out = run(D.dialectical_inference(D.DialecticalRequest(
        group1=x.tolist(), group2=(y + 50).tolist(), test_type="correlation")))
    assert out.frequentist.effect_label == "r"
    assert -1 < out.bayesian.credible_lower < out.bayesian.posterior_mean < out.bayesian.credible_upper < 1


@pytest.mark.smoke
def test_spf_noise_is_mostly_unstable_and_holm_is_monotone():
    rng = np.random.default_rng(1)
    unstable = 0
    for _ in range(40):
        res = run(S.scale_persistent_features(S.SPFRequest(
            data={"a": rng.normal(size=60).tolist(), "b": rng.normal(size=60).tolist()})))
        unstable += res.features[0].persistence_label == "Unstable"
    assert unstable >= 24
    assert S._holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.smoke
def test_complexity_intercept_cv_is_out_of_sample():
    rng = np.random.default_rng(2)
    y = 3 + rng.normal(size=30)
    out = run(C.navigate_complexity(C.ComplexityRequest(x=[[v] for v in range(30)], y=y.tolist())))
    assert out.complexity_path[0].cv_rmse > out.complexity_path[0].rmse


@pytest.mark.smoke
def test_cca_decision_flip_and_no_size_word_without_metric():
    specs = [A.Specification(effect_size=e, p_value=p, decisions={"outlier": o})
             for e, p, o in [(0.4, 0.01, "keep"), (0.5, 0.001, "keep"), (0.45, 0.002, "keep"),
                             (-0.1, 0.5, "drop"), (-0.2, 0.3, "drop")]]
    out = run(A.convergent_core_analysis(A.CCARequest(specifications=specs)))
    assert out.boundary_conditions[0]["flip_rate"] > 0
    assert "Cohen" not in out.core_claim
