"""
Regression tests for the follow-up check endpoints: TOST, What Would It Take,
Interpretation Confidence (PPV), Reviewer 2, Triangulation and Analytic
Decision Sensitivity. Reference values come from statsmodels (ttost_ind,
ttost_paired, TTestIndPower, FTestAnovaPower) and published G*Power results.
"""

import asyncio
import math

import numpy as np
import pytest
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from statsmodels.stats.weightstats import ttost_ind, ttost_paired

from app.routers import equivalence as E
from app.routers import interpretation_confidence as I
from app.routers import reviewer2 as R
from app.routers import sensitivity_decisions as S
from app.routers import triangulation as T
from app.routers import what_would_it_take as W

run = asyncio.run


# ── TOST ─────────────────────────────────────────────────────────────────

@pytest.mark.validation
def test_tost_matches_statsmodels_welch_and_paired():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.1, 1, 25), rng.normal(0, 1.5, 40)
    r = run(E.analyze_equivalence(E.EquivalenceRequest(
        test_type="ttest", group1=a.tolist(), group2=b.tolist(), equivalence_bound=0.6, bound_type="raw")))
    assert r.tost_p == pytest.approx(ttost_ind(a, b, -0.6, 0.6, usevar="unequal")[0], abs=1e-12)
    p, q = rng.normal(0, 1, 20), rng.normal(0, 1, 20)
    r = run(E.analyze_equivalence(E.EquivalenceRequest(
        test_type="paired", group1=p.tolist(), group2=q.tolist(), equivalence_bound=0.4, bound_type="raw")))
    assert r.tost_p == pytest.approx(ttost_paired(p, q, -0.4, 0.4)[0], abs=1e-12)


@pytest.mark.smoke
def test_tost_decision_equals_ci_inside_bounds_and_d_bound_is_standardised():
    g1, g2 = [10, 11, 12, 13, 14] * 4, [10.2, 11.1, 12.3, 12.9, 14.1] * 4
    r = run(E.analyze_equivalence(E.EquivalenceRequest(test_type="ttest", group1=g1, group2=g2, equivalence_bound=0.5)))
    assert r.standardized_bounds == pytest.approx([-0.5, 0.5])
    lo, hi = r.equivalence_bounds
    assert r.tost_significant == (lo < r.confidence_interval[0] and r.confidence_interval[1] < hi)


@pytest.mark.smoke
def test_tost_rejects_degenerate_proportions():
    with pytest.raises(Exception):
        run(E.analyze_equivalence(E.EquivalenceRequest(test_type="proportion", successes1=0, n1=10, successes2=0, n2=10)))


# ── What Would It Take ───────────────────────────────────────────────────

@pytest.mark.validation
def test_power_matches_statsmodels():
    assert W.power_two_sample(0.5, 128, 0.05) == pytest.approx(TTestIndPower().power(0.5, nobs1=64, alpha=0.05), abs=1e-6)
    # ANOVA: exact Poisson-mixture series for the noncentral F (statsmodels agrees to ~2e-6)
    from scipy import stats
    f, n, k = 0.25, 180, 4
    d1, d2, lam = k - 1, n - k, f * f * n
    crit = stats.f.ppf(0.95, d1, d2)
    x = d1 * crit / (d1 * crit + d2)
    j = np.arange(400)
    exact = float((stats.poisson.pmf(j, lam / 2) * stats.beta.sf(x, d1 / 2 + j, d2 / 2)).sum())
    assert W.power_anova(f, n, 0.05, k) == pytest.approx(exact, abs=1e-12)
    assert W.power_anova(f, n, 0.05, k) == pytest.approx(FTestAnovaPower().power(f, nobs=n, alpha=0.05, k_groups=k), abs=1e-5)


@pytest.mark.validation
@pytest.mark.parametrize("power_fn,es,n_min,expected", [
    (lambda e, m: W.power_correlation(e, m, 0.05), 0.3, 4, 84),  # G*Power exact
    (lambda e, m: W.power_correlation(e, m, 0.05), 0.5, 4, 29),
    (lambda e, m: W.power_regression(e, m, 0.05, 3), 0.15 / 1.15, 5, 77),  # f² = .15, 3 predictors
])
def test_required_n_matches_gpower(power_fn, es, n_min, expected):
    assert W._required_n(power_fn, es, 0.8, n_min, 1) == expected


@pytest.mark.smoke
def test_correlation_power_at_null_equals_alpha():
    assert W.power_correlation(0.0, 30, 0.05) == pytest.approx(0.05, abs=1e-6)


@pytest.mark.smoke
def test_wwit_endpoint_reports_sensitivity_and_no_fragility_index():
    out = run(W.what_would_it_take(W.SensitivityRequest(test_type="ttest", observed_es=0.45, n=60, observed_p=0.087)))
    assert out.required_n_80_power == 158 and out.n_for_significance == 79
    assert not hasattr(out, "fragility_index")


# ── PPV ──────────────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_ppv_formula():
    out = run(I.compute_ppv(I.PPVRequest(power=0.8, prior_probability=0.1, alpha=0.05)))
    assert out.ppv == pytest.approx(0.08 / (0.08 + 0.045), abs=1e-4)
    with pytest.raises(Exception):
        run(I.compute_ppv(I.PPVRequest(prior_probability=0.5)))


# ── Reviewer 2 ───────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_reviewer2_uses_sensitivity_not_observed_power_and_metric_benchmarks():
    out = run(R.adversarial_review(R.ReviewRequest(
        test_type="anova", results={"p_value": 0.2, "effect_size": 0.03}, sample_size=90, design_info={"groups": 3})))
    cats = {c.category for c in out.critiques}
    assert "Sensitivity" in cats and "Statistical Power" not in cats
    assert "Effect Size" not in cats  # η² = .03 is above the .01 benchmark (a d-threshold of .2 would wrongly flag it)


# ── Triangulation ────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_triangulation_effect_signs_agree_for_a_real_difference():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.8, 1, 30), rng.normal(0, 1, 30)
    out = run(T.triangulate(T.TriangulationRequest(data_type="group_comparison", group1=a.tolist(), group2=b.tolist())))
    assert out.convergence.direction_agree
    assert all(m.effect_size > 0 for m in out.methods)


# ── Decision sensitivity ─────────────────────────────────────────────────

@pytest.mark.validation
def test_yuen_and_akp():
    from scipy import stats
    from scipy.stats import mstats
    rng = np.random.default_rng(2)
    a, b = rng.lognormal(0, 0.8, 23), rng.lognormal(0.4, 0.8, 31)
    spec = S._group_spec(a, b, "none", "raw", "robust", 0.05)

    def parts(x, tr=0.2):
        n = len(x); g = int(math.floor(tr * n)); h = n - 2 * g
        w = np.asarray(mstats.winsorize(x, limits=(tr, tr)))
        return stats.trim_mean(x, tr), (n - 1) * w.var(ddof=1) / (h * (h - 1)), h
    m1, d1, h1 = parts(a); m2, d2, h2 = parts(b)
    t = (m1 - m2) / math.sqrt(d1 + d2)
    df = (d1 + d2) ** 2 / (d1 ** 2 / (h1 - 1) + d2 ** 2 / (h2 - 1))
    assert spec.p_value == pytest.approx(2 * stats.t.sf(abs(t), df), abs=1e-6)
    big1, big2 = rng.normal(0.5, 1, 40000), rng.normal(0, 1, 40000)
    assert S._akp_d(big1, big2) == pytest.approx(S._cohens_d(big1, big2), abs=0.02)


@pytest.mark.smoke
def test_sensitivity_signs_consistent_and_shift_invariant():
    rng = np.random.default_rng(3)
    g1, g2 = rng.normal(5.6, 1, 30), rng.normal(5, 1, 30)
    o = run(S.analyze_sensitivity(S.SensitivityRequest(data_type="group_comparison", group1=g1.tolist(), group2=g2.tolist())))
    assert {np.sign(s.effect_size) for s in o.specifications if s.effect_size} == {1.0}
    o2 = run(S.analyze_sensitivity(S.SensitivityRequest(
        data_type="group_comparison", group1=(g1 - 100).tolist(), group2=(g2 - 100).tolist())))
    for s, t in zip(o.specifications, o2.specifications):
        if s.transform in ("raw", "rank"):
            assert s.p_value == pytest.approx(t.p_value, abs=1e-9)
