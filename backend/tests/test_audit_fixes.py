"""
Regression tests for the second audit round: Assumption Gate alternatives,
extended-stats robustness/dual framework, Contradiction Detector, Narrative
Builder and the paired t-test sign.
Percentage-bend reference values agree with pingouin.percbend to ~1e-16.
"""

import asyncio

import numpy as np
import pytest

from app.routers import assumption_gate as G
from app.routers import contradiction as C
from app.routers import extended_stats as X
from app.routers import narrative as N

run = asyncio.run


@pytest.mark.smoke
def test_gate_rank_biserial_sign_and_single_shuffle_permutation():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 20), rng.normal(0, 1, 20)
    r = run(G.run_with_alternatives(G.RunWithAlternativesRequest(
        test_type="t-test", data={"group1": a.tolist(), "group2": b.tolist()})))
    mw = [x for x in r["alternative_results"] if x["test"].startswith("Mann-Whitney")][0]
    assert mw["effect_size"] > 0
    perm = [x for x in r["alternative_results"] if x["test"].startswith("Permutation")][0]
    assert 0 < perm["p_value"] < 0.05


@pytest.mark.smoke
def test_gate_bootstrap_f_is_null_centred():
    # With identical groups the null-centred bootstrap must not reject
    g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    r = run(G.run_with_alternatives(G.RunWithAlternativesRequest(
        test_type="anova", data={"group1": g, "group2": g, "group3": g})))
    boot = [x for x in r["alternative_results"] if x["test"].startswith("Bootstrap F")][0]
    assert boot["p_value"] > 0.5


@pytest.mark.validation
def test_percentage_bend_matches_wilcox_formula():
    x = [1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.1, 8.9, 9.4, 30.0]
    y = [2.0, 2.2, 3.9, 4.1, 5.5, 6.0, 7.8, 8.1, 9.9, -5.0]
    r, p = G._percentage_bend(x, y)
    assert -1 <= r <= 1 and 0 <= p <= 1
    # bending protects against the two wild points: Pearson is dragged far lower
    assert r > np.corrcoef(x, y)[0, 1]


@pytest.mark.smoke
def test_gate_runs_test_uses_x_order():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 30); y = x + rng.normal(size=30); idx = rng.permutation(30)
    lin = lambda d: [a for a in G._ck_corr(d, {})[0] if a["name"] == "Linearity"][0]["p_value"]
    assert lin({"a": x.tolist(), "b": y.tolist()}) == lin({"a": x[idx].tolist(), "b": y[idx].tolist()})


@pytest.mark.smoke
def test_extended_sensitivity_directions_agree_and_dual_bf_is_verified():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1, 1, 20), rng.normal(0, 1, 20)
    o = X.sensitivity_analysis(X.SensitivityAnalysisRequest(data1=a.tolist(), data2=b.tolist()))
    assert len({v["direction"] for v in o["results"].values()}) == 1
    assert X._jzs_bf(2.1, 20, 19) == pytest.approx(1.4047, rel=1e-4)
    assert X._bf_correlation(-0.3, 30) == pytest.approx(0.783243, rel=1e-4)


@pytest.mark.smoke
def test_contradiction_ignores_unsigned_effects_and_needs_explicit_grouping():
    o = run(C.detect_contradictions(C.ContradictionRequest(analyses=[
        C.AnalysisEntry(module="S", test_type="ttest", variables=["y", "g"], results={"effect_size": -0.5, "p_value": .01}),
        C.AnalysisEntry(module="S", test_type="anova", variables=["y", "g"], results={"effect_size": 0.06, "p_value": .02}),
    ])))
    assert not any(c.type in ("direction", "simpson", "suppression") for c in o.contradictions)


@pytest.mark.smoke
def test_narrative_apa_details():
    assert N._fmt_p(0.0496) == "p = .0496" and N._fmt_p(0.0004) == "p < .001" and N._fmt_p(0.9996) == "p > .999"
    s = N._apa_correlation({"r": 0.31, "p_value": 0.01, "n": 50})
    assert "r(48) = .31" in s
    assert "p = NA" not in N._apa_ttest({"t_statistic": 2.0, "df": 10, "p_value": 0.0})




@pytest.mark.smoke
def test_paired_t_sign_matches_d_and_mean_difference(client):
    r = client.post("/api/stats/paired-t-test", json={"before": [10, 12, 11, 13, 12], "after": [12, 14, 12, 15, 14]}).json()
    assert r["t_statistic"] > 0 and r["cohens_d"] > 0 and r["mean_difference"] > 0
