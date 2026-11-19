"""Size caps and client-safe errors for the ANOVA, advanced stats, assumptions, Bayesian,
contradiction, ethnography, extended stats, latent class / LTA / DCM / IRT, MICE, mediation,
medical and network psychometrics routers.

Oversized requests must be refused with a 422 before any heavy work starts; error responses
must not carry exception text, and must not reflect raw request strings.
"""

import numpy as np
import pytest

SECRET = "SECRET-INTERNAL-DETAIL /srv/app/lib.py"


def _boom(*args, **kwargs):
    raise RuntimeError(SECRET)


def _clean(r):
    assert SECRET not in r.text, r.text[:300]
    assert "Traceback" not in r.text


rng = np.random.default_rng(0)


# ── DoS caps ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,body", [
    # advanced stats
    ("/api/advanced/correlation-matrix", {"variables": [[1.0, 2.0, 3.0]] * 201}),
    ("/api/advanced/mediation", {"x": list(range(2000)), "m": list(range(2000)), "y": list(range(2000)),
                                 "n_bootstrap": 20000}),
    ("/api/advanced/multiple-regression", {"predictors": [[1.0, 2.0, 3.0]] * 201, "outcome": [1.0, 2.0, 3.0]}),
    ("/api/advanced/repeated-measures", {"measurements": [[1.0, 2.0, 3.0]] * 101}),
    ("/api/advanced/time-series", {"values": list(range(30)), "arima_order": [50, 0, 50]}),
    # advanced ANOVA
    ("/api/advanced-anova/mixed-anova", {"data": [[1.0, 2.0]] * 10_001, "groups": [0, 1] * 5000 + [0]}),
    ("/api/advanced-anova/discriminant", {"predictors": [[float(i % 7) for i in range(5001)]],
                                          "groups": [i % 2 for i in range(5001)]}),
    ("/api/advanced-anova/repeated-measures-anova", {"data": [[1.0] * 101] * 3}),
    # Bayesian
    ("/api/bayesian/bayes-factor/proportion", {"successes1": 1, "trials1": 10 ** 13, "successes2": 1, "trials2": 5}),
    ("/api/bayesian-advanced/mcmc", {"data": [1.0, 2.0, 3.0], "n_samples": 50_001}),
    ("/api/bayesian-advanced/mcmc", {"data": [1.0, 2.0, 3.0], "n_chains": 8, "n_samples": 50_000}),
    ("/api/bayesian-advanced/mcmc", {"data": [1.0, 2.0, 3.0], "n_samples": 0}),
    ("/api/bayesian-advanced/posterior-predictive", {"data": [1.0, 2.0, 3.0], "n_samples": 100_000,
                                                     "n_simulations": 10_000}),
    ("/api/bayesian-advanced/model-compare", {"data": [1.0, 2.0, 3.0], "models": ["normal"] * 11}),
    # assumptions
    ("/api/assumptions/check", {"test_type": "anova", "groups": [[1.0, 2.0, 3.0]] * 201}),
    # ethnography
    ("/api/ethnography/cultural-consensus", {"responses": [[1, 0, 1]] * 1001}),
    ("/api/ethnography/intercoder-reliability", {"coder_assignments": [["a"] * 101] * 3}),
    ("/api/ethnography/pile-sort", {"items": [f"i{i}" for i in range(301)], "sorts": [[list(range(301))]]}),
    # extended stats
    ("/api/extended-stats/roc-auc", {"true_labels": [i % 2 for i in range(20_001)],
                                     "scores": [float(i) for i in range(20_001)]}),
    ("/api/extended-stats/sensitivity-analysis", {"data1": [1.0] * 5001, "data2": [2.0] * 5000}),
    # latent class
    ("/api/latent-class/lca", {"data": [[0, 1], [1, 0]] * 10, "tol": 0}),
    ("/api/latent-class/lca", {"data": [[0, 1], [1, 0]] * 10, "n_starts": 51}),
    ("/api/latent-class/lca", {"data": [[0, 1], [1, 0]] * 10, "max_iter": 5001}),
    ("/api/latent-class/compare-models", {"data": [[0, 1], [1, 0]] * 10, "blrt_samples": 501}),
    ("/api/latent-class/lca", {"data": [[0, 1]] * 20_001}),
    ("/api/latent-class/lpa", {"data": [[float(j) for j in range(100)]] * 5000, "n_classes": 10,
                               "max_iter": 5000, "n_starts": 50}),
    ("/api/latent-class/class-comparison", {"assignments": list(range(21)) * 2,
                                            "auxiliary_variables": {"v": [1.0] * 42}}),
    # LTA / DCM / IRT
    ("/api/lta/analyze", {"indicators": [[[0, 1]] * 40] * 2, "tol": 0}),
    ("/api/lta/analyze", {"indicators": [[[0, 1]] * 40] * 2, "max_iter": 10_001}),
    ("/api/lta/analyze", {"indicators": [[[0, 1]] * 40] * 2, "n_classes": 11}),
    ("/api/dcm/dina", {"responses": [[0, 1]] * 30, "q_matrix": [[1], [1]], "random_starts": 21}),
    ("/api/dcm/dina", {"responses": [[0, 1]] * 30, "q_matrix": [[1], [1]], "tol": 0}),
    ("/api/irt/rasch", {"responses": [[0, 1]] * 20, "tol": 0}),
    ("/api/irt/cat-simulation", {"item_bank": [{"a": 1, "b": 0}] * 600, "n_items": 501}),
    ("/api/irt/rasch", {"responses": [[0, 1] * 51, [1, 0] * 51] * 10}),
    # MICE
    ("/api/mice/impute", {"data": [{"a": 1.0, "b": None}, {"a": 2.0, "b": 1.0}], "max_iterations": 101}),
    # mediation
    ("/api/mediation/simple-mediation", {"x": [1.0] * 20, "m": [1.0] * 20, "y": [1.0] * 20, "n_bootstrap": 50_001}),
    ("/api/mediation/serial-mediation", {"x": [1.0] * 3000, "m1": [1.0] * 3000, "m2": [1.0] * 3000,
                                         "y": [1.0] * 3000, "n_bootstrap": 10_000}),
    ("/api/mediation/parallel-mediation", {"x": [1.0] * 20, "y": [1.0] * 20, "mediators": [[1.0] * 20] * 11}),
    ("/api/mediation/moderation", {"x": [1.0] * 20, "w": [1.0] * 20, "y": [1.0] * 20, "probe_values": [0.0] * 21}),
    # medical
    ("/api/medical/cox-regression", {"durations": [1.0] * 12, "events": [1] * 12, "covariates": [[0.0] * 51] * 12}),
    ("/api/medical/kaplan-meier", {"durations": [1.0] * 60, "events": [1] * 60,
                                   "group_labels": [f"g{i}" for i in range(60)]}),
    # network psychometrics
    ("/api/network-psychometrics/estimate", {"data": [[float(j * i % 7) for j in range(101)] for i in range(5)]}),
    ("/api/network-psychometrics/centrality", {"adjacency": [[0.0] * 201] * 201}),
])
def test_oversized_requests_are_refused(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 422, (path, r.status_code, r.text[:300])


def test_pile_sort_rejects_repeated_items(client):
    # Repeating an index would let one pile grow without bound (pairs are counted in Python)
    r = client.post("/api/ethnography/pile-sort", json={"items": ["a", "b", "c"], "sorts": [[[0, 0, 0, 0, 1, 2]]]})
    assert r.status_code == 400


def test_network_compare_work_cap(client):
    g = rng.normal(size=(200, 60)).tolist()
    r = client.post("/api/network-psychometrics/compare",
                    json={"data_group1": g, "data_group2": g, "n_permutations": 100})
    assert r.status_code == 422
    assert "2,000,000,000" in r.text


def test_mediation_bootstrap_still_matches_with_blocks(client):
    """The blocked bootstrap gives the same numbers whatever the block size."""
    from app.routers import mediation as M
    n, B = 2500, 1000  # several blocks of resamples
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(size=n)
    y = 0.4 * m + rng.normal(size=n)
    out = M.simple_mediation(M.SimpleMediationRequest(x=x.tolist(), m=m.tolist(), y=y.tolist(),
                                                      n_bootstrap=B, seed=3))
    # Unblocked reference: all resamples at once, as before the change
    idx = np.random.default_rng(3).integers(0, n, size=(B, n))
    Xa, Xbc = M._build_design(x), M._build_design(x, m)
    ab = M._batched_coef(m[idx], Xa[idx], 1) * M._batched_coef(y[idx], Xbc[idx], 2)
    assert out["indirect_effect"]["bootstrap_se"] == float(np.std(ab, ddof=1))
    assert out["indirect_effect"]["percentile_ci"] == list(M._percentile_ci(ab, 0.95))


# ── Error responses carry no exception text ───────────────────────────────

def test_mixed_model_error_is_safe(client, monkeypatch):
    from app.routers import advanced_stats
    monkeypatch.setattr(advanced_stats, "fit_mixedlm", _boom)
    n = 20
    r = client.post("/api/advanced/mixed-model", json={
        "dependent": rng.normal(size=n).tolist(), "fixed_effects": [rng.normal(size=n).tolist()],
        "groups": [i % 4 for i in range(n)]})
    assert r.status_code == 422
    _clean(r)


def test_cox_error_is_safe(client, monkeypatch):
    from app.routers import medical
    monkeypatch.setattr(medical.CoxPHFitter, "fit", _boom)
    n = 12
    r = client.post("/api/medical/cox-regression", json={
        "durations": list(range(1, n + 1)), "events": [1, 0] * 6, "covariates": [[float(i)] for i in range(n)]})
    assert r.status_code == 422
    _clean(r)


def test_logistic_error_is_safe(client, monkeypatch):
    import statsmodels.api as sm
    monkeypatch.setattr(sm, "Logit", _boom)
    r = client.post("/api/extended-stats/logistic-regression", json={
        "dependent": [0, 1] * 10, "predictors": [[float(i)] for i in range(20)]})
    assert r.status_code == 422
    _clean(r)


def test_mediation_ols_error_is_safe(client, monkeypatch):
    from app.routers import mediation
    monkeypatch.setattr(mediation.sm, "OLS", _boom)
    r = client.post("/api/mediation/serial-mediation", json={
        "x": rng.normal(size=20).tolist(), "m1": rng.normal(size=20).tolist(),
        "m2": rng.normal(size=20).tolist(), "y": rng.normal(size=20).tolist()})
    assert r.status_code == 422
    _clean(r)


def test_assumptions_regression_error_is_safe(client, monkeypatch):
    import statsmodels.api as sm
    monkeypatch.setattr(sm, "OLS", _boom)
    r = client.post("/api/assumptions/auto-check-and-route", json={
        "test_type": "regression", "data1": rng.normal(size=20).tolist(), "data2": rng.normal(size=20).tolist()})
    assert r.status_code == 422
    _clean(r)


def test_pile_sort_mds_error_is_safe(client, monkeypatch):
    from app.routers import ethnography_backend
    monkeypatch.setattr(ethnography_backend, "MDS", _boom)
    r = client.post("/api/ethnography/pile-sort", json={
        "items": ["a", "b", "c", "d"], "sorts": [[[0, 1], [2, 3]], [[0, 2], [1, 3]]]})
    assert r.status_code == 422
    _clean(r)


# /api/contradiction needs a signed-in user, so its handler is called directly.

def test_contradiction_caps_analyses():
    from pydantic import ValidationError
    from app.routers import contradiction
    a = {"module": "m", "test_type": "pearson", "variables": ["x"], "results": {}}
    with pytest.raises(ValidationError):
        contradiction.ContradictionRequest(analyses=[a] * 201)


def test_contradiction_error_is_safe(monkeypatch):
    from fastapi import HTTPException
    from app.routers import contradiction
    monkeypatch.setattr(contradiction, "_detect_contradictions", _boom)
    a = {"module": "m", "test_type": "pearson", "variables": ["x"], "results": {}}
    with pytest.raises(HTTPException) as e:
        contradiction.detect_contradictions(contradiction.ContradictionRequest(analyses=[a, a]))
    assert e.value.status_code == 422
    assert SECRET not in str(e.value.detail)


def test_model_compare_per_model_error_is_safe(client, monkeypatch):
    from scipy import stats
    monkeypatch.setattr(stats.t, "fit", _boom)
    r = client.post("/api/bayesian-advanced/model-compare", json={"data": rng.normal(size=30).tolist(),
                                                                  "models": ["normal", "t"]})
    assert r.status_code == 200
    _clean(r)
    assert any("error" in m for m in r.json()["models"])


def test_latent_class_per_model_error_is_safe(client, monkeypatch):
    from app.routers import latent_class
    monkeypatch.setattr(latent_class, "_run_lca", _boom)
    data = rng.integers(0, 2, size=(40, 4)).tolist()
    r = client.post("/api/latent-class/compare-models", json={"data": data, "max_classes": 2, "blrt_samples": 0})
    assert r.status_code == 200
    _clean(r)


def test_box_m_error_is_safe(monkeypatch):
    from app.routers import advanced_anova
    monkeypatch.setattr(advanced_anova.np, "cov", _boom)
    Y = rng.normal(size=(10, 2))
    groups = np.array([0, 1] * 5)
    out = advanced_anova._box_m_test(Y, groups, [0, 1], {0: np.arange(0, 10, 2), 1: np.arange(1, 10, 2)},
                                     {0: 5, 1: 5}, 2, 2, 10)
    assert SECRET not in str(out)


def test_arima_failure_in_time_series_summary_is_safe(client, monkeypatch):
    from app.routers import advanced_stats
    monkeypatch.setattr(advanced_stats._ts, "arima_model", _boom)
    r = client.post("/api/advanced/time-series", json={"values": rng.normal(size=40).tolist()})
    assert r.status_code == 200
    _clean(r)


# ── No reflection of raw request strings ──────────────────────────────────

PAYLOAD = "<script>alert(1)</script>"


@pytest.mark.parametrize("path,body", [
    ("/api/bayesian-advanced/mcmc", {"data": [1.0, 2.0, 3.0], "model": PAYLOAD}),
    ("/api/bayesian-advanced/posterior-predictive", {"data": [1.0, 2.0, 3.0], "model": PAYLOAD}),
    ("/api/advanced/non-parametric", {"test": PAYLOAD}),
    ("/api/extended-stats/effect-size-benchmarks", {"effect_size": 0.5, "effect_type": PAYLOAD, "domain": "x"}),
    ("/api/extended-stats/effect-size-benchmarks", {"effect_size": 0.5, "effect_type": "cohens_d", "domain": PAYLOAD}),
])
def test_bad_options_are_not_reflected(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 400
    assert PAYLOAD.lower() not in r.text.lower()
