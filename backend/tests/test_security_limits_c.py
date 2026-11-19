"""Request-size caps and client-safe errors for the network, spatial, resampling, meta-analysis,
pipeline, sociology, multilevel, propensity, conjoint and criminology routers.

Oversized requests must be refused with 422 before any heavy work, and error responses must not
carry exception text or long user-supplied strings.
"""

import numpy as np
import pytest

SECRET = "SECRET_INTERNAL_/srv/app/lib.py_detail"


def _ok_422(r):
    assert r.status_code == 422, (r.status_code, r.text[:300])
    assert SECRET not in r.text
    return r


# ── ERGM ──────────────────────────────────────────────────────────────

def test_ergm_caps(client):
    big = [[0] * 201 for _ in range(201)]
    _ok_422(client.post("/api/ergm/fit", json={"adjacency": big, "terms": ["edges"]}))
    A = [[0] * 100 for _ in range(100)]
    # 100 nodes -> 50,000 MCMC steps per simulation; 100 simulations exceed the budget
    r = _ok_422(client.post("/api/ergm/gof", json={"adjacency": A, "parameters": {"edges": -1}, "n_simulations": 100}))
    assert "simulations" in r.json()["detail"]
    _ok_422(client.post("/api/ergm/gof", json={"adjacency": A, "parameters": {"edges": -1}, "n_simulations": 5000}))
    _ok_422(client.post("/api/ergm/simulate", json={"n_nodes": 100_000, "parameters": {"edges": -1}}))
    _ok_422(client.post("/api/ergm/simulate", json={"n_nodes": 10, "parameters": {"edges": -1}, "n_iterations": 10**9}))
    S = [[0] * 5 for _ in range(5)]
    _ok_422(client.post("/api/ergm/siena", json={"adjacency_t1": S, "adjacency_t2": S, "n_iterations": 50_000}))
    ragged = [[0, 1, 0], [1, 0], [0, 1, 0]]
    _ok_422(client.post("/api/ergm/fit", json={"adjacency": ragged, "terms": ["edges"]}))


def test_ergm_unknown_term_not_echoed(client):
    A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    r = _ok_422(client.post("/api/ergm/fit", json={"adjacency": A, "terms": ["<script>" + "x" * 500]}))
    assert "<script>" not in r.text


# ── Social graph ──────────────────────────────────────────────────────

def test_social_graph_caps(client):
    many_edges = [[i, i + 1] for i in range(20_001)]
    _ok_422(client.post("/api/social-graph/analyze", json={"edges": many_edges}))
    many_nodes = [[f"a{i}", f"b{i}"] for i in range(1_001)]  # 2,002 nodes
    _ok_422(client.post("/api/social-graph/analyze", json={"edges": many_nodes}))
    ring = [[i, (i + 1) % 50] for i in range(50)]
    _ok_422(client.post("/api/social-graph/influence-diffusion",
                        json={"edges": ring, "seed_nodes": [0], "n_simulations": 1_000_000}))
    _ok_422(client.post("/api/social-graph/ergm-approximate", json={"edges": ring, "n_simulations": 100_000}))
    _ok_422(client.post("/api/social-graph/key-player", json={"edges": ring, "k": 10_000}))
    _ok_422(client.post("/api/social-graph/temporal-network", json={"snapshots": [ring] * 101}))


def test_social_graph_weight_error_not_echoed(client):
    r = _ok_422(client.post("/api/social-graph/analyze", json={"edges": [["a", "b", "<b>" + "y" * 300]]}))
    assert "<b>" not in r.text


# ── Spatial ───────────────────────────────────────────────────────────

def test_spatial_caps(client):
    rng = np.random.default_rng(0)
    n = 5_001
    vals, coords = rng.normal(size=n).tolist(), rng.random((n, 2)).tolist()
    _ok_422(client.post("/api/spatial/morans-i", json={"values": vals, "coordinates": coords}))
    n = 1_500  # 999 permutations x n^2 exceeds the local Moran budget
    _ok_422(client.post("/api/spatial/local-morans",
                        json={"values": rng.normal(size=n).tolist(), "coordinates": rng.random((n, 2)).tolist()}))
    n = 2_001
    _ok_422(client.post("/api/spatial/gwr", json={"y": rng.normal(size=n).tolist(), "X": rng.normal(size=(n, 1)).tolist(),
                                                   "coordinates": rng.random((n, 2)).tolist()}))
    _ok_422(client.post("/api/spatial/spatial-regression",
                        json={"y": rng.normal(size=n).tolist(), "X": rng.normal(size=(n, 1)).tolist(),
                              "coordinates": rng.random((n, 2)).tolist()}))


# ── Novel modules ─────────────────────────────────────────────────────

def test_novel_caps(client):
    rng = np.random.default_rng(1)
    data = {"a": rng.normal(size=50).tolist(), "b": rng.normal(size=50).tolist()}
    _ok_422(client.post("/api/novel/spf/analyze", json={"data": data, "n_bootstrap": 50_000}))
    wide = {f"v{i}": rng.normal(size=20).tolist() for i in range(51)}
    _ok_422(client.post("/api/novel/spf/analyze", json={"data": wide}))
    lengths = {"<i>" + "k" * 200: [1.0] * 12, "b": [1.0] * 11}
    r = _ok_422(client.post("/api/novel/spf/analyze", json={"data": lengths}))
    assert "<i>" not in r.text

    specs = [{"effect_size": 0.1 * (i % 3 - 1), "p_value": 0.04, "decisions": {f"d{i}": "x"}} for i in range(60)]
    _ok_422(client.post("/api/novel/cca/analyze", json={"specifications": specs}))
    _ok_422(client.post("/api/novel/cca/from-multiverse", json={"specifications": [1, 2, 3]}))
    _ok_422(client.post("/api/novel/cca/from-multiverse",
                        json={"specifications": [{"effect_size": "abc", "p_value": 0.1}] * 3}))

    x = rng.normal(size=(40, 1)).tolist()
    y = rng.normal(size=40).tolist()
    _ok_422(client.post("/api/novel/complexity/navigate", json={"x": x, "y": y, "max_poly_degree": 1_000}))
    _ok_422(client.post("/api/novel/complexity/navigate", json={"x": x, "y": y, "cv_folds": 100_000}))
    _ok_422(client.post("/api/novel/dialectical/analyze",
                        json={"group1": [1, 2, 3, 4], "group2": [2, 3, 4, 5], "prior_scales": [0.5] * 100}))


# ── Non-parametric ────────────────────────────────────────────────────

def test_nonparametric_caps(client):
    g = list(np.random.default_rng(2).normal(size=30))
    _ok_422(client.post("/api/nonparametric/permutation-test", json={"group1": g, "group2": g, "n_permutations": 1_000_000}))
    _ok_422(client.post("/api/nonparametric/two-sample-bootstrap", json={"group1": g, "group2": g, "n_bootstrap": 10_000_000}))
    _ok_422(client.post("/api/nonparametric/bootstrap-ci", json={"data": list(range(25_000))}))
    # 10,000 resamples x 5,000 cases exceeds resamples x n <= 20,000,000
    _ok_422(client.post("/api/nonparametric/bootstrap-ci", json={"data": list(range(5_000)), "n_bootstrap": 10_000}))


def test_bootstrap_ci_chunked_jackknife_unchanged():
    from app.routers import nonparametric as N
    data = np.random.default_rng(3).normal(size=2_500)
    for name in ("mean", "median", "std", "trimmed_mean"):
        stat = N._STATS[name]
        full = stat(np.array([np.delete(data, j) for j in range(len(data))]), 0.1)
        assert np.array_equal(N._jackknife(stat, data, 0.1), full)


# ── Meta-analysis ─────────────────────────────────────────────────────

def test_meta_analysis_caps(client):
    k = 5_001
    _ok_422(client.post("/api/meta-analysis/random-effects", json={"effect_sizes": [0.1] * k, "standard_errors": [0.1] * k}))
    _ok_422(client.post("/api/meta-analysis/moderator-analysis",
                        json={"effect_sizes": [0.1] * 60, "standard_errors": [0.1] * 60, "moderators": [[1.0] * 51] * 60}))


# ── Pipeline ──────────────────────────────────────────────────────────

def test_pipeline_caps_and_errors(client, monkeypatch):
    from app.routers import pipeline as P
    data = {"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "b", "a", "b"]}
    steps = [{"id": f"s{i}", "module": "stats", "action": "descriptives"} for i in range(51)]
    _ok_422(client.post("/api/pipeline/run", json={"data": data, "steps": steps}))

    def boom(df, params):
        raise ValueError(SECRET)

    monkeypatch.setitem(P.EXECUTORS, ("stats", "descriptives"), boom)
    r = client.post("/api/pipeline/run", json={"data": data, "steps": [{"id": "s1", "module": "stats", "action": "descriptives"}]})
    assert r.status_code == 200 and SECRET not in r.text
    assert r.json()["results"][0]["status"] == "error"

    long = "<img>" + "c" * 500
    r = client.post("/api/pipeline/run", json={"data": data, "steps": [
        {"id": "s1", "module": "stats", "action": "regression", "params": {"dv": long, "ivs": ["x"]}}]})
    err = r.json()["results"][0]["results"]["error"]
    assert "Column not found" in err and len(err) < 120


# ── Sociology ─────────────────────────────────────────────────────────

def test_sociology_caps(client):
    _ok_422(client.post("/api/sociology/event-history",
                        json={"durations": [1, 2, 3], "events": [1, 0, 1], "time_intervals": list(range(1, 300))}))
    _ok_422(client.post("/api/sociology/segregation-indices",
                        json={"group_a_counts": [1] * 1_001, "group_b_counts": [1] * 1_001}))
    _ok_422(client.post("/api/sociology/social-mobility",
                        json={"origin": list(range(150)), "destination": list(range(150))}))
    _ok_422(client.post("/api/sociology/survey-weighted",
                        json={"values": [1.0] * 1_500, "weights": [1.0] * 1_500, "groups": list(range(1_500))}))
    _ok_422(client.post("/api/sociology/inequality-indices", json={"values": [1.0, 2.0] * 15_000}))


def test_sociology_model_error_not_echoed(client, monkeypatch):
    import statsmodels.api as sm

    def bad_logit(*a, **k):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(sm, "Logit", bad_logit)
    rng = np.random.default_rng(4)
    n = 40
    r = client.post("/api/sociology/event-history", json={
        "durations": rng.integers(1, 6, size=n).tolist(), "events": rng.integers(0, 2, size=n).tolist(),
        "covariates": rng.normal(size=(n, 1)).tolist()})
    assert r.status_code == 200 and SECRET not in r.text
    assert "failed" in r.json()["discrete_time_logistic"]["error"]


# ── HLM / multilevel SEM ──────────────────────────────────────────────

def test_hlm_caps(client):
    n = 4_100
    groups = [i // 2 for i in range(n)]  # 2,050 groups
    _ok_422(client.post("/api/hlm/two-level", json={"dependent": [0.0] * n, "level1_predictors": [[0.0] * n], "groups": groups}))
    m = 40
    preds = [[float(j * i % 7) for j in range(m)] for i in range(6)]
    _ok_422(client.post("/api/hlm/two-level", json={"dependent": [0.0] * m, "level1_predictors": preds,
                                                     "groups": [j % 4 for j in range(m)], "random_slopes": True}))


def _msem_rows(n=200, clusters=20, seed=5):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(clusters), n // clusters)
    x = rng.normal(size=n) + rng.normal(size=clusters)[g]
    y = 0.5 * x + rng.normal(size=n)
    return [{"g": int(gi), "x": float(xi), "y": float(yi)} for gi, xi, yi in zip(g, x, y)]


def test_msem_caps_and_errors(client, monkeypatch):
    rows = [{"g": i % 50, "x": 1.0, "y": 2.0} for i in range(20_001)]
    _ok_422(client.post("/api/multilevel-sem/msem-two-level",
                        json={"data": rows, "group_ids": "g", "within_model": ["y ~ x"], "between_model": []}))
    long = "q" * 400
    r = client.post("/api/multilevel-sem/within-between-decomposition",
                    json={"data": _msem_rows(), "group_ids": "g", "variables": [long]})
    assert r.status_code == 400 and long not in r.text

    from app import sem_engine

    def boom(self, *a, **k):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(sem_engine.TwoLevelSEM, "fit", boom)
    r = _ok_422(client.post("/api/multilevel-sem/msem-two-level",
                            json={"data": _msem_rows(), "group_ids": "g", "within_model": ["y ~ x"], "between_model": ["y ~ x"]}))


# ── Propensity / conjoint / criminology ───────────────────────────────

def test_propensity_caps(client):
    n = 6_000
    t = [i % 2 for i in range(n)]
    cov = [[float(i % 13) for i in range(n)]]
    _ok_422(client.post("/api/propensity-score/match", json={"treatment": t, "outcome": [0.0] * n, "covariates": cov}))
    m = 40
    t2 = [i % 2 for i in range(m)]
    cov2 = [[float((i * 7) % 11) for i in range(m)]]
    _ok_422(client.post("/api/propensity-score/iptw", json={"treatment": t2, "outcome": [0.0] * m, "covariates": cov2,
                                                      "n_bootstrap": 1_000_000}))
    _ok_422(client.post("/api/propensity-score/balance", json={"treatment": t2, "covariates": cov2,
                                                         "matched_indices_treated": [1, 99_999],
                                                         "matched_indices_control": [0, 2]}))


def _conjoint_tasks(n_tasks=30, seed=6):
    rng = np.random.default_rng(seed)
    profiles, choices = [], []
    for _ in range(n_tasks):
        task = [{"price": str(rng.choice(["low", "mid", "high"])), "brand": str(rng.choice(["A", "B"]))} for _ in range(3)]
        profiles.append(task)
        choices.append(int(rng.integers(0, 3)))
    return profiles, choices


def test_conjoint_caps(client):
    profiles, choices = _conjoint_tasks()
    _ok_422(client.post("/api/conjoint/importance", json={"profiles": profiles, "choices": choices, "n_bootstrap": 1_000_000}))
    r = _ok_422(client.post("/api/conjoint/analyze", json={"profiles": profiles, "choices": choices,
                                                           "attribute_names": ["<svg>" + "z" * 100]}))
    assert "<svg>" not in r.text
    wide = [[{"a": f"L{i}-{j}"} for j in range(2)] for i in range(40)]  # 79 parameters
    _ok_422(client.post("/api/conjoint/analyze", json={"profiles": wide, "choices": [0] * 40}))


def test_criminology_caps(client):
    incidents = [{"lat": 40 + i * 1e-4, "lng": -74 + i * 1e-4} for i in range(200)]
    _ok_422(client.post("/api/criminology/geographic-profile", json={"incidents": incidents, "grid_resolution": 200}))
    pts = [[float(i), float(i % 7)] for i in range(200)]
    _ok_422(client.post("/api/criminology/spatial-cluster", json={"points": pts, "n_clusters": 100}))
    n = 20_001
    _ok_422(client.post("/api/criminology/recidivism-survival", json={"time_to_event": [1.0] * n, "event_occurred": [1] * n}))
    m = 6_000
    _ok_422(client.post("/api/criminology/propensity-score",
                        json={"treatment": [i % 2 for i in range(m)], "covariates": [[1.0]] * m, "outcome": [0.0] * m}))
