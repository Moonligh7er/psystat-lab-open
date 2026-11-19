"""Size caps and client-safe errors for the simulator, multiverse, sensitivity, causal, effect-size,
ML, text-analysis, time-series, political-science and power routers.

Oversized requests must be refused with a 422 before any heavy work starts, and error responses
must not carry exception text from numpy/scipy/statsmodels/sklearn.
"""

import time

import numpy as np
import pytest

LIB_TEXT = ("Traceback", "f(a) and f(b)", "LinAlgError", "could not convert", "setting an array element",
            "inhomogeneous", "Got:")


def _no_lib_text(r):
    body = r.text
    for frag in LIB_TEXT:
        assert frag not in body, f"exception text leaked: {frag!r} in {body[:300]}"


# ── Simulator ────────────────────────────────────────────────────────────

def test_simulator_caps_n(client):
    r = client.post("/api/simulator/generate", json={"n": 100_001, "design": "between", "params": {}})
    assert r.status_code == 422


def test_simulator_caps_factorial_cells(client):
    params = {"factors": [{"name": f"F{i}", "levels": 10} for i in range(4)]}  # 10,000 cells
    r = client.post("/api/simulator/generate", json={"n": 100, "design": "factorial", "params": params})
    assert r.status_code == 422


def test_simulator_caps_group_sizes(client):
    r = client.post("/api/simulator/generate",
                    json={"n": 100, "design": "between", "params": {"group_sizes": [10_000_000, 10]}})
    assert r.status_code == 422


def test_power_simulation_caps(client):
    base = {"design": "between", "params": {}, "test_type": "t-test"}
    r = client.post("/api/simulator/power-simulation", json={**base, "sample_sizes": list(range(10, 61))})
    assert r.status_code == 422  # 51 sample sizes
    r = client.post("/api/simulator/power-simulation", json={**base, "sample_sizes": [200_000]})
    assert r.status_code == 422  # n above 100,000
    r = client.post("/api/simulator/power-simulation",
                    json={**base, "n_simulations": 50_000, "sample_sizes": [20, 40, 60]})
    assert r.status_code == 422  # 150,000 simulations
    r = client.post("/api/simulator/power-simulation",
                    json={**base, "n_simulations": 10_000, "sample_sizes": [5_000]})
    assert r.status_code == 422  # 100M simulated values
    r = client.post("/api/simulator/power-simulation", json={**base, "n_simulations": 50, "sample_sizes": [20, 40]})
    assert r.status_code == 200, r.text


# ── Multiverse and sensitivity ──────────────────────────────────────────

def test_multiverse_refuses_huge_grid_before_building_it(client):
    decisions = [{"name": f"d{i}", "options": [{"method": "none"}] * 50} for i in range(12)]  # 50^12 specs
    t0 = time.time()
    r = client.post("/api/multiverse/run", json={
        "data": {"y": [1.0, 2.0, 3.0, 4.0], "g": [0, 0, 1, 1]}, "dv": "y", "iv": "g",
        "test_type": "t-test", "decisions": decisions})
    assert r.status_code == 422
    assert time.time() - t0 < 5
    assert "2,000" in r.text


def test_sensitivity_dedupes_options(client):
    rng = np.random.default_rng(0)
    r = client.post("/api/sensitivity-decisions/analyze", json={
        "data_type": "group_comparison",
        "group1": rng.normal(0, 1, 20).tolist(), "group2": rng.normal(0.5, 1, 20).tolist(),
        "decisions": {"outlier_method": ["none"] * 1000, "transform": ["raw"] * 1000,
                      "test_variant": ["parametric"] * 1000}})
    assert r.status_code == 200, r.text
    assert r.json()["total_specifications"] == 1


# ── Causal inference ────────────────────────────────────────────────────

def test_dag_caps_nodes_and_edges(client):
    nodes = [f"n{i}" for i in range(21)]
    r = client.post("/api/causal/dag-analysis", json={"nodes": nodes, "edges": [], "exposure": "n0", "outcome": "n1"})
    assert r.status_code == 422
    nodes = [f"n{i}" for i in range(20)]
    edges = [[f"n{i}", f"n{j}"] for i in range(20) for j in range(i + 1, 20)][:61]
    r = client.post("/api/causal/dag-analysis", json={"nodes": nodes, "edges": edges, "exposure": "n0", "outcome": "n19"})
    assert r.status_code == 422


def test_dense_dag_is_bounded(client):
    nodes = [f"n{i}" for i in range(20)]
    edges = [[f"n{i}", f"n{j}"] for i in range(20) for j in range(i + 1, 20)][:60]
    t0 = time.time()
    r = client.post("/api/causal/dag-analysis", json={"nodes": nodes, "edges": edges, "exposure": "n5", "outcome": "n19"})
    assert r.status_code in (200, 422)
    assert time.time() - t0 < 30


def test_dag_bad_edge_does_not_echo_input(client):
    secret = "<script>alert(1)</script>"
    r = client.post("/api/causal/adjustment-sets",
                    json={"nodes": ["a", "b"], "edges": [["a", "b", secret]], "exposure": "a", "outcome": "b"})
    assert r.status_code == 422
    assert secret not in r.text
    assert "[from, to] pair" in r.text
    r = client.post("/api/causal/adjustment-sets",
                    json={"nodes": ["a", "b"], "edges": [["a", secret]], "exposure": "a", "outcome": "b"})
    assert r.status_code == 422
    assert secret not in r.text


def test_g_computation_caps_bootstrap(client):
    rng = np.random.default_rng(1)
    n = 50
    body = {"outcome": rng.normal(size=n).tolist(), "treatment": [0, 1] * (n // 2),
            "covariates": [rng.normal(size=n).tolist()], "n_bootstrap": 20_000}
    r = client.post("/api/causal/g-computation", json=body)
    assert r.status_code == 422
    r = client.post("/api/causal/doubly-robust", json=body)
    assert r.status_code == 422


# ── Effect size ─────────────────────────────────────────────────────────

def test_power_from_effect_tiny_effect_is_fast(client):
    t0 = time.time()
    r = client.post("/api/effect-size/power-from-effect",
                    json={"effect_size_type": "d", "value": 0.001, "n": 100, "test_type": "two_sample_t"})
    assert r.status_code == 200, r.text
    assert time.time() - t0 < 10


def test_power_from_effect_required_n_unchanged(client):
    # d = 0.5, two-sided alpha .05: 64 per group for 80% power (G*Power), i.e. N = 128.
    r = client.post("/api/effect-size/power-from-effect",
                    json={"effect_size_type": "d", "value": 0.5, "n": 100, "test_type": "two_sample_t"})
    assert r.status_code == 200, r.text
    assert r.json()["required_n_for_80_power"] == 128


def test_first_n_matches_linear_scan():
    from app.routers.effect_size import _first_n
    f = lambda n: 1 - np.exp(-n / 5_000.0)  # monotone, crosses .8 at ~8047
    expected = next(n for n in range(4, 100000) if f(n) >= 0.8)
    assert _first_n(4, 100000, f, 0.8) == expected
    assert _first_n(4, 100000, lambda n: 0.0, 0.8) is None


# ── Machine learning ────────────────────────────────────────────────────

def _cls_data(n=60, k=3):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n, k))
    y = (X[:, 0] > 0).astype(int)
    return X.tolist(), y.tolist()


def test_ml_caps_hyperparameters(client):
    X, y = _cls_data()
    r = client.post("/api/ml/classification", json={"features": X, "labels": y, "algorithm": "random_forest",
                                                    "hyperparameters": {"n_estimators": 1_000_000}})
    assert r.status_code == 422
    r = client.post("/api/ml/classification", json={"features": X, "labels": y, "algorithm": "logistic_regression",
                                                    "hyperparameters": {"max_iter": 10_000_000}})
    assert r.status_code == 422


def test_ml_caps_cv_and_clusters(client):
    X, y = _cls_data()
    r = client.post("/api/ml/cross-validation", json={"features": X, "target": y, "n_splits": 1000})
    assert r.status_code == 422
    r = client.post("/api/ml/cross-validation", json={"features": X, "target": y, "cv_strategy": "repeated",
                                                      "n_repeats": 1000})
    assert r.status_code == 422
    r = client.post("/api/ml/clustering", json={"features": X, "n_clusters": 10_000})
    assert r.status_code == 422


def test_ml_bad_features_message_is_generic(client):
    r = client.post("/api/ml/classification", json={"features": [[1, 2], [3]], "labels": [0, 1]})
    assert r.status_code == 400
    _no_lib_text(r)


# ── Text analysis ───────────────────────────────────────────────────────

def test_content_coding_patterns_are_literal(client):
    t0 = time.time()
    r = client.post("/api/text-analysis/content-coding", json={
        "texts": ["a" * 5_000 + "!", "a+b (x) here"],
        "coding_scheme": {"evil": ["(a+)+$"], "plus": ["a+b"]}})
    assert r.status_code == 200, r.text
    assert time.time() - t0 < 5
    out = r.json()
    assert out["coding_matrix"] == [[0, 0], [0, 1]]  # "a+b" matched literally, the regex never ran


def test_content_coding_caps(client):
    r = client.post("/api/text-analysis/content-coding",
                    json={"texts": ["x"], "coding_scheme": {"c": [f"k{i}" for i in range(201)]}})
    assert r.status_code == 422
    r = client.post("/api/text-analysis/content-coding", json={"texts": ["x"], "coding_scheme": {"c": ["k" * 201]}})
    assert r.status_code == 422


def test_text_analysis_caps(client):
    r = client.post("/api/text-analysis/topic-model", json={"texts": ["a b", "c d"], "n_topics": 1000})
    assert r.status_code == 422
    r = client.post("/api/text-analysis/word-frequency",
                    json={"texts": ["hello world"], "target_words": [f"w{i}" for i in range(51)]})
    assert r.status_code == 422
    r = client.post("/api/text-analysis/word-frequency", json={"texts": ["hello world"], "top_n": 10**9})
    assert r.status_code == 422


# ── Time series ─────────────────────────────────────────────────────────

def test_time_series_caps(client):
    y = np.sin(np.arange(100) / 3.0).tolist()
    r = client.post("/api/time-series/arima", json={"values": y, "order": [50, 0, 0]})
    assert r.status_code == 422
    r = client.post("/api/time-series/var", json={"series": {"a": y, "b": y[::-1]}, "max_lag": 1000})
    assert r.status_code == 422
    r = client.post("/api/time-series/stationarity", json={"values": y, "max_diff": 100})
    assert r.status_code == 422


def test_binseg_many_changepoints_no_recursion_error(client):
    y = np.repeat(np.arange(2_000) % 2 * 100.0, 5) + np.random.default_rng(3).normal(0, 0.1, 10_000)
    r = client.post("/api/time-series/changepoint",
                    json={"values": y.tolist(), "method": "binseg", "min_segment_length": 2})
    assert r.status_code == 200, r.text


def test_binseg_order_matches_recursive():
    from app.routers import time_series as T

    def recursive(y, start, end, pen, m, out):
        cp = T._binseg_split(y, start, end, pen, m)
        if cp is not None:
            out.append(cp)
            recursive(y, start, cp, pen, m, out)
            recursive(y, cp, end, pen, m, out)

    y = np.concatenate([np.zeros(30), np.ones(30) * 5, np.zeros(30), np.ones(30) * -4]) + \
        np.random.default_rng(4).normal(0, 0.3, 120)
    pen = 2 * np.log(len(y))
    ref = []
    recursive(y, 0, len(y), pen, 5, ref)
    assert T._binary_segmentation(y, pen, 5) == ref


def test_decomposition_unknown_method_passes_through(client):
    r = client.post("/api/time-series/decomposition",
                    json={"values": list(range(40)), "period": 4, "method": "nope"})
    assert r.status_code == 400
    assert "Unknown method" in r.json()["detail"]


# ── Political science ───────────────────────────────────────────────────

def test_ideal_point_agreement_matches_pairwise_loop(client):
    rng = np.random.default_rng(5)
    V = rng.integers(-1, 2, size=(25, 12))
    r = client.post("/api/political-science/ideal-point", json={"vote_matrix": V.tolist(), "dimensions": 1})
    assert r.status_code == 200, r.text
    got = np.array([p["position"] for p in r.json()["legislator_positions"]])

    # Reference: the original pair-by-pair agreement matrix
    mask = V >= 0
    W = np.where(mask, V, np.nan)
    n = V.shape[0]
    agree = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            both = mask[i] & mask[j]
            a = np.sum(W[i, both] == W[j, both]) / np.sum(both) if np.sum(both) > 0 else 0.5
            agree[i, j] = agree[j, i] = a
    B = agree - agree.mean(axis=1, keepdims=True) - agree.mean(axis=0, keepdims=True) + agree.mean()
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    expected = vecs[:, idx[0]] * np.sqrt(max(vals[idx[0]], 0))
    np.testing.assert_array_equal(got, expected)


def test_political_science_caps(client):
    r = client.post("/api/political-science/ideal-point", json={"vote_matrix": [[1, 0, 1]] * 2_001})
    assert r.status_code == 422
    r = client.post("/api/political-science/vote-scaling", json={"responses": [[1, 0] * 101] * 10})
    assert r.status_code == 422


# ── Power analysis ──────────────────────────────────────────────────────

def test_power_unsolvable_has_no_scipy_text(client):
    r = client.post("/api/power/t-test", json={"effect_size": 0.0001, "alpha": 0.05, "power": 0.99})
    if r.status_code == 422 and "Could not solve" in r.text:
        _no_lib_text(r)
    r = client.post("/api/power/t-test", json={"alpha": 0.05, "power": 0.9999, "n": 2})
    assert r.status_code in (200, 422)
    _no_lib_text(r)
