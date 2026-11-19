"""Security limits for the group-D routers: request-size caps (422 instead of unbounded work),
no exception text or raw user input in error responses, and no blocking async handlers."""

import inspect

import numpy as np
import pytest

SECRET = "SECRET_INTERNAL_DETAIL_/srv/app/x.py"


def _detail(resp) -> str:
    return str(resp.json().get("detail"))


# ── Denial-of-service caps ────────────────────────────────────────────────

def test_roc_rating_span_capped(client):
    r = client.post("/api/signal-detection/roc", json={"signal_ratings": [0, 10**12], "noise_ratings": [0, 1]})
    assert r.status_code == 422
    assert "1,000" in _detail(r)


def test_sdt_conditions_capped(client):
    cond = {"hits": 5, "misses": 5, "false_alarms": 5, "correct_rejections": 5}
    r = client.post("/api/signal-detection/multi-condition", json={"conditions": [cond] * 201})
    assert r.status_code == 422


def test_d_study_grid_capped(client):
    vc = {"person": 1, "item": 0.5, "occasion": 0.2, "person_x_item": 0.3, "person_x_occasion": 0.2,
          "item_x_occasion": 0.1, "residual": 0.5}
    r = client.post("/api/generalizability/d-study", json={
        "variance_components": vc, "design": "p_x_i_x_o",
        "proposed_counts": {"n_items": list(range(1, 201)), "n_occasions": list(range(1, 201))}})
    assert r.status_code == 422
    assert "10,000" in _detail(r)


def test_power_curve_lists_capped(client):
    r = client.post("/api/study-planner/power-curve", json={"effect_sizes": [0.5] * 101, "sample_sizes": [10]})
    assert r.status_code == 422


def test_anova_groups_capped(client):
    r = client.post("/api/stats/anova", json={"groups": [[1.0, 2.0, 3.0]] * 101})
    assert r.status_code == 422


def test_regression_predictors_capped(client):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(250, 201)).tolist()
    r = client.post("/api/stats/regression", json={"x": X, "y": rng.normal(size=250).tolist()})
    assert r.status_code == 422


def test_what_would_it_take_n_capped(client):
    r = client.post("/api/what-would-it-take/analyze", json={"test_type": "ttest", "observed_es": 0.3, "n": 10**12})
    assert r.status_code == 422


def test_reviewer2_sample_size_capped(client):
    r = client.post("/api/reviewer2/analyze", json={"test_type": "ttest", "results": {"p_value": 0.03},
                                                    "sample_size": 10**12})
    assert r.status_code == 422


def test_missingness_columns_capped(client):
    data = {f"c{i}": [1.0, None, 2.0] for i in range(101)}
    r = client.post("/api/missingness/diagnose", json={"data": data})
    assert r.status_code == 422


def test_missingness_cells_capped(client):
    data = {"a": list(range(100_000)), **{f"e{i}": [] for i in range(30)}}
    r = client.post("/api/missingness/diagnose", json={"data": data})
    assert r.status_code == 422


def test_imputation_methods_capped(client):
    r = client.post("/api/missingness/imputation-comparison", json={"data": [1, None, 2], "methods": ["mean"] * 11})
    assert r.status_code == 422


def test_q_sorts_capped(client):
    rng = np.random.default_rng(1)
    sorts = [rng.permutation(10).tolist() for _ in range(501)]
    r = client.post("/api/q-methodology/correlate", json={"q_sorts": sorts})
    assert r.status_code == 422


def test_q_interpret_factor_count_capped(client):
    r = client.post("/api/q-methodology/interpret", json={
        "factor_arrays": {f"F{i}": [0.0] * 5 for i in range(21)}, "statement_labels": list("abcde"), "n_sorts": 5})
    assert r.status_code == 422


def test_ca_table_capped(client):
    r = client.post("/api/correspondence/simple", json={"table": [[1, 2]] * 1001})
    assert r.status_code == 422


def test_mca_observations_capped(client):
    data = [["a", "x"], ["b", "y"]] * 2501
    r = client.post("/api/correspondence/multiple", json={"data": data})
    assert r.status_code == 422
    assert "5,000" in _detail(r)


def test_ca_dimensions_must_be_positive(client):
    r = client.post("/api/correspondence/simple", json={"table": [[1, 2], [3, 4]], "n_dimensions": -3})
    assert r.status_code == 422


def test_triangulation_bootstrap_work_capped(client):
    x = np.arange(5000.0)
    r = client.post("/api/triangulation/analyze", json={"data_type": "correlation", "x": x.tolist(),
                                                          "y": (x * 0.5 + np.sin(x)).tolist()})
    assert r.status_code == 422
    assert "4,000" in _detail(r)


def test_assumption_gate_manova_capped(client):
    rng = np.random.default_rng(2)
    data = {"a": rng.normal(size=5001).tolist(), "b": rng.normal(size=5001).tolist()}
    r = client.post("/api/assumption-gate/check", json={"test_type": "manova", "data": data})
    assert r.status_code == 422


def test_assumption_gate_alternatives_resampling_capped(client):
    rng = np.random.default_rng(3)
    data = {"group1": rng.normal(size=6000).tolist(), "group2": rng.normal(size=6000).tolist()}
    r = client.post("/api/assumption-gate/run-with-alternatives", json={"test_type": "t-test", "data": data})
    assert r.status_code == 422


def test_assumption_gate_regression_large_n_uses_no_hat_matrix(client):
    # 20,000 cases: building the n x n hat matrix would need 3.2 GB; leverages are computed row-wise.
    rng = np.random.default_rng(4)
    x = rng.normal(size=20_000)
    data = {"x": x.tolist(), "y": (2 * x + rng.normal(size=20_000)).tolist()}
    r = client.post("/api/assumption-gate/check", json={"test_type": "regression", "data": data,
                                                         "params": {"dv": "y", "ivs": ["x"]}})
    assert r.status_code == 200


def test_leverage_matches_hat_matrix():
    rng = np.random.default_rng(5)
    Xc = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    inv = np.linalg.inv(Xc.T @ Xc)
    h_old = np.diag(Xc @ inv @ Xc.T)
    h_new = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
    assert np.allclose(h_old, h_new, rtol=0, atol=1e-12)


def test_nl_query_length_capped(client):
    r = client.post("/api/nl-pipeline/parse-query", json={"query": "before " * 400})
    assert r.status_code == 422


def test_nl_variable_name_length_capped(client):
    r = client.post("/api/nl-pipeline/parse-query", json={"query": "compare groups",
                                                           "available_variables": ["a" * 201]})
    assert r.status_code == 422


def test_ema_lagged_variables_capped(client):
    obs = [{"participant_id": "p1", "timestamp": t, "variables": {"v0": float(t)}} for t in range(5)]
    r = client.post("/api/ema/lagged", json={"observations": obs, "variables": [f"v{i}" for i in range(51)]})
    assert r.status_code == 422


def test_mtmm_cfa_variables_capped(client):
    names = [f"v{i}" for i in range(31)]
    r = client.post("/api/mtmm/cfa", json={
        "correlation_matrix": np.eye(31).tolist(), "variable_names": names, "trait_labels": ["t"],
        "method_labels": ["m"], "variable_trait_map": {n: "t" for n in names},
        "variable_method_map": {n: "m" for n in names}, "sample_size": 100})
    assert r.status_code == 422


def test_factor_analysis_n_factors_capped(client):
    r = client.post("/api/psychometrics/factor-analysis", json={"data": [[1, 2, 3]] * 10, "n_factors": 51})
    assert r.status_code == 422


def test_sem_variable_count_capped(client):
    factors = {f"F{i}": [f"x{i}_{j}" for j in range(3)] for i in range(21)}  # 63 indicators
    rows = [{f"x{i}_{j}": float(k + i + j) for i in range(21) for j in range(3)} for k in range(5)]
    r = client.post("/api/sem/cfa", json={"data": rows, "factors": factors})
    assert r.status_code == 422
    assert "60" in _detail(r)


def test_sem_mediators_capped(client):
    r = client.post("/api/sem/path-analysis", json={"data": [{"a": 1, "b": 2}], "paths": [{"from": "a", "to": "b"}],
                                                     "mediators": [f"m{i}" for i in range(11)]})
    assert r.status_code == 422


def test_invariance_groups_capped(client):
    g = [[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]]
    r = client.post("/api/invariance/configural", json={"data_groups": [g] * 21, "factor_model": {"F": [0, 1, 2]}})
    assert r.status_code == 422


def test_diagnostics_predictors_capped(client):
    preds = [f"p{i}" for i in range(201)]
    r = client.post("/api/diagnostics/multicollinearity", json={"predictors": preds,
                                                                  "data": {p: [1.0, 2.0] for p in preds}})
    assert r.status_code == 422


def test_dual_report_variable_count_capped(client):
    data = {f"g{i}": [1.0, 2.0, 3.0] for i in range(201)}
    r = client.post("/api/dual-report/dual-test", json={"test_type": "anova", "data": data})
    assert r.status_code == 422


def test_biometrics_sampling_rate_capped(client):
    r = client.post("/api/biometrics/eda-analysis", json={"eda_signal": [1.0] * 20, "sampling_rate": 1e308})
    assert r.status_code == 422


def test_outlier_clean_data_is_linear(client):
    # A negative threshold flags every value; the clean-data filter used to be O(n^2) here.
    import time
    t0 = time.perf_counter()
    r = client.post("/api/data/outliers", json={"data": list(range(60_000)), "method": "zscore", "threshold": -1})
    assert r.status_code == 200
    assert r.json()["n_outliers"] == 60_000
    assert time.perf_counter() - t0 < 20


# ── No exception text or raw input in errors ─────────────────────────────

def test_dual_report_error_has_no_exception_text(client):
    data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 1.0, 4.0, 3.0, 5.0]}
    r = client.post("/api/dual-report/dual-test", json={"test_type": "correlation", "data": data,
                                                          "params": {"kappa": SECRET}})
    assert r.status_code == 422
    assert SECRET not in r.text and "could not convert" not in r.text


def test_dual_report_unknown_type_not_echoed(client):
    r = client.post("/api/dual-report/dual-test", json={"test_type": SECRET, "data": {"a": [1, 2, 3]}})
    assert r.status_code == 422
    assert SECRET not in r.text


def test_assumption_gate_error_is_generic(client):
    data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 5.0, 9.0]}
    r = client.post("/api/assumption-gate/check", json={"test_type": "regression", "data": data,
                                                         "params": {"dv": "y", "ivs": [SECRET]}})
    assert r.status_code == 422
    assert SECRET not in r.text


def test_triangulation_robust_failure_label_is_generic(client, monkeypatch):
    import statsmodels.api as sm

    def boom(*a, **k):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(sm, "RLM", boom)
    rng = np.random.default_rng(6)
    x = rng.normal(size=40)
    r = client.post("/api/triangulation/analyze", json={"data_type": "regression", "x": x.tolist(),
                                                          "y": (x + rng.normal(size=40)).tolist()})
    assert r.status_code == 200
    assert SECRET not in r.text
    assert any("unavailable for these data" in m["method"] for m in r.json()["methods"])


def test_gonogo_trial_type_not_echoed(client):
    r = client.post("/api/cognitive/gonogo", json={"trials": [{"trial_type": SECRET, "responded": True}]})
    assert r.status_code == 422
    assert SECRET not in r.text


@pytest.mark.parametrize("path,body", [
    ("/api/study-planner/sample-size", {"test_type": SECRET, "effect_size": 0.5}),
    ("/api/study-planner/effect-size-convert", {"value": 0.5, "from_type": SECRET, "to_type": "r"}),
    ("/api/stats/power-analysis", {"effect_size": 0.5, "test_type": SECRET}),
    ("/api/data/transform", {"data": [1.0, 2.0, 3.0], "transform": SECRET}),
    ("/api/data/outliers", {"data": [1.0, 2.0, 3.0, 4.0], "method": SECRET}),
])
def test_unknown_options_not_echoed(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code in (400, 422)
    assert SECRET not in r.text


def test_apa_formatter_non_numeric_is_422(client):
    r = client.post("/api/apa/format", json={"test_type": "ttest", "results": {"t_statistic": "abc"}})
    assert r.status_code == 422
    assert "abc" not in _detail(r)


def test_long_column_names_truncated(client):
    long = "Z" * 500
    r = client.post("/api/diagnostics/residual-diagnostics", json={"dv": "y", "ivs": [long],
                                                                     "data": {"y": [1.0, 2.0, 3.0, 4.0]}})
    assert r.status_code == 422
    assert long not in r.text


# ── Event loop ────────────────────────────────────────────────────────────

def test_cpu_handlers_are_sync():
    from app.routers import (assumption_gate, diagnostics, dual_report, equivalence, interpretation_confidence,
                             narrative, preregistration, reviewer2, triangulation, what_would_it_take)
    handlers = [assumption_gate.check_assumptions, assumption_gate.run_with_alternatives, diagnostics.qq_plot,
                diagnostics.residual_diagnostics, diagnostics.homogeneity_plot, diagnostics.multicollinearity,
                diagnostics.outlier_diagnostics, dual_report.dual_test, dual_report.dual_report_apa,
                equivalence.analyze_equivalence, interpretation_confidence.compute_ppv, narrative.generate_narrative,
                preregistration.generate_preregistration, preregistration.preregistration_checklist,
                reviewer2.adversarial_review, triangulation.triangulate, what_would_it_take.what_would_it_take]
    assert not any(inspect.iscoroutinefunction(h) for h in handlers)
