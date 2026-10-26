"""
Cognitive Lab scoring (/api/cognitive/*): RT effects against statsmodels' Welch
test, d′ against the log-linear formula (Hautus, 1995), and input validation.
"""

import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW


def _rt_data(seed=0, n=40, effect=100):
    rng = np.random.default_rng(seed)
    cong = (480 + 60 * rng.normal(size=n) + rng.exponential(80, n)).round().tolist()
    incong = (480 + effect + 60 * rng.normal(size=n) + rng.exponential(80, n)).round().tolist()
    cong_ok = (rng.random(n) > 0.05).tolist()
    incong_ok = (rng.random(n) > 0.12).tolist()
    return cong, incong, cong_ok, incong_ok


@pytest.mark.validation
@pytest.mark.parametrize("task,effect_key", [("stroop", "interference_ms"), ("flanker", "flanker_effect_ms")])
def test_rt_effect_uses_correct_trials_and_welch(client, record_validation, task, effect_key):
    cong, incong, cong_ok, incong_ok = _rt_data()
    r = client.post(f"/api/cognitive/{task}", json={
        "congruent_rts": cong, "incongruent_rts": incong, "congruent_correct": cong_ok, "incongruent_correct": incong_ok})
    assert r.status_code == 200, r.text
    j = r.json()
    a = np.array(incong)[np.array(incong_ok)]
    b = np.array(cong)[np.array(cong_ok)]
    t, p, df = CompareMeans(DescrStatsW(a), DescrStatsW(b)).ttest_ind(usevar="unequal")
    assert j[effect_key] == pytest.approx(a.mean() - b.mean(), abs=1e-9)
    assert j["t_statistic"] == pytest.approx(t, rel=1e-9) and j["df"] == pytest.approx(df, rel=1e-9)
    assert j["p_value"] == pytest.approx(p, rel=1e-8)
    assert j["n_correct_incongruent"] == int(np.sum(incong_ok))
    record_validation(
        method=f"Cognitive Lab: {task} RT effect (Welch t)", endpoint=f"/api/cognitive/{task}",
        statistic="t", observed=float(j["t_statistic"]), expected=float(t),
        reference="statsmodels CompareMeans.ttest_ind(usevar='unequal') on correct trials", reference_kind="scipy-ref",
        dataset="simulated ex-Gaussian RTs, 40 trials per condition", tolerance_rel=1e-9,
    )


def test_task_switching_costs(client):
    sw, rp, sw_ok, rp_ok = _rt_data(seed=3, effect=90)
    j = client.post("/api/cognitive/task-switching", json={
        "switch_rts": rp, "repeat_rts": sw, "switch_correct": rp_ok, "repeat_correct": sw_ok}).json()
    a, b = np.array(rp)[np.array(rp_ok)], np.array(sw)[np.array(sw_ok)]
    assert j["switch_cost_ms"] == pytest.approx(a.mean() - b.mean())
    assert j["accuracy_switch_cost"] == pytest.approx(np.mean(sw_ok) - np.mean(rp_ok))


@pytest.mark.validation
def test_nback_dprime_log_linear(client, record_validation):
    # 12 targets (9 hits), 30 non-targets (4 false alarms)
    trials = ([{"is_target": True, "responded": True, "rt": 500}] * 9 + [{"is_target": True, "responded": False, "rt": None}] * 3
              + [{"is_target": False, "responded": True, "rt": 450}] * 4 + [{"is_target": False, "responded": False, "rt": None}] * 26)
    j = client.post("/api/cognitive/nback", json={"responses": trials}).json()
    H, F = (9 + 0.5) / (12 + 1), (4 + 0.5) / (30 + 1)
    assert j["d_prime"] == pytest.approx(norm.ppf(H) - norm.ppf(F), abs=1e-12)
    assert j["criterion"] == pytest.approx(-0.5 * (norm.ppf(H) + norm.ppf(F)), abs=1e-12)
    assert j["hit_rate"] == pytest.approx(9 / 12) and j["false_alarm_rate"] == pytest.approx(4 / 30)
    record_validation(
        method="Cognitive Lab: N-Back d′ (log-linear)", endpoint="/api/cognitive/nback",
        statistic="d_prime", observed=float(j["d_prime"]), expected=float(norm.ppf(H) - norm.ppf(F)),
        reference="Hautus (1995) log-linear correction, closed form", reference_kind="analytical",
        dataset="9/12 hits, 4/30 false alarms", tolerance_abs=1e-12,
    )


def test_gonogo_perfect_performance_stays_finite(client):
    trials = [{"trial_type": "go", "responded": True, "rt": 300}] * 20 + [{"trial_type": "nogo", "responded": False, "rt": None}] * 10
    j = client.post("/api/cognitive/gonogo", json={"trials": trials}).json()
    assert j["d_prime"] == pytest.approx(norm.ppf(20.5 / 21) - norm.ppf(0.5 / 11))
    assert j["commission_errors"] == 0 and j["omission_rate"] == 0


def test_too_few_correct_trials_reports_not_computable(client):
    j = client.post("/api/cognitive/stroop", json={
        "congruent_rts": [500, 510], "incongruent_rts": [600, 620],
        "congruent_correct": [True, False], "incongruent_correct": [True, True]}).json()
    assert j["t_statistic"] is None and j["p_value"] is None and j["significant_interference"] is None
    assert j["interference_ms"] == pytest.approx(110)


def test_mismatched_lengths_and_bad_values_are_422(client):
    r = client.post("/api/cognitive/flanker", json={
        "congruent_rts": [500, 510], "incongruent_rts": [600], "congruent_correct": [True], "incongruent_correct": [True]})
    assert r.status_code == 422 and "correctness flags" in r.text
    r = client.post("/api/cognitive/stroop", json={
        "congruent_rts": [-5], "incongruent_rts": [600], "congruent_correct": [True], "incongruent_correct": [True]})
    assert r.status_code == 422
    r = client.post("/api/cognitive/gonogo", json={"trials": [{"trial_type": "stop", "responded": True}]})
    assert r.status_code == 422
    assert client.post("/api/cognitive/nback", json={"responses": []}).status_code == 422
