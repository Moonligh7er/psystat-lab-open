"""
Medical Stats (/api/medical/*) against published values and closed forms.

Kaplan–Meier / log-rank: Freireich et al. (1963) 6-MP leukaemia remission data as
tabulated in Kleinbaum & Klein, Survival Analysis (3rd ed.), ch. 2.
"""

import math

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

# Remission times (weeks); 0 = censored
MP6_T = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
MP6_E = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
PLC_T = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
PLC_E = [1] * 21

# Kleinbaum & Klein Table: KM estimates for the 6-MP group
MP6_KM = {6: 0.857, 7: 0.807, 10: 0.753, 13: 0.690, 16: 0.627, 22: 0.538, 23: 0.448}


@pytest.mark.validation
def test_kaplan_meier_freireich(client, record_validation):
    r = client.post("/api/medical/kaplan-meier", json={
        "durations": MP6_T + PLC_T, "events": MP6_E + PLC_E,
        "group_labels": ["6-MP"] * 21 + ["placebo"] * 21})
    assert r.status_code == 200, r.text
    j = r.json()
    g = j["groups"]["6-MP"]
    curve = dict(zip(g["timeline"], g["survival_probability"]))
    for t, s in MP6_KM.items():
        assert curve[t] == pytest.approx(s, abs=5e-4)
    assert g["median_survival"] == 23 and j["groups"]["placebo"]["median_survival"] == 8
    assert j["log_rank_test"]["test_statistic"] == pytest.approx(16.79, abs=0.01)
    assert j["log_rank_test"]["df"] == 1
    record_validation(
        method="Kaplan–Meier with log-rank test", endpoint="/api/medical/kaplan-meier",
        statistic="log-rank chi-square", observed=float(j["log_rank_test"]["test_statistic"]), expected=16.79,
        reference="Kleinbaum & Klein, Survival Analysis (2012), Freireich 6-MP data", reference_kind="analytical",
        dataset="Freireich et al. (1963) leukaemia remission", tolerance_abs=0.01,
    )


def test_rmst_is_step_area(client):
    j = client.post("/api/medical/kaplan-meier", json={"durations": MP6_T, "events": MP6_E}).json()
    t = j["survival_table"]["timeline"]
    s = j["survival_table"]["survival_probability"]
    area = sum(s[i] * (t[i + 1] - t[i]) for i in range(len(t) - 1))
    assert j["restricted_mean_survival"] == pytest.approx(area, rel=1e-9)
    assert j["rmst_horizon"] == 35


def test_three_group_log_rank_has_two_df(client):
    j = client.post("/api/medical/kaplan-meier", json={
        "durations": MP6_T + PLC_T + [3, 5, 9, 14, 20, 26], "events": MP6_E + PLC_E + [1, 1, 0, 1, 1, 0],
        "group_labels": ["a"] * 21 + ["b"] * 21 + ["c"] * 6}).json()
    assert j["log_rank_test"]["df"] == 2 and list(j["groups"]) == ["a", "b", "c"]


def test_kaplan_meier_rejects_mismatched_labels(client):
    r = client.post("/api/medical/kaplan-meier", json={"durations": [1, 2, 3], "events": [1, 0, 1], "group_labels": ["a", "b"]})
    assert r.status_code == 422 and "group labels" in r.text


@pytest.mark.validation
def test_nnt_altman_interval(client, record_validation):
    # 15/100 vs 30/100: ARR = .15, CI excludes 0
    j = client.post("/api/medical/nnt", json={"events_treatment": 15, "n_treatment": 100, "events_control": 30, "n_control": 100}).json()
    se = math.sqrt(.15 * .85 / 100 + .3 * .7 / 100)
    lo, hi = .15 - 1.959964 * se, .15 + 1.959964 * se
    assert j["absolute_risk_reduction"] == pytest.approx(.15)
    assert j["nnt_rounded"] == 7  # 1/.15 = 6.67, rounded up
    assert j["nnt_ci_95"] == pytest.approx([1 / hi, 1 / lo], rel=1e-5)
    record_validation(
        method="Number needed to treat (Altman CI)", endpoint="/api/medical/nnt",
        statistic="NNT", observed=float(j["nnt"]), expected=1 / .15,
        reference="Altman (1998) BMJ 317:1309, closed form", reference_kind="analytical",
        dataset="15/100 vs 30/100", tolerance_rel=1e-9,
    )
    # 20/100 vs 25/100: CI for ARR includes 0 -> NNTB .. inf .. NNTH
    j = client.post("/api/medical/nnt", json={"events_treatment": 20, "n_treatment": 100, "events_control": 25, "n_control": 100}).json()
    assert j["nnt_ci_95"] is None and "∞" in j["nnt_ci_text"] and "NNTH" in j["nnt_ci_text"]


def test_risk_ratio_closed_form_and_zero_cell(client):
    j = client.post("/api/medical/risk-ratio", json={"exposed_events": 40, "exposed_total": 200, "unexposed_events": 20, "unexposed_total": 200}).json()
    assert j["risk_ratio"] == pytest.approx(2.0) and j["odds_ratio"] == pytest.approx((40 * 180) / (160 * 20))
    se = math.sqrt(1 / 40 - 1 / 200 + 1 / 20 - 1 / 200)
    assert j["rr_ci_95"] == pytest.approx([2 * math.exp(-1.959964 * se), 2 * math.exp(1.959964 * se)], rel=1e-5)
    assert not j["zero_cell_correction"]
    z = client.post("/api/medical/risk-ratio", json={"exposed_events": 5, "exposed_total": 50, "unexposed_events": 0, "unexposed_total": 50}).json()
    assert z["zero_cell_correction"] and z["risk_ratio"] == pytest.approx((5.5 / 51) / (0.5 / 51))
    assert z["rr_ci_95"][0] < z["risk_ratio"] < z["rr_ci_95"][1]


@pytest.mark.validation
def test_diagnostic_accuracy_wilson(client, record_validation):
    j = client.post("/api/medical/diagnostic-accuracy", json={"true_positives": 85, "false_positives": 10, "true_negatives": 90, "false_negatives": 15}).json()
    assert j["sensitivity"] == pytest.approx(85 / 100) and j["specificity"] == pytest.approx(90 / 100)
    ref = proportion_confint(85, 100, method="wilson")
    assert j["sensitivity_ci_95"] == pytest.approx(list(ref), abs=1e-6)
    assert j["lr_positive"] == pytest.approx(0.85 / 0.10)
    record_validation(
        method="Diagnostic accuracy (Wilson CIs)", endpoint="/api/medical/diagnostic-accuracy",
        statistic="sensitivity CI lower", observed=float(j["sensitivity_ci_95"][0]), expected=float(ref[0]),
        reference="statsmodels proportion_confint(method='wilson')", reference_kind="scipy-ref",
        dataset="TP 85, FP 10, TN 90, FN 15", tolerance_abs=1e-6,
    )
    # No negatives: specificity undefined -> null, not 0
    j = client.post("/api/medical/diagnostic-accuracy", json={"true_positives": 5, "false_positives": 0, "true_negatives": 0, "false_negatives": 1}).json()
    assert j["specificity"] is None and j["lr_positive"] is None and j["youdens_j"] is None


def test_count_validation(client):
    r = client.post("/api/medical/nnt", json={"events_treatment": 120, "n_treatment": 100, "events_control": 30, "n_control": 100})
    assert r.status_code == 422
