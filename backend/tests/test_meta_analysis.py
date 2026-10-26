"""
Meta-analysis (/api/meta-analysis/*) against metafor 5.2.1 on the BCG vaccine data
(dat.bcg; log risk ratios from escalc(measure="RR")).

R code that reproduces every reference value:

    library(metafor)
    dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
    rma(yi, vi, data=dat, method="FE"); rma(yi, vi, data=dat, method="DL")
    re <- rma(yi, vi, data=dat); confint(re, control=list(tol=1e-12))
    rma(yi, vi, data=dat, test="knha")
    fe <- rma(yi, vi, data=dat, method="FE")
    regtest(fe, model="lm", predictor="sei"); ranktest(fe); trimfill(fe); fsn(yi, vi, data=dat)
    rma(yi, vi, mods=~ablat, data=dat, method="DL"); rma(yi, vi, mods=~ablat, data=dat, method="FE")
    # mixed-effects subgroup test: DL within each alloc group, then rma(est, sei=se, method="FE")$QE
"""

import asyncio
import math

import numpy as np
import pytest

from app.routers import meta_analysis as M

YI = [-0.889311333920, -1.585388657201, -1.348073148300, -1.441551190021, -0.217547322211, -0.786115585819,
      -1.620898223598, 0.011952333524, -0.469417648738, -1.371344803473, -0.339358828338, 0.445913400571,
      -0.017313948217]
VI = [0.325584765004, 0.194581121398, 0.415367965368, 0.020010031902, 0.051210172170, 0.006905618456,
      0.223017247572, 0.003961579298, 0.056434210463, 0.073024793613, 0.012412213972, 0.532505845200,
      0.071404659684]
SEI = [math.sqrt(v) for v in VI]
ABLAT = [44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33]
ALLOC = ["random"] * 4 + ["alternate"] * 2 + ["random"] * 3 + ["systematic"] * 4
REF = "metafor 5.2.1 (R 4.5.2)"
DATA = "BCG vaccine trials (dat.bcg), log risk ratios"

run = asyncio.run


def rec(record, method, endpoint, stat, observed, expected, tol=1e-5):
    assert observed == pytest.approx(expected, abs=tol), f"{method} {stat}: {observed} vs {expected}"
    record(method=method, endpoint=endpoint, statistic=stat, observed=float(observed), expected=float(expected),
           reference=REF, reference_kind="r", dataset=DATA, tolerance_abs=tol)


@pytest.mark.validation
def test_fixed_effect(record_validation):
    r = run(M.fixed_effects_meta(M.FixedEffectsRequest(effect_sizes=YI, standard_errors=SEI)))
    e = "/api/meta-analysis/fixed-effects"
    rec(record_validation, "Fixed-effect meta-analysis", e, "estimate", r["pooled_effect"], -0.4302851637)
    rec(record_validation, "Fixed-effect meta-analysis", e, "SE", r["pooled_se"], 0.0404987517)
    rec(record_validation, "Fixed-effect meta-analysis", e, "Q", r["heterogeneity"]["Q"], 152.2330080824)
    assert r["prediction_interval"] is None


@pytest.mark.validation
def test_random_effects_dl(record_validation):
    r = run(M.random_effects_meta(M.RandomEffectsRequest(effect_sizes=YI, standard_errors=SEI)))
    e = "/api/meta-analysis/random-effects"
    m = "Random-effects meta-analysis (DerSimonian–Laird)"
    rec(record_validation, m, e, "estimate", r["pooled_effect"], -0.7141172221)
    rec(record_validation, m, e, "SE", r["pooled_se"], 0.1787420895)
    rec(record_validation, m, e, "tau^2", r["tau_squared"], 0.3087602629)
    rec(record_validation, m, e, "I^2", r["heterogeneity"]["I_squared"], 92.1173468546, tol=0.005)


@pytest.mark.validation
def test_random_effects_reml_qprofile_and_knha(record_validation):
    r = run(M.random_effects_meta(M.RandomEffectsRequest(effect_sizes=YI, standard_errors=SEI, use_reml=True)))
    e = "/api/meta-analysis/random-effects"
    m = "Random-effects meta-analysis (REML)"
    rec(record_validation, m, e, "estimate", r["pooled_effect"], -0.7145323484)
    rec(record_validation, m, e, "SE", r["pooled_se"], 0.1797815318)
    rec(record_validation, m, e, "tau^2", r["tau_squared"], 0.3132433260)
    rec(record_validation, m, e, "I^2", r["heterogeneity"]["I_squared"], 92.2213860750, tol=0.005)
    rec(record_validation, "tau^2 Q-profile CI", e, "lower", r["tau_squared_CI"][0], 0.1197183611)
    rec(record_validation, "tau^2 Q-profile CI", e, "upper", r["tau_squared_CI"][1], 1.1114790841)
    kh = run(M.random_effects_meta(M.RandomEffectsRequest(effect_sizes=YI, standard_errors=SEI, use_reml=True, knapp_hartung=True)))
    rec(record_validation, "Knapp–Hartung adjustment", e, "SE", kh["pooled_se"], 0.1807917455)
    rec(record_validation, "Knapp–Hartung adjustment", e, "CI lower", kh["ci"][0], -1.1084437230)
    assert kh["test_df"] == 12 and kh["p"] == pytest.approx(1.9200150851e-03, rel=1e-4)
    # Leave-one-out uses the same estimator as the main model
    assert all(x["tau_squared"] is not None for x in r["leave_one_out"])


def test_prediction_interval_is_higgins_t():
    r = run(M.random_effects_meta(M.RandomEffectsRequest(effect_sizes=YI, standard_errors=SEI, use_reml=True)))
    from scipy import stats
    half = stats.t.ppf(0.975, 11) * math.sqrt(r["tau_squared"] + r["pooled_se"] ** 2)
    assert r["prediction_interval"] == pytest.approx([r["pooled_effect"] - half, r["pooled_effect"] + half], abs=1e-5)


@pytest.mark.validation
def test_publication_bias(record_validation):
    r = run(M.publication_bias(M.PublicationBiasRequest(effect_sizes=YI, standard_errors=SEI)))
    e = "/api/meta-analysis/publication-bias"
    rec(record_validation, "Egger regression test", e, "t", r["egger"]["t"], -1.4012820023, tol=1e-4)
    rec(record_validation, "Egger regression test", e, "p", r["egger"]["p"], 0.1887069951)
    rec(record_validation, "Begg & Mazumdar rank test", e, "Kendall tau", r["begg"]["kendalls_tau"], 0.0256410256)
    rec(record_validation, "Begg & Mazumdar rank test", e, "p", r["begg"]["p"], 0.9523619080)
    tf = r["trim_and_fill"]
    assert tf["imputed_studies"] == 4 and tf["side"] == "right"
    rec(record_validation, "Trim-and-fill (L0, fixed-effect)", e, "adjusted estimate", tf["adjusted_es"], -0.2910259921)
    rec(record_validation, "Trim-and-fill (L0, fixed-effect)", e, "adjusted SE", tf["adjusted_se"], 0.0382641366)
    rec(record_validation, "Rosenthal fail-safe N", e, "N", r["failsafe_N"]["rosenthal_N"], 598, tol=0)


@pytest.mark.validation
def test_moderators(record_validation):
    e = "/api/meta-analysis/moderator-analysis"
    s = run(M.moderator_analysis(M.ModeratorRequest(effect_sizes=YI, standard_errors=SEI, categorical_moderator=ALLOC)))["subgroup_analysis"]
    rec(record_validation, "Subgroup analysis (mixed effects)", e, "Q between", s["Q_between"], 1.5367688721)
    assert s["Q_between_df"] == 2
    m = run(M.moderator_analysis(M.ModeratorRequest(effect_sizes=YI, standard_errors=SEI, moderators=[[a] for a in ABLAT])))["meta_regression"]
    meth = "Meta-regression (mixed effects, DL residual tau^2)"
    rec(record_validation, meth, e, "slope", m["coefficients"][1]["estimate"], -0.0292287388)
    rec(record_validation, meth, e, "slope SE", m["coefficients"][1]["se"], 0.0067330108)
    rec(record_validation, meth, e, "residual tau^2", m["tau_squared"], 0.0633005024)
    rec(record_validation, meth, e, "QM", m["Q_model"], 18.8452335426, tol=1e-4)
    rec(record_validation, meth, e, "QE", m["Q_residual"], 30.7330900107, tol=1e-4)
    rec(record_validation, meth, e, "R^2", m["R_squared_analog"], 0.794984944502, tol=1e-4)
    f = run(M.moderator_analysis(M.ModeratorRequest(effect_sizes=YI, standard_errors=SEI, moderators=[[a] for a in ABLAT], mixed_effects=False)))["meta_regression"]
    rec(record_validation, "Meta-regression (fixed effect)", e, "slope", f["coefficients"][1]["estimate"], -0.0292369343)
    rec(record_validation, "Meta-regression (fixed effect)", e, "QM", f["Q_model"], 121.4999180716, tol=1e-4)


def test_effect_size_conversions_closed_form():
    r = run(M.convert_effect_sizes(M.EffectSizeRequest(means=[10.0, 8.0], sds=[2.0, 2.5], ns=[20, 25])))
    sp = math.sqrt((19 * 4 + 24 * 6.25) / 43)
    d = 2 / sp
    var_d = 45 / 500 + d ** 2 / 90
    J = 1 - 3 / (4 * 43 - 1)
    assert r["cohens_d"]["estimate"] == pytest.approx(d, abs=1e-6)
    assert r["cohens_d"]["se"] == pytest.approx(math.sqrt(var_d), abs=1e-6)
    assert r["hedges_g"]["estimate"] == pytest.approx(d * J, abs=1e-6)
    a = 45 ** 2 / 500
    assert r["r"]["estimate"] == pytest.approx(d / math.sqrt(d ** 2 + a), abs=1e-6)
    # r -> d with Borenstein's variance, and no invented SE when N is missing
    r2 = run(M.convert_effect_sizes(M.EffectSizeRequest(r_value=0.3, total_n=50)))
    assert r2["cohens_d"]["se"] == pytest.approx(math.sqrt(4 / (49 * 0.91)), abs=1e-6)
    r3 = run(M.convert_effect_sizes(M.EffectSizeRequest(r_value=0.3)))
    assert r3["cohens_d"]["se"] is None and r3["notes"]
    r4 = run(M.convert_effect_sizes(M.EffectSizeRequest(odds_ratio=2.0, total_n=100)))
    assert r4["cohens_d"]["estimate"] == pytest.approx(math.log(2) * math.sqrt(3) / math.pi, abs=1e-6)
    assert r4["cohens_d"]["se"] is None


def test_validation_errors(client):
    bad = client.post("/api/meta-analysis/publication-bias", json={"effect_sizes": [0.1, 0.2, 0.3], "standard_errors": [0.1, 0.1, 0.1], "study_labels": ["a"]})
    assert bad.status_code == 422 and "study_labels" in bad.text
    bad = client.post("/api/meta-analysis/effect-sizes", json={"t_value": 2.0, "total_n": 1})
    assert bad.status_code == 422
    bad = client.post("/api/meta-analysis/cumulative", json={"effect_sizes": [0.1, 0.2], "standard_errors": [0.1, 0.1], "sort_values": [1]})
    assert bad.status_code == 422
