"""
SEM (/api/sem/*, /api/invariance/*) against lavaan 0.7.3 and sirt (R 4.5.2).

Data (tests/data): Holzinger & Swineford (1939) nine ability tests with school
(lavaan::HolzingerSwineford1939, rounded to 4 decimals), Bollen's PoliticalDemocracy, and
lavaan's simulated Demo.growth (t1-t4, rounded to 6 decimals). Every model uses lavaan's
defaults (ML, N in the denominator, expected-information SEs, marker identification).
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers import measurement_invariance as MI
from app.routers import sem as R
from app.sem_engine import SEM, Model, cfa_params

DATA = Path(__file__).parent / "data"
LAV = "lavaan 0.7.3 (R 4.5.2)"
FAC = {"visual": ["x1", "x2", "x3"], "textual": ["x4", "x5", "x6"], "speed": ["x7", "x8", "x9"]}
FMOD = {"visual": [0, 1, 2], "textual": [3, 4, 5], "speed": [6, 7, 8]}


def rows(name):
    with open(DATA / name, newline="") as f:
        return [{k: (v if k == "school" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def hs_groups():
    hs = rows("holzinger_swineford_school.csv")
    return [[[r[f"x{i}"] for i in range(1, 10)] for r in hs if r["school"] == s] for s in ("Pasteur", "Grant-White")]


def _rec(rv, method, endpoint, stat, obs, exp, tol, dataset, ref=LAV):
    assert obs == pytest.approx(exp, abs=tol), (stat, obs, exp)
    rv(method=method, endpoint=endpoint, statistic=stat, observed=float(obs), expected=float(exp),
       reference=ref, reference_kind="r", dataset=dataset, tolerance_abs=tol)


@pytest.mark.validation
def test_cfa_matches_lavaan(record_validation):
    r = R.confirmatory_factor_analysis(R.CFARequest(data=rows("holzinger_swineford_school.csv"), factors=FAC))
    m, ep, ds = "Confirmatory factor analysis (ML)", "/api/sem/cfa", "Holzinger-Swineford 1939"
    fi = r["fit_indices"]
    for stat, exp, tol in (("chi_square", 85.30661326, 1e-5), ("cfi", 0.93055860, 1e-6), ("tli", 0.89583790, 1e-6),
                           ("rmsea", 0.09212230, 1e-6), ("rmsea_ci90_lower", 0.07141934, 1e-6),
                           ("rmsea_ci90_upper", 0.11367881, 1e-6), ("srmr", 0.06520541, 1e-6),
                           ("aic", 7517.48692134, 1e-4), ("bic", 7595.33623690, 1e-4)):
        _rec(record_validation, m, ep, stat, fi[stat], exp, tol, ds)
    assert fi["df"] == 24
    L = {(l["factor"], l["indicator"]): l for l in r["loadings"]}
    _rec(record_validation, m, ep, "loading visual=~x2", L[("visual", "x2")]["estimate"], 0.55349950, 1e-5, ds)
    _rec(record_validation, m, ep, "SE visual=~x2", L[("visual", "x2")]["se"], 0.09966494, 1e-5, ds)
    _rec(record_validation, m, ep, "std.all visual=~x2", L[("visual", "x2")]["standardized"], 0.42360016, 1e-5, ds)
    _rec(record_validation, m, ep, "SE speed=~x9", L[("speed", "x9")]["se"], 0.15116830, 1e-5, ds)
    fc = {(c["factor_1"], c["factor_2"]): c for c in r["factor_correlations"]}
    _rec(record_validation, m, ep, "cov visual~~textual", fc[("visual", "textual")]["covariance"], 0.40823145, 1e-5, ds)
    _rec(record_validation, m, ep, "cor textual~~speed", fc[("textual", "speed")]["correlation"], 0.28298313, 1e-5, ds)
    assert r["residual_variances"]["x1"]["se"] == pytest.approx(0.11360033, abs=1e-5)


@pytest.mark.validation
def test_path_analysis_matches_lavaan(record_validation):
    paths = [{"from": "x1", "to": "x4"}, {"from": "x4", "to": "x7"}, {"from": "x1", "to": "x7"}, {"from": "x4", "to": "x8"}]
    r = R.path_analysis(R.PathAnalysisRequest(data=rows("holzinger_swineford_school.csv"), paths=paths, mediators=["x4"]))
    m, ep, ds = "Path analysis (ML, fixed.x)", "/api/sem/path-analysis", "Holzinger-Swineford x1, x4, x7, x8"
    fi = r["fit_indices"]
    for stat, exp, tol in (("chi_square", 12.22728370, 1e-5), ("cfi", 0.92290024, 1e-6), ("tli", 0.53740144, 1e-6),
                           ("rmsea", 0.19313194, 1e-6), ("srmr", 0.06543881, 1e-6), ("aic", 2586.85350467, 1e-4)):
        _rec(record_validation, m, ep, stat, fi[stat], exp, tol, ds)
    d = {(e["from"], e["to"]): e for e in r["direct_effects"]}
    _rec(record_validation, m, ep, "x4~x1", d[("x1", "x4")]["coefficient"], 0.37164807, 1e-6, ds)
    _rec(record_validation, m, ep, "SE x7~x1", d[("x1", "x7")]["std_error"], 0.04984403, 1e-6, ds)
    ind = next(i for i in r["indirect_effects"] if i["to"] == "x7")
    _rec(record_validation, m, ep, "indirect x1->x4->x7", ind["indirect_effect"], 0.07380813, 1e-6, ds)
    _rec(record_validation, m, ep, "SE indirect (delta)", ind["se"], 0.02353271, 1e-6, ds)
    rc = r["residual_covariances"][0]
    assert {rc["var_1"], rc["var_2"]} == {"x7", "x8"} and rc["estimate"] == pytest.approx(0.53564873, abs=1e-6)


@pytest.mark.validation
def test_full_sem_matches_lavaan(record_validation):
    meas = {"ind60": ["x1", "x2", "x3"], "dem60": ["y1", "y2", "y3", "y4"], "dem65": ["y5", "y6", "y7", "y8"]}
    st = [{"from": "ind60", "to": "dem60"}, {"from": "ind60", "to": "dem65"}, {"from": "dem60", "to": "dem65"}]
    r = R.structural_equation_model(R.SEMRequest(data=rows("political_democracy.csv"), measurement=meas, structural=st))
    m, ep, ds = "Structural equation model (ML)", "/api/sem/sem", "PoliticalDemocracy (Bollen)"
    fi = r["fit_indices"]
    for stat, exp, tol in (("chi_square", 72.46160700, 1e-5), ("cfi", 0.95343533, 1e-6), ("rmsea", 0.10115047, 1e-6),
                           ("srmr", 0.05474240, 1e-6), ("bic", 3237.85547841, 1e-4)):
        _rec(record_validation, m, ep, stat, fi[stat], exp, tol, ds)
    p = {(s["from"], s["to"]): s for s in r["structural_paths"]}
    for (a, b), est, se, std in ((("ind60", "dem60"), 1.47374230, 0.39166965, 0.44822021),
                                  (("dem60", "dem65"), 0.86439401, 0.11268865, 0.91281796)):
        _rec(record_validation, m, ep, f"{b}~{a}", p[(a, b)]["coefficient"], est, 1e-5, ds)
        _rec(record_validation, m, ep, f"SE {b}~{a}", p[(a, b)]["std_error"], se, 1e-5, ds)
        assert p[(a, b)]["standardized"] == pytest.approx(std, abs=1e-5)
    _rec(record_validation, m, ep, "R2 dem65", r["r_squared"]["dem65"], 0.97354965, 1e-5, ds)


@pytest.mark.validation
def test_latent_growth_matches_lavaan(record_validation):
    with open(DATA / "demo_growth_lavaan.csv", newline="") as f:
        Y = [[float(v) for v in row.values()] for row in csv.DictReader(f)]
    r = R.latent_growth(R.LatentGrowthRequest(data=Y, time_points=[0, 1, 2, 3]))
    m, ep, ds = "Latent growth curve model (ML)", "/api/sem/latent-growth", "lavaan Demo.growth t1-t4"
    _rec(record_validation, m, ep, "chi_square", r["fit_indices"]["chi_square"], 8.06873289, 1e-5, ds)
    _rec(record_validation, m, ep, "srmr", r["fit_indices"]["srmr"], 0.01211705, 1e-6, ds)
    _rec(record_validation, m, ep, "intercept mean", r["intercept"]["mean"], 0.61469576, 1e-6, ds)
    _rec(record_validation, m, ep, "SE intercept mean", r["intercept"]["mean_se"], 0.07677168, 1e-6, ds)
    _rec(record_validation, m, ep, "slope mean", r["slope"]["mean"], 1.00629434, 1e-6, ds)
    _rec(record_validation, m, ep, "slope variance", r["slope"]["variance"], 0.58691419, 1e-6, ds)
    _rec(record_validation, m, ep, "i~~s correlation", r["correlation_intercept_slope"], 0.58023443, 1e-6, ds)


@pytest.mark.validation
def test_invariance_levels_match_lavaan(record_validation):
    g = hs_groups()
    ref = {"configural": (115.85279175, 48, 0.92339701), "metric": (124.04507631, 54, None),
           "scalar": (164.10452348, 60, 0.88247031)}
    for level, (chi, df, cfi) in ref.items():
        r = MI._level_endpoint(MI.InvarianceRequest(data_groups=g, factor_model=FMOD), level)
        m = f"Measurement invariance ({level})"
        _rec(record_validation, m, f"/api/invariance/{level}", "chi_square", r["chi_square"], chi, 1e-5,
             "Holzinger-Swineford by school")
        assert r["df"] == df
        if cfi is not None:
            assert r["cfi"] == pytest.approx(cfi, abs=1e-6)
    sc = MI._level_endpoint(MI.InvarianceRequest(data_groups=g, factor_model=FMOD), "scalar")
    tm = next(x for x in sc["latent_means"] if x["factor"] == "textual")
    _rec(record_validation, "Measurement invariance (scalar)", "/api/invariance/scalar", "textual mean, group 2",
         tm["mean"], 0.57635905, 1e-5, "Holzinger-Swineford by school")
    assert tm["se"] == pytest.approx(0.11718995, abs=1e-5)
    two = R.measurement_invariance(R.MeasurementInvarianceRequest(data_group1=g[0], data_group2=g[1], n_factors=3,
                                                                  factor_model=FMOD))
    assert [x["level"] for x in two["models"]] == ["configural", "metric", "scalar", "strict"]
    assert two["models"][3]["chi_square"] == pytest.approx(181.513, abs=1e-3)
    assert two["conclusion"].startswith("Metric invariance holds")


@pytest.mark.validation
def test_partial_invariance_matches_lavaan_group_partial(record_validation):
    r = MI.partial_invariance(MI.PartialInvarianceRequest(data_groups=hs_groups(), factor_model=FMOD,
                                                          invariance_level="scalar"))
    assert r["released_items"] == ["item3", "item7"] and r["partial_invariance_holds"]
    for step, exp in zip(r["steps"][1:], (144.58040592, 129.42405260)):
        _rec(record_validation, "Partial scalar invariance", "/api/invariance/partial-invariance",
             f"chi_square releasing up to {step['released']}", step["chi_square"], exp, 1e-5,
             "Holzinger-Swineford by school", LAV + " group.partial")


@pytest.mark.validation
def test_alignment_matches_sirt(record_validation):
    lam = np.array([[1.0, .8, .9, 1.1, .7, .95], [1.2, .95, 1.1, 1.3, .85, 1.15],
                    [.9, .72, .81, .99, .3, .86], [1.5, 1.2, 1.35, 1.65, 1.05, 1.42]])
    nu = np.array([[3.0, 2.5, 2.8, 3.2, 2.2, 2.9], [3.4, 2.8, 3.2, 3.7, 2.5, 3.3],
                   [2.7, 2.3, 2.5, 1.9, 2.0, 2.6], [3.9, 3.2, 3.6, 4.1, 2.9, 3.8]])
    r = MI.alignment_method(MI.AlignmentRequest(group_loadings=[l[:, None].tolist() for l in lam],
                                                group_intercepts=nu.tolist(), group_sizes=[200, 350, 150, 400]))
    f = r["factors"][0]
    ref = "sirt (R 4.5.2) invariance.alignment"
    for g, (a, s) in enumerate(zip([0, 0.416946, -0.313206, 0.907128], [1, 1.201904, 0.901231, 1.500020])):
        _rec(record_validation, "Alignment (Asparouhov & Muthen)", "/api/invariance/alignment", f"group {g + 1} mean",
             f["groups"][g]["mean"], a, 1e-5, "4 groups x 6 items (constructed)", ref)
        _rec(record_validation, "Alignment (Asparouhov & Muthen)", "/api/invariance/alignment", f"group {g + 1} SD",
             f["groups"][g]["sd"], s, 1e-5, "4 groups x 6 items (constructed)", ref)
    assert f["r_squared"]["loadings"] == pytest.approx(0.996248, abs=1e-5)
    assert f["r_squared"]["intercepts"] == pytest.approx(0.996767, abs=1e-5)
    assert f["rbar"]["intercepts"] == pytest.approx(0.712587, abs=1e-5)


def test_alignment_from_data_uses_lavaan_std_lv_configural_estimates():
    g = hs_groups()
    r = MI.alignment_method(MI.AlignmentRequest(data_groups=g, factor_model=FMOD))
    t = next(f for f in r["factors"] if f["factor"] == "textual")
    assert t["configural_loadings"][0] == pytest.approx([0.94567254, 1.11904965, 0.82743008], abs=1e-5)
    assert t["configural_intercepts"][1] == pytest.approx([3.31724138, 4.71206897, 2.46896483], abs=1e-5)


def test_engine_gradient_is_exact():
    hs = rows("holzinger_swineford_school.csv")
    obs = [f"x{i}" for i in range(1, 10)]
    X = np.array([[r[o] for o in obs] for r in hs])
    S = np.cov(X, rowvar=False, bias=True)
    sem = SEM(Model(obs, list(FAC), cfa_params(FAC, 1, S_list=[S], observed=obs)), [X])
    th = sem.theta0 + 0.05
    g = sem.gradient(th)
    num = np.array([(sem.objective(th + e) - sem.objective(th - e)) / 2e-6 for e in np.eye(len(th)) * 1e-6])
    assert np.max(np.abs(g - num)) < 1e-6


def test_sem_input_validation():
    hs = rows("holzinger_swineford_school.csv")
    with pytest.raises(HTTPException):  # unknown variable
        R.confirmatory_factor_analysis(R.CFARequest(data=hs, factors={"F": ["x1", "x2", "nope"]}))
    with pytest.raises(HTTPException):  # single factor with 2 indicators: not identified
        R.confirmatory_factor_analysis(R.CFARequest(data=hs, factors={"F": ["x1", "x2"]}))
    with pytest.raises(HTTPException):  # feedback loop
        R.path_analysis(R.PathAnalysisRequest(data=hs, paths=[{"from": "x1", "to": "x2"}, {"from": "x2", "to": "x1"}]))
    with pytest.raises(HTTPException):  # multi-factor invariance without a factor model
        g = hs_groups()
        R.measurement_invariance(R.MeasurementInvarianceRequest(data_group1=g[0], data_group2=g[1], n_factors=2))
