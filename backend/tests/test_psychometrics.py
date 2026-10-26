"""
Reliability and exploratory factor analysis (/api/psychometrics/*) against R 4.5.2.

Alpha: psych alpha() and alpha.ci() (Feldt). Factor analysis: stats::factanal (ML, varimax /
promax), psych::KMO and psych::cortest.bartlett, on the Holzinger & Swineford (1939) nine
ability tests (lavaan::HolzingerSwineford1939 x1-x9, rounded to 4 decimals, n = 301).
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers import psychometrics as P

ITEMS = [[4, 5, 3, 4], [5, 4, 4, 5], [3, 3, 2, 3], [4, 5, 4, 4], [5, 5, 5, 5],
         [2, 3, 2, 3], [4, 4, 3, 4], [3, 4, 3, 3], [5, 5, 4, 5], [4, 3, 3, 4]]
PSYCH = "psych 2.6.9 (R 4.5.2)"
FACTANAL = "stats::factanal (R 4.5.2)"


def hs():
    with open(Path(__file__).parent / "data" / "holzinger_swineford_1939.csv", newline="") as f:
        r = csv.reader(f)
        next(r)
        return [[float(v) for v in row] for row in r]


def _rec(rv, method, stat, obs, exp, tol, dataset, endpoint, ref):
    assert obs == pytest.approx(exp, abs=tol), (stat, obs, exp)
    rv(method=method, endpoint=endpoint, statistic=stat, observed=float(obs), expected=float(exp),
       reference=ref, reference_kind="r", dataset=dataset, tolerance_abs=tol)


@pytest.mark.validation
def test_alpha_matches_psych(record_validation):
    r = P.cronbachs_alpha(P.ReliabilityRequest(items=ITEMS))
    m, ep, ds = "Cronbach's alpha", "/api/psychometrics/reliability", "10 x 4 Likert example"
    _rec(record_validation, m, "raw alpha", r["cronbachs_alpha"], 0.9350649351, 1e-9, ds, ep, PSYCH)
    _rec(record_validation, m, "standardized alpha", r["standardized_alpha"], 0.9361103126, 1e-9, ds, ep, PSYCH)
    _rec(record_validation, m, "Feldt 95% CI lower", r["alpha_ci"][0], 0.83, 0.005, ds, ep, PSYCH + ", alpha.ci (2 d.p.)")
    _rec(record_validation, m, "Feldt 95% CI upper", r["alpha_ci"][1], 0.98, 0.005, ds, ep, PSYCH + ", alpha.ci (2 d.p.)")
    r_drop = [0.8983407144, 0.7064942185, 0.9084775750, 0.8952329535]
    a_drop = [0.8988549618, 0.9575471698, 0.8944444444, 0.9034941764]
    s_drop = [0.8991630861, 0.9615105692, 0.8968459802, 0.9036614258]
    for j in range(4):
        it = r["item_statistics"][j]
        _rec(record_validation, m, f"r.drop item {j + 1}", it["item_total_correlation"], r_drop[j], 1e-9, ds, ep, PSYCH)
        _rec(record_validation, m, f"alpha if dropped item {j + 1}", it["alpha_if_deleted"], a_drop[j], 1e-9, ds, ep, PSYCH)
        assert it["std_alpha_if_deleted"] == pytest.approx(s_drop[j], abs=1e-9)
    r = P.cronbachs_alpha(P.ReliabilityRequest(items=[row[:3] for row in hs()]))
    _rec(record_validation, m, "raw alpha", r["cronbachs_alpha"], 0.6261175745, 1e-9, "Holzinger-Swineford x1-x3", ep, PSYCH)


@pytest.mark.validation
@pytest.mark.parametrize("rotation,row1", [
    ("varimax", [0.27729131359, 0.62266175764, 0.15124995582]),
    ("none", [0.4880483939, 0.3135246355, 0.38856885396]),
    ("promax", [0.145673915867, 0.62390619715, 0.008761445802]),
])
def test_ml_factor_analysis_matches_factanal(record_validation, rotation, row1):
    r = P.factor_analysis(P.FactorAnalysisRequest(data=hs(), n_factors=3, rotation=rotation))
    ep, ds = "/api/psychometrics/factor-analysis", "Holzinger-Swineford 1939"
    m = f"ML factor analysis ({rotation})"
    # factanal's optim stops at a looser tolerance, hence 2e-5 on loadings and uniquenesses
    for k in range(3):
        _rec(record_validation, m, f"loading x1,F{k + 1}", r["loadings"][0][k], row1[k], 2e-5, ds, ep, FACTANAL)
    uniq = [0.5125256051, 0.7487374298, 0.5427750223, 0.2791938696, 0.2428761608, 0.3052150715,
            0.5022063886, 0.4685521627, 0.5432406266]
    for j in (0, 4, 8):
        _rec(record_validation, m, f"uniqueness x{j + 1}", r["uniquenesses"][j], uniq[j], 2e-5, ds, ep, FACTANAL)
    _rec(record_validation, m, "LR chi-square", r["fit_test"]["chi_square"], 22.37723, 1e-4, ds, ep, FACTANAL)
    _rec(record_validation, m, "LR p-value", r["fit_test"]["p_value"], 0.03350316, 1e-7, ds, ep, FACTANAL)
    assert r["fit_test"]["df"] == 12
    if rotation == "promax":
        fc = np.array(r["factor_correlations"])
        _rec(record_validation, m, "factor correlation F1,F2", fc[0, 1], 0.3993970856, 1e-5, ds, ep, FACTANAL)
        _rec(record_validation, m, "factor correlation F2,F3", fc[1, 2], 0.3390591935, 1e-5, ds, ep, FACTANAL)
    else:
        # orthogonal solutions: communalities are the row sums of squared loadings
        assert np.allclose((np.array(r["loadings"]) ** 2).sum(axis=1), r["communalities"], atol=1e-6)


@pytest.mark.validation
def test_kmo_and_bartlett_match_psych(record_validation):
    r = P.factor_analysis(P.FactorAnalysisRequest(data=hs(), n_factors=3))
    m, ep, ds = "KMO / Bartlett sphericity", "/api/psychometrics/factor-analysis", "Holzinger-Swineford 1939"
    _rec(record_validation, m, "KMO", r["kmo"], 0.7522444, 1e-7, ds, ep, PSYCH)
    _rec(record_validation, m, "MSA x7", r["kmo_per_variable"][6], 0.5930468845, 1e-9, ds, ep, PSYCH)
    _rec(record_validation, m, "Bartlett chi-square", r["bartlett_sphericity"]["chi_square"], 904.0993, 1e-3, ds, ep, PSYCH)
    assert r["bartlett_sphericity"]["df"] == 36
    one = P.factor_analysis(P.FactorAnalysisRequest(data=hs(), n_factors=1))
    _rec(record_validation, "ML factor analysis (1 factor)", "LR chi-square", one["fit_test"]["chi_square"],
         306.5602, 1e-3, ds, ep, FACTANAL)


def test_factor_analysis_rejects_bad_requests():
    with pytest.raises(HTTPException) as e:
        P.factor_analysis(P.FactorAnalysisRequest(data=ITEMS, n_factors=2))
    assert "too many" in e.value.detail
    with pytest.raises(HTTPException):
        P.factor_analysis(P.FactorAnalysisRequest(data=hs(), n_factors=2, rotation="oblimin"))
    with pytest.raises(HTTPException):  # constant item
        P.cronbachs_alpha(P.ReliabilityRequest(items=[[1, 2, 3]] * 5))
