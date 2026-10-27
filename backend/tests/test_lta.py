"""
Latent transition analysis (/api/lta/*) against depmixS4 (R 4.5.2) and an independent
mover-stayer likelihood maximised with optim() in R.

Data (tests/data/lta_*.csv): simulated with a fixed seed, 600 people, 4 binary items, 2 classes;
columns are the 4 items at time 1, then at time 2 (and time 3). lta_t3_stayers has a 40% stayer
group. depmixS4 holds item probabilities equal over time and uses one (homogeneous) transition
matrix; with 2 occasions that equals time-specific transitions.
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers import lta as L

DATA = Path(__file__).parent / "data"


def load(name, T):
    W = np.loadtxt(DATA / f"{name}.csv", delimiter=",")
    return [W[:, t * 4:(t + 1) * 4].tolist() for t in range(T)]


@pytest.mark.validation
@pytest.mark.parametrize("name,T,transitions,exp", [
    ("lta_t2", 2, "time_specific", -2851.553920),
    ("lta_t3", 3, "homogeneous", -4210.663354),
])
def test_lta_loglik_matches_depmixs4(record_validation, name, T, transitions, exp):
    r = L.lta_analyze(L.LTAAnalyzeRequest(indicators=load(name, T), n_classes=2, transitions=transitions))
    assert r["fit"]["log_likelihood"] == pytest.approx(exp, abs=1e-4)
    assert r["fit"]["n_params"] == 11
    record_validation(method="Latent transition analysis (EM)", endpoint="/api/lta/analyze",
                      statistic=f"log-likelihood, T = {T}", observed=r["fit"]["log_likelihood"], expected=exp,
                      reference="depmixS4 (R 4.5.2)", reference_kind="r", dataset=f"simulated {name}", tolerance_abs=1e-4)
    # transition rows are probabilities; profiles are ordered with class_0 highest
    for m in r["transition_matrices"].values():
        assert np.allclose(np.sum(m, axis=1), 1)
    prof = next(iter(r["class_profiles"].values()))
    assert np.mean(prof["class_0"]) > np.mean(prof["class_1"])


@pytest.mark.validation
def test_mover_stayer_matches_independent_ml(record_validation):
    m = L.lta_mover_stayer(L.LTAMoverStayerRequest(indicators=load("lta_t3_stayers", 3), n_classes=2,
                                                   transitions="homogeneous"))
    ref = "independent likelihood + optim (R 4.5.2)"
    assert m["mover_stayer_fit"]["log_likelihood"] == pytest.approx(-4238.694696, abs=1e-4)
    record_validation(method="Mover-stayer LTA", endpoint="/api/lta/mover-stayer", statistic="log-likelihood",
                      observed=m["mover_stayer_fit"]["log_likelihood"], expected=-4238.694696, reference=ref,
                      reference_kind="r", dataset="simulated lta_t3_stayers", tolerance_abs=1e-4)
    got = sorted([*m["stayers_by_class"].values(), *m["movers_by_initial_class"].values()])
    exp = sorted([0.204817, 0.240888, 0.359861, 0.194434])
    for g, e in zip(got, exp):
        assert g == pytest.approx(e, abs=1e-4)
    assert m["proportion_stayers"] == pytest.approx(0.194434 + 0.359861, abs=1e-4)
    assert m["mover_stayer_fit"]["log_likelihood"] >= m["standard_lta_fit"]["log_likelihood"] - 1e-6


def test_lta_validation():
    with pytest.raises(HTTPException):  # non-binary
        L.lta_analyze(L.LTAAnalyzeRequest(indicators=[[[0, 2]] * 40, [[1, 0]] * 40], n_classes=2))
    with pytest.raises(HTTPException):  # one time point
        L.lta_analyze(L.LTAAnalyzeRequest(indicators=[[[0, 1]] * 40], n_classes=2))
    with pytest.raises(HTTPException):  # bad transitions option
        L.lta_analyze(L.LTAAnalyzeRequest(indicators=load("lta_t2", 2), n_classes=2, transitions="other"))
