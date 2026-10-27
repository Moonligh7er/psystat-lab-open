"""Diagnostic classification models checked against the R package GDINA 2.13.2 (R 4.5.2).

Data: tests/data/dcm_sim.csv (1000 simulated examinees, 12 items) with Q-matrix
tests/data/dcm_sim_q.csv (3 attributes). Items 1-6 measure one attribute; items 7, 10, 12
were generated DINA-like, item 8 DINO-like and items 9 and 11 additive.
R reference:

    fit <- GDINA(X, Q, model = "DINA" / "DINO" / "GDINA", control = list(conv.crit = 1e-9, maxitr = 20000))
    logLik(fit); extract(fit, "npar"); coef(fit, what = "catprob", digits = 12)
    extract(fit, "posterior.prob"); CA(fit)$tau_k; CA(fit)$tau; personparm(fit, "mp", digits = 10)
    anova(DINA, GDINA); Qval(GDINA, method = "PVAF", eps = 0.95, digits = 10)
"""

from pathlib import Path

import numpy as np
import pytest

from app.routers.dcm import (
    CompareRequest, DinaRequest, GdinaRequest, QMatrixValidateRequest,
    compare_models, dina_model, dino_model, gdina_model, qmatrix_validate,
)

HERE = Path(__file__).parent / "data"
X = np.loadtxt(HERE / "dcm_sim.csv", delimiter=",", skiprows=1).astype(int).tolist()
Q = np.loadtxt(HERE / "dcm_sim_q.csv", delimiter=",", skiprows=1).astype(int).tolist()

R = {
    "dina": {
        "ll": -6909.478785, "npar": 31, "aic": 13880.957570, "bic": 14033.097984,
        "prob": [0.20827730, 0.83026174, 0.17032060, 0.84261744, 0.15924613, 0.79437712, 0.18991612, 0.88224349,
                 0.19821326, 0.84783967, 0.16642383, 0.78377001, 0.14544111, 0.14544111, 0.14544111, 0.90126258,
                 0.48582906, 0.48582906, 0.48582906, 0.85812454, 0.28889903, 0.28889903, 0.28889903, 0.82656431,
                 0.12509764, 0.12509764, 0.12509764, 0.90719228, 0.25259259, 0.25259259, 0.25259259, 0.81603698,
                 0.12073437, 0.12073437, 0.12073437, 0.12073437, 0.12073437, 0.12073437, 0.12073437, 0.80957216],
        "pi": [0.21453380, 0.10584310, 0.06062208, 0.06657345, 0.10295965, 0.08846084, 0.07027734, 0.29072974],
        "tau_k": [0.92327813, 0.93879237, 0.90158096], "tau": 0.79447304,
    },
    "dino": {
        "ll": -7065.387641, "npar": 31, "aic": 14192.775281, "bic": 14344.915695,
        "prob": [0.40805181, 0.85762904, 0.31639394, 0.87853103, 0.25132850, 0.87036752, 0.40775682, 0.92044350,
                 0.34122217, 0.87933927, 0.23802896, 0.88674852, 0.13953092, 0.90755464, 0.90755464, 0.90755464,
                 0.39456394, 0.83587630, 0.83587630, 0.83587630, 0.22478906, 0.73393069, 0.73393069, 0.73393069,
                 0.13054433, 0.89600331, 0.89600331, 0.89600331, 0.18729295, 0.71713086, 0.71713086, 0.71713086,
                 0.10427796, 0.51151247, 0.51151247, 0.51151247, 0.51151247, 0.51151247, 0.51151247, 0.51151247],
        "pi": [0.46782006, 0.02503041, 0.00548710, 0.13705011, 0.12095704, 0.00256263, 0.02052219, 0.22057046],
        "tau_k": [0.94854502, 0.95791831, 0.89481200], "tau": 0.83815069,
    },
    "gdina": {
        "ll": -6770.416149, "npar": 47, "aic": 13634.832298, "bic": 13865.496796,
        "prob": [0.21234066, 0.82962684, 0.18285568, 0.84901966, 0.19177139, 0.83865795, 0.18866235, 0.88561984,
                 0.19755118, 0.86626999, 0.19428760, 0.83127852, 0.15733407, 0.15234001, 0.13451947, 0.91358268,
                 0.16070296, 0.79410476, 0.81712141, 0.83656570, 0.14430224, 0.51020682, 0.53776378, 0.81725789,
                 0.12947137, 0.15484220, 0.10981640, 0.91522036, 0.11904847, 0.45054578, 0.49004537, 0.81646313,
                 0.13372199, 0.13990923, 0.06858019, 0.09931233, 0.05421591, 0.14859593, 0.23178568, 0.87068278],
        "pi": [0.21676448, 0.13684990, 0.07386002, 0.06968057, 0.11614170, 0.06610357, 0.05380892, 0.26679085],
        "tau_k": [0.93067103, 0.94119990, 0.93405587], "tau": 0.82936472,
    },
}
ENDPOINT = {"dina": dina_model, "dino": dino_model, "gdina": gdina_model}


@pytest.mark.validation
@pytest.mark.parametrize("model", ["dina", "dino", "gdina"])
def test_matches_gdina_package(record_validation, model):
    ref = R[model]
    res = ENDPOINT[model](DinaRequest(responses=X, q_matrix=Q))
    fit = res["model_fit"]
    assert fit["log_likelihood"] == pytest.approx(ref["ll"], abs=1e-4)
    record_validation(method=f"{res['model']} (EM, MMLE)", endpoint=f"/api/dcm/{model}", statistic="log-likelihood",
                      observed=fit["log_likelihood"], expected=ref["ll"], tolerance_abs=1e-4, reference="GDINA 2.13.2 (R 4.5.2)", reference_kind="r", dataset="simulated dcm_sim (N = 1000, 12 items, 3 attributes)")
    assert fit["n_parameters"] == ref["npar"]
    assert fit["AIC"] == pytest.approx(ref["aic"], abs=1e-3)
    assert fit["BIC"] == pytest.approx(ref["bic"], abs=1e-3)
    probs = [p["probability"] for it in res["item_parameters"] for p in it["success_probabilities"]]
    if model != "gdina":  # GDINA lists DINA/DINO items by every reduced pattern
        probs = []
        for it, q in zip(res["item_parameters"], Q):
            kj = sum(q)
            g, s = it["guess"], it["slip"]
            from app.routers.dcm import _patterns
            for row in _patterns(kj):
                full = row.all() if model == "dina" else row.any()
                probs.append(1 - s if full else g)
    assert probs == pytest.approx(ref["prob"], abs=2e-6)
    assert [c["proportion"] for c in res["latent_classes"]] == pytest.approx(ref["pi"], abs=2e-6)
    ca = res["classification_accuracy"]
    assert list(ca["attribute"].values()) == pytest.approx(ref["tau_k"], abs=2e-6)
    assert ca["pattern"] == pytest.approx(ref["tau"], abs=2e-6)


def test_marginal_mastery_and_map():
    res = gdina_model(GdinaRequest(responses=X, q_matrix=Q))
    mp = np.array(res["attribute_mastery_probabilities"][:3])
    ref = np.array([0.99554621, 0.99858208, 0.99951388, 0.81440502, 0.46541159, 0.01799640,
                    0.99994868, 0.99997597, 0.99952628]).reshape(3, 3)
    assert np.max(np.abs(mp - ref)) < 2e-6
    assert res["most_likely_profiles"][:5] == ["111", "100", "111", "110", "000"]


def test_compare_lr_tests():
    res = compare_models(CompareRequest(responses=X, q_matrix=Q))
    lr = {t["reduced_model"]: t for t in res["likelihood_ratio_tests"]}
    assert lr["DINA"]["chi_square"] == pytest.approx(278.13, abs=0.01)
    assert lr["DINO"]["chi_square"] == pytest.approx(589.94, abs=0.01)
    assert lr["DINA"]["df"] == lr["DINO"]["df"] == 16
    assert res["best_by_bic"] == "G-DINA"


PVAF = [
    [0.99506818, 0.10709700, 0.06657156, 0.99710501, 0.99646684, 0.13138454, 1.0],
    [0.11318099, 0.99427168, 0.16966701, 0.99500682, 0.23213104, 0.99919180, 1.0],
    [0.08138579, 0.09381718, 0.99591933, 0.13956160, 0.99638041, 0.99882803, 1.0],
    [0.99718640, 0.12785132, 0.07788060, 0.99764520, 0.99869878, 0.15804103, 1.0],
    [0.12162012, 0.99858734, 0.10479437, 0.99880835, 0.18260982, 0.99963128, 1.0],
    [0.05648609, 0.11608057, 0.99639465, 0.14482842, 0.99744537, 0.99770354, 1.0],
    [0.44044103, 0.57717674, 0.14484138, 0.99987447, 0.51138135, 0.60150265, 1.0],
    [0.57889927, 0.11868405, 0.39209235, 0.59404739, 0.99962826, 0.42359903, 1.0],
    [0.08563074, 0.65148353, 0.68293277, 0.65523101, 0.69114091, 0.99255025, 1.0],
    [0.46868666, 0.56930473, 0.13257837, 0.99952860, 0.53408033, 0.59463750, 1.0],
    [0.14641361, 0.63861928, 0.70836433, 0.65415091, 0.73598859, 0.99725482, 1.0],
    [0.23356914, 0.30526457, 0.47816365, 0.50816541, 0.74122882, 0.83520964, 1.0],
]


@pytest.mark.validation
def test_qval_pvaf_matches_gdina(record_validation):
    res = qmatrix_validate(QMatrixValidateRequest(responses=X, q_matrix=Q))
    for it, ref in zip(res["items"], PVAF):
        assert list(it["pvaf"].keys()) == ["100", "010", "001", "110", "101", "011", "111"]
        assert list(it["pvaf"].values()) == pytest.approx(ref, abs=2e-6)
    assert res["suggested_q_matrix"] == Q  # GDINA suggests no changes either
    record_validation(method="Q-matrix validation (PVAF)", endpoint="/api/dcm/qmatrix-validate", statistic="PVAF item 12, q = 011",
                      observed=res["items"][11]["pvaf"]["011"], expected=0.83520964, tolerance_abs=2e-6, reference="GDINA 2.13.2 (R 4.5.2)", reference_kind="r", dataset="simulated dcm_sim (N = 1000, 12 items, 3 attributes)")
    assert res["n_suggested_changes"] == 0


def test_qval_misspecified_entry_matches_gdina():
    # Item 7 really needs A1 and A2. With the G-DINA fit to the wrong Q-matrix, a single
    # PVAF pass keeps the under-specified 100 (PVAF .996 >= .95): GDINA::Qval does the same.
    Qbad = [row[:] for row in Q]
    Qbad[6] = [1, 0, 0]
    res = qmatrix_validate(QMatrixValidateRequest(responses=X, q_matrix=Qbad))
    ref = [0.9961546, 0.5262278, 0.1352373, 0.9997664, 0.9966932, 0.5515310, 1.0]
    assert list(res["items"][6]["pvaf"].values()) == pytest.approx(ref, abs=2e-6)
    assert res["items"][6]["suggested_q"] == "100"


def test_rejects_bad_q():
    from fastapi import HTTPException
    Qz = [row[:] for row in Q]
    Qz[0] = [0, 0, 0]
    with pytest.raises(HTTPException):
        dina_model(DinaRequest(responses=X, q_matrix=Qz))
