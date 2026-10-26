"""
IRT (/api/irt/* and /api/psychometrics/irt*) against mirt 1.48 (R 4.5.2), converged with TOL = 1e-10.

Data: LSAT7 (Bock & Lieberman, 1970; 1000 examinees x 5 items, from mirt::LSAT7) and Science
(Karlheinz & Melich, 1992; 392 x 4 four-category items, from mirt::Science, recoded 0-3).
Rasch: mirt itemtype "Rasch" (slopes 1, latent variance estimated). Item fit: mirt::itemfit
(fit_stats = "infit", method = "EAP"). Reliability: mirt::empirical_rxx.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.irt_engine import _Dichotomous
from app.routers import irt_advanced as I
from app.routers import psychometrics as P

DATA = Path(__file__).parent / "data"
LSAT7_COUNTS = ("00000:12,00001:19,00010:1,00011:7,00100:3,00101:19,00110:3,00111:17,01000:10,01001:5,"
                "01010:3,01011:7,01100:7,01101:23,01110:8,01111:28,10000:7,10001:39,10010:11,10011:34,"
                "10100:14,10101:51,10110:15,10111:90,11000:6,11001:25,11010:7,11011:35,11100:18,11101:136,"
                "11110:32,11111:308")
REF = "mirt 1.48 (R 4.5.2)"


def lsat7():
    rows = []
    for item in LSAT7_COUNTS.split(","):
        pat, k = item.split(":")
        rows += [[int(ch) for ch in pat]] * int(k)
    return rows


def science():
    with open(DATA / "science_mirt.csv", newline="") as f:
        r = csv.reader(f)
        next(r)
        return [[int(v) - 1 for v in row] for row in r]


def _rec(rv, method, endpoint, stat, obs, exp, tol, dataset):
    assert obs == pytest.approx(exp, abs=tol), (stat, obs, exp)
    rv(method=method, endpoint=endpoint, statistic=stat, observed=float(obs), expected=float(exp),
       reference=REF, reference_kind="r", dataset=dataset, tolerance_abs=tol)


@pytest.mark.validation
def test_rasch_matches_mirt(record_validation):
    r = I.rasch_model(I.RaschRequest(responses=lsat7()))
    m, ep = "Rasch model (MML)", "/api/irt/rasch"
    b = [-1.86826050, -0.79100432, -1.46097621, -0.52150468, -1.99297041]
    for j, exp in enumerate(b):
        _rec(record_validation, m, ep, f"b{j + 1}", r["item_parameters"][j]["difficulty"], exp, 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "latent variance", r["latent_sd"] ** 2, 1.02264709, 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "log-likelihood", r["log_likelihood"], -2664.90089095, 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "empirical reliability", r["reliability"], 0.43162395, 1e-4, "LSAT7")
    infit = [0.9393162, 0.8261344, 0.8595587, 0.8175020, 0.9935414]
    outfit = [0.7443282, 0.7575904, 0.7112288, 0.7697234, 0.7972851]
    for j in range(5):
        _rec(record_validation, "Rasch item fit (EAP)", ep, f"infit item {j + 1}",
             r["item_parameters"][j]["infit"], infit[j], 1e-4, "LSAT7")
        _rec(record_validation, "Rasch item fit (EAP)", ep, f"outfit item {j + 1}",
             r["item_parameters"][j]["outfit"], outfit[j], 1e-4, "LSAT7")
    # the basic Psychometrics endpoint uses the same engine
    basic = P.item_response_theory(P.IRTRequest(responses=lsat7()))
    assert basic["item_parameters"][0]["difficulty"] == pytest.approx(b[0], abs=1e-4)


@pytest.mark.validation
def test_2pl_matches_mirt(record_validation):
    r = I.two_pl_model(I.TwoPLRequest(responses=lsat7()))
    m, ep = "2PL model (MML)", "/api/irt/two-pl"
    a = [0.98754590, 1.08083714, 1.70747766, 0.76498988, 0.73567272]
    b = [-1.87926014, -0.74754055, -1.05723562, -0.63530226, -2.52076438]
    for j in range(5):
        _rec(record_validation, m, ep, f"a{j + 1}", r["item_parameters"][j]["discrimination"], a[j], 1e-4, "LSAT7")
        _rec(record_validation, m, ep, f"b{j + 1}", r["item_parameters"][j]["difficulty"], b[j], 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "SE(a1)", r["item_parameters"][0]["se_discrimination"], 0.17719483, 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "SE(b1)", r["item_parameters"][0]["se_difficulty"], 0.26396738, 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "log-likelihood", r["log_likelihood"], -2658.80511363, 1e-4, "LSAT7")
    _rec(record_validation, m, ep, "empirical reliability", r["reliability"], 0.45230498, 1e-4, "LSAT7")
    infit = [0.9581757, 0.8159859, 0.7500994, 0.8814275, 0.9892206]
    outfit = [0.7758136, 0.7367778, 0.4843680, 0.8552091, 0.8662292]
    for j in range(5):
        assert r["item_parameters"][j]["infit"] == pytest.approx(infit[j], abs=1e-4)
        assert r["item_parameters"][j]["outfit"] == pytest.approx(outfit[j], abs=1e-4)
    basic = P.irt_2pl(P.IRT2PLRequest(responses=lsat7()))
    assert basic["item_parameters"][2]["discrimination"] == pytest.approx(a[2], abs=1e-4)


@pytest.mark.validation
def test_3pl_likelihood_and_posterior_mode(record_validation):
    """3PL with a Beta(5, 17) prior on guessing (mirt parprior "expbeta", 5, 17).

    mirt's EM stops at a slightly lower point of the same objective (marginal LL plus the
    Beta log density): -2651.6875 with every mirt optimiser tried, against -2651.6142 here.
    So the test checks that our marginal likelihood reproduces mirt's logLik at mirt's
    estimates, and that our estimate is at least as good on that shared objective; it does
    not claim the estimates are equal."""
    from scipy.special import logit
    from scipy.stats import beta
    X = np.array(lsat7(), dtype=float)
    a = np.array([1.08617420, 1.45660250, 2.09514381, 0.90515278, 0.80635723])
    b = np.array([-1.46047755, -0.27223711, -0.74035362, -0.06135620, -1.97366225])
    g = np.array([0.19899010, 0.20238070, 0.19755344, 0.19391062, 0.19972033])
    flat = _Dichotomous(X, "3pl", (1.0, 1.0))  # Beta(1, 1): the objective is the marginal LL itself
    ll_at_mirt = -flat.negll_grad(np.concatenate([a, b, logit(g)]))[0]
    m, ep = "3PL model (MML, Beta prior)", "/api/irt/three-pl"
    _rec(record_validation, m, ep, "log-likelihood at mirt estimates", ll_at_mirt, -2659.29519692, 1e-6, "LSAT7")
    r = I.three_pl_model(I.ThreePLRequest(responses=lsat7(), guessing_prior=0.2))
    c = np.array([it["guessing"] for it in r["item_parameters"]])
    ours = r["log_likelihood"] + beta.logpdf(c, 5, 17).sum()
    assert r["log_prior"] == pytest.approx(beta.logpdf(c, 5, 17).sum())
    assert ours >= -2651.687466 - 1e-6
    record_validation(method=m, endpoint=ep, statistic="log posterior (must be >= mirt's)", observed=float(ours),
                      expected=-2651.687466, reference=REF, reference_kind="r", dataset="LSAT7", tolerance_abs=0.0)
    assert all(0.15 < x < 0.25 for x in c) and r["converged"]


@pytest.mark.validation
def test_graded_response_matches_mirt(record_validation):
    r = I.graded_response_model(I.GradedResponseRequest(responses=science()))
    m, ep = "Graded response model (MML)", "/api/irt/graded-response"
    ref = {0: (1.040641521, [-4.672811382, -2.5360934626, 1.4082242492]),
           1: (1.225832734, [-2.385317569, -0.7351170965, 1.8488898426]),
           2: (2.300594758, [-2.279921377, -0.9643779009, 0.8552119456]),
           3: (1.093799468, [-3.059870287, -0.9063997464, 1.5428156474])}
    for j, (a, bs) in ref.items():
        it = r["item_parameters"][j]
        _rec(record_validation, m, ep, f"a{j + 1}", it["discrimination"], a, 1e-4, "Science")
        for k, bk in enumerate(bs):
            _rec(record_validation, m, ep, f"b{j + 1},{k + 1}", it["thresholds"][k], bk, 1e-4, "Science")
    _rec(record_validation, m, ep, "log-likelihood", r["log_likelihood"], -1608.86941312, 1e-4, "Science")
    _rec(record_validation, m, ep, "empirical reliability", r["reliability"], 0.66703761, 1e-4, "Science")


def test_irt_input_validation():
    with pytest.raises(HTTPException):  # not 0/1
        I.two_pl_model(I.TwoPLRequest(responses=[[0, 2]] * 20))
    with pytest.raises(HTTPException):  # an item everyone got right
        I.rasch_model(I.RaschRequest(responses=[[1, i % 2] for i in range(20)]))
    with pytest.raises(HTTPException):  # 3PL needs n >= 100
        I.three_pl_model(I.ThreePLRequest(responses=lsat7()[::20]))
    with pytest.raises(HTTPException):  # ragged
        I.rasch_model(I.RaschRequest(responses=[[0, 1], [1]] * 10))
    with pytest.raises(HTTPException):  # GRM category gap
        I.graded_response_model(I.GradedResponseRequest(responses=[[0, 2], [2, 0]] * 15))


@pytest.mark.validation
def test_partial_credit_matches_mirt(record_validation):
    r = I.partial_credit_model(I.PartialCreditRequest(responses=science()))
    m, ep = "Partial credit model (MML)", "/api/irt/partial-credit"
    steps = [[-3.087966, -2.5953185, 1.388795], [-1.896367, -0.9109218, 1.858756],
             [-2.647599, -1.4212167, 1.135212], [-2.449166, -0.8997573, 1.358033]]
    for j, row in enumerate(steps):
        for k, d in enumerate(row):
            _rec(record_validation, m, ep, f"step {j + 1},{k + 1}", r["item_parameters"][j]["step_parameters"][k], d, 3e-4, "Science")
    _rec(record_validation, m, ep, "latent variance", r["latent_sd"] ** 2, 1.002532, 1e-4, "Science")
    _rec(record_validation, m, ep, "log-likelihood", r["log_likelihood"], -1619.27410089, 1e-4, "Science")
    _rec(record_validation, m, ep, "empirical reliability", r["reliability"], 0.60888910, 1e-4, "Science")
    infit = [0.9270181, 0.8809121, 0.7682106, 0.8411602]
    outfit = [0.9039714, 0.8623146, 0.7534683, 0.8323332]
    for j in range(4):
        assert r["item_parameters"][j]["infit"] == pytest.approx(infit[j], abs=1e-4)
        assert r["item_parameters"][j]["outfit"] == pytest.approx(outfit[j], abs=1e-4)


def verbal():
    with open(DATA / "verbal_difR.csv", newline="") as f:
        r = csv.reader(f)
        next(r)
        rows = [[int(v) for v in row] for row in r]
    return [row[:24] for row in rows], [row[24] for row in rows]


DIFR = "difR 6.1.0 (R 4.5.2)"


@pytest.mark.validation
def test_dif_matches_difR(record_validation):
    X, g = verbal()
    r = I.differential_item_functioning(I.DIFRequest(responses=X, group=g, purification=False))
    ep, ds = "/api/irt/dif", "difR verbal (Gender)"
    mh = [1.70763713, 2.14859270, 0.99259272, 1.93019714, 2.95399139, 9.60320864]
    alpha = [1.7004655, 1.7701795, 1.4480967, 1.9394745, 1.9799015, 2.8803829]
    lr = [2.001354, 3.354099, 2.474220, 4.729633, 4.140395, 11.411094]
    dr2 = [0.00673190, 0.01009785, 0.00738698, 0.01767887, 0.01247953, 0.03427300]
    lord = [1.235678, 2.208876, 0.974123, 3.972956, 2.846341, 5.340329]
    for j in range(6):
        it = r["items"][j]
        for meth, stat, obs, exp, tol, ref in (
                ("Mantel-Haenszel DIF", f"MH chi-square item {j + 1}", it["mantel_haenszel"]["chi_square"], mh[j], 1e-6, DIFR),
                ("Mantel-Haenszel DIF", f"MH common odds ratio item {j + 1}", it["mantel_haenszel"]["odds_ratio"], alpha[j], 1e-6, DIFR),
                ("Logistic regression DIF", f"LR chi-square (2 df) item {j + 1}", it["logistic_regression"]["overall_dif"]["chi_square"], lr[j], 1e-5, DIFR),
                ("Logistic regression DIF", f"Nagelkerke delta R2 item {j + 1}", it["logistic_regression"]["delta_nagelkerke_r2"], dr2[j], 1e-7, DIFR),
                # Lord: per-group 2PL by mirt (Oakes SEs) fed to difR::difLord; our SEs come from a
                # finite-difference Hessian, hence the looser tolerance
                ("Lord's chi-square DIF", f"Lord chi-square item {j + 1}", it["lord"]["chi_square"], lord[j], 5e-3, DIFR + " + mirt 1.48")):
            assert obs == pytest.approx(exp, abs=tol), (stat, obs, exp)
            record_validation(method=meth, endpoint=ep, statistic=stat, observed=float(obs), expected=float(exp),
                              reference=ref, reference_kind="r", dataset=ds, tolerance_abs=tol)
    pur = I.differential_item_functioning(I.DIFRequest(responses=X, group=g, method="mh", purification=True))
    assert pur["summary"]["flagged_items"]["mh"] == [5, 7, 13, 15, 16, 18, 19, 21, 22]  # difR: 6 8 14 16 17 19 20 22 23
    assert pur["purification"]["mh"]["iterations"] == 6
    _rec_mh = pur["items"][16]["mantel_haenszel"]["chi_square"]
    assert _rec_mh == pytest.approx(11.94363843, abs=1e-6)
    record_validation(method="Mantel-Haenszel DIF (purified)", endpoint=ep, statistic="MH chi-square item 17",
                      observed=float(_rec_mh), expected=11.94363843, reference=DIFR, reference_kind="r",
                      dataset=ds, tolerance_abs=1e-6)


@pytest.mark.validation
def test_linking_matches_equateIRT(record_validation):
    ax, bx, cx = [0.8, 1.2, 1.5, 0.9, 1.1, 0.7], [-1.2, -0.4, 0.3, 0.8, 1.5, -0.1], [0.2, 0.15, 0.25, 0.1, 0.2, 0.18]
    ay, by, cy = [0.7, 1.05, 1.4, 0.8, 0.95, 0.66], [-0.9, -0.1, 0.7, 1.1, 1.95, 0.25], [0.19, 0.16, 0.22, 0.12, 0.21, 0.17]
    X = [dict(a=a, b=b, c=c) for a, b, c in zip(ax, bx, cx)]
    Y = [dict(a=a, b=b, c=c) for a, b, c in zip(ay, by, cy)]
    r = I.test_equating(I.TestEquatingRequest(params_form_x=X, params_form_y=Y,
                                              anchor_indices_x=list(range(6)), anchor_indices_y=list(range(6))))
    ref = {"mean_mean": (1.11510791, 0.33273381), "mean_sigma": (1.04714887, 0.34292767),
           "haebara": (1.08030261, 0.33999163), "stocking_lord": (1.09037942, 0.33370045)}
    for k, (A, B) in ref.items():
        for stat, obs, exp in (("A", r[k]["A"], A), ("B", r[k]["B"], B)):
            assert obs == pytest.approx(exp, abs=1e-6)
            record_validation(method=f"IRT linking ({k.replace('_', '-')})", endpoint="/api/irt/test-equating",
                              statistic=stat, observed=float(obs), expected=float(exp),
                              reference="equateIRT 2.5.2 (R 4.5.2)", reference_kind="r",
                              dataset="6 common 3PL items (constructed)", tolerance_abs=1e-6)


def test_3pl_information_and_cat():
    # 3PL information equals (dP/dtheta)^2 / (P Q)
    a, b, c, th = 1.3, 0.4, 0.2, np.linspace(-3, 3, 13)
    h = 1e-6
    dP = (I._icc_3pl(th + h, a, b, c) - I._icc_3pl(th - h, a, b, c)) / (2 * h)
    P = I._icc_3pl(th, a, b, c)
    assert np.allclose(I._info_3pl(th, a, b, c), dP ** 2 / (P * (1 - P)), rtol=1e-6)
    bank = [dict(a=1 + 0.1 * (i % 5), b=-2 + 0.2 * i, c=0.15) for i in range(21)]
    r1 = I.cat_simulation(I.CATSimulationRequest(item_bank=bank, n_items=10, true_theta=1.0, seed=7))
    r2 = I.cat_simulation(I.CATSimulationRequest(item_bank=bank, n_items=10, true_theta=1.0, seed=7))
    assert r1["responses"] == r2["responses"] and len(set(r1["administered_items"])) == 10
    r3 = I.cat_simulation(I.CATSimulationRequest(item_bank=bank, n_items=21, stopping="se", se_threshold=0.6, seed=7))
    assert r3["final_se"] <= 0.6 or r3["n_items_administered"] == 21
