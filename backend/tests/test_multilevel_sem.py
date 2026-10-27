"""Two-level SEM checked against lavaan 0.7.3 and lme4 (R 4.5.2).

Data (simulated): tests/data/twolevel_path.csv — 654 people in 80 clusters (4-12 each);
x, m, y vary within clusters, w is a cluster-level variable. tests/data/twolevel_cfa.csv —
the same clusters, four items with one factor at each level.
lavaan reference: sem(model, data, cluster = "cluster", fixed.x = FALSE) / cfa(model, data,
cluster = "cluster"). lavaan's unrestricted (h1) model is fitted by EM to a looser tolerance
than ours, so chi-square and SRMR agree to about 1e-3; estimates and SEs to about 1e-6.
"""

from pathlib import Path

import pandas as pd
import pytest

from app.routers.multilevel_sem import (
    CrossLevelMediationRequest, MultilevelCFARequest, RandomSlopesSEMRequest, TwoLevelSEMRequest,
    WithinBetweenDecompRequest, cross_level_mediation, measurement_model_multilevel, msem_two_level,
    random_slopes_sem, within_between_decomposition,
)

HERE = Path(__file__).parent / "data"
PATH = pd.read_csv(HERE / "twolevel_path.csv")
CFA = pd.read_csv(HERE / "twolevel_cfa.csv")


def _find(rows, lhs, op, rhs=""):
    return next(r for r in rows if r["lhs"] == lhs and r["op"] == op and r["rhs"] == rhs)


@pytest.mark.validation
def test_two_level_path_model_matches_lavaan(record_validation):
    res = msem_two_level(TwoLevelSEMRequest(
        data=PATH.to_dict("records"), group_ids="cluster",
        within_model=["m ~ x", "y ~ m"], between_model=["m ~ x + w", "y ~ m"]))
    ref = {
        "within": [("m", "~", "x", 0.40852795, 0.03802144), ("y", "~", "m", 0.43856329, 0.03757349),
                   ("m", "~~", "m", 0.88326143, 0.05125130), ("y", "~~", "y", 0.89060774, 0.05234113),
                   ("x", "~~", "x", 1.05535719, 0.06253194)],
        "between": [("m", "~", "x", 0.49259811, 0.17471934), ("m", "~", "w", 0.29982816, 0.04575177),
                    ("y", "~", "m", 1.09042326, 0.17501744), ("m", "~~", "m", 0.03485160, 0.02024758),
                    ("y", "~~", "y", 0.05784312, 0.03071932), ("x", "~~", "x", 0.12250931, 0.04246546),
                    ("x", "~~", "w", -0.00904249, 0.05155592), ("w", "~~", "w", 0.83476634, 0.13198792),
                    ("m", "~1", "", 0.10286026, 1.72601327), ("y", "~1", "", -2.52654478, 0.87166930),
                    ("x", "~1", "", 9.88397854, 0.05680817), ("w", "~1", "", -0.00135935, 0.10214979)],
    }
    for lvl, rows in ref.items():
        for lhs, op, rhs, est, se in rows:
            r = _find(res["estimates"][lvl], lhs, op, rhs)
            assert r["est"] == pytest.approx(est, abs=2e-5), (lvl, lhs, op, rhs)
            assert r["se"] == pytest.approx(se, abs=2e-5), (lvl, lhs, op, rhs)
    fit = res["fit_indices"]
    assert fit["loglik"] == pytest.approx(-2887.41131765, abs=1e-5)
    record_validation(method="Two-level path model (ML)", endpoint="/api/multilevel-sem/msem-two-level", statistic="log-likelihood",
                      observed=fit["loglik"], expected=-2887.41131765, tolerance_abs=1e-5, reference="lavaan 0.7.3 (R 4.5.2)", reference_kind="r", dataset="simulated twolevel_path (654 people, 80 clusters)")
    assert fit["n_parameters"] == 17 and fit["df"] == 3
    assert fit["aic"] == pytest.approx(5808.82263529, abs=1e-4)
    assert fit["bic"] == pytest.approx(5885.03546027, abs=1e-4)
    assert fit["chi_square"] == pytest.approx(26.24458043, abs=2e-3)
    assert fit["baseline_chi_square"] == pytest.approx(348.67353844, abs=2e-3)
    assert fit["baseline_df"] == 8
    assert fit["cfi"] == pytest.approx(0.93176875, abs=1e-4)
    assert fit["tli"] == pytest.approx(0.81805001, abs=3e-4)
    assert fit["rmsea"] == pytest.approx(0.10884572, abs=1e-5)
    assert fit["srmr_within"] == pytest.approx(0.07062770, abs=2e-4)
    assert fit["srmr_between"] == pytest.approx(0.06088356, abs=2e-4)
    ind = {(e["from"], e["to"]): e for e in res["indirect_effects"]["between"]}
    assert ind[("x", "y")]["est"] == pytest.approx(0.53714044, abs=2e-5)
    assert ind[("x", "y")]["se"] == pytest.approx(0.19102988, abs=2e-5)
    assert res["indirect_effects"]["within"][0]["est"] == pytest.approx(0.17916537, abs=2e-6)
    assert res["indirect_effects"]["within"][0]["se"] == pytest.approx(0.02277680, abs=2e-6)
    assert res["icc"] == pytest.approx({"m": 0.11875857, "y": 0.16280392, "x": 0.10467020}, abs=2e-4)


def _mediation(model_type, x, y):
    return cross_level_mediation(CrossLevelMediationRequest(
        x=PATH[x].tolist(), m=PATH["m"].tolist(), y=PATH[y].tolist(),
        group_ids=PATH["cluster"].tolist(), model_type=model_type, n_monte_carlo=5000))


@pytest.mark.validation
def test_mediation_111_matches_lavaan(record_validation):
    res = _mediation("1-1-1", "x", "y")
    p = {(r["path"], r["level"]): r for r in res["path_coefficients"]}
    assert p[("a (X->M)", "within")]["estimate"] == pytest.approx(0.40692065, abs=2e-6)
    assert p[("b (M->Y)", "between")]["estimate"] == pytest.approx(1.19179765, abs=2e-5)
    assert p[("b (M->Y)", "between")]["se"] == pytest.approx(0.25498850, abs=2e-5)
    ind = {e["effect"]: e for e in res["indirect_effects"]}
    assert ind["indirect (within)"]["est"] == pytest.approx(0.14448198, abs=2e-6)
    assert ind["indirect (within)"]["se"] == pytest.approx(0.02145985, abs=2e-6)
    assert ind["indirect (between)"]["est"] == pytest.approx(0.58991624, abs=2e-5)
    assert ind["indirect (between)"]["se"] == pytest.approx(0.31006839, abs=2e-5)
    record_validation(method="Multilevel mediation 1-1-1", endpoint="/api/multilevel-sem/cross-level-mediation", statistic="between indirect effect",
                      observed=ind["indirect (between)"]["est"], expected=0.58991624, tolerance_abs=2e-5, reference="lavaan 0.7.3 (R 4.5.2)", reference_kind="r", dataset="simulated twolevel_path (654 people, 80 clusters)")
    ctx = {c["path"]: c for c in res["contextual_effects"]}
    assert ctx["a"]["est"] == pytest.approx(0.08805955, abs=2e-5)
    assert ctx["a"]["se"] == pytest.approx(0.23179355, abs=2e-5)
    assert res["fit_indices"]["loglik"] == pytest.approx(-2786.60838569, abs=1e-4)
    assert ind["indirect (within)"]["ci_lower_mc"] < 0.14448 < ind["indirect (within)"]["ci_upper_mc"]


def test_mediation_211_and_112_match_lavaan():
    r211 = _mediation("2-1-1", "w", "y")
    ind = r211["indirect_effects"]
    assert len(ind) == 1 and ind[0]["effect"] == "indirect (between)"
    assert ind[0]["est"] == pytest.approx(0.26290370, abs=2e-5)
    assert ind[0]["se"] == pytest.approx(0.10573359, abs=2e-5)
    assert r211["fit_indices"]["loglik"] == pytest.approx(-1979.80887172, abs=1e-4)
    r112 = _mediation("1-1-2", "x", "w")
    ind = r112["indirect_effects"][0]
    assert ind["est"] == pytest.approx(1.10797056, abs=5e-5)
    assert ind["se"] == pytest.approx(0.57899856, abs=5e-5)
    assert r112["fit_indices"]["loglik"] == pytest.approx(-1976.71531900, abs=1e-4)


def test_mediation_checks_levels():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _mediation("2-1-1", "x", "y")  # x varies within clusters


@pytest.mark.validation
def test_two_level_cfa_matches_lavaan(record_validation):
    res = measurement_model_multilevel(MultilevelCFARequest(
        data=CFA.to_dict("records"), group_ids="cluster", factor_model={"f": ["y1", "y2", "y3", "y4"]}))
    ref = {
        "within": [("f", "=~", "y2", 0.82916711, 0.03875240), ("f", "=~", "y3", 0.75476968, 0.03851665),
                   ("f", "=~", "y4", 0.94998179, 0.04416180), ("y1", "~~", "y1", 0.38971867, 0.03388383),
                   ("f", "~~", "f", 0.88868307, 0.07613514)],
        "between": [("f", "=~", "y2", 0.96203547, 0.10892542), ("f", "=~", "y3", 0.62627476, 0.09550724),
                    ("y1", "~~", "y1", -0.00271167, 0.01409985), ("y4", "~~", "y4", 0.03850952, 0.01789488),
                    ("f", "~~", "f", 0.23789890, 0.06549520), ("y1", "~1", "", 2.14681423, 0.07076759)],
    }
    for lvl, rows in ref.items():
        for lhs, op, rhs, est, se in rows:
            r = _find(res["estimates"][lvl], lhs, op, rhs)
            assert r["est"] == pytest.approx(est, abs=2e-5), (lvl, lhs, op, rhs)
            assert r["se"] == pytest.approx(se, abs=2e-5), (lvl, lhs, op, rhs)
    fit = res["fit_indices"]
    assert fit["loglik"] == pytest.approx(-3222.81420374, abs=1e-4)
    record_validation(method="Two-level CFA (ML)", endpoint="/api/multilevel-sem/measurement-model-multilevel", statistic="log-likelihood",
                      observed=fit["loglik"], expected=-3222.81420374, tolerance_abs=1e-4, reference="lavaan 0.7.3 (R 4.5.2)",
                      reference_kind="r", dataset="simulated twolevel_cfa (654 people, 80 clusters)")
    assert fit["chi_square"] == pytest.approx(2.49786179, abs=3e-3)
    assert fit["df"] == 4 and fit["baseline_df"] == 12
    assert fit["baseline_chi_square"] == pytest.approx(1444.90470963, abs=3e-3)
    assert res["icc_per_item"] == pytest.approx({"y1": 0.15422689, "y2": 0.20919600, "y3": 0.12573923, "y4": 0.16203207}, abs=2e-4)
    assert any("Negative variance" in n for n in res["notes"])  # y1's between residual variance is -0.003


def test_decomposition_icc_matches_lavaan():
    res = within_between_decomposition(WithinBetweenDecompRequest(
        data=PATH.to_dict("records"), group_ids="cluster", variables=["m", "y", "x"]))
    icc = {v: s["icc_ml"] for v, s in res["variable_summary"].items()}
    assert icc == pytest.approx({"m": 0.11673082, "y": 0.16403692, "x": 0.10483997}, abs=2e-4)


@pytest.mark.validation
def test_random_slopes_matches_lme4(record_validation):
    res = random_slopes_sem(RandomSlopesSEMRequest(
        data=PATH.to_dict("records"), group_ids="cluster", outcome="y", predictors=["x", "m"], random_slopes=["x"]))
    fe = {f["name"]: f for f in res["fixed_effects"]}
    # lme4 with optimizer = "bobyqa" (the default nloptwrap stops at REML 1827.844 here; the
    # optimum is a singular fit with slope variance ~3e-5)
    assert fe["Intercept"]["estimate"] == pytest.approx(-0.94393101, abs=2e-5)
    assert fe["x"]["estimate"] == pytest.approx(0.18684027, abs=2e-6)
    assert fe["m"]["estimate"] == pytest.approx(0.40015975, abs=2e-6)
    assert fe["x"]["se"] == pytest.approx(0.03891065, abs=2e-5)
    assert fe["m"]["se"] == pytest.approx(0.03873387, abs=2e-5)
    assert res["model_fit"]["reml_criterion"] == pytest.approx(1827.144900, abs=1e-5)
    record_validation(method="Random-slope multilevel regression (REML)", endpoint="/api/multilevel-sem/random-slopes-sem", statistic="slope SE",
                      observed=fe["x"]["se"], expected=0.03891065, tolerance_abs=2e-5, reference="lme4 2.0.6, bobyqa (R 4.5.2)", reference_kind="r", dataset="simulated twolevel_path (654 people, 80 clusters)")
    assert res["residual_variance"] == pytest.approx(0.85635953, abs=1e-5)
    assert res["random_effects"]["variances"]["Group"] == pytest.approx(0.17032426, abs=1e-4)
    ctx = res["contextual_effects"]["x"]
    assert ctx["within_effect"] == pytest.approx(0.35463293, abs=1e-5)
    assert ctx["between_effect"] == pytest.approx(0.32477047, abs=1e-4)
    assert ctx["contextual_se"] == pytest.approx(0.13011848, abs=1e-4)
