"""Network psychometrics checked against qgraph 1.9.8, networktools 1.6.1, bootnet 1.6 and
NetworkComparisonTest 2.2.4 (R 4.5.2).

Data: tests/data/network_sim.csv, simulated from two known Gaussian graphical models
(8 variables; group 1 n = 500, group 2 n = 450).
R reference (qgraph's graphical lasso uses thr = 1e-4, so agreement is to about 1e-5):

    d <- read.csv("network_sim.csv"); X <- d[d$group == 1, -1]; X2 <- d[d$group == 2, -1]
    net <- EBICglasso(cor(X), n = nrow(X))
    centrality(net)
    bridge(net, communities = c(rep("anx", 4), rep("dep", 4)))
    NCT(X, X2, it = 20, gamma = 0.5)$glstrinv.real / $nwinv.real
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers.network_psychometrics import (
    BridgeRequest, CompareRequest, EstimateRequest, StabilityRequest,
    _cs_coefficient, _ebicglasso, bridge_centrality, compare_networks,
    estimate_network, stability_analysis,
)

DATA = np.genfromtxt(Path(__file__).parent / "data" / "network_sim.csv", delimiter=",", skip_header=1)
G1 = DATA[DATA[:, 0] == 1, 1:]
G2 = DATA[DATA[:, 0] == 2, 1:]
P = 8


def _from_upper_colmajor(v):
    """Rebuild a symmetric matrix from R's m[upper.tri(m)] (column-major order)."""
    m = np.zeros((P, P))
    k = 0
    for j in range(P):
        for i in range(j):
            m[i, j] = m[j, i] = v[k]
            k += 1
    return m


R_NET = _from_upper_colmajor([
    0.2911270102, 0.0583771293, 0.3017614296, 0.1302879639, 0.0215898872, 0.2073238179, 0.0000000000,
    0.0633449596, -0.0419668964, 0.1568574806, 0.0000000000, 0.1145152009, 0.0000000000, 0.0000000000,
    0.1709345311, -0.0106304822, 0.0000000000, -0.0937965951, 0.0000000000, 0.0000000000, 0.2823687041,
    -0.0436160496, 0.0000000000, 0.0000000000, 0.0000000000, 0.1146418650, 0.0162581890, 0.2448055932,
])
R_NET2 = _from_upper_colmajor([
    0.0629114596, 0.0000000000, 0.3182746755, 0.0948842608, 0.0218194833, 0.2842811460, 0.0000000000,
    0.0000000000, 0.0000000000, 0.2878559127, 0.0000000000, 0.1243114007, 0.0000000000, 0.0000000000,
    0.2445382719, 0.0000000000, 0.0248017972, -0.1496637164, 0.0000000000, 0.0930050409, 0.2435223609,
    0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.1376009164, 0.0000000000, 0.1793816441,
])
R_PCOR = _from_upper_colmajor([
    0.3131756747, 0.0605286003, 0.3302792210, 0.1562367015, 0.0199352132, 0.2552090477, 0.0080006923,
    0.0997039139, -0.1415019953, 0.2199360405, 0.0304507880, 0.1461691934, 0.0256531005, -0.0830333704,
    0.2219277548, -0.0340934604, -0.0292900940, -0.1444844426, 0.0629523825, -0.0679347167, 0.3357972699,
    -0.1120440311, 0.0445815665, 0.0352131439, 0.0223658760, 0.1526461486, 0.0161652985, 0.2821149973,
])


@pytest.mark.validation
def test_ebicglasso_matches_qgraph(record_validation):
    res = estimate_network(EstimateRequest(data=G1.tolist()))
    net = np.array(res["adjacency_matrix"])
    assert np.array_equal(net != 0, R_NET != 0)  # same edges selected
    assert np.max(np.abs(net - R_NET)) < 5e-5
    assert res["estimation"]["lambda"] == pytest.approx(0.0453480893, rel=1e-8)
    assert res["estimation"]["ebic"] == pytest.approx(3727.557880, abs=1e-3)
    assert res["n_edges"] == 18
    record_validation(method="EBICglasso network", endpoint="/api/network-psychometrics/estimate", statistic="EBIC of selected network",
                      observed=res["estimation"]["ebic"], expected=3727.557880, tolerance_abs=1e-3, reference="qgraph 1.9.8 / networktools 1.6.1 / NetworkComparisonTest 2.2.4 (R 4.5.2)", reference_kind="r", dataset="simulated network_sim (n = 500 / 450)")
    record_validation(method="EBICglasso network", endpoint="/api/network-psychometrics/estimate", statistic="max |edge weight difference|",
                      observed=float(np.max(np.abs(net - R_NET))), expected=0.0, tolerance_abs=5e-5, reference="qgraph 1.9.8 / networktools 1.6.1 / NetworkComparisonTest 2.2.4 (R 4.5.2)", reference_kind="r", dataset="simulated network_sim (n = 500 / 450)")


def test_ebicglasso_solution_satisfies_kkt():
    """At the selected penalty the solution is the exact graphical lasso optimum."""
    from sklearn.covariance import graphical_lasso
    R = np.corrcoef(G1, rowvar=False)
    _, info = _ebicglasso(R, len(G1))
    lam = info["lambda"]
    _, K = graphical_lasso(R, alpha=lam, tol=1e-8, enet_tol=1e-10, max_iter=2000)
    W = np.linalg.inv(K)
    off = ~np.eye(P, dtype=bool)
    assert np.allclose(np.diag(W), 1.0, atol=1e-6)  # diagonal not penalized
    g = (W - R)[off]
    k = K[off]
    assert np.all(np.abs(g) <= lam + 1e-6)
    assert np.allclose(g[k != 0], lam * np.sign(k[k != 0]), atol=1e-6)


def test_partial_correlations_match_r():
    res = estimate_network(EstimateRequest(data=G1.tolist(), method="partial"))
    assert np.max(np.abs(np.array(res["adjacency_matrix"]) - R_PCOR)) < 1e-9


def test_centrality_matches_qgraph():
    res = estimate_network(EstimateRequest(data=G1.tolist()))
    t = res["centrality_table"]
    strength = [0.5340386352, 0.7923384875, 0.7032258683, 0.5160591497, 0.5477457328, 0.5840766251, 0.6316013746, 0.4193216969]
    ei = [0.4255455715, 0.7923384875, 0.4316988853, 0.5160591497, 0.4638119399, 0.5840766251, 0.4227472200, 0.3320895976]
    clo = [0.012566043340, 0.014978194389, 0.015738467213, 0.014322683319, 0.014265039163, 0.016079856332, 0.014055424557, 0.011569942958]
    bet = [0, 10, 6, 6, 4, 10, 8, 0]
    assert [r["strength"] for r in t] == pytest.approx(strength, abs=2e-4)
    assert [r["expected_influence"] for r in t] == pytest.approx(ei, abs=2e-4)
    assert [r["closeness"] for r in t] == pytest.approx(clo, rel=1e-3)
    assert [r["betweenness"] for r in t] == bet


@pytest.mark.validation
def test_centrality_on_r_network_exact(record_validation):
    """Given qgraph's own network, the indices agree to rounding."""
    from app.routers.network_psychometrics import _compute_centrality
    t = _compute_centrality(R_NET, [str(i) for i in range(P)])["centrality_table"]
    clo = [0.012566043340, 0.014978194389, 0.015738467213, 0.014322683319, 0.014265039163, 0.016079856332, 0.014055424557, 0.011569942958]
    assert [r["closeness"] for r in t] == pytest.approx(clo, abs=1e-11)
    assert [r["betweenness"] for r in t] == [0, 10, 6, 6, 4, 10, 8, 0]
    record_validation(method="Network centrality", endpoint="/api/network-psychometrics/centrality", statistic="closeness (node 1)",
                      observed=t[0]["closeness"], expected=clo[0], tolerance_abs=1e-11, reference="qgraph 1.9.8 / networktools 1.6.1 / NetworkComparisonTest 2.2.4 (R 4.5.2)", reference_kind="r", dataset="simulated network_sim (n = 500 / 450)")


def test_betweenness_shares_tied_paths():
    # Square 0-1-2-3-0 with equal weights: the two shortest paths 0->2 are tied, so
    # nodes 1 and 3 each get half of that pair (both directions -> 1 per pair).
    W = np.zeros((4, 4))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        W[i, j] = W[j, i] = 0.5
    from app.routers.network_psychometrics import _closeness_betweenness
    _, b = _closeness_betweenness(W)
    assert b.tolist() == [1.0, 1.0, 1.0, 1.0]  # igraph::betweenness(directed) gives 1 each


@pytest.mark.validation
def test_bridge_matches_networktools(record_validation):
    res = bridge_centrality(BridgeRequest(adjacency=R_NET.tolist(), communities=[0, 0, 0, 0, 1, 1, 1, 1]))
    t = res["bridge_centrality_table"]
    exp = {
        "bridge_strength": [0.0542465319, 0.1778601605, 0.1357634915, 0.1568574806, 0.2621693367, 0.1145152009, 0.1044270773, 0.0436160496],
        "bridge_betweenness": [0, 6, 1, 3, 3, 7, 3, 0],
        "bridge_closeness": [0.064808280500, 0.077104465300, 0.068369731392, 0.080864395513, 0.086698719286, 0.088551202921, 0.067411017692, 0.056398988177],
        "bridge_expected_influence": [-0.0542465319, 0.1778601605, -0.1357634915, 0.1568574806, 0.1782355438, 0.1145152009, -0.1044270773, -0.0436160496],
        "bridge_expected_influence_2step": [-0.0119463894, 0.1963476260, -0.1141705294, 0.1702775799, 0.2642628595, 0.1851528626, -0.1410954815, -0.0678119535],
    }
    for k, v in exp.items():
        assert [r[k] for r in t] == pytest.approx(v, abs=1e-9), k
    record_validation(method="Bridge centrality", endpoint="/api/network-psychometrics/bridge", statistic="bridge EI 2-step (node 5)",
                      observed=t[4]["bridge_expected_influence_2step"], expected=0.2642628595, tolerance_abs=1e-9, reference="qgraph 1.9.8 / networktools 1.6.1 / NetworkComparisonTest 2.2.4 (R 4.5.2)", reference_kind="r", dataset="simulated network_sim (n = 500 / 450)")


@pytest.mark.validation
def test_nct_observed_statistics_match(record_validation):
    names = ["worry", "restless", "tension", "irritable", "sad_mood", "anhedonia", "fatigue", "sleep"]
    res = compare_networks(CompareRequest(data_group1=G1.tolist(), data_group2=G2.tolist(), n_permutations=20, variable_names=names))
    assert np.max(np.abs(np.array(res["adjacency_group2"]) - R_NET2)) < 5e-5
    assert res["global_strength"]["S"] == pytest.approx(0.0973516988, abs=2e-4)
    assert res["network_invariance"]["M"] == pytest.approx(0.2282155506, abs=1e-4)
    record_validation(method="Network Comparison Test", endpoint="/api/network-psychometrics/compare", statistic="global strength difference S",
                      observed=res["global_strength"]["S"], expected=0.0973516988, tolerance_abs=2e-4, reference="qgraph 1.9.8 / networktools 1.6.1 / NetworkComparisonTest 2.2.4 (R 4.5.2)", reference_kind="r", dataset="simulated network_sim (n = 500 / 450)")
    for key in ("network_invariance", "global_strength"):
        pv = res[key]["p_value"]
        assert 1 / 21 <= pv <= 1  # (b + 1) / (B + 1)
    # The edge whose weight was changed in the simulation (worry - restless) differs most.
    top = max(res["edge_tests"], key=lambda e: e["abs_difference"])
    assert (top["from"], top["to"]) == ("worry", "restless")


def test_nct_is_reproducible_with_seed():
    a = compare_networks(CompareRequest(data_group1=G1.tolist(), data_group2=G2.tolist(), n_permutations=20, seed=7))
    b = compare_networks(CompareRequest(data_group1=G1.tolist(), data_group2=G2.tolist(), n_permutations=20, seed=7))
    assert a["global_strength"]["p_value"] == b["global_strength"]["p_value"]
    assert a["network_invariance"]["p_value"] == b["network_invariance"]["p_value"]


def test_cs_coefficient_rule():
    # More than 95% of correlations must exceed 0.7; the largest such level counts even
    # if an intermediate level fails (bootnet takes max over all passing levels).
    levels = [0.1, 0.3, 0.5, 0.7]
    good = [0.9] * 96 + [0.5] * 4    # 96% above 0.7
    edge = [0.9] * 95 + [0.5] * 5    # exactly 95%: not more than 95%
    assert _cs_coefficient(levels, [good, good, edge, good]) == 0.7
    assert _cs_coefficient(levels, [edge, edge, edge, edge]) == 0.0
    assert _cs_coefficient(levels, [good, good, edge, edge]) == 0.3


def test_stability_runs_and_is_seeded():
    req = StabilityRequest(data=G1.tolist(), n_boots=10, drop_proportions=[0.25, 0.5, 0.7], seed=3)
    a = stability_analysis(req)
    b = stability_analysis(req)
    assert a["cs_coefficient"] == b["cs_coefficient"]
    assert a["stability_curves"]["strength"][0]["keep_n"] == 375
    assert 0 <= a["cs_coefficient"]["strength"] <= 0.7


def test_rejects_singular_and_missing():
    X = G1[:, :3]
    X = np.column_stack([X, X[:, 0] + X[:, 1]])
    with pytest.raises(HTTPException):
        estimate_network(EstimateRequest(data=X.tolist()))
    with pytest.raises(HTTPException):
        estimate_network(EstimateRequest(data=[[1, 2], [3, float("nan")], [2, 2], [4, 1]]))
