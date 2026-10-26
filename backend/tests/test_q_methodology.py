"""
Q-methodology (/api/q-methodology/analyze, Ethnography) against R qmethod 1.8.4 on the
Lipset (1963) value-pattern data shipped with qmethod (data(lipset), 9 Q-sorts x 33 statements):

    qmethod(lipset[[1]], nfactors = 3, rotation = "varimax")
"""

import collections

import pytest

from app.routers import q_methodology as Q

# One row per Q-sort (participant), 33 statements each
LIPSET = [
    [-1, 0, -2, 0, -2, 1, 0, -1, 0, -1, 1, 1, 2, 3, -1, -4, -3, -3, -1, -4, 3, 2, 3, 1, 1, 0, 4, 2, 2, -3, -2, -2, 4],
    [-1, 0, -1, -3, 2, 3, 1, 1, -4, 0, 2, -1, 4, -1, 1, -3, 0, -2, -2, 0, 3, -2, 1, 0, 2, 1, 2, 3, -1, -4, -2, -3, 4],
    [2, -2, -2, 4, -1, 0, -4, -3, 1, -4, -3, 3, 3, 2, 0, -3, 2, -1, 1, 0, 1, -2, 0, 2, -2, 3, 1, -1, 0, -1, -1, 4, 1],
    [3, 1, -3, -1, -1, 3, -3, -2, 0, -4, -1, 0, 3, -2, -3, -4, -1, 2, 4, -2, 4, 1, 1, -2, 0, 1, -1, 2, 0, 2, 1, 2, 0],
    [-4, -1, 3, -1, 1, 1, 4, 2, -4, 4, 2, 0, -1, 3, 0, 2, 0, -2, -2, 0, -1, 2, 1, -3, -3, -3, 3, 0, 1, -2, 1, -2, -1],
    [1, -3, 0, 3, 3, 4, -2, 0, -2, -2, 1, -1, 1, 0, -4, 3, -1, 0, 1, -1, -2, -3, -1, -4, 2, 2, 0, 4, -1, 2, 1, 2, -3],
    [2, 0, -2, 1, 0, 1, -1, -3, 0, -1, 1, -1, 1, 2, -4, 4, 3, 2, 0, 2, -2, -1, -3, -4, 3, -2, 0, 4, -3, 1, -1, 3, -2],
    [-2, 2, 0, -3, -4, 4, 0, -1, -1, -1, 1, -1, 1, -1, 4, 0, 0, 1, -3, 1, 2, 2, 3, 3, -3, 1, -4, 3, 0, -2, -2, -2, 2],
    [3, 1, 0, 1, -4, -3, 2, 2, -2, 0, 0, -2, 1, 4, -1, -2, 2, -1, 2, 1, 3, -3, -3, -2, 1, 0, -1, -4, 0, -1, 3, -1, 4],
]


@pytest.mark.validation
def test_q_analysis_matches_qmethod(record_validation):
    r = Q.q_analyze(Q.QAnalyzeRequest(q_sorts=LIPSET, n_factors=3))
    loa1 = [row["loadings"][0] for row in r["loadings"]]
    assert loa1 == pytest.approx([0.190900, -0.065524, 0.810739, 0.775649, -0.833453, 0.146058, 0.141311, -0.089066, 0.203772], abs=1e-4)
    assert [row["flagged_on"] for row in r["loadings"]] == [2, 2, 1, 1, 1, 3, 3, 2, 3]
    z = r["factor_arrays"][0]["z_scores"][:6]
    assert z == pytest.approx([1.659916, -0.004243, -1.449094, 0.791342, -0.543941, 0.240352], abs=1e-4)
    assert r["factor_arrays"][0]["se"] == pytest.approx(0.277350, abs=1e-4)
    cats = collections.Counter(s["category"] for s in r["statements"])
    assert cats["consensus"] == 2 and cats["distinguishes all"] == 5 and cats["distinguishes factor 1 only"] == 8
    record_validation(method="Q-methodology (PCA, varimax, qmethod flagging)", endpoint="/api/q-methodology/analyze",
                      statistic="factor 1 z-score, statement 1", observed=float(z[0]), expected=1.659916,
                      reference="R qmethod 1.8.4", reference_kind="r", dataset="Lipset (1963), 9 Q-sorts x 33 statements",
                      tolerance_abs=1e-4)


def test_ragged_sorts_are_422(client):
    r = client.post("/api/q-methodology/correlate", json={"q_sorts": [[1, 2, 3], [1, 2]]})
    assert r.status_code == 422
