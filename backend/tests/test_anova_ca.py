"""
Mixed ANOVA, MANOVA / LDA (/api/advanced-anova/*) and correspondence analysis
(/api/correspondence/*) against R 4.5.2.

    afex::aov_ez("id", "y", d, between="grp", within="cond", type=3)   # mixed ANOVA, GG/HF, Mauchly
    manova(cbind(Sepal.Length, Sepal.Width, Petal.Length) ~ Species, iris[c(1:15,51:65,101:115),])
    MASS::lda(...)$scaling
    ca::ca(N); ca::mjca(X, lambda="indicator"); ca::mjca(X, lambda="adjusted")

Note on Mauchly's p: R's mauchly.test uses the number of conditions instead of the
contrast dimension in one term of Box's correction; the textbook formula used here
differs from R in the 5th decimal of p (W is identical).
"""

import numpy as np
import pytest

from app.routers import advanced_anova as A
from app.routers import correspondence as C

MIXED_Y = [[10, 12, 15, 14], [9, 11, 13, 15], [11, 14, 16, 15], [8, 10, 12, 14], [12, 13, 17, 18],
           [14, 15, 15, 16], [13, 15, 14, 17], [15, 17, 18, 17], [12, 14, 15, 15], [16, 16, 17, 19],
           [14, 16, 16, 18], [13, 14, 15, 16]]
MIXED_G = ["A"] * 5 + ["B"] * 7

# First 15 of each species from R's iris (Sepal.Length, Sepal.Width, Petal.Length)
IRIS45 = np.array([
    [5.1, 3.5, 1.4], [4.9, 3.0, 1.4], [4.7, 3.2, 1.3], [4.6, 3.1, 1.5], [5.0, 3.6, 1.4], [5.4, 3.9, 1.7], [4.6, 3.4, 1.4],
    [5.0, 3.4, 1.5], [4.4, 2.9, 1.4], [4.9, 3.1, 1.5], [5.4, 3.7, 1.5], [4.8, 3.4, 1.6], [4.8, 3.0, 1.4], [4.3, 3.0, 1.1],
    [5.8, 4.0, 1.2],
    [7.0, 3.2, 4.7], [6.4, 3.2, 4.5], [6.9, 3.1, 4.9], [5.5, 2.3, 4.0], [6.5, 2.8, 4.6], [5.7, 2.8, 4.5], [6.3, 3.3, 4.7],
    [4.9, 2.4, 3.3], [6.6, 2.9, 4.6], [5.2, 2.7, 3.9], [5.0, 2.0, 3.5], [5.9, 3.0, 4.2], [6.0, 2.2, 4.0], [6.1, 2.9, 4.7],
    [5.6, 2.9, 3.6],
    [6.3, 3.3, 6.0], [5.8, 2.7, 5.1], [7.1, 3.0, 5.9], [6.3, 2.9, 5.6], [6.5, 3.0, 5.8], [7.6, 3.0, 6.6], [4.9, 2.5, 4.5],
    [7.3, 2.9, 6.3], [6.7, 2.5, 5.8], [7.2, 3.6, 6.1], [6.5, 3.2, 5.1], [6.4, 2.7, 5.3], [6.8, 3.0, 5.5], [5.7, 2.5, 5.0],
    [5.8, 2.8, 5.1]])
SPECIES = ["setosa"] * 15 + ["versicolor"] * 15 + ["virginica"] * 15


def chk(rec, method, endpoint, stat, obs, exp, tol, dataset):
    assert obs == pytest.approx(exp, abs=tol), f"{method} {stat}: {obs} vs {exp}"
    rec(method=method, endpoint=endpoint, statistic=stat, observed=float(obs), expected=float(exp),
        reference="R 4.5.2", reference_kind="r", dataset=dataset, tolerance_abs=tol)


@pytest.mark.validation
def test_mixed_anova_matches_afex_type3(record_validation):
    r = A.mixed_anova(A.MixedAnovaRequest(data=MIXED_Y, groups=MIXED_G))
    e, ds, m = "/api/advanced-anova/mixed-anova", "12 subjects (5 + 7, unbalanced) x 4 conditions", "Mixed ANOVA (Type III)"
    chk(record_validation, m, e, "F between", r["between_subjects"]["F"], 9.878011, 1e-5, ds)
    chk(record_validation, m, e, "F within", r["within_subjects"]["F"], 70.878308, 1e-5, ds)
    chk(record_validation, m, e, "F interaction", r["interaction"]["F"], 9.266652, 1e-5, ds)
    chk(record_validation, m, e, "partial eta^2 within", r["within_subjects"]["partial_eta_squared"], 0.8763575, 1e-6, ds)
    chk(record_validation, m, e, "GG epsilon", r["within_subjects"]["epsilon_gg"], 0.6171613, 1e-6, ds)
    chk(record_validation, m, e, "HF epsilon", r["within_subjects"]["epsilon_hf"], 0.7513156, 1e-6, ds)
    chk(record_validation, m, e, "GG-corrected p (within)", r["corrected_within"]["p"], 3.203459e-09, 1e-14, ds)
    assert r["sphericity_test"]["W"] == pytest.approx(0.3626125309, abs=1e-8)
    assert r["sphericity_test"]["p"] == pytest.approx(0.1173623236, abs=1e-4)


@pytest.mark.validation
def test_manova_and_lda_match_r(record_validation):
    X = [IRIS45[:, j].tolist() for j in range(3)]
    r = A.manova(A.ManovaRequest(dependent_variables=X, groups=SPECIES))
    e, ds = "/api/advanced-anova/manova", "iris, first 15 per species, 3 DVs"
    mt = r["multivariate_tests"]
    chk(record_validation, "MANOVA", e, "Pillai trace", mt["pillai_trace"]["value"], 1.1610724271, 1e-9, ds)
    chk(record_validation, "MANOVA", e, "Pillai F", mt["pillai_trace"]["F"], 18.9146123583, 1e-8, ds)
    chk(record_validation, "MANOVA", e, "Wilks lambda", mt["wilks_lambda"]["value"], 0.0229595413, 1e-9, ds)
    chk(record_validation, "MANOVA", e, "Wilks F", mt["wilks_lambda"]["F"], 74.6614919464, 1e-7, ds)
    chk(record_validation, "MANOVA", e, "Hotelling-Lawley F", mt["hotelling_lawley"]["F"], 224.5060180810, 1e-6, ds)
    chk(record_validation, "MANOVA", e, "Roy F", mt["roys_largest_root"]["F"], 468.8450961158, 1e-6, ds)
    assert mt["wilks_lambda"]["p"] == pytest.approx(1.035927e-30, rel=1e-4)  # not floored at 1e-16
    ld1 = r["discriminant_coefficients"][0]["coefficients"]
    assert list(ld1.values()) == pytest.approx([-1.572686709, -1.000177681, 3.726755969], abs=1e-6)


@pytest.mark.validation
def test_correspondence_analysis_matches_ca_package(record_validation):
    r = C.simple_ca(C.SimpleCARequest(table=[[20, 10, 5, 8], [10, 25, 10, 6], [5, 10, 30, 9]]))
    e, ds = "/api/correspondence/simple", "3x4 contingency table"
    chk(record_validation, "Simple correspondence analysis", e, "eigenvalue 1", r["dimensions"][0]["eigenvalue"], 0.2087690884, 1e-9, ds)
    chk(record_validation, "Simple correspondence analysis", e, "eigenvalue 2", r["dimensions"][1]["eigenvalue"], 0.0789075796, 1e-9, ds)
    rows1 = [abs(x["coordinates"][0]) for x in r["row_coordinates"]]
    assert rows1 == pytest.approx([0.51840902, 0.17152084, 0.57479909], abs=1e-7)
    assert [x["cos2"][0] for x in r["row_coordinates"]] == pytest.approx([0.74682966, 0.17472131, 0.96360178], abs=1e-7)
    assert len(r["dimensions"]) == 2  # no trivial zero-inertia dimension

    X = [["x", "p", "u"], ["y", "p", "v"], ["x", "q", "u"], ["z", "q", "v"], ["y", "p", "u"],
         ["x", "q", "u"], ["z", "p", "v"], ["z", "q", "v"], ["y", "q", "u"], ["x", "p", "v"]]
    m = C.multiple_ca(C.MCARequest(data=X, n_dimensions=4))
    e2, ds2 = "/api/correspondence/multiple", "10 observations, 3 categorical variables"
    assert m["total_inertia"] == pytest.approx(4 / 3)
    assert [d["eigenvalue"] for d in m["dimensions"]] == pytest.approx([0.5527621359, 0.4044062151, 0.2989969442, 0.0771680381], abs=1e-9)
    chk(record_validation, "Multiple correspondence analysis", e2, "adjusted % dim 1 (Greenacre)", m["dimensions"][0]["adjusted_percent"], 60.186249, 1e-5, ds2)
    chk(record_validation, "Multiple correspondence analysis", e2, "adjusted eigenvalue 1 (Benzecri)", m["dimensions"][0]["adjusted_eigenvalue"], 0.1083352486, 1e-9, ds2)


def test_validation(client):
    assert client.post("/api/correspondence/supplementary", json={"original_table": [[1, -2], [3, 4]], "supplementary_rows": [[1, 1]]}).status_code == 422
    assert client.post("/api/advanced-anova/mixed-anova", json={"data": MIXED_Y, "groups": MIXED_G, "group_labels": ["only one"]}).status_code == 400
