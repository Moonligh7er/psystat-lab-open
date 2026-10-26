"""
Spatial statistics (/api/spatial/*, Criminology) against R spdep / spatialreg on spatial_data().

    lw <- nb2listw(knn2nb(knearneigh(co, k = 5)), style = "W")
    moran.test(d$y, lw, randomisation = TRUE); lm.RStests(lm(y ~ x1, d), lw, test = "all")
    lagsarlm(y ~ x1, d, lw, method = "eigen")
"""

import pytest

from app.routers import spatial_stats as S


def spatial_data():
    def lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) % 2 ** 31
            yield x / 2 ** 31 * 2 - 1
    g = lcg(21)
    co, x1s, ys = [], [], []
    for i in range(40):
        cx = (i % 8) + 0.35 * next(g); cy = (i // 8) + 0.35 * next(g)
        x1 = 2 + next(g)
        y = 1 + 0.8 * x1 + 0.6 * cx + 0.3 * cy + 0.7 * next(g)
        co.append([round(cx, 5), round(cy, 5)]); x1s.append(round(x1, 5)); ys.append(round(y, 5))
    return co, x1s, ys


def rec(record, method, endpoint, stat, obs, exp, tol):
    assert obs == pytest.approx(exp, abs=tol), f"{method} {stat}"
    record(method=method, endpoint=endpoint, statistic=stat, observed=float(obs), expected=float(exp),
           reference="R spdep 1.4 / spatialreg", reference_kind="r", dataset="40 jittered grid points, kNN (k = 5) weights",
           tolerance_abs=tol)


@pytest.mark.validation
def test_morans_i_matches_spdep(record_validation):
    co, _, y = spatial_data()
    r = S.global_morans_i(S.MoransIRequest(values=y, coordinates=co))
    rec(record_validation, "Global Moran's I", "/api/spatial/morans-i", "I", r["morans_i"], 0.7207706807, 1e-9)
    rec(record_validation, "Global Moran's I", "/api/spatial/morans-i", "variance (randomisation)", r["variance"], 0.0077523362, 1e-10)
    assert r["z_score"] == pytest.approx(8.4773895763, abs=1e-8)


@pytest.mark.validation
def test_lm_tests_and_ml_lag_match_spatialreg(record_validation):
    co, x1, y = spatial_data()
    r = S.spatial_regression(S.SpatialRegressionRequest(y=y, X=[[v] for v in x1], coordinates=co))
    e = "/api/spatial/spatial-regression"
    lm = r["lagrange_multiplier_tests"]
    for key, ref in (("lm_error", 66.5872160311), ("lm_lag", 62.8948865770), ("robust_lm_error", 7.4612871935), ("robust_lm_lag", 3.7689577394)):
        rec(record_validation, "Spatial LM diagnostics", e, key, lm[key]["statistic"], ref, 1e-7)
    L = r["spatial_lag"]
    rec(record_validation, "Spatial lag model (ML)", e, "rho", L["rho"], 0.9408453040, 1e-7)
    rec(record_validation, "Spatial lag model (ML)", e, "rho SE", L["rho_se"], 0.0351329236, 1e-7)
    rec(record_validation, "Spatial lag model (ML)", e, "slope SE", L["coefficients"][1]["se"], 0.1609979326, 1e-7)
    rec(record_validation, "Spatial lag model (ML)", e, "log-likelihood", L["log_likelihood"], -43.9716687786, 1e-6)


def test_local_morans_values_match_spdep():
    co, _, y = spatial_data()
    r = S.local_morans(S.LocalMoransRequest(values=y, coordinates=co))
    assert [l["local_morans_i"] for l in r["locations"][:5]] == pytest.approx([2.15310900, 0.89314444, 0.50391694, 0.11831784, -0.00685061], abs=1e-7)


def test_gwr_reduces_to_ols_with_huge_bandwidth():
    co, x1, y = spatial_data()
    g = S.geographically_weighted_regression(S.GWRRequest(y=y, X=[[v] for v in x1], coordinates=co, bandwidth=1e6))
    o = S.spatial_regression(S.SpatialRegressionRequest(y=y, X=[[v] for v in x1], coordinates=co))
    assert g["coefficient_variation"]["X1"]["mean"] == pytest.approx(o["ols"]["coefficients"][1]["estimate"], rel=1e-8)
    assert g["effective_parameters"] == pytest.approx(2, rel=1e-6)


def test_bad_weight_matrix_is_422(client):
    r = client.post("/api/spatial/local-morans", json={"values": [1, 2, 3, 4], "coordinates": [[0, 0], [1, 0], [0, 1], [1, 1]], "weight_matrix": [[0, 1], [1, 0]]})
    assert r.status_code == 422
