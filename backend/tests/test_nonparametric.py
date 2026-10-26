"""
Bootstrap, permutation and robust statistics (/api/nonparametric/*, Stats Studio).

    Exact permutation p: scipy.stats.permutation_test(..., n_resamples=np.inf) and enumeration
    Huber location: R MASS::huber(d, k = 1.345)$mu = 14.5438425219 (MASS stops at tol 1e-6)
"""

import itertools

import numpy as np
import pytest
from scipy import stats

from app.routers import nonparametric as N

D = [12, 15, 11, 19, 14, 16, 13, 18, 17, 10, 25, 9]
G1 = [12, 15, 11, 19, 14, 16, 13]
G2 = [9, 8, 12, 7, 11, 10]


@pytest.mark.validation
def test_exact_permutation_p(record_validation):
    r = N.permutation_test(N.PermutationTestRequest(group1=G1, group2=G2))
    ref = stats.permutation_test((np.array(G1), np.array(G2)), lambda a, b, axis: a.mean(axis) - b.mean(axis),
                                 permutation_type="independent", n_resamples=np.inf).pvalue
    assert r["method"].startswith("exact") and r["p_value"] == pytest.approx(ref, rel=1e-12)
    record_validation(method="Permutation test (exact)", endpoint="/api/nonparametric/permutation-test",
                      statistic="p", observed=float(r["p_value"]), expected=float(ref),
                      reference="scipy 1.14 permutation_test (exact) and full enumeration", reference_kind="scipy-ref",
                      dataset="7 vs 6 observations", tolerance_rel=1e-12)


def test_monte_carlo_permutation_p_is_never_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(3, 1, 30), rng.normal(0, 1, 30)
    r = N.permutation_test(N.PermutationTestRequest(group1=a.tolist(), group2=b.tolist(), n_permutations=2000))
    assert r["method"].startswith("Monte Carlo") and r["p_value"] == pytest.approx(1 / 2001)


@pytest.mark.validation
def test_huber_location_matches_mass(record_validation):
    r = N.robust_statistics(N.RobustStatsRequest(data=D))
    assert r["huber_location"] == pytest.approx(14.5438425219, abs=1e-5)
    record_validation(method="Huber M-estimator of location", endpoint="/api/nonparametric/robust-stats",
                      statistic="location", observed=float(r["huber_location"]), expected=14.5438425219,
                      reference="R MASS::huber (k = 1.345)", reference_kind="r", dataset="12 values", tolerance_abs=1e-5)


def test_bootstrap_is_reproducible_and_bca_close_to_scipy():
    r1 = N.bootstrap_confidence_interval(N.BootstrapCIRequest(data=D))
    r2 = N.bootstrap_confidence_interval(N.BootstrapCIRequest(data=D))
    assert r1["bca_ci"] == r2["bca_ci"] and r1["observed_value"] == pytest.approx(np.mean(D))
    ref = stats.bootstrap((np.array(D),), np.mean, n_resamples=20000, method="BCa", random_state=1).confidence_interval
    assert r1["bca_ci"]["lower"] == pytest.approx(ref.low, abs=0.3) and r1["bca_ci"]["upper"] == pytest.approx(ref.high, abs=0.3)


def test_resample_limits(client):
    assert client.post("/api/nonparametric/bootstrap-ci", json={"data": D, "n_bootstrap": 10_000_000}).status_code == 400
    assert client.post("/api/nonparametric/permutation-test", json={"group1": G1, "group2": G2, "statistic": "ks", "alternative": "greater"}).status_code == 400
