"""
Mediation & moderation (/api/mediation/*, used by Stats Studio).

Moderation references from R 4.5.2 on mod_data() (lm(y ~ xc * wc) with mean-centred
predictors, simple slopes from vcov, Johnson–Neyman by the quadratic in W).
Mediation paths are OLS; the bootstrap is checked against explicit refits.
"""

import numpy as np
import pytest
import statsmodels.api as sm

from app.routers import mediation as M


def mod_data():
    def lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) % 2 ** 31
            yield x / 2 ** 31 * 2 - 1
    g = lcg(11)
    X, W, Y = [], [], []
    for _ in range(40):
        x = 5 + 2 * next(g); w = 3 + 1.5 * next(g); e = next(g)
        X.append(round(x, 4)); W.append(round(w, 4)); Y.append(round(1 + 0.3 * x + 0.2 * w + 0.25 * x * w + 1.2 * e, 4))
    return X, W, Y


@pytest.mark.validation
def test_moderation_matches_r(record_validation):
    x, w, y = mod_data()
    r = M.moderation_analysis(M.ModerationRequest(x=x, w=w, y=y))
    e, ds = "/api/mediation/moderation", "40 observations (LCG), mean-centred"
    def chk(stat, obs, exp, tol):
        assert obs == pytest.approx(exp, abs=tol)
        record_validation(method="Moderation (PROCESS model 1)", endpoint=e, statistic=stat, observed=float(obs),
                          expected=float(exp), reference="R 4.5.2 lm + vcov", reference_kind="r", dataset=ds, tolerance_abs=tol)
    chk("interaction b", r["coefficients"]["x_w_interaction"]["coeff"], 0.2829684636, 1e-9)
    chk("interaction p", r["coefficients"]["x_w_interaction"]["p"], 2.2317499415e-02, 1e-10)
    chk("delta R^2", r["r_squared_change_interaction"], 0.0226918839, 1e-9)
    s = r["simple_slopes"]
    chk("simple slope at -1 SD", s[0]["slope"], 0.8585435950, 1e-9)
    chk("simple slope SE at +1 SD", s[2]["se"], 0.1525133562, 1e-9)
    # J-N roots are -22.37 and 0.919; both fall outside the observed range of W
    assert r["johnson_neyman"]["n_points_found"] == 0


def test_moderation_without_johnson_neyman():
    x, w, y = mod_data()
    r = M.moderation_analysis(M.ModerationRequest(x=x, w=w, y=y, johnson_neyman=False))
    assert r["johnson_neyman"] is None and len(r["visualization"]["slopes"]) == 100


def test_simple_mediation_bootstrap_is_reproducible_and_exact():
    rng = np.random.default_rng(3)
    n = 60
    x = rng.normal(size=n); m = 0.5 * x + rng.normal(size=n); y = 0.4 * m + 0.2 * x + rng.normal(size=n)
    req = dict(x=x.tolist(), m=m.tolist(), y=y.tolist(), n_bootstrap=2000, seed=5)
    r1 = M.simple_mediation(M.SimpleMediationRequest(**req))
    r2 = M.simple_mediation(M.SimpleMediationRequest(**req))
    assert r1["indirect_effect"]["percentile_ci"] == r2["indirect_effect"]["percentile_ci"]
    a = sm.OLS(m, sm.add_constant(x)).fit().params[1]
    b = sm.OLS(y, sm.add_constant(np.column_stack([x, m]))).fit().params[2]
    assert r1["indirect_effect"]["point_estimate"] == pytest.approx(a * b)
    idx = np.random.default_rng(5).integers(0, n, size=(2000, n))
    ab = [sm.OLS(m[i], sm.add_constant(x[i])).fit().params[1]
          * sm.OLS(y[i], sm.add_constant(np.column_stack([x[i], m[i]]))).fit().params[2] for i in idx]
    assert r1["indirect_effect"]["percentile_ci"] == pytest.approx(list(np.percentile(ab, [2.5, 97.5])), rel=1e-10)
