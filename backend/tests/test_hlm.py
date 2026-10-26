"""
Two-level HLM (/api/hlm/two-level, Stats Studio) against lme4 2.0.6 (REML).

R² by hand in R: var(predict(m, re.form = NA)) for the fixed part; mean(diag(Z S Z')) for the
random part (Johnson, 2014); sigma(m)^2 residual (Nakagawa & Schielzeth, 2013).
"""

import warnings

import pytest

from app.routers import hlm as H
from tests.test_advanced_stats import mixed_data


@pytest.mark.validation
@pytest.mark.parametrize("slopes,r2m,r2c,se", [(False, 0.6829536624, 0.7743825053, 0.58690718),
                                                (True, 0.6820131649, 0.7948920456, 0.73931970)])
def test_two_level_r2_and_se_match_lme4(record_validation, slopes, r2m, r2c, se):
    subj, day, y = mixed_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = H.two_level_hlm(H.TwoLevelRequest(dependent=y, level1_predictors=[day], groups=subj,
                                              predictor_names=["day"], random_slopes=slopes))
    m = f"Two-level HLM ({'random slope' if slopes else 'random intercept'})"
    for stat, obs, exp, tol in (("R2 marginal", r["r_squared"]["marginal"], r2m, 1e-4),
                                ("R2 conditional", r["r_squared"]["conditional"], r2c, 1e-4),
                                ("slope SE", r["fixed_effects"][1]["se"], se, 1e-4)):
        assert obs == pytest.approx(exp, abs=tol)
        record_validation(method=m, endpoint="/api/hlm/two-level", statistic=stat, observed=float(obs), expected=float(exp),
                          reference="lme4 2.0.6 (R 4.5.2)", reference_kind="r",
                          dataset="deterministic 12 subjects x 8 days (LCG)", tolerance_abs=tol)
    assert "z" in r["fixed_effects"][1] and r["model_fit"]["aic"] is not None
