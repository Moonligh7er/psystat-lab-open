"""
Logistic regression and ROC/AUC (/api/extended-stats/*) against R 4.5.2.

    glm(y ~ x, family = binomial)
    pROC::roc(lab, sc, direction = "<"); pROC::ci.auc(r, method = "delong")
"""

import pytest

from app.routers import extended_stats as E

LAB = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
SC = [0.9, 0.3, 0.8, 0.4, 0.2, 0.4, 0.85, 0.5, 0.75, 0.95, 0.15, 0.88, 0.3, 0.25, 0.35, 0.82, 0.6, 0.78, 0.91, 0.6]
Y = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
X = [1.2, 2.1, 3.3, 1.8, 4.0, 3.1, 2.5, 3.9, 2.0, 2.8, 3.4, 2.2, 3.6, 4.1, 1.9, 3.0, 2.7, 2.6, 3.8, 2.4]


def _rec(rec, method, endpoint, stat, obs, exp, tol, ref):
    assert obs == pytest.approx(exp, abs=tol)
    rec(method=method, endpoint=endpoint, statistic=stat, observed=float(obs), expected=float(exp),
        reference=ref, reference_kind="r", dataset="20 observations", tolerance_abs=tol)


@pytest.mark.validation
def test_roc_auc_delong_matches_proc(record_validation):
    r = E.roc_auc(E.RocAucRequest(true_labels=LAB, scores=SC))
    e = "/api/extended-stats/roc-auc"
    _rec(record_validation, "ROC AUC with DeLong CI", e, "AUC", r["auc"], 0.921875, 1e-12, "pROC::roc / ci.auc (DeLong)")
    _rec(record_validation, "ROC AUC with DeLong CI", e, "CI lower", r["auc_ci_lower"], 0.8068409219, 1e-9, "pROC::ci.auc (DeLong)")
    # Youden: same split as pROC (which reports the midpoint 0.675 between adjacent scores)
    assert (r["sensitivity_at_optimal"], r["specificity_at_optimal"]) == (0.75, 1.0)


def test_auc_below_half_is_not_flipped():
    r = E.roc_auc(E.RocAucRequest(true_labels=LAB, scores=[-s for s in SC]))
    assert r["auc"] == pytest.approx(1 - 0.921875) and r["auc_note"]


@pytest.mark.validation
def test_logistic_regression_matches_glm(record_validation):
    r = E.logistic_regression(E.LogisticRegressionRequest(dependent=Y, predictors=[[v] for v in X]))
    e, ref = "/api/extended-stats/logistic-regression", "R glm(binomial)"
    c = r["coefficient_table"][1]
    _rec(record_validation, "Logistic regression", e, "slope", c["coefficient"], 2.5170485808, 1e-6, ref)
    _rec(record_validation, "Logistic regression", e, "slope SE", c["std_error"], 1.0856464890, 1e-4, ref)
    _rec(record_validation, "Logistic regression", e, "LR chi-square", r["likelihood_ratio_test"]["chi_square"], 10.2196965046, 1e-6, ref)
    _rec(record_validation, "Logistic regression", e, "AIC", r["aic"], 21.3058560440, 1e-6, ref)


@pytest.mark.validation
def test_anova_power_per_group_matches_gpower(client, record_validation):
    """statsmodels' FTestAnovaPower returns TOTAL N; per-group N must divide by k (was multiplied)."""
    r = client.post("/api/stats/power-analysis", json={"effect_size": 0.25, "alpha": 0.05, "power": 0.8, "test_type": "anova", "n_groups": 3}).json()
    assert r["required_n_per_group"] == 53 and r["total_n"] == 159
    record_validation(method="Power analysis, one-way ANOVA", endpoint="/api/stats/power-analysis", statistic="total N",
                      observed=float(r["total_n"]), expected=159.0, reference="G*Power 3.1 (f = 0.25, k = 3, power = .80)",
                      reference_kind="analytical", dataset="f = 0.25, alpha = .05, 3 groups", tolerance_abs=0)
