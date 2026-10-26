"""
Pipeline Builder (/api/pipeline/run): every executor against an independent
reference (statsmodels or the closed-form formula), plus the request contract
the PipelineBuilder screen sends — mixed numeric / categorical columns, missing
cells, step chaining via input_from, and errors reported per step.
"""

import asyncio
import math

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats.mstats import winsorize
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from app.routers import pipeline as P

run = asyncio.run


def _data(seed=0, n=40):
    rng = np.random.default_rng(seed)
    group = ["control" if i % 2 == 0 else "treatment" for i in range(n)]
    cond = ["A", "B", "C", "D"] * (n // 4)
    base = rng.normal(50, 10, n)
    out = 40 + 0.6 * base + np.where(np.array(group) == "treatment", 4, 0) + rng.normal(0, 6, n)
    age = rng.normal(30, 8, n)
    return {"group": group, "condition": cond, "outcome": out.tolist(), "baseline": base.tolist(), "age": age.tolist()}


def _run(data, *steps, chain=True):
    ids = [f"s{i}" for i in range(len(steps))]
    req = P.PipelineRequest(name="t", data=data, steps=[
        P.PipelineStep(id=ids[i], module=m, action=a, params=p, input_from=(ids[i - 1] if chain and i else None))
        for i, (m, a, p) in enumerate(steps)
    ])
    return run(P.run_pipeline(req))


def _rec(record, method, statistic, observed, expected, reference, kind, dataset, tol):
    record(method=method, endpoint="/api/pipeline/run", statistic=statistic, observed=float(observed),
           expected=float(expected), reference=reference, reference_kind=kind, dataset=dataset, tolerance_abs=tol)


def _ok(resp, i=0):
    r = resp.results[i]
    assert r.status == "success", r.results
    return r.results


@pytest.mark.validation
def test_welch_t_and_cohens_d_match_statsmodels(record_validation):
    d = _data()
    r = _ok(_run(d, ("stats", "t-test", {"group_col": "group", "value_col": "outcome"})))
    y, g = np.array(d["outcome"]), np.array(d["group"])
    a, b = y[g == "control"], y[g == "treatment"]
    t, p, df = CompareMeans(DescrStatsW(a), DescrStatsW(b)).ttest_ind(usevar="unequal")
    assert r["groups"] == ["control", "treatment"]
    assert r["t"] == pytest.approx(t, abs=1e-4) and r["df"] == pytest.approx(df, abs=1e-4) and r["p"] == pytest.approx(p, abs=1e-6)
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    assert r["cohens_d"] == pytest.approx((a.mean() - b.mean()) / sp, abs=1e-4)
    # Student's version when requested
    r2 = _ok(_run(d, ("stats", "t-test", {"group_col": "group", "value_col": "outcome", "equal_var": True})))
    t2, p2, df2 = CompareMeans(DescrStatsW(a), DescrStatsW(b)).ttest_ind(usevar="pooled")
    assert r2["t"] == pytest.approx(t2, abs=1e-4) and r2["df"] == len(a) + len(b) - 2 and r2["p"] == pytest.approx(p2, abs=1e-6)
    _rec(record_validation, "Pipeline: Welch t-test", "t", r["t"], t, "statsmodels CompareMeans.ttest_ind(usevar='unequal')",
         "scipy-ref", "simulated 2-group data, n=40", 1e-4)


@pytest.mark.validation
def test_paired_t_uses_n_minus_1_df_and_dz(record_validation):
    d = _data(1)
    r = _ok(_run(d, ("stats", "paired-t-test", {"col_1": "outcome", "col_2": "baseline"})))
    diff = np.array(d["outcome"]) - np.array(d["baseline"])
    n = len(diff)
    t = diff.mean() / (diff.std(ddof=1) / math.sqrt(n))
    assert r["df"] == n - 1
    assert r["t"] == pytest.approx(t, abs=1e-4)
    assert r["p"] == pytest.approx(2 * stats.t.sf(abs(t), n - 1), abs=1e-6)
    assert r["cohens_dz"] == pytest.approx(diff.mean() / diff.std(ddof=1), abs=1e-4)
    _rec(record_validation, "Pipeline: paired t-test", "t", r["t"], t, "closed form: mean(d) / (sd(d)/sqrt(n)), df = n - 1",
         "analytical", "simulated paired data, n=40", 1e-4)


@pytest.mark.validation
def test_one_way_anova_matches_statsmodels_ols(record_validation):
    import pandas as pd
    d = _data(2)
    r = _ok(_run(d, ("stats", "anova", {"group_col": "condition", "value_col": "outcome"})))
    df = pd.DataFrame(d)
    tab = sm.stats.anova_lm(smf.ols("outcome ~ C(condition)", df).fit(), typ=2)
    ss_b, ss_w = tab.loc["C(condition)", "sum_sq"], tab.loc["Residual", "sum_sq"]
    assert r["F"] == pytest.approx(tab.loc["C(condition)", "F"], abs=1e-4)
    assert r["p"] == pytest.approx(tab.loc["C(condition)", "PR(>F)"], abs=1e-6)
    assert (r["df1"], r["df2"]) == (3, len(df) - 4)
    assert r["eta_squared"] == pytest.approx(ss_b / (ss_b + ss_w), abs=1e-4)
    ms_w = ss_w / (len(df) - 4)
    assert r["omega_squared"] == pytest.approx((ss_b - 3 * ms_w) / (ss_b + ss_w + ms_w), abs=1e-4)
    _rec(record_validation, "Pipeline: one-way ANOVA", "F", r["F"], tab.loc["C(condition)", "F"], "statsmodels anova_lm(ols)",
         "scipy-ref", "simulated 4-group data, n=40", 1e-4)


@pytest.mark.validation
def test_brown_forsythe_equals_anova_on_median_deviations(record_validation):
    d = _data(3)
    r = _ok(_run(d, ("assumptions", "homogeneity", {"group_col": "condition", "value_col": "outcome"})))
    y, g = np.array(d["outcome"]), np.array(d["condition"])
    devs = [np.abs(y[g == k] - np.median(y[g == k])) for k in "ABCD"]
    F, p = stats.f_oneway(*devs)
    assert r["F"] == pytest.approx(F, abs=1e-5) and r["p"] == pytest.approx(p, abs=1e-6)
    _rec(record_validation, "Pipeline: Brown–Forsythe test", "F", r["F"], F, "one-way ANOVA on |x - group median| (Brown & Forsythe, 1974)",
         "analytical", "simulated 4-group data, n=40", 1e-5)


@pytest.mark.validation
def test_correlation_pairs_match_closed_form(record_validation):
    d = _data(4)
    r = _ok(_run(d, ("stats", "correlation", {"columns": ["outcome", "baseline", "age"]})))
    x, y = np.array(d["outcome"]), np.array(d["baseline"])
    rr = np.corrcoef(x, y)[0, 1]
    n = len(x)
    p = 2 * stats.t.sf(abs(rr) * math.sqrt((n - 2) / (1 - rr ** 2)), n - 2)
    pair = next(q for q in r["pairs"] if (q["x"], q["y"]) == ("outcome", "baseline"))
    assert pair["r"] == pytest.approx(rr, abs=1e-4) and pair["p"] == pytest.approx(p, abs=1e-6)
    assert len(r["pairs"]) == 3
    _rec(record_validation, "Pipeline: Pearson correlation", "r", pair["r"], rr, "closed form; p from t = r sqrt((n-2)/(1-r^2))",
         "analytical", "simulated data, n=40", 1e-4)


@pytest.mark.validation
def test_regression_matches_least_squares(record_validation):
    d = _data(5)
    r = _ok(_run(d, ("stats", "regression", {"dv": "outcome", "ivs": ["baseline", "age"]})))
    X = np.column_stack([np.ones(len(d["outcome"])), d["baseline"], d["age"]])
    y = np.array(d["outcome"])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    assert r["coefficients"]["baseline"]["b"] == pytest.approx(beta[1], abs=1e-4)
    assert r["coefficients"]["const"]["b"] == pytest.approx(beta[0], abs=1e-4)
    assert r["r_squared"] == pytest.approx(r2, abs=1e-4)
    assert (r["df1"], r["df2"]) == (2, len(y) - 3)
    _rec(record_validation, "Pipeline: linear regression", "R^2", r["r_squared"], r2, "numpy least squares",
         "analytical", "simulated data, n=40, 2 predictors", 1e-4)


@pytest.mark.validation
def test_vif_equals_one_over_one_minus_r2(record_validation):
    d = _data(6)
    d["collinear"] = (np.array(d["baseline"]) * 0.8 + np.random.default_rng(9).normal(0, 3, 40)).tolist()
    r = _ok(_run(d, ("assumptions", "multicollinearity", {"columns": ["baseline", "age", "collinear"]})))
    X = np.column_stack([np.ones(40), d["age"], d["collinear"]])
    y = np.array(d["baseline"])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    assert r["vif"]["baseline"] == pytest.approx(1 / (1 - r2), abs=1e-3)
    _rec(record_validation, "Pipeline: variance inflation factor", "VIF", r["vif"]["baseline"], 1 / (1 - r2),
         "1 / (1 - R^2) of each predictor on the others", "analytical", "simulated collinear predictors, n=40", 1e-3)


def test_missing_cells_and_categoricals_are_accepted():
    d = _data(7)
    d["baseline"][3] = None
    d["baseline"][10] = None
    resp = _run(d,
                ("stats", "descriptives", {"columns": ["baseline"]}),
                ("data-cleaning", "impute-mean", {"columns": ["baseline"]}),
                ("stats", "descriptives", {"columns": ["baseline"]}))
    first, _, third = _ok(resp, 0), _ok(resp, 1), _ok(resp, 2)
    assert first["baseline"]["n"] == 38 and first["baseline"]["missing"] == 2
    assert third["baseline"]["n"] == 40 and third["baseline"]["missing"] == 0
    assert third["baseline"]["mean"] == pytest.approx(first["baseline"]["mean"], abs=1e-4)
    assert resp.final_data["group"][0] == "control"


def test_chaining_passes_filtered_rows_forward():
    d = _data(8)
    resp = _run(d,
                ("transform", "filter", {"column": "condition", "operator": "!=", "value": "D"}),
                ("stats", "anova", {"group_col": "condition", "value_col": "outcome"}))
    assert _ok(resp, 0)["rows_after"] == 30
    assert _ok(resp, 1)["df1"] == 2 and resp.results[1].n_rows == 30
    # without chaining the ANOVA sees the original four conditions
    resp2 = _run(d,
                 ("transform", "filter", {"column": "condition", "operator": "!=", "value": "D"}),
                 ("stats", "anova", {"group_col": "condition", "value_col": "outcome"}), chain=False)
    assert _ok(resp2, 1)["df1"] == 3


def test_outlier_removal_keeps_rows_with_missing_values():
    d = {"x": [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 100.0, None]}
    r = _ok(_run(d, ("data-cleaning", "remove-outliers", {"threshold": 2.5})))
    assert r["rows_removed"] == 1


def test_winsorize_matches_scipy_and_ignores_missing():
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 50)
    vals = x.tolist() + [None]
    resp = _run({"x": vals}, ("data-cleaning", "winsorize", {"percentile": 10}))
    _ok(resp)
    expected = np.asarray(winsorize(x, limits=(0.1, 0.1)))
    assert resp.final_data["x"][:50] == pytest.approx(expected.tolist(), abs=1e-12)
    assert resp.final_data["x"][50] is None


def test_recode_and_filter_coerce_string_params_on_numeric_columns():
    d = {"code": [1.0, 2.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0]}
    resp = _run(d,
                ("transform", "recode", {"column": "code", "mapping": "1=10, 2=20"}),
                ("transform", "filter", {"column": "code", "operator": ">=", "value": "20"}))
    assert _ok(resp, 0)["cells_recoded"] == 3
    assert _ok(resp, 1)["rows_after"] == 2
    assert resp.final_data["code"] == [20.0, 20.0]


def test_failures_are_reported_not_hidden():
    d = _data(11)
    resp = _run(d,
                ("stats", "t-test", {"group_col": "condition", "value_col": "outcome"}),
                ("data-cleaning", "log-transform", {"columns": ["outcome"], "offset": -1000}),
                ("cleaning", "remove_outliers", {}),
                chain=False)
    assert [r.status for r in resp.results] == ["error", "error", "error"]
    assert "exactly 2 groups" in resp.results[0].results["error"]
    assert "log is undefined" in resp.results[1].results["error"]
    assert "Unknown action" in resp.results[2].results["error"]
    assert resp.steps_completed == 0


def test_unequal_column_lengths_rejected():
    with pytest.raises(ValueError):
        P.PipelineRequest(name="t", data={"a": [1.0, 2.0], "b": [1.0]}, steps=[])
