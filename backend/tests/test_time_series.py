"""
Time Series (/api/time-series/*): each method against an independent reference.

  - Phillips–Perron Z_t: frozen values from arch 8.0.0 (PhillipsPerron, trend="c")
  - PELT change points: exhaustive optimal partitioning (same cost and penalty)
  - VAR Granger test: restricted vs unrestricted OLS F statistic
  - ITS segmented regression: least squares by hand
  - ARIMA: interval level honoured, Ljung–Box df adjusted, d chosen by KPSS
  - Decomposition strength: 1 - Var(R)/Var(T+R) from the returned components
"""

import math

import numpy as np
import pytest
from scipy import stats

from app.routers import time_series as T


def lcg_series(n, phi, seed=1):
    """AR(1) driven by a linear congruential generator, reproducible anywhere."""
    x, y = seed, [0.0]
    for _ in range(1, n):
        x = (1103515245 * x + 12345) % 2 ** 31
        y.append(phi * y[-1] + (x / 2 ** 31) * 2 - 1)
    return y


# arch 8.0.0: PhillipsPerron(lcg_series(n, phi), trend="c", test_type="tau")
PP_REFERENCE = [
    (150, 0.5, -6.619931569831973, 6.07300314909679e-09, 14),
    (150, 1.0, -1.8163217230003503, 0.3723890467309342, 14),
    (80, 0.9, -1.4010850635840792, 0.5817558149899436, 12),
]


@pytest.mark.validation
@pytest.mark.parametrize("n,phi,stat,pval,lags", PP_REFERENCE)
def test_phillips_perron_matches_arch(n, phi, stat, pval, lags, record_validation):
    r = T.stationarity_tests(T.StationarityRequest(values=lcg_series(n, phi)))["phillips_perron"]
    assert r["test_statistic"] == pytest.approx(stat, abs=1e-8)
    assert r["p_value"] == pytest.approx(pval, rel=1e-6, abs=1e-12)
    assert r["lags_used"] == lags
    for name, obs, exp in (("Z_t", r["test_statistic"], stat), ("p_value", r["p_value"], pval)):
        record_validation(
            method="Phillips–Perron unit root test", endpoint="/api/time-series/stationarity",
            statistic=name, observed=float(obs), expected=float(exp),
            reference="arch 8.0.0 PhillipsPerron(trend='c')", reference_kind="scipy-ref",
            dataset=f"LCG AR(1), n={n}, phi={phi}", tolerance_abs=1e-8,
        )


def _optimal_partition(y, pen, m):
    n = len(y)
    F, last = [math.inf] * (n + 1), [0] * (n + 1)
    F[0] = -pen
    cost = lambda a, b: (b - a) * math.log(max(np.var(y[a:b]), 1e-15))
    for t in range(m, n + 1):
        for s in [0] + list(range(m, t - m + 1)):
            v = F[s] + cost(s, t) + pen
            if v < F[t]:
                F[t], last[t] = v, s
    cps, t = [], n
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(t)
    return sorted(cps)


@pytest.mark.validation
def test_pelt_is_exact(record_validation):
    mismatches = 0
    for seed in range(120):
        rng = np.random.default_rng(seed)
        n, m = int(rng.integers(12, 80)), int(rng.integers(2, 6))
        k = int(rng.integers(0, 4))
        cuts = np.sort(rng.choice(np.arange(m, n - m), min(k, n - 2 * m), replace=False))
        y = np.concatenate([rng.normal(rng.normal(0, 2), rng.uniform(0.3, 2), int(L)) for L in np.diff(np.r_[0, cuts, n])])
        pen = 2 * math.log(len(y)) * rng.uniform(0.3, 1.5)
        mismatches += T._pelt(y, pen, m) != _optimal_partition(y, pen, m)
    assert mismatches == 0
    record_validation(
        method="PELT change point detection", endpoint="/api/time-series/changepoint",
        statistic="segmentations differing from exhaustive optimum (120 random series)",
        observed=float(mismatches), expected=0.0,
        reference="Exhaustive optimal partitioning (Jackson et al., 2005)", reference_kind="analytical",
        dataset="120 simulated piecewise-Gaussian series", tolerance_abs=0,
    )


def test_changepoint_default_penalty_is_bic():
    y = lcg_series(60, 0.0) + [v + 3 for v in lcg_series(60, 0.0, seed=7)]
    r = T.changepoint_detection(T.ChangePointRequest(values=y))
    assert r["penalty"] == pytest.approx(2 * math.log(120))
    assert r["changepoints"] == [60]
    assert r["model_comparison"]["changepoints_preferred"]


@pytest.mark.validation
def test_var_granger_equals_restricted_ols_f(record_validation):
    rng = np.random.default_rng(3)
    n = 120
    a, b = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        a[t] = 0.5 * a[t - 1] + rng.normal()
        b[t] = 0.3 * b[t - 1] + 0.4 * a[t - 1] + rng.normal()
    r = T.var_model(T.VARRequest(series={"a": a.tolist(), "b": b.tolist()}, max_lag=4))
    p = r["selected_lag"]
    g = r["granger_causality"]["a -> b"]
    # Unrestricted: b_t on const + p lags of a and b; restricted: drop the lags of a
    Y = b[p:]
    lags = lambda x: np.column_stack([x[p - j:n - j] for j in range(1, p + 1)])
    Xu = np.column_stack([np.ones(n - p), lags(a), lags(b)])
    Xr = np.column_stack([np.ones(n - p), lags(b)])
    rss = lambda X: float(np.sum((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2))
    df_resid = len(Y) - Xu.shape[1]
    F = ((rss(Xr) - rss(Xu)) / p) / (rss(Xu) / df_resid)
    assert g["f_statistic"] == pytest.approx(F, rel=1e-8)
    # statsmodels / R vars::causality use the system residual df (k equations x df_resid)
    assert g["df"] == [p, 2 * df_resid]
    assert g["p_value"] == pytest.approx(stats.f.sf(F, p, 2 * df_resid), rel=1e-8)
    assert g["significant"] and not r["granger_causality"]["b -> a"]["significant"]
    record_validation(
        method="VAR Granger causality (Wald F)", endpoint="/api/time-series/var",
        statistic="F", observed=float(g["f_statistic"]), expected=float(F),
        reference="Restricted vs unrestricted OLS F (Lütkepohl, 2005, §3.6)", reference_kind="analytical",
        dataset="simulated bivariate VAR(1), n=120, seed 3", tolerance_rel=1e-8,
    )


@pytest.mark.validation
def test_its_segmented_regression_matches_least_squares(record_validation):
    y = [10, 12, 11, 13, 12, 14, 13, 15, 25, 28, 27, 30, 29, 31, 30, 32]
    k = 8
    r = T.interrupted_time_series(T.ITSRequest(outcome=y, intervention_point=k))
    t = np.arange(16.0)
    X = np.column_stack([np.ones(16), t, (t >= k).astype(float), np.where(t >= k, t - k, 0.0)])
    beta = np.linalg.lstsq(X, np.array(y, float), rcond=None)[0]
    assert r["level_change"]["value"] == pytest.approx(beta[2], abs=1e-9)
    assert r["slope_change"]["value"] == pytest.approx(beta[3], abs=1e-9)
    cf = X[:, :2] @ beta[:2]
    assert r["plot_data"]["counterfactual"] == pytest.approx(cf.tolist(), abs=1e-9)
    record_validation(
        method="Interrupted time series (segmented regression)", endpoint="/api/time-series/interrupted-time-series",
        statistic="level change", observed=float(r["level_change"]["value"]), expected=float(beta[2]),
        reference="Least squares by hand (Wagner et al., 2002 model)", reference_kind="analytical",
        dataset="16-point step series, intervention at 8", tolerance_abs=1e-9,
    )


def test_arima_interval_level_and_ljung_box_df():
    y = lcg_series(120, 0.7)
    r95 = T.arima_model(T.ARIMARequest(values=y, order=[1, 0, 1], n_forecast=3))
    r80 = T.arima_model(T.ARIMARequest(values=y, order=[1, 0, 1], n_forecast=3, confidence_level=0.8))
    w = lambda r: np.array(r["forecast"]["upper_ci"]) - np.array(r["forecast"]["lower_ci"])
    # Gaussian forecast intervals: width ratio = z(.90) / z(.975)
    assert w(r80) / w(r95) == pytest.approx(np.full(3, stats.norm.ppf(0.9) / stats.norm.ppf(0.975)), rel=1e-6)
    d = r95["residual_diagnostics"]
    assert d["ljung_box_df"] == d["ljung_box_lag"] - 2


def test_arima_auto_order_differences_a_random_walk_only():
    walk = np.cumsum(np.random.default_rng(5).normal(size=200)).tolist()
    noise = np.random.default_rng(6).normal(size=200).tolist()
    assert T.arima_model(T.ARIMARequest(values=walk))["order"]["d"] == 1
    assert T.arima_model(T.ARIMARequest(values=noise))["order"]["d"] == 0


def test_decomposition_strength_formula():
    t = np.arange(72)
    y = (0.2 * t + 4 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(8).normal(0, 1, 72)).tolist()
    r = T.decomposition(T.DecompositionRequest(values=y, period=12))
    Tr, S, R = (np.array(r[k], float) for k in ("trend", "seasonal", "residual"))
    assert r["trend_strength"] == pytest.approx(max(0, 1 - np.var(R, ddof=1) / np.var(Tr + R, ddof=1)))
    assert r["seasonal_strength"] == pytest.approx(max(0, 1 - np.var(R, ddof=1) / np.var(S + R, ddof=1)))


def test_screen_payloads_round_trip(client):
    """The exact request shapes TimeSeries.tsx sends, and the fields it reads back."""
    ex = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140]
    r = client.post("/api/time-series/arima", json={"values": ex, "order": None})
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("order", "order_selection", "parameters", "aic", "aicc", "bic", "original_acf", "original_pacf"):
        assert k in j
    assert {"mean", "lower_ci", "upper_ci", "confidence_level"} <= set(j["forecast"])
    assert {"ljung_box_p", "ljung_box_lag", "ljung_box_df", "residual_acf"} <= set(j["residual_diagnostics"])

    sales = [20.1, 21.4, 20.8, 22.5, 23.9, 23.1, 24.8, 26.2, 25.4, 27.0, 28.6, 27.9, 29.3, 30.8, 30.1, 31.7, 33.2, 32.4, 34.0, 35.5, 34.9, 36.3, 37.9, 37.2]
    ads = [5.2, 5.9, 5.4, 6.3, 6.8, 6.1, 7.0, 7.6, 6.9, 7.8, 8.3, 7.7, 8.6, 9.1, 8.4, 9.3, 9.9, 9.2, 10.1, 10.6, 9.9, 10.8, 11.4, 10.7]
    r = client.post("/api/time-series/var", json={"series": {"sales": sales, "ads": ads}, "max_lag": 4})
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("selected_lag", "lag_note", "stability", "granger_causality", "impulse_response_functions", "irf_type", "forecast_error_variance_decomposition"):
        assert k in j

    r = client.post("/api/time-series/interrupted-time-series", json={
        "outcome": [10, 12, 11, 13, 12, 14, 13, 15, 25, 28, 27, 30, 29, 31, 30, 32], "intervention_point": 8, "covariates": None})
    assert r.status_code == 200, r.text
    j = r.json()
    assert {"value", "p_value"} <= set(j["level_change"]) and {"value", "p_value"} <= set(j["slope_change"])
    assert {"durbin_watson", "se_type", "autocorrelation_detected"} <= set(j["autocorrelation"])
    assert {"observed", "counterfactual"} <= set(j["plot_data"]) and "r_squared" in j["model_fit"]

    r = client.post("/api/time-series/decomposition", json={"values": ex, "period": 12})
    assert r.status_code == 200, r.text
    assert {"method", "period", "trend", "seasonal", "residual", "trend_strength", "seasonal_strength"} <= set(r.json())

    r = client.post("/api/time-series/stationarity", json={"values": [1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.3, 2.5, 2.4, 2.8, 3.0, 2.9, 3.2, 3.5, 3.4, 3.8, 4.0, 3.9, 4.2, 4.5]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert {"stationary", "interpretation", "recommended_differencing_order"} <= set(j["conclusion"])
    for k in ("adf", "kpss", "phillips_perron"):
        assert {"test_statistic", "p_value", "critical_values", "null_hypothesis"} <= set(j[k])

    r = client.post("/api/time-series/changepoint", json={"values": [10, 11, 12, 10, 11, 13, 12, 25, 26, 28, 27, 29, 30, 28, 15, 14, 16, 15, 17, 14]})
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("method", "changepoints", "segments", "model_comparison", "penalty", "penalty_rule"):
        assert k in j


def test_decomposition_withholds_strength_below_three_periods():
    ex = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140]
    r = T.decomposition(T.DecompositionRequest(values=ex, period=12))
    assert r["trend_strength"] is None and r["seasonal_strength"] is None and r["strength_note"]
    r3 = T.decomposition(T.DecompositionRequest(values=ex + [145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166], period=12))
    assert 0 < r3["seasonal_strength"] < 1 and r3["strength_note"] is None


def test_var_warns_when_unstable():
    trend = [float(t) + 0.1 * ((t * 7) % 5) for t in range(30)]
    other = [2.0 * float(t) + 0.2 * ((t * 3) % 4) for t in range(30)]
    r = T.var_model(T.VARRequest(series={"a": trend, "b": other}, max_lag=3))
    assert (r["stability"]["warning"] is None) == r["stability"]["is_stable"]
    assert not r["stability"]["is_stable"] and "not stable" in r["stability"]["warning"]
