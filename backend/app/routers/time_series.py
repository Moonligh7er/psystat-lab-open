"""Time Series Analysis — ARIMA, VAR, ITS, decomposition, stationarity, change-point detection."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.util import comp_matrix as var_comp_matrix
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
import statsmodels.api as sm
import pandas as pd
import warnings

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class ARIMARequest(BaseModel):
    values: list[float]
    order: list[int] | None = None  # [p, d, q] — auto-select if None
    n_forecast: int = 10
    confidence_level: float = 0.95


class VARRequest(BaseModel):
    series: dict[str, list[float]]  # named series: {"gdp": [...], "cpi": [...]}
    max_lag: int = 8
    n_forecast: int = 10
    irf_periods: int = 10


class ITSRequest(BaseModel):
    outcome: list[float]
    intervention_point: int  # index where intervention begins
    covariates: list[list[float]] | None = None  # optional exogenous covariates
    confidence_level: float = 0.95


class DecompositionRequest(BaseModel):
    values: list[float]
    period: int
    method: str = "stl"  # "stl", "additive", "multiplicative"


class StationarityRequest(BaseModel):
    values: list[float]
    max_diff: int = 3  # max differencing order to recommend


class ChangePointRequest(BaseModel):
    values: list[float]
    method: str = "pelt"  # "pelt" or "binseg"
    min_segment_length: int = 5
    penalty: float | None = None  # BIC penalty; auto-computed if None


# ── Helpers ───────────────────────────────────────────────────────────

def _safe_float(v):
    """Convert numpy value to Python float, handling NaN/Inf."""
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _safe_list(arr):
    """Convert numpy array to list of safe floats."""
    return [_safe_float(x) for x in arr]


# ── 1. ARIMA ──────────────────────────────────────────────────────────

@router.post("/arima")
def arima_model(req: ARIMARequest):
    """ARIMA modeling with optional auto-order selection via AIC grid search."""
    y = np.array(req.values, dtype=float)
    n = len(y)
    if n < 10:
        raise HTTPException(400, "Need at least 10 observations for ARIMA.")

    if not 0 < req.confidence_level < 1:
        raise HTTPException(400, "confidence_level must be between 0 and 1.")
    if not 1 <= req.n_forecast <= 200:
        raise HTTPException(400, "n_forecast must be between 1 and 200.")

    # Determine order
    if req.order is not None:
        if len(req.order) != 3 or any(o < 0 for o in req.order) or req.order[1] > 2:
            raise HTTPException(400, "order must be [p, d, q] with non-negative values and d ≤ 2.")
        order = tuple(req.order)
        selection = "user-specified"
    else:
        # As in Hyndman & Khandakar (2008): choose d by repeated KPSS tests, then
        # (p, q) by AICc. AIC is not comparable across different d (the likelihood
        # is for a differently differenced series), so d is never chosen by AIC.
        d = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while d < 2:
                yd = np.diff(y, n=d) if d else y
                # Truncation lag as in R forecast::ndiffs / urca "short": trunc(4 (n/100)^(1/4))
                if len(yd) < 10 or kpss(yd, regression="c", nlags=int(4 * (len(yd) / 100) ** 0.25))[1] >= 0.05:
                    break
                d += 1
            best_aicc, best_order = np.inf, None
            for p in range(4):
                for q in range(4):
                    if p + q + 2 >= n - d:
                        continue
                    try:
                        f = ARIMA(y, order=(p, d, q)).fit()
                    except Exception:
                        continue
                    if np.isfinite(f.aicc) and f.aicc < best_aicc:
                        best_aicc, best_order = f.aicc, (p, d, q)
        if best_order is None:
            raise HTTPException(400, "No ARIMA model could be fitted to this series.")
        order = best_order
        selection = "d by KPSS (α = .05), then p, q ∈ {0..3} by AICc"

    # Fit the selected model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(y, order=order).fit()
    except Exception as e:
        raise HTTPException(400, f"ARIMA fitting failed: {str(e)}")

    # Residual diagnostics (the first d residuals are start-up values, not innovations)
    resid = np.asarray(fit.resid)[order[1]:]
    model_df = order[0] + order[2]
    lb_lags = min(max(10, model_df + 5), len(resid) // 2)
    if lb_lags <= model_df:
        raise HTTPException(400, "Series too short for residual diagnostics at this order.")
    ljung_box = acorr_ljungbox(resid, lags=lb_lags, model_df=model_df, return_df=True)
    n_acf_lags = min(20, n // 2 - 1)
    resid_acf = acf(resid, nlags=n_acf_lags, fft=True)
    resid_pacf = pacf(resid, nlags=n_acf_lags)

    # ACF/PACF of original series (for model identification)
    orig_acf = acf(y, nlags=n_acf_lags, fft=True)
    orig_pacf = pacf(y, nlags=n_acf_lags)

    # Forecast
    forecast_res = fit.get_forecast(steps=req.n_forecast)
    forecast_mean = np.asarray(forecast_res.predicted_mean)
    forecast_ci = np.asarray(forecast_res.conf_int(alpha=1 - req.confidence_level))

    # Parameter details
    params = fit.params
    bse = fit.bse
    pvalues = fit.pvalues
    param_names = fit.param_names if hasattr(fit, 'param_names') else [f"param_{i}" for i in range(len(params))]

    parameters = []
    for name, coef, se, pv in zip(param_names, params, bse, pvalues):
        parameters.append({
            "name": name,
            "coefficient": _safe_float(coef),
            "std_error": _safe_float(se),
            "p_value": _safe_float(pv),
        })

    return {
        "order": {"p": order[0], "d": order[1], "q": order[2]},
        "auto_selected": req.order is None,
        "order_selection": selection,
        "aicc": _safe_float(fit.aicc),
        "parameters": parameters,
        "aic": _safe_float(fit.aic),
        "bic": _safe_float(fit.bic),
        "log_likelihood": _safe_float(fit.llf),
        "residual_diagnostics": {
            "ljung_box_lag": int(lb_lags),
            "ljung_box_df": int(lb_lags - model_df),
            "ljung_box_p": _safe_float(ljung_box["lb_pvalue"].iloc[-1]),
            "ljung_box": [
                {"lag": int(lag), "statistic": _safe_float(row["lb_stat"]), "p_value": _safe_float(row["lb_pvalue"])}
                for lag, row in ljung_box.iterrows()
            ],
            "residual_acf": _safe_list(resid_acf),
            "residual_pacf": _safe_list(resid_pacf),
            "residual_mean": _safe_float(np.mean(resid)),
            "residual_std": _safe_float(np.std(resid, ddof=1)),
        },
        "original_acf": _safe_list(orig_acf),
        "original_pacf": _safe_list(orig_pacf),
        "fitted_values": _safe_list(fit.fittedvalues),
        "forecast": {
            "mean": _safe_list(forecast_mean),
            "lower_ci": _safe_list(forecast_ci[:, 0]),
            "upper_ci": _safe_list(forecast_ci[:, 1]),
            "confidence_level": req.confidence_level,
        },
    }


# ── 2. VAR ────────────────────────────────────────────────────────────

@router.post("/var")
def var_model(req: VARRequest):
    """Vector Autoregression with Granger causality and impulse responses."""
    if len(req.series) < 2:
        raise HTTPException(400, "VAR requires at least 2 series.")

    names = list(req.series.keys())
    lengths = {k: len(v) for k, v in req.series.items()}
    if len(set(lengths.values())) != 1:
        raise HTTPException(400, f"All series must have equal length. Got: {lengths}")

    n = list(lengths.values())[0]
    if n < 15:
        raise HTTPException(400, "Need at least 15 observations for VAR.")

    df = pd.DataFrame(req.series)

    try:
        model = VAR(df)
        # Lag order selection
        lag_selection = model.select_order(maxlags=min(req.max_lag, n // 3))
        lag_info = {
            "aic": int(lag_selection.aic),
            "bic": int(lag_selection.bic),
            "hqic": int(lag_selection.hqic),
            "fpe": int(lag_selection.fpe),
        }
        selected_lag = int(lag_selection.aic)  # use AIC-selected lag
        lag_note = None
        if selected_lag == 0:
            selected_lag = 1
            lag_note = "AIC selected lag 0 (no dynamics); a VAR(1) is fitted so the tests below are defined."

        fit = model.fit(maxlags=selected_lag)
    except Exception as e:
        raise HTTPException(400, f"VAR fitting failed: {str(e)}")

    # Coefficient matrices per lag
    coef_matrices = {}
    for lag_i in range(1, selected_lag + 1):
        mat = fit.coefs[lag_i - 1]
        coef_matrices[f"lag_{lag_i}"] = {
            "coefficients": [[_safe_float(c) for c in row] for row in mat],
            "variable_names": names,
        }

    # Granger causality: Wald F-test that all lags of `causing` are zero in the
    # `caused` equation of the fitted VAR (so it conditions on the other series),
    # at the selected lag only — taking the best p over several lags would inflate
    # the false-positive rate.
    granger_results = {}
    for caused in names:
        for causing in names:
            if caused == causing:
                continue
            try:
                gc = fit.test_causality(caused, [causing], kind="f")
                granger_results[f"{causing} -> {caused}"] = {
                    "f_statistic": _safe_float(gc.test_statistic),
                    "df": [int(gc.df[0]), int(gc.df[1])],
                    "p_value": _safe_float(gc.pvalue),
                    "lag": selected_lag,
                    "significant": bool(gc.pvalue < 0.05),
                }
            except Exception:
                granger_results[f"{causing} -> {caused}"] = {
                    "error": "Test could not be computed",
                }

    # Orthogonalised impulse responses (Cholesky; depends on the variable order given)
    irf = fit.irf(req.irf_periods)
    irf_data = {}
    for i, impulse_var in enumerate(names):
        irf_data[impulse_var] = {}
        for j, response_var in enumerate(names):
            irf_data[impulse_var][response_var] = _safe_list(irf.orth_irfs[:, j, i])

    # Forecast error variance decomposition
    fevd = fit.fevd(req.irf_periods)
    fevd_data = {}
    for i, var_name in enumerate(names):
        decomp = fevd.decomp[i]  # shape: (periods, n_vars)
        fevd_data[var_name] = {
            other: _safe_list(decomp[:, j]) for j, other in enumerate(names)
        }

    # Stability check — eigenvalues of companion matrix
    eigenvalues = np.abs(np.linalg.eigvals(var_comp_matrix(fit.coefs)))
    is_stable = bool(np.all(eigenvalues < 1))

    # Forecast
    forecast_vals = fit.forecast(df.values[-selected_lag:], steps=req.n_forecast)
    forecast_out = {
        name: _safe_list(forecast_vals[:, i]) for i, name in enumerate(names)
    }

    return {
        "selected_lag": selected_lag,
        "lag_note": lag_note,
        "lag_selection_criteria": lag_info,
        "coefficient_matrices": coef_matrices,
        "granger_causality": granger_results,
        "impulse_response_functions": irf_data,
        "irf_type": "orthogonalised (Cholesky, in the order the series were given), one-SD shocks",
        "forecast_error_variance_decomposition": fevd_data,
        "stability": {
            "is_stable": is_stable,
            "warning": None if is_stable else (
                "The fitted VAR is not stable (a root lies on or outside the unit circle), which usually means "
                "trending or unit-root series. Granger tests, impulse responses and forecasts are unreliable; "
                "difference or detrend the series and refit."
            ),
            "max_eigenvalue_modulus": _safe_float(np.max(eigenvalues)),
            "eigenvalue_moduli": _safe_list(eigenvalues),
        },
        "forecast": forecast_out,
        "n_observations": n,
        "variable_names": names,
    }


# ── 3. Interrupted Time Series ────────────────────────────────────────

@router.post("/interrupted-time-series")
def interrupted_time_series(req: ITSRequest):
    """Segmented regression for interrupted time series analysis."""
    y = np.array(req.outcome, dtype=float)
    n = len(y)
    if n < 8:
        raise HTTPException(400, "Need at least 8 observations for ITS.")
    if req.intervention_point < 2 or req.intervention_point >= n - 2:
        raise HTTPException(400, "Intervention point must allow at least 2 observations on each side.")

    # Build design matrix: time, intervention (0/1), time_after_intervention
    time = np.arange(n, dtype=float)
    intervention = np.zeros(n, dtype=float)
    intervention[req.intervention_point:] = 1.0
    time_after = np.zeros(n, dtype=float)
    time_after[req.intervention_point:] = time[req.intervention_point:] - req.intervention_point

    X = np.column_stack([time, intervention, time_after])
    col_names = ["time", "intervention_level", "time_after_intervention"]

    # Add optional covariates
    if req.covariates:
        for i, cov in enumerate(req.covariates):
            if len(cov) != n:
                raise HTTPException(400, f"Covariate {i} length ({len(cov)}) must match outcome length ({n}).")
            X = np.column_stack([X, np.array(cov, dtype=float)])
            col_names.append(f"covariate_{i + 1}")

    X = sm.add_constant(X)
    col_names = ["intercept"] + col_names

    # Fit OLS first
    ols_model = OLS(y, X)
    ols_fit = ols_model.fit()

    # Check autocorrelation with Durbin-Watson
    dw_stat = durbin_watson(ols_fit.resid)
    autocorrelation_detected = dw_stat < 1.5 or dw_stat > 2.5

    # If autocorrelated, refit with Newey-West SEs
    if autocorrelation_detected:
        nw_fit = ols_fit.get_robustcov_results(cov_type="HAC", maxlags=int(np.ceil(n ** 0.25)))
        final_fit = nw_fit
        se_type = "Newey-West HAC"
    else:
        final_fit = ols_fit
        se_type = "OLS"

    # Extract results
    alpha = 1 - req.confidence_level
    coefficients = []
    for i, name in enumerate(col_names):
        coef = final_fit.params[i]
        se = final_fit.bse[i]
        t_val = final_fit.tvalues[i]
        p_val = final_fit.pvalues[i]
        ci = final_fit.conf_int(alpha=alpha)[i]
        coefficients.append({
            "name": name,
            "coefficient": _safe_float(coef),
            "std_error": _safe_float(se),
            "t_value": _safe_float(t_val),
            "p_value": _safe_float(p_val),
            "ci_lower": _safe_float(ci[0]),
            "ci_upper": _safe_float(ci[1]),
        })

    # Counterfactual predictions (what would have happened without intervention)
    X_counterfactual = X.copy()
    X_counterfactual[:, 2] = 0  # zero out intervention
    X_counterfactual[:, 3] = 0  # zero out time_after
    counterfactual = ols_fit.predict(X_counterfactual)

    # Fitted values
    fitted = ols_fit.predict(X)

    return {
        "coefficients": coefficients,
        "level_change": {
            "name": "intervention_level (b2)",
            "value": _safe_float(final_fit.params[2]),
            "p_value": _safe_float(final_fit.pvalues[2]),
            "interpretation": "Immediate change in outcome level at intervention",
        },
        "slope_change": {
            "name": "time_after_intervention (b3)",
            "value": _safe_float(final_fit.params[3]),
            "p_value": _safe_float(final_fit.pvalues[3]),
            "interpretation": "Change in trend (slope) after intervention",
        },
        "model_fit": {
            "r_squared": _safe_float(ols_fit.rsquared),
            "adj_r_squared": _safe_float(ols_fit.rsquared_adj),
            "f_statistic": _safe_float(ols_fit.fvalue),
            "f_p_value": _safe_float(ols_fit.f_pvalue),
            "aic": _safe_float(ols_fit.aic),
            "bic": _safe_float(ols_fit.bic),
        },
        "autocorrelation": {
            "durbin_watson": _safe_float(dw_stat),
            "autocorrelation_detected": autocorrelation_detected,
            "se_type": se_type,
        },
        "plot_data": {
            "observed": _safe_list(y),
            "fitted": _safe_list(fitted),
            "counterfactual": _safe_list(counterfactual),
            "time": _safe_list(time),
            "intervention_point": req.intervention_point,
        },
    }


# ── 4. Seasonal Decomposition ────────────────────────────────────────

@router.post("/decomposition")
def decomposition(req: DecompositionRequest):
    """Seasonal decomposition via STL or classical methods."""
    y = np.array(req.values, dtype=float)
    n = len(y)
    if n < 2 * req.period:
        raise HTTPException(400, f"Need at least 2 full periods ({2 * req.period} observations). Got {n}.")
    if req.period < 2:
        raise HTTPException(400, "Period must be at least 2.")

    series = pd.Series(y)

    try:
        if req.method == "stl":
            stl = STL(series, period=req.period)
            result = stl.fit()
            trend = result.trend.values
            seasonal = result.seasonal.values
            residual = result.resid.values
        elif req.method in ("additive", "multiplicative"):
            result = seasonal_decompose(series, model=req.method, period=req.period)
            trend = result.trend.values
            seasonal = result.seasonal.values
            residual = result.resid.values
        else:
            raise HTTPException(400, f"Unknown method '{req.method}'. Use 'stl', 'additive', or 'multiplicative'.")
    except Exception as e:
        raise HTTPException(400, f"Decomposition failed: {str(e)}")

    # Strength of trend / seasonality (Wang, Smith & Hyndman, 2006):
    #   F_T = max(0, 1 - Var(R) / Var(T + R)),  F_S = max(0, 1 - Var(R) / Var(S + R))
    # Defined for additive components; for a multiplicative fit they are computed on
    # the log scale, where the decomposition is additive.
    ok = ~(np.isnan(trend) | np.isnan(seasonal) | np.isnan(residual))
    T_, S_, R_ = trend[ok], seasonal[ok], residual[ok]
    if req.method == "multiplicative":
        if (T_ <= 0).any() or (S_ <= 0).any() or (R_ <= 0).any():
            T_ = S_ = R_ = np.array([])
        else:
            T_, S_, R_ = np.log(T_), np.log(S_), np.log(R_)
    strength_note = None
    if n < 3 * req.period:
        # With only two cycles the seasonal and trend fits can absorb (almost) all
        # variation, leaving a near-zero remainder; strengths would read ~1 regardless.
        T_ = S_ = R_ = np.array([])
        strength_note = "Strength measures need at least 3 full periods; with fewer the remainder is near zero by construction."
    if len(R_) > 1:
        var_r = np.var(R_, ddof=1)
        v_tr, v_sr = np.var(T_ + R_, ddof=1), np.var(S_ + R_, ddof=1)
        trend_strength = max(0.0, 1 - var_r / v_tr) if v_tr > 0 else None
        seasonal_strength = max(0.0, 1 - var_r / v_sr) if v_sr > 0 else None
    else:
        trend_strength = seasonal_strength = None

    return {
        "method": req.method,
        "period": req.period,
        "trend": _safe_list(trend),
        "seasonal": _safe_list(seasonal),
        "residual": _safe_list(residual),
        "trend_strength": _safe_float(trend_strength) if trend_strength is not None else None,
        "seasonal_strength": _safe_float(seasonal_strength) if seasonal_strength is not None else None,
        "strength_note": strength_note,
        "n_observations": n,
    }


# ── 5. Stationarity Tests ────────────────────────────────────────────

@router.post("/stationarity")
def stationarity_tests(req: StationarityRequest):
    """ADF, KPSS, and Phillips-Perron stationarity tests with differencing recommendation."""
    y = np.array(req.values, dtype=float)
    n = len(y)
    if n < 10:
        raise HTTPException(400, "Need at least 10 observations for stationarity tests.")

    results = {}

    # 1. Augmented Dickey-Fuller
    try:
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(y, autolag="AIC")
        results["adf"] = {
            "test_statistic": _safe_float(adf_stat),
            "p_value": _safe_float(adf_p),
            "lags_used": int(adf_lags),
            "n_observations": int(adf_nobs),
            "critical_values": {k: _safe_float(v) for k, v in adf_crit.items()},
            "stationary": adf_p < 0.05,
            "null_hypothesis": "Unit root (non-stationary)",
        }
    except Exception as e:
        results["adf"] = {"error": str(e)}

    # 2. KPSS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(y, regression="c", nlags="auto")
        results["kpss"] = {
            "test_statistic": _safe_float(kpss_stat),
            "p_value": _safe_float(kpss_p),
            "lags_used": int(kpss_lags),
            "critical_values": {k: _safe_float(v) for k, v in kpss_crit.items()},
            "stationary": kpss_p >= 0.05,  # KPSS: H0 = stationary
            "null_hypothesis": "Stationary",
        }
    except Exception as e:
        results["kpss"] = {"error": str(e)}

    # 3. Phillips–Perron Z_t (constant, no trend)
    try:
        z_t, pp_lags, T_pp = _phillips_perron_zt(y)
        pp_p = float(mackinnonp(z_t, regression="c", N=1))
        crit = mackinnoncrit(N=1, regression="c", nobs=T_pp)
        results["phillips_perron"] = {
            "test_statistic": _safe_float(z_t),
            "p_value": _safe_float(pp_p),
            "lags_used": int(pp_lags),
            "critical_values": {"1%": _safe_float(crit[0]), "5%": _safe_float(crit[1]), "10%": _safe_float(crit[2])},
            "stationary": pp_p < 0.05,
            "null_hypothesis": "Unit root (non-stationary)",
            "note": "Z_t statistic with Newey–West (Bartlett) long-run variance; MacKinnon (1994, 2010) p-values",
        }
    except Exception as e:
        results["phillips_perron"] = {"error": str(e)}

    # Overall conclusion and recommended differencing
    adf_stationary = results.get("adf", {}).get("stationary", None)
    kpss_stationary = results.get("kpss", {}).get("stationary", None)
    pp_stationary = results.get("phillips_perron", {}).get("stationary", None)

    votes = [v for v in [adf_stationary, kpss_stationary, pp_stationary] if v is not None]
    is_stationary = sum(votes) > len(votes) / 2 if votes else None

    # Determine recommended differencing order
    recommended_d = 0
    if not is_stationary:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for d in range(1, req.max_diff + 1):
                y_d = np.diff(y, n=d)
                if len(y_d) < 10:
                    break
                try:
                    _, p, *_ = adfuller(y_d, autolag="AIC")
                    if p < 0.05:
                        recommended_d = d
                        break
                except Exception:
                    continue
            else:
                recommended_d = req.max_diff

    results["conclusion"] = {
        "stationary": is_stationary,
        "recommended_differencing_order": recommended_d,
        "interpretation": (
            "Series appears stationary — no differencing needed."
            if is_stationary
            else f"Series appears non-stationary — recommend differencing of order {recommended_d}."
        ),
    }

    return results


def _phillips_perron_zt(y):
    """Phillips–Perron Z_t for y_t = a + rho*y_{t-1} + u_t (Hamilton 1994, eq. 17.6.12).

    Lags for the Bartlett-kernel long-run variance follow the common default
    ceil(12 * (T/100)^(1/4)).
    """
    y = np.asarray(y, dtype=float)
    yt, ylag = y[1:], y[:-1]
    T = len(yt)
    X = np.column_stack([np.ones(T), ylag])
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)  # regress y_t on (1, y_{t-1})
    u = yt - X @ beta
    k = X.shape[1]
    s2 = u @ u / (T - k)
    se_rho = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    t_rho = (beta[1] - 1) / se_rho
    L = int(np.ceil(12 * (T / 100) ** 0.25))
    L = min(L, T - 1)
    gamma0 = u @ u / T
    lam2 = gamma0 + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) / T for j in range(1, L + 1))
    z_t = np.sqrt(gamma0 / lam2) * t_rho - 0.5 * (lam2 - gamma0) / np.sqrt(lam2) * (T * se_rho / np.sqrt(s2))
    return float(z_t), L, T


# ── 6. Change Point Detection ────────────────────────────────────────

@router.post("/changepoint")
def changepoint_detection(req: ChangePointRequest):
    """Change point detection via CUSUM-based PELT or binary segmentation."""
    y = np.array(req.values, dtype=float)
    n = len(y)
    if req.min_segment_length < 2:
        raise HTTPException(400, "min_segment_length must be at least 2 (each segment estimates a variance).")
    if n < 2 * req.min_segment_length:
        raise HTTPException(400, f"Need at least {2 * req.min_segment_length} observations.")
    if n > 20000:
        raise HTTPException(400, "Change point detection is limited to 20,000 observations.")

    # Each extra segment adds a mean and a variance, so the BIC penalty per change
    # point is 2·ln(n); with it PELT minimises exactly the BIC reported below.
    penalty = req.penalty if req.penalty is not None else 2 * np.log(n)

    if req.method == "pelt":
        changepoints = _pelt(y, penalty, req.min_segment_length)
    elif req.method == "binseg":
        changepoints = _binary_segmentation(y, penalty, req.min_segment_length)
    else:
        raise HTTPException(400, f"Unknown method '{req.method}'. Use 'pelt' or 'binseg'.")

    # Compute segment statistics
    boundaries = [0] + sorted(changepoints) + [n]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg = y[start:end]
        segments.append({
            "start": start,
            "end": end - 1,
            "n": len(seg),
            "mean": _safe_float(np.mean(seg)),
            "std": _safe_float(np.std(seg, ddof=1)) if len(seg) > 1 else 0,
            "min": _safe_float(np.min(seg)),
            "max": _safe_float(np.max(seg)),
        })

    # BIC comparison: model with vs without change points
    bic_no_cp = _gaussian_bic(y, [])
    bic_with_cp = _gaussian_bic(y, changepoints)

    return {
        "method": req.method,
        "n_changepoints": len(changepoints),
        "changepoints": sorted(changepoints),
        "segments": segments,
        "model_comparison": {
            "bic_no_changepoints": _safe_float(bic_no_cp),
            "bic_with_changepoints": _safe_float(bic_with_cp),
            "delta_bic": _safe_float(bic_no_cp - bic_with_cp),
            "changepoints_preferred": bic_with_cp < bic_no_cp,
        },
        "penalty": _safe_float(penalty),
        "penalty_rule": "user-specified" if req.penalty is not None else "BIC: 2·ln(n) per change point (mean + variance)",
        "n_observations": n,
    }


def _segment_cost(y):
    """Gaussian negative log-likelihood cost for a segment (up to constant)."""
    n = len(y)
    if n == 0:
        return 0.0
    return n * np.log(max(np.var(y), 1e-15))


def _pelt(y, penalty, min_seg):
    """Pruned Exact Linear Time (PELT) change point detection using Gaussian cost."""
    n = len(y)
    # cumsum and cumsum-of-squares for O(1) cost computation
    cumsum = np.concatenate([[0], np.cumsum(y)])
    cumsum2 = np.concatenate([[0], np.cumsum(y ** 2)])

    def cost(start, end):
        """Gaussian cost for segment y[start:end]."""
        length = end - start
        if length <= 1:
            return 0.0
        s = cumsum[end] - cumsum[start]
        s2 = cumsum2[end] - cumsum2[start]
        var = s2 / length - (s / length) ** 2
        return length * np.log(max(var, 1e-15))

    def cost_vec(starts, end):
        length = end - starts
        sm_ = cumsum[end] - cumsum[starts]
        sq_ = cumsum2[end] - cumsum2[starts]
        var = np.maximum(sq_ / length - (sm_ / length) ** 2, 1e-15)
        return length * np.log(var)

    # F[t] = min penalised cost of segmenting y[0:t]
    F = np.full(n + 1, np.inf)
    F[0] = -penalty  # offset so first segment has no extra penalty
    last = np.zeros(n + 1, dtype=int)
    active = np.array([0])
    prune_at = np.full(n + 1, n + 1)

    # Pruning (Killick et al., 2012): if F[s] + C(s, t) > F[t], then s can never be the
    # last change point for any T at which t is itself a valid candidate. With a minimum
    # segment length that is only from T = t + min_seg onwards, so pruning is delayed.
    for t in range(min_seg, n + 1):
        active = active[prune_at[active] > t]
        cand = active[t - active >= min_seg]
        if not len(cand):
            continue  # y[0:t] cannot be split into segments of length >= min_seg
        c = cost_vec(cand, t)
        vals = F[cand] + c + penalty
        i = int(np.argmin(vals))
        F[t], last[t] = vals[i], cand[i]
        newly = cand[(F[cand] + c > F[t]) & (prune_at[cand] > n)]
        prune_at[newly] = t + min_seg
        active = np.append(active, t)

    cps, t = [], n
    while t > 0:
        t = int(last[t])
        if t > 0:
            cps.append(t)
    return sorted(cps)


def _binary_segmentation(y, penalty, min_seg):
    """Recursive binary segmentation for change point detection."""
    changepoints = []
    _binseg_recurse(y, 0, len(y), penalty, min_seg, changepoints)
    return changepoints


def _binseg_recurse(y, start, end, penalty, min_seg, changepoints):
    """Recursive helper for binary segmentation."""
    n = end - start
    if n < 2 * min_seg:
        return

    segment = np.asarray(y[start:end], dtype=float)
    base_cost = _segment_cost(segment)

    # Cost of every admissible split at once, from cumulative sums
    cs = np.concatenate([[0.0], np.cumsum(segment)])
    cs2 = np.concatenate([[0.0], np.cumsum(segment ** 2)])
    k = np.arange(min_seg, n - min_seg + 1)
    def seg_cost(sum_, sq_, length):
        return length * np.log(np.maximum(sq_ / length - (sum_ / length) ** 2, 1e-15))
    split = seg_cost(cs[k], cs2[k], k) + seg_cost(cs[n] - cs[k], cs2[n] - cs2[k], n - k)
    gains = base_cost - split
    i = int(np.argmax(gains))
    best_gain, best_cp = float(gains[i]), int(k[i])

    if best_cp is not None and best_gain > penalty:
        abs_cp = start + best_cp
        changepoints.append(abs_cp)
        _binseg_recurse(y, start, abs_cp, penalty, min_seg, changepoints)
        _binseg_recurse(y, abs_cp, end, penalty, min_seg, changepoints)


def _gaussian_bic(y, changepoints):
    """Compute BIC for a piecewise-constant Gaussian model."""
    n = len(y)
    boundaries = [0] + sorted(changepoints) + [n]
    k = len(changepoints)  # number of change points
    n_params = 2 * (k + 1)  # mean + variance per segment

    log_likelihood = 0.0
    for i in range(len(boundaries) - 1):
        seg = y[boundaries[i]:boundaries[i + 1]]
        seg_n = len(seg)
        if seg_n == 0:
            continue
        var = np.var(seg)
        if var < 1e-15:
            var = 1e-15
        log_likelihood += -0.5 * seg_n * (np.log(2 * np.pi * var) + 1)

    return n_params * np.log(n) - 2 * log_likelihood
