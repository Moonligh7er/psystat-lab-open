from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(tags=["diagnostics"])


# ── Request / Response Models ───────────────────────────────────────────────

class QQPlotRequest(BaseModel):
    data: List[float]
    distribution: str = "norm"

class QQPlotResponse(BaseModel):
    theoretical: List[float]
    observed: List[float]
    fit_line: Dict[str, float]
    shapiro_stat: float
    shapiro_p: float

class ResidualRequest(BaseModel):
    dv: str
    ivs: List[str]
    data: Dict[str, List[float]]

class ResidualResponse(BaseModel):
    fitted_values: List[float]
    residuals: List[float]
    standardized_residuals: List[float]
    cooks_d: List[float]
    leverage: List[float]
    influential_points: List[int]
    breusch_pagan: Dict[str, float]
    durbin_watson: float
    normality_of_residuals: Dict[str, float]

class HomogeneityRequest(BaseModel):
    group_col: str
    value_col: str
    data: Dict[str, List]

class GroupInfo(BaseModel):
    name: str
    mean: float
    sd: float
    n: int
    values: List[float]

class HomogeneityResponse(BaseModel):
    groups: List[GroupInfo]
    levene: Dict[str, float]
    bartlett: Dict[str, float]
    brown_forsythe: Dict[str, float]
    variance_ratio: float

class MulticollinearityRequest(BaseModel):
    predictors: List[str]
    data: Dict[str, List[float]]

class VIFEntry(BaseModel):
    variable: str
    vif_value: float
    tolerance: float
    problematic: bool

class MulticollinearityResponse(BaseModel):
    vif: List[VIFEntry]
    condition_number: float
    correlation_matrix: List[List[float]]
    eigenvalues: List[float]

class OutlierRequest(BaseModel):
    data: Dict[str, List[float]]
    columns: List[str]

class ColumnOutlierResult(BaseModel):
    z_scores: List[float]
    outliers_3sd: List[int]
    outliers_iqr: List[int]

class OutlierResponse(BaseModel):
    per_column: Dict[str, ColumnOutlierResult]
    mahalanobis_distances: Optional[List[float]] = None
    grubbs_tests: Dict[str, Dict[str, Any]]


# ── Helpers ─────────────────────────────────────────────────────────────────

_QQ_DISTS = {"norm": stats.norm, "t": stats.t, "lognorm": stats.lognorm, "expon": stats.expon,
             "gamma": stats.gamma, "uniform": stats.uniform, "logistic": stats.logistic}


def _frame(data: Dict[str, List[Any]], columns: List[str], numeric: bool = True) -> pd.DataFrame:
    """Build a DataFrame from column lists, checking names, lengths and numeric content."""
    if not data:
        raise HTTPException(422, "No data supplied.")
    lengths = {len(v) for v in data.values()}
    if len(lengths) > 1:
        raise HTTPException(422, "All data columns must have the same length.")
    missing = [c for c in columns if c not in data]
    if missing:
        raise HTTPException(422, f"Column(s) not found in data: {', '.join(missing)}")
    df = pd.DataFrame(data)
    if numeric:
        for c in columns:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() < df[c].notna().sum():
                raise HTTPException(422, f"Column {c!r} must be numeric.")
            df[c] = conv
    return df


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/qq-plot", response_model=QQPlotResponse)
async def qq_plot(req: QQPlotRequest):
    """Generate theoretical vs observed quantiles for QQ plot rendering."""
    arr = np.array(req.data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        raise HTTPException(422, "Need at least 3 values.")
    dist = _QQ_DISTS.get(req.distribution)
    if dist is None:
        raise HTTPException(422, f"distribution must be one of: {', '.join(_QQ_DISTS)}")
    params = dist.fit(arr)  # maximum-likelihood fit of the reference distribution

    sorted_data = np.sort(arr)
    n = len(sorted_data)
    theoretical_q = dist.ppf(
        (np.arange(1, n + 1) - 0.5) / n, *params[:-2], loc=params[-2], scale=params[-1]
    )

    slope, intercept, _, _, _ = stats.linregress(theoretical_q, sorted_data)

    # Shapiro-Wilk tests normality, so it is only reported for the normal reference
    if req.distribution == "norm":
        shapiro_stat, shapiro_p = stats.shapiro(arr)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")

    return QQPlotResponse(
        theoretical=theoretical_q.tolist(),
        observed=sorted_data.tolist(),
        fit_line={"slope": float(slope), "intercept": float(intercept)},
        shapiro_stat=float(shapiro_stat),
        shapiro_p=float(shapiro_p),
    )


@router.post("/residual-diagnostics", response_model=ResidualResponse)
async def residual_diagnostics(req: ResidualRequest):
    """Compute full residual diagnostics for an OLS regression model."""
    if not req.ivs:
        raise HTTPException(422, "Provide at least one predictor (ivs).")
    df = _frame(req.data, [req.dv] + req.ivs).dropna(subset=[req.dv] + req.ivs).reset_index(drop=True)
    if len(df) <= len(req.ivs) + 2:
        raise HTTPException(422, "Not enough complete rows for this many predictors.")
    y = df[req.dv].values.astype(float)
    X = df[req.ivs].values.astype(float)
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()
    influence = model.get_influence()

    residuals = model.resid
    fitted = model.fittedvalues
    std_resid = influence.resid_studentized_internal
    cooks = influence.cooks_distance[0]
    hat_vals = influence.hat_matrix_diag
    n = len(y)
    influential = [int(i) for i in range(n) if cooks[i] > 4.0 / n]

    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)
    dw = durbin_watson(residuals)

    if len(residuals) >= 3:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = float("nan"), float("nan")

    return ResidualResponse(
        fitted_values=fitted.tolist(),
        residuals=residuals.tolist(),
        standardized_residuals=std_resid.tolist(),
        cooks_d=cooks.tolist(),
        leverage=hat_vals.tolist(),
        influential_points=influential,
        breusch_pagan={"statistic": float(bp_stat), "p_value": float(bp_p)},
        durbin_watson=float(dw),
        normality_of_residuals={"shapiro_stat": float(sw_stat), "shapiro_p": float(sw_p)},
    )


@router.post("/homogeneity-plot", response_model=HomogeneityResponse)
async def homogeneity_plot(req: HomogeneityRequest):
    """Assess homogeneity of variance across groups for ANOVA / t-test."""
    df = _frame(req.data, [req.group_col], numeric=False)
    df = _frame(req.data, [req.value_col]).assign(**{req.group_col: df[req.group_col]})
    group_names = df[req.group_col].dropna().unique().tolist()
    if len(group_names) < 2:
        raise HTTPException(422, "Need at least 2 groups.")

    group_data: Dict[str, np.ndarray] = {}
    group_infos: List[GroupInfo] = []
    for g in group_names:
        vals = df.loc[df[req.group_col] == g, req.value_col].astype(float).dropna().values
        group_data[str(g)] = vals
        group_infos.append(GroupInfo(
            name=str(g),
            mean=float(np.mean(vals)),
            sd=float(np.std(vals, ddof=1)),
            n=int(len(vals)),
            values=vals.tolist(),
        ))

    arrays = list(group_data.values())
    if any(len(a) < 2 for a in arrays):
        raise HTTPException(422, "Every group needs at least 2 values.")

    lev_stat, lev_p = stats.levene(*arrays, center="mean")
    bart_stat, bart_p = stats.bartlett(*arrays)
    bf_stat, bf_p = stats.levene(*arrays, center="median")

    variances = [float(np.var(a, ddof=1)) for a in arrays]
    variance_ratio = max(variances) / min(variances) if min(variances) > 0 else float("inf")

    return HomogeneityResponse(
        groups=group_infos,
        levene={"statistic": float(lev_stat), "p_value": float(lev_p)},
        bartlett={"statistic": float(bart_stat), "p_value": float(bart_p)},
        brown_forsythe={"statistic": float(bf_stat), "p_value": float(bf_p)},
        variance_ratio=float(variance_ratio),
    )


@router.post("/multicollinearity", response_model=MulticollinearityResponse)
async def multicollinearity(req: MulticollinearityRequest):
    """Compute VIF, condition number, correlation matrix, and eigenvalues."""
    if len(req.predictors) < 2:
        raise HTTPException(422, "Need at least 2 predictors.")
    df = _frame(req.data, req.predictors)[req.predictors].dropna()
    if len(df) <= len(req.predictors) + 1:
        raise HTTPException(422, "Not enough complete rows.")
    X = df.values
    X_const = sm.add_constant(X)

    vif_entries: List[VIFEntry] = []
    for i, col in enumerate(req.predictors):
        vif_val = float(variance_inflation_factor(X_const, i + 1))  # skip constant
        vif_entries.append(VIFEntry(
            variable=col,
            vif_value=round(vif_val, 4),
            tolerance=round(1.0 / vif_val, 4) if vif_val != 0 else 0.0,
            problematic=vif_val > 10,
        ))

    corr = np.corrcoef(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    # Condition number of the predictor correlation matrix: sqrt(largest / smallest eigenvalue)
    smallest = float(np.min(np.abs(eigenvalues)))
    condition_number = float(np.sqrt(max(eigenvalues) / smallest)) if smallest > 1e-15 else float("inf")

    return MulticollinearityResponse(
        vif=vif_entries,
        condition_number=round(condition_number, 4),
        correlation_matrix=np.round(corr, 6).tolist(),
        eigenvalues=np.round(np.sort(eigenvalues)[::-1], 6).tolist(),
    )


def _grubbs_test(data: np.ndarray):
    """Two-sided Grubbs test for a single outlier."""
    n = len(data)
    if n < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "outlier_index": None}
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    if sd == 0:
        return {"statistic": 0.0, "p_value": 1.0, "outlier_index": None}

    abs_dev = np.abs(data - mean)
    max_idx = int(np.argmax(abs_dev))
    G = float(abs_dev[max_idx] / sd)

    # G = (n-1)/sqrt(n) * sqrt(t^2 / (n-2+t^2))  <=>  t^2 = n(n-2)G^2 / ((n-1)^2 - nG^2)
    denom = (n - 1) ** 2 - n * G ** 2
    if denom <= 0:
        p_value = 0.0
    else:
        t_stat = np.sqrt(n * (n - 2) * G ** 2 / denom)
        p_value = float(min(1.0, 2 * n * stats.t.sf(t_stat, n - 2)))  # two-sided (as outliers::grubbs.test)
    g_crit = (n - 1) / np.sqrt(n) * np.sqrt(stats.t.ppf(1 - 0.05 / (2 * n), n - 2) ** 2
                                           / (n - 2 + stats.t.ppf(1 - 0.05 / (2 * n), n - 2) ** 2))

    return {"statistic": round(G, 6), "p_value": round(p_value, 8), "critical_value_05": round(float(g_crit), 6),
            "outlier_index": max_idx, "outlier_value": float(data[max_idx])}


@router.post("/outlier-diagnostics", response_model=OutlierResponse)
async def outlier_diagnostics(req: OutlierRequest):
    """Detect outliers via z-scores, IQR, Mahalanobis distance, and Grubbs test."""
    if not req.columns:
        raise HTTPException(422, "Provide at least one column.")
    df = _frame(req.data, req.columns)
    per_column: Dict[str, ColumnOutlierResult] = {}
    grubbs_tests: Dict[str, Dict[str, Any]] = {}

    for col in req.columns:
        vals = df[col].astype(float).values
        mean, sd = np.nanmean(vals), np.nanstd(vals, ddof=1)
        z = ((vals - mean) / sd if sd > 0 else np.zeros_like(vals)).tolist()

        outliers_3sd = [int(i) for i, v in enumerate(z) if abs(v) > 3]

        q1, q3 = np.nanpercentile(vals, 25), np.nanpercentile(vals, 75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_iqr = [int(i) for i, v in enumerate(vals) if v < lower or v > upper]

        per_column[col] = ColumnOutlierResult(
            z_scores=[round(v, 6) for v in z],
            outliers_3sd=outliers_3sd,
            outliers_iqr=outliers_iqr,
        )
        grubbs_tests[col] = _grubbs_test(vals[~np.isnan(vals)])

    # Mahalanobis distances when multiple columns are requested
    mahal: Optional[List[float]] = None
    if len(req.columns) > 1:
        sub = df[req.columns].astype(float).dropna()
        if len(sub) > len(req.columns):
            mean_vec = sub.mean().values
            cov = sub.cov().values
            try:
                cov_inv = np.linalg.inv(cov)
                diff = sub.values - mean_vec
                mahal = [round(float(np.sqrt(d @ cov_inv @ d)), 6) for d in diff]
            except np.linalg.LinAlgError:
                mahal = None

    return OutlierResponse(
        per_column=per_column,
        mahalanobis_distances=mahal,
        grubbs_tests=grubbs_tests,
    )
