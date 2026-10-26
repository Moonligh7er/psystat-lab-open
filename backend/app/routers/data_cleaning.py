"""Data Cleaning & Preprocessing — outlier detection, missing data, transformations."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


class OutlierRequest(BaseModel):
    data: list[float]
    method: str = "iqr"  # "iqr", "zscore", "grubbs", "modified_zscore"
    threshold: float | None = None  # custom threshold (default depends on method)


class MissingDataRequest(BaseModel):
    columns: list[list[float | None]]  # each inner list is a variable
    labels: list[str] | None = None
    method: str = "listwise"  # "listwise", "mean", "median", "mode", "linear"


class TransformRequest(BaseModel):
    data: list[float]
    transform: str  # "log", "sqrt", "reciprocal", "zscore", "minmax", "boxcox", "rank"


class NormalityRequest(BaseModel):
    data: list[float]


class DescribeColumnsRequest(BaseModel):
    columns: list[list[float | None]]
    labels: list[str] | None = None


@router.post("/outliers")
def detect_outliers(req: OutlierRequest):
    """Detect outliers using various methods."""
    data = np.array(req.data, dtype=float)
    n = len(data)
    if n < 4:
        raise HTTPException(400, "Need at least 4 observations for outlier detection.")

    method = req.method.lower()
    outlier_indices = []
    details = {}

    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        multiplier = req.threshold or 1.5
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_indices = [int(i) for i in range(n) if data[i] < lower or data[i] > upper]
        details = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "lower_fence": float(lower), "upper_fence": float(upper), "multiplier": multiplier}

    elif method == "zscore":
        threshold = req.threshold or 3.0
        z_scores = np.abs((data - data.mean()) / data.std(ddof=1))
        outlier_indices = [int(i) for i in range(n) if z_scores[i] > threshold]
        details = {"mean": float(data.mean()), "std": float(data.std(ddof=1)), "threshold": threshold, "z_scores": [float(z) for z in z_scores]}

    elif method == "modified_zscore":
        threshold = req.threshold or 3.5
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / mad if mad > 0 else np.zeros(n)
        outlier_indices = [int(i) for i in range(n) if abs(modified_z[i]) > threshold]
        details = {"median": float(median), "mad": float(mad), "threshold": threshold, "modified_z_scores": [float(z) for z in modified_z]}

    elif method == "grubbs":
        threshold = req.threshold or 0.05
        mean, std = data.mean(), data.std(ddof=1)
        if std == 0:
            return {"outlier_indices": [], "outlier_values": [], "method": method, "n": n, "details": {}}
        abs_devs = np.abs(data - mean)
        max_idx = int(np.argmax(abs_devs))
        G = abs_devs[max_idx] / std
        # Critical value
        t_crit = sp_stats.t.ppf(1 - threshold / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        if G > G_crit:
            outlier_indices = [max_idx]
        details = {"G_statistic": float(G), "G_critical": float(G_crit), "alpha": threshold, "suspect_index": max_idx, "suspect_value": float(data[max_idx])}

    else:
        raise HTTPException(400, f"Unknown method: {method}. Options: iqr, zscore, modified_zscore, grubbs")

    return {
        "outlier_indices": outlier_indices,
        "outlier_values": [float(data[i]) for i in outlier_indices],
        "n_outliers": len(outlier_indices),
        "n_total": n,
        "percent_outliers": float(len(outlier_indices) / n * 100),
        "clean_data": [float(data[i]) for i in range(n) if i not in outlier_indices],
        "clean_n": n - len(outlier_indices),
        "method": method,
        "details": details,
    }


@router.post("/missing-data")
def handle_missing_data(req: MissingDataRequest):
    """Analyze and impute missing data."""
    n_vars = len(req.columns)
    if n_vars == 0:
        raise HTTPException(400, "No columns provided.")

    labels = req.labels or [f"V{i+1}" for i in range(n_vars)]
    n_obs = max(len(col) for col in req.columns)

    # Analysis
    missing_report = []
    for i, col in enumerate(req.columns):
        n_miss = sum(1 for v in col if v is None)
        missing_report.append({
            "variable": labels[i],
            "n_total": len(col),
            "n_missing": n_miss,
            "percent_missing": float(n_miss / len(col) * 100) if len(col) > 0 else 0,
        })

    total_cells = sum(len(col) for col in req.columns)
    total_missing = sum(r["n_missing"] for r in missing_report)

    # Imputation
    method = req.method.lower()
    imputed_columns = []

    if method == "listwise":
        # Remove rows where any value is None
        valid_rows = set(range(n_obs))
        for col in req.columns:
            for j in range(min(len(col), n_obs)):
                if col[j] is None:
                    valid_rows.discard(j)
        for col in req.columns:
            imputed_columns.append([col[j] for j in sorted(valid_rows) if j < len(col)])
        n_removed = n_obs - len(valid_rows)
    elif method in ("mean", "median", "mode"):
        n_removed = 0
        for col in req.columns:
            valid = [v for v in col if v is not None]
            if not valid:
                imputed_columns.append(col)
                continue
            if method == "mean":
                fill = float(np.mean(valid))
            elif method == "median":
                fill = float(np.median(valid))
            else:
                from scipy.stats import mode as sp_mode
                fill = float(sp_mode(valid, keepdims=True).mode[0])
            imputed_columns.append([v if v is not None else fill for v in col])
    elif method == "linear":
        n_removed = 0
        for col in req.columns:
            arr = np.array([float(v) if v is not None else np.nan for v in col])
            nans = np.isnan(arr)
            if nans.all() or not nans.any():
                imputed_columns.append(col)
                continue
            valid_idx = np.where(~nans)[0]
            nan_idx = np.where(nans)[0]
            arr[nan_idx] = np.interp(nan_idx, valid_idx, arr[valid_idx])
            imputed_columns.append([float(v) for v in arr])
    else:
        raise HTTPException(400, f"Unknown method: {method}. Options: listwise, mean, median, mode, linear")

    return {
        "missing_report": missing_report,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "percent_missing": float(total_missing / total_cells * 100) if total_cells > 0 else 0,
        "method": method,
        "imputed_columns": imputed_columns,
        "rows_after_imputation": len(imputed_columns[0]) if imputed_columns else 0,
    }


@router.post("/transform")
def transform_data(req: TransformRequest):
    """Apply data transformations."""
    data = np.array(req.data, dtype=float)
    n = len(data)
    if n < 2:
        raise HTTPException(400, "Need at least 2 observations.")

    transform = req.transform.lower()

    if transform == "log":
        if np.any(data <= 0):
            # Shift to make all positive
            shift = abs(data.min()) + 1
            transformed = np.log(data + shift)
            note = f"Data shifted by +{shift:.2f} to make positive before log"
        else:
            transformed = np.log(data)
            note = "Natural log applied"
    elif transform == "sqrt":
        if np.any(data < 0):
            shift = abs(data.min())
            transformed = np.sqrt(data + shift)
            note = f"Data shifted by +{shift:.2f} before sqrt"
        else:
            transformed = np.sqrt(data)
            note = "Square root applied"
    elif transform == "reciprocal":
        if np.any(data == 0):
            data[data == 0] = 1e-10
        transformed = 1 / data
        note = "Reciprocal (1/x) applied"
    elif transform == "zscore":
        transformed = (data - data.mean()) / data.std(ddof=1)
        note = f"Z-scored (M={data.mean():.3f}, SD={data.std(ddof=1):.3f})"
    elif transform == "minmax":
        mn, mx = data.min(), data.max()
        rng = mx - mn or 1
        transformed = (data - mn) / rng
        note = f"Min-max scaled to [0, 1] (range: {mn:.3f}–{mx:.3f})"
    elif transform == "boxcox":
        if np.any(data <= 0):
            shift = abs(data.min()) + 1
            transformed, lam = sp_stats.boxcox(data + shift)
            note = f"Box-Cox (λ={lam:.4f}), shifted by +{shift:.2f}"
        else:
            transformed, lam = sp_stats.boxcox(data)
            note = f"Box-Cox (λ={lam:.4f})"
    elif transform == "rank":
        transformed = sp_stats.rankdata(data).astype(float)
        note = "Rank-ordered"
    else:
        raise HTTPException(400, f"Unknown transform: {transform}. Options: log, sqrt, reciprocal, zscore, minmax, boxcox, rank")

    # Test normality before and after
    _, orig_p = sp_stats.shapiro(data) if n <= 5000 else (None, None)
    _, trans_p = sp_stats.shapiro(transformed) if n <= 5000 else (None, None)

    return {
        "original": [float(v) for v in data],
        "transformed": [float(v) for v in transformed],
        "transform": transform,
        "note": note,
        "original_shapiro_p": float(orig_p) if orig_p is not None else None,
        "transformed_shapiro_p": float(trans_p) if trans_p is not None else None,
        "normality_improved": trans_p > orig_p if orig_p is not None and trans_p is not None else None,
        "original_skewness": float(sp_stats.skew(data)),
        "transformed_skewness": float(sp_stats.skew(transformed)),
        "original_kurtosis": float(sp_stats.kurtosis(data)),
        "transformed_kurtosis": float(sp_stats.kurtosis(transformed)),
    }


@router.post("/normality")
def normality_tests(req: NormalityRequest):
    """Comprehensive normality testing."""
    data = np.array(req.data, dtype=float)
    n = len(data)
    if n < 4:
        raise HTTPException(400, "Need at least 4 observations.")

    results = {
        "n": n,
        "skewness": float(sp_stats.skew(data)),
        "kurtosis": float(sp_stats.kurtosis(data)),
    }

    # Shapiro-Wilk (best for n < 5000)
    if n <= 5000:
        sw_stat, sw_p = sp_stats.shapiro(data)
        results["shapiro_wilk"] = {"statistic": float(sw_stat), "p_value": float(sw_p), "normal": bool(sw_p > 0.05)}

    # D'Agostino-Pearson (for n >= 8)
    if n >= 8:
        try:
            dp_stat, dp_p = sp_stats.normaltest(data)
            results["dagostino_pearson"] = {"statistic": float(dp_stat), "p_value": float(dp_p), "normal": bool(dp_p > 0.05)}
        except Exception:
            pass

    # Kolmogorov-Smirnov
    ks_stat, ks_p = sp_stats.kstest(data, 'norm', args=(data.mean(), data.std(ddof=1)))
    results["kolmogorov_smirnov"] = {"statistic": float(ks_stat), "p_value": float(ks_p), "normal": bool(ks_p > 0.05)}

    # Anderson-Darling
    ad_result = sp_stats.anderson(data, dist='norm')
    results["anderson_darling"] = {
        "statistic": float(ad_result.statistic),
        "critical_values": {f"{sl}%": float(cv) for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)},
        "normal_at_5pct": bool(ad_result.statistic < ad_result.critical_values[2]),  # 5% level
    }

    # Q-Q data for plotting
    sorted_data = np.sort(data)
    theoretical_quantiles = [float(sp_stats.norm.ppf((i + 0.5) / n)) for i in range(n)]
    results["qq_data"] = {
        "theoretical": theoretical_quantiles,
        "observed": [float(v) for v in sorted_data],
    }

    # Overall assessment
    tests_normal = []
    if "shapiro_wilk" in results:
        tests_normal.append(results["shapiro_wilk"]["normal"])
    if "dagostino_pearson" in results:
        tests_normal.append(results["dagostino_pearson"]["normal"])
    tests_normal.append(results["kolmogorov_smirnov"]["normal"])
    tests_normal.append(results["anderson_darling"]["normal_at_5pct"])

    results["consensus_normal"] = sum(tests_normal) > len(tests_normal) / 2
    results["recommendation"] = (
        "Data appears normally distributed. Parametric tests are appropriate."
        if results["consensus_normal"]
        else "Data deviates from normality. Consider non-parametric tests or data transformation."
    )

    return results


@router.post("/describe")
def describe_columns(req: DescribeColumnsRequest):
    """Comprehensive descriptive statistics for multiple columns."""
    labels = req.labels or [f"V{i+1}" for i in range(len(req.columns))]
    summaries = []

    for i, col in enumerate(req.columns):
        valid = np.array([v for v in col if v is not None], dtype=float)
        n_missing = len(col) - len(valid)

        if len(valid) == 0:
            summaries.append({"variable": labels[i], "n": 0, "n_missing": n_missing, "all_missing": True})
            continue

        summary = {
            "variable": labels[i],
            "n": len(valid),
            "n_missing": n_missing,
            "percent_missing": float(n_missing / len(col) * 100),
            "mean": float(valid.mean()),
            "median": float(np.median(valid)),
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0,
            "min": float(valid.min()),
            "max": float(valid.max()),
            "range": float(valid.max() - valid.min()),
            "q1": float(np.percentile(valid, 25)),
            "q3": float(np.percentile(valid, 75)),
            "iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
            "skewness": float(sp_stats.skew(valid)),
            "kurtosis": float(sp_stats.kurtosis(valid)),
            "sem": float(sp_stats.sem(valid)) if len(valid) > 1 else 0,
        }

        if len(valid) >= 3:
            _, p = sp_stats.shapiro(valid) if len(valid) <= 5000 else (None, None)
            if p is not None:
                summary["shapiro_p"] = float(p)
                summary["normal"] = p > 0.05

        summaries.append(summary)

    return {"variables": summaries, "n_variables": len(summaries)}
