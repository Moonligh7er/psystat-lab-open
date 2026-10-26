from fastapi import APIRouter, HTTPException
import math
import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator
from typing import List, Dict, Any, Optional, Union
import uuid
from datetime import datetime
import time

from scipy import stats as scipy_stats
from scipy.stats.mstats import winsorize as scipy_winsorize
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

router = APIRouter(tags=["pipeline"])

Cell = Union[float, str, None]
MAX_CELLS = 1_000_000


class PipelineStep(BaseModel):
    id: str
    module: str
    action: str
    params: Dict[str, Any] = {}
    input_from: Optional[str] = None


class PipelineRequest(BaseModel):
    name: str = "Pipeline"
    # Column name -> values. Numeric columns hold numbers, categorical columns
    # hold strings; missing cells are null.
    data: Dict[str, List[Cell]]
    steps: List[PipelineStep]

    @model_validator(mode="after")
    def _check_shape(self):
        lengths = {len(v) for v in self.data.values()}
        if len(lengths) > 1:
            raise ValueError("All data columns must have the same number of rows")
        if lengths and next(iter(lengths)) * len(self.data) > MAX_CELLS:
            raise ValueError(f"Data too large (limit {MAX_CELLS:,} cells)")
        return self


class StepResult(BaseModel):
    step_id: str
    module: str
    action: str
    status: str  # "success" | "error"
    results: Dict[str, Any] = {}
    n_rows: int = 0
    output_data: Optional[Dict[str, List[Cell]]] = None
    timestamp: str


class PipelineResponse(BaseModel):
    pipeline_id: str
    name: str
    steps_completed: int
    steps_total: int
    results: List[StepResult]
    final_data: Dict[str, List[Cell]]
    execution_time_ms: float
    created_at: str


class ValidationIssue(BaseModel):
    step_id: str
    issue: str
    severity: str  # "error" | "warning"


class ValidationResponse(BaseModel):
    valid: bool
    issues: List[ValidationIssue]


SUPPORTED_ACTIONS: Dict[str, List[str]] = {
    "data-cleaning": [
        "remove-outliers", "log-transform", "z-score", "winsorize", "impute-mean",
    ],
    "assumptions": ["normality", "homogeneity", "multicollinearity"],
    "stats": ["descriptives", "t-test", "paired-t-test", "anova", "correlation", "regression"],
    "effect-size": ["compute"],
    "transform": ["recode", "filter"],
}

REQUIRED_PARAMS: Dict[str, Dict[str, List[str]]] = {
    "data-cleaning": {
        "log-transform": ["columns"],
        "z-score": ["columns"],
    },
    "assumptions": {
        "homogeneity": ["group_col", "value_col"],
        "multicollinearity": ["columns"],
    },
    "stats": {
        "t-test": ["group_col", "value_col"],
        "paired-t-test": ["col_1", "col_2"],
        "anova": ["group_col", "value_col"],
        "regression": ["dv", "ivs"],
    },
    "effect-size": {
        "compute": ["group_col", "value_col"],
    },
    "transform": {
        "recode": ["column", "mapping"],
        "filter": ["column", "operator", "value"],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _f(v, digits: int = 6):
    """Round to a JSON-safe float (NaN / ±inf -> None)."""
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


def _df_to_dict(df: pd.DataFrame) -> Dict[str, List[Cell]]:
    out: Dict[str, List[Cell]] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[col] = [float(v) if pd.notna(v) and math.isfinite(float(v)) else None for v in s]
        else:
            out[col] = [None if pd.isna(v) else str(v) for v in s]
    return out


def _to_frame(data: Dict[str, List[Cell]]) -> pd.DataFrame:
    df = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in data.items()})
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        # A column is numeric when every non-missing cell is a number.
        if converted.notna().sum() == df[c].notna().sum():
            df[c] = converted.astype(float)
    return df


def _col(df: pd.DataFrame, name: Any, numeric: bool = False) -> str:
    if not isinstance(name, str) or name not in df.columns:
        raise ValueError(f"Column not found: {name!r}")
    if numeric and not pd.api.types.is_numeric_dtype(df[name]):
        raise ValueError(f"Column {name!r} is not numeric")
    return name


def _cols(df: pd.DataFrame, params: Dict[str, Any], key: str = "columns", min_n: int = 1) -> List[str]:
    """Explicit numeric columns from params[key], or every numeric column."""
    raw = params.get(key)
    if raw in (None, "", []):
        cols = list(df.select_dtypes(include="number").columns)
    else:
        if isinstance(raw, str):
            raw = [c.strip() for c in raw.split(",") if c.strip()]
        cols = [_col(df, c, numeric=True) for c in raw]
    if len(cols) < min_n:
        raise ValueError(f"Need at least {min_n} numeric column{'s' if min_n > 1 else ''}")
    return cols


def _num(params: Dict[str, Any], key: str, default: float) -> float:
    v = params.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter {key!r} must be a number")


def _groups(df: pd.DataFrame, params: Dict[str, Any]):
    """(sorted level labels, arrays of values) for group_col / value_col, complete cases only."""
    g = _col(df, params.get("group_col"))
    v = _col(df, params.get("value_col"), numeric=True)
    if g == v:
        raise ValueError("group_col and value_col must be different columns")
    sub = df[[g, v]].dropna()
    levels = sorted(sub[g].unique(), key=lambda x: (isinstance(x, str), x))
    arrays = [sub.loc[sub[g] == lv, v].to_numpy(dtype=float) for lv in levels]
    return [str(int(lv)) if isinstance(lv, float) and lv.is_integer() else str(lv) for lv in levels], arrays


def _pooled_d(a: np.ndarray, b: np.ndarray):
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp else float("nan")
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)  # Hedges' small-sample correction
    return d, d * j


# ── Executors ────────────────────────────────────────────────────────────
# Each returns {"results": {...}, "df": DataFrame} and raises ValueError on bad input.

def _exec_remove_outliers(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    threshold = _num(params, "threshold", 3)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    cols = _cols(df, params)
    sub = df[cols]
    z = ((sub - sub.mean()) / sub.std(ddof=1)).abs()
    # A row is an outlier if any of its non-missing values has |z| >= threshold.
    outlier = (z >= threshold).any(axis=1)
    df_out = df.loc[~outlier].reset_index(drop=True)
    return {"results": {"rows_removed": int(outlier.sum()), "threshold_sd": threshold, "columns": cols}, "df": df_out}


def _exec_log_transform(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    cols = _cols(df, params)
    base = str(params.get("base", "e")).strip().lower()
    offset = _num(params, "offset", 0)
    logf = {"e": np.log, "ln": np.log, "10": np.log10, "2": np.log2}.get(base)
    if logf is None:
        raise ValueError("base must be 'e', '10' or '2'")
    for c in cols:
        shifted = df[c] + offset
        if (shifted.dropna() <= 0).any():
            raise ValueError(f"Column {c!r} has values ≤ 0 after adding offset {offset:g}; the log is undefined. Increase 'offset'.")
    df = df.copy()
    for c in cols:
        df[c] = logf(df[c] + offset)
    label = {"e": "ln", "ln": "ln", "10": "log10", "2": "log2"}[base]
    return {"results": {"transformed_columns": cols, "transform": f"{label}(x + {offset:g})"}, "df": df}


def _exec_z_score(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    cols = _cols(df, params)
    df = df.copy()
    constant = []
    for c in cols:
        sd = df[c].std(ddof=1)
        if not sd or not math.isfinite(sd):
            constant.append(c)
            continue
        df[c] = (df[c] - df[c].mean()) / sd
    if constant:
        raise ValueError(f"Cannot standardise constant column(s): {', '.join(constant)}")
    return {"results": {"normalized_columns": cols}, "df": df}


def _exec_winsorize(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    pct = _num(params, "percentile", 5)
    if not 0 < pct < 50:
        raise ValueError("percentile must be between 0 and 50 (per tail)")
    cols = _cols(df, params)
    df = df.copy()
    changed = {}
    for c in cols:
        mask = df[c].notna()
        x = df.loc[mask, c].to_numpy(dtype=float)
        w = np.asarray(scipy_winsorize(x, limits=(pct / 100, pct / 100)), dtype=float)
        changed[c] = int((w != x).sum())
        df.loc[mask, c] = w
    return {"results": {"percentile_per_tail": pct, "columns": cols, "values_changed": changed}, "df": df}


def _exec_impute_mean(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    cols = _cols(df, params)
    df = df.copy()
    imputed = []
    for c in cols:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            df[c] = df[c].fillna(df[c].mean())
            imputed.append({"column": c, "imputed_count": n_missing})
    return {"results": {
        "imputed": imputed,
        "note": "Mean imputation shrinks variances and correlations; prefer multiple imputation for inference.",
    }, "df": df}


def _exec_normality(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for c in _cols(df, params):
        x = df[c].dropna().to_numpy(dtype=float)
        if len(x) < 3:
            results[c] = {"W": None, "p": None, "n": len(x), "note": "Shapiro–Wilk needs n ≥ 3"}
            continue
        if np.ptp(x) == 0:
            results[c] = {"W": None, "p": None, "n": len(x), "note": "constant column"}
            continue
        W, p = scipy_stats.shapiro(x)
        results[c] = {"W": _f(W), "p": _f(p), "n": len(x)}
        if len(x) > 5000:
            results[c]["note"] = "n > 5000: Shapiro–Wilk p-value may be inaccurate"
    return {"results": {"test": "Shapiro–Wilk", "columns": results}, "df": df}


def _exec_homogeneity(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    center = str(params.get("center", "median"))
    if center not in ("median", "mean"):
        raise ValueError("center must be 'median' (Brown–Forsythe) or 'mean' (original Levene)")
    levels, groups = _groups(df, params)
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        raise ValueError("Need at least 2 groups with 2+ observations each")
    stat, p = scipy_stats.levene(*groups, center=center)
    N, k = sum(len(g) for g in groups), len(groups)
    return {"results": {
        "test": "Brown–Forsythe (Levene, median-centred)" if center == "median" else "Levene (mean-centred)",
        "F": _f(stat), "df1": k - 1, "df2": N - k, "p": _f(p), "groups": levels,
    }, "df": df}


def _exec_multicollinearity(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    cols = _cols(df, params, min_n=2)
    sub = df[cols].dropna()
    if len(sub) <= len(cols) + 1:
        raise ValueError("Not enough complete rows to compute VIFs")
    X = sm.add_constant(sub, has_constant="add").to_numpy(dtype=float)
    vifs = {c: _f(variance_inflation_factor(X, i + 1), 4) for i, c in enumerate(cols)}
    return {"results": {"vif": vifs, "n": len(sub),
                        "note": "VIF = inf (shown as null) means a predictor is a perfect linear combination of the others."}, "df": df}


def _exec_descriptives(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for c in _cols(df, params):
        x = df[c].dropna()
        results[c] = {
            "n": int(len(x)), "missing": int(df[c].isna().sum()),
            "mean": _f(x.mean(), 4) if len(x) else None,
            "sd": _f(x.std(ddof=1), 4) if len(x) > 1 else None,
            "median": _f(x.median(), 4) if len(x) else None,
            "min": _f(x.min(), 4) if len(x) else None,
            "max": _f(x.max(), 4) if len(x) else None,
            # pandas: bias-adjusted skewness (G1) and excess kurtosis (G2), as in SPSS
            "skew": _f(x.skew(), 4) if len(x) > 2 else None,
            "kurtosis": _f(x.kurtosis(), 4) if len(x) > 3 else None,
        }
    return {"results": results, "df": df}


def _exec_t_test(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    equal_var = str(params.get("equal_var", False)).lower() in ("true", "1", "yes")
    levels, groups = _groups(df, params)
    if len(groups) != 2:
        raise ValueError(f"Independent t-test needs exactly 2 groups in {params.get('group_col')!r}; found {len(groups)}")
    a, b = groups
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 observations")
    res = scipy_stats.ttest_ind(a, b, equal_var=equal_var)
    d, g = _pooled_d(a, b)
    return {"results": {
        "test": "Student's t (equal variances)" if equal_var else "Welch's t (unequal variances)",
        "groups": levels, "n_1": len(a), "n_2": len(b),
        "mean_1": _f(a.mean(), 4), "mean_2": _f(b.mean(), 4),
        "t": _f(res.statistic, 4), "df": _f(res.df, 4), "p": _f(res.pvalue),
        "cohens_d": _f(d, 4), "hedges_g": _f(g, 4),
    }, "df": df}


def _exec_paired_t_test(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    c1 = _col(df, params.get("col_1"), numeric=True)
    c2 = _col(df, params.get("col_2"), numeric=True)
    if c1 == c2:
        raise ValueError("col_1 and col_2 must be different columns")
    sub = df[[c1, c2]].dropna()
    if len(sub) < 2:
        raise ValueError("Need at least 2 rows with both columns present")
    diff = (sub[c1] - sub[c2]).to_numpy(dtype=float)
    res = scipy_stats.ttest_rel(sub[c1], sub[c2])
    sd = diff.std(ddof=1)
    return {"results": {
        "test": "Paired t", "n_pairs": len(sub),
        "mean_1": _f(sub[c1].mean(), 4), "mean_2": _f(sub[c2].mean(), 4), "mean_difference": _f(diff.mean(), 4),
        "t": _f(res.statistic, 4), "df": len(sub) - 1, "p": _f(res.pvalue),
        "cohens_dz": _f(diff.mean() / sd, 4) if sd else None,
    }, "df": df}


def _exec_anova(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    levels, groups = _groups(df, params)
    k, N = len(groups), sum(len(g) for g in groups)
    if k < 2 or N - k < 1:
        raise ValueError("Need at least 2 groups and more observations than groups")
    f_stat, p = scipy_stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ss_t = float(((allv - allv.mean()) ** 2).sum())
    ms_w = (ss_t - ss_b) / (N - k)
    return {"results": {
        "groups": levels, "group_n": [len(g) for g in groups], "group_means": [_f(g.mean(), 4) for g in groups],
        "F": _f(f_stat, 4), "df1": k - 1, "df2": N - k, "p": _f(p),
        "eta_squared": _f(ss_b / ss_t, 4) if ss_t else None,
        "omega_squared": _f((ss_b - (k - 1) * ms_w) / (ss_t + ms_w), 4) if ss_t else None,
    }, "df": df}


def _exec_correlation(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(params.get("method", "pearson")).lower()
    fn = {"pearson": scipy_stats.pearsonr, "spearman": scipy_stats.spearmanr, "kendall": scipy_stats.kendalltau}.get(method)
    if fn is None:
        raise ValueError("method must be pearson, spearman or kendall")
    cols = _cols(df, params, min_n=2)
    pairs = []
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            sub = df[[x, y]].dropna()  # pairwise deletion
            if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
                pairs.append({"x": x, "y": y, "r": None, "p": None, "n": len(sub)})
                continue
            r, p = fn(sub[x], sub[y])
            pairs.append({"x": x, "y": y, "r": _f(r, 4), "p": _f(p), "n": len(sub)})
    return {"results": {"method": method, "missing": "pairwise deletion", "pairs": pairs}, "df": df}


def _exec_regression(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    dv = _col(df, params.get("dv"), numeric=True)
    ivs = _cols(df, params, key="ivs")
    if dv in ivs:
        raise ValueError("The outcome (dv) cannot also be a predictor")
    sub = df[[dv] + ivs].dropna()
    if len(sub) <= len(ivs) + 1:
        raise ValueError("Not enough complete rows for this many predictors")
    model = sm.OLS(sub[dv], sm.add_constant(sub[ivs], has_constant="add")).fit()
    coefficients = {
        name: {"b": _f(b, 4), "se": _f(se, 4), "t": _f(t, 4), "p": _f(p)}
        for name, b, se, t, p in zip(model.params.index, model.params, model.bse, model.tvalues, model.pvalues)
    }
    return {"results": {
        "dv": dv, "n": int(model.nobs),
        "r_squared": _f(model.rsquared, 4), "adj_r_squared": _f(model.rsquared_adj, 4),
        "F": _f(model.fvalue, 4), "df1": int(model.df_model), "df2": int(model.df_resid), "p": _f(model.f_pvalue),
        "coefficients": coefficients,
    }, "df": df}


def _exec_effect_size(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    levels, groups = _groups(df, params)
    if len(groups) == 2:
        a, b = groups
        if len(a) < 2 or len(b) < 2:
            raise ValueError("Each group needs at least 2 observations")
        d, g = _pooled_d(a, b)
        return {"results": {"groups": levels, "cohens_d": _f(d, 4), "hedges_g": _f(g, 4),
                            "note": "d uses the pooled SD; sign is group 1 minus group 2."}, "df": df}
    if len(groups) > 2:
        allv = np.concatenate(groups)
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
        ss_t = float(((allv - allv.mean()) ** 2).sum())
        return {"results": {"groups": levels, "eta_squared": _f(ss_b / ss_t, 4) if ss_t else None}, "df": df}
    raise ValueError("Need at least 2 groups")


def _coerce_like(series: pd.Series, v: Any):
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{v!r} is not a number, but column {series.name!r} is numeric")
    return str(v)


def _exec_recode(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    col = _col(df, params.get("column"))
    mapping = params.get("mapping")
    if isinstance(mapping, str):  # "1=low, 2=high"
        pairs = [p.split("=", 1) for p in mapping.split(",") if "=" in p]
        mapping = {k.strip(): v.strip() for k, v in pairs}
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be like '1=low, 2=high'")
    keyed = {_coerce_like(df[col], k): v for k, v in mapping.items()}
    new_vals = list(keyed.values())
    numeric_out = all(_is_number(v) for v in new_vals)
    df = df.copy()
    hits = int(df[col].isin(list(keyed)).sum())
    replaced = df[col].map(lambda x: keyed.get(x, x) if pd.notna(x) else x)
    df[col] = pd.to_numeric(replaced, errors="coerce") if numeric_out and pd.api.types.is_numeric_dtype(df[col]) else replaced.astype(object)
    return {"results": {"column": col, "mapping_applied": {str(k): v for k, v in keyed.items()}, "cells_recoded": hits}, "df": df}


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _exec_filter(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    col = _col(df, params.get("column"))
    op = str(params.get("operator", ""))
    val = _coerce_like(df[col], params.get("value"))
    ops = {
        "==": lambda s: s == val, "!=": lambda s: s != val,
        ">": lambda s: s > val, ">=": lambda s: s >= val,
        "<": lambda s: s < val, "<=": lambda s: s <= val,
    }
    if op not in ops:
        raise ValueError(f"Unsupported operator {op!r}; use ==, !=, >, >=, <, <=")
    if isinstance(val, str) and op not in ("==", "!="):
        raise ValueError(f"Column {col!r} is categorical; only == and != are allowed")
    keep = ops[op](df[col]) & df[col].notna()
    before = len(df)
    df = df.loc[keep].reset_index(drop=True)
    return {"results": {"rows_before": before, "rows_after": len(df), "filter": f"{col} {op} {params.get('value')}"}, "df": df}


EXECUTORS = {
    ("data-cleaning", "remove-outliers"): _exec_remove_outliers,
    ("data-cleaning", "log-transform"): _exec_log_transform,
    ("data-cleaning", "z-score"): _exec_z_score,
    ("data-cleaning", "winsorize"): _exec_winsorize,
    ("data-cleaning", "impute-mean"): _exec_impute_mean,
    ("assumptions", "normality"): _exec_normality,
    ("assumptions", "homogeneity"): _exec_homogeneity,
    ("assumptions", "multicollinearity"): _exec_multicollinearity,
    ("stats", "descriptives"): _exec_descriptives,
    ("stats", "t-test"): _exec_t_test,
    ("stats", "paired-t-test"): _exec_paired_t_test,
    ("stats", "anova"): _exec_anova,
    ("stats", "correlation"): _exec_correlation,
    ("stats", "regression"): _exec_regression,
    ("effect-size", "compute"): _exec_effect_size,
    ("transform", "recode"): _exec_recode,
    ("transform", "filter"): _exec_filter,
}


@router.post("/run", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    start = time.time()
    if not req.data:
        raise HTTPException(status_code=422, detail="No data supplied")
    original_df = _to_frame(req.data)
    outputs: Dict[str, pd.DataFrame] = {}
    result_list: List[StepResult] = []
    completed = 0
    current_df = original_df

    for step in req.steps:
        ts = datetime.utcnow().isoformat() + "Z"
        # Each step reads the output of `input_from` (or the original data when unset).
        if step.input_from is None:
            input_df = original_df
        elif step.input_from in outputs:
            input_df = outputs[step.input_from]
        else:
            input_df = None

        executor = EXECUTORS.get((step.module, step.action))
        try:
            if executor is None:
                raise ValueError(f"Unknown action: {step.module}/{step.action}")
            if input_df is None:
                raise ValueError(f"input_from references a step that has not run: {step.input_from}")
            out = executor(input_df.copy(), step.params)
            out_df = out.get("df", input_df)
            status, results = "success", out.get("results", {})
            completed += 1
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            out_df = input_df if input_df is not None else current_df
            status, results = "error", {"error": str(exc)}

        outputs[step.id] = out_df
        current_df = out_df
        result_list.append(StepResult(
            step_id=step.id, module=step.module, action=step.action, status=status,
            results=results, n_rows=len(out_df), output_data=None, timestamp=ts,
        ))

    return PipelineResponse(
        pipeline_id=str(uuid.uuid4()),
        name=req.name,
        steps_completed=completed,
        steps_total=len(req.steps),
        results=result_list,
        final_data=_df_to_dict(current_df),
        execution_time_ms=round((time.time() - start) * 1000, 2),
        created_at=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_pipeline(req: PipelineRequest):
    issues: List[ValidationIssue] = []
    seen_ids: set = set()

    for step in req.steps:
        # Duplicate IDs
        if step.id in seen_ids:
            issues.append(ValidationIssue(step_id=step.id, issue="Duplicate step ID", severity="error"))
        seen_ids.add(step.id)

        # Unknown module/action
        if step.module not in SUPPORTED_ACTIONS:
            issues.append(ValidationIssue(step_id=step.id, issue=f"Unknown module: {step.module}", severity="error"))
        elif step.action not in SUPPORTED_ACTIONS[step.module]:
            issues.append(ValidationIssue(
                step_id=step.id,
                issue=f"Unknown action: {step.action} for module {step.module}",
                severity="error",
            ))

        # Invalid input_from reference
        if step.input_from and step.input_from not in seen_ids:
            issues.append(ValidationIssue(
                step_id=step.id,
                issue=f"input_from references unknown step: {step.input_from}",
                severity="error",
            ))

        # Required params
        req_params = REQUIRED_PARAMS.get(step.module, {}).get(step.action, [])
        for rp in req_params:
            if rp not in step.params:
                issues.append(ValidationIssue(
                    step_id=step.id,
                    issue=f"Missing required param: {rp}",
                    severity="error",
                ))

    return ValidationResponse(valid=len([i for i in issues if i.severity == "error"]) == 0, issues=issues)

@router.post("/templates")
async def get_templates():
    _s = lambda **kw: kw  # shorthand step builder
    gp = {"group_col": "group", "value_col": "score"}
    templates = [
        {"name": "Basic Analysis",
         "description": "Descriptives -> Normality -> T-test -> Effect Size",
         "steps": [
             _s(id="desc", module="stats", action="descriptives", params={}, input_from=None),
             _s(id="norm", module="assumptions", action="normality", params={}, input_from=None),
             _s(id="ttest", module="stats", action="t-test", params=gp, input_from=None),
             _s(id="effect", module="effect-size", action="compute", params=gp, input_from=None),
         ]},
        {"name": "Regression Prep",
         "description": "Impute -> Check Multicollinearity -> Regression",
         "steps": [
             _s(id="impute", module="data-cleaning", action="impute-mean", params={}, input_from=None),
             _s(id="vif", module="assumptions", action="multicollinearity", params={}, input_from="impute"),
             _s(id="reg", module="stats", action="regression",
                params={"dv": "outcome", "ivs": ["pred1", "pred2"]}, input_from="impute"),
         ]},
        {"name": "ANOVA Pipeline",
         "description": "Outlier removal -> Normality -> Homogeneity -> ANOVA -> Effect Sizes",
         "steps": [
             _s(id="outliers", module="data-cleaning", action="remove-outliers", params={"threshold": 3}, input_from=None),
             _s(id="norm", module="assumptions", action="normality", params={}, input_from="outliers"),
             _s(id="levene", module="assumptions", action="homogeneity", params=gp, input_from="outliers"),
             _s(id="anova", module="stats", action="anova", params=gp, input_from="outliers"),
             _s(id="effect", module="effect-size", action="compute", params=gp, input_from="outliers"),
         ]},
        {"name": "Psychometric",
         "description": "Descriptives -> Normality -> Correlation Matrix",
         "steps": [
             _s(id="desc", module="stats", action="descriptives", params={}, input_from=None),
             _s(id="norm", module="assumptions", action="normality", params={}, input_from=None),
             _s(id="corr", module="stats", action="correlation", params={"method": "pearson"}, input_from=None),
         ]},
    ]
    return {"templates": templates}
