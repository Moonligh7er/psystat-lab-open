"""Multiverse / Specification Curve Analysis router.
Runs every combination of analyst decisions and reports how robust the
finding is across the full specification space.
"""
from __future__ import annotations
import math
from itertools import product
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats
import statsmodels.api as sm

router = APIRouter()

# ── Models ───────────────────────────────────────────────────────────────

class DecisionPoint(BaseModel):
    name: str
    options: List[Dict[str, Any]]

class MultiverseRequest(BaseModel):
    data: Dict[str, List[float]]
    dv: str; iv: str
    covariates: Optional[List[str]] = None
    test_type: str  # "t-test", "correlation", "regression", "anova"
    decisions: List[DecisionPoint]

class MultiverseSummaryRequest(BaseModel):
    total_specifications: int; significant_count: int; significant_percentage: float
    median_effect_size: float; effect_size_range: List[float]
    specification_curve: List[Dict[str, Any]]
    decision_influence: List[Dict[str, Any]]
    robustness_verdict: str
    test_type: str = "t-test"; dv: str = "dv"; iv: str = "iv"

# ── Helpers ──────────────────────────────────────────────────────────────

def _safe(v) -> Optional[float]:
    v = float(v)
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)

def _apply_outlier(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    s = df[col].dropna()
    if method == "none" or s.empty:
        return df
    if method in ("2sd", "3sd"):
        nsd = int(method[0])
        lo, hi = s.mean() - nsd * s.std(), s.mean() + nsd * s.std()
        return df[(df[col] >= lo) & (df[col] <= hi)].copy()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iq = q3 - q1
        return df[(df[col] >= q1 - 1.5*iq) & (df[col] <= q3 + 1.5*iq)].copy()
    if method.startswith("winsorize"):
        pct = int(method.split("_")[1]) / 100
        lo, hi = s.quantile(pct), s.quantile(1 - pct)
        df = df.copy(); df.loc[df[col] < lo, col] = lo; df.loc[df[col] > hi, col] = hi
        return df
    return df

def _apply_missing(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    if method == "listwise":
        return df.dropna(subset=[col]).copy()
    if method == "mean_imputation":
        df = df.copy(); df[col] = df[col].fillna(df[col].mean()); return df
    if method == "median_imputation":
        df = df.copy(); df[col] = df[col].fillna(df[col].median()); return df
    return df

def _apply_transform(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    if method == "none":
        return df
    df = df.copy()
    if method == "log":
        shift = abs(df[col].min()) + 1 if df[col].min() <= 0 else 0
        df[col] = np.log(df[col] + shift)
    elif method == "sqrt":
        shift = abs(df[col].min()) if df[col].min() < 0 else 0
        df[col] = np.sqrt(df[col] + shift)
    elif method == "rank":
        df[col] = df[col].rank()
    return df

def _pooled_sd(g1, g2):
    return np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))

def _run_test(df, dv, iv, covariates, test_type, variant="default"):
    """Run the statistical test; return dict with stat/p/es/n or None."""
    try:
        if test_type == "t-test":
            groups = df[iv].unique()
            if len(groups) < 2: return None
            g1, g2 = df.loc[df[iv]==groups[0], dv].dropna(), df.loc[df[iv]==groups[1], dv].dropna()
            if len(g1) < 2 or len(g2) < 2: return None
            if variant == "mann_whitney":
                stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                es = 1 - (2*stat)/(len(g1)*len(g2))
            else:
                eq = variant != "welch"
                stat, p = sp_stats.ttest_ind(g1, g2, equal_var=eq)
                ps = _pooled_sd(g1, g2)
                es = (g1.mean()-g2.mean())/ps if ps > 0 else 0
            return {"stat": stat, "p": p, "es": es, "n": len(g1)+len(g2)}
        elif test_type == "correlation":
            common = df[[iv, dv]].dropna()
            if len(common) < 3: return None
            r, p = sp_stats.pearsonr(common[iv], common[dv])
            return {"stat": r, "p": p, "es": r, "n": len(common)}
        elif test_type == "regression":
            cols = [iv] + (covariates or [])
            sub = df[[dv]+cols].dropna()
            if len(sub) < len(cols)+2: return None
            X = sm.add_constant(sub[cols]); model = sm.OLS(sub[dv], X).fit()
            beta = model.params.get(iv, model.params.iloc[1])
            p = model.pvalues.get(iv, model.pvalues.iloc[1])
            es = beta*(sub[iv].std()/sub[dv].std()) if sub[dv].std() > 0 else 0
            return {"stat": float(model.fvalue), "p": float(p), "es": float(es), "n": len(sub)}
        elif test_type == "anova":
            groups = [g[dv].dropna().values for _, g in df.groupby(iv)]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2: return None
            stat, p = sp_stats.f_oneway(*groups)
            grand = np.concatenate(groups)
            ssb = sum(len(g)*(g.mean()-grand.mean())**2 for g in groups)
            sst = np.sum((grand-grand.mean())**2)
            return {"stat": stat, "p": p, "es": ssb/sst if sst > 0 else 0, "n": len(grand)}
    except Exception:
        return None
    return None

# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/run")
async def run_multiverse(req: MultiverseRequest):
    """Run every specification and return the specification curve."""
    if req.dv not in req.data:
        raise HTTPException(400, f"DV '{req.dv}' not found in data columns")
    if req.iv not in req.data:
        raise HTTPException(400, f"IV '{req.iv}' not found in data columns")
    df_base = pd.DataFrame(req.data)
    decision_names = [d.name for d in req.decisions]
    combos = list(product(*[d.options for d in req.decisions]))
    if len(combos) > 5000:
        raise HTTPException(400, f"Too many specifications ({len(combos)}). Keep under 5 000.")
    results: List[Dict[str, Any]] = []
    for spec_id, combo in enumerate(combos, 1):
        dmap = dict(zip(decision_names, combo))
        df, variant = df_base.copy(), "default"
        for name, opt in dmap.items():
            val = opt.get("method", opt.get("value", "none"))
            if name == "outlier_handling": df = _apply_outlier(df, req.dv, val)
            elif name == "missing_data": df = _apply_missing(df, req.dv, val)
            elif name == "transform": df = _apply_transform(df, req.dv, val)
            elif name == "subgroup":
                if val != "all" and "group_col" in opt and "group_val" in opt:
                    gcol = opt["group_col"]
                    if gcol in df.columns: df = df[df[gcol] == opt["group_val"]].copy()
            elif name == "test_variant": variant = val
        # Covariates for this spec
        cd = dmap.get("covariates")
        covs = cd.get("include") if cd else (req.covariates or None)
        if covs is not None and len(covs) == 0: covs = None
        out = _run_test(df, req.dv, req.iv, covs, req.test_type, variant)
        if out is None: continue
        dstr = {n: o.get("method", o.get("value", str(o))) for n, o in dmap.items()}
        results.append({"spec_id": spec_id, "decisions": dstr, "effect_size": _safe(out["es"]),
                        "p_value": _safe(out["p"]), "significant": bool(out["p"] < .05),
                        "n": int(out["n"]), "test_statistic": _safe(out["stat"])})
    if not results:
        raise HTTPException(422, "No valid specifications produced results.")
    results.sort(key=lambda r: (r["effect_size"] is None, r["effect_size"]))
    sig_count = sum(1 for r in results if r["significant"])
    es_vals = [r["effect_size"] for r in results if r["effect_size"] is not None]
    # Decision influence
    influence: List[Dict[str, Any]] = []
    for dn in decision_names:
        og: Dict[str, List[Dict]] = {}
        for r in results: og.setdefault(r["decisions"].get(dn, "?"), []).append(r)
        for ov, specs in og.items():
            ev = [s["effect_size"] for s in specs if s["effect_size"] is not None]
            sv = [s["significant"] for s in specs]
            influence.append({"decision_name": dn, "option": ov,
                              "mean_effect_size": _safe(np.mean(ev)) if ev else None,
                              "pct_significant": round(sum(sv)/len(sv)*100, 1) if sv else 0})
    spct = round(sig_count/len(results)*100, 1)
    verdict = "Highly robust" if spct >= 90 else ("Moderately robust" if spct >= 60 else "Fragile")
    return {
        "total_specifications": len(results), "significant_count": sig_count,
        "significant_percentage": spct,
        "median_effect_size": _safe(float(np.median(es_vals))) if es_vals else None,
        "effect_size_range": [_safe(min(es_vals)), _safe(max(es_vals))] if es_vals else [None, None],
        "specification_curve": results, "decision_influence": influence,
        "robustness_verdict": verdict,
    }

@router.post("/summary")
async def multiverse_summary(req: MultiverseSummaryRequest):
    """Return a plain-English + APA-formatted summary of multiverse results."""
    total, sig, pct = req.total_specifications, req.significant_count, req.significant_percentage
    med_es = req.median_effect_size
    es_lo, es_hi = (req.effect_size_range + [None, None])[:2]
    verdict = req.robustness_verdict
    parts = [
        f"A multiverse analysis was conducted across {total} unique analytical specifications.",
        f"Of these, {sig} ({pct}%) produced statistically significant results (p < .05).",
    ]
    if med_es is not None:
        parts.append(f"The median effect size was {med_es:.3f} (range: {es_lo:.3f} to {es_hi:.3f}).")
    parts.append(f"Overall robustness: {verdict}.")
    # Most influential decision
    if req.decision_influence:
        by_d: Dict[str, List[float]] = {}
        for item in req.decision_influence:
            mes = item.get("mean_effect_size")
            if mes is not None: by_d.setdefault(item["decision_name"], []).append(mes)
        d_ranges = {d: max(v)-min(v) for d, v in by_d.items()}
        if d_ranges:
            top = max(d_ranges, key=d_ranges.get)
            parts.append(f"The most influential decision was '{top}', shifting mean effect size "
                         f"by {d_ranges[top]:.3f} across its options.")
    narrative = " ".join(parts)
    apa = (f"A specification curve analysis examined the robustness of the relationship "
           f"between {req.iv} and {req.dv} across {total} specifications varying analyst decisions. "
           f"{sig} specifications ({pct}%) yielded significant results (p < .05). ")
    if med_es is not None:
        apa += f"The median effect size was {med_es:.2f} (range = {es_lo:.2f}\u2013{es_hi:.2f}). "
    apa += f"The finding was classified as \u201c{verdict.lower()}\u201d."
    return {"narrative": narrative, "apa_text": apa, "robustness_verdict": verdict}
