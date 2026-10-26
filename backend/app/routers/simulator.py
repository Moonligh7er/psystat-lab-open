"""Data Simulation Engine — generate synthetic datasets for teaching and
power validation.  Supports between, within, factorial, correlational, and
regression designs with configurable effect sizes, missingness, and
Monte Carlo power simulation.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats

router = APIRouter()

# ── Models ───────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    n: int
    design: str  # "between", "within", "factorial", "correlational", "regression"
    params: Dict[str, Any]
    seed: Optional[int] = None
    missing_rate: float = 0.0
    missing_mechanism: str = "mcar"  # "mcar", "mar", "mnar"

class PowerSimRequest(BaseModel):
    design: str
    params: Dict[str, Any]
    test_type: str  # "t-test", "anova", "correlation", "regression"
    n_simulations: int = 1000
    alpha: float = 0.05
    sample_sizes: List[int]

# ── Helpers ──────────────────────────────────────────────────────────────

def _safe(v) -> Optional[float]:
    v = float(v)
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)

def _desc(arr: np.ndarray) -> Dict[str, Optional[float]]:
    c = arr[~np.isnan(arr)] if np.any(np.isnan(arr)) else arr
    if len(c) == 0:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {"mean": _safe(np.mean(c)), "sd": _safe(np.std(c, ddof=1)),
            "min": _safe(np.min(c)), "max": _safe(np.max(c))}

def _inject_missing(rng, data: Dict[str, np.ndarray], rate: float, mechanism: str) -> int:
    if rate <= 0:
        return 0
    total = 0
    cols = list(data.keys())
    for col in cols:
        arr = data[col].astype(float)
        n = len(arr)
        nm = int(round(n * rate))
        if nm == 0:
            continue
        if mechanism == "mar":
            other = [c for c in cols if c != col]
            if other:
                ref = np.nan_to_num(data[other[0]].astype(float), nan=0.0)
                pr = (ref - ref.min()) + 1e-6; pr /= pr.sum()
                idx = rng.choice(n, min(nm, n), replace=False, p=pr)
            else:
                idx = rng.choice(n, min(nm, n), replace=False)
        elif mechanism == "mnar":
            pr = (arr - np.nanmin(arr)) + 1e-6; pr /= pr.sum()
            idx = rng.choice(n, min(nm, n), replace=False, p=pr)
        else:  # mcar
            idx = rng.choice(n, min(nm, n), replace=False)
        arr[idx] = np.nan
        data[col] = arr
        total += len(idx)
    return total

# ── Generators ───────────────────────────────────────────────────────────

def _gen_between(rng, n, p):
    d, mu, sd = float(p.get("effect_size", .5)), float(p.get("dv_mean", 50)), float(p.get("dv_sd", 10))
    dv, gn = p.get("dv_name", "score"), p.get("group_name", "group")
    gs = p.get("group_sizes")
    n1, n2 = (int(gs[0]), int(gs[1])) if gs and len(gs) == 2 else (n // 2, n - n // 2)
    g1, g2 = rng.normal(mu, sd, n1), rng.normal(mu + d * sd, sd, n2)
    data = {dv: np.concatenate([g1, g2]), gn: np.array([0]*n1 + [1]*n2, dtype=float)}
    gt = {"true_effect_size": d, "true_parameters": {"mean_diff": d * sd},
          "population_means": {"group_0": mu, "group_1": mu + d * sd}}
    return data, gt, [dv, gn]

def _gen_within(rng, n, p):
    d, ntp = float(p.get("effect_size", .5)), int(p.get("n_timepoints", 2))
    corr, dv = float(p.get("correlation", .5)), p.get("dv_name", "score")
    mu, sd = float(p.get("dv_mean", 50)), float(p.get("dv_sd", 10))
    cov = np.full((ntp, ntp), corr * sd**2); np.fill_diagonal(cov, sd**2)
    means = [mu + d * sd * t / max(ntp - 1, 1) for t in range(ntp)]
    raw = rng.multivariate_normal(means, cov, n)
    data, vnames = {}, []
    for t in range(ntp):
        c = f"{dv}_t{t+1}"; data[c] = raw[:, t]; vnames.append(c)
    gt = {"true_effect_size": d, "true_parameters": {"correlation": corr, "n_timepoints": ntp},
          "population_means": {f"t{t+1}": means[t] for t in range(ntp)}}
    return data, gt, vnames

def _gen_factorial(rng, n, p):
    factors = p.get("factors", [{"name": "A", "levels": 2}, {"name": "B", "levels": 2}])
    me = p.get("main_effects", [.5]*len(factors))
    ix = float(p.get("interaction_effect", 0)); dv = p.get("dv_name", "score")
    mu, sd = float(p.get("dv_mean", 50)), float(p.get("dv_sd", 10))
    lvls = [list(range(f["levels"])) for f in factors]
    cells = list(np.array(np.meshgrid(*lvls)).T.reshape(-1, len(factors)))
    npc = max(n // len(cells), 2)
    all_dv, fd = [], {f["name"]: [] for f in factors}
    for cell in cells:
        cm = mu + sum(me[i]*sd*(cell[i]-(factors[i]["levels"]-1)/2) for i in range(len(cell)))
        if len(cell) >= 2:
            cm += ix * sd * np.prod([(cell[i]-(factors[i]["levels"]-1)/2) for i in range(len(cell))])
        all_dv.extend(rng.normal(cm, sd, npc))
        for i, f in enumerate(factors):
            fd[f["name"]].extend([float(cell[i])]*npc)
    data = {dv: np.array(all_dv)}; vnames = [dv]
    for fn, fv in fd.items():
        data[fn] = np.array(fv); vnames.append(fn)
    gt = {"true_effect_size": me[0] if me else 0,
          "true_parameters": {"main_effects": {factors[i]["name"]: me[i] for i in range(len(factors))},
                              "interaction_effect": ix},
          "population_means": {"grand_mean": mu}}
    return data, gt, vnames

def _gen_correlational(rng, n, p):
    variables = p.get("variables", [{"name": "x", "mean": 0, "sd": 1},
                                     {"name": "y", "mean": 0, "sd": 1}])
    k = len(variables)
    R = np.array(p.get("correlation_matrix", np.eye(k).tolist()), dtype=float)
    means, sds = [v["mean"] for v in variables], [v["sd"] for v in variables]
    D = np.diag(sds); cov = D @ R @ D
    try:
        raw = rng.multivariate_normal(means, cov, n)
    except np.linalg.LinAlgError:
        cov += np.eye(k)*1e-6; raw = rng.multivariate_normal(means, cov, n)
    data, vn = {}, []
    for i, v in enumerate(variables):
        data[v["name"]] = raw[:, i]; vn.append(v["name"])
    gt = {"true_effect_size": float(R[0, 1]) if k >= 2 else 0,
          "true_parameters": {"correlation_matrix": R.tolist()},
          "population_means": {v["name"]: v["mean"] for v in variables}}
    return data, gt, vn

def _gen_regression(rng, n, p):
    preds = p.get("predictors", [{"name": "x", "mean": 0, "sd": 1, "beta": .5}])
    dv = p.get("dv_name", "y"); r2 = float(p.get("r_squared", .25))
    b0 = float(p.get("intercept", 0))
    data, vn, y = {}, [], np.full(n, b0, dtype=float)
    for pr in preds:
        x = rng.normal(pr["mean"], pr["sd"], n); data[pr["name"]] = x
        vn.append(pr["name"]); y += pr["beta"] * x
    sv = np.var(y - b0) if np.var(y - b0) > 0 else 1
    nv = sv * (1 - r2) / r2 if 0 < r2 < 1 else 0
    y += rng.normal(0, np.sqrt(max(nv, 0)), n)
    data[dv] = y; vn.append(dv)
    gt = {"true_effect_size": r2,
          "true_parameters": {"intercept": b0, "betas": {pr["name"]: pr["beta"] for pr in preds}, "r_squared": r2},
          "population_means": {pr["name"]: pr["mean"] for pr in preds}}
    return data, gt, vn

_GEN = {"between": _gen_between, "within": _gen_within, "factorial": _gen_factorial,
        "correlational": _gen_correlational, "regression": _gen_regression}

# ── Power test runner ────────────────────────────────────────────────────

def _p_value(data, test_type, params):
    """Return p-value for a single simulated dataset."""
    cols = list(data.keys())
    try:
        if test_type == "t-test":
            gn, dn = params.get("group_name", "group"), params.get("dv_name", "score")
            if gn not in data or dn not in data:
                dn, gn = cols[0], cols[1] if len(cols) >= 2 else (None, None)
            if gn is None: return None
            g = data[gn]; dv = data[dn]
            a, b = dv[g == 0], dv[g == 1]
            return float(sp_stats.ttest_ind(a, b).pvalue) if len(a) >= 2 and len(b) >= 2 else None
        if test_type == "anova":
            dn = params.get("dv_name", "score")
            fns = [f["name"] for f in params.get("factors", [])] or [c for c in cols if c != dn]
            if not fns or dn not in data: return None
            gc = data[fns[0]]; dv = data[dn]
            gs = [dv[gc == lv] for lv in np.unique(gc)]
            gs = [g for g in gs if len(g) >= 2]
            return float(sp_stats.f_oneway(*gs).pvalue) if len(gs) >= 2 else None
        if test_type == "repeated":
            # One-way repeated-measures ANOVA over the time-point columns (sphericity assumed);
            # with two time points this equals the paired t-test (F = t²).
            mat = np.column_stack([data[c] for c in cols])
            mat = mat[~np.isnan(mat).any(axis=1)]
            n_s, k = mat.shape
            if n_s < 3 or k < 2:
                return None
            grand = mat.mean()
            ss_time = n_s * np.sum((mat.mean(axis=0) - grand) ** 2)
            ss_subj = k * np.sum((mat.mean(axis=1) - grand) ** 2)
            ss_err = np.sum((mat - grand) ** 2) - ss_time - ss_subj
            df1, df2 = k - 1, (n_s - 1) * (k - 1)
            if ss_err <= 0:
                return None
            return float(sp_stats.f.sf((ss_time / df1) / (ss_err / df2), df1, df2))
        if test_type == "correlation":
            vn = [v["name"] for v in params.get("variables", [])]
            x, y = (data[vn[0]], data[vn[1]]) if len(vn) >= 2 else (data[cols[0]], data[cols[1]])
            return float(sp_stats.pearsonr(x, y).pvalue)
        if test_type == "regression":
            dn = params.get("dv_name", "y"); pn = [pr["name"] for pr in params.get("predictors", [])]
            if dn not in data: return None
            y = data[dn]; xs = np.column_stack([data[c] for c in pn if c in data])
            if xs.size == 0: return None
            xc = np.column_stack([np.ones(len(y)), xs])
            beta = np.linalg.lstsq(xc, y, rcond=None)[0]
            yh = xc @ beta; ssr = np.sum((yh - y.mean())**2); sse = np.sum((y - yh)**2)
            dfr, dfe = xs.shape[1], len(y) - xs.shape[1] - 1
            if dfe <= 0 or sse <= 0: return None
            return float(sp_stats.f.sf((ssr/dfr)/(sse/dfe), dfr, dfe))
    except Exception:
        return None
    return None

# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/generate")
async def generate_data(req: SimulationRequest):
    """Generate a synthetic dataset for the requested design."""
    if req.design not in _GEN:
        raise HTTPException(400, f"Unknown design '{req.design}'. Supported: {list(_GEN.keys())}")
    if req.n < 4:
        raise HTTPException(400, "Sample size must be at least 4.")
    seed = req.seed if req.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)
    data, ground_truth, variables = _GEN[req.design](rng, req.n, req.params)
    missing_count = _inject_missing(rng, data, req.missing_rate, req.missing_mechanism)
    out_data: Dict[str, list] = {}
    desc: Dict[str, Any] = {}
    for col, arr in data.items():
        out_data[col] = [None if (isinstance(v, float) and math.isnan(v)) else _safe(v) for v in arr]
        desc[col] = _desc(arr)
    return {"data": out_data, "n": req.n, "design": req.design, "ground_truth": ground_truth,
            "missing_count": missing_count, "seed": seed, "variables": variables, "descriptives": desc}

@router.post("/power-simulation")
async def power_simulation(req: PowerSimRequest):
    """Monte Carlo power simulation across multiple sample sizes."""
    if req.design not in _GEN:
        raise HTTPException(400, f"Unknown design '{req.design}'. Supported: {list(_GEN.keys())}")
    if req.n_simulations < 10 or req.n_simulations > 50000:
        raise HTTPException(400, "n_simulations must be between 10 and 50 000.")
    if not req.sample_sizes:
        raise HTTPException(400, "sample_sizes must contain at least one value.")
    rng = np.random.default_rng(42)
    power_curve: List[Dict[str, Any]] = []
    recommended_n: Optional[int] = None
    true_es: Optional[float] = None
    for n in sorted(req.sample_sizes):
        if n < 4:
            continue
        sig = 0
        for _ in range(req.n_simulations):
            s = int(rng.integers(0, 2**31))
            data, gt, _ = _GEN[req.design](np.random.default_rng(s), n, req.params)
            if true_es is None:
                true_es = gt.get("true_effect_size", 0)
            pv = _p_value(data, req.test_type, req.params)
            if pv is not None and pv < req.alpha:
                sig += 1
        pw = sig / req.n_simulations
        se = math.sqrt(pw * (1 - pw) / req.n_simulations)
        power_curve.append({"n": n, "power": round(pw, 4), "se": round(se, 4)})
        if recommended_n is None and pw >= 0.80:
            recommended_n = n
    return {"power_curve": power_curve, "recommended_n": recommended_n,
            "n_simulations": req.n_simulations,
            "true_effect_size": _safe(true_es) if true_es is not None else None}
