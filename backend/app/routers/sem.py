"""Structural Equation Modeling — CFA, path analysis, full SEM, measurement invariance
and latent growth, all fitted by maximum likelihood with app/sem_engine.py (lavaan's
defaults; reference values in tests/test_sem.py)."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.sem_engine import SEM, Model, cfa_params, growth_params, structural_params

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class CFARequest(BaseModel):
    """Confirmatory Factor Analysis."""
    data: list[dict]  # rows as dicts
    factors: dict[str, list[str]]  # factor_name -> [indicator_names]
    residual_covariances: list[list[str]] | None = None  # e.g. [["x1", "x4"]]


class PathAnalysisRequest(BaseModel):
    """Path Analysis — directed paths between observed variables."""
    data: list[dict]
    paths: list[dict]  # [{"from": "x1", "to": "y"}, ...]
    mediators: list[str] | None = None
    residual_covariances: list[list[str]] | None = None


class SEMRequest(BaseModel):
    """Full SEM: measurement model + structural model."""
    data: list[dict]
    measurement: dict[str, list[str]]  # latent -> [indicators]
    structural: list[dict]  # [{"from": "latent1", "to": "latent2"}, ...]
    residual_covariances: list[list[str]] | None = None


class MeasurementInvarianceRequest(BaseModel):
    data_group1: list[list[float]]
    data_group2: list[list[float]]
    n_factors: int = 1
    factor_model: dict[str, list[int]] | None = None  # factor -> item column indices (0-based)


class LatentGrowthRequest(BaseModel):
    data: list[list[float]]  # persons x time points
    time_points: list[float]


# ── Shared helpers ────────────────────────────────────────────────────

ESTIMATION_NOTE = ("Maximum likelihood (lavaan defaults: N in the denominator, expected-information standard "
                   "errors, marker-variable identification). Rows with missing values are dropped (listwise).")


def _frame(rows: list[dict], columns: list[str]) -> tuple[np.ndarray, int]:
    if not rows:
        raise HTTPException(422, "No data supplied.")
    df = pd.DataFrame(rows)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Variable(s) not found in the data: {', '.join(missing)}")
    sub = df[columns].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in columns if sub[c].notna().sum() < df[c].notna().sum()]
    if bad:
        raise HTTPException(422, f"Variable(s) must be numeric: {', '.join(bad)}")
    clean = sub.dropna()
    return clean.values.astype(float), len(df) - len(clean)


def _check_cov(X: np.ndarray, names: list[str]):
    sd = X.std(axis=0)
    flat = [names[i] for i in np.where(sd == 0)[0]]
    if flat:
        raise HTTPException(422, f"Variable(s) with no variance: {', '.join(flat)}")
    if np.linalg.eigvalsh(np.cov(X, rowvar=False)).min() < 1e-10:
        raise HTTPException(422, "The covariance matrix is singular (a variable is a linear combination of others).")


def _fit_dict(fm: dict) -> dict:
    ci = fm.get("rmsea_ci90") or [None, None]
    return {
        "chi_square": fm["chi_square"], "df": fm["df"], "p_value": fm["p_value"],
        "cfi": fm["cfi"], "tli": fm["tli"], "rmsea": fm["rmsea"],
        "rmsea_ci90_lower": ci[0], "rmsea_ci90_upper": ci[1], "srmr": fm["srmr"],
        "aic": fm["aic"], "bic": fm["bic"],
    }


def _interpret(fm: dict) -> dict:
    """Plain-language reading of the fit indices against common conventions (Hu & Bentler, 1999).
    These are rules of thumb, not tests."""
    out = {}
    if fm["df"] == 0:
        out["model"] = "Saturated (df = 0): the model reproduces the data exactly, so fit cannot be assessed."
        return out
    out["chi_square"] = (f"chi2({fm['df']:.0f}) = {fm['chi_square']:.2f}, p = {fm['p_value']:.3g} "
                         + ("(exact fit not rejected)" if fm["p_value"] >= .05 else "(exact fit rejected; common with large N)"))
    out["cfi"] = f"{fm['cfi']:.3f} ({'good' if fm['cfi'] >= .95 else 'acceptable' if fm['cfi'] >= .90 else 'poor'}; >= .95 good, >= .90 acceptable)"
    if fm["tli"] is not None:
        out["tli"] = f"{fm['tli']:.3f} ({'good' if fm['tli'] >= .95 else 'acceptable' if fm['tli'] >= .90 else 'poor'})"
    if fm["rmsea"] is not None:
        lo, hi = fm["rmsea_ci90"]
        out["rmsea"] = (f"{fm['rmsea']:.3f}, 90% CI [{lo:.3f}, {hi:.3f}] "
                        f"({'good' if fm['rmsea'] <= .06 else 'acceptable' if fm['rmsea'] <= .08 else 'poor'}; <= .06 good, <= .08 acceptable)")
    out["srmr"] = f"{fm['srmr']:.3f} ({'good' if fm['srmr'] <= .08 else 'poor'}; <= .08 good)"
    out["note"] = "Cut-offs are conventions (Hu & Bentler, 1999), not significance tests."
    return out


def _run(model: Model, data: list[np.ndarray], n_fixed_x=0, exo=()):
    try:
        sem = SEM(model, data)
        fitted = sem.fit()
    except np.linalg.LinAlgError:
        raise HTTPException(422, "The model could not be estimated (singular matrices). Check the specification.")
    fm = sem.fit_measures(fitted, n_fixed_x, exo)
    if fm["df"] < 0:
        raise HTTPException(422, f"The model is not identified: it has {fm['n_parameters']} free parameters but the data "
                                 f"provide only {int(sem.n_moments() - n_fixed_x)} variances/covariances (df = {fm['df']:.0f}).")
    return sem, fitted, fm


def _warnings(sem, fitted, est_rows) -> list[str]:
    w = []
    if not fitted.converged:
        w.append("The optimiser did not fully converge; estimates may be unreliable.")
    neg = [f"{r['lhs']}" for r in est_rows if r["op"] == "~~" and r["lhs"] == r["rhs"] and r["est"] < 0]
    if neg:
        w.append(f"Negative variance estimate(s) for {', '.join(neg)} (Heywood case): the solution is improper.")
    if np.isnan(fitted.se).any():
        w.append("Standard errors could not be computed (the information matrix is singular): the model may not be identified.")
    return w


def _row(r):
    return {"estimate": r["est"], "se": r["se"], "z": r["z"], "p_value": r["p_value"], "standardized": r["std_all"]}


# ── CFA ───────────────────────────────────────────────────────────────

def _validate_factors(factors: dict[str, list[str]]):
    if not factors:
        raise HTTPException(422, "Specify at least one factor.")
    seen = set()
    for f, inds in factors.items():
        if len(inds) < 2:
            raise HTTPException(422, f"Factor '{f}' needs at least 2 indicators.")
        dup = seen.intersection(inds)
        if dup:
            raise HTTPException(422, f"Indicator(s) {', '.join(sorted(dup))} load on more than one factor; "
                                     "cross-loadings are not supported in this form.")
        seen.update(inds)
    if len(factors) == 1 and len(next(iter(factors.values()))) < 3:
        raise HTTPException(422, "A single factor needs at least 3 indicators to be identified.")


@router.post("/cfa")
def confirmatory_factor_analysis(req: CFARequest):
    """Confirmatory factor analysis by maximum likelihood."""
    _validate_factors(req.factors)
    obs = [i for inds in req.factors.values() for i in inds]
    X, dropped = _frame(req.data, obs)
    if len(X) < len(obs) + 2:
        raise HTTPException(422, f"Need more complete observations ({len(X)}) than indicators ({len(obs)}).")
    _check_cov(X, obs)
    S = np.cov(X, rowvar=False, bias=True)
    params = cfa_params(req.factors, 1, S_list=[S], observed=obs,
                        residual_covariances=[tuple(p) for p in (req.residual_covariances or [])])
    sem, fitted, fm = _run(Model(obs, list(req.factors), params), [X])
    est = sem.estimates(fitted)
    r2 = sem.r_squared(fitted)
    loadings = [{"factor": r["lhs"], "indicator": r["rhs"], "marker": not r["free"], **_row(r)}
                for r in est if r["op"] == "=~"]
    lat = list(req.factors)
    corrs = [{"factor_1": r["lhs"], "factor_2": r["rhs"], "covariance": r["est"], "se": r["se"],
              "p_value": r["p_value"], "correlation": r["std_all"]}
             for r in est if r["op"] == "~~" and r["lhs"] in lat and r["rhs"] in lat and r["lhs"] != r["rhs"]]
    return {
        "n_observations": len(X),
        "n_dropped_missing": dropped,
        "n_factors": len(lat),
        "factor_names": lat,
        "estimation": ESTIMATION_NOTE,
        "fit_indices": _fit_dict(fm),
        "interpretation": _interpret(fm),
        "factor_loadings": {f: {l["indicator"]: l["standardized"] for l in loadings if l["factor"] == f} for f in lat},
        "loadings": loadings,
        "factor_correlations": corrs,
        "residual_variances": {r["lhs"]: _row(r) for r in est if r["op"] == "~~" and r["lhs"] == r["rhs"] and r["lhs"] in obs},
        "factor_variances": {r["lhs"]: _row(r) for r in est if r["op"] == "~~" and r["lhs"] == r["rhs"] and r["lhs"] in lat},
        "r_squared": {k: v for k, v in r2.items() if k in obs},
        "loglik": fm["loglik"],
        "n_parameters": fm["n_parameters"],
        "converged": fitted.converged,
        "warnings": _warnings(sem, fitted, est),
        "note": "Loadings in factor_loadings are standardized (std.all); the first indicator of each factor is the marker (unstandardized loading fixed to 1).",
    }


# ── Path analysis ─────────────────────────────────────────────────────

def _check_acyclic(edges: list[tuple[str, str]]):
    graph: dict[str, set] = {}
    for y, x in edges:
        graph.setdefault(x, set()).add(y)
    state: dict[str, int] = {}

    def visit(v):
        state[v] = 1
        for w in graph.get(v, ()):
            if state.get(w) == 1 or (state.get(w) is None and visit(w)):
                return True
        state[v] = 2
        return False
    for v in list(graph):
        if state.get(v) is None and visit(v):
            raise HTTPException(422, "The paths contain a feedback loop; only recursive (acyclic) models are supported.")


def _effects(sem, fitted, variables: list[str], mediators: list[str] | None):
    """Indirect effects through each listed mediator and total effects, with delta-method SEs."""
    idx = sem.idx

    def total(i, j):  # total effect of variable j on i: [(I - A)^-1 - I][i, j]
        def fn(s, th):
            A, _, _ = s.matrices(th, 0)
            B = np.linalg.inv(np.eye(s.k) - A)
            return B[i, j] - (1.0 if i == j else 0.0)
        return fn

    A, _, _ = sem.matrices(fitted.theta, 0)
    indirect = []
    for m in mediators or []:
        mi = idx[m]
        for x in variables:
            for y in variables:
                xi, yi = idx[x], idx[y]
                if x == y or x == m or y == m or A[mi, xi] == 0 or A[yi, mi] == 0:
                    continue
                d = sem.defined(fitted, lambda s, th, mi=mi, xi=xi, yi=yi: (
                    s.matrices(th, 0)[0][yi, mi] * s.matrices(th, 0)[0][mi, xi]))
                indirect.append({"from": x, "mediator": m, "to": y, "indirect_effect": d["est"], "se": d["se"],
                                 "z": d["z"], "p_value": d["p_value"], "sobel_z": d["z"], "sobel_p": d["p_value"],
                                 "significant": d["p_value"] is not None and d["p_value"] < .05})
    B = np.linalg.inv(np.eye(sem.k) - A)
    totals = []
    for x in variables:
        for y in variables:
            xi, yi = idx[x], idx[y]
            if x != y and abs(B[yi, xi]) > 1e-12:
                d = sem.defined(fitted, total(yi, xi))
                totals.append({"from": x, "to": y, "total_effect": d["est"], "se": d["se"], "p_value": d["p_value"],
                               "direct_effect": float(A[yi, xi])})
    return indirect, totals


@router.post("/path-analysis")
def path_analysis(req: PathAnalysisRequest):
    """Path analysis among observed variables by maximum likelihood (all equations estimated jointly)."""
    if not req.paths:
        raise HTTPException(422, "Specify at least one path.")
    edges = []
    for pth in req.paths:
        if "from" not in pth or "to" not in pth or pth["from"] == pth["to"]:
            raise HTTPException(422, "Each path needs distinct 'from' and 'to' variables.")
        edges.append((str(pth["to"]), str(pth["from"])))
    if len(set(edges)) != len(edges):
        raise HTTPException(422, "A path is listed twice.")
    _check_acyclic(edges)
    obs = list(dict.fromkeys([x for _, x in edges] + [y for y, _ in edges]))
    for m in req.mediators or []:
        if m not in obs:
            raise HTTPException(422, f"Mediator '{m}' is not part of any path.")
    X, dropped = _frame(req.data, obs)
    if len(X) < len(obs) + 2:
        raise HTTPException(422, "Too few complete observations for this many variables.")
    _check_cov(X, obs)
    S = np.cov(X, rowvar=False, bias=True)
    params, nfx, exo = structural_params(edges, obs, None, S, [tuple(p) for p in (req.residual_covariances or [])])
    sem, fitted, fm = _run(Model(obs, [], params), [X], nfx, exo)
    est = sem.estimates(fitted)
    direct = [{"from": r["rhs"], "to": r["lhs"], "coefficient": r["est"], "std_error": r["se"], "z": r["z"],
               "p_value": r["p_value"], "significant": r["p_value"] is not None and r["p_value"] < .05,
               "standardized": r["std_all"]} for r in est if r["op"] == "~"]
    indirect, totals = _effects(sem, fitted, obs, req.mediators)
    return {
        "n_observations": len(X),
        "n_dropped_missing": dropped,
        "n_paths": len(edges),
        "estimation": ESTIMATION_NOTE + " Exogenous variables are treated as fixed (fixed.x).",
        "fit_indices": _fit_dict(fm),
        "interpretation": _interpret(fm),
        "direct_effects": direct,
        "indirect_effects": indirect,
        "total_effects": totals,
        "r_squared": sem.r_squared(fitted),
        "residual_covariances": [{"var_1": r["lhs"], "var_2": r["rhs"], **_row(r)} for r in est
                                 if r["op"] == "~~" and r["lhs"] != r["rhs"] and r["free"]],
        "converged": fitted.converged,
        "warnings": _warnings(sem, fitted, est),
        "note": ("Indirect effects are products of path coefficients with delta-method standard errors "
                 "(equivalent to the Sobel test); for small samples a bootstrap CI is more accurate (Mediation module)."),
    }


# ── Full SEM ──────────────────────────────────────────────────────────

@router.post("/sem")
def structural_equation_model(req: SEMRequest):
    """Full SEM: measurement model and structural paths estimated simultaneously by ML."""
    _validate_factors(req.measurement)
    lat = list(req.measurement)
    indicators = [i for inds in req.measurement.values() for i in inds]
    edges = []
    extra_obs = []
    for pth in req.structural:
        if "from" not in pth or "to" not in pth or pth["from"] == pth["to"]:
            raise HTTPException(422, "Each structural path needs distinct 'from' and 'to' variables.")
        for v in (pth["from"], pth["to"]):
            if v not in lat and v not in indicators and v not in extra_obs:
                extra_obs.append(v)
        edges.append((str(pth["to"]), str(pth["from"])))
    if not edges:
        raise HTTPException(422, "Specify at least one structural path.")
    _check_acyclic(edges)
    obs = indicators + extra_obs
    X, dropped = _frame(req.data, obs)
    if len(X) < len(obs) + 2:
        raise HTTPException(422, "Too few complete observations for this many variables.")
    _check_cov(X, obs)
    S = np.cov(X, rowvar=False, bias=True)
    params, nfx, exo = structural_params(edges, obs, req.measurement, S,
                                         [tuple(p) for p in (req.residual_covariances or [])])
    sem, fitted, fm = _run(Model(obs, lat, params), [X], nfx, exo)
    est = sem.estimates(fitted)
    structural = [{"from": r["rhs"], "to": r["lhs"], "coefficient": r["est"], "std_error": r["se"], "z": r["z"],
                   "p_value": r["p_value"], "standardized": r["std_all"],
                   "significant": r["p_value"] is not None and r["p_value"] < .05} for r in est if r["op"] == "~"]
    indirect, totals = _effects(sem, fitted, lat + extra_obs, None)
    return {
        "n_observations": len(X),
        "n_dropped_missing": dropped,
        "estimation": ESTIMATION_NOTE,
        "fit_indices": _fit_dict(fm),
        "interpretation": _interpret(fm),
        "measurement_model": [{"factor": r["lhs"], "indicator": r["rhs"], "marker": not r["free"], **_row(r)}
                              for r in est if r["op"] == "=~"],
        "factor_loadings": {f: {r["rhs"]: r["std_all"] for r in est if r["op"] == "=~" and r["lhs"] == f} for f in lat},
        "structural_paths": structural,
        "total_effects": totals,
        "latent_covariances": [{"var_1": r["lhs"], "var_2": r["rhs"], **_row(r)} for r in est
                               if r["op"] == "~~" and r["lhs"] != r["rhs"] and r["lhs"] in lat],
        "r_squared": sem.r_squared(fitted),
        "converged": fitted.converged,
        "warnings": _warnings(sem, fitted, est),
    }


# ── Measurement invariance (two groups) ───────────────────────────────

LEVELS = [("configural", ()), ("metric", ("loadings",)), ("scalar", ("loadings", "intercepts")),
          ("strict", ("loadings", "intercepts", "residuals"))]


def invariance_sequence(groups: list[np.ndarray], factor_model: dict[str, list[int]]) -> list[dict]:
    """Fit configural, metric, scalar and strict multi-group CFAs (lavaan group.equal)
    and compare each with the previous level (chi-square difference, delta CFI / RMSEA)."""
    p = groups[0].shape[1]
    obs = [f"item{j + 1}" for j in range(p)]
    factors = {f: [obs[j] for j in idx] for f, idx in factor_model.items()}
    used = [o for inds in factors.values() for o in inds]
    keep = [obs.index(o) for o in used]
    data = [g[:, keep] for g in groups]
    Ss = [np.cov(d, rowvar=False, bias=True) for d in data]
    ys = [d.mean(axis=0) for d in data]
    out, prev = [], None
    from scipy import stats as sp_stats
    for level, eq in LEVELS:
        params = cfa_params(factors, len(groups), meanstructure=True, equal=eq, S_list=Ss, ybar_list=ys, observed=used)
        sem, fitted, fm = _run(Model(used, list(factors), params, n_groups=len(groups), meanstructure=True), data)
        row = {"level": level, "name": level.capitalize(), **_fit_dict(fm), "n_parameters": fm["n_parameters"],
               "converged": fitted.converged}
        if prev is not None:
            dchi = fm["chi_square"] - prev["chi_square"]
            ddf = fm["df"] - prev["df"]
            row.update({
                "delta_chi_square": dchi, "delta_df": ddf,
                "delta_p_value": float(sp_stats.chi2.sf(max(dchi, 0), ddf)) if ddf > 0 else None,
                "delta_cfi": fm["cfi"] - prev["cfi"],
                "delta_rmsea": fm["rmsea"] - prev["rmsea"],
            })
            row["passed"] = bool(row["delta_cfi"] > -0.01 and row["delta_rmsea"] < 0.015)
        else:
            row["passed"] = bool(fm["cfi"] >= 0.90)
        if level == "scalar":
            est = sem.estimates(fitted)
            row["latent_mean_differences"] = [
                {"factor": r["lhs"], "group": r["group"], "mean": r["est"], "se": r["se"], "p_value": r["p_value"]}
                for r in est if r["op"] == "~1" and r["lhs"] in factors and r["free"]]
        out.append(row)
        prev = fm
    return out


def _conclusion(models: list[dict]) -> str:
    supported = "none"
    for m in models:
        if not m["passed"]:
            break
        supported = m["level"]
    text = {
        "none": "The configural model fits poorly (CFI < .90), so even the same factor structure is doubtful across groups.",
        "configural": "Configural invariance holds (same structure), but loadings differ: factor relations and means are not comparable.",
        "metric": "Metric invariance holds: loadings are equal, so relations with factors can be compared, but intercepts differ, so latent means cannot.",
        "scalar": "Scalar invariance holds: loadings and intercepts are equal, so latent means can be compared across groups.",
        "strict": "Strict invariance holds: loadings, intercepts and residual variances are equal, so observed scores are comparable across groups.",
    }[supported]
    return text + " (Decisions use Chen's 2007 criteria: delta CFI > -.010 and delta RMSEA < .015 relative to the previous level.)"


@router.post("/measurement-invariance")
def measurement_invariance(req: MeasurementInvarianceRequest):
    """Two-group measurement invariance (configural, metric, scalar, strict)."""
    g1, g2 = np.array(req.data_group1, dtype=float), np.array(req.data_group2, dtype=float)
    if g1.ndim != 2 or g2.ndim != 2 or g1.shape[1] != g2.shape[1]:
        raise HTTPException(422, "Both groups need rectangular data with the same items (columns).")
    p = g1.shape[1]
    if req.factor_model is None:
        if req.n_factors != 1:
            raise HTTPException(422, "With more than one factor, say which items belong to which factor "
                                     "(factor_model), or use the Advanced Invariance tool.")
        factor_model = {"F1": list(range(p))}
    else:
        factor_model = req.factor_model
    _validate_index_model(factor_model, p)
    groups = [g[~np.isnan(g).any(axis=1)] for g in (g1, g2)]
    for i, g in enumerate(groups):
        if len(g) < p + 2:
            raise HTTPException(422, f"Group {i + 1} needs more complete observations than items.")
        _check_cov(g, [f"item{j + 1}" for j in range(p)])
    models = invariance_sequence(groups, factor_model)
    return {
        "models": models,
        "conclusion": _conclusion(models),
        "n_group1": len(groups[0]), "n_group2": len(groups[1]),
        "n_indicators": p, "n_factors": len(factor_model),
        "estimation": ESTIMATION_NOTE + " Intercepts are free and latent means fixed at 0 until intercepts are constrained (scalar), "
                                        "after which the second group's latent means are estimated.",
    }


def _validate_index_model(factor_model: dict[str, list[int]], p: int):
    if not factor_model:
        raise HTTPException(422, "Specify at least one factor.")
    seen = set()
    for f, idx in factor_model.items():
        if len(idx) < 2:
            raise HTTPException(422, f"Factor '{f}' needs at least 2 items.")
        if any(i < 0 or i >= p for i in idx):
            raise HTTPException(422, f"Factor '{f}' refers to an item outside 0..{p - 1}.")
        if seen.intersection(idx):
            raise HTTPException(422, "An item is assigned to more than one factor.")
        seen.update(idx)
    if len(factor_model) == 1 and len(next(iter(factor_model.values()))) < 3:
        raise HTTPException(422, "A single factor needs at least 3 items.")


# ── Latent growth ─────────────────────────────────────────────────────

@router.post("/latent-growth")
def latent_growth(req: LatentGrowthRequest):
    """Linear latent growth curve model by ML (lavaan growth())."""
    Y = np.array(req.data, dtype=float)
    t = [float(x) for x in req.time_points]
    if Y.ndim != 2:
        raise HTTPException(422, "Data must be a persons x time points matrix.")
    T = Y.shape[1]
    if T < 3:
        raise HTTPException(422, "A linear growth model needs at least 3 time points.")
    if len(t) != T:
        raise HTTPException(422, "time_points must have one value per column.")
    if len(set(t)) != T:
        raise HTTPException(422, "Time points must be distinct.")
    keep = ~np.isnan(Y).any(axis=1)
    Y = Y[keep]
    if len(Y) < T + 3:
        raise HTTPException(422, "Too few complete cases for a growth model.")
    cols = [f"t{j + 1}" for j in range(T)]
    _check_cov(Y, cols)
    S = np.cov(Y, rowvar=False, bias=True)
    sem, fitted, fm = _run(Model(cols, ["i", "s"], growth_params(cols, t, Y.mean(axis=0), S), meanstructure=True), [Y])
    est = {(r["lhs"], r["op"], r["rhs"]): r for r in sem.estimates(fitted)}

    def part(k):
        r = est[k]
        return {"estimate": r["est"], "se": r["se"], "p_value": r["p_value"]}
    i_mean, s_mean = est[("i", "~1", "")], est[("s", "~1", "")]
    i_var, s_var, cov = est[("i", "~~", "i")], est[("s", "~~", "s")], est[("i", "~~", "s")]
    return {
        "n_persons": len(Y),
        "n_dropped_missing": int((~keep).sum()),
        "time_points": t,
        "estimation": ESTIMATION_NOTE + " Linear growth: intercept loadings 1, slope loadings = time points, "
                                        "observed intercepts 0, residual variances free per time point.",
        "intercept": {"mean": i_mean["est"], "mean_se": i_mean["se"], "mean_p_value": i_mean["p_value"],
                      "variance": i_var["est"], "variance_se": i_var["se"], "variance_p_value": i_var["p_value"]},
        "slope": {"mean": s_mean["est"], "mean_se": s_mean["se"], "p_value": s_mean["p_value"],
                  "variance": s_var["est"], "variance_se": s_var["se"], "variance_p_value": s_var["p_value"]},
        "intercept_slope_covariance": part(("i", "~~", "s")),
        "correlation_intercept_slope": cov["std_all"],
        "residual_variances": {c: part((c, "~~", c)) for c in cols},
        "r_squared": {k: v for k, v in sem.r_squared(fitted).items() if k in cols},
        "fit_indices": _fit_dict(fm),
        "interpretation": _interpret(fm),
        "converged": fitted.converged,
        "warnings": _warnings(sem, fitted, list(est.values())),
    }
