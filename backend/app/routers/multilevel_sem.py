"""Multilevel Structural Equation Modeling — two-level path models, cross-level mediation,
multilevel CFA, within-between decomposition and random-slope regression.

Two-level models are fitted by full maximum likelihood for clustered data, the same model
lavaan fits with sem(..., cluster = ): every variable that varies within clusters is split
into a within-cluster part and a between-cluster (cluster-mean) part, and separate models
are specified for each level (sem_engine.TwoLevelSEM). Complete cases only; exogenous
variables' variances and covariances are estimated (lavaan fixed.x = FALSE). Checked against
lavaan 0.7.3 in tests/test_multilevel_sem.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats

from app.mixed import fit_mixedlm, fixed_effect_se
from app.sem_engine import (Model, Param, TwoLevelSEM, _key, cfa_params, two_level_fit_measures,
                            two_level_reference_models)

router = APIRouter()

LEVELS = ("within", "between")


# ══════════════════════════════════════════════════════════════════════
# Request Models
# ══════════════════════════════════════════════════════════════════════

class TwoLevelSEMRequest(BaseModel):
    """Two-level SEM with within- and between-level path models."""
    data: list[dict]                      # rows as dicts
    group_ids: str                        # column name for cluster ID
    within_model: list[str]               # e.g. ["Y ~ X1 + X2", "M ~ X1"]
    between_model: list[str]              # e.g. ["Y ~ W1"]
    confidence: float = 0.95


class RandomSlopesSEMRequest(BaseModel):
    """Multilevel regression with random slopes (one outcome)."""
    data: list[dict]
    group_ids: str
    outcome: str
    predictors: list[str]
    random_slopes: list[str]             # subset of predictors with random slopes
    level2_predictors: list[str] | None = None
    confidence: float = 0.95


class WithinBetweenDecompRequest(BaseModel):
    """Within-between covariance decomposition."""
    data: list[dict]
    group_ids: str
    variables: list[str]


class CrossLevelMediationRequest(BaseModel):
    """Multilevel mediation (1-1-1, 2-1-1, 1-1-2)."""
    x: list[float]
    m: list[float]
    y: list[float]
    group_ids: list[int | str]
    model_type: str = "1-1-1"           # "1-1-1", "2-1-1", or "1-1-2"
    level2_x: list[float] | None = None  # cluster-level X for 2-1-1, one value per cluster (in order of first appearance)
    n_monte_carlo: int = 20000
    confidence: float = 0.95
    seed: int = 1


class MultilevelCFARequest(BaseModel):
    """Two-level confirmatory factor analysis."""
    data: list[dict]
    group_ids: str
    factor_model: dict[str, list[str]]   # factor_name -> [item_names]
    equal_loadings: bool = False         # constrain loadings equal across levels


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """Parse 'Y ~ X1 + X2' into (outcome, [predictors])."""
    parts = formula.replace(" ", "").split("~")
    if len(parts) != 2 or not parts[0]:
        raise HTTPException(400, f"Invalid model formula: '{formula}'. Use 'Y ~ X1 + X2'.")
    preds = [p for p in parts[1].split("+") if p]
    if not preds:
        raise HTTPException(400, f"No predictors found in formula: '{formula}'.")
    return parts[0], preds


def _frame(data: list[dict], cols: list[str], cluster: str) -> tuple[pd.DataFrame, int]:
    df = pd.DataFrame(data)
    missing = [c for c in cols + [cluster] if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns in data: {missing}")
    df = df[cols + [cluster]]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_before = len(df)
    df = df.dropna()
    return df, n_before - len(df)


def _varies_within(df: pd.DataFrame, v: str, cluster: str) -> bool:
    sd_w = df.groupby(cluster)[v].transform(lambda s: s - s.mean()).abs().max()
    scale = max(float(df[v].std(ddof=0)), 1e-12)
    return bool(sd_w > 1e-10 * scale)


def _check_clusters(df: pd.DataFrame, cluster: str, n_vars: int):
    J = df[cluster].nunique()
    if len(df) < 20:
        raise HTTPException(400, f"Need at least 20 complete observations, got {len(df)}.")
    if J < max(10, n_vars + 2):
        raise HTTPException(400, f"Need at least {max(10, n_vars + 2)} clusters for a two-level model, got {J}.")
    return J


def _level_params(regs: list[tuple[str, str]], level_vars: list[str], g: int,
                  S: np.ndarray, mu: np.ndarray | None) -> tuple[list[Param], list[str]]:
    """lavaan defaults for one level of a path model (fixed.x = FALSE): residual variances of
    endogenous variables, variances and covariances of exogenous variables and residual
    covariances among pure outcomes are free; intercepts free at the between level."""
    ix = {v: i for i, v in enumerate(level_vars)}
    outcomes = {y for y, _ in regs}
    preds = {x for _, x in regs}
    exo = [v for v in level_vars if v not in outcomes]
    P = [Param(y, "~", x, g, value=0.0) for y, x in regs]
    for v in level_vars:
        if v in outcomes:
            P.append(Param(v, "~~", v, g, value=max(0.5 * S[ix[v], ix[v]], 1e-3)))
    pure_y = [v for v in level_vars if v in outcomes and v not in preds]
    for i, a in enumerate(pure_y):
        for b in pure_y[i + 1:]:
            P.append(Param(a, "~~", b, g, value=0.0))
    for i, a in enumerate(exo):
        for b in exo[i:]:
            P.append(Param(a, "~~", b, g, value=float(S[ix[a], ix[b]]) if a != b else max(float(S[ix[a], ix[a]]), 1e-3)))
    if mu is not None:
        for v in level_vars:
            P.append(Param(v, "~1", "", g, value=float(mu[ix[v]])))
    return P, exo


def _fit_two_level(df, cluster, observed, within_vars, build):
    """Fit a two-level model. build(SW, SB, mu) -> (params, latent, exo_within, exo_between)."""
    probe = TwoLevelSEM(Model(observed, [], [], 2, True), df[observed].values, df[cluster].values, within_vars)
    SW, SB, mu = probe.moment_starts()
    params, latent, exo_w, exo_b = build(SW, SB, mu)
    sem = TwoLevelSEM(Model(observed, latent, params, 2, True), df[observed].values, df[cluster].values, within_vars)
    fitted = sem.fit()
    fm = two_level_fit_measures(sem, fitted, exo_w, exo_b)
    return sem, fitted, fm


def _icc_from(fm: dict, sem: TwoLevelSEM) -> dict:
    """ICC per within-varying variable from the unrestricted two-level estimates (lavaan's lavInspect(fit, "icc"))."""
    u = fm["unrestricted"]
    SW, SB = u["sigma_within"], u["sigma_between"]
    out = {}
    for k, v in enumerate(sem.within_vars):
        b = SB[sem.obs.index(v), sem.obs.index(v)]
        out[v] = float(b / (b + SW[k, k])) if b + SW[k, k] > 0 else None
    return out


def _estimates(sem: TwoLevelSEM, fitted, confidence: float) -> dict:
    zc = sp_stats.norm.ppf(0.5 + confidence / 2)
    out = {lvl: [] for lvl in LEVELS}
    for r in sem.estimates(fitted):
        if not r["free"]:
            continue
        lvl = LEVELS[r["group"] - 1]
        row = {k: r[k] for k in ("lhs", "op", "rhs", "est", "se", "z", "p_value", "std_all")}
        row["ci_lower"] = r["est"] - zc * r["se"] if r["se"] is not None else None
        row["ci_upper"] = r["est"] + zc * r["se"] if r["se"] is not None else None
        out[lvl].append(row)
    return out


def _clean_fit(fm: dict) -> dict:
    return {k: v for k, v in fm.items() if k != "unrestricted"}


def _fit_notes(fitted, fm, sem) -> list[str]:
    notes = []
    if not fitted.converged:
        notes.append("The optimizer did not fully converge; treat the estimates with caution.")
    neg = [f"{prm.lhs} ({LEVELS[prm.group]})" for prm in sem.m.params
           if prm.op == "~~" and prm.lhs == prm.rhs and prm.free and fitted.theta[sem.pos[_key(prm)]] < 0]
    if neg:
        notes.append(f"Negative variance estimate(s): {', '.join(neg)}. Often a sign that a between-level variance is near zero.")
    if fm["n_clusters"] < 50:
        notes.append(f"Only {fm['n_clusters']} clusters: between-level estimates and standard errors can be unreliable with fewer than about 50 clusters.")
    return notes


def _mc_ci(sem, fitted, fn, n_draws: int, confidence: float, seed: int) -> tuple[float, float]:
    """Monte Carlo CI for a function of the parameters (draws from N(theta, vcov))."""
    rng = np.random.default_rng(seed)
    try:
        draws = rng.multivariate_normal(fitted.theta, fitted.vcov, size=n_draws, method="eigh")
    except (ValueError, np.linalg.LinAlgError):
        return None, None
    vals = np.array([fn(sem, t) for t in draws])
    a = (1 - confidence) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


# ══════════════════════════════════════════════════════════════════════
# 1. Two-Level SEM
# ══════════════════════════════════════════════════════════════════════

@router.post("/msem-two-level")
def msem_two_level(req: TwoLevelSEMRequest):
    """Two-level path model (lavaan sem(..., cluster = )) with indirect effects per level."""
    regs = {lvl: [] for lvl in LEVELS}
    for lvl, formulas in (("within", req.within_model), ("between", req.between_model)):
        for f in formulas:
            y, xs = _parse_formula(f)
            regs[lvl] += [(y, x) for x in xs]
    if not regs["within"] and not regs["between"]:
        raise HTTPException(400, "Specify at least one regression.")
    var_set = []
    for lvl in LEVELS:
        for y, x in regs[lvl]:
            for v in (y, x):
                if v not in var_set:
                    var_set.append(v)
    df, dropped = _frame(req.data, var_set, req.group_ids)
    J = _check_clusters(df, req.group_ids, len(var_set))

    within_vars = [v for v in var_set if _varies_within(df, v, req.group_ids)]
    between_only = [v for v in var_set if v not in within_vars]
    bad = sorted({v for y, x in regs["within"] for v in (y, x) if v in between_only})
    if bad:
        raise HTTPException(400, f"{', '.join(bad)} do not vary within clusters, so they can only appear in the between-level model.")
    observed = within_vars + between_only
    notes = []
    in_w = {v for y, x in regs["within"] for v in (y, x)}
    in_b = {v for y, x in regs["between"] for v in (y, x)}
    added_w = [v for v in within_vars if v not in in_w]
    added_b = [v for v in observed if v not in in_b]
    if added_w or added_b:
        notes.append(
            "Variables that vary within clusters have a part at both levels. "
            + (f"Not in the within model, so added there as exogenous: {', '.join(added_w)}. " if added_w else "")
            + (f"Not in the between model, so added there as exogenous: {', '.join(added_b)}. " if added_b else "")
            + "(lavaan instead treats a variable mentioned at only one level as belonging to that level only.)"
        )

    exo = {}

    def build(SW, SB, mu):
        pw, ew = _level_params(regs["within"], within_vars, 0, SW, None)
        pb, eb = _level_params(regs["between"], observed, 1, SB, mu)
        exo["within"], exo["between"] = ew, eb
        return pw + pb, [], ew, eb

    sem, fitted, fm = _fit_two_level(df, req.group_ids, observed, within_vars, build)

    indirect = {}
    for g, lvl in enumerate(LEVELS):
        out = []
        r = regs[lvl]
        for (m, x) in r:
            for (y, m2) in r:
                if m2 == m:
                    fn = lambda s, t, x=x, m=m, y=y, g=g: s.value(t, m, "~", x, g) * s.value(t, y, "~", m, g)
                    d = sem.defined(fitted, fn)
                    zc = sp_stats.norm.ppf(0.5 + req.confidence / 2)
                    out.append({"from": x, "through": m, "to": y, **d,
                                "ci_lower": d["est"] - zc * d["se"], "ci_upper": d["est"] + zc * d["se"]})
        indirect[lvl] = out

    return {
        "n_observations": int(fm["n_observations"]),
        "n_groups": J,
        "n_dropped_incomplete": dropped,
        "group_sizes": {"mean": float(np.mean(sem.cluster_sizes)), "min": int(np.min(sem.cluster_sizes)),
                        "max": int(np.max(sem.cluster_sizes))},
        "within_variables": within_vars,
        "between_only_variables": between_only,
        "icc": _icc_from(fm, sem),
        "estimates": _estimates(sem, fitted, req.confidence),
        "r_squared": {lvl: sem.r_squared(fitted, g) for g, lvl in enumerate(LEVELS)},
        "indirect_effects": indirect,
        "fit_indices": _clean_fit(fm),
        "estimation": "Two-level maximum likelihood (complete cases); standard errors from the observed information.",
        "notes": notes + _fit_notes(fitted, fm, sem)
                 + (["Indirect-effect CIs here are delta-method (normal); the cross-level-mediation endpoint gives Monte Carlo CIs."]
                    if any(indirect.values()) else []),
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Random Slopes (multilevel regression)
# ══════════════════════════════════════════════════════════════════════

@router.post("/random-slopes-sem")
def random_slopes_sem(req: RandomSlopesSEMRequest):
    """Multilevel regression for one outcome with random intercepts and slopes (REML, as
    lme4::lmer's default), cross-level interactions, and within / between (contextual)
    effects of each predictor from a random-intercept model with the predictor split into
    its cluster mean and the deviation from it."""
    l2 = req.level2_predictors or []
    cols = list(dict.fromkeys([req.outcome] + req.predictors + l2))
    df, dropped = _frame(req.data, cols, req.group_ids)
    J = df[req.group_ids].nunique()
    N = len(df)
    if N < 20 or J < 5:
        raise HTTPException(400, "Need at least 20 complete observations in at least 5 clusters.")
    for rs in req.random_slopes:
        if rs not in req.predictors:
            raise HTTPException(400, f"Random slope '{rs}' must be in predictors list.")
    zc = sp_stats.norm.ppf(0.5 + req.confidence / 2)

    fixed_terms = list(req.predictors) + l2
    for w in l2:
        for rs in req.random_slopes:
            col = f"{w}_x_{rs}"
            df[col] = df[w] * df[rs]
            fixed_terms.append(col)
    formula = f"{req.outcome} ~ " + " + ".join(fixed_terms)
    re_formula = "~ " + " + ".join(req.random_slopes) if req.random_slopes else "~ 1"
    try:
        model = fit_mixedlm(smf.mixedlm(formula, df, groups=df[req.group_ids], re_formula=re_formula), reml=True)
    except Exception as exc:
        raise HTTPException(422, f"Model failed to converge: {exc}. Try centering predictors or removing random slopes.")

    fe, bse = model.fe_params, fixed_effect_se(model)[0]
    fixed = []
    for name in fe.index:
        est, se = float(fe[name]), float(bse[name])
        z = est / se if se > 0 else None
        p = float(2 * sp_stats.norm.sf(abs(z))) if z is not None else None
        fixed.append({"name": name, "estimate": est, "se": se, "z": z, "p": p,
                      "ci_lower": est - zc * se, "ci_upper": est + zc * se})

    cov_re = model.cov_re
    re_names = list(cov_re.index)
    random = {"variances": {n: float(cov_re.loc[n, n]) for n in re_names}, "correlations": {}}
    for i, a in enumerate(re_names):
        for b in re_names[i + 1:]:
            den = np.sqrt(cov_re.loc[a, a] * cov_re.loc[b, b])
            random["correlations"][f"{a} ~ {b}"] = float(cov_re.loc[a, b] / den) if den > 0 else None

    cross = [f for f in fixed if "_x_" in f["name"]]

    # Within / between (contextual) effects, random-intercept model per predictor
    contextual = {}
    for x in req.predictors:
        d2 = df[[req.outcome, x, req.group_ids]].copy()
        d2["x_between"] = d2.groupby(req.group_ids)[x].transform("mean")
        d2["x_within"] = d2[x] - d2["x_between"]
        if d2["x_between"].std() == 0 or d2["x_within"].std() == 0:
            continue
        try:
            m2 = fit_mixedlm(smf.mixedlm(f"{req.outcome} ~ x_within + x_between", d2, groups=d2[req.group_ids]), reml=True)
        except Exception:
            continue
        b = m2.fe_params
        se2, V = fixed_effect_se(m2)
        ctx = float(b["x_between"] - b["x_within"])
        ctx_se = float(np.sqrt(V.loc["x_between", "x_between"] + V.loc["x_within", "x_within"] - 2 * V.loc["x_between", "x_within"]))
        contextual[x] = {
            "within_effect": float(b["x_within"]), "within_se": float(se2["x_within"]),
            "between_effect": float(b["x_between"]), "between_se": float(se2["x_between"]),
            "contextual_effect": ctx, "contextual_se": ctx_se,
            "contextual_p": float(2 * sp_stats.norm.sf(abs(ctx / ctx_se))) if ctx_se > 0 else None,
        }

    k_re = len(re_names)
    n_params = len(fe) + k_re * (k_re + 1) // 2 + 1
    return {
        "n_observations": N,
        "n_groups": J,
        "n_dropped_incomplete": dropped,
        "formula": formula,
        "random_effects_formula": re_formula,
        "fixed_effects": fixed,
        "random_effects": random,
        "residual_variance": float(model.scale),
        "cross_level_interactions": cross,
        "contextual_effects": contextual,
        "model_fit": {
            "reml_criterion": float(-2 * model.llf),
            "n_parameters": n_params,
            "aic_reml": float(-2 * model.llf + 2 * n_params),
            "converged": bool(model.converged),
        },
        "notes": [
            "This is a single-equation multilevel regression (one outcome), estimated by REML.",
            "REML-based AIC can only compare models with the same fixed effects; refit with ML to compare fixed effects.",
            "Cross-level interaction terms use the variables as entered; centre them first if the main effects should be interpretable at the mean.",
            "Contextual effect = between-cluster effect minus within-cluster effect of the predictor (random-intercept model with the cluster mean and the deviation from it).",
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# 3. Within-Between Decomposition
# ══════════════════════════════════════════════════════════════════════

@router.post("/within-between-decomposition")
def within_between_decomposition(req: WithinBetweenDecompRequest):
    """Pooled-within and between covariance matrices, ICCs and design effects."""
    df, dropped = _frame(req.data, req.variables, req.group_ids)
    J = df[req.group_ids].nunique()
    N = len(df)
    if J < 3 or N < 10:
        raise HTTPException(400, "Need at least 10 complete observations in at least 3 clusters.")
    within_vars = [v for v in req.variables if _varies_within(df, v, req.group_ids)]
    if not within_vars:
        raise HTTPException(400, "No variable varies within clusters.")
    sem = TwoLevelSEM(Model(within_vars, [], [], 2, True), df[within_vars].values, df[req.group_ids].values, within_vars)
    SW = sem.SSW / (N - J)
    sizes = sem.cluster_sizes
    n_tilde = (N - np.sum(sizes ** 2) / N) / (J - 1)
    MSB = sizes[:, None, None] * np.einsum("ji,jk->jik", sem.cluster_means - sem.cluster_means.T @ sizes / N,
                                           sem.cluster_means - sem.cluster_means.T @ sizes / N)
    SB_mom = (MSB.sum(axis=0) / (J - 1) - SW) / n_tilde

    # ML ICCs from the unrestricted two-level model
    sat, _ = two_level_reference_models(sem, [], [])
    fs = sat.fit()
    SWml, SBml, _ = sat._levels(fs.theta)

    def corr(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
        return (M / np.outer(d, d)).tolist()

    out = {}
    for k, v in enumerate(within_vars):
        icc_mom = SB_mom[k, k] / (SB_mom[k, k] + SW[k, k]) if SB_mom[k, k] + SW[k, k] > 0 else 0.0
        icc_ml = SBml[k, k] / (SBml[k, k] + SWml[k, k])
        deff = 1 + (float(np.mean(sizes)) - 1) * max(icc_ml, 0.0)
        out[v] = {"icc_ml": float(icc_ml), "icc_anova": float(icc_mom), "design_effect": float(deff),
                  "effective_n": float(N / deff)}
    return {
        "n_observations": N,
        "n_groups": J,
        "n_dropped_incomplete": dropped,
        "avg_group_size": float(np.mean(sizes)),
        "variables": within_vars,
        "between_only_variables": [v for v in req.variables if v not in within_vars],
        "within_covariance": SW.tolist(),
        "between_covariance_anova": SB_mom.tolist(),
        "within_covariance_ml": SWml.tolist(),
        "between_covariance_ml": SBml.tolist(),
        "within_correlation_ml": corr(SWml),
        "between_correlation_ml": corr(SBml),
        "variable_summary": out,
        "notes": [
            "Within covariance = pooled within-cluster covariance (divisor N - J). The ANOVA between covariance is "
            "(MSB - MSW) / n~ with n~ = (N - sum n_j^2 / N) / (J - 1), and can have negative variances.",
            "ML matrices are the unrestricted two-level estimates (as lavaan's h1); ICC (ML) matches lavaan's lavInspect(fit, 'icc').",
            "Design effect = 1 + (mean cluster size - 1) x ICC (Kish), using the ML ICC.",
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# 4. Cross-Level Mediation
# ══════════════════════════════════════════════════════════════════════

@router.post("/cross-level-mediation")
def cross_level_mediation(req: CrossLevelMediationRequest):
    """Multilevel mediation as a two-level SEM (Preacher, Zyphur & Zhang, 2010).

    1-1-1: X, M, Y all vary within clusters; separate within and between indirect effects.
    2-1-1: X is a cluster-level variable; the indirect effect exists only between clusters.
    1-1-2: Y is a cluster-level variable; the indirect effect exists only between clusters.
    Indirect effects get delta-method SEs and Monte Carlo CIs (draws from the estimated
    sampling distribution of all parameters).
    """
    n = len(req.m)
    if not (len(req.y) == n and len(req.group_ids) == n and (len(req.x) == n or req.level2_x is not None)):
        raise HTTPException(400, "x, m, y and group_ids must have the same length.")
    mt = req.model_type.replace(" ", "")
    if mt not in ("1-1-1", "2-1-1", "1-1-2"):
        raise HTTPException(400, f"model_type must be '1-1-1', '2-1-1', or '1-1-2'. Got '{req.model_type}'.")
    groups = pd.Series(req.group_ids, dtype=str)
    if mt == "2-1-1" and req.level2_x is not None:
        order = list(dict.fromkeys(groups))
        if len(req.level2_x) != len(order):
            raise HTTPException(400, f"level2_x needs one value per cluster ({len(order)}), in order of first appearance.")
        xmap = dict(zip(order, req.level2_x))
        x = groups.map(xmap).astype(float).values
    else:
        x = np.asarray(req.x, dtype=float)
    data = pd.DataFrame({"x": x, "m": np.asarray(req.m, float), "y": np.asarray(req.y, float), "cluster": groups})
    data = data.dropna()
    J = _check_clusters(data, "cluster", 3)
    conf = float(np.clip(req.confidence, 0.5, 0.999))
    n_mc = int(max(1000, min(req.n_monte_carlo, 100000)))

    varies = {v: _varies_within(data, v, "cluster") for v in ("x", "m", "y")}
    expected = {"1-1-1": {"x": True, "m": True, "y": True},
                "2-1-1": {"x": False, "m": True, "y": True},
                "1-1-2": {"x": True, "m": True, "y": False}}[mt]
    for v, should in expected.items():
        if varies[v] != should:
            raise HTTPException(
                400, f"For a {mt} model, {v.upper()} must {'vary' if should else 'be constant'} within clusters, but it "
                     f"{'does not' if should else 'varies'}.")
    if mt == "1-1-1":
        reg_w, reg_b = [("m", "x"), ("y", "m"), ("y", "x")], [("m", "x"), ("y", "m"), ("y", "x")]
        within_vars, observed = ["m", "y", "x"], ["m", "y", "x"]
    elif mt == "2-1-1":
        reg_w, reg_b = [("y", "m")], [("m", "x"), ("y", "m"), ("y", "x")]
        within_vars, observed = ["m", "y"], ["m", "y", "x"]
    else:
        reg_w, reg_b = [("m", "x")], [("m", "x"), ("y", "m"), ("y", "x")]
        within_vars, observed = ["m", "x"], ["m", "x", "y"]

    def build(SW, SB, mu):
        pw, ew = _level_params(reg_w, within_vars, 0, SW, None)
        pb, eb = _level_params(reg_b, observed, 1, SB, mu)
        return pw + pb, [], ew, eb

    sem, fitted, fm = _fit_two_level(data, "cluster", observed, within_vars, build)
    zc = sp_stats.norm.ppf(0.5 + conf / 2)

    def effect(fn, label, seed_offset):
        d = sem.defined(fitted, fn)
        lo, hi = _mc_ci(sem, fitted, fn, n_mc, conf, req.seed + seed_offset)
        return {"effect": label, **d, "ci_lower_mc": lo, "ci_upper_mc": hi,
                "significant": bool(lo is not None and (lo > 0 or hi < 0))}

    v = lambda s, t, lhs, rhs, g: s.value(t, lhs, "~", rhs, g)
    effects = []
    if mt == "1-1-1":
        effects.append(effect(lambda s, t: v(s, t, "m", "x", 0) * v(s, t, "y", "m", 0), "indirect (within)", 0))
    effects.append(effect(lambda s, t: v(s, t, "m", "x", 1) * v(s, t, "y", "m", 1), "indirect (between)", 1))
    contextual = []
    if mt == "1-1-1":
        for (lhs, rhs, name) in (("m", "x", "a"), ("y", "m", "b"), ("y", "x", "c'")):
            d = sem.defined(fitted, lambda s, t, lhs=lhs, rhs=rhs: v(s, t, lhs, rhs, 1) - v(s, t, lhs, rhs, 0))
            contextual.append({"path": name, **d, "ci_lower": d["est"] - zc * d["se"], "ci_upper": d["est"] + zc * d["se"]})

    paths = []
    for g, lvl in enumerate(LEVELS):
        for lhs, rhs in (reg_w if g == 0 else reg_b):
            name = {"m~x": "a (X->M)", "y~m": "b (M->Y)", "y~x": "c' (X->Y direct)"}[f"{lhs}~{rhs}"]
            est = sem.value(fitted.theta, lhs, "~", rhs, g)
            se = fitted.se[sem.pos[f"{lhs}~{rhs}@g{g}"]]
            z = est / se if se > 0 else None
            paths.append({"path": name, "level": lvl, "estimate": float(est), "se": float(se), "z": z,
                          "p": float(2 * sp_stats.norm.sf(abs(z))) if z is not None else None,
                          "ci_lower": float(est - zc * se), "ci_upper": float(est + zc * se)})

    return {
        "model_type": mt,
        "n_observations": int(fm["n_observations"]),
        "n_groups": J,
        "icc": _icc_from(fm, sem),
        "path_coefficients": paths,
        "indirect_effects": effects,
        "contextual_effects": contextual,
        "fit_indices": _clean_fit(fm),
        "n_monte_carlo": n_mc,
        "confidence_level": conf,
        "seed": req.seed,
        "notes": _fit_notes(fitted, fm, sem) + [
            "Between-level paths relate the latent cluster means (not the observed cluster averages), so they are not biased by within-cluster sampling error.",
        ] + (["Contextual effect = between-level path minus within-level path."] if contextual else []),
    }


# ══════════════════════════════════════════════════════════════════════
# 5. Multilevel CFA (Measurement Model)
# ══════════════════════════════════════════════════════════════════════

@router.post("/measurement-model-multilevel")
def measurement_model_multilevel(req: MultilevelCFARequest):
    """Two-level CFA: the same factor structure at the within and between level
    (lavaan cfa(..., cluster = )), optionally with loadings equal across levels."""
    items, factors = [], list(req.factor_model)
    for f, its in req.factor_model.items():
        if len(its) < 2:
            raise HTTPException(400, f"Factor '{f}' needs at least 2 items.")
        for it in its:
            if it in items:
                raise HTTPException(400, f"Item '{it}' assigned to multiple factors.")
            items.append(it)
    if len(items) < 3:
        raise HTTPException(400, "Need at least 3 items for CFA.")
    df, dropped = _frame(req.data, items, req.group_ids)
    J = _check_clusters(df, req.group_ids, len(items))
    flat = [it for it in items if not _varies_within(df, it, req.group_ids)]
    if flat:
        raise HTTPException(400, f"Items {flat} do not vary within clusters.")

    def build(SW, SB, mu):
        P = cfa_params(req.factor_model, groups=2, meanstructure=True, equal=("loadings",) if req.equal_loadings else (),
                       S_list=[SW, SB], ybar_list=[mu, mu], observed=items)
        P = [p for p in P if not (p.group == 0 and p.op == "~1")]
        return P, factors, [], []

    sem, fitted, fm = _fit_two_level(df, req.group_ids, items, items, build)
    est = _estimates(sem, fitted, 0.95)

    levels = {}
    for g, lvl in enumerate(LEVELS):
        A, S, M = sem.matrices(fitted.theta, g)
        loadings = {f: [] for f in factors}
        for r in est[lvl]:
            if r["op"] == "=~":
                loadings[r["lhs"]].append(r)
        for f in factors:  # marker loadings (fixed at 1)
            first = req.factor_model[f][0]
            std = next((x["std_all"] for x in sem.estimates(fitted) if x["group"] == g + 1 and x["op"] == "=~"
                        and x["lhs"] == f and x["rhs"] == first), None)
            loadings[f].insert(0, {"lhs": f, "op": "=~", "rhs": first, "est": 1.0, "se": None, "z": None,
                                   "p_value": None, "std_all": std, "fixed": True})
        fi = [sem.idx[f] for f in factors]
        Phi = S[np.ix_(fi, fi)]
        d = np.sqrt(np.clip(np.diag(Phi), 1e-12, None))
        fcor = {f"{factors[i]} ~~ {factors[j]}": float(Phi[i, j] / (d[i] * d[j]))
                for i in range(len(factors)) for j in range(i + 1, len(factors))}
        omega = {}
        for i, f in enumerate(factors):
            idx = [sem.idx[it] for it in req.factor_model[f]]
            lam = A[idx, sem.idx[f]]
            th = np.array([S[k, k] for k in idx])
            num = lam.sum() ** 2 * Phi[i, i]
            omega[f] = float(num / (num + th.sum())) if num + th.sum() > 0 else None
        levels[lvl] = {
            "loadings": loadings,
            "residual_variances": {it: float(S[sem.idx[it], sem.idx[it]]) for it in items},
            "factor_variances": {f: float(Phi[i, i]) for i, f in enumerate(factors)},
            "factor_correlations": fcor,
            "omega": omega,
            "r_squared": sem.r_squared(fitted, g),
        }
    levels["between"]["intercepts"] = {r["lhs"]: r["est"] for r in est["between"] if r["op"] == "~1"}

    return {
        "n_observations": int(fm["n_observations"]),
        "n_groups": J,
        "n_dropped_incomplete": dropped,
        "n_items": len(items),
        "n_factors": len(factors),
        "factor_names": factors,
        "equal_loadings_across_levels": req.equal_loadings,
        "within_level": levels["within"],
        "between_level": levels["between"],
        "estimates": est,
        "icc_per_item": _icc_from(fm, sem),
        "fit_indices": _clean_fit(fm),
        "notes": _fit_notes(fitted, fm, sem) + [
            "Omega per level = (sum of loadings)^2 x factor variance / ((sum of loadings)^2 x factor variance + sum of residual variances).",
            "Identification: first loading of each factor fixed to 1 at each level.",
        ],
    }
