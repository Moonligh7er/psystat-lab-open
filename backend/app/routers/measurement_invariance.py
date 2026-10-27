"""Measurement invariance across groups: configural, metric, scalar and strict multi-group
CFA (lavaan's group.equal), partial invariance by sequential likelihood-ratio freeing
(lavaan's group.partial), and the alignment method (Asparouhov & Muthén, 2014, as
sirt::invariance.alignment). Model fitting uses app/sem_engine.py; reference values are
in tests/test_sem.py."""

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import optimize
from scipy import stats as sp_stats

from app.routers.sem import ESTIMATION_NOTE, _check_cov, _fit_dict, _interpret, _run, _validate_index_model
from app.sem_engine import SEM, Model, Param, cfa_params

router = APIRouter()


class InvarianceRequest(BaseModel):
    """Measurement invariance testing request."""
    data_groups: list[list[list[float]]]  # list of data matrices (rows=obs, cols=items)
    factor_model: dict[str, list[int]]    # factor_name -> list of item column indices (0-based)


class PartialInvarianceRequest(BaseModel):
    """Partial invariance: find the parameters to release."""
    data_groups: list[list[list[float]]]
    factor_model: dict[str, list[int]]
    invariance_level: str = "metric"  # "metric" or "scalar"
    alpha: float = 0.05


class AlignmentRequest(BaseModel):
    """Alignment: either raw data (a configural model is fitted with factor means 0 and
    variances 1 in every group), or configural estimates for a one-factor model per factor."""
    data_groups: list[list[list[float]]] | None = None
    factor_model: dict[str, list[int]] | None = None
    group_loadings: list[list[list[float]]] | None = None     # n_groups x p x n_factors
    group_intercepts: list[list[float]] | None = None         # n_groups x p
    group_sizes: list[int] | None = None                      # weights sqrt(N_g); equal if omitted


LEVEL_EQUAL = {
    "configural": (),
    "metric": ("loadings",),
    "scalar": ("loadings", "intercepts"),
    "strict": ("loadings", "intercepts", "residuals"),
}


def _groups(data_groups, factor_model):
    if len(data_groups) < 2:
        raise HTTPException(422, "At least 2 groups are required.")
    arrays = []
    for i, g in enumerate(data_groups):
        if not g or len({len(r) for r in g}) != 1:
            raise HTTPException(422, f"Group {i + 1} must be a rectangular matrix.")
        arrays.append(np.array(g, dtype=float))
    p = arrays[0].shape[1]
    if any(a.shape[1] != p for a in arrays):
        raise HTTPException(422, "All groups must have the same number of items.")
    _validate_index_model(factor_model, p)
    used = sorted({j for idx in factor_model.values() for j in idx})
    names = [f"item{j + 1}" for j in range(p)]
    out = []
    for i, a in enumerate(arrays):
        a = a[~np.isnan(a).any(axis=1)][:, used]
        if len(a) < len(used) + 2:
            raise HTTPException(422, f"Group {i + 1} needs more complete observations ({len(a)}) than items ({len(used)}).")
        _check_cov(a, [names[j] for j in used])
        out.append(a)
    factors = {f: [names[j] for j in idx] for f, idx in factor_model.items()}
    observed = [names[j] for j in used]
    return out, factors, observed


def _fit_level(data, factors, observed, level, release=None):
    Ss = [np.cov(d, rowvar=False, bias=True) for d in data]
    ys = [d.mean(axis=0) for d in data]
    params = cfa_params(factors, len(data), meanstructure=True, equal=LEVEL_EQUAL[level], S_list=Ss,
                        ybar_list=ys, observed=observed, release=release)
    return _run(Model(observed, list(factors), params, n_groups=len(data), meanstructure=True), data)


def _level_endpoint(req: InvarianceRequest, level: str):
    data, factors, observed = _groups(req.data_groups, req.factor_model)
    sem, fitted, fm = _fit_level(data, factors, observed, level)
    est = sem.estimates(fitted)
    out = {
        "level": level,
        **_fit_dict(fm),
        "fit_indices": _fit_dict(fm),
        "interpretation": _interpret(fm),
        "n_parameters": fm["n_parameters"],
        "n_groups": len(data),
        "group_sizes": [len(d) for d in data],
        "estimation": ESTIMATION_NOTE,
        "constraints": {"configural": "same factor structure; all parameters free in each group",
                        "metric": "loadings equal across groups",
                        "scalar": "loadings and intercepts equal; latent means free in groups 2..G",
                        "strict": "loadings, intercepts and residual variances equal"}[level],
        "converged": fitted.converged,
    }
    if level in ("scalar", "strict"):
        out["latent_means"] = [{"factor": r["lhs"], "group": r["group"], "mean": r["est"], "se": r["se"],
                                "p_value": r["p_value"]} for r in est if r["op"] == "~1" and r["lhs"] in factors and r["free"]]
    return out


@router.post("/configural")
def configural_invariance(req: InvarianceRequest):
    """Configural model: the same factor structure in every group, nothing constrained."""
    return _level_endpoint(req, "configural")


@router.post("/metric")
def metric_invariance(req: InvarianceRequest):
    """Metric (weak) invariance: equal loadings."""
    return _level_endpoint(req, "metric")


@router.post("/scalar")
def scalar_invariance(req: InvarianceRequest):
    """Scalar (strong) invariance: equal loadings and intercepts."""
    return _level_endpoint(req, "scalar")


@router.post("/strict")
def strict_invariance(req: InvarianceRequest):
    """Strict invariance: equal loadings, intercepts and residual variances."""
    return _level_endpoint(req, "strict")


# ── Partial invariance ───────────────────────────────────────────────

@router.post("/partial-invariance")
def partial_invariance(req: PartialInvarianceRequest):
    """Sequentially release the equality constraint that most improves fit (largest drop in
    chi-square) until the partially constrained model no longer fits significantly worse than
    the less constrained baseline (configural for metric, metric for scalar). At least two
    items per factor stay constrained (Byrne, Shavelson & Muthén, 1989)."""
    if req.invariance_level not in ("metric", "scalar"):
        raise HTTPException(422, "invariance_level must be 'metric' or 'scalar'.")
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    data, factors, observed = _groups(req.data_groups, req.factor_model)
    baseline_level = "configural" if req.invariance_level == "metric" else "metric"
    kind = "loadings" if req.invariance_level == "metric" else "intercepts"
    _, _, base = _fit_level(data, factors, observed, baseline_level)
    # loadings: the marker (first indicator) is fixed to 1 in every group, so it can't be released
    candidates = {f: list(inds[1:] if kind == "loadings" else inds) for f, inds in factors.items()}
    min_constrained = {f: (2 - 1 if kind == "loadings" else 2) for f in factors}  # marker counts as constrained

    def compare(fm):
        dchi = fm["chi_square"] - base["chi_square"]
        ddf = fm["df"] - base["df"]
        return {"delta_chi_square": dchi, "delta_df": ddf,
                "p_value": float(sp_stats.chi2.sf(max(dchi, 0), ddf)) if ddf > 0 else None,
                "delta_cfi": fm["cfi"] - base["cfi"], "delta_rmsea": fm["rmsea"] - base["rmsea"]}

    released: set[str] = set()
    _, _, cur = _fit_level(data, factors, observed, req.invariance_level)
    steps = [{"released": None, **_fit_dict(cur), **compare(cur)}]
    while True:
        comp = compare(cur)
        if comp["p_value"] is None or comp["p_value"] >= req.alpha:
            break
        best = None
        for f, items in candidates.items():
            still = [i for i in items if i not in released]
            if len(still) <= min_constrained[f]:
                continue
            for item in still:
                _, _, fm = _fit_level(data, factors, observed, req.invariance_level, {kind: released | {item}})
                if best is None or fm["chi_square"] < best[1]["chi_square"]:
                    best = (item, fm, f)
        if best is None:
            break
        released.add(best[0])
        prev = cur
        cur = best[1]
        steps.append({"released": best[0], "factor": best[2], **_fit_dict(cur), **compare(cur),
                      "chi_square_drop": prev["chi_square"] - cur["chi_square"]})
    final = compare(cur)
    holds = final["p_value"] is None or final["p_value"] >= req.alpha
    return {
        "invariance_level": req.invariance_level,
        "parameter_type": kind,
        "baseline_level": baseline_level,
        "baseline_fit": _fit_dict(base),
        "steps": steps,
        "released_items": sorted(released, key=observed.index),
        "partial_invariance_holds": bool(holds),
        "final_comparison": final,
        "n_groups": len(data),
        "estimation": ESTIMATION_NOTE,
        "interpretation": (
            (f"Partial {req.invariance_level} invariance holds after releasing the {kind} of "
             f"{', '.join(sorted(released, key=observed.index)) or 'no items'}: the constrained model does not fit "
             f"significantly worse than the {baseline_level} model (chi-square difference test, alpha = {req.alpha}).")
            if holds else
            (f"Even after releasing the {kind} of {', '.join(sorted(released, key=observed.index)) or 'no items'}, the model "
             f"still fits significantly worse than the {baseline_level} model while keeping at least two items per "
             f"factor constrained, so partial {req.invariance_level} invariance is not supported.")),
        "note": ("Released items are chosen greedily by the largest chi-square drop. With large samples the "
                 "chi-square difference test is sensitive to trivial differences; compare delta CFI too."),
    }


# ── Alignment (Asparouhov & Muthén, 2014; sirt::invariance.alignment) ─

ALIGN_EPS = 0.001


def _align_loss(alpha, psi, lam, nu, wpair, eps):
    """sum over items and ordered group pairs of w_g w_h [f(lam1_g - lam1_h) + f(nu1_g - nu1_h)],
    f(x) = (x^2 + eps)^(1/4); lam1 = lam / psi, nu1 = nu - alpha * lam1 (psi = factor SD)."""
    l1 = lam / psi[:, None]
    n1 = nu - alpha[:, None] * l1
    dl = l1[:, None, :] - l1[None, :, :]
    dn = n1[:, None, :] - n1[None, :, :]
    return float(np.sum(wpair * ((dl ** 2 + eps) ** 0.25 + (dn ** 2 + eps) ** 0.25)))


def align_one_factor(lam: np.ndarray, nu: np.ndarray, wgt: np.ndarray):
    """Alignment for one factor. lam, nu: groups x items configural estimates (factor mean 0,
    SD 1 in every group). Group 1 is the reference (alpha = 0, psi = 1)."""
    G, I = lam.shape
    w = wgt / wgt.sum(axis=0, keepdims=True)              # sirt normalises weights per item
    wpair = w[:, None, :] * w[None, :, :]
    psi0 = np.median(lam, axis=1)
    psi0 = psi0 / psi0[0]
    alpha0 = np.median(nu, axis=1) - np.median(nu, axis=1)[0]
    x = np.concatenate([alpha0[1:], np.log(np.clip(psi0[1:], 1e-3, None))])

    def unpack(z):
        return np.concatenate([[0.0], z[: G - 1]]), np.concatenate([[1.0], np.exp(z[G - 1:])])
    # continuation over eps (as sirt): 10^0, 10^-0.5, ... down to the target eps
    eps_seq = sorted({float(e) for e in 10.0 ** np.arange(0, -10.5, -0.5) if e >= ALIGN_EPS} | {ALIGN_EPS}, reverse=True)
    for eps in eps_seq:
        res = optimize.minimize(lambda z: _align_loss(*unpack(z), lam, nu, wpair, eps), x, method="BFGS",
                                options={"gtol": 1e-10, "maxiter": 5000})
        x = res.x
    alpha, psi = unpack(x)
    l1 = lam / psi[:, None]
    n1 = nu - alpha[:, None] * l1
    m_l, m_n = l1.mean(axis=0), n1.mean(axis=0)
    r2_l = 1 - np.sum((lam - psi[:, None] * m_l) ** 2) / np.sum(lam ** 2)
    r2_n = 1 - np.sum((nu - (m_n + alpha[:, None] * m_l)) ** 2) / np.sum(nu ** 2)

    def rbar(M):
        if M.shape[0] < 2:
            return None
        c = np.corrcoef(M)
        return float((c.sum() - len(c)) / (len(c) ** 2 - len(c)))
    return {
        "alpha": alpha, "psi": psi, "lambda_aligned": l1, "nu_aligned": n1,
        "r2_loadings": float(r2_l), "r2_intercepts": float(r2_n),
        "rbar_loadings": rbar(l1), "rbar_intercepts": rbar(n1),
        "loss": _align_loss(alpha, psi, lam, nu, wpair, ALIGN_EPS),
        "converged": bool(res.success),
    }


def _configural_std(data, factors, observed):
    """Per-group CFA with factor means 0 and variances 1 (lavaan std.lv = TRUE), intercepts free."""
    lam, nu = [], []
    lat = list(factors)
    for d in data:
        S = np.cov(d, rowvar=False, bias=True)
        params = []
        for f, inds in factors.items():
            for ind in inds:
                params.append(Param(f, "=~", ind, value=float(np.sqrt(max(S[observed.index(ind), observed.index(ind)], 1e-6)) * 0.7)))
            params.append(Param(f, "~~", f, free=False, value=1.0))
            params.append(Param(f, "~1", "", free=False, value=0.0))
        for i, f in enumerate(lat):
            for f2 in lat[i + 1:]:
                params.append(Param(f, "~~", f2, value=0.0))
        for j, ind in enumerate(observed):
            params.append(Param(ind, "~~", ind, value=0.5 * S[j, j]))
            params.append(Param(ind, "~1", "", value=float(d[:, j].mean())))
        sem = SEM(Model(observed, lat, params, meanstructure=True), [d])
        fitted = sem.fit()
        est = sem.estimates(fitted)
        L = {(r["lhs"], r["rhs"]): r["est"] for r in est if r["op"] == "=~"}
        N = {r["lhs"]: r["est"] for r in est if r["op"] == "~1" and r["lhs"] in observed}
        # orient each factor so its loadings sum positive (sign is arbitrary under std.lv)
        for f, inds in factors.items():
            if sum(L[(f, i)] for i in inds) < 0:
                for i in inds:
                    L[(f, i)] = -L[(f, i)]
        lam.append(L)
        nu.append(N)
    return lam, nu


@router.post("/alignment")
def alignment_method(req: AlignmentRequest):
    """Alignment optimization (Asparouhov & Muthén, 2014): estimates group factor means and SDs
    from the configural model by minimising a component loss that favours a few large
    non-invariances over many small ones. Reproduces sirt::invariance.alignment (align.pow 0.25,
    eps = 0.001, group 1 as reference)."""
    if req.data_groups is not None:
        if req.factor_model is None:
            raise HTTPException(422, "With raw data, give the factor_model too.")
        data, factors, observed = _groups(req.data_groups, req.factor_model)
        lam_g, nu_g = _configural_std(data, factors, observed)
        sizes = np.array([len(d) for d in data], dtype=float)
        blocks = {f: (np.array([[lg[(f, i)] for i in inds] for lg in lam_g]),
                      np.array([[ng[i] for i in inds] for ng in nu_g]), inds) for f, inds in factors.items()}
    else:
        if req.group_loadings is None or req.group_intercepts is None:
            raise HTTPException(422, "Give either data_groups + factor_model, or group_loadings + group_intercepts.")
        L = [np.array(g, dtype=float) for g in req.group_loadings]
        Nu = [np.array(g, dtype=float) for g in req.group_intercepts]
        if len(L) < 2 or len(L) != len(Nu):
            raise HTTPException(422, "Need at least 2 groups, with one loading matrix and one intercept vector each.")
        if any(x.ndim != 2 or x.shape != L[0].shape for x in L) or any(len(n) != L[0].shape[0] for n in Nu):
            raise HTTPException(422, "Every group needs a p x n_factors loading matrix and p intercepts.")
        sizes = np.array(req.group_sizes, dtype=float) if req.group_sizes else np.ones(len(L))
        if len(sizes) != len(L) or np.any(sizes <= 0):
            raise HTTPException(422, "group_sizes must give one positive size per group.")
        p, q = L[0].shape
        blocks = {}
        for f in range(q):
            items = [j for j in range(p) if any(abs(Lg[j, f]) > 0 for Lg in L)]
            if any(any(abs(Lg[j, k]) > 0 for k in range(q) if k != f) for Lg in L for j in items):
                raise HTTPException(422, "Alignment needs simple structure: each item loads on one factor.")
            blocks[f"F{f + 1}"] = (np.array([[Lg[j, f] for j in items] for Lg in L]),
                                   np.array([[Nu[g][j] for j in items] for g in range(len(L))]),
                                   [f"item{j + 1}" for j in items])
    G = len(sizes)
    out_f = []
    for f, (lam, nu, items) in blocks.items():
        if lam.shape[1] < 2:
            raise HTTPException(422, f"Factor {f} needs at least 2 items for alignment.")
        wgt = np.tile(np.sqrt(sizes)[:, None], (1, lam.shape[1]))
        r = align_one_factor(lam, nu, wgt)
        out_f.append({
            "factor": f,
            "items": items,
            "groups": [{"group": g + 1, "mean": float(r["alpha"][g]), "sd": float(r["psi"][g]),
                        "variance": float(r["psi"][g] ** 2)} for g in range(G)],
            "aligned_loadings": r["lambda_aligned"].tolist(),
            "aligned_intercepts": r["nu_aligned"].tolist(),
            "configural_loadings": lam.tolist(),
            "configural_intercepts": nu.tolist(),
            "r_squared": {"loadings": r["r2_loadings"], "intercepts": r["r2_intercepts"]},
            "rbar": {"loadings": r["rbar_loadings"], "intercepts": r["rbar_intercepts"]},
            "loss": r["loss"],
            "converged": r["converged"],
        })
    return {
        "method": "alignment (Asparouhov & Muthén, 2014)",
        "n_groups": G,
        "group_sizes": sizes.tolist(),
        "factors": out_f,
        "settings": {"loss": "f(x) = (x^2 + 0.001)^(1/4)", "weights": "sqrt(group size), normalised per item",
                     "reference_group": 1, "identification": "group 1 mean 0, SD 1"},
        "interpretation": (
            "Group factor means and SDs are chosen so that aligned loadings and intercepts are as similar across "
            "groups as possible, tolerating a few large differences. R-squared near 1 means the configural "
            "differences are almost entirely explained by group differences in factor mean and SD (approximate "
            "invariance); Asparouhov & Muthén suggest the means are trustworthy when roughly 75% or more of the "
            "parameters are invariant. rbar is the average correlation of aligned parameters between groups."),
        "note": ("Item-level significance tests of non-invariance (as in Mplus) need standard errors of the "
                 "configural estimates and are not computed here."),
    }
