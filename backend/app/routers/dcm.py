"""Diagnostic Classification Models — DINA, DINO, G-DINA cognitive diagnostic models.

Marginal maximum likelihood via EM with a saturated (unstructured) attribute distribution,
identity link, item success probabilities bounded to [1e-4, 1 - 1e-4] and no monotonicity
constraint, as in the R package GDINA (Ma & de la Torre, 2020). DINA and DINO are the
G-DINA model with the latent classes of each item collapsed into two groups ("has every
required attribute" / "has at least one"), so the M-step is closed form for all three.
Q-matrix validation uses the PVAF method of de la Torre & Chiu (2016) on the G-DINA fit.
Checked against GDINA 2.13.2 in tests/test_dcm.py.
"""

from itertools import combinations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()

P_BOUNDS = (1e-4, 1 - 1e-4)
MODELS = {"dina": "DINA", "dino": "DINO", "gdina": "G-DINA"}


# ── Request Models ────────────────────────────────────────────────────

class DinaRequest(BaseModel):
    responses: list[list[int]]  # N × J binary response matrix
    q_matrix: list[list[int]]   # J × K binary Q-matrix (items × attributes)
    max_iter: int = 5000
    tol: float = 1e-8
    random_starts: int = 5
    seed: int = 1

class DinoRequest(DinaRequest):
    pass

class GdinaRequest(DinaRequest):
    pass

class CompareRequest(DinaRequest):
    pass

class QMatrixValidateRequest(DinaRequest):
    eps: float = 0.95  # PVAF cut-off


# ── Helpers ───────────────────────────────────────────────────────────

def _patterns(K: int) -> np.ndarray:
    """All 2^K attribute patterns, ordered as GDINA::attributepattern (by number mastered)."""
    rows = []
    for m in range(K + 1):
        for idx in combinations(range(K), m):
            r = np.zeros(K, dtype=int)
            r[list(idx)] = 1
            rows.append(r)
    return np.array(rows, dtype=int)


def _pattern_label(row) -> str:
    return "".join(str(int(v)) for v in row)


def _item_groups(patterns: np.ndarray, q_row: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Group index of every latent class for one item, plus the group patterns.

    G-DINA: one group per reduced pattern of the required attributes (GDINA order).
    DINA: 0 = lacks a required attribute, 1 = has them all. DINO: 0 = has none, 1 = has one or more.
    """
    req = np.where(q_row > 0)[0]
    sub = patterns[:, req]
    if model == "dina":
        return sub.all(axis=1).astype(int), np.array([[0], [1]])
    if model == "dino":
        return sub.any(axis=1).astype(int), np.array([[0], [1]])
    reduced = _patterns(len(req))
    lookup = {tuple(r): i for i, r in enumerate(reduced)}
    return np.array([lookup[tuple(r)] for r in sub]), reduced


def _fit(X: np.ndarray, Q: np.ndarray, model: str, max_iter: int, tol: float,
         random_starts: int = 5, seed: int = 1) -> dict:
    """Fit a DCM by EM from a rational start plus `random_starts` random ones; keep the best."""
    N, J = X.shape
    K = Q.shape[1]
    patterns = _patterns(K)
    L = len(patterns)
    groups = [_item_groups(patterns, Q[j], model) for j in range(J)]
    rng = np.random.default_rng(seed)

    def em(start_p: list[np.ndarray], pi: np.ndarray):
        gp = [p.copy() for p in start_p]
        ll_old = -np.inf
        converged = False
        for it in range(1, max_iter + 1):
            P = np.column_stack([gp[j][groups[j][0]] for j in range(J)])  # L × J
            loglik = X @ np.log(P).T + (1 - X) @ np.log1p(-P).T + np.log(pi)
            m = loglik.max(axis=1, keepdims=True)
            post = np.exp(loglik - m)
            s = post.sum(axis=1, keepdims=True)
            ll = float(np.sum(m.ravel() + np.log(s.ravel())))
            post /= s
            Nc = post.sum(axis=0)
            R1 = post.T @ X  # expected number correct per class and item
            new_gp = []
            for j in range(J):
                g = groups[j][0]
                n_g = np.bincount(g, weights=Nc, minlength=len(groups[j][1]))
                r_g = np.bincount(g, weights=R1[:, j], minlength=len(groups[j][1]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    pg = np.where(n_g > 0, r_g / n_g, gp[j])
                new_gp.append(np.clip(pg, *P_BOUNDS))
            new_pi = np.maximum(Nc / N, 1e-300)
            change = max(max(np.max(np.abs(a - b)) for a, b in zip(new_gp, gp)), np.max(np.abs(new_pi - pi)))
            gp, pi = new_gp, new_pi
            if change < tol and abs(ll - ll_old) < 1e-6:
                converged = True
                break
            ll_old = ll
        # Final E-step at the returned parameters
        P = np.column_stack([gp[j][groups[j][0]] for j in range(J)])
        loglik = X @ np.log(P).T + (1 - X) @ np.log1p(-P).T + np.log(pi)
        m = loglik.max(axis=1, keepdims=True)
        post = np.exp(loglik - m)
        s = post.sum(axis=1, keepdims=True)
        ll = float(np.sum(m.ravel() + np.log(s.ravel())))
        return {"gp": gp, "pi": pi, "post": post / s, "ll": ll, "iterations": it, "converged": converged}

    rational = [0.2 + 0.6 * grp[1].sum(axis=1) / max(1, grp[1].shape[1]) for grp in groups]
    runs = [em(rational, np.full(L, 1.0 / L))]
    for _ in range(max(0, random_starts)):
        start = [np.sort(rng.uniform(0.05, 0.95, len(grp[1]))) if model != "gdina"
                 else 0.1 + 0.8 * (grp[1].sum(axis=1) / max(1, grp[1].shape[1])) * rng.uniform(0.6, 1.0, len(grp[1]))
                 for grp in groups]
        runs.append(em(start, rng.dirichlet(np.ones(L))))
    lls = np.array([r["ll"] for r in runs])
    best = runs[int(np.argmax(lls))]
    best["n_starts"] = len(runs)
    best["n_reaching_best"] = int(np.sum(lls > lls.max() - 1e-4))
    best["patterns"] = patterns
    best["groups"] = groups
    return best


def _summarize(fit: dict, X: np.ndarray, Q: np.ndarray, model: str, attr_names: list[str]) -> dict:
    N, J = X.shape
    K = Q.shape[1]
    patterns, post, pi, gp, groups = fit["patterns"], fit["post"], fit["pi"], fit["gp"], fit["groups"]
    L = len(patterns)
    n_item = sum(len(g) for g in gp)
    npar = n_item + L - 1
    ll = fit["ll"]

    mastery = post @ patterns  # marginal P(mastery) per person and attribute
    map_idx = post.argmax(axis=1)

    items = []
    for j in range(J):
        req = [attr_names[k] for k in np.where(Q[j] > 0)[0]]
        entry = {"item": j + 1, "attributes": req}
        if model in ("dina", "dino"):
            entry.update({"guess": float(gp[j][0]), "slip": float(1 - gp[j][1]),
                          "discrimination": float(gp[j][1] - gp[j][0])})
        entry["success_probabilities"] = [
            {"pattern": _pattern_label(r), "probability": float(p)} for r, p in zip(groups[j][1], gp[j])
        ]
        items.append(entry)

    notes = []
    if not fit["converged"]:
        notes.append("EM reached the iteration limit before converging; treat the estimates with caution.")
    if fit["n_reaching_best"] < 2 and fit["n_starts"] > 1:
        notes.append("Only one of the starting values reached the best log-likelihood; the solution may be a local maximum.")
    if N < 10 * L:
        notes.append(f"With {L} latent classes, {N} people is a small sample; class proportions and item parameters will be imprecise.")

    return {
        "model": MODELS[model],
        "n_subjects": N,
        "n_items": J,
        "n_attributes": K,
        "attribute_names": attr_names,
        "item_parameters": items,
        "latent_classes": [
            {"pattern": _pattern_label(r), "proportion": float(p)} for r, p in zip(patterns, pi)
        ],
        "attribute_prevalence": {attr_names[k]: float(pi @ patterns[:, k]) for k in range(K)},
        "attribute_mastery_probabilities": mastery.tolist(),
        "most_likely_profiles": [_pattern_label(patterns[i]) for i in map_idx],
        "classification_accuracy": {
            # Posterior-based estimates for MAP classification (Wang et al., 2015), as GDINA::CA
            "attribute": {attr_names[k]: float(np.mean(np.where(patterns[map_idx, k] == 1, mastery[:, k], 1 - mastery[:, k])))
                          for k in range(K)},
            "pattern": float(np.mean(post.max(axis=1))),
        },
        "model_fit": {
            "log_likelihood": ll,
            "deviance": -2 * ll,
            "n_parameters": npar,
            "AIC": -2 * ll + 2 * npar,
            "BIC": -2 * ll + npar * np.log(N),
        },
        "estimation": {
            "iterations": fit["iterations"], "converged": fit["converged"],
            "starts": fit["n_starts"], "starts_reaching_best": fit["n_reaching_best"],
        },
        "notes": notes,
    }


def _validate_inputs(responses, q_matrix):
    """Common validation for DCM inputs."""
    try:
        R = np.array(responses, dtype=float)
        Q = np.array(q_matrix, dtype=int)
    except ValueError:
        raise HTTPException(400, "Responses and Q-matrix rows must all have the same length.")
    if R.ndim != 2:
        raise HTTPException(400, "Responses must be a 2D binary matrix (subjects x items).")
    if Q.ndim != 2:
        raise HTTPException(400, "Q-matrix must be a 2D binary matrix (items x attributes).")
    N, J = R.shape
    if Q.shape[0] != J:
        raise HTTPException(400, f"Q-matrix rows ({Q.shape[0]}) must equal number of items ({J}).")
    K = Q.shape[1]
    if K < 1 or K > 8:
        raise HTTPException(400, "Number of attributes must be between 1 and 8.")
    if J < 2:
        raise HTTPException(400, "Need at least 2 items.")
    if N < 2 ** K + 10:
        raise HTTPException(400, f"Need more subjects than latent classes: at least {2 ** K + 10} for {K} attributes.")
    if not np.all((R == 0) | (R == 1)):
        raise HTTPException(400, "Response matrix must be binary (0/1) with no missing values.")
    if not np.all((Q == 0) | (Q == 1)):
        raise HTTPException(400, "Q-matrix must be binary (0/1).")
    if np.any(Q.sum(axis=1) == 0):
        raise HTTPException(400, "Every item must measure at least one attribute (a Q-matrix row is all zeros).")
    if np.any(Q.sum(axis=0) == 0):
        raise HTTPException(400, "Every attribute must be measured by at least one item (a Q-matrix column is all zeros).")
    return R, Q, N, J, K


def _run(req, model: str) -> dict:
    X, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)
    names = [f"A{k + 1}" for k in range(K)]
    fit = _fit(X, Q, model, req.max_iter, req.tol, req.random_starts, req.seed)
    return _summarize(fit, X, Q, model, names)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/dina")
def dina_model(req: DinaRequest):
    """DINA (Deterministic Inputs, Noisy And) model.

    Two parameters per item: guess = P(correct | lacks a required attribute) and
    slip = P(wrong | has every required attribute).
    """
    return _run(req, "dina")


@router.post("/dino")
def dino_model(req: DinoRequest):
    """DINO (Deterministic Inputs, Noisy Or) model.

    Two parameters per item: guess = P(correct | has none of the required attributes) and
    slip = P(wrong | has at least one).
    """
    return _run(req, "dino")


@router.post("/gdina")
def gdina_model(req: GdinaRequest):
    """G-DINA (Generalized DINA) model, identity link.

    One success probability per combination of the item's required attributes; DINA and
    DINO are special cases.
    """
    return _run(req, "gdina")


@router.post("/compare")
def compare_models(req: CompareRequest):
    """Fit DINA, DINO and G-DINA; compare by AIC/BIC and likelihood-ratio tests.

    DINA and DINO are nested in G-DINA (identity link), so 2·(LL_GDINA − LL_reduced) is
    compared with a chi-square on the difference in parameter counts, as GDINA::anova does.
    """
    X, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)
    names = [f"A{k + 1}" for k in range(K)]
    res = {m: _summarize(_fit(X, Q, m, req.max_iter, req.tol, req.random_starts, req.seed), X, Q, m, names)
           for m in ("dina", "dino", "gdina")}

    fit_table = [{"model": r["model"], **r["model_fit"]} for r in res.values()]
    g = res["gdina"]["model_fit"]
    lr_tests = []
    for m in ("dina", "dino"):
        f = res[m]["model_fit"]
        df = g["n_parameters"] - f["n_parameters"]
        chi2 = max(0.0, 2 * (g["log_likelihood"] - f["log_likelihood"]))
        lr_tests.append({
            "reduced_model": MODELS[m], "full_model": "G-DINA", "chi_square": chi2, "df": df,
            "p_value": float(sp_stats.chi2.sf(chi2, df)) if df > 0 else None,
        })
    notes = [
        "Likelihood-ratio tests assume regular conditions; when item probabilities sit at their bounds the p-values are approximate.",
    ]
    if all(Q.sum(axis=1) == 1):
        notes.append("Every item measures one attribute, so DINA, DINO and G-DINA are the same model here.")
    return {
        "n_subjects": N,
        "n_items": J,
        "n_attributes": K,
        "fit_comparison": fit_table,
        "best_by_aic": min(fit_table, key=lambda r: r["AIC"])["model"],
        "best_by_bic": min(fit_table, key=lambda r: r["BIC"])["model"],
        "likelihood_ratio_tests": lr_tests,
        "classification_accuracy": {MODELS[m]: r["classification_accuracy"] for m, r in res.items()},
        "notes": notes,
    }


@router.post("/qmatrix-validate")
def qmatrix_validate(req: QMatrixValidateRequest):
    """Empirical Q-matrix validation, PVAF method (de la Torre & Chiu, 2016; GDINA::Qval).

    Using the G-DINA fit, every possible q-vector for an item is scored by the proportion
    of variance accounted for (PVAF): the posterior-weighted variance of the item's success
    probabilities across the classes it distinguishes, relative to the q-vector that uses
    all attributes. The suggested q-vector is the one with the fewest attributes whose PVAF
    reaches `eps` (largest PVAF breaks ties).
    """
    X, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)
    if not 0 < req.eps < 1:
        raise HTTPException(400, "eps must be between 0 and 1.")
    fit = _fit(X, Q, "gdina", req.max_iter, req.tol, req.random_starts, req.seed)
    patterns, post, pi = fit["patterns"], fit["post"], fit["pi"]
    Nc = post.sum(axis=0)
    R1 = post.T @ X
    candidates = _patterns(K)[1:]

    def zeta2(j: int, q: np.ndarray) -> float:
        g, _ = _item_groups(patterns, q, "gdina")
        n_groups = g.max() + 1
        w = np.bincount(g, weights=pi, minlength=n_groups)
        n_g = np.bincount(g, weights=Nc, minlength=n_groups)
        r_g = np.bincount(g, weights=R1[:, j], minlength=n_groups)
        P = np.divide(r_g, n_g, out=np.zeros(n_groups), where=n_g > 0)
        return float(np.sum(w * (P - np.sum(w * P)) ** 2))

    items = []
    Q_suggested = Q.copy()
    for j in range(J):
        full = zeta2(j, np.ones(K, dtype=int))
        pvaf = [zeta2(j, q) / full if full > 0 else 0.0 for q in candidates]
        ok = [i for i, v in enumerate(pvaf) if v >= req.eps]
        best = min(ok, key=lambda i: (candidates[i].sum(), -pvaf[i])) if ok else len(candidates) - 1
        sug = candidates[best]
        Q_suggested[j] = sug
        items.append({
            "item": j + 1,
            "current_q": _pattern_label(Q[j]),
            "suggested_q": _pattern_label(sug),
            "changed": bool(np.any(sug != Q[j])),
            "pvaf": {_pattern_label(q): float(v) for q, v in zip(candidates, pvaf)},
        })

    changed = [it for it in items if it["changed"]]
    return {
        "n_items": J,
        "n_attributes": K,
        "eps": req.eps,
        "original_q_matrix": Q.tolist(),
        "suggested_q_matrix": Q_suggested.tolist(),
        "n_suggested_changes": len(changed),
        "items": items,
        "notes": [
            "Suggestions are empirical and should be checked against the content of each item before the Q-matrix is changed.",
            "This is a single pass on the G-DINA fit to the Q-matrix as given, so an item that is missing an attribute "
            "can still look adequate; refit with any changed q-vectors and validate again.",
        ],
    }
