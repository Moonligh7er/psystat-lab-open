"""Latent Transition Analysis — latent class models over time for binary indicators.

The model is a hidden Markov model fitted by EM (forward-backward, Baum-Welch) by maximum
likelihood:
  * item-response probabilities are held equal across time by default (measurement
    invariance), so "class k" means the same profile at every time point and transitions are
    interpretable; time-specific item probabilities are available on request;
  * transition probabilities are time-specific (one matrix per pair of adjacent occasions,
    as Mplus) or homogeneous (one matrix, as depmixS4);
  * the mover-stayer model adds a latent "stayer" group whose members keep their class at every
    occasion (Langeheine & van de Pol, 1990), fitted as a 2K-state hidden Markov model.
Several random starts are run and the best log-likelihood is reported together with how many
starts reached it. Reference values: tests/test_lta.py (depmixS4; mover-stayer likelihood
maximised independently in R).
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp_stats
from scipy.special import logsumexp

router = APIRouter()

TINY = 1e-300


class LTAAnalyzeRequest(BaseModel):
    """Latent transition analysis with a given number of classes."""
    indicators: list[list[list[float]]]  # [time_points][participants][indicators], 0/1
    n_classes: int = 2
    measurement_invariance: bool = True
    transitions: str = "time_specific"  # or "homogeneous"
    max_iter: int = 2000
    tol: float = 1e-8
    n_starts: int = 20


class LTACompareRequest(BaseModel):
    indicators: list[list[list[float]]]
    min_classes: int = 2
    max_classes: int = 5
    measurement_invariance: bool = True
    transitions: str = "time_specific"
    max_iter: int = 2000
    tol: float = 1e-8
    n_starts: int = 20


class LTAMoverStayerRequest(BaseModel):
    indicators: list[list[list[float]]]
    n_classes: int = 2
    measurement_invariance: bool = True
    transitions: str = "time_specific"
    max_iter: int = 3000
    tol: float = 1e-8
    n_starts: int = 20


# ── data ──────────────────────────────────────────────────────────────

def _prepare(indicators, n_classes):
    T = len(indicators)
    if T < 2:
        raise HTTPException(422, "Need at least 2 time points.")
    arrays = []
    for t, block in enumerate(indicators):
        if not block or len({len(r) for r in block}) != 1:
            raise HTTPException(422, f"Time point {t + 1} must be a rectangular participants x indicators matrix.")
        arrays.append(np.array(block, dtype=float))
    N, J = arrays[0].shape
    if any(a.shape != (N, J) for a in arrays):
        raise HTTPException(422, "Every time point needs the same participants (rows) and indicators (columns).")
    if not all(np.isin(a, (0, 1)).all() for a in arrays):
        raise HTTPException(422, "Indicators must be coded 0/1 with no missing values.")
    if J < 2:
        raise HTTPException(422, "Need at least 2 indicators.")
    if n_classes < 2:
        raise HTTPException(422, "n_classes must be at least 2.")
    if N < 10 * n_classes:
        raise HTTPException(422, f"Too few participants ({N}) for {n_classes} classes; aim for at least {10 * n_classes}.")
    return np.stack(arrays), N, J, T  # X: (T, N, J)


# ── engine ────────────────────────────────────────────────────────────

def _log_emission(Xt, rho_t):
    """log P(x | state) for binary indicators: (N, S)."""
    lr = np.log(np.clip(rho_t, TINY, 1))
    l1 = np.log(np.clip(1 - rho_t, TINY, 1))
    return Xt @ lr.T + (1 - Xt) @ l1.T


def _fit_hmm(X, K, *, invariant, homogeneous, stayer, max_iter, tol, rng, init=None):
    """EM for an LTA with K classes; with stayer=True a mover-stayer model with 2K states
    (states 0..K-1 movers, K..2K-1 stayers; a stayer keeps its class)."""
    T, N, J = X.shape
    S = 2 * K if stayer else K
    n_tr = 1 if homogeneous else T - 1
    if init is not None:
        rho, pi, tau = init["rho"].copy(), init["pi"].copy(), init["tau"].copy()
    else:
        rho = rng.uniform(0.2, 0.8, size=(1 if invariant else T, K, J))
        pi = rng.dirichlet(np.ones(S))
        tau = np.array([rng.dirichlet(np.ones(K) + 2 * np.eye(K)[k]) for _ in range(n_tr) for k in range(K)]).reshape(n_tr, K, K)

    def trans(t):
        A = np.zeros((S, S))
        A[:K, :K] = tau[0 if homogeneous else t]
        if stayer:
            A[K:, K:] = np.eye(K)
        return A

    def emis(t):
        lr = _log_emission(X[t], rho[0 if invariant else t])
        return np.concatenate([lr, lr], axis=1) if stayer else lr

    prev = -np.inf
    converged = False
    for it in range(max_iter):
        le = [emis(t) for t in range(T)]
        la = [None] * T
        la[0] = le[0] + np.log(np.clip(pi, TINY, 1))
        for t in range(1, T):
            lA = np.log(np.clip(trans(t - 1), TINY, 1))
            la[t] = le[t] + logsumexp(la[t - 1][:, :, None] + lA[None], axis=1)
        ll = float(logsumexp(la[-1], axis=1).sum())
        lb = [None] * T
        lb[-1] = np.zeros((N, S))
        for t in range(T - 2, -1, -1):
            lA = np.log(np.clip(trans(t), TINY, 1))
            lb[t] = logsumexp(lA[None] + (le[t + 1] + lb[t + 1])[:, None, :], axis=2)
        gam = []
        for t in range(T):
            g = la[t] + lb[t]
            gam.append(np.exp(g - logsumexp(g, axis=1, keepdims=True)))
        # M-step
        pi = gam[0].mean(axis=0)
        xi_sum = np.zeros((n_tr, K, K))
        for t in range(T - 1):
            lA = np.log(np.clip(trans(t), TINY, 1))
            lx = la[t][:, :, None] + lA[None] + (le[t + 1] + lb[t + 1])[:, None, :]
            lx -= logsumexp(lx.reshape(N, -1), axis=1)[:, None, None]
            xi_sum[0 if homogeneous else t] += np.exp(lx[:, :K, :K]).sum(axis=0)
        for m in range(n_tr):
            rs = xi_sum[m].sum(axis=1, keepdims=True)
            tau[m] = np.where(rs > 0, xi_sum[m] / np.where(rs > 0, rs, 1), 1.0 / K)
        # class posteriors (movers and stayers with the same class share the profile)
        cls = [g[:, :K] + (g[:, K:] if stayer else 0) for g in gam]
        if invariant:
            w = np.sum([c.sum(axis=0) for c in cls], axis=0)
            num = np.sum([c.T @ X[t] for t, c in enumerate(cls)], axis=0)
            rho[0] = num / np.where(w[:, None] > 0, w[:, None], 1)
        else:
            for t in range(T):
                w = cls[t].sum(axis=0)
                rho[t] = (cls[t].T @ X[t]) / np.where(w[:, None] > 0, w[:, None], 1)
        if abs(ll - prev) < tol * (1 + abs(ll)):
            converged = True
            break
        prev = ll
    npar = (S - 1) + rho.size + n_tr * K * (K - 1)
    return {"ll": ll, "pi": pi, "rho": rho, "tau": tau, "gamma": gam, "npar": npar, "converged": converged,
            "iterations": it + 1}


def _best(X, K, *, invariant, homogeneous, stayer, max_iter, tol, n_starts):
    if not 1 <= n_starts <= 200:
        raise HTTPException(422, "n_starts must be between 1 and 200.")
    runs = []
    for s in range(n_starts):
        rng = np.random.default_rng(1000 + s)
        runs.append(_fit_hmm(X, K, invariant=invariant, homogeneous=homogeneous, stayer=stayer,
                             max_iter=max_iter, tol=tol, rng=rng))
    best = max(runs, key=lambda r: r["ll"])
    replicated = int(sum(abs(r["ll"] - best["ll"]) < 1e-3 * (1 + abs(best["ll"]) / 1000) for r in runs))
    # EM converges slowly near the optimum: continue from the best start with a tight tolerance
    best = _fit_hmm(X, K, invariant=invariant, homogeneous=homogeneous, stayer=stayer, max_iter=200000,
                    tol=1e-13, rng=None, init=best)
    best["starts"] = n_starts
    best["best_ll_replicated"] = replicated
    return best


def _order_classes(res, K):
    """Sort classes by their average item probability (highest first) so labels are reproducible."""
    order = np.argsort(-res["rho"].mean(axis=(0, 2)))
    res["rho"] = res["rho"][:, order]
    res["tau"] = res["tau"][:, order][:, :, order]
    S = len(res["pi"])
    full = np.concatenate([order, order + K]) if S == 2 * K else order
    res["pi"] = res["pi"][full]
    res["gamma"] = [g[:, full] for g in res["gamma"]]
    return res


def _fit_stats(ll, npar, N):
    return {"log_likelihood": float(ll), "n_params": int(npar),
            "aic": float(-2 * ll + 2 * npar), "bic": float(-2 * ll + npar * np.log(N)),
            "sa_bic": float(-2 * ll + npar * np.log((N + 2) / 24))}


def _entropy(gamma, K):
    """Relative entropy (Ramaswamy et al., 1993) of the class posteriors pooled over occasions:
    1 = perfectly separated classes."""
    post = np.concatenate([g[:, :K] + (g[:, K:] if g.shape[1] > K else 0) for g in gamma])
    h = -np.sum(post * np.log(np.clip(post, TINY, 1))) / len(post)
    return float(1 - h / np.log(K))


def _describe(res, K, T, N, transitions):
    rho, tau, pi = res["rho"], res["tau"], res["pi"]
    S = len(pi)
    stayer = S == 2 * K
    p1 = pi[:K] + (pi[K:] if stayer else 0)
    # model-implied class prevalences: movers follow the transitions, stayers keep their class
    prev = [p1]
    for t in range(T - 1):
        moved = pi[:K] @ _cum(tau, t, transitions)
        prev.append(moved + (pi[K:] if stayer else 0))
    gam = res["gamma"]
    cls = [g[:, :K] + (g[:, K:] if stayer else 0) for g in gam]
    modal = [np.bincount(np.argmax(c, axis=1), minlength=K) / N for c in cls]
    profiles = {(f"time_{t}" if rho.shape[0] > 1 else "all_times"): {f"class_{k}": rho[t, k].tolist() for k in range(K)}
                for t in range(rho.shape[0])}
    tm = ({"all_transitions": tau[0].tolist()} if transitions == "homogeneous" else
          {f"time_{t}_to_{t + 1}": tau[t].tolist() for t in range(T - 1)})
    return {
        "class_profiles": profiles,
        "transition_matrices": tm,
        "class_prevalences": {f"time_{t}": prev[t].tolist() for t in range(T)},
        "modal_class_proportions": {f"time_{t}": modal[t].tolist() for t in range(T)},
        "initial_class_probabilities": p1.tolist(),
    }


def _cum(tau, t, transitions):
    """Product of mover transition matrices from occasion 0 to occasion t+1."""
    K = tau.shape[1]
    M = np.eye(K)
    for s in range(t + 1):
        M = M @ tau[0 if transitions == "homogeneous" else s]
    return M


def _check_options(transitions):
    if transitions not in ("time_specific", "homogeneous"):
        raise HTTPException(422, "transitions must be 'time_specific' or 'homogeneous'.")


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/analyze")
def lta_analyze(req: LTAAnalyzeRequest):
    """Latent transition analysis with K classes (EM, best of several random starts)."""
    _check_options(req.transitions)
    X, N, J, T = _prepare(req.indicators, req.n_classes)
    K = req.n_classes
    res = _order_classes(_best(X, K, invariant=req.measurement_invariance, homogeneous=req.transitions == "homogeneous",
                               stayer=False, max_iter=req.max_iter, tol=req.tol, n_starts=req.n_starts), K)
    out = {
        "analysis": "latent_transition_analysis",
        "n_classes": K, "n_timepoints": T, "n_participants": N, "n_indicators": J,
        "measurement_invariance": req.measurement_invariance,
        "transitions": req.transitions,
        **_describe(res, K, T, N, req.transitions),
        "fit": {**_fit_stats(res["ll"], res["npar"], N), "entropy": _entropy(res["gamma"], K)},
        "estimation": {
            "method": "maximum likelihood, EM (forward-backward)",
            "random_starts": res["starts"],
            "best_loglik_replicated": res["best_ll_replicated"],
            "converged": res["converged"], "iterations": res["iterations"],
        },
        "individual_posteriors": {f"time_{t}": res["gamma"][t].tolist() for t in range(T)},
        "notes": [
            "Item-response probabilities are equal across time (measurement invariance), so class k describes the same profile at every occasion."
            if req.measurement_invariance else
            "Item-response probabilities differ by occasion: check that classes keep the same meaning before reading transitions.",
            "Classes are ordered by average item probability (class_0 highest).",
            ("The best log-likelihood was reached by only one start; increase n_starts to make sure it is the global maximum."
             if res["best_ll_replicated"] < 2 else
             f"The best log-likelihood was reached by {res['best_ll_replicated']} of {res['starts']} starts."),
        ],
    }
    return out


@router.post("/compare")
def lta_compare(req: LTACompareRequest):
    """Fit LTAs with min..max classes and compare information criteria."""
    _check_options(req.transitions)
    if req.max_classes < req.min_classes or req.max_classes > 8:
        raise HTTPException(422, "Choose 2 <= min_classes <= max_classes <= 8.")
    X, N, J, T = _prepare(req.indicators, req.min_classes)
    table = []
    for K in range(req.min_classes, req.max_classes + 1):
        if N < 10 * K:
            break
        res = _best(X, K, invariant=req.measurement_invariance, homogeneous=req.transitions == "homogeneous",
                    stayer=False, max_iter=req.max_iter, tol=req.tol, n_starts=req.n_starts)
        table.append({"n_classes": K, **_fit_stats(res["ll"], res["npar"], N), "entropy": _entropy(res["gamma"], K),
                      "best_loglik_replicated": res["best_ll_replicated"], "converged": res["converged"]})
    best_bic = min(table, key=lambda r: r["bic"])["n_classes"] if table else None
    return {
        "analysis": "lta_model_comparison",
        "comparison_table": table,
        "best_by_bic": best_bic,
        "interpretation": ("Lower BIC indicates the better balance of fit and parsimony; BIC is the most reliable of these "
                           "criteria for choosing the number of classes (Nylund et al., 2007). Entropy describes "
                           "classification certainty, not fit."),
    }


@router.post("/mover-stayer")
def lta_mover_stayer(req: LTAMoverStayerRequest):
    """Mover-stayer LTA: a latent group of stayers keeps its class at every occasion; movers
    follow the transition matrices. Compared with the standard LTA by likelihood ratio and BIC."""
    _check_options(req.transitions)
    X, N, J, T = _prepare(req.indicators, req.n_classes)
    K = req.n_classes
    hom = req.transitions == "homogeneous"
    base = _order_classes(_best(X, K, invariant=req.measurement_invariance, homogeneous=hom, stayer=False,
                                max_iter=req.max_iter, tol=req.tol, n_starts=req.n_starts), K)
    ms = _order_classes(_best(X, K, invariant=req.measurement_invariance, homogeneous=hom, stayer=True,
                              max_iter=req.max_iter, tol=req.tol, n_starts=req.n_starts), K)
    lr = max(2 * (ms["ll"] - base["ll"]), 0.0)
    ddf = ms["npar"] - base["npar"]
    pi = ms["pi"]
    stayer_share = pi[K:]
    return {
        "analysis": "mover_stayer_model",
        "n_classes": K, "n_participants": N, "n_timepoints": T,
        "proportion_stayers": float(stayer_share.sum()),
        "stayers_by_class": {f"class_{k}": float(stayer_share[k]) for k in range(K)},
        "movers_by_initial_class": {f"class_{k}": float(pi[k]) for k in range(K)},
        "mover_transition_matrices": ({"all_transitions": ms["tau"][0].tolist()} if hom else
                                      {f"time_{t}_to_{t + 1}": ms["tau"][t].tolist() for t in range(T - 1)}),
        "class_profiles": {f"class_{k}": ms["rho"][0, k].tolist() for k in range(K)} if req.measurement_invariance
        else {f"time_{t}": {f"class_{k}": ms["rho"][t, k].tolist() for k in range(K)} for t in range(T)},
        "standard_lta_fit": _fit_stats(base["ll"], base["npar"], N),
        "mover_stayer_fit": _fit_stats(ms["ll"], ms["npar"], N),
        "likelihood_ratio_test": {
            "chi_square": lr, "df": ddf,
            "p_value_naive": float(sp_stats.chi2.sf(lr, ddf)) if ddf > 0 else None,
            "note": ("The standard LTA is the mover-stayer model with no stayers, a boundary value, so the "
                     "chi-square reference distribution is only approximate (the naive p-value is conservative). "
                     "Prefer the BIC comparison."),
        },
        "preferred_by_bic": "mover_stayer" if _fit_stats(ms["ll"], ms["npar"], N)["bic"] < _fit_stats(base["ll"], base["npar"], N)["bic"] else "standard",
        "estimation": {"random_starts": req.n_starts, "best_loglik_replicated": ms["best_ll_replicated"],
                       "converged": ms["converged"]},
    }
