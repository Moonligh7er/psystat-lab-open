"""Advanced Item Response Theory — Rasch, 2PL, 3PL, GRM, PCM, DIF, CAT, equating."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit  # logistic function

from app.irt_engine import fit_dichotomous, fit_graded, fit_partial_credit

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class RaschRequest(BaseModel):
    responses: list[list[int]]  # persons x items, 0/1
    max_iter: int = 100
    tol: float = 1e-6


class TwoPLRequest(BaseModel):
    responses: list[list[int]]  # persons x items, 0/1
    n_quadrature: int = 41
    max_iter: int = 200
    tol: float = 1e-6


class ThreePLRequest(BaseModel):
    responses: list[list[int]]
    guessing_prior: float = 0.2  # prior mean of the guessing parameters
    n_quadrature: int = 41
    max_iter: int = 300
    tol: float = 1e-6


class GradedResponseRequest(BaseModel):
    responses: list[list[int]]  # persons x items, values 0..K-1
    n_quadrature: int = 41
    max_iter: int = 200
    tol: float = 1e-6


class PartialCreditRequest(BaseModel):
    responses: list[list[int]]  # persons x items, values 0..K-1
    max_iter: int = 100
    tol: float = 1e-6


class DIFRequest(BaseModel):
    responses: list[list[int]]  # persons x items, 0/1
    group: list[int]  # 0 = reference, 1 = focal
    method: str = "all"  # "mh", "lord", "logistic", "all"
    purification: bool = True
    max_purification_iter: int = 10


class CATSimulationRequest(BaseModel):
    item_bank: list[dict]  # each: {"a": float, "b": float, "c": float}
    n_items: int = 20
    start_theta: float = 0.0
    se_threshold: float = 0.3
    stopping: str = "fixed"  # "fixed" or "se"
    true_theta: float = 0.0  # ability of the simulated examinee
    seed: int | None = 42  # random seed for the simulated responses (None = unseeded)


class TestEquatingRequest(BaseModel):
    params_form_x: list[dict]  # each: {"a": float, "b": float} or {"a", "b", "c"}
    params_form_y: list[dict]
    anchor_indices_x: list[int]  # indices of anchor items in form X
    anchor_indices_y: list[int]  # corresponding indices in form Y
    theta_points: list[float] | None = None  # optional evaluation points (equal weights)
    n_quadrature: int = 30  # otherwise: Gauss-Hermite N(0, 1) points on the Form Y scale


# ── Helper functions ──────────────────────────────────────────────────

def _icc_2pl(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """2PL ICC: P(X=1|theta) = 1 / (1 + exp(-a*(theta-b)))."""
    return expit(a * (theta - b))


def _icc_3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """3PL ICC: P = c + (1-c) / (1 + exp(-a*(theta-b)))."""
    return c + (1.0 - c) * expit(a * (theta - b))


def _info_2pl(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """Item information for 2PL: I = a^2 * P * Q."""
    p = _icc_2pl(theta, a, b)
    return a ** 2 * p * (1.0 - p)


def _info_3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """3PL item information: a^2 (1 - c) P*^2 Q* / P, i.e. a^2 (P - c)^2 Q / ((1 - c)^2 P)."""
    p = _icc_3pl(theta, a, b, c)
    p_star = expit(a * (theta - b))
    return a ** 2 * (1.0 - c) * (p_star ** 2) * (1.0 - p_star) / p


def _gauss_hermite_quadrature(n_points: int):
    """Return Gauss-Hermite quadrature nodes and weights for N(0,1)."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    # Transform from Hermite to standard normal
    nodes = nodes * np.sqrt(2)
    weights = weights / np.sqrt(np.pi)
    return nodes, weights


def _safe_float(x) -> float:
    """Convert numpy scalar to Python float, handling NaN/Inf."""
    v = float(x)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


# ── 1. Rasch / 1PL Model ─────────────────────────────────────────────

def _validate_binary(responses):
    if not responses or len({len(r) for r in responses}) != 1:
        raise HTTPException(422, "Responses must be a rectangular persons x items matrix.")
    X = np.array(responses, dtype=float)
    if not np.isin(X, (0, 1)).all():
        raise HTTPException(422, "Dichotomous models need 0/1 responses.")
    n, J = X.shape
    if J < 2 or n < 10:
        raise HTTPException(422, "Need at least 2 items and 10 respondents.")
    pc = X.mean(axis=0)
    flat = [j + 1 for j in range(J) if pc[j] in (0.0, 1.0)]
    if flat:
        raise HTTPException(422, f"Item(s) {flat} were answered the same way by everyone; their parameters cannot be estimated.")
    return X


def _person_fit_stats(X, P_eap):
    """Infit / outfit mean squares per item from the EAP-based expected scores."""
    var = P_eap * (1 - P_eap)
    res2 = (X - P_eap) ** 2
    infit = res2.sum(axis=0) / var.sum(axis=0)
    outfit = (res2 / var).mean(axis=0)
    return infit, outfit


def irt_dichotomous_response(responses, model: str, guessing_prior: float = 0.2) -> dict:
    """Shared response for Rasch / 2PL / 3PL used by /api/irt/* and /api/psychometrics/irt*."""
    X = _validate_binary(responses)
    n, J = X.shape
    if model == "3pl" and n < 100:
        raise HTTPException(422, "The 3PL model needs at least 100 respondents to estimate guessing parameters.")
    if not 0 < guessing_prior < 0.5:
        raise HTTPException(422, "guessing_prior must be between 0 and 0.5.")
    prior = (20 * guessing_prior + 1, 20 * (1 - guessing_prior) + 1)
    r = fit_dichotomous(X, model, prior)
    a, b, c = r["a"], r["b"], r["c"]
    th = r["theta_eap"]
    s = 1 / (1 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    P = c[None, :] + (1 - c[None, :]) * s
    infit, outfit = _person_fit_stats(X, P)
    grid = r["info_grid"]
    curve_idx = np.arange(0, len(grid), 5)  # every 0.5 logits
    items = []
    for j in range(J):
        it = {
            "item": j,
            "difficulty": float(b[j]),
            "se_difficulty": _safe_float(r["se_b"][j]),
            "discrimination": float(a[j]),
            "se_discrimination": None if model == "rasch" else _safe_float(r["se_a"][j]),
            "guessing": float(c[j]) if model == "3pl" else None,
            "se_guessing": _safe_float(r["se_c"][j]) if model == "3pl" else None,
            "proportion_correct": float(X[:, j].mean()),
            "information": float(r["item_information"][:, j].max()),
            "information_peak_theta": float(grid[int(np.argmax(r["item_information"][:, j]))]),
            "infit": float(infit[j]),
            "outfit": float(outfit[j]),
        }
        if model == "rasch":
            it["fit_flag"] = "misfit" if not (0.7 <= infit[j] <= 1.3 and 0.7 <= outfit[j] <= 1.3) else "ok"
        items.append(it)
    names = {"rasch": "Rasch (1PL)", "2pl": "Two-parameter logistic (2PL)", "3pl": "Three-parameter logistic (3PL)"}
    out = {
        "model": names[model],
        "estimation": ("Marginal maximum likelihood (61-point Gauss-Hermite quadrature); EAP person scores"
                       + ("; guessing with a Beta(%g, %g) prior, mode = guessing_prior (posterior mode)" % prior if model == "3pl" else "")),
        "item_parameters": items,
        "information_curve": [{"theta": float(grid[i]), "information": float(r["test_information"][i])} for i in curve_idx],
        "max_info": float(r["test_information"].max()),
        "reliability": r["empirical_reliability"],
        "reliability_type": "empirical (EAP)",
        "log_likelihood": r["log_likelihood"],
        "log_prior": r["log_prior"],
        "n_parameters": r["n_parameters"],
        "aic": r["aic"],
        "bic": r["bic"],
        "fit_indices": {"log_likelihood": r["log_likelihood"], "aic": r["aic"], "bic": r["bic"]},
        "person_ability_summary": {"mean": float(th.mean()), "std": float(th.std(ddof=1)),
                                   "min": float(th.min()), "max": float(th.max())},
        "person_abilities": [{"theta": float(t), "se": float(e)} for t, e in zip(th, r["theta_psd"])],
        "n_persons": int(n),
        "n_items": int(J),
        "converged": r["converged"],
    }
    if model == "rasch":
        out["latent_sd"] = r["latent_sd"]
        out["fit_note"] = "Infit/outfit mean squares from EAP abilities; 0.7-1.3 treated as acceptable."
    out["interpretation"] = (f"{names[model]} fitted to {n} respondents and {J} items by MML. "
                             f"Empirical reliability = {r['empirical_reliability']:.3f}.")
    return out


@router.post("/rasch")
def rasch_model(req: RaschRequest):
    """Rasch model by marginal maximum likelihood (item slopes fixed at 1, latent SD estimated)."""
    return irt_dichotomous_response(req.responses, "rasch")


@router.post("/two-pl")
def two_pl_model(req: TwoPLRequest):
    """Two-parameter logistic model by marginal maximum likelihood."""
    return irt_dichotomous_response(req.responses, "2pl")


@router.post("/three-pl")
def three_pl_model(req: ThreePLRequest):
    """Three-parameter logistic model by marginal maximum likelihood, with a Beta prior on guessing."""
    return irt_dichotomous_response(req.responses, "3pl", req.guessing_prior)


@router.post("/graded-response")
def graded_response_model(req: GradedResponseRequest):
    """Samejima's graded response model by marginal maximum likelihood."""
    X = _validate_polytomous(req.responses)
    n, J = X.shape
    r = fit_graded(X)
    grid = r["info_grid"]
    curve_idx = np.arange(0, len(grid), 5)
    th = r["theta_eap"]
    items = [{
        "item": j,
        "discrimination": it["a"],
        "se_discrimination": _safe_float(it["se_a"]),
        "thresholds": it["b"],
        "difficulty": float(np.mean(it["b"])),
        "n_categories": len(it["b"]) + 1,
        "information": float(r["item_information"][:, j].max()),
    } for j, it in enumerate(r["items"])]
    return {
        "model": "Graded response model (Samejima)",
        "estimation": "Marginal maximum likelihood (61-point Gauss-Hermite quadrature); EAP person scores",
        "item_parameters": items,
        "information_curve": [{"theta": float(grid[i]), "information": float(r["test_information"][i])} for i in curve_idx],
        "max_info": float(r["test_information"].max()),
        "reliability": r["empirical_reliability"],
        "reliability_type": "empirical (EAP)",
        "log_likelihood": r["log_likelihood"],
        "n_parameters": r["n_parameters"],
        "aic": r["aic"],
        "bic": r["bic"],
        "person_ability_summary": {"mean": float(th.mean()), "std": float(th.std(ddof=1)),
                                   "min": float(th.min()), "max": float(th.max())},
        "n_persons": int(n),
        "n_items": int(J),
        "converged": r["converged"],
        "note": "'difficulty' is the mean of an item's category thresholds.",
    }


def _validate_polytomous(responses, min_persons=20):
    if not responses or len({len(r) for r in responses}) != 1:
        raise HTTPException(422, "Responses must be a rectangular persons x items matrix.")
    X = np.array(responses, dtype=float)
    if np.any(X < 0) or np.any(X != np.round(X)):
        raise HTTPException(422, "Responses must be integer categories starting at 0.")
    X = X.astype(int)
    n, J = X.shape
    if J < 2 or n < min_persons:
        raise HTTPException(422, f"Need at least 2 items and {min_persons} respondents.")
    for j in range(J):
        used = np.unique(X[:, j])
        if len(used) < 2 or not np.array_equal(used, np.arange(used.max() + 1)):
            raise HTTPException(422, f"Item {j + 1}: categories must run 0, 1, 2, ... with every category observed.")
    return X


@router.post("/partial-credit")
def partial_credit_model(req: PartialCreditRequest):
    """Masters' partial credit model by marginal maximum likelihood (latent SD estimated)."""
    X = _validate_polytomous(req.responses)
    r = fit_partial_credit(X)
    grid = r["info_grid"]
    th = r["theta_eap"]
    J = X.shape[1]
    return {
        "model": "Partial credit model (Masters)",
        "estimation": "Marginal maximum likelihood (61-point Gauss-Hermite quadrature); EAP person scores",
        "item_parameters": [
            {
                "item": int(j),
                "step_parameters": it["steps"],
                "se_step_parameters": [_safe_float(v) for v in it["se_steps"]],
                "difficulty": float(np.mean(it["steps"])),
                "n_categories": len(it["steps"]) + 1,
                "infit": r["infit"][j],
                "outfit": r["outfit"][j],
                "information": float(r["item_information"][:, j].max()),
            }
            for j, it in enumerate(r["items"])
        ],
        "category_probability_curves": {
            "theta": grid.tolist(),
            "curves": {f"item_{j}": {f"category_{k}": r["category_curves"][j][:, k].tolist()
                                     for k in range(r["category_curves"][j].shape[1])} for j in range(J)},
        },
        "item_information": {"theta": grid.tolist(),
                             "curves": {f"item_{j}": r["item_information"][:, j].tolist() for j in range(J)}},
        "test_information": {"theta": grid.tolist(), "information": r["test_information"].tolist()},
        "information_curve": [{"theta": float(grid[i]), "information": float(r["test_information"][i])}
                              for i in range(0, len(grid), 5)],
        "max_info": float(r["test_information"].max()),
        "latent_sd": r["latent_sd"],
        "reliability": r["empirical_reliability"],
        "reliability_type": "empirical (EAP)",
        "log_likelihood": r["log_likelihood"],
        "n_parameters": r["n_parameters"],
        "aic": r["aic"],
        "bic": r["bic"],
        "person_ability_summary": {"mean": float(th.mean()), "std": float(th.std(ddof=1)),
                                   "min": float(th.min()), "max": float(th.max())},
        "converged": r["converged"],
        "n_iterations": r["n_iterations"],
        "fit_note": "Infit/outfit mean squares from EAP abilities (as mirt::itemfit, method = 'EAP').",
    }


# ── 6. Differential Item Functioning ─────────────────────────────────

def _mh_item(X, i, score, ref, foc):
    """Mantel-Haenszel statistics for item i, stratified on `score` (difR::difMH, correct = TRUE)."""
    num = den = dev = var = 0.0
    for s in np.unique(score):
        m = score == s
        nr, nf = int((m & ref).sum()), int((m & foc).sum())
        if nr == 0 or nf == 0:
            continue
        N = nr + nf
        A = X[m & ref, i].sum()
        B = nr - A
        C = X[m & foc, i].sum()
        D = nf - C
        num += A * D / N
        den += B * C / N
        dev += A - nr * (A + C) / N
        var += nr * nf * (A + C) * (B + D) / (N ** 2 * (N - 1))
    alpha = num / den if den > 0 else np.nan
    chi2 = (abs(dev) - 0.5) ** 2 / var if var > 0 else np.nan
    delta = -2.35 * np.log(alpha) if alpha and np.isfinite(alpha) and alpha > 0 else np.nan
    return chi2, alpha, delta


def _logit_fit(Xd, y, max_iter=100):
    """Logistic regression by Newton-Raphson; returns (deviance, coefficients, converged)."""
    beta = np.zeros(Xd.shape[1])
    dev_old = np.inf
    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -30, 30)
        p = expit(eta)
        W = p * (1 - p)
        H = Xd.T @ (Xd * W[:, None])
        try:
            beta = beta + np.linalg.solve(H, Xd.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        p = np.clip(expit(np.clip(Xd @ beta, -30, 30)), 1e-15, 1 - 1e-15)
        dev = float(-2 * (y * np.log(p) + (1 - y) * np.log1p(-p)).sum())
        if abs(dev - dev_old) < 1e-10 * (abs(dev) + 1):
            return dev, beta, True
        dev_old = dev
    return dev_old, beta, False


def _logistic_item(X, i, score, g):
    """Swaminathan-Rogers logistic DIF (difR::difLogistic, type = 'both', criterion = 'LRT')."""
    y = X[:, i]
    n = len(y)
    one = np.ones(n)
    p0 = np.clip(y.mean(), 1e-15, 1 - 1e-15)
    null_dev = float(-2 * n * (p0 * np.log(p0) + (1 - p0) * np.log1p(-p0)))
    d1, _, c1 = _logit_fit(np.column_stack([one, score]), y)
    d2, b2, c2 = _logit_fit(np.column_stack([one, score, g]), y)
    d3, b3, c3 = _logit_fit(np.column_stack([one, score, g, score * g]), y)

    def r2(dev):  # Nagelkerke R^2
        return (1 - np.exp(-null_dev / n + dev / n)) / (1 - np.exp(-null_dev / n))
    chi_u, chi_nu, chi_all = max(d1 - d2, 0.0), max(d2 - d3, 0.0), max(d1 - d3, 0.0)
    dr2 = float(r2(d3) - r2(d1))
    return {
        "uniform_dif": {"chi_square": chi_u, "df": 1, "p_value": float(sp_stats.chi2.sf(chi_u, 1)),
                        "group_coefficient": _safe_float(b2[2])},
        "non_uniform_dif": {"chi_square": chi_nu, "df": 1, "p_value": float(sp_stats.chi2.sf(chi_nu, 1)),
                            "interaction_coefficient": _safe_float(b3[3])},
        "overall_dif": {"chi_square": chi_all, "df": 2, "p_value": float(sp_stats.chi2.sf(chi_all, 2))},
        "delta_nagelkerke_r2": dr2,
        "effect_size_zumbo_thomas": "A" if dr2 < 0.13 else ("B" if dr2 <= 0.26 else "C"),
        "effect_size_jodoin_gierl": "A" if dr2 < 0.035 else ("B" if dr2 <= 0.07 else "C"),
        "converged": bool(c1 and c2 and c3),
    }


def _lord(fit_r, fit_f, anchor):
    """Lord's chi-square for the 2PL (difR::difLord): focal parameters put on the reference
    scale by mean-sigma on the anchor difficulties, then a 2-df Wald test per item."""
    bR, bF = fit_r["b"], fit_f["b"]
    A = np.std(bR[anchor], ddof=1) / np.std(bF[anchor], ddof=1)
    B = bR[anchor].mean() - A * bF[anchor].mean()
    aF, bFn = fit_f["a"] / A, A * bF + B
    seaF, sebF = fit_f["se_a"] / A, fit_f["se_b"] * A
    out = []
    for j in range(len(bR)):
        S = np.array([[fit_r["se_a"][j] ** 2 + seaF[j] ** 2, fit_r["cov_ab"][j] + fit_f["cov_ab"][j]],
                      [fit_r["cov_ab"][j] + fit_f["cov_ab"][j], fit_r["se_b"][j] ** 2 + sebF[j] ** 2]])
        d = np.array([fit_r["a"][j] - aF[j], bR[j] - bFn[j]])
        try:
            chi2 = float(d @ np.linalg.solve(S, d))
        except np.linalg.LinAlgError:
            chi2 = float("nan")
        out.append({"chi_square": _safe_float(chi2), "df": 2,
                    "p_value": _safe_float(sp_stats.chi2.sf(chi2, 2)) if np.isfinite(chi2) else None,
                    "a_reference": float(fit_r["a"][j]), "b_reference": float(bR[j]),
                    "a_focal_rescaled": float(aF[j]), "b_focal_rescaled": float(bFn[j])})
    return out, {"A": float(A), "B": float(B)}


def _purify(stat_fn, n_items, alpha, purify, max_iter):
    """difR-style purification: re-run with flagged items removed from the anchor until the
    flagged set stops changing. stat_fn(anchor) -> list of p-values."""
    anchor = list(range(n_items))
    pvals = stat_fn(anchor)
    flagged = [j for j, p in enumerate(pvals) if p is not None and p < alpha]
    it, converged = 0, True
    if purify and flagged:
        converged = False
        for it in range(1, max_iter + 1):
            anchor = [j for j in range(n_items) if j not in flagged]
            if len(anchor) < 2:
                break
            pvals = stat_fn(anchor)
            new = [j for j, p in enumerate(pvals) if p is not None and p < alpha]
            if new == flagged:
                converged = True
                break
            flagged = new
    return flagged, anchor, it, converged


@router.post("/dif")
def differential_item_functioning(req: DIFRequest):
    """DIF: Mantel-Haenszel, logistic regression and Lord's chi-square (as difR)."""
    X = _validate_binary(req.responses)
    group = np.array(req.group)
    n_persons, n_items = X.shape
    if len(group) != n_persons:
        raise HTTPException(422, f"The group vector has {len(group)} entries but there are {n_persons} respondents.")
    if not set(np.unique(group).tolist()) <= {0, 1}:
        raise HTTPException(422, "Group must contain only 0 (reference) and 1 (focal).")
    if req.method not in ("mh", "logistic", "lord", "all"):
        raise HTTPException(422, "method must be 'mh', 'logistic', 'lord' or 'all'.")
    ref, foc = group == 0, group == 1
    if ref.sum() < 5 or foc.sum() < 5:
        raise HTTPException(422, "Each group needs at least 5 respondents.")
    alpha_level = 0.05
    g = foc.astype(float)
    methods = ["mh", "logistic", "lord"] if req.method == "all" else [req.method]
    notes = []
    if "lord" in methods and min(ref.sum(), foc.sum()) < 50:
        if req.method == "lord":
            raise HTTPException(422, "Lord's chi-square fits a 2PL in each group and needs at least 50 respondents per group.")
        methods.remove("lord")
        notes.append("Lord's chi-square skipped: it fits a 2PL in each group and needs at least 50 respondents per group.")
    elif "lord" in methods and min(ref.sum(), foc.sum()) < 200:
        notes.append("Lord's chi-square: a group has fewer than 200 respondents, so its 2PL estimates and the test are imprecise.")

    def matching(anchor, i):
        cols = sorted(set(anchor) | {i})  # the studied item is always part of the matching score
        return X[:, cols].sum(axis=1)

    out_items = [{"item": j} for j in range(n_items)]
    summary_flags = {}
    purification = {}

    if "mh" in methods:
        cache = {}

        def mh_stats(anchor):
            res = [_mh_item(X, i, matching(anchor, i), ref, foc) for i in range(n_items)]
            cache["res"] = res
            return [float(sp_stats.chi2.sf(c, 1)) if np.isfinite(c) else None for c, _, _ in res]
        flagged, anchor, it, conv = _purify(mh_stats, n_items, alpha_level, req.purification, req.max_purification_iter)
        for i, (chi2, a_mh, delta) in enumerate(cache["res"]):
            ad = abs(delta) if np.isfinite(delta) else np.nan
            out_items[i]["mantel_haenszel"] = {
                "chi_square": _safe_float(chi2),
                "p_value": _safe_float(sp_stats.chi2.sf(chi2, 1)) if np.isfinite(chi2) else None,
                "odds_ratio": _safe_float(a_mh),
                "delta_mh": _safe_float(delta),
                "ets_classification": None if not np.isfinite(ad) else ("A" if ad < 1 else ("B" if ad < 1.5 else "C")),
                "flagged": i in flagged,
            }
        summary_flags["mh"] = flagged
        purification["mh"] = {"iterations": it, "converged": conv, "anchor_items": anchor}

    if "logistic" in methods:
        cache = {}

        def lr_stats(anchor):
            res = [_logistic_item(X, i, matching(anchor, i).astype(float), g) for i in range(n_items)]
            cache["res"] = res
            return [r["overall_dif"]["p_value"] for r in res]
        flagged, anchor, it, conv = _purify(lr_stats, n_items, alpha_level, req.purification, req.max_purification_iter)
        for i, r in enumerate(cache["res"]):
            r["flagged"] = i in flagged
            out_items[i]["logistic_regression"] = r
        summary_flags["logistic"] = flagged
        purification["logistic"] = {"iterations": it, "converged": conv, "anchor_items": anchor}

    if "lord" in methods:
        fr = fit_dichotomous(X[ref], "2pl")
        ff = fit_dichotomous(X[foc], "2pl")
        cache = {}

        def lord_stats(anchor):
            res, link = _lord(fr, ff, np.array(anchor))
            cache["res"], cache["link"] = res, link
            return [r["p_value"] for r in res]
        flagged, anchor, it, conv = _purify(lord_stats, n_items, alpha_level, req.purification, req.max_purification_iter)
        for i, r in enumerate(cache["res"]):
            r["flagged"] = i in flagged
            out_items[i]["lord"] = r
        summary_flags["lord"] = flagged
        purification["lord"] = {"iterations": it, "converged": conv, "anchor_items": anchor,
                                "linking": cache["link"]}

    return {
        "items": out_items,
        "summary": {
            "methods": methods,
            "total_items": int(n_items),
            "flagged_items": summary_flags,
            "alpha": alpha_level,
            "purification": req.purification,
        },
        "purification": purification,
        "notes": notes + [
            "Items are flagged when p < .05 (MH chi-square with continuity correction; logistic 2-df LR test of "
            "uniform + non-uniform DIF; Lord's 2-df Wald test). No multiple-comparison adjustment is applied.",
            "ETS A/B/C uses |delta_MH| < 1, 1-1.5, >= 1.5 (magnitude only, as difR); logistic effect sizes use "
            "Nagelkerke delta R^2 with the Zumbo-Thomas and Jodoin-Gierl cut-offs.",
        ],
        "n_persons": int(n_persons),
        "n_items": int(n_items),
        "n_reference": int(ref.sum()),
        "n_focal": int(foc.sum()),
    }


# ── 7. Computerized Adaptive Testing Simulation ──────────────────────

@router.post("/cat-simulation")
def cat_simulation(req: CATSimulationRequest):
    """Simulate a computerized adaptive test: maximum-information item selection, EAP scoring
    with a N(0, 1) prior, responses drawn from the 3PL at true_theta."""
    n_bank = len(req.item_bank)
    if req.n_items < 1:
        raise HTTPException(422, "n_items must be at least 1.")
    if n_bank < req.n_items:
        raise HTTPException(422, f"The item bank ({n_bank}) must have at least as many items as n_items ({req.n_items}).")
    if req.stopping not in ("fixed", "se"):
        raise HTTPException(422, "stopping must be 'fixed' or 'se'.")
    try:
        a_bank = np.array([float(it.get("a", 1.0)) for it in req.item_bank])
        b_bank = np.array([float(it.get("b", 0.0)) for it in req.item_bank])
        c_bank = np.array([float(it.get("c", 0.0)) for it in req.item_bank])
    except (TypeError, ValueError):
        raise HTTPException(422, "Item bank entries need numeric a, b and (optionally) c.")
    if np.any(a_bank <= 0) or np.any((c_bank < 0) | (c_bank >= 1)):
        raise HTTPException(422, "Each item needs a > 0 and 0 <= c < 1.")

    rng = np.random.default_rng(req.seed)
    nodes = np.linspace(-4, 4, 81)
    log_post = sp_stats.norm.logpdf(nodes)
    theta, se = req.start_theta, None
    administered, responses, theta_hist, se_hist = [], [], [], []
    available = np.ones(n_bank, dtype=bool)
    for _ in range(req.n_items):
        info = _info_3pl(np.full(n_bank, theta), a_bank, b_bank, c_bank)
        info[~available] = -np.inf
        j = int(np.argmax(info))
        available[j] = False
        p_true = float(_icc_3pl(np.array([req.true_theta]), a_bank[j], b_bank[j], c_bank[j])[0])
        u = int(rng.random() < p_true)
        administered.append(j)
        responses.append(u)
        pj = np.clip(_icc_3pl(nodes, a_bank[j], b_bank[j], c_bank[j]), 1e-15, 1 - 1e-15)
        log_post = log_post + (np.log(pj) if u else np.log1p(-pj))
        w = np.exp(log_post - log_post.max())
        w /= w.sum()
        theta = float(w @ nodes)
        se = float(np.sqrt(max(w @ nodes ** 2 - theta ** 2, 0.0)))
        theta_hist.append(theta)
        se_hist.append(se)
        if req.stopping == "se" and se <= req.se_threshold:
            break

    cat_info = float(_info_3pl(np.full(len(administered), theta), a_bank[administered],
                               b_bank[administered], c_bank[administered]).sum())
    bank_info = float(_info_3pl(np.full(n_bank, theta), a_bank, b_bank, c_bank).sum())
    return {
        "administered_items": administered,
        "responses": responses,
        "theta_after_each_item": theta_hist,
        "se_after_each_item": se_hist,
        "final_theta": theta,
        "final_se": se,
        "n_items_administered": len(administered),
        "test_information_at_final_theta": cat_info,
        "efficiency_vs_full_test": {
            "cat_information": cat_info,
            "full_bank_information": bank_info,
            "efficiency_ratio": cat_info / bank_info if bank_info > 0 else None,
            "items_used_fraction": len(administered) / n_bank,
        },
        "true_theta": req.true_theta,
        "seed": req.seed,
        "stopping_criterion": req.stopping,
        "method": ("Maximum Fisher information selection at the current estimate; EAP scoring with a N(0, 1) "
                   "prior on an 81-point grid over [-4, 4]; responses simulated from the 3PL at true_theta."),
    }


# ── 8. Test Equating via IRT ─────────────────────────────────────────

def _item_params(items, idx, what):
    try:
        return (np.array([float(items[i].get("a", 1.0)) for i in idx]),
                np.array([float(items[i].get("b", 0.0)) for i in idx]),
                np.array([float(items[i].get("c", 0.0)) for i in idx]))
    except IndexError:
        raise HTTPException(422, f"An anchor index is outside {what}.")
    except (TypeError, ValueError):
        raise HTTPException(422, f"{what} items need numeric a, b and (optionally) c.")


@router.post("/test-equating")
def test_equating(req: TestEquatingRequest):
    """IRT scale linking of Form X onto Form Y with common items (Kolen & Brennan, 2014, ch. 6):
    mean-mean, mean-sigma, Haebara and Stocking-Lord, as equateIRT::direc (D = 1)."""
    ax_, ay_ = req.anchor_indices_x, req.anchor_indices_y
    if len(ax_) != len(ay_):
        raise HTTPException(422, "Anchor index lists must have the same length.")
    if len(ax_) < 2:
        raise HTTPException(422, "Need at least 2 anchor items.")
    a_x, b_x, c_x = _item_params(req.params_form_x, ax_, "Form X")
    a_y, b_y, c_y = _item_params(req.params_form_y, ay_, "Form Y")
    if np.any(a_x <= 0) or np.any(a_y <= 0):
        raise HTTPException(422, "Discriminations must be positive.")

    # theta_Y = A theta_X + B;  b_Y = A b_X + B;  a_Y = a_X / A
    A_mm = a_x.mean() / a_y.mean()
    B_mm = b_y.mean() - A_mm * b_x.mean()
    sd_bx = b_x.std(ddof=1)
    A_ms = b_y.std(ddof=1) / sd_bx if sd_bx > 0 else np.nan
    B_ms = b_y.mean() - A_ms * b_x.mean() if np.isfinite(A_ms) else np.nan

    if req.theta_points:
        theta = np.array(req.theta_points, dtype=float)
        wt = np.ones(len(theta))
        points = "user-supplied points, equal weights"
    else:
        if not 5 <= req.n_quadrature <= 200:
            raise HTTPException(422, "n_quadrature must be between 5 and 200.")
        x, w = np.polynomial.hermite.hermgauss(req.n_quadrature)
        theta, wt = x * np.sqrt(2.0), w / np.sqrt(np.pi)
        points = f"{req.n_quadrature}-point Gauss-Hermite quadrature for N(0, 1) on the Form Y scale"
    P_y = c_y[None, :] + (1 - c_y[None, :]) * expit(a_y[None, :] * (theta[:, None] - b_y[None, :]))

    def criterion(par, kind):
        A, B = par
        if A <= 0:
            return 1e10
        P_x = c_x[None, :] + (1 - c_x[None, :]) * expit((a_x / A)[None, :] * (theta[:, None] - (A * b_x + B)[None, :]))
        if kind == "haebara":
            return 0.5 * float((((P_y - P_x) ** 2).sum(axis=1) * wt).sum())
        return 0.5 * float((((P_y.sum(axis=1) - P_x.sum(axis=1)) ** 2) * wt).sum())

    chars = {}
    for kind in ("haebara", "stocking_lord"):
        res = minimize(criterion, x0=[A_mm, B_mm], args=(kind,), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000})
        chars[kind] = {"A": float(res.x[0]), "B": float(res.x[1]), "converged": bool(res.success),
                       "objective_value": float(res.fun)}
    A_sl, B_sl = chars["stocking_lord"]["A"], chars["stocking_lord"]["B"]

    equated = []
    for it in req.params_form_x:
        a0, b0, c0 = float(it.get("a", 1.0)), float(it.get("b", 0.0)), float(it.get("c", 0.0))
        equated.append({"a": a0 / A_sl, "b": A_sl * b0 + B_sl, "c": c0})
    return {
        "stocking_lord": chars["stocking_lord"],
        "haebara": chars["haebara"],
        "mean_mean": {"A": float(A_mm), "B": float(B_mm)},
        "mean_sigma": {"A": _safe_float(A_ms), "B": _safe_float(B_ms)},
        "equated_parameters": equated,
        "equated_with": "stocking_lord",
        "concordance_table": [{"theta_x": float(t), "theta_y": float(A_sl * t + B_sl)} for t in np.linspace(-4, 4, 41)],
        "transformation": "theta_Y = A * theta_X + B; b* = A * b + B; a* = a / A; c unchanged",
        "criterion_points": points,
        "n_anchor_items": len(ax_),
        "n_form_x_items": len(req.params_form_x),
        "n_form_y_items": len(req.params_form_y),
    }
