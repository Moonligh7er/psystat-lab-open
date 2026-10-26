"""Psychometrics — reliability analysis, exploratory factor analysis, IRT.

Reference implementations used for validation:
  * Cronbach's alpha: psych::alpha (raw and standardised alpha, r.drop, alpha if dropped),
    Feldt (1965) confidence interval as psych::alpha.ci.
  * Factor analysis: stats::factanal (maximum likelihood on the correlation matrix,
    Kaiser-normalised varimax / promax(m = 4), sorted and sign-aligned loadings,
    Bartlett-corrected likelihood-ratio test). KMO as psych::KMO, Bartlett's test of
    sphericity as psych::cortest.bartlett.
  * IRT: delegated to app.irt_engine (marginal maximum likelihood, checked against mirt).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import optimize
from scipy import stats as sp_stats

from app.routers.irt_advanced import irt_dichotomous_response

router = APIRouter()


class ReliabilityRequest(BaseModel):
    items: list[list[float]]  # rows = participants, cols = items
    confidence_level: float = 0.95


class FactorAnalysisRequest(BaseModel):
    data: list[list[float]]  # rows = participants, cols = variables
    n_factors: int = 2
    rotation: str = "varimax"  # "varimax", "promax" or "none"


class IRTRequest(BaseModel):
    responses: list[list[int]]  # rows = participants, cols = items (0/1)


class IRT2PLRequest(BaseModel):
    responses: list[list[int]]  # rows = persons, cols = items (0/1)


class IRT3PLRequest(BaseModel):
    responses: list[list[int]]  # rows = persons, cols = items (0/1)
    guessing_prior: float = 0.2  # prior mean of the guessing parameters (e.g. 0.2 for 5 options)


def _matrix(rows, what: str) -> np.ndarray:
    if not rows or len({len(r) for r in rows}) != 1:
        raise HTTPException(422, f"{what} must be a rectangular table (rows of equal length).")
    X = np.array(rows, dtype=float)
    if not np.isfinite(X).all():
        raise HTTPException(422, f"{what} contains missing or non-numeric values; remove incomplete rows first.")
    return X


def _alpha(data: np.ndarray) -> float:
    k = data.shape[1]
    total_var = data.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - data.var(axis=0, ddof=1).sum() / total_var))


def _std_alpha(R: np.ndarray) -> float:
    k = R.shape[0]
    rbar = (R.sum() - k) / (k * (k - 1))
    return float(k * rbar / (1 + (k - 1) * rbar))


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/reliability")
def cronbachs_alpha(req: ReliabilityRequest):
    """Cronbach's alpha with a Feldt confidence interval and item-total statistics."""
    data = _matrix(req.items, "Items")
    n, k = data.shape
    if k < 2:
        raise HTTPException(422, "Need at least 2 items.")
    if n < 3:
        raise HTTPException(422, "Need at least 3 participants.")
    if not 0 < req.confidence_level < 1:
        raise HTTPException(422, "confidence_level must be between 0 and 1.")
    sds = data.std(axis=0, ddof=1)
    if np.any(sds == 0):
        flat = [int(j) + 1 for j in np.where(sds == 0)[0]]
        raise HTTPException(422, f"Item(s) {flat} have zero variance; alpha is undefined with constant items.")
    total = data.sum(axis=1)
    alpha = _alpha(data)
    R = np.corrcoef(data, rowvar=False)
    std_alpha = _std_alpha(R)

    # Feldt (1965): (1 - alpha_pop) / (1 - alpha) ~ F(n - 1, (n - 1)(k - 1))
    q = (1 - req.confidence_level) / 2
    df1, df2 = n - 1, (n - 1) * (k - 1)
    ci = [float(1 - (1 - alpha) * sp_stats.f.ppf(1 - q, df1, df2)),
          float(1 - (1 - alpha) * sp_stats.f.ppf(q, df1, df2))]

    item_stats = []
    for i in range(k):
        rest = np.delete(data, i, axis=1)
        r_drop = np.corrcoef(data[:, i], rest.sum(axis=1))[0, 1]
        item_stats.append({
            "item": i,
            "mean": float(data[:, i].mean()),
            "std": float(sds[i]),
            "item_total_correlation": float(r_drop) if np.isfinite(r_drop) else None,
            "alpha_if_deleted": _alpha(rest) if k > 2 else None,
            "std_alpha_if_deleted": _std_alpha(np.delete(np.delete(R, i, 0), i, 1)) if k > 2 else None,
        })

    return {
        "cronbachs_alpha": alpha,
        "standardized_alpha": std_alpha,
        "alpha_ci": ci,
        "confidence_level": req.confidence_level,
        "ci_method": "Feldt (1965) F-distribution interval",
        "average_inter_item_r": float((R.sum() - k) / (k * (k - 1))),
        "n_items": k,
        "n_participants": n,
        "item_statistics": item_stats,
        "item_total_note": "Corrected item-total correlation: each item against the sum of the other items.",
        "scale_mean": float(total.mean()),
        "scale_std": float(total.std(ddof=1)),
        "interpretation": (
            "Excellent" if alpha >= 0.9 else
            "Good" if alpha >= 0.8 else
            "Acceptable" if alpha >= 0.7 else
            "Questionable" if alpha >= 0.6 else
            "Poor" if alpha >= 0.5 else
            "Unacceptable"
        ),
        "interpretation_note": "Labels follow George & Mallery's widely used rule of thumb; they are conventions, not tests.",
    }


# ── Maximum-likelihood factor analysis (stats::factanal) ───────────────

def _fa_loadings(psi: np.ndarray, S: np.ndarray, q: int) -> np.ndarray:
    sc = 1 / np.sqrt(psi)
    vals, vecs = np.linalg.eigh(S * np.outer(sc, sc))
    vals, vecs = vals[::-1], vecs[:, ::-1]
    L = vecs[:, :q] * np.sqrt(np.maximum(vals[:q] - 1, 0))
    return np.sqrt(psi)[:, None] * L


def _fa_objective(psi: np.ndarray, S: np.ndarray, q: int) -> float:
    sc = 1 / np.sqrt(psi)
    e = np.linalg.eigvalsh(S * np.outer(sc, sc))[::-1][q:]
    return float(-(np.sum(np.log(e) - e) - q + S.shape[0]))


def _fa_gradient(psi: np.ndarray, S: np.ndarray, q: int) -> np.ndarray:
    load = _fa_loadings(psi, S, q)
    g = load @ load.T + np.diag(psi) - S
    return np.diag(g) / psi ** 2


def _varimax(x: np.ndarray, eps: float = 1e-5, max_iter: int = 1000):
    """stats::varimax(normalize = TRUE); returns (loadings, rotation matrix)."""
    p, nc = x.shape
    sc = np.sqrt((x ** 2).sum(axis=1))
    xn = x / sc[:, None]
    TT = np.eye(nc)
    d = 0.0
    for _ in range(max_iter):
        z = xn @ TT
        B = xn.T @ (z ** 3 - z @ np.diag((z ** 2).sum(axis=0)) / p)
        u, sv, vt = np.linalg.svd(B)
        TT = u @ vt
        d_past, d = d, sv.sum()
        if d < d_past * (1 + eps):
            break
    return (xn @ TT) * sc[:, None], TT


def _promax(x: np.ndarray, m: int = 4):
    """stats::promax(m = 4); returns (pattern loadings, rotation matrix)."""
    xv, Tv = _varimax(x)
    Q = xv * np.abs(xv) ** (m - 1)
    U = np.linalg.lstsq(xv, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    return xv @ U, Tv @ U


def _sort_loadings(L: np.ndarray, T: np.ndarray | None):
    """factanal's sortLoadings: order factors by sum of squares, make column sums positive."""
    order = np.argsort(-(L ** 2).sum(axis=0), kind="stable")
    L = L[:, order]
    T = T[:, order] if T is not None else None
    neg = L.sum(axis=0) < 0
    L[:, neg] *= -1
    if T is not None:
        T[:, neg] *= -1
    return L, T


@router.post("/factor-analysis")
def factor_analysis(req: FactorAnalysisRequest):
    """Exploratory maximum-likelihood factor analysis with varimax / promax rotation."""
    data = _matrix(req.data, "Data")
    n, p = data.shape
    q = req.n_factors
    if req.rotation not in ("varimax", "promax", "none"):
        raise HTTPException(422, "rotation must be 'varimax', 'promax' or 'none'.")
    if p < 3:
        raise HTTPException(422, "Need at least 3 variables.")
    if q < 1:
        raise HTTPException(422, "n_factors must be at least 1.")
    dof = ((p - q) ** 2 - p - q) / 2
    if dof < 0:
        max_q = max(f for f in range(1, p) if ((p - f) ** 2 - p - f) >= 0) if p >= 3 else 0
        raise HTTPException(422, f"{q} factors are too many for {p} variables (the model is not identified); "
                                 f"the maximum is {max_q}.")
    if n <= p:
        raise HTTPException(422, "Need more observations than variables.")
    sds = data.std(axis=0, ddof=1)
    if np.any(sds == 0):
        raise HTTPException(422, "Every variable must vary; remove constant columns.")

    S = np.corrcoef(data, rowvar=False)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        raise HTTPException(422, "The correlation matrix is singular (a variable is a linear combination of others).")
    if np.linalg.eigvalsh(S).min() < 1e-10:
        raise HTTPException(422, "The correlation matrix is singular (a variable is a linear combination of others).")

    start = (1 - 0.5 * q / p) / np.diag(S_inv)
    res = optimize.minimize(_fa_objective, start, args=(S, q), jac=_fa_gradient, method="L-BFGS-B",
                            bounds=[(0.005, 1.0)] * p, options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10})
    psi = res.x
    L, _ = _sort_loadings(_fa_loadings(psi, S, q), None)
    T = None
    if q > 1 and req.rotation == "varimax":
        L, T = _varimax(L)
    elif q > 1 and req.rotation == "promax":
        L, T = _promax(L)
    if T is not None:
        L, T = _sort_loadings(L, T)
    factor_corr = None
    if req.rotation == "promax" and T is not None:
        Ti = np.linalg.inv(T)
        factor_corr = (Ti @ Ti.T).tolist()

    ss = (L ** 2).sum(axis=0)
    prop = ss / p
    objective = float(res.fun)
    stat = (n - 1 - (2 * p + 5) / 6 - 2 * q / 3) * objective
    lr = {"chi_square": float(stat), "df": float(dof),
          "p_value": float(sp_stats.chi2.sf(stat, dof)) if dof > 0 else None,
          "hypothesis": f"{q} factor(s) are sufficient"}

    # KMO (psych::KMO) and Bartlett's test of sphericity
    partial = -S_inv / np.sqrt(np.outer(np.diag(S_inv), np.diag(S_inv)))
    r2 = S ** 2
    np.fill_diagonal(r2, 0)
    q2 = partial ** 2
    np.fill_diagonal(q2, 0)
    kmo = float(r2.sum() / (r2.sum() + q2.sum()))
    msa = (r2.sum(axis=0) / (r2.sum(axis=0) + q2.sum(axis=0))).tolist()
    sign, logdet = np.linalg.slogdet(S)
    bart_stat = -(n - 1 - (2 * p + 5) / 6) * logdet
    bart_df = p * (p - 1) / 2

    communalities = 1 - psi
    return {
        "method": "Maximum likelihood (as R's factanal), correlation matrix",
        "loadings": L.tolist(),
        "uniquenesses": psi.tolist(),
        "communalities": communalities.tolist(),
        "ss_loadings": ss.tolist(),
        "variance_explained_pct": (prop * 100).tolist(),
        "cumulative_variance_pct": (np.cumsum(prop) * 100).tolist(),
        "variance_note": ("For promax the sums of squares are of the pattern loadings; with correlated factors they "
                          "do not add up to the total common variance." if req.rotation == "promax" else
                          "Sums of squared loadings of the common factors as a share of the total variance (p variables)."),
        "factor_correlations": factor_corr,
        "fit_test": lr,
        "converged": bool(res.success),
        "heywood_case": bool(np.any(psi <= 0.0051)),
        "correlation_eigenvalues": np.linalg.eigvalsh(S)[::-1].tolist(),
        "eigenvalue_note": "Eigenvalues of the correlation matrix (for scree / Kaiser checks); they are not factor variances.",
        "kmo": kmo,
        "kmo_per_variable": msa,
        "kmo_interpretation": (
            "Marvelous" if kmo >= 0.9 else
            "Meritorious" if kmo >= 0.8 else
            "Middling" if kmo >= 0.7 else
            "Mediocre" if kmo >= 0.6 else
            "Miserable" if kmo >= 0.5 else
            "Unacceptable"
        ),
        "bartlett_sphericity": {"chi_square": float(bart_stat), "df": float(bart_df),
                                "p_value": float(sp_stats.chi2.sf(bart_stat, bart_df))},
        "n_factors": q,
        "n_variables": p,
        "n_observations": n,
        "rotation": req.rotation if q > 1 else "none",
    }


# ── IRT (shared MML engine) ────────────────────────────────────────────

@router.post("/irt")
def item_response_theory(req: IRTRequest):
    """Rasch (1PL) model by marginal maximum likelihood."""
    return irt_dichotomous_response(req.responses, "rasch")


@router.post("/irt-2pl")
def irt_2pl(req: IRT2PLRequest):
    """Two-parameter logistic model by marginal maximum likelihood."""
    return irt_dichotomous_response(req.responses, "2pl")


@router.post("/irt-3pl")
def irt_3pl(req: IRT3PLRequest):
    """Three-parameter logistic model by marginal maximum likelihood with a Beta prior on guessing."""
    return irt_dichotomous_response(req.responses, "3pl", req.guessing_prior)
