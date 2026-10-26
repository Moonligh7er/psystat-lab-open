"""Spatial Statistics — Moran's I, LISA, spatial lag regression, and GWR.

Global and local spatial autocorrelation, Lagrange-multiplier diagnostics, the
maximum-likelihood spatial lag (SAR) model, and geographically weighted regression.
Reference values in tests/test_spatial.py come from R spdep 1.4 / spatialreg 1.4
(moran.test, localmoran, lm.RStests, lagsarlm with method = "eigen").
"""

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class MoransIRequest(BaseModel):
    """Global Moran's I test for spatial autocorrelation."""
    values: list[float]
    coordinates: list[list[float]]  # [[x, y], ...] per observation
    weight_matrix: list[list[float]] | None = None  # optional pre-built W
    weight_type: str = "knn"  # knn or distance_band
    k: int = 5  # k for knn
    distance_band: float | None = None  # threshold for distance_band
    n_permutations: int = 999
    row_standardize: bool = True


class LocalMoransRequest(BaseModel):
    """Local Indicators of Spatial Association (LISA)."""
    values: list[float]
    coordinates: list[list[float]]
    weight_matrix: list[list[float]] | None = None
    weight_type: str = "knn"
    k: int = 5
    distance_band: float | None = None
    n_permutations: int = 999
    row_standardize: bool = True
    alpha: float = 0.05


class SpatialRegressionRequest(BaseModel):
    """OLS with spatial diagnostics and the ML spatial lag (SAR) model."""
    y: list[float]
    X: list[list[float]]  # n x p design matrix (without intercept)
    coordinates: list[list[float]]
    weight_matrix: list[list[float]] | None = None
    weight_type: str = "knn"
    k: int = 5
    distance_band: float | None = None
    row_standardize: bool = True
    variable_names: list[str] | None = None


class GWRRequest(BaseModel):
    """Geographically Weighted Regression."""
    y: list[float]
    X: list[list[float]]  # n x p design matrix (without intercept)
    coordinates: list[list[float]]
    bandwidth: float | None = None  # None = choose by leave-one-out CV
    kernel: str = "gaussian"  # gaussian or bisquare
    variable_names: list[str] | None = None


# ── Shared helpers ────────────────────────────────────────────────────

def _coords(coordinates, n):
    c = np.asarray(coordinates, dtype=float)
    if c.ndim != 2 or c.shape != (n, 2):
        raise HTTPException(422, f"coordinates must be {n} [x, y] pairs (one per observation).")
    if not np.all(np.isfinite(c)):
        raise HTTPException(422, "Coordinates must be finite numbers.")
    return c


def _finite(a, name):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise HTTPException(422, f"{name} must be finite numbers.")
    return a


def _build_weight_matrix(coords: np.ndarray, weight_type: str, k: int = 5,
                         distance_band: float | None = None, row_standardize: bool = True) -> np.ndarray:
    """Spatial weights from coordinates: k nearest neighbours (as spdep::knearneigh) or a distance band."""
    n = coords.shape[0]
    D = cdist(coords, coords)
    W = np.zeros((n, n))
    if weight_type == "knn":
        if not 1 <= k < n:
            raise HTTPException(422, f"k must be between 1 and {n - 1}.")
        for i in range(n):
            dists = D[i].copy()
            dists[i] = np.inf
            W[i, np.argsort(dists, kind="stable")[:k]] = 1.0
    elif weight_type == "distance_band":
        if distance_band is None:
            distance_band = float(np.median(D[np.triu_indices(n, k=1)]))
        W = ((D <= distance_band) & ~np.eye(n, dtype=bool)).astype(float)
        isolated = np.where(W.sum(axis=1) == 0)[0]
        if len(isolated):
            raise HTTPException(422, f"{len(isolated)} observation(s) have no neighbour within the distance band; widen it.")
    else:
        raise HTTPException(422, f"Unknown weight_type '{weight_type}'. Use 'knn' or 'distance_band'.")
    return _standardize(W) if row_standardize else W


def _standardize(W):
    rs = W.sum(axis=1, keepdims=True)
    return W / np.where(rs == 0, 1, rs)


def _weights(req, n, coords):
    if req.weight_matrix is not None:
        W = np.asarray(req.weight_matrix, dtype=float)
        if W.shape != (n, n):
            raise HTTPException(422, f"weight_matrix must be {n}x{n}.")
        if not np.all(np.isfinite(W)) or np.any(W < 0) or np.any(np.diag(W) != 0):
            raise HTTPException(422, "weight_matrix must be non-negative with a zero diagonal.")
        return _standardize(W) if req.row_standardize else W
    return _build_weight_matrix(coords, req.weight_type, req.k, req.distance_band, req.row_standardize)


def _check_perms(m):
    if not 99 <= m <= 9999:
        raise HTTPException(422, "n_permutations must be between 99 and 9,999.")


# ── Moran's I ─────────────────────────────────────────────────────────

def _moran_moments(z, W):
    """E[I] and Var[I] under randomisation (Cliff & Ord, 1981), as spdep::moran.test."""
    n = len(z)
    S0 = W.sum()
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)
    EI = -1.0 / (n - 1)
    b2 = n * np.sum(z ** 4) / np.sum(z ** 2) ** 2
    num = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) - b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    EI2 = num / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)
    return EI, EI2 - EI ** 2


@router.post("/morans-i")
def global_morans_i(req: MoransIRequest):
    """Global Moran's I with the randomisation-moments test (as spdep::moran.test) and a
    permutation test (pseudo p = (b + 1) / (m + 1), one-sided in the direction of I)."""
    values = _finite(req.values, "values")
    n = len(values)
    if n < 4:
        raise HTTPException(422, "Need at least 4 observations.")
    coords = _coords(req.coordinates, n)
    _check_perms(req.n_permutations)
    W = _weights(req, n, coords)

    z = values - values.mean()
    if np.allclose(z, 0):
        raise HTTPException(422, "All values are identical; Moran's I is undefined.")
    S0 = W.sum()
    I = float(n / S0 * (z @ W @ z) / (z @ z))
    EI, VI = _moran_moments(z, W)
    z_score = (I - EI) / math.sqrt(VI)
    p_two = float(2 * sp_stats.norm.sf(abs(z_score)))

    rng = np.random.default_rng(42)
    perms = np.array([rng.permutation(z) for _ in range(req.n_permutations)])
    perm_I = n / S0 * np.einsum("pi,ij,pj->p", perms, W, perms) / (z @ z)
    b = int(np.sum(perm_I >= I)) if I >= EI else int(np.sum(perm_I <= I))
    p_perm = (b + 1) / (req.n_permutations + 1)

    direction = "positive" if I > EI else "negative"
    sig = p_two < 0.05
    return {
        "morans_i": I,
        "expected_i": EI,
        "variance": VI,
        "z_score": z_score,
        "p_value": p_two,
        "p_value_method": "normal approximation under randomisation (two-sided)",
        "p_value_permutation": p_perm,
        "n_permutations": req.n_permutations,
        "n_observations": n,
        "n_locations": n,
        "interpretation": (f"Moran's I = {I:.4f} (E[I] = {EI:.4f}, z = {z_score:.2f}, p = {p_two:.4g}): "
                           + (f"significant {direction} spatial autocorrelation." if sig
                              else "no significant spatial autocorrelation at the .05 level.")),
    }


@router.post("/local-morans")
def local_morans(req: LocalMoransRequest):
    """Local Moran's I (Anselin, 1995) with conditional-permutation pseudo p-values.

    I_i = z_i * sum_j w_ij z_j / m2, m2 = sum(z^2)/n (as spdep::localmoran). The pseudo
    p-value is folded: (min(#>=, #<=) + 1) / (m + 1), as in GeoDa. p-values are not
    adjusted for the n local tests.
    """
    values = _finite(req.values, "values")
    n = len(values)
    if n < 4:
        raise HTTPException(422, "Need at least 4 observations.")
    coords = _coords(req.coordinates, n)
    _check_perms(req.n_permutations)
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    W = _weights(req, n, coords)

    z = values - values.mean()
    m2 = float(np.sum(z ** 2)) / n
    if m2 == 0:
        raise HTTPException(422, "All values are identical; local Moran's I is undefined.")
    lag = W @ z
    local_I = z * lag / m2

    rng = np.random.default_rng(42)
    p_values = np.empty(n)
    for i in range(n):
        nbr = np.nonzero(W[i])[0]
        others = np.delete(z, i)
        k_i = len(nbr)
        # draw k_i neighbour values without replacement from the other observations, m times
        draws = np.argsort(rng.random((req.n_permutations, n - 1)), axis=1)[:, :k_i]
        perm_I = z[i] * (others[draws] @ W[i, nbr]) / m2
        ge, le = int(np.sum(perm_I >= local_I[i])), int(np.sum(perm_I <= local_I[i]))
        p_values[i] = (min(ge, le) + 1) / (req.n_permutations + 1)

    names = {"HH": "High-High", "LL": "Low-Low", "HL": "High-Low", "LH": "Low-High"}
    locations, counts = [], {}
    for i in range(n):
        if p_values[i] > req.alpha:
            code = "not_significant"
        else:
            code = ("H" if z[i] > 0 else "L") + ("H" if lag[i] > 0 else "L")
        counts[code] = counts.get(code, 0) + 1
        locations.append({"index": i, "value": float(values[i]), "local_morans_i": float(local_I[i]),
                          "p_value": float(p_values[i]), "cluster_type": code,
                          "cluster_label": names.get(code, "Not significant"), "coordinate": coords[i].tolist()})

    return {
        "locations": locations,
        "cluster_summary": counts,
        "n_observations": n,
        "alpha": req.alpha,
        "interpretation": ("HH = high value among high neighbours (hot spot), LL = cold spot, HL / LH = spatial outliers. "
                           "Pseudo p-values are unadjusted across locations, so some significant locations are expected by chance."),
    }


# ── Spatial regression ────────────────────────────────────────────────

def _ols(y, X):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    return b, e, XtX_inv


def _lm_tests(y, X, W, b, e, XtX_inv):
    """LM-error, LM-lag and their robust forms (Anselin, Bera, Florax & Yoon, 1996), as spdep::lm.RStests."""
    n = len(y)
    s2 = float(e @ e) / n  # ML residual variance
    T = float(np.trace(W.T @ W + W @ W))
    WXb = W @ (X @ b)
    M = np.eye(n) - X @ XtX_inv @ X.T
    nJ = float(WXb @ M @ WXb) / s2 + T
    d_err = float(e @ W @ e) / s2
    d_lag = float(e @ W @ y) / s2
    out = {
        "lm_error": d_err ** 2 / T,
        "lm_lag": d_lag ** 2 / nJ,
        "robust_lm_error": (d_err - T / nJ * d_lag) ** 2 / (T - T ** 2 / nJ),
        "robust_lm_lag": (d_lag - d_err) ** 2 / (nJ - T),
    }
    out["sarma"] = out["robust_lm_lag"] + out["lm_error"]
    tests = {k: {"statistic": v, "df": 2 if k == "sarma" else 1,
                 "p_value": float(sp_stats.chi2.sf(v, 2 if k == "sarma" else 1))} for k, v in out.items()}
    # Anselin's decision rule: look at the robust tests only when both simple tests are significant
    le, ll = tests["lm_error"]["p_value"] < 0.05, tests["lm_lag"]["p_value"] < 0.05
    if not le and not ll:
        rec = "ols_sufficient"
    elif le and not ll:
        rec = "spatial_error"
    elif ll and not le:
        rec = "spatial_lag"
    else:
        rle, rll = tests["robust_lm_error"]["p_value"], tests["robust_lm_lag"]["p_value"]
        rec = "spatial_lag" if rll < rle else "spatial_error"
    tests["recommendation"] = rec
    tests["recommendation_rule"] = "Anselin (2005): simple LM tests first; if both significant, the smaller robust-test p-value"
    return tests


def _sar_ml(y, X, W):
    """ML spatial lag model y = rho W y + X b + e (Ord, 1975), concentrated likelihood with
    log|I - rho W| from the eigenvalues of W (spatialreg::lagsarlm method = "eigen")."""
    n, k = X.shape
    omega = np.linalg.eigvals(W)
    real = omega.real
    lower = 1.0 / real.min() if real.min() < 0 else -1.0
    upper = 1.0 / real.max()
    XtX_inv = np.linalg.inv(X.T @ X)
    Wy = W @ y
    b0 = XtX_inv @ X.T @ y
    bL = XtX_inv @ X.T @ Wy
    e0, eL = y - X @ b0, Wy - X @ bL

    def negll(rho):
        e = e0 - rho * eL
        s2 = float(e @ e) / n
        logdet = float(np.sum(np.log(np.abs(1 - rho * omega))))
        return -(-n / 2 * math.log(s2) + logdet)

    eps = 1e-8
    opt = minimize_scalar(negll, bounds=(lower + eps, upper - eps), method="bounded", options={"xatol": 1e-10})
    rho = float(opt.x)
    b = b0 - rho * bL
    e = e0 - rho * eL
    s2 = float(e @ e) / n
    logdet = float(np.sum(np.log(np.abs(1 - rho * omega)))).real
    ll = -n / 2 * (math.log(2 * math.pi) + math.log(s2) + 1) + logdet

    # Asymptotic covariance from the information matrix (Anselin, 1988, ch. 6)
    A_inv = np.linalg.inv(np.eye(n) - rho * W)
    WA = W @ A_inv
    WAXb = WA @ X @ b
    info = np.zeros((k + 2, k + 2))
    info[:k, :k] = X.T @ X / s2
    info[:k, k] = info[k, :k] = X.T @ WAXb / s2
    info[k, k] = float(np.trace(WA @ WA) + np.trace(WA.T @ WA) + WAXb @ WAXb / s2)
    info[k, k + 1] = info[k + 1, k] = float(np.trace(WA)) / s2
    info[k + 1, k + 1] = n / (2 * s2 ** 2)
    cov = np.linalg.inv(info)
    se = np.sqrt(np.diag(cov))
    return {"rho": rho, "beta": b, "s2": s2, "ll": ll, "se_beta": se[:k], "se_rho": float(se[k]),
            "rho_bounds": [float(lower), float(upper)]}


@router.post("/spatial-regression")
def spatial_regression(req: SpatialRegressionRequest):
    """OLS with Lagrange-multiplier diagnostics, and the maximum-likelihood spatial lag model."""
    y = _finite(req.y, "y")
    n = len(y)
    X_raw = _finite(req.X, "X")
    if X_raw.ndim != 2 or X_raw.shape[0] != n:
        raise HTTPException(422, "X must have one row per observation.")
    p = X_raw.shape[1]
    if n < p + 5:
        raise HTTPException(422, f"Need at least {p + 5} observations for {p} predictor(s).")
    coords = _coords(req.coordinates, n)
    W = _weights(req, n, coords)
    X = np.column_stack([np.ones(n), X_raw])
    k = X.shape[1]
    if np.linalg.matrix_rank(X) < k:
        raise HTTPException(422, "Predictors are collinear or constant.")
    names = ["intercept"] + (list(req.variable_names[:p]) if req.variable_names and len(req.variable_names) >= p
                             else [f"X{i + 1}" for i in range(p)])

    # OLS
    b, e, XtX_inv = _ols(y, X)
    s2_ols = float(e @ e) / (n - k)
    se = np.sqrt(np.diag(s2_ols * XtX_inv))
    tvals = b / se
    sst = float(((y - y.mean()) ** 2).sum())
    ll_ols = -n / 2 * (math.log(2 * math.pi) + math.log(float(e @ e) / n) + 1)
    ols = {
        "coefficients": [{"variable": names[i], "estimate": float(b[i]), "se": float(se[i]), "t_value": float(tvals[i]),
                          "p_value": float(2 * sp_stats.t.sf(abs(tvals[i]), n - k))} for i in range(k)],
        "r_squared": 1 - float(e @ e) / sst,
        "log_likelihood": ll_ols,
        "aic": -2 * ll_ols + 2 * (k + 1),
    }
    moran_resid = float(n / W.sum() * (e @ W @ e) / (e @ e))
    lm = _lm_tests(y, X, W, b, e, XtX_inv)

    # ML spatial lag model
    sar = _sar_ml(y, X, W)
    zb = sar["beta"] / sar["se_beta"]
    lag = {
        "rho": sar["rho"],
        "rho_se": sar["se_rho"],
        "rho_z": sar["rho"] / sar["se_rho"],
        "rho_p_value": float(2 * sp_stats.norm.sf(abs(sar["rho"] / sar["se_rho"]))),
        "coefficients": [{"variable": names[i], "estimate": float(sar["beta"][i]), "se": float(sar["se_beta"][i]),
                          "z_value": float(zb[i]), "p_value": float(2 * sp_stats.norm.sf(abs(zb[i])))} for i in range(k)],
        "sigma2": sar["s2"],
        "log_likelihood": sar["ll"],
        "aic": -2 * sar["ll"] + 2 * (k + 2),
        "rho_bounds": sar["rho_bounds"],
        "estimation": "maximum likelihood (eigenvalue log-determinant), asymptotic standard errors",
    }
    lr = 2 * (lag["log_likelihood"] - ll_ols)
    lag["lr_test_vs_ols"] = {"statistic": lr, "df": 1, "p_value": float(sp_stats.chi2.sf(lr, 1))}
    preferred = "spatial_lag" if lag["aic"] < ols["aic"] else "ols"

    return {
        "ols": ols,
        "residual_morans_i": moran_resid,
        "lagrange_multiplier_tests": lm,
        "spatial_lag": lag,
        # flat fields for display
        "rho": lag["rho"],
        "coefficients": lag["coefficients"],
        "aic": lag["aic"],
        "r_squared": ols["r_squared"],
        "model_comparison": {"ols_aic": ols["aic"], "spatial_lag_aic": lag["aic"], "preferred_by_aic": preferred},
        "n_observations": n,
        "n_predictors": p,
        "interpretation": (
            f"Spatial lag rho = {lag['rho']:.3f} (SE {lag['rho_se']:.3f}, p = {lag['rho_p_value']:.4g}); "
            f"LR test vs OLS: chi2(1) = {lr:.2f}. LM tests recommend: {lm['recommendation'].replace('_', ' ')}. "
            "In a lag model the coefficients are not marginal effects: a change in X also spills over to neighbours "
            "(total effect = b / (1 - rho) under row-standardised weights)."
        ),
    }


# ── GWR ───────────────────────────────────────────────────────────────

def _kernel_weights(distances: np.ndarray, bandwidth: float, kernel: str = "gaussian") -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-0.5 * (distances / bandwidth) ** 2)
    w = np.zeros_like(distances)
    mask = distances < bandwidth
    w[mask] = (1 - (distances[mask] / bandwidth) ** 2) ** 2
    return w


@router.post("/gwr")
def geographically_weighted_regression(req: GWRRequest):
    """Geographically weighted regression (Fotheringham, Brunsdon & Charlton, 2002) with a fixed
    gaussian or bisquare kernel; bandwidth by leave-one-out cross-validation unless given.
    Reports AICc = 2n ln(sigma) + n ln(2 pi) + n (n + tr S) / (n - 2 - tr S)."""
    if req.kernel not in ("gaussian", "bisquare"):
        raise HTTPException(422, "kernel must be 'gaussian' or 'bisquare'.")
    y = _finite(req.y, "y")
    n = len(y)
    X_raw = _finite(req.X, "X")
    if X_raw.ndim != 2 or X_raw.shape[0] != n:
        raise HTTPException(422, "X must have one row per observation.")
    p = X_raw.shape[1]
    if n < p + 10:
        raise HTTPException(422, f"Need at least {p + 10} observations for GWR with {p} predictor(s).")
    coords = _coords(req.coordinates, n)
    if req.bandwidth is not None and req.bandwidth <= 0:
        raise HTTPException(422, "bandwidth must be positive.")
    X = np.column_stack([np.ones(n), X_raw])
    k = X.shape[1]
    names = ["intercept"] + (list(req.variable_names[:p]) if req.variable_names and len(req.variable_names) >= p
                             else [f"X{i + 1}" for i in range(p)])
    D = cdist(coords, coords)

    b_g, e_g, _ = _ols(y, X)
    sst = float(((y - y.mean()) ** 2).sum())
    r2_global = 1 - float(e_g @ e_g) / sst

    def local_fit(i, bw, loo=False):
        w = _kernel_weights(D[i], bw, req.kernel)
        if loo:
            w = w.copy()
            w[i] = 0.0
        XtW = X.T * w
        return np.linalg.solve(XtW @ X, XtW), w

    if req.bandwidth is None:
        def cv(bw):
            err = 0.0
            for i in range(n):
                try:
                    proj, _ = local_fit(i, bw, loo=True)
                    err += (y[i] - X[i] @ (proj @ y)) ** 2
                except np.linalg.LinAlgError:
                    return 1e30
            return err
        upper = D[np.triu_indices(n, k=1)]
        lo, hi = float(np.percentile(upper, 5)), float(upper.max())
        bandwidth = float(minimize_scalar(cv, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4 * hi}).x)
        bw_method = "leave-one-out CV"
    else:
        bandwidth, bw_method = float(req.bandwidth), "user-specified"

    betas, ses = np.zeros((n, k)), np.zeros((n, k))
    trS, trSS = 0.0, 0.0
    S_rows = np.zeros((n, n))
    local_r2 = np.zeros(n)
    for i in range(n):
        try:
            proj, w = local_fit(i, bandwidth)
        except np.linalg.LinAlgError:
            raise HTTPException(422, "Local design is singular at this bandwidth; increase it.")
        betas[i] = proj @ y
        S_rows[i] = X[i] @ proj
        yhat = X @ betas[i]
        ybar = float(np.sum(w * y) / np.sum(w))
        tot = float(np.sum(w * (y - ybar) ** 2))
        local_r2[i] = 1 - float(np.sum(w * (y - yhat) ** 2)) / tot if tot > 0 else 0.0
    resid = y - np.einsum("ij,ij->i", X, betas)
    rss = float(resid @ resid)
    trS = float(np.trace(S_rows))
    trSS = float(np.trace(S_rows.T @ S_rows))
    sigma2 = rss / (n - 2 * trS + trSS)
    for i in range(n):
        proj, _ = local_fit(i, bandwidth)
        ses[i] = np.sqrt(np.diag(proj @ proj.T) * sigma2)
    sigma_ml = math.sqrt(rss / n)
    aicc = 2 * n * math.log(sigma_ml) + n * math.log(2 * math.pi) + n * (n + trS) / (n - 2 - trS) if n - 2 - trS > 0 else None

    coef_stats = {}
    for j in range(k):
        col = betas[:, j]
        coef_stats[names[j]] = {"mean": float(col.mean()), "std": float(col.std(ddof=1)), "min": float(col.min()),
                                "q1": float(np.percentile(col, 25)), "median": float(np.median(col)),
                                "q3": float(np.percentile(col, 75)), "max": float(col.max())}
    observations = [{"index": i, "coordinate": coords[i].tolist(), "local_r2": float(local_r2[i]), "residual": float(resid[i]),
                     **{f"coef_{names[j]}": float(betas[i, j]) for j in range(k)},
                     **{f"se_{names[j]}": float(ses[i, j]) for j in range(k)}} for i in range(n)]
    r2_gwr = 1 - rss / sst
    return {
        "bandwidth": bandwidth,
        "bandwidth_method": bw_method,
        "kernel": req.kernel,
        "effective_parameters": trS,
        "r2_global": r2_global,
        "r2_local": r2_gwr,
        "aicc": aicc,
        "residual_ss": rss,
        "coefficient_variation": coef_stats,
        "observations": observations,
        "n_observations": n,
        "n_predictors": p,
        "r_squared": r2_gwr,
        "interpretation": (f"GWR bandwidth = {bandwidth:.3f} ({req.kernel} kernel, {bw_method}); effective parameters = {trS:.1f}. "
                           f"Global OLS R² = {r2_global:.3f}, GWR R² = {r2_gwr:.3f}. Higher R² is expected from the extra flexibility; "
                           "compare AICc with a global model before concluding the relationships vary across space."),
    }
