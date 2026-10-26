"""Latent Class & Latent Profile Analysis — EM-based mixture modelling."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


# ── Request / Config Models ──────────────────────────────────────────

class LCARequest(BaseModel):
    data: list[list[int]]  # N x J binary matrix
    n_classes: int = 2
    n_starts: int = 10
    max_iter: int = 1000
    tol: float = 1e-6

class LPARequest(BaseModel):
    data: list[list[float]]  # N x P continuous matrix
    n_classes: int = 2
    n_starts: int = 10
    max_iter: int = 1000
    tol: float = 1e-6
    covariance_type: str = "diagonal_unequal"  # diagonal_equal, diagonal_unequal, full_equal, full_unequal

class CompareModelsRequest(BaseModel):
    data: list[list[float]]
    model_type: str = "lca"  # "lca" or "lpa"
    min_classes: int = 1
    max_classes: int = 8
    n_starts: int = 10
    max_iter: int = 1000
    tol: float = 1e-6
    blrt_samples: int = 100
    covariance_type: str = "diagonal_unequal"

class GrowthMixtureRequest(BaseModel):
    data: list[list[float]]  # N x T longitudinal matrix (rows=subjects, cols=timepoints)
    n_classes: int = 2
    n_starts: int = 10
    max_iter: int = 500
    tol: float = 1e-6
    time_points: list[float] | None = None  # custom time coding; defaults to 0,1,2,...

class ClassComparisonRequest(BaseModel):
    assignments: list[int]  # class label per observation
    auxiliary_variables: dict[str, list[float]]  # name -> values per observation


# ── Helpers ───────────────────────────────────────────────────────────

def _compute_entropy(posterior: np.ndarray) -> float:
    """Classification entropy R (closer to 1 = better separation)."""
    N = posterior.shape[0]
    ent = -np.sum(posterior * np.log(np.clip(posterior, 1e-300, 1.0)))
    K = posterior.shape[1]
    if K <= 1:
        return 1.0
    max_ent = N * np.log(K)
    return float(1.0 - ent / max_ent) if max_ent > 0 else 1.0


def _aic(ll: float, n_params: int) -> float:
    return -2.0 * ll + 2.0 * n_params


def _bic(ll: float, n_params: int, n: int) -> float:
    return -2.0 * ll + n_params * np.log(n)


def _sabic(ll: float, n_params: int, n: int) -> float:
    return -2.0 * ll + n_params * np.log((n + 2.0) / 24.0)


def _classification_table(assignments: np.ndarray, K: int) -> dict:
    counts = {int(k): int(np.sum(assignments == k)) for k in range(K)}
    N = len(assignments)
    proportions = {int(k): round(c / N, 4) for k, c in counts.items()}
    return {"counts": counts, "proportions": proportions}


def _sort_classes_by_proportion(pi: np.ndarray, params: list, posterior: np.ndarray):
    """Sort classes by descending proportion to reduce label switching."""
    order = np.argsort(-pi)
    pi = pi[order]
    params = [params[i] for i in order]
    posterior = posterior[:, order]
    return pi, params, posterior


# ── LCA core ─────────────────────────────────────────────────────────

def _lca_em(X: np.ndarray, K: int, max_iter: int, tol: float, rng: np.random.Generator):
    """Single-start EM for Latent Class Analysis on binary indicators."""
    N, J = X.shape

    # Random initialisation
    pi = rng.dirichlet(np.ones(K))
    rho = rng.uniform(0.1, 0.9, size=(K, J))  # item-response probabilities

    ll_prev = -np.inf
    for iteration in range(max_iter):
        # E-step: log P(x_i | class k) for each observation
        log_lik_nk = np.zeros((N, K))
        for k in range(K):
            log_p = X * np.log(np.clip(rho[k], 1e-300, 1.0)) + \
                     (1 - X) * np.log(np.clip(1 - rho[k], 1e-300, 1.0))
            log_lik_nk[:, k] = np.sum(log_p, axis=1) + np.log(np.clip(pi[k], 1e-300, 1.0))

        # Log-sum-exp for numerical stability
        max_log = np.max(log_lik_nk, axis=1, keepdims=True)
        log_sum = max_log + np.log(np.sum(np.exp(log_lik_nk - max_log), axis=1, keepdims=True))
        log_posterior = log_lik_nk - log_sum
        posterior = np.exp(log_posterior)

        ll = float(np.sum(log_sum))

        if abs(ll - ll_prev) < tol and iteration > 0:
            break
        ll_prev = ll

        # M-step
        Nk = np.sum(posterior, axis=0)

        # Handle empty classes
        empty = Nk < 1e-10
        if np.any(empty):
            Nk[empty] = 1e-10

        pi = Nk / N
        for k in range(K):
            rho[k] = (posterior[:, k] @ X) / Nk[k]
            rho[k] = np.clip(rho[k], 1e-6, 1.0 - 1e-6)

    return pi, rho, posterior, ll, iteration + 1


def _run_lca(X: np.ndarray, K: int, n_starts: int, max_iter: int, tol: float):
    best_ll = -np.inf
    best_result = None
    for s in range(n_starts):
        rng = np.random.default_rng(seed=s * 42 + 7)
        try:
            pi, rho, posterior, ll, iters = _lca_em(X, K, max_iter, tol, rng)
            if ll > best_ll:
                best_ll = ll
                best_result = (pi, rho, posterior, ll, iters)
        except Exception:
            continue
    if best_result is None:
        raise HTTPException(500, "All EM starts failed for LCA. Check data.")
    return best_result


# ── LPA core ─────────────────────────────────────────────────────────

def _lpa_em(X: np.ndarray, K: int, max_iter: int, tol: float,
            covariance_type: str, rng: np.random.Generator):
    """Single-start EM for Latent Profile Analysis (Gaussian mixture)."""
    N, P = X.shape

    # Random initialisation — pick K random observations as means + jitter
    indices = rng.choice(N, size=K, replace=False) if N >= K else rng.choice(N, size=K, replace=True)
    means = X[indices] + rng.normal(0, 0.1, size=(K, P))
    pi = rng.dirichlet(np.ones(K))

    # Initialise covariance
    data_var = np.var(X, axis=0) + 1e-6
    if covariance_type in ("diagonal_equal", "diagonal_unequal"):
        covs = np.tile(np.diag(data_var), (K, 1, 1))
    else:
        covs = np.tile(np.diag(data_var), (K, 1, 1))

    MIN_VAR = 1e-6
    ll_prev = -np.inf

    for iteration in range(max_iter):
        # E-step
        log_lik_nk = np.zeros((N, K))
        for k in range(K):
            try:
                diff = X - means[k]
                cov_k = covs[k]
                # Regularise
                cov_k += np.eye(P) * MIN_VAR
                sign, logdet = np.linalg.slogdet(cov_k)
                if sign <= 0:
                    cov_k = np.diag(np.diag(cov_k)) + np.eye(P) * MIN_VAR
                    sign, logdet = np.linalg.slogdet(cov_k)
                cov_inv = np.linalg.inv(cov_k)
                mahal = np.sum((diff @ cov_inv) * diff, axis=1)
                log_lik_nk[:, k] = -0.5 * (P * np.log(2 * np.pi) + logdet + mahal) + \
                                    np.log(np.clip(pi[k], 1e-300, 1.0))
            except np.linalg.LinAlgError:
                log_lik_nk[:, k] = -1e10

        max_log = np.max(log_lik_nk, axis=1, keepdims=True)
        log_sum = max_log + np.log(np.sum(np.exp(log_lik_nk - max_log), axis=1, keepdims=True))
        log_posterior = log_lik_nk - log_sum
        posterior = np.exp(log_posterior)

        ll = float(np.sum(log_sum))
        if abs(ll - ll_prev) < tol and iteration > 0:
            break
        ll_prev = ll

        # M-step
        Nk = np.sum(posterior, axis=0)
        empty = Nk < 1e-10
        if np.any(empty):
            Nk[empty] = 1e-10

        pi = Nk / N
        new_means = np.zeros_like(means)
        new_covs = np.zeros_like(covs)

        for k in range(K):
            w = posterior[:, k]
            new_means[k] = (w @ X) / Nk[k]
            diff = X - new_means[k]
            weighted_diff = diff * w[:, None]

            if covariance_type in ("full_equal", "full_unequal"):
                new_covs[k] = (weighted_diff.T @ diff) / Nk[k]
            else:
                # Diagonal
                new_covs[k] = np.diag(np.sum(weighted_diff * diff, axis=0) / Nk[k])

            # Protect against degenerate variance
            diag_vals = np.diag(new_covs[k])
            diag_vals = np.maximum(diag_vals, MIN_VAR)
            np.fill_diagonal(new_covs[k], diag_vals)

        # Equal variance constraint
        if covariance_type in ("diagonal_equal", "full_equal"):
            pooled = sum(new_covs[k] * (Nk[k] / N) for k in range(K))
            for k in range(K):
                new_covs[k] = pooled.copy()

        means = new_means
        covs = new_covs

    return pi, means, covs, posterior, ll, iteration + 1


def _run_lpa(X: np.ndarray, K: int, n_starts: int, max_iter: int, tol: float,
             covariance_type: str):
    best_ll = -np.inf
    best_result = None
    for s in range(n_starts):
        rng = np.random.default_rng(seed=s * 42 + 7)
        try:
            pi, means, covs, posterior, ll, iters = _lpa_em(
                X, K, max_iter, tol, covariance_type, rng
            )
            if ll > best_ll:
                best_ll = ll
                best_result = (pi, means, covs, posterior, ll, iters)
        except Exception:
            continue
    if best_result is None:
        raise HTTPException(500, "All EM starts failed for LPA. Check data.")
    return best_result


# ── Growth Mixture helpers ────────────────────────────────────────────

def _fit_growth_mixture(X: np.ndarray, K: int, time_pts: np.ndarray,
                        n_starts: int, max_iter: int, tol: float):
    """EM for Growth Mixture Model (random intercept + slope per class)."""
    N, T_len = X.shape
    if T_len != len(time_pts):
        raise HTTPException(400, "Mismatch between data columns and time_points length.")

    # Design matrix for linear growth: [1, t]
    G = np.column_stack([np.ones(T_len), time_pts])  # T x 2

    best_ll = -np.inf
    best_result = None

    for s in range(n_starts):
        rng = np.random.default_rng(seed=s * 42 + 13)
        pi = rng.dirichlet(np.ones(K))
        # Each class has mean intercept, mean slope, and residual variance
        betas = rng.normal(0, 1, size=(K, 2))  # intercept, slope per class
        sigma2 = np.full(K, np.var(X) * 0.5 + 1e-6)

        ll_prev = -np.inf
        converged = False

        for iteration in range(max_iter):
            # E-step
            log_lik_nk = np.zeros((N, K))
            for k in range(K):
                pred = G @ betas[k]  # length T
                resid = X - pred[None, :]  # N x T
                ss = np.sum(resid ** 2, axis=1)  # N
                log_lik_nk[:, k] = (-T_len / 2.0) * np.log(2 * np.pi * max(sigma2[k], 1e-10)) \
                                    - ss / (2.0 * max(sigma2[k], 1e-10)) \
                                    + np.log(np.clip(pi[k], 1e-300, 1.0))

            max_log = np.max(log_lik_nk, axis=1, keepdims=True)
            log_sum = max_log + np.log(np.sum(np.exp(log_lik_nk - max_log), axis=1, keepdims=True))
            posterior = np.exp(log_lik_nk - log_sum)

            ll = float(np.sum(log_sum))
            if abs(ll - ll_prev) < tol and iteration > 0:
                converged = True
                break
            ll_prev = ll

            # M-step
            Nk = np.sum(posterior, axis=0)
            empty = Nk < 1e-10
            if np.any(empty):
                Nk[empty] = 1e-10

            pi = Nk / N
            for k in range(K):
                w = posterior[:, k]
                # Weighted OLS: beta_k = (G'WG)^{-1} G'Wy
                W = np.diag(np.repeat(w, T_len))
                Y_flat = X.flatten()  # N*T
                G_big = np.tile(G, (N, 1))  # (N*T) x 2
                # Efficient version without huge diagonal
                # Sum over observations
                GtWG = np.zeros((2, 2))
                GtWy = np.zeros(2)
                for i in range(N):
                    GtWG += w[i] * (G.T @ G)
                    GtWy += w[i] * (G.T @ X[i])
                try:
                    betas[k] = np.linalg.solve(GtWG + np.eye(2) * 1e-8, GtWy)
                except np.linalg.LinAlgError:
                    pass
                pred = G @ betas[k]
                resid = X - pred[None, :]
                sigma2[k] = np.sum(w[:, None] * resid ** 2) / (Nk[k] * T_len)
                sigma2[k] = max(sigma2[k], 1e-6)

        if ll > best_ll:
            best_ll = ll
            best_result = (pi, betas, sigma2, posterior, ll, iteration + 1, converged)

    if best_result is None:
        raise HTTPException(500, "All EM starts failed for GMM growth model.")
    return best_result


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/lca")
def latent_class_analysis(req: LCARequest):
    """Latent Class Analysis for binary/categorical indicators via EM."""
    X = np.array(req.data, dtype=np.float64)
    N, J = X.shape
    K = req.n_classes

    if K < 2 or K > 10:
        raise HTTPException(400, "n_classes must be between 2 and 10.")
    if N < K:
        raise HTTPException(400, "Number of observations must be >= n_classes.")
    if not np.all((X == 0) | (X == 1)):
        raise HTTPException(400, "LCA requires binary (0/1) indicators.")

    pi, rho, posterior, ll, iters = _run_lca(X, K, req.n_starts, req.max_iter, req.tol)

    # Sort by class proportion (descending)
    rho_list = [rho[k] for k in range(K)]
    pi, rho_list, posterior = _sort_classes_by_proportion(pi, rho_list, posterior)
    rho = np.array(rho_list)

    assignments = np.argmax(posterior, axis=1)
    n_params = (K - 1) + K * J  # class proportions + item-response probs

    return {
        "n_classes": K,
        "n_observations": N,
        "n_items": J,
        "class_proportions": [round(float(p), 6) for p in pi],
        "item_response_probabilities": {
            f"class_{k + 1}": [round(float(v), 6) for v in rho[k]]
            for k in range(K)
        },
        "posterior_probabilities": [[round(float(v), 6) for v in row] for row in posterior],
        "class_assignments": [int(a) + 1 for a in assignments],
        "log_likelihood": round(float(ll), 4),
        "n_parameters": n_params,
        "aic": round(_aic(ll, n_params), 4),
        "bic": round(_bic(ll, n_params, N), 4),
        "sabic": round(_sabic(ll, n_params, N), 4),
        "entropy": round(_compute_entropy(posterior), 4),
        "classification_table": _classification_table(assignments, K),
        "iterations": iters,
    }


@router.post("/lpa")
def latent_profile_analysis(req: LPARequest):
    """Latent Profile Analysis for continuous indicators via Gaussian mixture EM."""
    X = np.array(req.data, dtype=np.float64)
    N, P = X.shape
    K = req.n_classes

    if K < 2 or K > 10:
        raise HTTPException(400, "n_classes must be between 2 and 10.")
    if N < K:
        raise HTTPException(400, "Number of observations must be >= n_classes.")
    valid_cov = ("diagonal_equal", "diagonal_unequal", "full_equal", "full_unequal")
    if req.covariance_type not in valid_cov:
        raise HTTPException(400, f"covariance_type must be one of {valid_cov}.")

    pi, means, covs, posterior, ll, iters = _run_lpa(
        X, K, req.n_starts, req.max_iter, req.tol, req.covariance_type
    )

    # Sort by proportion
    params = list(range(K))
    pi, params, posterior = _sort_classes_by_proportion(pi, params, posterior)
    means = means[params]
    covs = covs[params]

    assignments = np.argmax(posterior, axis=1)

    # Count parameters
    if req.covariance_type == "diagonal_equal":
        cov_params = P
    elif req.covariance_type == "diagonal_unequal":
        cov_params = K * P
    elif req.covariance_type == "full_equal":
        cov_params = P * (P + 1) // 2
    else:  # full_unequal
        cov_params = K * P * (P + 1) // 2
    n_params = (K - 1) + K * P + cov_params

    # Within-class correlations
    within_corr = {}
    for k in range(K):
        cov_k = covs[k]
        d = np.sqrt(np.diag(cov_k))
        d[d < 1e-10] = 1e-10
        corr = cov_k / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        within_corr[f"class_{k + 1}"] = [[round(float(v), 6) for v in row] for row in corr]

    return {
        "n_classes": K,
        "n_observations": N,
        "n_variables": P,
        "covariance_type": req.covariance_type,
        "class_proportions": [round(float(p), 6) for p in pi],
        "class_means": {
            f"class_{k + 1}": [round(float(v), 6) for v in means[k]]
            for k in range(K)
        },
        "class_covariances": {
            f"class_{k + 1}": [[round(float(v), 6) for v in row] for row in covs[k]]
            for k in range(K)
        },
        "within_class_correlations": within_corr,
        "posterior_probabilities": [[round(float(v), 6) for v in row] for row in posterior],
        "class_assignments": [int(a) + 1 for a in assignments],
        "log_likelihood": round(float(ll), 4),
        "n_parameters": n_params,
        "aic": round(_aic(ll, n_params), 4),
        "bic": round(_bic(ll, n_params, N), 4),
        "sabic": round(_sabic(ll, n_params, N), 4),
        "entropy": round(_compute_entropy(posterior), 4),
        "classification_table": _classification_table(assignments, K),
        "iterations": iters,
    }


@router.post("/compare-models")
def compare_models(req: CompareModelsRequest):
    """Fit LCA or LPA across a range of class counts and compare fit indices."""
    X = np.array(req.data, dtype=np.float64)
    N = X.shape[0]

    if req.model_type not in ("lca", "lpa"):
        raise HTTPException(400, "model_type must be 'lca' or 'lpa'.")
    if req.min_classes < 1 or req.max_classes > 10 or req.min_classes > req.max_classes:
        raise HTTPException(400, "Invalid class range. min_classes >= 1, max_classes <= 10.")

    results = []

    for K in range(req.min_classes, req.max_classes + 1):
        try:
            if K == 1:
                # 1-class model — closed form
                if req.model_type == "lca":
                    J = X.shape[1]
                    rho = np.mean(X, axis=0)
                    ll = float(np.sum(
                        X * np.log(np.clip(rho, 1e-300, 1.0)) +
                        (1 - X) * np.log(np.clip(1 - rho, 1e-300, 1.0))
                    ))
                    n_params = J
                else:
                    P = X.shape[1]
                    mu = np.mean(X, axis=0)
                    cov = np.cov(X.T) if P > 1 else np.array([[np.var(X)]])
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]])
                    cov += np.eye(P) * 1e-6
                    rv = sp_stats.multivariate_normal(mean=mu, cov=cov, allow_singular=True)
                    ll = float(np.sum(rv.logpdf(X)))
                    n_params = P + P * (P + 1) // 2

                results.append({
                    "n_classes": 1,
                    "log_likelihood": round(ll, 4),
                    "n_parameters": n_params,
                    "aic": round(_aic(ll, n_params), 4),
                    "bic": round(_bic(ll, n_params, N), 4),
                    "sabic": round(_sabic(ll, n_params, N), 4),
                    "entropy": None,
                    "smallest_class_pct": 100.0,
                    "converged": True,
                })
            else:
                if req.model_type == "lca":
                    pi, rho, posterior, ll, iters = _run_lca(
                        X, K, req.n_starts, req.max_iter, req.tol
                    )
                    J = X.shape[1]
                    n_params = (K - 1) + K * J
                else:
                    pi, means, covs, posterior, ll, iters = _run_lpa(
                        X, K, req.n_starts, req.max_iter, req.tol, req.covariance_type
                    )
                    P = X.shape[1]
                    if req.covariance_type == "diagonal_equal":
                        cov_params = P
                    elif req.covariance_type == "diagonal_unequal":
                        cov_params = K * P
                    elif req.covariance_type == "full_equal":
                        cov_params = P * (P + 1) // 2
                    else:
                        cov_params = K * P * (P + 1) // 2
                    n_params = (K - 1) + K * P + cov_params

                assignments = np.argmax(posterior, axis=1)
                class_sizes = [np.sum(assignments == k) for k in range(K)]
                smallest_pct = round(float(min(class_sizes) / N * 100), 2)

                results.append({
                    "n_classes": K,
                    "log_likelihood": round(float(ll), 4),
                    "n_parameters": n_params,
                    "aic": round(_aic(ll, n_params), 4),
                    "bic": round(_bic(ll, n_params, N), 4),
                    "sabic": round(_sabic(ll, n_params, N), 4),
                    "entropy": round(_compute_entropy(posterior), 4),
                    "smallest_class_pct": smallest_pct,
                    "converged": True,
                })
        except Exception as exc:
            results.append({
                "n_classes": K,
                "log_likelihood": None,
                "n_parameters": None,
                "aic": None,
                "bic": None,
                "sabic": None,
                "entropy": None,
                "smallest_class_pct": None,
                "converged": False,
                "error": str(exc),
            })

    # BLRT: parametric bootstrap LR test between K and K-1 classes
    blrt_results = []
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        if prev["log_likelihood"] is None or curr["log_likelihood"] is None:
            blrt_results.append({"k_minus_1": prev["n_classes"], "k": curr["n_classes"],
                                 "lr_statistic": None, "p_value": None})
            continue

        observed_lr = -2.0 * (prev["log_likelihood"] - curr["log_likelihood"])
        K_prev = prev["n_classes"]

        # Parametric bootstrap under K-1 model
        boot_lrs = []
        for b in range(req.blrt_samples):
            try:
                rng = np.random.default_rng(seed=b * 101 + 3)
                if req.model_type == "lca":
                    if K_prev == 1:
                        p = np.mean(X, axis=0)
                        X_boot = rng.binomial(1, np.clip(p, 0.01, 0.99), size=(N, X.shape[1])).astype(float)
                    else:
                        pi0, rho0, _, ll0, _ = _run_lca(X, K_prev, 3, req.max_iter, req.tol)
                        classes = rng.choice(K_prev, size=N, p=pi0)
                        X_boot = np.zeros((N, X.shape[1]))
                        for n_i in range(N):
                            X_boot[n_i] = rng.binomial(1, np.clip(rho0[classes[n_i]], 0.01, 0.99))

                    if K_prev == 1:
                        p_boot = np.mean(X_boot, axis=0)
                        ll_null = float(np.sum(
                            X_boot * np.log(np.clip(p_boot, 1e-300, 1.0)) +
                            (1 - X_boot) * np.log(np.clip(1 - p_boot, 1e-300, 1.0))
                        ))
                    else:
                        _, _, _, ll_null, _ = _run_lca(X_boot, K_prev, 3, req.max_iter, req.tol)
                    _, _, _, ll_alt, _ = _run_lca(X_boot, curr["n_classes"], 3, req.max_iter, req.tol)
                else:
                    # LPA bootstrap
                    if K_prev == 1:
                        mu = np.mean(X, axis=0)
                        cov = np.cov(X.T) if X.shape[1] > 1 else np.array([[np.var(X)]])
                        if cov.ndim == 0:
                            cov = np.array([[float(cov)]])
                        cov += np.eye(X.shape[1]) * 1e-6
                        X_boot = rng.multivariate_normal(mu, cov, size=N)
                        rv = sp_stats.multivariate_normal(mean=np.mean(X_boot, axis=0),
                                                          cov=np.cov(X_boot.T) + np.eye(X.shape[1]) * 1e-6,
                                                          allow_singular=True)
                        ll_null = float(np.sum(rv.logpdf(X_boot)))
                    else:
                        pi0, mu0, cov0, _, ll0, _ = _run_lpa(
                            X, K_prev, 3, req.max_iter, req.tol, req.covariance_type
                        )
                        classes = rng.choice(K_prev, size=N, p=pi0)
                        X_boot = np.zeros((N, X.shape[1]))
                        for n_i in range(N):
                            X_boot[n_i] = rng.multivariate_normal(mu0[classes[n_i]],
                                                                    cov0[classes[n_i]] + np.eye(X.shape[1]) * 1e-6)
                        _, _, _, _, ll_null, _ = _run_lpa(
                            X_boot, K_prev, 3, req.max_iter, req.tol, req.covariance_type
                        )
                    _, _, _, _, ll_alt, _ = _run_lpa(
                        X_boot, curr["n_classes"], 3, req.max_iter, req.tol, req.covariance_type
                    )

                boot_lr = -2.0 * (ll_null - ll_alt)
                boot_lrs.append(boot_lr)
            except Exception:
                continue

        if len(boot_lrs) > 0:
            p_value = float(np.mean(np.array(boot_lrs) >= observed_lr))
        else:
            p_value = None

        blrt_results.append({
            "k_minus_1": prev["n_classes"],
            "k": curr["n_classes"],
            "lr_statistic": round(float(observed_lr), 4),
            "p_value": round(p_value, 4) if p_value is not None else None,
            "n_bootstrap_samples": len(boot_lrs),
        })

    # Elbow data
    elbow_data = {
        "n_classes": [r["n_classes"] for r in results],
        "aic": [r["aic"] for r in results],
        "bic": [r["bic"] for r in results],
        "sabic": [r["sabic"] for r in results],
        "log_likelihood": [r["log_likelihood"] for r in results],
    }

    return {
        "model_type": req.model_type,
        "comparison_table": results,
        "blrt": blrt_results,
        "elbow_plot_data": elbow_data,
        "recommendation": _recommend_classes(results),
    }


def _recommend_classes(results: list[dict]) -> str:
    """Simple heuristic: pick the model with lowest BIC that also has entropy > 0.6."""
    valid = [r for r in results if r["bic"] is not None]
    if not valid:
        return "No valid models to compare."
    best_bic = min(valid, key=lambda r: r["bic"])
    msg = f"Lowest BIC at {best_bic['n_classes']} class(es) (BIC = {best_bic['bic']})."
    if best_bic.get("entropy") is not None and best_bic["entropy"] < 0.6:
        msg += " Warning: entropy is below 0.6 — class separation may be poor."
    if best_bic.get("smallest_class_pct") is not None and best_bic["smallest_class_pct"] < 5:
        msg += " Warning: smallest class contains < 5% of the sample."
    return msg


@router.post("/growth-mixture")
def growth_mixture_model(req: GrowthMixtureRequest):
    """Growth Mixture Model — class-specific linear trajectories via EM."""
    X = np.array(req.data, dtype=np.float64)
    N, T_len = X.shape
    K = req.n_classes

    if K < 2 or K > 10:
        raise HTTPException(400, "n_classes must be between 2 and 10.")
    if N < K:
        raise HTTPException(400, "Number of observations must be >= n_classes.")
    if T_len < 2:
        raise HTTPException(400, "Need at least 2 time points.")

    time_pts = np.array(req.time_points, dtype=np.float64) if req.time_points else np.arange(T_len, dtype=np.float64)
    if len(time_pts) != T_len:
        raise HTTPException(400, "time_points length must match number of columns in data.")

    pi, betas, sigma2, posterior, ll, iters, converged = _fit_growth_mixture(
        X, K, time_pts, req.n_starts, req.max_iter, req.tol
    )

    # Sort by proportion
    params = list(range(K))
    pi, params, posterior = _sort_classes_by_proportion(pi, params, posterior)
    betas = betas[params]
    sigma2 = sigma2[params]

    assignments = np.argmax(posterior, axis=1)

    # n_params: (K-1) proportions + K*(intercept + slope + sigma2)
    n_params = (K - 1) + K * 3

    # Trajectory plot data
    t_fine = np.linspace(float(time_pts[0]), float(time_pts[-1]), 50)
    trajectories = {}
    for k in range(K):
        y_hat = betas[k, 0] + betas[k, 1] * t_fine
        trajectories[f"class_{k + 1}"] = {
            "time": [round(float(t), 4) for t in t_fine],
            "predicted": [round(float(y), 4) for y in y_hat],
            "intercept": round(float(betas[k, 0]), 6),
            "slope": round(float(betas[k, 1]), 6),
            "residual_variance": round(float(sigma2[k]), 6),
        }

    return {
        "n_classes": K,
        "n_observations": N,
        "n_timepoints": T_len,
        "time_points": [float(t) for t in time_pts],
        "class_proportions": [round(float(p), 6) for p in pi],
        "trajectories": trajectories,
        "posterior_probabilities": [[round(float(v), 6) for v in row] for row in posterior],
        "class_assignments": [int(a) + 1 for a in assignments],
        "log_likelihood": round(float(ll), 4),
        "n_parameters": n_params,
        "aic": round(_aic(ll, n_params), 4),
        "bic": round(_bic(ll, n_params, N), 4),
        "sabic": round(_sabic(ll, n_params, N), 4),
        "entropy": round(_compute_entropy(posterior), 4),
        "classification_table": _classification_table(assignments, K),
        "iterations": iters,
        "converged": converged,
    }


@router.post("/class-comparison")
def class_comparison(req: ClassComparisonRequest):
    """Compare latent classes on auxiliary/distal variables (3-step-style)."""
    assignments = np.array(req.assignments, dtype=int)
    N = len(assignments)

    if N == 0:
        raise HTTPException(400, "No observations provided.")

    classes = np.unique(assignments)
    K = len(classes)
    if K < 2:
        raise HTTPException(400, "Need at least 2 classes for comparison.")

    # Validate auxiliary variables
    for name, vals in req.auxiliary_variables.items():
        if len(vals) != N:
            raise HTTPException(400, f"Variable '{name}' length ({len(vals)}) != assignments length ({N}).")

    comparison_results = {}

    for var_name, var_vals in req.auxiliary_variables.items():
        y = np.array(var_vals, dtype=np.float64)

        class_stats = []
        group_means = []
        group_ns = []
        group_vars = []

        for k in classes:
            mask = assignments == k
            yk = y[mask]
            n_k = len(yk)
            if n_k == 0:
                class_stats.append({
                    "class": int(k), "n": 0, "mean": None, "sd": None, "se": None
                })
                continue
            m = float(np.mean(yk))
            sd = float(np.std(yk, ddof=1)) if n_k > 1 else 0.0
            se = sd / np.sqrt(n_k) if n_k > 0 else 0.0
            class_stats.append({
                "class": int(k), "n": int(n_k),
                "mean": round(m, 6), "sd": round(sd, 6), "se": round(se, 6),
            })
            group_means.append(m)
            group_ns.append(n_k)
            group_vars.append(sd ** 2)

        # Omnibus Wald test (chi-square)
        # Wald = sum over pairs of (m_j - m_k)^2 / (se_j^2 + se_k^2)
        # Simplified: use one-way ANOVA F -> chi-square approximation
        groups_for_test = [y[assignments == k] for k in classes if np.sum(assignments == k) > 0]
        if len(groups_for_test) >= 2 and all(len(g) > 0 for g in groups_for_test):
            f_stat, f_p = sp_stats.f_oneway(*groups_for_test)
            df_between = len(groups_for_test) - 1
            chi2_omnibus = float(f_stat * df_between)
            p_omnibus = float(sp_stats.chi2.sf(chi2_omnibus, df_between))
        else:
            chi2_omnibus = None
            p_omnibus = None
            df_between = None

        # Pairwise comparisons with Bonferroni correction
        valid_classes = [k for k in classes if np.sum(assignments == k) > 1]
        n_comparisons = len(valid_classes) * (len(valid_classes) - 1) // 2
        pairwise = []
        for i in range(len(valid_classes)):
            for j in range(i + 1, len(valid_classes)):
                k1, k2 = valid_classes[i], valid_classes[j]
                y1 = y[assignments == k1]
                y2 = y[assignments == k2]
                t_stat, p_val = sp_stats.ttest_ind(y1, y2)
                p_bonf = min(float(p_val) * n_comparisons, 1.0) if n_comparisons > 0 else float(p_val)
                cohens_d = (np.mean(y1) - np.mean(y2)) / np.sqrt(
                    ((len(y1) - 1) * np.var(y1, ddof=1) + (len(y2) - 1) * np.var(y2, ddof=1))
                    / (len(y1) + len(y2) - 2)
                ) if (len(y1) + len(y2)) > 2 else 0.0
                pairwise.append({
                    "class_1": int(k1), "class_2": int(k2),
                    "mean_diff": round(float(np.mean(y1) - np.mean(y2)), 6),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "p_bonferroni": round(p_bonf, 6),
                    "cohens_d": round(float(cohens_d), 4),
                    "significant_bonferroni": p_bonf < 0.05,
                })

        comparison_results[var_name] = {
            "class_descriptives": class_stats,
            "omnibus_wald_chi2": round(chi2_omnibus, 4) if chi2_omnibus is not None else None,
            "omnibus_df": int(df_between) if df_between is not None else None,
            "omnibus_p_value": round(p_omnibus, 6) if p_omnibus is not None else None,
            "pairwise_comparisons": pairwise,
        }

    return {
        "n_observations": N,
        "n_classes": K,
        "classes_found": [int(k) for k in classes],
        "n_auxiliary_variables": len(req.auxiliary_variables),
        "comparisons": comparison_results,
        "note": (
            "This analysis uses a classify-analyse approach. For more accurate results "
            "with latent class uncertainty, consider the BCH method or Vermunt's 3-step "
            "approach, which account for classification error in auxiliary variable comparisons."
        ),
    }
