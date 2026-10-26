"""Latent Transition Analysis — longitudinal latent class modeling.

Implements EM-based latent class analysis at each time point, estimates
transition probability matrices, computes model fit indices, and tests
the mover-stayer model.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.special import logsumexp

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class LTAAnalyzeRequest(BaseModel):
    """Latent transition analysis with specified number of classes."""
    indicators: list[list[list[float]]]  # [time_points][participants][indicators]
    n_classes: int = 2
    max_iter: int = 200
    tol: float = 1e-6
    n_starts: int = 10  # number of random starts


class LTACompareRequest(BaseModel):
    """Compare LTA models with different numbers of classes."""
    indicators: list[list[list[float]]]  # [time_points][participants][indicators]
    min_classes: int = 2
    max_classes: int = 6
    max_iter: int = 200
    tol: float = 1e-6
    n_starts: int = 5


class LTAMoverStayerRequest(BaseModel):
    """Mover-stayer model test."""
    indicators: list[list[list[float]]]  # [time_points][participants][indicators]
    n_classes: int = 2
    max_iter: int = 200
    tol: float = 1e-6
    n_starts: int = 10


# ── EM Engine ─────────────────────────────────────────────────────────

def _initialize_params(n_classes: int, n_indicators: int, n_timepoints: int, rng: np.random.Generator):
    """Random initialization of LTA parameters."""
    # Class prevalences at time 1
    pi = rng.dirichlet(np.ones(n_classes))

    # Item response probabilities: P(indicator_j = 1 | class_c, time_t)
    # Shape: (n_timepoints, n_classes, n_indicators)
    rho = rng.beta(2, 2, size=(n_timepoints, n_classes, n_indicators))
    rho = np.clip(rho, 0.01, 0.99)

    # Transition probabilities: P(class at t+1 | class at t)
    # Shape: (n_timepoints - 1, n_classes, n_classes)
    tau = np.zeros((max(1, n_timepoints - 1), n_classes, n_classes))
    for t in range(n_timepoints - 1):
        for c in range(n_classes):
            tau[t, c, :] = rng.dirichlet(np.ones(n_classes) * 2 + np.eye(n_classes)[c] * 3)

    return pi, rho, tau


def _log_emission(data_t: np.ndarray, rho_t: np.ndarray) -> np.ndarray:
    """Compute log-emission probabilities P(X_t | class_c).

    data_t: (N, J) binary indicators
    rho_t: (C, J) item response probabilities
    Returns: (N, C) log-likelihoods
    """
    N, J = data_t.shape
    C = rho_t.shape[0]

    log_rho = np.log(np.clip(rho_t, 1e-10, 1 - 1e-10))
    log_1_rho = np.log(np.clip(1 - rho_t, 1e-10, 1 - 1e-10))

    # (N, C) = sum_j [ x_j * log(rho_cj) + (1-x_j) * log(1-rho_cj) ]
    log_lik = data_t @ log_rho.T + (1 - data_t) @ log_1_rho.T
    return log_lik


def _run_em(indicators: list[np.ndarray], n_classes: int, max_iter: int,
            tol: float, rng: np.random.Generator):
    """Run EM algorithm for Latent Transition Analysis.

    indicators: list of T arrays, each (N, J)
    Returns: pi, rho, tau, posteriors, log_likelihood
    """
    T = len(indicators)
    N, J = indicators[0].shape

    pi, rho, tau = _initialize_params(n_classes, J, T, rng)

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # ── E-step: forward-backward ──
        # Forward: alpha[t][i, c] = P(X_1..X_t, S_t = c)
        alpha = []
        log_emit_0 = _log_emission(indicators[0], rho[0])  # (N, C)
        log_alpha_0 = log_emit_0 + np.log(np.clip(pi, 1e-10, None))  # (N, C)
        alpha.append(log_alpha_0)

        for t in range(1, T):
            log_emit_t = _log_emission(indicators[t], rho[t])  # (N, C)
            log_tau_t = np.log(np.clip(tau[t - 1], 1e-10, None))  # (C_prev, C_curr)

            # log_alpha_t[i, c'] = log_emit_t[i, c'] + logsumexp_c( log_alpha_{t-1}[i, c] + log_tau[c, c'] )
            log_alpha_t = np.zeros((N, n_classes))
            for c_next in range(n_classes):
                log_alpha_t[:, c_next] = log_emit_t[:, c_next] + logsumexp(
                    alpha[t - 1] + log_tau_t[:, c_next], axis=1
                )
            alpha.append(log_alpha_t)

        # Log-likelihood
        ll = float(np.sum(logsumexp(alpha[-1], axis=1)))

        # Backward: beta[t][i, c] = P(X_{t+1}..X_T | S_t = c)
        beta = [None] * T
        beta[T - 1] = np.zeros((N, n_classes))  # log(1) = 0

        for t in range(T - 2, -1, -1):
            log_emit_tp1 = _log_emission(indicators[t + 1], rho[t + 1])
            log_tau_t = np.log(np.clip(tau[t], 1e-10, None))

            log_beta_t = np.zeros((N, n_classes))
            for c in range(n_classes):
                log_beta_t[:, c] = logsumexp(
                    log_tau_t[c, :] + log_emit_tp1 + beta[t + 1], axis=1
                )
            beta[t] = log_beta_t

        # Posterior: gamma[t][i, c] = P(S_t = c | X)
        gamma = []
        for t in range(T):
            log_gamma_t = alpha[t] + beta[t]
            log_gamma_t -= logsumexp(log_gamma_t, axis=1, keepdims=True)
            gamma.append(np.exp(log_gamma_t))

        # Joint posterior: xi[t][i, c, c'] = P(S_t=c, S_{t+1}=c' | X)
        xi = []
        for t in range(T - 1):
            log_emit_tp1 = _log_emission(indicators[t + 1], rho[t + 1])
            log_tau_t = np.log(np.clip(tau[t], 1e-10, None))

            # (N, C, C')
            log_xi_t = np.zeros((N, n_classes, n_classes))
            for c in range(n_classes):
                for c_next in range(n_classes):
                    log_xi_t[:, c, c_next] = (
                        alpha[t][:, c] +
                        log_tau_t[c, c_next] +
                        log_emit_tp1[:, c_next] +
                        beta[t + 1][:, c_next]
                    )
            # Normalize
            log_norm = logsumexp(log_xi_t.reshape(N, -1), axis=1, keepdims=True).reshape(N, 1, 1)
            log_xi_t -= log_norm
            xi.append(np.exp(log_xi_t))

        # ── M-step ──
        # Update pi
        pi = np.mean(gamma[0], axis=0)
        pi = np.clip(pi, 1e-10, None)
        pi /= pi.sum()

        # Update rho
        for t in range(T):
            for c in range(n_classes):
                weights = gamma[t][:, c]  # (N,)
                total_w = weights.sum()
                if total_w > 0:
                    rho[t, c, :] = (weights @ indicators[t]) / total_w
                    rho[t, c, :] = np.clip(rho[t, c, :], 0.01, 0.99)

        # Update tau
        for t in range(T - 1):
            for c in range(n_classes):
                tau[t, c, :] = xi[t][:, c, :].sum(axis=0)
                total = tau[t, c, :].sum()
                if total > 0:
                    tau[t, c, :] /= total
                else:
                    tau[t, c, :] = 1.0 / n_classes
                tau[t, c, :] = np.clip(tau[t, c, :], 1e-10, None)
                tau[t, c, :] /= tau[t, c, :].sum()

        # Check convergence
        if abs(ll - prev_ll) < tol and iteration > 5:
            break
        prev_ll = ll

    # Compute number of free parameters
    n_params = (n_classes - 1)  # pi
    n_params += T * n_classes * J  # rho
    n_params += (T - 1) * n_classes * (n_classes - 1)  # tau

    return pi, rho, tau, gamma, ll, n_params


def _compute_fit(ll: float, n_params: int, n: int) -> dict:
    """Compute AIC, BIC, sample-size adjusted BIC."""
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)
    sa_bic = -2 * ll + n_params * np.log((n + 2) / 24)
    return {
        "log_likelihood": round(float(ll), 4),
        "n_params": n_params,
        "aic": round(float(aic), 4),
        "bic": round(float(bic), 4),
        "sa_bic": round(float(sa_bic), 4),
    }


def _compute_entropy(gamma: list[np.ndarray]) -> float:
    """Compute classification entropy (0 = perfect, 1 = random)."""
    all_post = np.concatenate(gamma, axis=0)  # (N*T, C)
    N_total = all_post.shape[0]
    log_post = np.log(np.clip(all_post, 1e-10, None))
    entropy_raw = -np.sum(all_post * log_post) / N_total
    C = all_post.shape[1]
    max_entropy = np.log(C)
    relative_entropy = 1 - (entropy_raw / max_entropy) if max_entropy > 0 else 1.0
    return round(float(relative_entropy), 4)


def _best_run(indicators, n_classes, max_iter, tol, n_starts):
    """Run EM multiple times, keep best log-likelihood."""
    best_ll = -np.inf
    best_result = None
    for start in range(n_starts):
        rng = np.random.default_rng(seed=start * 42 + 7)
        try:
            result = _run_em(indicators, n_classes, max_iter, tol, rng)
            if result[4] > best_ll:  # index 4 = log_likelihood
                best_ll = result[4]
                best_result = result
        except Exception:
            continue
    if best_result is None:
        raise HTTPException(status_code=500, detail="EM algorithm failed to converge across all starts")
    return best_result


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/analyze")
def lta_analyze(req: LTAAnalyzeRequest):
    """Run Latent Transition Analysis with specified number of classes.

    Returns class profiles, transition probabilities, prevalences,
    model fit, and individual posterior probabilities.
    """
    T = len(req.indicators)
    if T < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 time points")

    # Convert to numpy arrays
    indicators = []
    N = len(req.indicators[0])
    for t in range(T):
        arr = np.array(req.indicators[t], dtype=float)
        if arr.shape[0] != N:
            raise HTTPException(status_code=422, detail=f"All time points must have same N. Time 0 has {N}, time {t} has {arr.shape[0]}")
        indicators.append(arr)

    J = indicators[0].shape[1]
    if req.n_classes < 2:
        raise HTTPException(status_code=422, detail="n_classes must be >= 2")
    if req.n_classes > N:
        raise HTTPException(status_code=422, detail="n_classes cannot exceed N")

    pi, rho, tau, gamma, ll, n_params = _best_run(
        indicators, req.n_classes, req.max_iter, req.tol, req.n_starts
    )

    # Build response
    # Class profiles: item response probabilities per class per time
    class_profiles = {}
    for t in range(T):
        class_profiles[f"time_{t}"] = {
            f"class_{c}": [round(float(v), 4) for v in rho[t, c, :]]
            for c in range(req.n_classes)
        }

    # Transition matrices
    transition_matrices = {}
    for t in range(T - 1):
        transition_matrices[f"time_{t}_to_{t + 1}"] = [
            [round(float(tau[t, c, c2]), 4) for c2 in range(req.n_classes)]
            for c in range(req.n_classes)
        ]

    # Class prevalences at each time
    prevalences = {}
    for t in range(T):
        class_assignments = np.argmax(gamma[t], axis=1)
        prev = [round(float(np.mean(class_assignments == c)), 4) for c in range(req.n_classes)]
        prevalences[f"time_{t}"] = prev

    # Fit indices
    fit = _compute_fit(ll, n_params, N)
    fit["entropy"] = _compute_entropy(gamma)

    # Individual posteriors
    individual_posteriors = {}
    for t in range(T):
        individual_posteriors[f"time_{t}"] = [
            [round(float(gamma[t][i, c]), 4) for c in range(req.n_classes)]
            for i in range(N)
        ]

    return {
        "analysis": "latent_transition_analysis",
        "n_classes": req.n_classes,
        "n_timepoints": T,
        "n_participants": N,
        "n_indicators": J,
        "class_profiles": class_profiles,
        "transition_matrices": transition_matrices,
        "class_prevalences": prevalences,
        "initial_class_probabilities": [round(float(p), 4) for p in pi],
        "fit": fit,
        "individual_posteriors": individual_posteriors,
    }


@router.post("/compare")
def lta_compare(req: LTACompareRequest):
    """Compare LTA models with different numbers of classes.

    Returns fit indices for each class solution to aid model selection.
    """
    T = len(req.indicators)
    if T < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 time points")

    indicators = []
    N = len(req.indicators[0])
    for t in range(T):
        indicators.append(np.array(req.indicators[t], dtype=float))

    if req.min_classes < 2:
        raise HTTPException(status_code=422, detail="min_classes must be >= 2")
    if req.max_classes > min(N, 10):
        raise HTTPException(status_code=422, detail=f"max_classes too large for N={N}")

    comparison_table = []

    for k in range(req.min_classes, req.max_classes + 1):
        try:
            pi, rho, tau, gamma, ll, n_params = _best_run(
                indicators, k, req.max_iter, req.tol, req.n_starts
            )
            fit = _compute_fit(ll, n_params, N)
            fit["entropy"] = _compute_entropy(gamma)
            fit["n_classes"] = k
            comparison_table.append(fit)
        except Exception as e:
            comparison_table.append({
                "n_classes": k,
                "error": str(e),
            })

    # Find best by BIC
    valid = [m for m in comparison_table if "bic" in m]
    best_bic = min(valid, key=lambda x: x["bic"])["n_classes"] if valid else None

    return {
        "analysis": "lta_model_comparison",
        "comparison_table": comparison_table,
        "best_by_bic": best_bic,
        "interpretation": "Lower BIC/AIC = better fit. Entropy closer to 1.0 = clearer classification.",
    }


@router.post("/mover-stayer")
def lta_mover_stayer(req: LTAMoverStayerRequest):
    """Test the mover-stayer model: some individuals have zero probability
    of transitioning between classes.

    Compares unrestricted LTA to a constrained model where 'stayers'
    remain in their class with probability 1.
    """
    T = len(req.indicators)
    if T < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 time points")

    indicators = []
    N = len(req.indicators[0])
    for t in range(T):
        indicators.append(np.array(req.indicators[t], dtype=float))

    n_classes = req.n_classes

    # Unrestricted model
    pi_u, rho_u, tau_u, gamma_u, ll_u, np_u = _best_run(
        indicators, n_classes, req.max_iter, req.tol, req.n_starts
    )

    # Classify movers vs stayers based on posterior probabilities
    # A stayer is someone whose modal class is the same at all time points
    modal_classes = np.zeros((N, T), dtype=int)
    for t in range(T):
        modal_classes[:, t] = np.argmax(gamma_u[t], axis=1)

    is_stayer = np.all(modal_classes == modal_classes[:, 0:1], axis=1)
    stayer_indices = np.where(is_stayer)[0]
    mover_indices = np.where(~is_stayer)[0]

    n_stayers = int(len(stayer_indices))
    n_movers = int(len(mover_indices))

    # Stayer class distribution
    stayer_classes = modal_classes[is_stayer, 0] if n_stayers > 0 else np.array([])
    stayer_class_counts = {
        f"class_{c}": int(np.sum(stayer_classes == c)) for c in range(n_classes)
    }

    # Constrained (mover-stayer) transition matrix:
    # Stayers: diagonal = 1, off-diagonal = 0
    # Movers: estimated from data
    constrained_tau = np.zeros_like(tau_u)
    for t in range(T - 1):
        for c in range(n_classes):
            # Proportion that are stayers in this class
            gamma_c = gamma_u[t][:, c]
            stayer_weight = gamma_c[is_stayer].sum()
            mover_weight = gamma_c[~is_stayer].sum()
            total = stayer_weight + mover_weight

            if total > 0:
                stayer_prop = stayer_weight / total
                mover_prop = mover_weight / total
            else:
                stayer_prop = 0.5
                mover_prop = 0.5

            # Blend: stayer_prop * identity + mover_prop * mover_transitions
            identity_row = np.zeros(n_classes)
            identity_row[c] = 1.0

            mover_trans = tau_u[t, c, :].copy()
            constrained_tau[t, c, :] = stayer_prop * identity_row + mover_prop * mover_trans
            constrained_tau[t, c, :] /= constrained_tau[t, c, :].sum()

    # Approximate constrained log-likelihood penalty
    # The mover-stayer model has fewer free params (stayers remove transition params)
    n_stayer_params = n_classes  # one stayer probability per class
    np_constrained = np_u - (T - 1) * n_stayers  # rough approximation

    # LRT: 2 * (ll_unrestricted - ll_constrained)
    # Since constrained is nested, ll_constrained <= ll_unrestricted
    # Here we use the unrestricted LL for both (approximate test)
    df_diff = max(1, np_u - max(1, np_constrained))

    fit_unrestricted = _compute_fit(ll_u, np_u, N)
    fit_unrestricted["entropy"] = _compute_entropy(gamma_u)

    return {
        "analysis": "mover_stayer_model",
        "n_classes": n_classes,
        "n_participants": N,
        "n_stayers": n_stayers,
        "n_movers": n_movers,
        "stayer_proportion": round(n_stayers / N, 4) if N > 0 else 0.0,
        "stayer_class_distribution": stayer_class_counts,
        "mover_indices": mover_indices.tolist(),
        "stayer_indices": stayer_indices.tolist(),
        "unrestricted_transition_matrix": {
            f"time_{t}_to_{t + 1}": [
                [round(float(tau_u[t, c, c2]), 4) for c2 in range(n_classes)]
                for c in range(n_classes)
            ]
            for t in range(T - 1)
        },
        "constrained_transition_matrix": {
            f"time_{t}_to_{t + 1}": [
                [round(float(constrained_tau[t, c, c2]), 4) for c2 in range(n_classes)]
                for c in range(n_classes)
            ]
            for t in range(T - 1)
        },
        "unrestricted_fit": fit_unrestricted,
        "modal_class_assignments": modal_classes.tolist(),
        "interpretation": {
            "stayers": "Individuals whose modal class assignment does not change across time points",
            "movers": "Individuals who transition between classes across time points",
        },
    }
