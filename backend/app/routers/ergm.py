"""Exponential Random Graph Models (ERGM) and SIENA-style network modeling.

Implements pseudolikelihood ERGM estimation via logistic regression on change
statistics, goodness-of-fit via network simulation, network generation from
ERGM parameters, and simplified stochastic actor-oriented models (SIENA).
Uses only numpy, scipy, and standard library.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class ERGMFitRequest(BaseModel):
    """Fit an ERGM via pseudolikelihood estimation."""
    adjacency: list[list[int]]  # n×n binary adjacency matrix (0/1)
    terms: list[str]  # network statistics: edges, mutual, triangles, kstar2, kstar3, nodecov, nodematch
    node_attributes: list[float] | None = None  # attribute per node (for nodecov/nodematch)
    directed: bool = False


class ERGMGofRequest(BaseModel):
    """Goodness-of-fit assessment for a fitted ERGM."""
    adjacency: list[list[int]]
    parameters: dict[str, float]  # term -> coefficient from /fit
    directed: bool = False
    n_simulations: int = 100


class ERGMSimulateRequest(BaseModel):
    """Simulate networks from ERGM parameters."""
    n_nodes: int
    parameters: dict[str, float]  # term -> coefficient
    directed: bool = False
    n_iterations: int = 5000
    seed: int | None = None


class SIENARequest(BaseModel):
    """Simplified stochastic actor-oriented model (two time points)."""
    adjacency_t1: list[list[int]]  # adjacency at time 1
    adjacency_t2: list[list[int]]  # adjacency at time 2
    node_attributes: list[float] | None = None  # optional covariate per node
    directed: bool = True
    n_iterations: int = 200


# ── Change Statistic Helpers ──────────────────────────────────────────

def _compute_network_stat(A: np.ndarray, term: str,
                          node_attrs: np.ndarray | None = None,
                          directed: bool = False) -> float:
    """Compute a single network statistic on adjacency matrix A."""
    n = A.shape[0]
    if term == "edges":
        if directed:
            return float(np.sum(A))
        return float(np.sum(A) / 2)
    elif term == "mutual":
        return float(np.sum(A * A.T) / 2)
    elif term == "triangles":
        A2 = A @ A
        return float(np.trace(A @ A2) / 6)
    elif term == "kstar2":
        # number of 2-stars
        deg = np.sum(A, axis=1) if directed else np.sum(A, axis=1)
        return float(np.sum(deg * (deg - 1) / 2))
    elif term == "kstar3":
        deg = np.sum(A, axis=1)
        return float(np.sum(deg * (deg - 1) * (deg - 2) / 6))
    elif term == "nodecov":
        if node_attrs is None:
            return 0.0
        # sum of attribute values for nodes involved in edges
        s = 0.0
        for i in range(n):
            for j in range(i + 1 if not directed else 0, n):
                if A[i, j]:
                    s += node_attrs[i] + node_attrs[j]
        return s
    elif term == "nodematch":
        if node_attrs is None:
            return 0.0
        # number of edges where both nodes have the same attribute
        s = 0.0
        for i in range(n):
            for j in range(i + 1 if not directed else 0, n):
                if A[i, j] and node_attrs[i] == node_attrs[j]:
                    s += 1.0
        return s
    return 0.0


def _change_statistic(A: np.ndarray, i: int, j: int, term: str,
                      node_attrs: np.ndarray | None = None,
                      directed: bool = False) -> float:
    """Compute change statistic delta(term) when toggling tie (i,j).

    Change stat = stat(A with (i,j)=1) - stat(A with (i,j)=0).
    """
    n = A.shape[0]
    current = A[i, j]

    if term == "edges":
        return 1.0 if current == 0 else -1.0

    elif term == "mutual":
        if not directed:
            return 0.0
        # mutual changes only if the reciprocal tie exists
        toggle = 1.0 if current == 0 else -1.0
        return toggle * A[j, i]

    elif term == "triangles":
        # number of shared partners
        toggle = 1.0 if current == 0 else -1.0
        shared = 0
        for k in range(n):
            if k == i or k == j:
                continue
            if not directed:
                shared += int(A[i, k] and A[j, k])
            else:
                shared += int(A[i, k] and A[k, j])
                shared += int(A[k, i] and A[j, k])
        return toggle * shared

    elif term == "kstar2":
        toggle = 1.0 if current == 0 else -1.0
        # adding tie (i,j) creates new 2-stars centered at i and j
        deg_i = np.sum(A[i, :]) - current  # degree without (i,j)
        deg_j = np.sum(A[:, j]) if directed else np.sum(A[j, :]) - current
        return toggle * (deg_i + deg_j)

    elif term == "kstar3":
        toggle = 1.0 if current == 0 else -1.0
        deg_i = np.sum(A[i, :]) - current
        deg_j = np.sum(A[:, j]) if directed else np.sum(A[j, :]) - current
        # new 3-stars from adding a tie
        return toggle * (deg_i * (deg_i - 1) / 2 + deg_j * (deg_j - 1) / 2)

    elif term == "nodecov":
        if node_attrs is None:
            return 0.0
        toggle = 1.0 if current == 0 else -1.0
        return toggle * (node_attrs[i] + node_attrs[j])

    elif term == "nodematch":
        if node_attrs is None:
            return 0.0
        toggle = 1.0 if current == 0 else -1.0
        return toggle * (1.0 if node_attrs[i] == node_attrs[j] else 0.0)

    return 0.0


def _change_statistics_vector(A: np.ndarray, i: int, j: int,
                              terms: list[str],
                              node_attrs: np.ndarray | None,
                              directed: bool) -> np.ndarray:
    """Compute vector of change statistics for all terms."""
    return np.array([
        _change_statistic(A, i, j, t, node_attrs, directed)
        for t in terms
    ])


# ── MCMC simulation helpers ──────────────────────────────────────────

def _simulate_network_mcmc(n: int, params: dict[str, float],
                           directed: bool, n_iterations: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Simulate a network from ERGM parameters using Gibbs sampling."""
    terms = list(params.keys())
    theta = np.array([params[t] for t in terms])
    A = np.zeros((n, n), dtype=int)

    for _ in range(n_iterations):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        if not directed and j < i:
            i, j = j, i

        delta = _change_statistics_vector(A, i, j, terms, None, directed)

        if A[i, j] == 0:
            log_odds = float(theta @ delta)
        else:
            log_odds = -float(theta @ delta)

        prob = 1.0 / (1.0 + np.exp(-log_odds))
        if rng.random() < prob:
            if A[i, j] == 0:
                A[i, j] = 1
                if not directed:
                    A[j, i] = 1
            else:
                A[i, j] = 0
                if not directed:
                    A[j, i] = 0

    return A


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/fit")
def fit_ergm(req: ERGMFitRequest):
    """Fit an ERGM via pseudolikelihood estimation.

    Uses logistic regression on change statistics as a tractable
    approximation to MCMC-MLE.  Returns parameter estimates, standard
    errors, z-values, p-values, AIC/BIC, and a goodness-of-fit summary.
    """
    A = np.array(req.adjacency, dtype=int)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise HTTPException(422, "Adjacency matrix must be square.")
    if n < 3:
        raise HTTPException(422, "Network must have at least 3 nodes.")

    node_attrs = np.array(req.node_attributes) if req.node_attributes else None
    terms = req.terms
    if not terms:
        raise HTTPException(422, "Must specify at least one network term.")

    valid_terms = {"edges", "mutual", "triangles", "kstar2", "kstar3",
                   "nodecov", "nodematch"}
    for t in terms:
        if t not in valid_terms:
            raise HTTPException(422, f"Unknown term '{t}'. Valid: {sorted(valid_terms)}")

    if ("nodecov" in terms or "nodematch" in terms) and node_attrs is None:
        raise HTTPException(422, "node_attributes required for nodecov/nodematch terms.")

    # Build pseudolikelihood design matrix
    # For each dyad (i,j), y = A[i,j], X = change statistics
    pairs = []
    if req.directed:
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((i, j))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

    n_obs = len(pairs)
    p = len(terms)
    X = np.zeros((n_obs, p))
    y = np.zeros(n_obs)

    for idx, (i, j) in enumerate(pairs):
        y[idx] = A[i, j]
        # Change statistics computed with the tie set to 0
        A_tmp = A.copy()
        A_tmp[i, j] = 0
        if not req.directed:
            A_tmp[j, i] = 0
        X[idx] = _change_statistics_vector(A_tmp, i, j, terms, node_attrs, req.directed)

    # Logistic regression via maximum likelihood
    def neg_log_lik(theta):
        eta = X @ theta
        eta = np.clip(eta, -500, 500)
        ll = np.sum(y * eta - np.log(1 + np.exp(eta)))
        return -ll

    def neg_log_lik_grad(theta):
        eta = X @ theta
        eta = np.clip(eta, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-eta))
        return -X.T @ (y - prob)

    theta0 = np.zeros(p)
    result = minimize(neg_log_lik, theta0, jac=neg_log_lik_grad,
                      method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-10})
    theta_hat = result.x

    # Standard errors from inverse Fisher information (Hessian)
    eta = X @ theta_hat
    eta = np.clip(eta, -500, 500)
    prob = 1.0 / (1.0 + np.exp(-eta))
    W = prob * (1.0 - prob)
    H = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.abs(np.diag(cov)))

    z_vals = theta_hat / se
    p_vals = 2 * (sp_stats.norm.sf(np.abs(z_vals)))

    # AIC / BIC
    ll_fit = -neg_log_lik(theta_hat)
    aic = -2 * ll_fit + 2 * p
    bic = -2 * ll_fit + np.log(n_obs) * p

    # Observed network statistics
    obs_stats = {t: _compute_network_stat(A, t, node_attrs, req.directed) for t in terms}

    coefficients = {}
    for k, t in enumerate(terms):
        coefficients[t] = {
            "estimate": float(theta_hat[k]),
            "se": float(se[k]),
            "z": float(z_vals[k]),
            "p_value": float(p_vals[k]),
        }

    return {
        "coefficients": coefficients,
        "aic": float(aic),
        "bic": float(bic),
        "log_likelihood": float(ll_fit),
        "n_dyads": n_obs,
        "n_nodes": n,
        "observed_statistics": obs_stats,
        "convergence": bool(result.success),
        "interpretation": _interpret_ergm(coefficients),
    }


@router.post("/gof")
def ergm_gof(req: ERGMGofRequest):
    """Goodness-of-fit for a fitted ERGM.

    Simulates networks from the fitted model and compares degree
    distribution, geodesic distance distribution, and edgewise shared
    partner distribution to the observed network.
    """
    A_obs = np.array(req.adjacency, dtype=int)
    n = A_obs.shape[0]
    if A_obs.shape[0] != A_obs.shape[1]:
        raise HTTPException(422, "Adjacency matrix must be square.")
    if not req.parameters:
        raise HTTPException(422, "Must provide fitted parameters.")

    rng = np.random.default_rng(42)
    n_sim = req.n_simulations
    n_iter = max(n * n * 5, 5000)

    # Observed statistics
    obs_degree = _degree_distribution(A_obs, req.directed)
    obs_geodesic = _geodesic_distribution(A_obs)
    obs_esp = _esp_distribution(A_obs, req.directed)

    max_deg = max(obs_degree.keys()) if obs_degree else 0
    max_geo = max(obs_geodesic.keys()) if obs_geodesic else 0
    max_esp_k = max(obs_esp.keys()) if obs_esp else 0

    sim_degrees = []
    sim_geodesics = []
    sim_esps = []

    for _ in range(n_sim):
        A_sim = _simulate_network_mcmc(n, req.parameters, req.directed,
                                       n_iter, rng)
        sd = _degree_distribution(A_sim, req.directed)
        sg = _geodesic_distribution(A_sim)
        se = _esp_distribution(A_sim, req.directed)
        max_deg = max(max_deg, max(sd.keys()) if sd else 0)
        max_geo = max(max_geo, max(sg.keys()) if sg else 0)
        max_esp_k = max(max_esp_k, max(se.keys()) if se else 0)
        sim_degrees.append(sd)
        sim_geodesics.append(sg)
        sim_esps.append(se)

    # Compile results
    degree_gof = _compile_gof(obs_degree, sim_degrees, max_deg, "degree")
    geodesic_gof = _compile_gof(obs_geodesic, sim_geodesics, max_geo, "geodesic")
    esp_gof = _compile_gof(obs_esp, sim_esps, max_esp_k, "esp")

    return {
        "n_simulations": n_sim,
        "degree_distribution": degree_gof,
        "geodesic_distance_distribution": geodesic_gof,
        "edgewise_shared_partners": esp_gof,
        "interpretation": (
            "Compare observed values to simulated means. If observed values "
            "fall within 2 SDs of simulated means across most categories, "
            "the model fits well."
        ),
    }


@router.post("/simulate")
def simulate_ergm(req: ERGMSimulateRequest):
    """Simulate a network from ERGM parameters via MCMC (Gibbs sampling).

    Returns the simulated adjacency matrix and summary network statistics.
    """
    if req.n_nodes < 3:
        raise HTTPException(422, "n_nodes must be at least 3.")
    if not req.parameters:
        raise HTTPException(422, "Must provide at least one parameter.")

    rng = np.random.default_rng(req.seed)
    A = _simulate_network_mcmc(req.n_nodes, req.parameters, req.directed,
                               req.n_iterations, rng)

    terms = list(req.parameters.keys())
    sim_stats = {}
    for t in terms:
        sim_stats[t] = _compute_network_stat(A, t, None, req.directed)

    n = req.n_nodes
    degrees = np.sum(A, axis=1).tolist()
    density = float(np.sum(A)) / (n * (n - 1)) if req.directed else float(np.sum(A)) / (n * (n - 1))

    return {
        "adjacency": A.tolist(),
        "n_nodes": n,
        "n_edges": int(np.sum(A)) if req.directed else int(np.sum(A) // 2),
        "density": density,
        "degree_sequence": degrees,
        "network_statistics": sim_stats,
    }


@router.post("/siena")
def siena_model(req: SIENARequest):
    """Simplified stochastic actor-oriented model (SIENA) for two time points.

    Estimates rate parameter, outdegree, reciprocity, and transitivity
    effects via method-of-moments approximation. Returns parameter
    estimates, standard errors, and convergence t-ratios.
    """
    A1 = np.array(req.adjacency_t1, dtype=int)
    A2 = np.array(req.adjacency_t2, dtype=int)
    n = A1.shape[0]

    if A1.shape != A2.shape:
        raise HTTPException(422, "Both adjacency matrices must have the same dimensions.")
    if A1.shape[0] != A1.shape[1]:
        raise HTTPException(422, "Adjacency matrices must be square.")
    if n < 3:
        raise HTTPException(422, "Network must have at least 3 nodes.")

    node_attrs = np.array(req.node_attributes) if req.node_attributes else None

    # Observed statistics
    n_changes = int(np.sum(np.abs(A2 - A1)))
    if not req.directed:
        n_changes //= 2

    # Rate parameter: average number of opportunities for change per actor
    rate = n_changes / n if n > 0 else 0.0

    # Compute target statistics at time 2
    n_edges_2 = int(np.sum(A2)) if req.directed else int(np.sum(A2) // 2)
    n_mutual_2 = int(np.sum(A2 * A2.T) // 2)
    n_transitive_2 = 0
    for i in range(n):
        for j in range(n):
            if i != j and A2[i, j]:
                for h in range(n):
                    if h != i and h != j:
                        n_transitive_2 += int(A2[i, h] and A2[h, j])

    # Method-of-moments estimation via simulation
    # We approximate parameters using iterative updating
    effects = ["rate", "outdegree", "reciprocity", "transitivity"]
    if node_attrs is not None:
        effects.append("covariate")

    n_effects = len(effects)
    theta = np.zeros(n_effects)
    theta[0] = rate  # initial rate
    theta[1] = -1.0  # outdegree (negative = cost of ties)

    # Target statistics
    targets = np.array([
        n_changes,
        n_edges_2,
        n_mutual_2,
        n_transitive_2,
    ])
    if node_attrs is not None:
        # covariate effect: sum of attr for tie receivers
        cov_stat = sum(node_attrs[j] for i in range(n) for j in range(n) if i != j and A2[i, j])
        targets = np.append(targets, cov_stat)

    rng = np.random.default_rng(42)
    n_sim_per_iter = 20
    n_iter = req.n_iterations

    # Simplified Robbins-Monro updating
    gain = 0.1
    sim_stats_all = []

    for iteration in range(n_iter):
        # Simulate from current parameters
        sim_stats_batch = []
        for _ in range(n_sim_per_iter):
            sim_stats = _simulate_siena_step(A1, theta, effects, node_attrs,
                                             req.directed, rng)
            sim_stats_batch.append(sim_stats)

        mean_sim = np.mean(sim_stats_batch, axis=0)
        sim_stats_all.append(mean_sim)

        # Update: Robbins-Monro step
        step = gain / (iteration + 1) * (mean_sim - targets)
        theta -= step

    # Estimate SE from simulation variability
    if len(sim_stats_all) > 10:
        recent = np.array(sim_stats_all[-20:])
        se = np.std(recent, axis=0)
        se = np.where(se > 0, se, 0.01)
    else:
        se = np.ones(n_effects) * 0.1

    # Convergence t-ratios: (simulated mean - target) / simulated SD
    final_sims = []
    for _ in range(50):
        s = _simulate_siena_step(A1, theta, effects, node_attrs, req.directed, rng)
        final_sims.append(s)
    final_mean = np.mean(final_sims, axis=0)
    final_sd = np.std(final_sims, axis=0)
    final_sd = np.where(final_sd > 0, final_sd, 0.01)
    t_ratios = (final_mean - targets) / final_sd

    converged = bool(np.all(np.abs(t_ratios) < 0.1))

    parameters = {}
    for k, name in enumerate(effects):
        parameters[name] = {
            "estimate": float(theta[k]),
            "se": float(se[k]),
            "t_ratio": float(t_ratios[k]),
        }

    return {
        "parameters": parameters,
        "n_nodes": n,
        "n_changes": n_changes,
        "converged": converged,
        "convergence_criterion": "all |t-ratio| < 0.1",
        "target_statistics": {effects[k]: float(targets[k]) for k in range(n_effects)},
        "interpretation": _interpret_siena(parameters),
    }


# ── Private Helpers ───────────────────────────────────────────────────

def _degree_distribution(A: np.ndarray, directed: bool) -> dict[int, int]:
    """Compute degree distribution."""
    deg = np.sum(A, axis=1)
    dist = {}
    for d in deg:
        d_int = int(d)
        dist[d_int] = dist.get(d_int, 0) + 1
    return dist


def _geodesic_distribution(A: np.ndarray) -> dict[int, int]:
    """Compute geodesic (shortest-path) distance distribution via BFS."""
    n = A.shape[0]
    dist_count = {}
    for source in range(n):
        visited = {source: 0}
        queue = [source]
        while queue:
            node = queue.pop(0)
            d = visited[node]
            for neighbor in range(n):
                if A[node, neighbor] and neighbor not in visited:
                    visited[neighbor] = d + 1
                    queue.append(neighbor)
                    dist_count[d + 1] = dist_count.get(d + 1, 0) + 1
    return dist_count


def _esp_distribution(A: np.ndarray, directed: bool) -> dict[int, int]:
    """Edgewise shared partners distribution."""
    n = A.shape[0]
    esp_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                shared = 0
                for k in range(n):
                    if k != i and k != j and A[i, k] and A[j, k]:
                        shared += 1
                esp_count[shared] = esp_count.get(shared, 0) + 1
    return esp_count


def _compile_gof(obs: dict, sims: list[dict], max_val: int,
                 label: str) -> list[dict]:
    """Compile goodness-of-fit table for one statistic."""
    results = []
    for k in range(max_val + 1):
        obs_val = obs.get(k, 0)
        sim_vals = [s.get(k, 0) for s in sims]
        sim_mean = float(np.mean(sim_vals))
        sim_sd = float(np.std(sim_vals))
        if sim_sd > 0:
            p_val = float(2 * min(
                np.mean(np.array(sim_vals) >= obs_val),
                np.mean(np.array(sim_vals) <= obs_val),
            ))
        else:
            p_val = 1.0 if obs_val == sim_mean else 0.0
        results.append({
            label: k,
            "observed": obs_val,
            "simulated_mean": round(sim_mean, 3),
            "simulated_sd": round(sim_sd, 3),
            "p_value": round(min(p_val, 1.0), 4),
        })
    return results


def _simulate_siena_step(A1: np.ndarray, theta: np.ndarray,
                         effects: list[str],
                         node_attrs: np.ndarray | None,
                         directed: bool,
                         rng: np.random.Generator) -> np.ndarray:
    """Simulate one SIENA network evolution from A1 using current parameters.

    Returns vector of summary statistics matching the effects list.
    """
    n = A1.shape[0]
    A = A1.copy()
    rate = max(abs(theta[0]), 0.1)
    n_steps = int(rate * n)

    for _ in range(n_steps):
        actor = rng.integers(0, n)
        # Evaluate objective function change for each possible tie toggle
        best_j = -1
        best_val = -np.inf
        candidates = [j for j in range(n) if j != actor]
        # Sample a subset for speed
        if len(candidates) > 10:
            candidates = rng.choice(candidates, size=10, replace=False).tolist()

        for j in candidates:
            val = _eval_objective(A, actor, j, theta, effects, node_attrs, directed)
            exp_val = np.exp(min(val, 500))
            if rng.random() * (1 + exp_val) < exp_val:
                if exp_val > best_val:
                    best_val = exp_val
                    best_j = j

        if best_j >= 0:
            A[actor, best_j] = 1 - A[actor, best_j]
            if not directed:
                A[best_j, actor] = A[actor, best_j]

    # Compute statistics
    n_changes = int(np.sum(np.abs(A - A1)))
    if not directed:
        n_changes //= 2
    n_edges = int(np.sum(A)) if directed else int(np.sum(A) // 2)
    n_mutual = int(np.sum(A * A.T) // 2)
    n_trans = 0
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j]:
                for h in range(n):
                    if h != i and h != j:
                        n_trans += int(A[i, h] and A[h, j])

    stats = np.array([n_changes, n_edges, n_mutual, n_trans], dtype=float)
    if node_attrs is not None:
        cov_stat = sum(node_attrs[j] for i in range(n) for j in range(n) if i != j and A[i, j])
        stats = np.append(stats, cov_stat)
    return stats


def _eval_objective(A: np.ndarray, actor: int, alter: int,
                    theta: np.ndarray, effects: list[str],
                    node_attrs: np.ndarray | None,
                    directed: bool) -> float:
    """Evaluate objective function change for toggling tie (actor, alter)."""
    # theta[0] is rate (skip), start from index 1
    val = 0.0
    n = A.shape[0]

    # outdegree effect (index 1)
    toggle = 1.0 if A[actor, alter] == 0 else -1.0
    val += theta[1] * toggle

    # reciprocity (index 2)
    if len(theta) > 2:
        val += theta[2] * toggle * A[alter, actor]

    # transitivity (index 3)
    if len(theta) > 3:
        shared = 0
        for k in range(n):
            if k != actor and k != alter:
                shared += int(A[actor, k] and A[k, alter])
        val += theta[3] * toggle * shared

    # covariate (index 4)
    if len(theta) > 4 and node_attrs is not None:
        val += theta[4] * toggle * node_attrs[alter]

    return val


def _interpret_ergm(coefficients: dict) -> str:
    """Generate plain-language interpretation of ERGM results."""
    parts = []
    for term, vals in coefficients.items():
        est = vals["estimate"]
        sig = vals["p_value"] < 0.05
        direction = "increases" if est > 0 else "decreases"
        sig_str = "significantly " if sig else "non-significantly "

        if term == "edges":
            parts.append(
                f"The baseline tendency to form ties is {sig_str}"
                f"{'positive' if est > 0 else 'negative'} "
                f"(coef = {est:.3f}, p = {vals['p_value']:.4f})."
            )
        elif term == "mutual":
            parts.append(
                f"Reciprocity {sig_str}{direction} tie formation "
                f"(coef = {est:.3f}, p = {vals['p_value']:.4f})."
            )
        elif term == "triangles":
            parts.append(
                f"Transitivity (triangle closure) {sig_str}{direction} "
                f"tie probability (coef = {est:.3f}, p = {vals['p_value']:.4f})."
            )
        elif term == "kstar2":
            parts.append(
                f"2-star effects (degree heterogeneity) {sig_str}present "
                f"(coef = {est:.3f}, p = {vals['p_value']:.4f})."
            )
        else:
            parts.append(
                f"{term}: coef = {est:.3f}, p = {vals['p_value']:.4f} "
                f"({sig_str}{'positive' if est > 0 else 'negative'})."
            )
    return " ".join(parts)


def _interpret_siena(parameters: dict) -> str:
    """Generate plain-language interpretation of SIENA results."""
    parts = []
    for name, vals in parameters.items():
        est = vals["estimate"]
        converged = abs(vals["t_ratio"]) < 0.1
        conv_str = "well-converged" if converged else "not fully converged"

        if name == "rate":
            parts.append(
                f"Rate parameter = {est:.3f} ({conv_str}): actors had "
                f"approximately {est:.1f} opportunities to change ties."
            )
        elif name == "outdegree":
            parts.append(
                f"Outdegree effect = {est:.3f} ({conv_str}): "
                f"{'negative value indicates a cost to forming ties' if est < 0 else 'positive value indicates a tendency to form ties'}."
            )
        elif name == "reciprocity":
            parts.append(
                f"Reciprocity = {est:.3f} ({conv_str}): "
                f"{'actors tend to reciprocate ties' if est > 0 else 'no strong reciprocity tendency'}."
            )
        elif name == "transitivity":
            parts.append(
                f"Transitivity = {est:.3f} ({conv_str}): "
                f"{'actors tend to close triangles (friends of friends become friends)' if est > 0 else 'no transitivity tendency'}."
            )
        elif name == "covariate":
            parts.append(
                f"Covariate effect = {est:.3f} ({conv_str}): "
                f"{'higher attribute values attract more incoming ties' if est > 0 else 'attribute does not attract ties'}."
            )
    return " ".join(parts)
