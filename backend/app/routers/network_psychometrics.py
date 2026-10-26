"""Network Psychometrics — psychological network analysis (Borsboom, Epskamp)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    data: list[list[float]]  # rows = subjects, cols = variables
    variable_names: list[str] | None = None
    method: str = "partial"  # "partial" or "correlation"
    threshold: float | None = None  # optional manual threshold for correlation method

class CentralityRequest(BaseModel):
    adjacency: list[list[float]]  # weighted adjacency matrix
    variable_names: list[str] | None = None

class StabilityRequest(BaseModel):
    data: list[list[float]]  # raw data (subjects × variables)
    variable_names: list[str] | None = None
    method: str = "partial"
    n_boots: int = 100
    drop_proportions: list[float] | None = None  # defaults to [0.10, 0.20, ..., 0.75]

class BridgeRequest(BaseModel):
    adjacency: list[list[float]]
    communities: list[int]  # community assignment per node (0-indexed)
    variable_names: list[str] | None = None

class CompareRequest(BaseModel):
    data_group1: list[list[float]]  # group 1: subjects × variables
    data_group2: list[list[float]]  # group 2: subjects × variables
    variable_names: list[str] | None = None
    method: str = "partial"
    n_permutations: int = 1000


# ── Helpers ───────────────────────────────────────────────────────────

def _variable_names(names: list[str] | None, p: int) -> list[str]:
    """Return variable names, generating defaults if not provided."""
    if names and len(names) == p:
        return names
    return [f"V{i+1}" for i in range(p)]


def _partial_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix via precision matrix inversion."""
    cov = np.cov(data, rowvar=False)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    diag = np.sqrt(np.diag(prec))
    diag[diag == 0] = 1.0
    pcor = -prec / np.outer(diag, diag)
    np.fill_diagonal(pcor, 0.0)
    return pcor


def _correlation_network_bic(data: np.ndarray) -> np.ndarray:
    """Correlation network with BIC-based edge selection.

    For each pair (i, j), include the edge only if the partial correlation
    improves BIC over the null (no edge) model.
    """
    n, p = data.shape
    corr = np.corrcoef(data, rowvar=False)
    adj = np.zeros((p, p))

    for i in range(p):
        for j in range(i + 1, p):
            r = corr[i, j]
            # Null model: no edge (residual variance = var of y)
            var_y = np.var(data[:, j], ddof=1)
            if var_y <= 0:
                continue
            ss_null = np.sum((data[:, j] - np.mean(data[:, j])) ** 2)
            ll_null = -n / 2.0 * np.log(2 * np.pi * ss_null / n) - n / 2.0
            bic_null = -2 * ll_null + 1 * np.log(n)  # 1 param: intercept

            # Edge model: linear regression y ~ x
            x = data[:, i]
            y = data[:, j]
            slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2) if np.var(x) > 0 else 0
            intercept = y.mean() - slope * x.mean()
            residuals = y - (slope * x + intercept)
            ss_edge = np.sum(residuals ** 2)
            if ss_edge <= 0:
                ss_edge = 1e-300
            ll_edge = -n / 2.0 * np.log(2 * np.pi * ss_edge / n) - n / 2.0
            bic_edge = -2 * ll_edge + 2 * np.log(n)  # 2 params: intercept + slope

            if bic_edge < bic_null:
                adj[i, j] = r
                adj[j, i] = r

    return adj


def _adjacency_to_edgelist(adj: np.ndarray, names: list[str]) -> list[dict]:
    """Convert adjacency matrix to list of edges with weights."""
    edges = []
    p = adj.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            w = adj[i, j]
            if abs(w) > 1e-10:
                edges.append({
                    "from": names[i],
                    "to": names[j],
                    "weight": float(w),
                })
    return edges


def _shortest_paths(adj: np.ndarray) -> np.ndarray:
    """Floyd-Warshall shortest paths on absolute inverse weights."""
    p = adj.shape[0]
    dist = np.full((p, p), np.inf)
    np.fill_diagonal(dist, 0.0)
    for i in range(p):
        for j in range(p):
            if i != j and abs(adj[i, j]) > 1e-10:
                dist[i, j] = 1.0 / abs(adj[i, j])
    for k in range(p):
        for i in range(p):
            for j in range(p):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist


def _betweenness(adj: np.ndarray) -> np.ndarray:
    """Betweenness centrality via counting shortest paths (unweighted on presence)."""
    p = adj.shape[0]
    betw = np.zeros(p)
    # Use inverse absolute weight as distance
    dist = _shortest_paths(adj)

    for s in range(p):
        for t in range(p):
            if s == t:
                continue
            if np.isinf(dist[s, t]):
                continue
            for v in range(p):
                if v == s or v == t:
                    continue
                # Check if v is on a shortest path from s to t
                if abs(dist[s, v] + dist[v, t] - dist[s, t]) < 1e-10:
                    # Count all shortest paths through v (approximation: assume unique)
                    betw[v] += 1.0
    # Normalize
    denom = (p - 1) * (p - 2)
    if denom > 0:
        betw /= denom
    return betw


def _compute_centrality(adj: np.ndarray, names: list[str]) -> dict:
    """Compute all four centrality measures."""
    p = adj.shape[0]

    # Strength centrality
    strength = np.sum(np.abs(adj), axis=1)

    # Expected influence (signed)
    expected_influence = np.sum(adj, axis=1)

    # Closeness centrality
    dist = _shortest_paths(adj)
    closeness = np.zeros(p)
    for i in range(p):
        reachable = dist[i, :][dist[i, :] < np.inf]
        reachable = reachable[reachable > 0]
        if len(reachable) > 0:
            closeness[i] = len(reachable) / np.sum(reachable)

    # Betweenness
    betw = _betweenness(adj)

    # Standardize (z-scores)
    def _zscore(arr):
        m, s = arr.mean(), arr.std()
        return ((arr - m) / s if s > 0 else np.zeros_like(arr)).tolist()

    table = []
    for i in range(p):
        table.append({
            "variable": names[i],
            "strength": float(strength[i]),
            "closeness": float(closeness[i]),
            "betweenness": float(betw[i]),
            "expected_influence": float(expected_influence[i]),
        })

    z_strength = _zscore(strength)
    z_closeness = _zscore(closeness)
    z_betweenness = _zscore(betw)
    z_ei = _zscore(expected_influence)

    standardized = []
    for i in range(p):
        standardized.append({
            "variable": names[i],
            "strength": z_strength[i],
            "closeness": z_closeness[i],
            "betweenness": z_betweenness[i],
            "expected_influence": z_ei[i],
        })

    most_central = {
        "strength": names[int(np.argmax(strength))],
        "closeness": names[int(np.argmax(closeness))],
        "betweenness": names[int(np.argmax(betw))],
        "expected_influence": names[int(np.argmax(np.abs(expected_influence)))],
    }

    return {
        "centrality_table": table,
        "standardized_centrality": standardized,
        "most_central_nodes": most_central,
    }


def _estimate_network(data: np.ndarray, method: str) -> np.ndarray:
    """Estimate network adjacency matrix using specified method."""
    if method == "partial":
        return _partial_correlation_matrix(data)
    else:
        return _correlation_network_bic(data)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/estimate")
def estimate_network(req: EstimateRequest):
    """Estimate a psychological network from multivariate data.

    Supports partial correlation networks and correlation networks with
    BIC-based edge selection (LASSO-like thresholding).
    """
    data = np.array(req.data)
    if data.ndim != 2:
        raise HTTPException(400, "Data must be a 2D array (subjects × variables).")
    n, p = data.shape
    if n < 3:
        raise HTTPException(400, "Need at least 3 subjects.")
    if p < 2:
        raise HTTPException(400, "Need at least 2 variables.")
    if req.method not in ("partial", "correlation"):
        raise HTTPException(400, "Method must be 'partial' or 'correlation'.")

    names = _variable_names(req.variable_names, p)

    if req.method == "partial":
        adj = _partial_correlation_matrix(data)
    else:
        if req.threshold is not None:
            # Simple threshold on correlation matrix
            corr = np.corrcoef(data, rowvar=False)
            np.fill_diagonal(corr, 0.0)
            adj = np.where(np.abs(corr) >= req.threshold, corr, 0.0)
        else:
            adj = _correlation_network_bic(data)

    edges = _adjacency_to_edgelist(adj, names)
    max_edges = p * (p - 1) // 2
    n_edges = len(edges)
    sparsity = 1.0 - (n_edges / max_edges) if max_edges > 0 else 1.0

    return {
        "method": req.method,
        "n_subjects": n,
        "n_variables": p,
        "variable_names": names,
        "adjacency_matrix": adj.tolist(),
        "edge_list": edges,
        "n_edges": n_edges,
        "max_possible_edges": max_edges,
        "sparsity": float(sparsity),
    }


@router.post("/centrality")
def centrality_analysis(req: CentralityRequest):
    """Compute network centrality indices: strength, closeness, betweenness, expected influence."""
    adj = np.array(req.adjacency)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise HTTPException(400, "Adjacency must be a square 2D matrix.")
    p = adj.shape[0]
    if p < 2:
        raise HTTPException(400, "Need at least 2 nodes.")

    names = _variable_names(req.variable_names, p)
    result = _compute_centrality(adj, names)
    result["n_nodes"] = p
    return result


@router.post("/stability")
def stability_analysis(req: StabilityRequest):
    """Case-dropping bootstrap for network centrality stability.

    Computes the CS-coefficient: largest proportion of cases that can be
    dropped while maintaining centrality correlation with original >= 0.7.
    """
    data = np.array(req.data)
    if data.ndim != 2:
        raise HTTPException(400, "Data must be a 2D array (subjects × variables).")
    n, p = data.shape
    if n < 10:
        raise HTTPException(400, "Need at least 10 subjects for stability analysis.")
    if p < 2:
        raise HTTPException(400, "Need at least 2 variables.")

    names = _variable_names(req.variable_names, p)
    n_boots = max(10, min(req.n_boots, 500))

    drop_proportions = req.drop_proportions or [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]
    drop_proportions = [dp for dp in drop_proportions if 0 < dp < 1]

    # Estimate original network
    orig_adj = _estimate_network(data, req.method)
    orig_strength = np.sum(np.abs(orig_adj), axis=1)
    orig_ei = np.sum(orig_adj, axis=1)

    stability_curves = {"strength": [], "expected_influence": []}

    for drop_prop in sorted(drop_proportions):
        keep_n = max(p + 1, int(n * (1.0 - drop_prop)))
        cors_strength = []
        cors_ei = []

        for _ in range(n_boots):
            idx = np.random.choice(n, size=keep_n, replace=False)
            sub_data = data[idx, :]
            try:
                sub_adj = _estimate_network(sub_data, req.method)
                sub_strength = np.sum(np.abs(sub_adj), axis=1)
                sub_ei = np.sum(sub_adj, axis=1)
                r_s = np.corrcoef(orig_strength, sub_strength)[0, 1]
                r_e = np.corrcoef(orig_ei, sub_ei)[0, 1]
                if not np.isnan(r_s):
                    cors_strength.append(r_s)
                if not np.isnan(r_e):
                    cors_ei.append(r_e)
            except Exception:
                continue

        mean_r_s = float(np.mean(cors_strength)) if cors_strength else 0.0
        mean_r_e = float(np.mean(cors_ei)) if cors_ei else 0.0

        stability_curves["strength"].append({
            "drop_proportion": float(drop_prop),
            "keep_n": keep_n,
            "mean_correlation": mean_r_s,
            "n_successful_boots": len(cors_strength),
        })
        stability_curves["expected_influence"].append({
            "drop_proportion": float(drop_prop),
            "keep_n": keep_n,
            "mean_correlation": mean_r_e,
            "n_successful_boots": len(cors_ei),
        })

    # CS-coefficient: largest drop proportion where mean correlation >= 0.7
    def _cs_coeff(curve):
        cs = 0.0
        for pt in curve:
            if pt["mean_correlation"] >= 0.7:
                cs = pt["drop_proportion"]
            else:
                break
        return float(cs)

    cs_strength = _cs_coeff(stability_curves["strength"])
    cs_ei = _cs_coeff(stability_curves["expected_influence"])

    return {
        "n_subjects": n,
        "n_variables": p,
        "method": req.method,
        "n_bootstraps": n_boots,
        "cs_coefficient": {
            "strength": cs_strength,
            "expected_influence": cs_ei,
        },
        "interpretation": {
            "strength": (
                "Stable (CS >= 0.5)" if cs_strength >= 0.5 else
                "Moderate stability (CS >= 0.25)" if cs_strength >= 0.25 else
                "Unstable (CS < 0.25)"
            ),
            "expected_influence": (
                "Stable (CS >= 0.5)" if cs_ei >= 0.5 else
                "Moderate stability (CS >= 0.25)" if cs_ei >= 0.25 else
                "Unstable (CS < 0.25)"
            ),
        },
        "stability_curves": stability_curves,
    }


@router.post("/bridge")
def bridge_centrality(req: BridgeRequest):
    """Bridge centrality for community structure in networks.

    Identifies nodes that act as bridges connecting different communities.
    """
    adj = np.array(req.adjacency)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise HTTPException(400, "Adjacency must be a square 2D matrix.")
    p = adj.shape[0]
    communities = np.array(req.communities)
    if len(communities) != p:
        raise HTTPException(400, "Length of communities must equal number of nodes.")

    names = _variable_names(req.variable_names, p)
    unique_comms = np.unique(communities)
    n_communities = len(unique_comms)

    if n_communities < 2:
        raise HTTPException(400, "Need at least 2 communities for bridge centrality.")

    # Bridge strength: sum of absolute edge weights to nodes in OTHER communities
    bridge_strength = np.zeros(p)
    bridge_ei = np.zeros(p)
    for i in range(p):
        for j in range(p):
            if i != j and communities[i] != communities[j]:
                bridge_strength[i] += abs(adj[i, j])
                bridge_ei[i] += adj[i, j]

    # Bridge betweenness: betweenness restricted to paths crossing communities
    dist = _shortest_paths(adj)
    bridge_betw = np.zeros(p)
    for s in range(p):
        for t in range(p):
            if s == t or communities[s] == communities[t]:
                continue
            if np.isinf(dist[s, t]):
                continue
            for v in range(p):
                if v == s or v == t:
                    continue
                if abs(dist[s, v] + dist[v, t] - dist[s, t]) < 1e-10:
                    bridge_betw[v] += 1.0
    denom = (p - 1) * (p - 2)
    if denom > 0:
        bridge_betw /= denom

    # Bridge closeness: closeness restricted to nodes in other communities
    bridge_closeness = np.zeros(p)
    for i in range(p):
        other_dists = []
        for j in range(p):
            if communities[i] != communities[j] and not np.isinf(dist[i, j]) and dist[i, j] > 0:
                other_dists.append(dist[i, j])
        if other_dists:
            bridge_closeness[i] = len(other_dists) / sum(other_dists)

    # Build table
    table = []
    for i in range(p):
        table.append({
            "variable": names[i],
            "community": int(communities[i]),
            "bridge_strength": float(bridge_strength[i]),
            "bridge_betweenness": float(bridge_betw[i]),
            "bridge_closeness": float(bridge_closeness[i]),
            "bridge_expected_influence": float(bridge_ei[i]),
        })

    # Identify top bridge nodes (above mean + 1 SD on bridge strength)
    bs_mean = bridge_strength.mean()
    bs_std = bridge_strength.std()
    threshold_val = bs_mean + bs_std if bs_std > 0 else bs_mean
    bridge_nodes = [
        {"variable": names[i], "community": int(communities[i]),
         "bridge_strength": float(bridge_strength[i])}
        for i in range(p) if bridge_strength[i] > threshold_val
    ]

    return {
        "n_nodes": p,
        "n_communities": n_communities,
        "community_sizes": {int(c): int(np.sum(communities == c)) for c in unique_comms},
        "bridge_centrality_table": table,
        "identified_bridge_nodes": bridge_nodes,
    }


@router.post("/compare")
def compare_networks(req: CompareRequest):
    """Network comparison test between two groups via permutation testing.

    Tests differences in global network strength, individual edge weights,
    and individual node strengths.
    """
    d1 = np.array(req.data_group1)
    d2 = np.array(req.data_group2)
    if d1.ndim != 2 or d2.ndim != 2:
        raise HTTPException(400, "Both data matrices must be 2D.")
    if d1.shape[1] != d2.shape[1]:
        raise HTTPException(400, "Both groups must have the same number of variables.")
    p = d1.shape[1]
    n1, n2 = d1.shape[0], d2.shape[0]
    if n1 < 3 or n2 < 3:
        raise HTTPException(400, "Each group needs at least 3 subjects.")

    names = _variable_names(req.variable_names, p)
    n_perm = max(100, min(req.n_permutations, 5000))

    # Observed networks
    adj1 = _estimate_network(d1, req.method)
    adj2 = _estimate_network(d2, req.method)

    # Observed differences
    obs_global_diff = float(np.sum(np.abs(adj1)) - np.sum(np.abs(adj2)))
    obs_edge_diff = adj1 - adj2
    obs_node_strength_diff = np.sum(np.abs(adj1), axis=1) - np.sum(np.abs(adj2), axis=1)

    # Permutation test
    combined = np.vstack([d1, d2])
    n_total = n1 + n2
    perm_global = np.zeros(n_perm)
    perm_edge = np.zeros((n_perm, p, p))
    perm_node = np.zeros((n_perm, p))

    for b in range(n_perm):
        perm_idx = np.random.permutation(n_total)
        perm_d1 = combined[perm_idx[:n1], :]
        perm_d2 = combined[perm_idx[n1:], :]
        pa1 = _estimate_network(perm_d1, req.method)
        pa2 = _estimate_network(perm_d2, req.method)
        perm_global[b] = np.sum(np.abs(pa1)) - np.sum(np.abs(pa2))
        perm_edge[b] = pa1 - pa2
        perm_node[b] = np.sum(np.abs(pa1), axis=1) - np.sum(np.abs(pa2), axis=1)

    # p-values (two-tailed)
    p_global = float(np.mean(np.abs(perm_global) >= abs(obs_global_diff)))

    edge_pvalues = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            edge_pvalues[i, j] = float(np.mean(
                np.abs(perm_edge[:, i, j]) >= abs(obs_edge_diff[i, j])
            ))
            edge_pvalues[j, i] = edge_pvalues[i, j]

    node_pvalues = np.zeros(p)
    for i in range(p):
        node_pvalues[i] = float(np.mean(
            np.abs(perm_node[:, i]) >= abs(obs_node_strength_diff[i])
        ))

    # Significant edges
    sig_edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if edge_pvalues[i, j] < 0.05:
                sig_edges.append({
                    "from": names[i],
                    "to": names[j],
                    "observed_diff": float(obs_edge_diff[i, j]),
                    "p_value": float(edge_pvalues[i, j]),
                })

    # Node results
    node_results = []
    for i in range(p):
        node_results.append({
            "variable": names[i],
            "strength_group1": float(np.sum(np.abs(adj1[i, :]))),
            "strength_group2": float(np.sum(np.abs(adj2[i, :]))),
            "observed_diff": float(obs_node_strength_diff[i]),
            "p_value": float(node_pvalues[i]),
        })

    return {
        "n_group1": n1,
        "n_group2": n2,
        "n_variables": p,
        "method": req.method,
        "n_permutations": n_perm,
        "global_strength": {
            "group1": float(np.sum(np.abs(adj1))),
            "group2": float(np.sum(np.abs(adj2))),
            "observed_diff": obs_global_diff,
            "p_value": p_global,
        },
        "edge_differences": sig_edges,
        "node_strength_differences": node_results,
        "adjacency_group1": adj1.tolist(),
        "adjacency_group2": adj2.tolist(),
    }
