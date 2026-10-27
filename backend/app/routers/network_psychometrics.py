"""Network Psychometrics — psychological network analysis (Borsboom, Epskamp).

Estimation follows qgraph::EBICglasso (Epskamp & Fried, 2018): the graphical lasso is fit
to the Pearson correlation matrix over 100 penalty values (log-spaced from the largest
absolute correlation down to 1% of it, diagonal not penalized), and the network with the
lowest EBIC (gamma = 0.5 by default) is kept, without refitting. Edge weights are partial
correlations. Each graphical lasso fit is solved to tight tolerance (scikit-learn's
coordinate-descent solver), so the selected network reproduces qgraph's to about 1e-5.

Centrality follows qgraph::centrality: strength = sum |w|, expected influence = sum w,
closeness = 1 / (sum of shortest-path lengths to reachable nodes) and betweenness =
number of shortest paths through a node (counted in both directions, ties shared), with
path lengths 1/|w|. Bridge centrality follows networktools::bridge, the case-dropping
CS-coefficient follows bootnet::corStability, and the comparison test follows
NetworkComparisonTest::NCT. Checked against those packages in tests/test_network.py.
"""

import heapq
import warnings

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.covariance import graphical_lasso

router = APIRouter()

METHODS = ("ebicglasso", "partial", "correlation")


# ── Request Models ────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    data: list[list[float]]  # rows = subjects, cols = variables
    variable_names: list[str] | None = None
    method: str = "ebicglasso"  # "ebicglasso", "partial" or "correlation"
    gamma: float = 0.5  # EBIC hyperparameter (ebicglasso only)
    threshold: float | None = None  # optional |r| cut-off (correlation method only)

class CentralityRequest(BaseModel):
    adjacency: list[list[float]]  # weighted adjacency matrix
    variable_names: list[str] | None = None

class StabilityRequest(BaseModel):
    data: list[list[float]]  # raw data (subjects × variables)
    variable_names: list[str] | None = None
    method: str = "ebicglasso"
    gamma: float = 0.5
    n_boots: int = 30  # subsamples per drop proportion
    drop_proportions: list[float] | None = None  # defaults to bootnet's 10 levels, 0.05 to 0.75
    seed: int = 1

class BridgeRequest(BaseModel):
    adjacency: list[list[float]]
    communities: list[int]  # community assignment per node
    variable_names: list[str] | None = None

class CompareRequest(BaseModel):
    data_group1: list[list[float]]  # group 1: subjects × variables
    data_group2: list[list[float]]  # group 2: subjects × variables
    variable_names: list[str] | None = None
    method: str = "ebicglasso"
    gamma: float = 0.5
    n_permutations: int = 100
    seed: int = 1


# ── Helpers ───────────────────────────────────────────────────────────

def _variable_names(names: list[str] | None, p: int) -> list[str]:
    """Return variable names, generating defaults if not provided."""
    if names and len(names) == p:
        return names
    return [f"V{i+1}" for i in range(p)]


def _data_matrix(rows: list[list[float]], label: str = "Data") -> np.ndarray:
    try:
        data = np.array(rows, dtype=float)
    except ValueError:
        raise HTTPException(400, f"{label} rows must all have the same number of values.")
    if data.ndim != 2:
        raise HTTPException(400, f"{label} must be a 2D array (subjects × variables).")
    if not np.all(np.isfinite(data)):
        raise HTTPException(400, f"{label} must contain only finite numbers (no missing values).")
    if data.shape[1] < 2:
        raise HTTPException(400, f"{label} needs at least 2 variables.")
    if np.any(np.std(data, axis=0) == 0):
        raise HTTPException(400, f"Every variable in {label.lower()} must vary (a column is constant).")
    return data


def _check_method(method: str):
    if method not in METHODS:
        raise HTTPException(400, f"Method must be one of: {', '.join(METHODS)}.")


def _wi2net(K: np.ndarray) -> np.ndarray:
    """Precision matrix -> partial correlation network (qgraph::wi2net)."""
    d = np.sqrt(np.diag(K))
    net = -K / np.outer(d, d)
    net = np.triu(net, 1)
    return net + net.T


def _ebicglasso(R: np.ndarray, n: int, gamma: float = 0.5, nlambda: int = 100,
                lambda_min_ratio: float = 0.01, tol: float = 1e-8) -> tuple[np.ndarray, dict]:
    """qgraph::EBICglasso (refit = FALSE, threshold = FALSE, diagonal not penalized)."""
    p = R.shape[0]
    lam_max = float(np.max(np.abs(R - np.eye(p))))
    if lam_max == 0:
        return np.zeros((p, p)), {"lambda": None, "ebic": None, "lambda_max": 0.0}
    lambdas = np.exp(np.linspace(np.log(lambda_min_ratio * lam_max), np.log(lam_max), nlambda))
    lower = np.tril_indices(p, -1)
    best = None
    for i, lam in enumerate(lambdas):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, K = graphical_lasso(R, alpha=float(lam), tol=tol, enet_tol=tol * 1e-2, max_iter=2000)
        except (FloatingPointError, ValueError):
            continue
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            continue
        E = int(np.count_nonzero(K[lower]))
        loglik = n / 2.0 * (logdet - float(np.trace(K @ R)))
        ebic = -2.0 * loglik + E * np.log(n) + 4.0 * E * gamma * np.log(p)
        if best is None or ebic < best[0]:
            best = (ebic, i, K)
    if best is None:
        raise HTTPException(422, "The graphical lasso failed for every penalty value.")
    ebic, i, K = best
    notes = []
    if i == 0:
        notes.append("The least-penalized network on the path was selected, so the assumption of a sparse network may not hold.")
    elif lambdas[i] < 0.1 * lam_max:
        notes.append("A dense network was selected (lambda < 0.1 × lambda max). Small edges may be false positives; interpret them with care.")
    net = _wi2net(K)
    if not np.any(net):
        notes.append("The selected network has no edges.")
    return net, {"lambda": float(lambdas[i]), "lambda_max": lam_max, "ebic": float(ebic),
                 "gamma": gamma, "n_lambda": nlambda, "lambda_min_ratio": lambda_min_ratio, "notes": notes}


def _correlation(data: np.ndarray) -> np.ndarray:
    R = np.corrcoef(data, rowvar=False)
    if np.min(np.linalg.eigvalsh(R)) <= 1e-10:
        raise HTTPException(
            422,
            "The correlation matrix is not positive definite (more variables than independent cases, "
            "or a variable is an exact combination of others). Remove redundant variables or collect more data.",
        )
    return R


def _estimate_network(data: np.ndarray, method: str, gamma: float = 0.5,
                      threshold: float | None = None) -> tuple[np.ndarray, dict]:
    """Estimate a network; returns (adjacency, details)."""
    n = data.shape[0]
    if method == "correlation":
        R = np.corrcoef(data, rowvar=False)
        adj = R - np.eye(R.shape[0])
        if threshold is not None:
            adj = np.where(np.abs(adj) >= threshold, adj, 0.0)
        return adj, {"threshold": threshold}
    R = _correlation(data)
    if method == "partial":
        return _wi2net(np.linalg.inv(R)), {}
    return _ebicglasso(R, n, gamma)


def _adjacency_to_edgelist(adj: np.ndarray, names: list[str]) -> list[dict]:
    """Convert adjacency matrix to list of edges with weights."""
    p = adj.shape[0]
    return [{"from": names[i], "to": names[j], "weight": float(adj[i, j])}
            for i in range(p) for j in range(i + 1, p) if adj[i, j] != 0]


def _shortest_paths(D: np.ndarray, source: int, eps: float = 1e-10):
    """Dijkstra from one source on a distance matrix (inf = no edge).

    Returns (dist, sigma, preds, order): distances, number of shortest paths, the
    predecessors on shortest paths and the order nodes were settled. Path lengths within
    a relative 1e-10 count as ties, as in igraph.
    """
    p = D.shape[0]
    dist = np.full(p, np.inf)
    sigma = np.zeros(p)
    preds: list[list[int]] = [[] for _ in range(p)]
    dist[source] = 0.0
    sigma[source] = 1.0
    done = np.zeros(p, dtype=bool)
    order = []
    heap = [(0.0, source)]
    while heap:
        d, v = heapq.heappop(heap)
        if done[v]:
            continue
        done[v] = True
        order.append(v)
        for w in range(p):
            if done[w] or not np.isfinite(D[v, w]):
                continue
            nd = d + D[v, w]
            if np.isfinite(dist[w]) and abs(nd - dist[w]) <= eps * max(nd, dist[w]):
                sigma[w] += sigma[v]
                preds[w].append(v)
            elif nd < dist[w]:
                dist[w] = nd
                sigma[w] = sigma[v]
                preds[w] = [v]
                heapq.heappush(heap, (nd, w))
    return dist, sigma, preds, order


def _distances(W: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        D = np.where(W != 0, 1.0 / np.abs(W), np.inf)
    np.fill_diagonal(D, np.inf)
    return D


def _closeness_betweenness(W: np.ndarray) -> tuple[list, np.ndarray]:
    """igraph closeness and betweenness as called by qgraph::centrality."""
    p = W.shape[0]
    D = _distances(W)
    closeness: list = []
    betweenness = np.zeros(p)
    for s in range(p):
        dist, sigma, preds, order = _shortest_paths(D, s)
        reach = np.isfinite(dist) & (np.arange(p) != s)
        closeness.append(float(1.0 / dist[reach].sum()) if reach.any() else None)
        delta = np.zeros(p)
        for w in reversed(order):
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    return closeness, betweenness


def _square(adj_rows: list[list[float]]) -> np.ndarray:
    try:
        adj = np.array(adj_rows, dtype=float)
    except ValueError:
        raise HTTPException(400, "Adjacency must be a square 2D matrix.")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise HTTPException(400, "Adjacency must be a square 2D matrix.")
    if not np.all(np.isfinite(adj)):
        raise HTTPException(400, "Adjacency must contain only finite numbers.")
    if adj.shape[0] < 2:
        raise HTTPException(400, "Need at least 2 nodes.")
    if not np.allclose(adj, adj.T, atol=1e-10):
        raise HTTPException(400, "Adjacency must be symmetric (undirected network).")
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0.0)
    return adj


def _compute_centrality(adj: np.ndarray, names: list[str]) -> dict:
    """Strength, closeness, betweenness and expected influence (qgraph::centrality)."""
    p = adj.shape[0]
    strength = np.sum(np.abs(adj), axis=1)
    expected_influence = np.sum(adj, axis=1)
    closeness, betw = _closeness_betweenness(adj)

    def _zscore(arr):
        a = np.array([np.nan if v is None else v for v in arr], dtype=float)
        ok = np.isfinite(a)
        if ok.sum() < 2:
            return [None] * p
        m, s = a[ok].mean(), a[ok].std(ddof=1)
        return [None if not ok[i] else (float((a[i] - m) / s) if s > 0 else 0.0) for i in range(p)]

    table = [{
        "variable": names[i],
        "strength": float(strength[i]),
        "closeness": closeness[i],
        "betweenness": float(betw[i]),
        "expected_influence": float(expected_influence[i]),
    } for i in range(p)]

    z = {k: _zscore(v) for k, v in
         (("strength", strength), ("closeness", closeness), ("betweenness", betw), ("expected_influence", expected_influence))}
    standardized = [{"variable": names[i], **{k: z[k][i] for k in z}} for i in range(p)]

    clo = np.array([-np.inf if c is None else c for c in closeness])
    most_central = {
        "strength": names[int(np.argmax(strength))],
        "closeness": names[int(np.argmax(clo))] if np.isfinite(clo).any() else None,
        "betweenness": names[int(np.argmax(betw))] if betw.max() > 0 else None,
        "expected_influence": names[int(np.argmax(expected_influence))],
    }
    notes = []
    if any(c is None for c in closeness):
        notes.append("Closeness is undefined for nodes with no edges and is left blank for them.")
    if betw.max() == 0:
        notes.append("No node lies on a shortest path between two others, so betweenness is 0 everywhere.")
    return {
        "centrality_table": table,
        "standardized_centrality": standardized,
        "most_central_nodes": most_central,
        "centrality_notes": notes,
    }


def _cor0(x: np.ndarray, y: np.ndarray) -> float:
    """bootnet's cor0: the correlation, or 0 when either vector is constant."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _cs_coefficient(levels: list[float], cors: list[list[float]], cor: float = 0.7) -> float:
    """bootnet::corStability: the largest drop proportion at which more than 95% of the
    subsample correlations exceed `cor` (0 if there is none)."""
    ok = [lvl for lvl, cs in zip(levels, cors) if cs and np.mean(np.array(cs) > cor) > 0.95]
    return float(max(ok)) if ok else 0.0


def _cs_label(cs: float) -> str:
    if cs >= 0.5:
        return "Stable (CS ≥ 0.5)"
    if cs >= 0.25:
        return "Acceptable for interpretation (CS ≥ 0.25), but not ≥ 0.5"
    return "Unstable (CS < 0.25): do not interpret the ordering of nodes on this index"


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj.tolist()


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/estimate")
def estimate_network(req: EstimateRequest):
    """Estimate a psychological network from multivariate data, with centrality indices.

    Methods: "ebicglasso" (regularized partial correlations, qgraph::EBICglasso),
    "partial" (unregularized partial correlations) or "correlation" (Pearson
    correlations, optionally thresholded at |r| >= threshold).
    """
    _check_method(req.method)
    data = _data_matrix(req.data)
    n, p = data.shape
    if n < 3:
        raise HTTPException(400, "Need at least 3 subjects.")
    if req.method != "correlation" and n <= p:
        raise HTTPException(400, "Partial-correlation networks need more subjects than variables.")
    if not 0 <= req.gamma <= 1:
        raise HTTPException(400, "gamma must be between 0 and 1.")

    names = _variable_names(req.variable_names, p)
    adj, details = _estimate_network(data, req.method, req.gamma, req.threshold)

    edges = _adjacency_to_edgelist(adj, names)
    max_edges = p * (p - 1) // 2
    n_edges = len(edges)
    notes = list(details.pop("notes", []))
    if req.method == "partial":
        notes.append("Unregularized partial correlations: every pair gets an edge, including ones that are only sampling noise.")
    if req.method == "correlation":
        notes.append("Correlation networks show marginal associations, which include indirect ones; partial-correlation networks do not.")

    result = {
        "method": req.method,
        "n_subjects": n,
        "n_variables": p,
        "variable_names": names,
        "adjacency_matrix": adj.tolist(),
        "edge_list": edges,
        "n_edges": n_edges,
        "max_possible_edges": max_edges,
        "density": float(n_edges / max_edges) if max_edges else 0.0,
        "sparsity": float(1.0 - n_edges / max_edges) if max_edges else 1.0,
        "estimation": details,
        "notes": notes,
    }
    result.update(_compute_centrality(adj, names))
    return result


@router.post("/centrality")
def centrality_analysis(req: CentralityRequest):
    """Compute network centrality indices: strength, closeness, betweenness, expected influence."""
    adj = _square(req.adjacency)
    p = adj.shape[0]
    names = _variable_names(req.variable_names, p)
    result = _compute_centrality(adj, names)
    result["n_nodes"] = p
    return result


@router.post("/stability")
def stability_analysis(req: StabilityRequest):
    """Case-dropping subsample stability of strength and expected influence (bootnet).

    For each drop proportion, n_boots subsamples without replacement are drawn, the
    network is re-estimated and its centrality correlated with the full-sample values.
    CS-coefficient = largest drop proportion at which more than 95% of those correlations
    exceed 0.7 (Epskamp, Borsboom & Fried, 2018).
    """
    _check_method(req.method)
    data = _data_matrix(req.data)
    n, p = data.shape
    if n < 10:
        raise HTTPException(400, "Need at least 10 subjects for stability analysis.")

    names = _variable_names(req.variable_names, p)
    levels = req.drop_proportions or list(np.linspace(0.05, 0.75, 10))
    levels = sorted({float(dp) for dp in levels if 0 < dp < 1})
    if not levels:
        raise HTTPException(400, "drop_proportions must be between 0 and 1.")
    n_boots = max(10, min(req.n_boots, 200))
    if n_boots * len(levels) > 1000:
        raise HTTPException(400, "At most 1000 subsamples per request (n_boots × number of drop proportions).")
    rng = np.random.default_rng(req.seed)

    orig_adj, _ = _estimate_network(data, req.method, req.gamma)
    orig = {"strength": np.abs(orig_adj).sum(axis=1), "expected_influence": orig_adj.sum(axis=1)}

    curves = {k: [] for k in orig}
    cors_by_stat = {k: [] for k in orig}
    actual_levels = []
    for drop in levels:
        keep_n = int(round(n * (1.0 - drop)))
        if keep_n <= p + 1:
            continue
        actual = 1.0 - keep_n / n
        actual_levels.append(actual)
        cors = {k: [] for k in orig}
        failed = 0
        for _ in range(n_boots):
            idx = rng.choice(n, size=keep_n, replace=False)
            try:
                sub, _ = _estimate_network(data[idx], req.method, req.gamma)
            except HTTPException:
                failed += 1
                continue
            cors["strength"].append(_cor0(orig["strength"], np.abs(sub).sum(axis=1)))
            cors["expected_influence"].append(_cor0(orig["expected_influence"], sub.sum(axis=1)))
        for k in orig:
            arr = np.array(cors[k])
            cors_by_stat[k].append(cors[k])
            curves[k].append({
                "drop_proportion": actual,
                "keep_n": keep_n,
                "mean_correlation": float(arr.mean()) if arr.size else None,
                "q05_correlation": float(np.quantile(arr, 0.05)) if arr.size else None,
                "proportion_above_0_7": float(np.mean(arr > 0.7)) if arr.size else None,
                "n_successful_boots": int(arr.size),
                "n_failed": failed,
            })

    cs = {k: _cs_coefficient(actual_levels, cors_by_stat[k]) for k in orig}
    return {
        "n_subjects": n,
        "n_variables": p,
        "variable_names": names,
        "method": req.method,
        "n_boots_per_level": n_boots,
        "seed": req.seed,
        "cs_coefficient": cs,
        "interpretation": {k: _cs_label(v) for k, v in cs.items()},
        "stability_curves": curves,
        "notes": [
            "CS-coefficient = largest proportion of cases that can be dropped while more than 95% of subsample "
            "centrality values still correlate above 0.7 with the full sample (bootnet::corStability). "
            "Epskamp et al. (2018) suggest at least 0.25, preferably 0.5.",
            "Correlations are Pearson; bootnet's EBICglasso default uses polychoric correlations for ordinal items, "
            "so results differ from bootnet when items are ordinal.",
        ],
    }


@router.post("/bridge")
def bridge_centrality(req: BridgeRequest):
    """Bridge centrality for community structure in networks (networktools::bridge).

    Bridge strength = sum of |w| to nodes in other communities; bridge expected influence
    (1-step) = signed sum of those edges; 2-step adds each neighbour's influence on the
    other communities, weighted by the connecting edge. Bridge betweenness counts how often
    a node lies on a shortest path between nodes of different communities; bridge
    closeness = 1 / mean shortest-path length to nodes in other communities. As in
    networktools, the two path-based indices use positive edges only.
    """
    adj = _square(req.adjacency)
    p = adj.shape[0]
    communities = np.array(req.communities)
    if len(communities) != p:
        raise HTTPException(400, "Length of communities must equal number of nodes.")
    unique_comms = list(dict.fromkeys(communities.tolist()))
    if len(unique_comms) < 2:
        raise HTTPException(400, "Need at least 2 communities for bridge centrality.")

    names = _variable_names(req.variable_names, p)
    other = communities[:, None] != communities[None, :]
    bridge_strength = np.sum(np.abs(adj) * other, axis=1)
    bridge_ei1 = np.sum(adj * other, axis=1)

    # Influence of each node j on community c (1-step, excluding j itself)
    infl = {c: (adj * (communities == c)[None, :]).sum(axis=1) for c in unique_comms}
    bridge_ei2 = bridge_ei1.copy()
    for i in range(p):
        for c in unique_comms:
            if c != communities[i]:
                bridge_ei2[i] += float(adj[i] @ infl[c])

    # Path-based indices on the positive-edge graph
    D = _distances(np.where(adj > 0, adj, 0.0))
    paths = [_shortest_paths(D, s) for s in range(p)]
    dist = np.array([pt[0] for pt in paths])
    sigma = np.array([pt[1] for pt in paths])
    bridge_betw = np.zeros(p)
    for s in range(p):
        for t in range(p):
            if not other[s, t] or not np.isfinite(dist[s, t]):
                continue
            for v in range(p):
                if v in (s, t):
                    continue
                if np.isfinite(dist[s, v]) and np.isfinite(dist[v, t]) and \
                        abs(dist[s, v] + dist[v, t] - dist[s, t]) <= 1e-10 * dist[s, t]:
                    bridge_betw[v] += sigma[s, v] * sigma[v, t]
    bridge_betw /= 2.0
    bridge_closeness = []
    for i in range(p):
        d = dist[i, other[i]]
        d = d[np.isfinite(d)]
        bridge_closeness.append(float(1.0 / d.mean()) if d.size else None)

    table = [{
        "variable": names[i],
        "community": communities[i].item(),
        "bridge_strength": float(bridge_strength[i]),
        "bridge_betweenness": float(bridge_betw[i]),
        "bridge_closeness": bridge_closeness[i],
        "bridge_expected_influence": float(bridge_ei1[i]),
        "bridge_expected_influence_2step": float(bridge_ei2[i]),
    } for i in range(p)]

    # Top 20% on bridge strength (the cut-off Jones et al., 2021, used as an example)
    cut = float(np.quantile(bridge_strength, 0.8))
    bridge_nodes = [
        {"variable": names[i], "community": communities[i].item(), "bridge_strength": float(bridge_strength[i])}
        for i in range(p) if bridge_strength[i] >= cut and bridge_strength[i] > 0
    ]

    return {
        "n_nodes": p,
        "n_communities": len(unique_comms),
        "community_sizes": {str(c): int(np.sum(communities == c)) for c in unique_comms},
        "bridge_centrality_table": table,
        "identified_bridge_nodes": bridge_nodes,
        "notes": ["Bridge nodes listed are those in the top 20% on bridge strength. That cut-off is a convention, not a test."],
    }


@router.post("/compare")
def compare_networks(req: CompareRequest):
    """Network Comparison Test (van Borkulo et al., 2022) between two independent groups.

    Statistics: M = largest absolute difference in any edge (network structure invariance),
    S = absolute difference in global strength (sum of |edges|), each edge's absolute
    difference, and node strength / expected influence differences. Reference
    distributions come from permuting group membership; p = (exceedances + 1) / (permutations + 1).
    """
    _check_method(req.method)
    d1 = _data_matrix(req.data_group1, "Group 1 data")
    d2 = _data_matrix(req.data_group2, "Group 2 data")
    if d1.shape[1] != d2.shape[1]:
        raise HTTPException(400, "Both groups must have the same number of variables.")
    p = d1.shape[1]
    n1, n2 = d1.shape[0], d2.shape[0]
    if min(n1, n2) <= p:
        raise HTTPException(400, "Each group needs more subjects than variables.")

    names = _variable_names(req.variable_names, p)
    n_perm = max(20, min(req.n_permutations, 1000))
    rng = np.random.default_rng(req.seed)
    iu = np.triu_indices(p, 1)

    def stats(a1, a2):
        diff = np.abs(a1 - a2)[iu]
        return (abs(np.abs(a1[iu]).sum() - np.abs(a2[iu]).sum()), diff.max(), diff,
                np.abs(a1).sum(axis=1) - np.abs(a2).sum(axis=1), a1.sum(axis=1) - a2.sum(axis=1))

    adj1, _ = _estimate_network(d1, req.method, req.gamma)
    adj2, _ = _estimate_network(d2, req.method, req.gamma)
    S_obs, M_obs, edge_obs, str_obs, ei_obs = stats(adj1, adj2)

    combined = np.vstack([d1, d2])
    S_perm, M_perm, edge_perm, str_perm, ei_perm = [], [], [], [], []
    for _ in range(n_perm):
        idx = rng.permutation(n1 + n2)
        try:
            pa1, _ = _estimate_network(combined[idx[:n1]], req.method, req.gamma)
            pa2, _ = _estimate_network(combined[idx[n1:]], req.method, req.gamma)
        except HTTPException:
            continue
        s, m, e, st, ei = stats(pa1, pa2)
        S_perm.append(s); M_perm.append(m); edge_perm.append(e); str_perm.append(st); ei_perm.append(ei)
    B = len(S_perm)
    if B == 0:
        raise HTTPException(422, "Every permutation failed to estimate a network.")
    S_perm, M_perm = np.array(S_perm), np.array(M_perm)
    edge_perm, str_perm, ei_perm = np.array(edge_perm), np.array(str_perm), np.array(ei_perm)

    def pval(perm, obs):
        return (np.sum(perm >= obs - 1e-12, axis=0) + 1) / (B + 1)

    edge_p = pval(edge_perm, edge_obs)
    edge_p_holm = _holm(edge_p.tolist())
    edges = [{
        "from": names[i], "to": names[j],
        "weight_group1": float(adj1[i, j]), "weight_group2": float(adj2[i, j]),
        "abs_difference": float(edge_obs[k]),
        "p_value": float(edge_p[k]), "p_value_holm": float(edge_p_holm[k]),
    } for k, (i, j) in enumerate(zip(*iu))]

    str_p = pval(np.abs(str_perm), np.abs(str_obs))
    ei_p = pval(np.abs(ei_perm), np.abs(ei_obs))
    nodes = [{
        "variable": names[i],
        "strength_group1": float(np.abs(adj1[i]).sum()), "strength_group2": float(np.abs(adj2[i]).sum()),
        "strength_difference": float(str_obs[i]), "strength_p_value": float(str_p[i]),
        "expected_influence_difference": float(ei_obs[i]), "expected_influence_p_value": float(ei_p[i]),
    } for i in range(p)]

    return {
        "n_group1": n1,
        "n_group2": n2,
        "n_variables": p,
        "method": req.method,
        "n_permutations": B,
        "seed": req.seed,
        "network_invariance": {"M": float(M_obs), "p_value": float(pval(M_perm, M_obs))},
        "global_strength": {
            "group1": float(np.abs(adj1[iu]).sum()),
            "group2": float(np.abs(adj2[iu]).sum()),
            "S": float(S_obs),
            "p_value": float(pval(S_perm, S_obs)),
        },
        "edge_tests": edges,
        "significant_edges_holm": [e for e in edges if e["p_value_holm"] < 0.05],
        "node_tests": nodes,
        "adjacency_group1": adj1.tolist(),
        "adjacency_group2": adj2.tolist(),
        "notes": [
            "Edge and node p-values are per test; p_value_holm adjusts edge p-values for the number of edges.",
            f"With {B} permutations the smallest possible p-value is {1 / (B + 1):.4f}.",
        ],
    }
