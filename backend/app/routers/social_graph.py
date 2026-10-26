"""Social Graph — network analysis using NetworkX."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
from math import comb

router = APIRouter()


class NetworkRequest(BaseModel):
    edges: list[list[str | int]]  # [[source, target], ...] or [[source, target, weight], ...]
    directed: bool = False


class CommunityRequest(BaseModel):
    edges: list[list[str | int]]
    directed: bool = False


class ERGMRequest(BaseModel):
    edges: list[list[str | int]]
    directed: bool = False
    n_simulations: int = 100


class DiffusionRequest(BaseModel):
    edges: list[list[str | int]]
    directed: bool = False
    seed_nodes: list[str | int]
    model: str = "independent_cascade"
    probability: float = 0.1
    threshold: float = 0.3
    n_simulations: int = 100


class StructuralEquivalenceRequest(BaseModel):
    edges: list[list[str | int]]
    directed: bool = False
    n_positions: int = 3


class KeyPlayerRequest(BaseModel):
    edges: list[list[str | int]]
    directed: bool = False
    k: int = 3
    mode: str = "removal"


class TemporalNetworkRequest(BaseModel):
    snapshots: list[list[list[str | int]]]
    directed: bool = False
    period_labels: list[str] | None = None


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/analyze")
def analyze_network(req: NetworkRequest):
    """Comprehensive network analysis — centrality, structure, properties."""
    G = _build_graph(req.edges, req.directed)

    if G.number_of_nodes() == 0:
        raise HTTPException(400, "Network has no nodes.")

    # Basic metrics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    # Degree statistics
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))

    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        eigenvector = {n: 0.0 for n in G.nodes()}

    # Node-level data (sorted by betweenness)
    node_data = []
    for node in G.nodes():
        node_data.append({
            "id": str(node),
            "degree": degrees[node],
            "degree_centrality": float(degree_centrality[node]),
            "betweenness_centrality": float(betweenness[node]),
            "closeness_centrality": float(closeness[node]),
            "eigenvector_centrality": float(eigenvector[node]),
        })
    node_data.sort(key=lambda x: x["betweenness_centrality"], reverse=True)

    # Clustering
    if not req.directed:
        clustering = nx.clustering(G)
        avg_clustering = nx.average_clustering(G)
        transitivity = nx.transitivity(G)
    else:
        clustering = {}
        avg_clustering = 0
        transitivity = 0

    # Components
    if req.directed:
        n_strong = nx.number_strongly_connected_components(G)
        n_weak = nx.number_weakly_connected_components(G)
        components_info = {"strongly_connected": n_strong, "weakly_connected": n_weak}
    else:
        components = list(nx.connected_components(G))
        components_info = {
            "n_components": len(components),
            "largest_component_size": len(max(components, key=len)) if components else 0,
        }
        # Diameter and avg path length (only if connected)
        if nx.is_connected(G):
            components_info["diameter"] = nx.diameter(G)
            components_info["avg_path_length"] = float(nx.average_shortest_path_length(G))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(density),
        "avg_degree": float(avg_degree),
        "avg_clustering": float(avg_clustering),
        "transitivity": float(transitivity),
        "components": components_info,
        "nodes": node_data,
        "degree_distribution": _degree_distribution(G),
    }


@router.post("/communities")
def detect_communities(req: CommunityRequest):
    """Community detection by greedy modularity maximisation (Clauset, Newman & Moore, 2004).

    Directed networks are analysed as undirected. If edges carry weights, both the partition
    and the reported modularity use them.
    """
    G = _build_graph(req.edges, req.directed)
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "Network has no nodes.")

    # Use greedy modularity (works with networkx built-in)
    if req.directed:
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    w = "weight" if _is_weighted(G_undirected) else None
    communities = sorted(nx.community.greedy_modularity_communities(G_undirected, weight=w), key=len, reverse=True)
    modularity = nx.community.modularity(G_undirected, communities, weight=w)

    community_data = []
    for i, comm in enumerate(communities):
        members = [str(n) for n in comm]
        subgraph = G_undirected.subgraph(comm)
        community_data.append({
            "id": i,
            "size": len(members),
            "members": members,
            "internal_density": float(nx.density(subgraph)),
        })

    # Node-to-community mapping
    node_community = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_community[str(n)] = i

    return {
        "n_communities": len(communities),
        "modularity": float(modularity),
        "method": "greedy modularity (Clauset-Newman-Moore)" + (", weighted" if w else ""),
        "communities": community_data,
        "node_community_map": node_community,
    }


# ── ERGM Approximate ──────────────────────────────────────────────────

@router.post("/ergm-approximate")
def ergm_approximate(req: ERGMRequest):
    """Approximate ERGM by computing sufficient statistics and comparing to
    random Erdos-Renyi graphs of equal size/density."""
    G = _build_graph(req.edges, req.directed)
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "Network has no nodes.")
    if req.n_simulations < 1:
        raise HTTPException(400, "n_simulations must be >= 1.")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    nodes = list(G.nodes())

    # ── Observed statistics ──
    obs = _ergm_statistics(G, req.directed)

    # ── Observed GOF distributions ──
    obs_degree_dist = _degree_distribution(G)
    obs_geodesic_dist = _geodesic_distribution(G, req.directed)

    # ── Simulate Erdos-Renyi graphs ──
    if req.directed:
        max_edges = n * (n - 1)
    else:
        max_edges = n * (n - 1) / 2
    p = m / max_edges if max_edges > 0 else 0

    sim_stats: dict[str, list[float]] = {k: [] for k in obs}
    sim_degree_dists: list[dict] = []
    sim_geodesic_dists: list[dict] = []

    for _ in range(req.n_simulations):
        if req.directed:
            G_rand = nx.gnp_random_graph(n, p, directed=True)
        else:
            G_rand = nx.gnp_random_graph(n, p, directed=False)
        stats = _ergm_statistics(G_rand, req.directed)
        for k, v in stats.items():
            sim_stats[k].append(v)
        sim_degree_dists.append(_degree_distribution(G_rand))
        sim_geodesic_dists.append(_geodesic_distribution(G_rand, req.directed))

    # ── Summary: mean, sd, z-scores ──
    sim_summary = {}
    z_scores = {}
    interpretations = []
    for k in obs:
        vals = np.array(sim_stats[k])
        mean_v = float(np.mean(vals))
        sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sim_summary[k] = {"mean": mean_v, "sd": sd_v}
        z = (obs[k] - mean_v) / sd_v if sd_v > 0 else 0.0
        z_scores[k] = float(z)
        if abs(z) > 2:
            direction = "over" if z > 0 else "under"
            interpretations.append(
                f"{k} is significantly {direction}-represented "
                f"(z={z:.2f}, observed={obs[k]:.4f}, expected={mean_v:.4f})"
            )

    # ── GOF: aggregate simulated degree/geodesic distributions ──
    gof_degree = _aggregate_distribution(sim_degree_dists)
    gof_geodesic = _aggregate_distribution(sim_geodesic_dists)

    return {
        "observed_statistics": {k: float(v) for k, v in obs.items()},
        "simulated_statistics": sim_summary,
        "z_scores": z_scores,
        "interpretation": interpretations if interpretations else [
            "No structural features differ significantly from random."
        ],
        "gof": {
            "degree_distribution": {
                "observed": obs_degree_dist,
                "simulated_mean": gof_degree,
            },
            "geodesic_distribution": {
                "observed": obs_geodesic_dist,
                "simulated_mean": gof_geodesic,
            },
        },
    }


# ── Influence Diffusion ───────────────────────────────────────────────

@router.post("/influence-diffusion")
def influence_diffusion(req: DiffusionRequest):
    """Simulate information/influence diffusion through the network using
    Independent Cascade or Linear Threshold models."""
    G = _build_graph(req.edges, req.directed)
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "Network has no nodes.")
    if req.model not in ("independent_cascade", "linear_threshold"):
        raise HTTPException(400, "model must be 'independent_cascade' or 'linear_threshold'.")
    if not req.seed_nodes:
        raise HTTPException(400, "seed_nodes must not be empty.")
    if req.n_simulations < 1:
        raise HTTPException(400, "n_simulations must be >= 1.")

    all_nodes = set(G.nodes())
    seeds = set()
    for s in req.seed_nodes:
        if s in all_nodes:
            seeds.add(s)
        elif str(s) in all_nodes:
            seeds.add(str(s))
    if not seeds:
        raise HTTPException(400, "None of the seed_nodes exist in the network.")

    spread_counts: list[int] = []
    activation_counts: dict = defaultdict(int)
    adoption_times: dict = defaultdict(list)

    for _ in range(req.n_simulations):
        if req.model == "independent_cascade":
            activated, times = _independent_cascade(G, seeds, req.probability)
        else:
            activated, times = _linear_threshold(G, seeds, req.threshold)

        spread_counts.append(len(activated))
        for node in activated:
            activation_counts[str(node)] += 1
        for node, t in times.items():
            adoption_times[str(node)].append(t)

    n_sims = req.n_simulations
    avg_spread = float(np.mean(spread_counts))

    # Spread distribution histogram
    spread_arr = np.array(spread_counts)
    hist_bins = min(20, max(1, int(spread_arr.max() - spread_arr.min() + 1)))
    counts, bin_edges = np.histogram(spread_arr, bins=hist_bins)
    spread_histogram = [
        {"bin_start": float(bin_edges[i]), "bin_end": float(bin_edges[i + 1]),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]

    # Per-node activation probability
    per_node_prob = {
        str(node): float(activation_counts[str(node)] / n_sims)
        for node in all_nodes
    }

    # Average time-to-adoption
    avg_adoption_time = {}
    for node, times_list in adoption_times.items():
        avg_adoption_time[node] = float(np.mean(times_list))

    # Key spreaders: activated earliest and most often (excluding seeds)
    non_seed_activation = {
        n: prob for n, prob in per_node_prob.items()
        if n not in {str(s) for s in seeds}
    }
    key_spreaders = sorted(
        non_seed_activation.items(),
        key=lambda x: (-x[1], avg_adoption_time.get(x[0], float("inf"))),
    )[:10]

    return {
        "average_spread": avg_spread,
        "spread_distribution": spread_histogram,
        "per_node_activation_probability": per_node_prob,
        "average_time_to_adoption": avg_adoption_time,
        "key_spreaders": [
            {"node": n, "activation_probability": p,
             "avg_time_to_adoption": avg_adoption_time.get(n, None)}
            for n, p in key_spreaders
        ],
        "n_simulations": n_sims,
        "model": req.model,
    }


# ── Structural Equivalence ────────────────────────────────────────────

@router.post("/structural-equivalence")
def structural_equivalence(req: StructuralEquivalenceRequest):
    """Find structurally equivalent nodes via adjacency profile clustering."""
    G = _build_graph(req.edges, req.directed)
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "Network has no nodes.")
    if req.n_positions < 1:
        raise HTTPException(400, "n_positions must be >= 1.")

    nodes = sorted(G.nodes(), key=str)
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    if n < 2:
        raise HTTPException(400, "Need at least 2 nodes for structural equivalence.")

    n_positions = min(req.n_positions, n)

    # Build adjacency matrix
    adj = nx.to_numpy_array(G, nodelist=nodes)

    # Euclidean distance matrix based on adjacency row profiles
    # Each node's profile = its row + its column in the adjacency matrix
    if req.directed:
        profiles = np.hstack([adj, adj.T])
    else:
        profiles = adj

    dist_condensed = pdist(profiles, metric="euclidean")
    dist_matrix = squareform(dist_condensed)

    # Hierarchical clustering
    Z = linkage(dist_condensed, method="ward")
    labels = fcluster(Z, t=n_positions, criterion="maxclust")

    # Position assignments
    position_map = {}
    for i, node in enumerate(nodes):
        position_map[str(node)] = int(labels[i])

    # Blockmodel: density between positions
    positions = sorted(set(labels))
    pos_members: dict[int, list] = {p: [] for p in positions}
    for i, node in enumerate(nodes):
        pos_members[int(labels[i])].append(node)

    density_table: dict[str, dict[str, float]] = {}
    blockmodel: dict[str, dict[str, float]] = {}
    for pi in positions:
        density_table[str(pi)] = {}
        blockmodel[str(pi)] = {}
        for pj in positions:
            members_i = pos_members[pi]
            members_j = pos_members[pj]
            edge_count = 0
            for ni in members_i:
                for nj in members_j:
                    if ni != nj and G.has_edge(ni, nj):
                        edge_count += 1
            if pi == pj:
                possible = len(members_i) * (len(members_i) - 1)
                if not req.directed:
                    possible //= 2
                    # Count undirected edges once
                    edge_count //= 2
            else:
                possible = len(members_i) * len(members_j)
                if not req.directed:
                    edge_count //= 2
                    possible //= 2
            dens = edge_count / possible if possible > 0 else 0.0
            density_table[str(pi)][str(pj)] = float(dens)
            blockmodel[str(pi)][str(pj)] = float(edge_count)

    # Dendrogram data (convert linkage matrix to serializable list)
    dendrogram_data = Z.tolist()

    # Distance matrix as dict
    dist_dict: dict[str, dict[str, float]] = {}
    for i, ni in enumerate(nodes):
        dist_dict[str(ni)] = {}
        for j, nj in enumerate(nodes):
            dist_dict[str(ni)][str(nj)] = float(dist_matrix[i][j])

    return {
        "distance_matrix": dist_dict,
        "position_assignments": position_map,
        "blockmodel": blockmodel,
        "density_table": density_table,
        "dendrogram": dendrogram_data,
        "positions": {
            str(p): [str(m) for m in members]
            for p, members in pos_members.items()
        },
    }


# ── Key Player ────────────────────────────────────────────────────────

@router.post("/key-player")
def key_player(req: KeyPlayerRequest):
    """Identify key players for network disruption (removal) or diffusion
    (reach maximization) using greedy optimization."""
    G = _build_graph(req.edges, req.directed)
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "Network has no nodes.")
    if req.k < 1:
        raise HTTPException(400, "k must be >= 1.")
    if req.mode not in ("removal", "diffusion"):
        raise HTTPException(400, "mode must be 'removal' or 'diffusion'.")

    nodes = list(G.nodes())
    k = min(req.k, len(nodes))

    if req.mode == "removal":
        return _key_player_removal(G, nodes, k)
    else:
        return _key_player_diffusion(G, nodes, k)


# ── Temporal Network ──────────────────────────────────────────────────

@router.post("/temporal-network")
def temporal_network(req: TemporalNetworkRequest):
    """Analyze how a network evolves across time snapshots."""
    if not req.snapshots:
        raise HTTPException(400, "snapshots must not be empty.")

    n_periods = len(req.snapshots)
    labels = req.period_labels or [f"period_{i}" for i in range(n_periods)]
    if len(labels) != n_periods:
        raise HTTPException(400, "period_labels length must match number of snapshots.")

    graphs = [_build_graph(snap, req.directed) for snap in req.snapshots]

    # ── Per-period metrics ──
    per_period = []
    for i, G in enumerate(graphs):
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = float(nx.density(G)) if n_nodes > 0 else 0.0
        clustering = float(nx.average_clustering(G.to_undirected())) if n_nodes > 0 else 0.0
        if req.directed:
            n_comp = nx.number_weakly_connected_components(G) if n_nodes > 0 else 0
        else:
            n_comp = nx.number_connected_components(G) if n_nodes > 0 else 0
        per_period.append({
            "label": labels[i],
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "avg_clustering": clustering,
            "n_components": n_comp,
        })

    # ── Network stability: Jaccard similarity of edge sets ──
    stability = []
    for i in range(1, n_periods):
        edges_prev = set(_edge_set(graphs[i - 1], req.directed))
        edges_curr = set(_edge_set(graphs[i], req.directed))
        intersection = len(edges_prev & edges_curr)
        union = len(edges_prev | edges_curr)
        jaccard = intersection / union if union > 0 else 0.0
        stability.append({
            "from": labels[i - 1],
            "to": labels[i],
            "jaccard_similarity": float(jaccard),
            "edges_added": len(edges_curr - edges_prev),
            "edges_removed": len(edges_prev - edges_curr),
            "edges_persisted": intersection,
        })

    # ── Node persistence ──
    node_periods: dict[str, int] = defaultdict(int)
    for G in graphs:
        for node in G.nodes():
            node_periods[str(node)] += 1

    node_persistence = {
        node: {"periods_present": count, "fraction": float(count / n_periods)}
        for node, count in sorted(node_periods.items(), key=lambda x: -x[1])
    }

    # ── Edge persistence ──
    edge_period_count: dict[tuple, int] = defaultdict(int)
    for G in graphs:
        for e in _edge_set(G, req.directed):
            edge_period_count[e] += 1

    edge_persistence = [
        {"edge": list(e), "periods_present": count,
         "fraction": float(count / n_periods)}
        for e, count in sorted(edge_period_count.items(), key=lambda x: -x[1])
    ]

    # ── Growth/decline metrics ──
    growth = []
    for i in range(1, n_periods):
        prev = per_period[i - 1]
        curr = per_period[i]
        growth.append({
            "from": labels[i - 1],
            "to": labels[i],
            "node_change": curr["n_nodes"] - prev["n_nodes"],
            "edge_change": curr["n_edges"] - prev["n_edges"],
            "density_change": curr["density"] - prev["density"],
        })

    # ── Core-periphery evolution ──
    # Core = nodes present in all periods; periphery = rest
    all_periods_nodes = set(node_periods.keys())
    core_nodes = {n for n, c in node_periods.items() if c == n_periods}
    periphery_nodes = all_periods_nodes - core_nodes

    core_periphery = {
        "core_nodes": sorted(core_nodes),
        "periphery_nodes": sorted(periphery_nodes),
        "core_size_per_period": [],
    }
    for i, G in enumerate(graphs):
        g_nodes = {str(n) for n in G.nodes()}
        core_in_period = len(core_nodes & g_nodes)
        core_periphery["core_size_per_period"].append({
            "label": labels[i],
            "core_present": core_in_period,
            "total_nodes": G.number_of_nodes(),
        })

    return {
        "n_periods": n_periods,
        "per_period_metrics": per_period,
        "stability": stability,
        "node_persistence": node_persistence,
        "edge_persistence": edge_persistence[:100],  # cap for large networks
        "growth_decline": growth,
        "core_periphery": core_periphery,
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _build_graph(edges: list, directed: bool):
    """Graph from [source, target] or [source, target, weight] rows. Node ids are compared as
    strings (so 1 and "1" are the same node); weights must be positive numbers."""
    G = nx.DiGraph() if directed else nx.Graph()
    for i, edge in enumerate(edges):
        if len(edge) not in (2, 3):
            raise HTTPException(422, f"Edge {i + 1} must be [source, target] or [source, target, weight].")
        u, v = str(edge[0]).strip(), str(edge[1]).strip()
        if not u or not v:
            raise HTTPException(422, f"Edge {i + 1} has an empty node id.")
        if len(edge) == 3:
            try:
                w = float(edge[2])
            except (TypeError, ValueError):
                raise HTTPException(422, f"Edge {i + 1}: weight {edge[2]!r} is not a number.")
            if not w > 0:
                raise HTTPException(422, f"Edge {i + 1}: weights must be positive.")
            G.add_edge(u, v, weight=w)
        else:
            G.add_edge(u, v)
    return G


def _is_weighted(G) -> bool:
    return any("weight" in d for _, _, d in G.edges(data=True))


def _degree_distribution(G) -> dict:
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return {}
    counts: dict[int, int] = {}
    for d in degrees:
        counts[d] = counts.get(d, 0) + 1
    return {str(k): v for k, v in sorted(counts.items())}


def _ergm_statistics(G, directed: bool) -> dict:
    """Compute sufficient statistics used in ERGM models."""
    stats: dict[str, float] = {}
    stats["edges"] = float(G.number_of_edges())
    stats["transitivity"] = float(nx.transitivity(G) if G.number_of_nodes() > 2 else 0.0)

    # Triangles
    if not directed:
        triangle_dict = nx.triangles(G)
        stats["triangles"] = float(sum(triangle_dict.values()) // 3)
    else:
        # For directed graphs, count directed triangles via triadic census
        stats["triangles"] = float(sum(nx.triangles(G.to_undirected()).values()) // 3)

    # K-stars
    degrees = [d for _, d in G.degree()]
    stats["two_stars"] = float(sum(comb(d, 2) for d in degrees))
    stats["three_stars"] = float(sum(comb(d, 3) for d in degrees))

    # Reciprocity (directed only)
    if directed:
        stats["reciprocity"] = float(nx.reciprocity(G)) if G.number_of_edges() > 0 else 0.0
    else:
        stats["reciprocity"] = 1.0  # all edges are reciprocal in undirected

    return stats


def _geodesic_distribution(G, directed: bool) -> dict:
    """Distribution of shortest path lengths."""
    dist_counts: dict[int, int] = defaultdict(int)
    if directed:
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, d in lengths.items():
                if source != target:
                    dist_counts[d] += 1
    else:
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, d in lengths.items():
                if source != target and str(source) <= str(target):
                    dist_counts[d] += 1
    return {str(k): v for k, v in sorted(dist_counts.items())}


def _aggregate_distribution(dists: list[dict]) -> dict:
    """Average a list of distribution dicts."""
    all_keys: set[str] = set()
    for d in dists:
        all_keys.update(d.keys())
    result = {}
    for k in sorted(all_keys):
        vals = [d.get(k, 0) for d in dists]
        result[k] = float(np.mean(vals))
    return result


def _independent_cascade(G, seeds: set, p: float) -> tuple[set, dict]:
    """Run one Independent Cascade simulation. Returns (activated, time_map)."""
    activated = set(seeds)
    time_map = {s: 0 for s in seeds}
    newly_activated = list(seeds)
    t = 0
    while newly_activated:
        t += 1
        next_activated = []
        for node in newly_activated:
            neighbors = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
            for nb in neighbors:
                if nb not in activated:
                    if np.random.random() < p:
                        activated.add(nb)
                        next_activated.append(nb)
                        time_map[nb] = t
        newly_activated = next_activated
    return activated, time_map


def _linear_threshold(G, seeds: set, threshold: float) -> tuple[set, dict]:
    """Run one Linear Threshold simulation. Returns (activated, time_map)."""
    activated = set(seeds)
    time_map = {s: 0 for s in seeds}
    changed = True
    t = 0
    while changed:
        t += 1
        changed = False
        for node in list(G.nodes()):
            if node in activated:
                continue
            neighbors = list(G.predecessors(node)) if G.is_directed() else list(G.neighbors(node))
            if not neighbors:
                continue
            active_neighbors = sum(1 for nb in neighbors if nb in activated)
            fraction = active_neighbors / len(neighbors)
            if fraction >= threshold:
                activated.add(node)
                time_map[node] = t
                changed = True
    return activated, time_map


def _fragmentation(G) -> float:
    """Borgatti fragmentation score: 1 - sum(s_k*(s_k-1)) / (n*(n-1))
    where s_k is the size of component k."""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    reachable_pairs = sum(len(c) * (len(c) - 1) for c in components)
    return 1.0 - reachable_pairs / (n * (n - 1))


def _key_player_removal(G, nodes: list, k: int) -> dict:
    """Greedy key player identification for network fragmentation."""
    node_scores: dict[str, float] = {}
    for node in nodes:
        H = G.copy()
        H.remove_node(node)
        node_scores[str(node)] = _fragmentation(H)

    # Greedy selection
    selected = []
    remaining = set(nodes)
    G_temp = G.copy()
    frag_baseline = _fragmentation(G)

    for _ in range(k):
        best_node = None
        best_frag = -1.0
        for node in remaining:
            H = G_temp.copy()
            H.remove_node(node)
            f = _fragmentation(H)
            if f > best_frag:
                best_frag = f
                best_node = node
        if best_node is not None:
            selected.append(best_node)
            remaining.discard(best_node)
            G_temp.remove_node(best_node)

    frag_after = _fragmentation(G_temp)

    # Random removal comparison (average over 50 trials)
    random_frags = []
    for _ in range(50):
        rand_nodes = list(np.random.choice(nodes, size=min(k, len(nodes)), replace=False))
        H = G.copy()
        H.remove_nodes_from(rand_nodes)
        random_frags.append(_fragmentation(H))
    avg_random_frag = float(np.mean(random_frags))

    if G.is_directed():
        comp_before = nx.number_weakly_connected_components(G)
        comp_after = nx.number_weakly_connected_components(G_temp) if G_temp.number_of_nodes() > 0 else 0
    else:
        comp_before = nx.number_connected_components(G)
        comp_after = nx.number_connected_components(G_temp) if G_temp.number_of_nodes() > 0 else 0

    return {
        "key_players": [str(n) for n in selected],
        "mode": "removal",
        "fragmentation_before": float(frag_baseline),
        "fragmentation_after": float(frag_after),
        "components_before": comp_before,
        "components_after": comp_after,
        "random_removal_avg_fragmentation": avg_random_frag,
        "node_scores": {k_: float(v) for k_, v in sorted(
            node_scores.items(), key=lambda x: -x[1])},
    }


def _key_player_diffusion(G, nodes: list, k: int) -> dict:
    """Greedy key player identification for maximum reach."""
    # Compute reach at 1, 2, 3 steps for each node
    node_reach: dict[str, dict[str, int]] = {}
    for node in nodes:
        reach = {}
        for steps in (1, 2, 3):
            reachable = set()
            if G.is_directed():
                lengths = nx.single_source_shortest_path_length(G, node, cutoff=steps)
            else:
                lengths = nx.single_source_shortest_path_length(G, node, cutoff=steps)
            reachable = {n for n, d in lengths.items() if d > 0}
            reach[f"step_{steps}"] = len(reachable)
        node_reach[str(node)] = reach

    # Greedy selection maximizing combined 2-step reach
    selected = []
    remaining = set(nodes)
    covered: set = set()

    for _ in range(k):
        best_node = None
        best_gain = -1
        for node in remaining:
            reachable = set()
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=2)
            reachable = {n for n, d in lengths.items() if d > 0}
            gain = len(reachable - covered)
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is not None:
            selected.append(best_node)
            remaining.discard(best_node)
            lengths = nx.single_source_shortest_path_length(G, best_node, cutoff=2)
            covered.update(n for n, d in lengths.items() if d > 0)

    # Total reach from selected set at 1, 2, 3 steps
    total_reach = {}
    for steps in (1, 2, 3):
        reachable = set()
        for node in selected:
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=steps)
            reachable.update(n for n, d in lengths.items() if d > 0)
        total_reach[f"step_{steps}"] = len(reachable)

    # Random selection comparison
    random_reaches = []
    for _ in range(50):
        rand_nodes = list(np.random.choice(nodes, size=min(k, len(nodes)), replace=False))
        reachable = set()
        for node in rand_nodes:
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=2)
            reachable.update(n for n, d in lengths.items() if d > 0)
        random_reaches.append(len(reachable))
    avg_random_reach = float(np.mean(random_reaches))

    return {
        "key_players": [str(n) for n in selected],
        "mode": "diffusion",
        "total_reach": total_reach,
        "random_selection_avg_reach_step2": avg_random_reach,
        "node_scores": node_reach,
    }


def _edge_set(G, directed: bool) -> list[tuple]:
    """Return edges as sorted tuples for comparison."""
    if directed:
        return [(str(u), str(v)) for u, v in G.edges()]
    else:
        return [tuple(sorted([str(u), str(v)])) for u, v in G.edges()]
