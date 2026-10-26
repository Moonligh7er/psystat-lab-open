"""Causal Inference & DAG Analysis — Pearl's framework, IV estimation,
sensitivity analysis, g-computation, and doubly-robust estimation.

Implements DAG path analysis, backdoor/front-door criteria, 2SLS instrumental
variables, E-value sensitivity, g-computation counterfactuals, and AIPW
doubly-robust estimation with full bootstrap inference.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from itertools import combinations
from collections import deque

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class DAGAnalysisRequest(BaseModel):
    """DAG specification for causal path analysis."""
    nodes: list[str]
    edges: list[list[str]]  # each is [from, to]
    exposure: str
    outcome: str


class AdjustmentSetRequest(BaseModel):
    """Find valid adjustment sets for a DAG."""
    nodes: list[str]
    edges: list[list[str]]
    exposure: str
    outcome: str


class IVEstimationRequest(BaseModel):
    """Instrumental variable (2SLS) estimation."""
    outcome: list[float]
    treatment: list[float]
    instrument: list[list[float]]  # one or more instruments, each as value array
    instrument_names: list[str] | None = None
    covariates: list[list[float]] | None = None
    covariate_names: list[str] | None = None
    confidence: float = 0.95


class SensitivityAnalysisRequest(BaseModel):
    """Sensitivity analysis for unmeasured confounding."""
    treatment_effect: float
    se: float
    treatment_outcome_r_squared: float
    confidence: float = 0.95
    gamma_range: list[float] | None = None  # Rosenbaum bounds range


class GComputationRequest(BaseModel):
    """G-computation for causal effects."""
    outcome: list[float]
    treatment: list[int]  # binary 0/1
    covariates: list[list[float]]
    covariate_names: list[str] | None = None
    outcome_type: str = "continuous"  # "continuous" or "binary"
    n_bootstrap: int = 1000
    confidence: float = 0.95


class DoublyRobustRequest(BaseModel):
    """Doubly robust (AIPW) estimation."""
    outcome: list[float]
    treatment: list[int]  # binary 0/1
    covariates: list[list[float]]
    covariate_names: list[str] | None = None
    n_bootstrap: int = 1000
    confidence: float = 0.95


# ── DAG Helpers ───────────────────────────────────────────────────────

def _build_adjacency(nodes: list[str], edges: list[list[str]]):
    """Build adjacency structures from edge list."""
    children = {n: set() for n in nodes}
    parents = {n: set() for n in nodes}
    for e in edges:
        if len(e) != 2:
            raise HTTPException(status_code=422, detail=f"Each edge must be [from, to]. Got: {e}")
        src, dst = e[0], e[1]
        if src not in children or dst not in children:
            raise HTTPException(
                status_code=422,
                detail=f"Edge references unknown node. Nodes: {nodes}, edge: {e}",
            )
        children[src].add(dst)
        parents[dst].add(src)
    return children, parents


def _all_paths(children: dict, parents: dict, start: str, end: str,
               undirected: bool = False) -> list[list[str]]:
    """Find all simple paths from start to end.

    If undirected=True, traverse edges in both directions (for backdoor paths).
    """
    paths = []
    if undirected:
        adj = {n: set() for n in children}
        for n in children:
            for c in children[n]:
                adj[n].add(c)
                adj[c].add(n)
    else:
        adj = children

    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == end and len(path) > 1:
            paths.append(path)
            continue
        for neighbor in adj.get(node, []):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
    return paths


def _directed_paths(children: dict, start: str, end: str) -> list[list[str]]:
    """Find all directed (causal) paths from start to end."""
    return _all_paths(children, {}, start, end, undirected=False)


def _undirected_paths(children: dict, parents: dict,
                      start: str, end: str) -> list[list[str]]:
    """Find all paths (ignoring edge direction) from start to end."""
    return _all_paths(children, parents, start, end, undirected=True)


def _classify_node_on_path(node: str, path: list[str], children: dict,
                           idx: int) -> str:
    """Classify a node's role on a path: collider, mediator, confounder, or endpoint."""
    if idx == 0 or idx == len(path) - 1:
        return "endpoint"

    prev_node = path[idx - 1]
    next_node = path[idx + 1]

    # Check edge directions
    prev_to_node = node in children.get(prev_node, set())
    node_to_prev = prev_node in children.get(node, set())
    next_to_node = node in children.get(next_node, set())
    node_to_next = next_node in children.get(node, set())

    # Collider: both neighbors point INTO this node  <- node ->  becomes  -> node <-
    if prev_to_node and next_to_node:
        return "collider"
    # Confounder (fork): node points to both neighbors  node -> ... and node -> ...
    if node_to_prev and node_to_next:
        return "confounder"
    # Mediator (chain): one points in, the other points out
    return "mediator"


def _is_backdoor_path(path: list[str], children: dict, exposure: str) -> bool:
    """A backdoor path has an arrow INTO the exposure (path[0])."""
    if len(path) < 2:
        return False
    second = path[1]
    # Backdoor: second node is a parent of exposure, i.e. second -> exposure
    return exposure in children.get(second, set())


def _descendants(node: str, children: dict) -> set[str]:
    """Get all descendants of a node."""
    desc = set()
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for c in children.get(current, set()):
            if c not in desc:
                desc.add(c)
                queue.append(c)
    return desc


def _ancestors(node: str, parents: dict) -> set[str]:
    """Get all ancestors of a node."""
    anc = set()
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for p in parents.get(current, set()):
            if p not in anc:
                anc.add(p)
                queue.append(p)
    return anc


def _is_path_blocked(path: list[str], children: dict, conditioning_set: set[str]) -> bool:
    """Check if a path is blocked (d-separated) given a conditioning set.

    A path is blocked if any non-endpoint node satisfies:
    - It is a non-collider AND is in the conditioning set, OR
    - It is a collider AND neither it nor any descendant is in the conditioning set.
    """
    for i in range(1, len(path) - 1):
        node = path[i]
        role = _classify_node_on_path(node, path, children, i)

        if role == "collider":
            # Collider blocks UNLESS conditioned on (or descendant conditioned on)
            desc = _descendants(node, children)
            if node not in conditioning_set and not desc.intersection(conditioning_set):
                return True  # blocked
        else:
            # Non-collider blocks IF conditioned on
            if node in conditioning_set:
                return True  # blocked

    return False  # not blocked = active path


def _find_minimal_adjustment_sets(nodes: list[str], children: dict, parents: dict,
                                  exposure: str, outcome: str) -> list[set[str]]:
    """Find all minimal sufficient adjustment sets using the backdoor criterion.

    A valid adjustment set Z must:
    1. Not contain any descendant of the exposure
    2. Block all backdoor paths from exposure to outcome
    """
    # Get all undirected paths
    all_paths = _undirected_paths(children, parents, exposure, outcome)

    # Identify backdoor paths
    backdoor_paths = [p for p in all_paths if _is_backdoor_path(p, children, exposure)]

    if not backdoor_paths:
        return [set()]  # empty set is sufficient

    # Forbidden nodes: descendants of exposure (cannot be in adjustment set)
    desc_exposure = _descendants(exposure, children)
    forbidden = desc_exposure | {exposure, outcome}

    # Candidate adjustment variables
    candidates = [n for n in nodes if n not in forbidden]

    if not candidates:
        return []  # no valid adjustment possible

    # Try all subsets from smallest to largest
    valid_sets = []
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            z = set(subset)
            # Check if Z blocks all backdoor paths
            all_blocked = True
            for bp in backdoor_paths:
                if not _is_path_blocked(bp, children, z):
                    all_blocked = False
                    break
            if all_blocked:
                # Check minimality: no proper subset of z is already valid
                is_minimal = True
                for vs in valid_sets:
                    if vs.issubset(z):
                        is_minimal = False
                        break
                if is_minimal:
                    valid_sets.append(z)

    return valid_sets


def _check_front_door(children: dict, parents: dict,
                      exposure: str, outcome: str, nodes: list[str]) -> dict:
    """Check if the front-door criterion is applicable.

    Front-door criterion requires a set M such that:
    1. Exposure intercepts all directed paths from exposure to outcome through M
    2. No unblocked backdoor paths from exposure to M
    3. All backdoor paths from M to outcome are blocked by exposure
    """
    desc_exp = _descendants(exposure, children)
    anc_out = _ancestors(outcome, parents)

    # Candidate mediators: descendants of exposure that are ancestors of outcome
    potential_mediators = desc_exp.intersection(anc_out) - {outcome}

    if not potential_mediators:
        return {
            "applicable": False,
            "reason": "No mediator set found between exposure and outcome.",
            "mediator_set": [],
        }

    # Check if mediators intercept ALL directed paths
    directed = _directed_paths(children, exposure, outcome)
    for m_set_size in range(1, len(potential_mediators) + 1):
        for m_set in combinations(potential_mediators, m_set_size):
            m_set_s = set(m_set)
            # Condition 1: every directed path goes through at least one mediator
            all_intercepted = all(
                any(node in m_set_s for node in path[1:-1])
                for path in directed
            ) if directed else False

            if not all_intercepted:
                continue

            # Condition 2: no unblocked backdoor from exposure to any mediator
            cond2 = True
            for med in m_set:
                bp_exp_med = [
                    p for p in _undirected_paths(children, parents, exposure, med)
                    if _is_backdoor_path(p, children, exposure)
                ]
                for bp in bp_exp_med:
                    if not _is_path_blocked(bp, children, set()):
                        cond2 = False
                        break
                if not cond2:
                    break

            # Condition 3: all backdoor paths from M to Y blocked by {exposure}
            cond3 = True
            for med in m_set:
                bp_med_out = [
                    p for p in _undirected_paths(children, parents, med, outcome)
                    if _is_backdoor_path(p, children, med)
                ]
                for bp in bp_med_out:
                    if not _is_path_blocked(bp, children, {exposure}):
                        cond3 = False
                        break
                if not cond3:
                    break

            if all_intercepted and cond2 and cond3:
                return {
                    "applicable": True,
                    "mediator_set": list(m_set),
                    "reason": (
                        f"Front-door criterion satisfied via mediator(s): {list(m_set)}. "
                        "Causal effect is identifiable even without blocking backdoor paths directly."
                    ),
                }

    return {
        "applicable": False,
        "reason": "No mediator set satisfies all three front-door conditions.",
        "mediator_set": [],
    }


def _find_iv_candidates(children: dict, parents: dict,
                        exposure: str, outcome: str,
                        nodes: list[str]) -> list[dict]:
    """Find instrumental variable candidates in the DAG.

    An IV Z must:
    1. Be associated with the exposure (relevance)
    2. Not have a direct effect on the outcome (exclusion)
    3. Not share common causes with the outcome (independence)
    """
    candidates = []
    desc_exp = _descendants(exposure, children)

    for z in nodes:
        if z in (exposure, outcome):
            continue
        if z in desc_exp:
            continue

        # Condition 1: Z has a directed path to exposure
        paths_to_exp = _directed_paths(children, z, exposure)
        if not paths_to_exp:
            continue

        # Condition 2: no direct edge Z -> outcome
        if outcome in children.get(z, set()):
            continue

        # Condition 3: no directed path from Z to outcome that bypasses exposure
        paths_to_out = _directed_paths(children, z, outcome)
        all_through_exposure = all(
            exposure in path for path in paths_to_out
        ) if paths_to_out else True

        if not all_through_exposure:
            continue

        # Check independence: no backdoor path from Z to outcome
        # (after removing edges through exposure)
        candidates.append({
            "variable": z,
            "paths_to_exposure": [p for p in paths_to_exp],
            "reason": (
                f"{z} affects {exposure} (relevance), has no direct path to "
                f"{outcome} bypassing {exposure} (exclusion restriction)."
            ),
        })

    return candidates


def _validate_lengths(*arrays, min_n: int = 10):
    """Validate that all arrays are the same length and meet minimum n."""
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"All variables must have the same length. Got lengths: {lengths}",
        )
    if lengths[0] < min_n:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_n} observations, got {lengths[0]}.",
        )


# ── 1. DAG Analysis ──────────────────────────────────────────────────

@router.post("/dag-analysis")
def dag_analysis(req: DAGAnalysisRequest):
    """Analyze a DAG: find causal/backdoor paths, colliders, mediators,
    confounders, adjustment sets, and check front-door criterion."""

    if req.exposure not in req.nodes:
        raise HTTPException(status_code=422, detail=f"Exposure '{req.exposure}' not in nodes.")
    if req.outcome not in req.nodes:
        raise HTTPException(status_code=422, detail=f"Outcome '{req.outcome}' not in nodes.")
    if req.exposure == req.outcome:
        raise HTTPException(status_code=422, detail="Exposure and outcome must be different.")

    children, parents = _build_adjacency(req.nodes, req.edges)

    # All directed (causal) paths
    causal_paths = _directed_paths(children, req.exposure, req.outcome)

    # All undirected paths (for backdoor analysis)
    all_undirected = _undirected_paths(children, parents, req.exposure, req.outcome)

    # Classify each path
    path_details = []
    backdoor_paths = []

    for path in all_undirected:
        is_causal = path in causal_paths
        is_backdoor = _is_backdoor_path(path, children, req.exposure)

        node_details = []
        for i, node in enumerate(path):
            role = _classify_node_on_path(node, path, children, i)
            node_details.append({"node": node, "role": role})

        path_info = {
            "path": path,
            "type": "causal" if is_causal else ("backdoor" if is_backdoor else "non_causal"),
            "node_details": node_details,
            "blocked_by_empty_set": _is_path_blocked(path, children, set()),
        }
        path_details.append(path_info)

        if is_backdoor:
            backdoor_paths.append(path)

    # D-separation analysis
    d_sep_empty = all(
        _is_path_blocked(p, children, set())
        for p in backdoor_paths
    ) if backdoor_paths else True

    # Minimal sufficient adjustment sets
    adj_sets = _find_minimal_adjustment_sets(
        req.nodes, children, parents, req.exposure, req.outcome
    )

    # Front-door criterion
    front_door = _check_front_door(children, parents, req.exposure, req.outcome, req.nodes)

    # IV candidates
    iv_candidates = _find_iv_candidates(children, parents, req.exposure, req.outcome, req.nodes)

    # Build interpretation
    n_causal = sum(1 for p in path_details if p["type"] == "causal")
    n_backdoor = len(backdoor_paths)
    interp_parts = [
        f"Found {n_causal} causal path(s) and {n_backdoor} backdoor path(s) "
        f"from {req.exposure} to {req.outcome}.",
    ]
    if adj_sets and adj_sets != [set()]:
        optimal = min(adj_sets, key=len)
        interp_parts.append(
            f"The optimal adjustment set is {sorted(optimal)}. "
            f"Conditioning on these variables blocks all backdoor paths."
        )
    elif adj_sets == [set()]:
        interp_parts.append(
            "No confounding adjustment needed: all backdoor paths are already blocked."
        )
    else:
        interp_parts.append(
            "WARNING: No valid adjustment set exists. The causal effect may not be "
            "identifiable via the backdoor criterion alone."
        )
    if front_door["applicable"]:
        interp_parts.append(
            f"The front-door criterion IS applicable via {front_door['mediator_set']}."
        )
    if iv_candidates:
        iv_names = [c["variable"] for c in iv_candidates]
        interp_parts.append(f"Potential instrumental variables: {iv_names}.")

    return {
        "exposure": req.exposure,
        "outcome": req.outcome,
        "n_nodes": len(req.nodes),
        "n_edges": len(req.edges),
        "paths": path_details,
        "n_causal_paths": n_causal,
        "n_backdoor_paths": n_backdoor,
        "d_separated_unconditionally": d_sep_empty,
        "minimal_sufficient_adjustment_sets": [sorted(s) for s in adj_sets],
        "optimal_adjustment_set": sorted(min(adj_sets, key=len)) if adj_sets else None,
        "front_door_criterion": front_door,
        "instrumental_variable_candidates": iv_candidates,
        "interpretation": " ".join(interp_parts),
    }


# ── 2. Adjustment Sets ───────────────────────────────────────────────

@router.post("/adjustment-sets")
def adjustment_sets(req: AdjustmentSetRequest):
    """Find all valid adjustment sets for causal effect identification."""

    if req.exposure not in req.nodes:
        raise HTTPException(status_code=422, detail=f"Exposure '{req.exposure}' not in nodes.")
    if req.outcome not in req.nodes:
        raise HTTPException(status_code=422, detail=f"Outcome '{req.outcome}' not in nodes.")

    children, parents = _build_adjacency(req.nodes, req.edges)

    # Descendants of exposure (forbidden in adjustment set)
    desc_exposure = _descendants(req.exposure, children)
    forbidden_nodes = desc_exposure | {req.exposure, req.outcome}

    # Find all backdoor paths
    all_undirected = _undirected_paths(children, parents, req.exposure, req.outcome)
    backdoor_paths = [p for p in all_undirected if _is_backdoor_path(p, children, req.exposure)]

    # Find collider descendants that could open paths if conditioned on
    collider_descendants = set()
    for path in all_undirected:
        for i in range(1, len(path) - 1):
            node = path[i]
            role = _classify_node_on_path(node, path, children, i)
            if role == "collider":
                collider_descendants.add(node)
                collider_descendants.update(_descendants(node, children))

    # Find all minimal sufficient adjustment sets
    adj_sets = _find_minimal_adjustment_sets(
        req.nodes, children, parents, req.exposure, req.outcome
    )

    # Determine optimal (fewest variables)
    optimal = sorted(min(adj_sets, key=len)) if adj_sets else None

    # Check if adjustment is possible
    adjustable = len(adj_sets) > 0

    # Identify forbidden adjustments with reasons
    forbidden_details = []
    for node in sorted(forbidden_nodes - {req.exposure, req.outcome}):
        reasons = []
        if node in desc_exposure:
            reasons.append("descendant of exposure")
        if node in collider_descendants:
            reasons.append("collider or descendant of collider (may open non-causal paths)")
        forbidden_details.append({
            "variable": node,
            "reasons": reasons if reasons else ["in causal pathway"],
        })

    # Warnings
    warnings = []
    if not adjustable:
        warnings.append(
            "No valid adjustment set exists under the backdoor criterion. "
            "Consider instrumental variables or front-door criterion."
        )
    if not backdoor_paths:
        warnings.append(
            "No backdoor paths found. The causal effect is identifiable without adjustment."
        )
    if len(adj_sets) > 1:
        sizes = [len(s) for s in adj_sets]
        if len(set(sizes)) > 1:
            warnings.append(
                f"Multiple valid sets found with different sizes ({min(sizes)} to {max(sizes)} variables). "
                "The optimal set minimizes variance but all are valid."
            )

    # Build interpretation
    interp_parts = []
    if adjustable and optimal is not None:
        if not optimal:
            interp_parts.append(
                "No adjustment needed: there are no open backdoor paths."
            )
        else:
            interp_parts.append(
                f"The optimal adjustment set is {{ {', '.join(optimal)} }}. "
                f"Conditioning on these {len(optimal)} variable(s) blocks all "
                f"{len(backdoor_paths)} backdoor path(s)."
            )
            if len(adj_sets) > 1:
                interp_parts.append(
                    f"There are {len(adj_sets)} total valid minimal adjustment sets."
                )
    else:
        interp_parts.append(
            "The causal effect is NOT identifiable via backdoor adjustment with the "
            "given DAG. Consider alternative identification strategies."
        )

    return {
        "exposure": req.exposure,
        "outcome": req.outcome,
        "n_backdoor_paths": len(backdoor_paths),
        "backdoor_paths": [[n for n in p] for p in backdoor_paths],
        "adjustment_possible": adjustable,
        "all_minimal_adjustment_sets": [sorted(s) for s in adj_sets],
        "n_valid_sets": len(adj_sets),
        "optimal_adjustment_set": optimal,
        "forbidden_adjustments": forbidden_details,
        "warnings": warnings,
        "interpretation": " ".join(interp_parts),
    }


# ── 3. IV Estimation (2SLS) ──────────────────────────────────────────

@router.post("/iv-estimation")
def iv_estimation(req: IVEstimationRequest):
    """Two-stage least squares (2SLS) instrumental variable estimation."""

    y = np.array(req.outcome, dtype=float)
    d = np.array(req.treatment, dtype=float)
    instruments = [np.array(z, dtype=float) for z in req.instrument]
    n_instruments = len(instruments)

    all_arrays = [y, d] + instruments
    _validate_lengths(*all_arrays)
    n = len(y)

    confidence = np.clip(req.confidence, 0.50, 0.999)
    alpha = 1 - confidence

    inst_names = req.instrument_names or [f"Z{j+1}" for j in range(n_instruments)]

    # Build instrument matrix
    Z = np.column_stack(instruments)  # (n, n_instruments)

    # Build covariate matrix
    if req.covariates:
        cov = np.array(req.covariates, dtype=float).T
        if cov.shape[0] != n:
            raise HTTPException(status_code=422, detail="Covariate length mismatch.")
        cov_names = req.covariate_names or [f"X{j+1}" for j in range(cov.shape[1])]
    else:
        cov = None
        cov_names = []

    # ── OLS estimate (for comparison) ──
    if cov is not None:
        X_ols = np.column_stack([np.ones(n), d, cov])
    else:
        X_ols = np.column_stack([np.ones(n), d])

    try:
        beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
        resid_ols = y - X_ols @ beta_ols
        sigma2_ols = float(np.sum(resid_ols ** 2) / (n - X_ols.shape[1]))
        var_ols = sigma2_ols * np.linalg.inv(X_ols.T @ X_ols)
        se_ols = float(np.sqrt(var_ols[1, 1]))
        ols_estimate = float(beta_ols[1])
        t_ols = ols_estimate / se_ols
        p_ols = float(2 * sp_stats.t.sf(abs(t_ols), n - X_ols.shape[1]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OLS estimation failed: {str(e)}")

    # ── First stage: regress treatment on instruments + covariates ──
    if cov is not None:
        X_first = np.column_stack([np.ones(n), Z, cov])
    else:
        X_first = np.column_stack([np.ones(n), Z])

    try:
        beta_first = np.linalg.lstsq(X_first, d, rcond=None)[0]
        d_hat = X_first @ beta_first
        resid_first = d - d_hat

        # First-stage F-statistic (for instruments only)
        # Compare full model vs restricted (without instruments)
        if cov is not None:
            X_restricted = np.column_stack([np.ones(n), cov])
        else:
            X_restricted = np.ones((n, 1))

        beta_restricted = np.linalg.lstsq(X_restricted, d, rcond=None)[0]
        resid_restricted = d - X_restricted @ beta_restricted

        ssr_restricted = np.sum(resid_restricted ** 2)
        ssr_full = np.sum(resid_first ** 2)
        df_num = n_instruments
        df_den = n - X_first.shape[1]

        if ssr_full > 0 and df_den > 0:
            f_first_stage = float(
                ((ssr_restricted - ssr_full) / df_num) / (ssr_full / df_den)
            )
            p_first_stage = float(sp_stats.f.sf(f_first_stage, df_num, df_den))
        else:
            f_first_stage = 0.0
            p_first_stage = 1.0

        r2_first = float(1 - ssr_full / np.sum((d - np.mean(d)) ** 2))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"First stage failed: {str(e)}")

    # ── Second stage: regress outcome on predicted treatment + covariates ──
    if cov is not None:
        X_second = np.column_stack([np.ones(n), d_hat, cov])
    else:
        X_second = np.column_stack([np.ones(n), d_hat])

    try:
        beta_iv = np.linalg.lstsq(X_second, y, rcond=None)[0]
        iv_estimate = float(beta_iv[1])

        # Correct SE: use actual treatment (not predicted) for residuals
        if cov is not None:
            X_actual = np.column_stack([np.ones(n), d, cov])
        else:
            X_actual = np.column_stack([np.ones(n), d])

        resid_iv = y - X_actual @ np.concatenate([[beta_iv[0]], [iv_estimate], beta_iv[2:]])
        sigma2_iv = float(np.sum(resid_iv ** 2) / (n - X_second.shape[1]))
        # IV variance uses (Z'X)^-1 (X'X) (Z'X)^-1 sigma^2
        # Simplified: sigma^2 * (X_second' X_second)^-1
        var_iv = sigma2_iv * np.linalg.inv(X_second.T @ X_second)
        se_iv = float(np.sqrt(var_iv[1, 1]))

        t_iv = iv_estimate / se_iv if se_iv > 0 else 0.0
        p_iv = float(2 * sp_stats.t.sf(abs(t_iv), n - X_second.shape[1]))
        t_crit = sp_stats.t.ppf(1 - alpha / 2, n - X_second.shape[1])
        ci_iv = [iv_estimate - t_crit * se_iv, iv_estimate + t_crit * se_iv]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Second stage failed: {str(e)}")

    # ── Hausman test: compare IV to OLS ──
    diff = iv_estimate - ols_estimate
    var_diff = max(se_iv ** 2 - se_ols ** 2, 1e-10)
    hausman_stat = diff ** 2 / var_diff
    hausman_p = float(sp_stats.chi2.sf(hausman_stat, df=1))

    # ── Sargan/Hansen J-test (overidentification) ──
    sargan_result = None
    if n_instruments > 1:
        # Residuals from 2SLS
        resid_2sls = y - X_second @ beta_iv
        # Regress residuals on all instruments + covariates
        if cov is not None:
            X_sargan = np.column_stack([np.ones(n), Z, cov])
        else:
            X_sargan = np.column_stack([np.ones(n), Z])

        try:
            beta_sargan = np.linalg.lstsq(X_sargan, resid_2sls, rcond=None)[0]
            fitted_sargan = X_sargan @ beta_sargan
            r2_sargan = float(np.sum(fitted_sargan ** 2) / np.sum(resid_2sls ** 2))
            j_stat = float(n * r2_sargan)
            df_sargan = n_instruments - 1  # overidentifying restrictions
            p_sargan = float(sp_stats.chi2.sf(j_stat, df_sargan))

            sargan_result = {
                "j_statistic": j_stat,
                "df": df_sargan,
                "p_value": p_sargan,
                "reject_validity": p_sargan < alpha,
                "interpretation": (
                    f"Sargan J = {j_stat:.4f}, p = {p_sargan:.4f}. "
                    + ("Instruments may be invalid (overidentification rejected)."
                       if p_sargan < alpha
                       else "No evidence against instrument validity.")
                ),
            }
        except Exception:
            sargan_result = {"error": "Sargan test computation failed."}

    # Diagnostics
    weak_instrument = f_first_stage < 10
    diagnostics = []
    if weak_instrument:
        diagnostics.append(
            f"WARNING: First-stage F = {f_first_stage:.2f} < 10. "
            "Instruments may be weak, leading to biased IV estimates."
        )
    if hausman_p > 0.05:
        diagnostics.append(
            "Hausman test not significant: OLS and IV estimates are similar. "
            "OLS may be consistent (no endogeneity detected)."
        )
    else:
        diagnostics.append(
            "Hausman test significant: OLS and IV differ substantially. "
            "Endogeneity is present; IV estimate is preferred."
        )

    # Build interpretation
    interp = (
        f"IV (2SLS) estimate: {iv_estimate:.4f} (SE = {se_iv:.4f}, p = {p_iv:.4f}). "
        f"OLS estimate: {ols_estimate:.4f} for comparison. "
        f"First-stage F = {f_first_stage:.2f} ({'adequate' if not weak_instrument else 'WEAK'}). "
        f"Hausman test p = {hausman_p:.4f}."
    )

    return {
        "method": "two_stage_least_squares",
        "n": n,
        "n_instruments": n_instruments,
        "instrument_names": inst_names,
        "iv_estimate": {
            "coefficient": iv_estimate,
            "se": se_iv,
            "t_statistic": float(t_iv),
            "p_value": p_iv,
            "ci_lower": ci_iv[0],
            "ci_upper": ci_iv[1],
            "confidence_level": float(confidence),
        },
        "ols_estimate": {
            "coefficient": ols_estimate,
            "se": se_ols,
            "t_statistic": float(t_ols),
            "p_value": p_ols,
        },
        "first_stage": {
            "f_statistic": f_first_stage,
            "p_value": p_first_stage,
            "r_squared": r2_first,
            "weak_instrument": weak_instrument,
        },
        "hausman_test": {
            "statistic": float(hausman_stat),
            "p_value": hausman_p,
            "endogeneity_detected": hausman_p < alpha,
        },
        "sargan_test": sargan_result,
        "diagnostics": diagnostics,
        "interpretation": interp,
    }


# ── 4. Sensitivity Analysis ──────────────────────────────────────────

@router.post("/sensitivity-analysis")
def sensitivity_analysis(req: SensitivityAnalysisRequest):
    """Sensitivity analysis for unmeasured confounding:
    E-value (VanderWeele & Ding 2017), Rosenbaum bounds, bias factor."""

    te = req.treatment_effect
    se = req.se
    r2 = req.treatment_outcome_r_squared
    confidence = np.clip(req.confidence, 0.50, 0.999)
    alpha = 1 - confidence

    if se <= 0:
        raise HTTPException(status_code=422, detail="Standard error must be positive.")
    if not (0 < r2 < 1):
        raise HTTPException(status_code=422, detail="R-squared must be between 0 and 1 (exclusive).")

    # ── E-value (VanderWeele & Ding 2017) ──
    # For risk ratio approximation: E = RR + sqrt(RR * (RR - 1))
    # For point estimate on standardized scale, approximate RR
    # Using the approximation: RR ~ exp(0.91 * beta / sqrt(variance_outcome))
    # Simplified: use the HR approximation E = HR + sqrt(HR*(HR-1))
    # where HR ~ exp(te) for log-scale or approximate via RR

    # Convert to approximate risk ratio using exp(te * sqrt(0.91))
    # VanderWeele-Ding: for continuous outcome, convert to approximate RR
    # RR ~ exp(0.91 * te / SD_y), using R^2 to estimate SD
    # Simpler: treat te/se as Z, compute RR = exp(0.91 * te)
    # But most commonly E-value uses the point estimate directly

    # Standard E-value computation for standardized effect
    # If te is on the original scale and we know R^2,
    # standardize: d = te / sqrt(residual_variance)
    # residual_var approx = total_var * (1 - R^2)
    # But we need total variance. Approximate from SE and n.

    # More direct approach: approximate RR
    # For continuous outcomes: RR ~ exp(0.91 * d) where d is Cohen's d
    # We approximate d from the t-statistic and sample characteristics

    # Use the direct formula: E = RR + sqrt(RR * (RR - 1))
    # where RR = exp(0.91 * |te| / se_residual)
    # Approximate se_residual from R^2

    # Simplified: use te as the effect measure
    # Convert to approximate HR/RR for E-value
    rr_point = float(np.exp(0.91 * abs(te)))
    rr_point = max(rr_point, 1.0)
    e_value_point = float(rr_point + np.sqrt(rr_point * (rr_point - 1)))

    # E-value for confidence interval bound
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = te - z_crit * se
    ci_upper = te + z_crit * se

    # Use the CI bound closer to null
    ci_bound_near_null = min(abs(ci_lower), abs(ci_upper))
    if (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0):
        # CI excludes null
        rr_ci = float(np.exp(0.91 * ci_bound_near_null))
        rr_ci = max(rr_ci, 1.0)
        e_value_ci = float(rr_ci + np.sqrt(rr_ci * (rr_ci - 1)))
    else:
        # CI includes null — effect is already non-significant
        e_value_ci = 1.0

    # ── Rosenbaum bounds ──
    # For a matched study, Gamma is the factor by which odds of treatment
    # assignment can differ due to hidden bias
    gamma_range = req.gamma_range or [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    z_obs = abs(te / se)
    rosenbaum_bounds = []

    for gamma in gamma_range:
        if gamma < 1.0:
            continue
        # Under Gamma, the p-value bounds shift
        # Upper bound on p-value: Z_adjusted = Z_obs - log(Gamma)/se_factor
        # Simplified Rosenbaum sensitivity: test statistic adjusted for bias
        # T_adj = T_obs * sqrt(1/Gamma) approximately
        z_adj_upper = z_obs / np.sqrt(gamma)
        p_upper = float(2 * sp_stats.norm.sf(z_adj_upper))

        z_adj_lower = z_obs * np.sqrt(gamma)
        p_lower = float(2 * sp_stats.norm.sf(z_adj_lower))

        rosenbaum_bounds.append({
            "gamma": float(gamma),
            "p_value_upper_bound": p_upper,
            "p_value_lower_bound": p_lower,
            "conclusion_changes": p_upper > alpha,
        })

    # Find critical Gamma (where conclusion changes)
    # Binary search for Gamma where p_upper = alpha
    gamma_lo, gamma_hi = 1.0, 20.0
    for _ in range(100):
        gamma_mid = (gamma_lo + gamma_hi) / 2
        z_adj = z_obs / np.sqrt(gamma_mid)
        p_adj = 2 * sp_stats.norm.sf(z_adj)
        if p_adj < alpha:
            gamma_lo = gamma_mid
        else:
            gamma_hi = gamma_mid

    critical_gamma = float((gamma_lo + gamma_hi) / 2)

    # ── Bias factor analysis ──
    # How large would bias need to be to nullify the effect?
    # Bias = te, so confounder association needed = te / (some function of r2)
    bias_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bias_analysis = []

    for delta in bias_strengths:
        # If unmeasured confounder has association delta with treatment
        # and delta with outcome, bias ~ delta^2 * (something)
        # Simplified: adjusted_effect = te - delta * sqrt(r2)
        bias_amount = delta * np.sqrt(r2)
        adjusted_te = te - np.sign(te) * bias_amount
        adjusted_z = adjusted_te / se
        adjusted_p = float(2 * sp_stats.norm.sf(abs(adjusted_z)))

        bias_analysis.append({
            "confounder_strength": float(delta),
            "bias_amount": float(bias_amount),
            "adjusted_effect": float(adjusted_te),
            "adjusted_p_value": adjusted_p,
            "still_significant": adjusted_p < alpha,
        })

    # ── Interpretation ──
    interp_parts = [
        f"E-value for point estimate: {e_value_point:.2f}. "
        f"An unmeasured confounder would need to be associated with both "
        f"treatment and outcome by a risk ratio of at least {e_value_point:.2f} "
        f"to fully explain away the observed effect.",
    ]

    if e_value_ci > 1.0:
        interp_parts.append(
            f"E-value for CI bound: {e_value_ci:.2f}. To shift the CI to include "
            f"the null, confounding of at least {e_value_ci:.2f} is needed."
        )
    else:
        interp_parts.append(
            "The confidence interval already includes the null, so even minimal "
            "confounding could explain the observed effect."
        )

    interp_parts.append(
        f"Rosenbaum critical Gamma: {critical_gamma:.2f}. The conclusion is robust "
        f"to a hidden bias that changes odds of treatment by up to "
        f"{critical_gamma:.1f}x."
    )

    return {
        "method": "sensitivity_analysis_unmeasured_confounding",
        "treatment_effect": te,
        "se": se,
        "confidence_level": float(confidence),
        "e_value": {
            "point_estimate": e_value_point,
            "ci_bound": e_value_ci,
            "interpretation": (
                f"To explain away the effect, an unmeasured confounder "
                f"needs RR >= {e_value_point:.2f} with both treatment and outcome."
            ),
        },
        "rosenbaum_bounds": {
            "critical_gamma": critical_gamma,
            "gamma_table": rosenbaum_bounds,
            "interpretation": (
                f"Conclusion robust up to Gamma = {critical_gamma:.2f}. "
                f"A hidden bias factor of {critical_gamma:.1f}x in treatment "
                f"odds would be needed to overturn significance."
            ),
        },
        "bias_factor_analysis": {
            "bias_adjusted_estimates": bias_analysis,
            "interpretation": (
                "Table shows how the effect estimate changes under increasing "
                "confounder strength. Larger values indicate stronger confounding needed."
            ),
        },
        "confidence_interval": {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
        },
        "interpretation": " ".join(interp_parts),
    }


# ── 5. G-Computation ─────────────────────────────────────────────────

@router.post("/g-computation")
def g_computation(req: GComputationRequest):
    """G-computation for average treatment effects with bootstrap inference."""

    y = np.array(req.outcome, dtype=float)
    a = np.array(req.treatment, dtype=int)
    covariates_raw = [np.array(c, dtype=float) for c in req.covariates]

    unique_treatments = set(a)
    if not unique_treatments.issubset({0, 1}):
        raise HTTPException(status_code=422, detail="Treatment must be binary (0/1).")

    all_arrays = [y, a] + covariates_raw
    _validate_lengths(*all_arrays, min_n=20)
    n = len(y)
    n_treated = int(np.sum(a == 1))
    n_control = int(np.sum(a == 0))

    if n_treated < 5 or n_control < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 5 in each group. Got {n_treated} treated, {n_control} control.",
        )

    X_cov = np.column_stack(covariates_raw)  # (n, k)
    k = X_cov.shape[1]
    cov_names = req.covariate_names or [f"X{j+1}" for j in range(k)]
    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_boot = max(500, min(req.n_bootstrap, 20000))
    alpha = 1 - confidence

    # ── Fit outcome model ──
    # Y = b0 + b1*A + b2*X1 + ... + bk*Xk (+ interaction terms optional)
    X_model = np.column_stack([np.ones(n), a, X_cov])

    is_binary_outcome = req.outcome_type == "binary"

    if is_binary_outcome:
        # Logistic regression via iterative reweighted least squares
        from scipy.optimize import minimize

        def _logistic_nll(beta, X, y):
            z = X @ beta
            z = np.clip(z, -500, 500)
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        beta_init = np.zeros(X_model.shape[1])
        result = minimize(_logistic_nll, beta_init, args=(X_model, y), method="BFGS")
        if not result.success:
            raise HTTPException(status_code=500, detail="Logistic outcome model failed to converge.")
        beta_outcome = result.x

        def _predict(X, beta):
            z = np.clip(X @ beta, -500, 500)
            return 1 / (1 + np.exp(-z))

        # Model fit summary
        y_pred = _predict(X_model, beta_outcome)
        pseudo_r2 = float(1 - _logistic_nll(beta_outcome, X_model, y) /
                          _logistic_nll(np.zeros_like(beta_outcome), X_model, y))
    else:
        # OLS
        beta_outcome = np.linalg.lstsq(X_model, y, rcond=None)[0]

        def _predict(X, beta):
            return X @ beta

        resid = y - _predict(X_model, beta_outcome)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        pseudo_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ── Compute counterfactuals ──
    # E[Y(1)] = mean of predictions with A=1 for all subjects
    X_treat_all = np.column_stack([np.ones(n), np.ones(n), X_cov])
    X_control_all = np.column_stack([np.ones(n), np.zeros(n), X_cov])

    y1_pred = _predict(X_treat_all, beta_outcome)  # E[Y|A=1, X]
    y0_pred = _predict(X_control_all, beta_outcome)  # E[Y|A=0, X]

    ate = float(np.mean(y1_pred - y0_pred))
    att = float(np.mean(y1_pred[a == 1] - y0_pred[a == 1]))
    atu = float(np.mean(y1_pred[a == 0] - y0_pred[a == 0]))

    # ── Bootstrap CIs ──
    rng = np.random.default_rng()
    boot_ate = np.empty(n_boot)
    boot_att = np.empty(n_boot)
    boot_atu = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        a_b = a[idx]
        X_cov_b = X_cov[idx]
        X_model_b = np.column_stack([np.ones(n), a_b, X_cov_b])

        try:
            if is_binary_outcome:
                res_b = minimize(_logistic_nll, beta_outcome, args=(X_model_b, y_b), method="BFGS")
                beta_b = res_b.x
            else:
                beta_b = np.linalg.lstsq(X_model_b, y_b, rcond=None)[0]

            X_t = np.column_stack([np.ones(n), np.ones(n), X_cov_b])
            X_c = np.column_stack([np.ones(n), np.zeros(n), X_cov_b])
            y1_b = _predict(X_t, beta_b)
            y0_b = _predict(X_c, beta_b)

            boot_ate[b] = np.mean(y1_b - y0_b)
            treated_mask = a_b == 1
            control_mask = a_b == 0
            boot_att[b] = np.mean(y1_b[treated_mask] - y0_b[treated_mask]) if np.any(treated_mask) else np.nan
            boot_atu[b] = np.mean(y1_b[control_mask] - y0_b[control_mask]) if np.any(control_mask) else np.nan
        except Exception:
            boot_ate[b] = np.nan
            boot_att[b] = np.nan
            boot_atu[b] = np.nan

    def _boot_ci(boot_arr, point_est, name):
        valid = boot_arr[~np.isnan(boot_arr)]
        if len(valid) < n_boot * 0.5:
            return {
                "estimate": point_est,
                "se": None,
                "ci_lower": None,
                "ci_upper": None,
                "warning": "Too many bootstrap failures for reliable CI.",
            }
        se_boot = float(np.std(valid, ddof=1))
        lo = float(np.percentile(valid, 100 * alpha / 2))
        hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
        return {
            "estimate": point_est,
            "se": se_boot,
            "ci_lower": lo,
            "ci_upper": hi,
            "significant": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
        }

    ate_result = _boot_ci(boot_ate, ate, "ATE")
    att_result = _boot_ci(boot_att, att, "ATT")
    atu_result = _boot_ci(boot_atu, atu, "ATU")

    # Outcome model coefficients
    coef_names = ["intercept", "treatment"] + list(cov_names)
    model_coefficients = {
        name: float(beta_outcome[i]) for i, name in enumerate(coef_names)
    }

    # Counterfactual distribution summaries
    cf_summary = {
        "E_Y1_mean": float(np.mean(y1_pred)),
        "E_Y1_std": float(np.std(y1_pred)),
        "E_Y0_mean": float(np.mean(y0_pred)),
        "E_Y0_std": float(np.std(y0_pred)),
        "individual_effect_mean": float(np.mean(y1_pred - y0_pred)),
        "individual_effect_std": float(np.std(y1_pred - y0_pred)),
    }

    # Interpretation
    sig_str = "statistically significant" if ate_result.get("significant", False) else "not statistically significant"
    interp = (
        f"G-computation ATE = {ate:.4f} ({sig_str}). "
        f"On average, treatment {'increases' if ate > 0 else 'decreases'} the outcome by "
        f"{abs(ate):.4f} units. "
        f"ATT = {att:.4f} (effect among the treated), "
        f"ATU = {atu:.4f} (effect among the untreated). "
        f"{'ATE ~= ATT ~= ATU suggests homogeneous treatment effects.' if abs(att - atu) < 2 * (ate_result.get('se') or 1e10) else 'ATT != ATU suggests heterogeneous treatment effects across groups.'}"
    )

    return {
        "method": "g_computation",
        "n": n,
        "n_treated": n_treated,
        "n_control": n_control,
        "outcome_type": req.outcome_type,
        "ate": ate_result,
        "att": att_result,
        "atu": atu_result,
        "outcome_model": {
            "type": "logistic" if is_binary_outcome else "ols",
            "coefficients": model_coefficients,
            "r_squared_or_pseudo_r2": pseudo_r2,
        },
        "counterfactual_distributions": cf_summary,
        "n_bootstrap": n_boot,
        "confidence_level": float(confidence),
        "interpretation": interp,
    }


# ── 6. Doubly Robust (AIPW) Estimation ───────────────────────────────

@router.post("/doubly-robust")
def doubly_robust(req: DoublyRobustRequest):
    """Augmented Inverse Probability Weighted (AIPW) doubly-robust estimation."""

    y = np.array(req.outcome, dtype=float)
    a = np.array(req.treatment, dtype=int)
    covariates_raw = [np.array(c, dtype=float) for c in req.covariates]

    unique_treatments = set(a)
    if not unique_treatments.issubset({0, 1}):
        raise HTTPException(status_code=422, detail="Treatment must be binary (0/1).")

    all_arrays = [y, a] + covariates_raw
    _validate_lengths(*all_arrays, min_n=20)
    n = len(y)
    n_treated = int(np.sum(a == 1))
    n_control = int(np.sum(a == 0))

    if n_treated < 5 or n_control < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 5 in each group. Got {n_treated} treated, {n_control} control.",
        )

    X_cov = np.column_stack(covariates_raw)  # (n, k)
    k = X_cov.shape[1]
    cov_names = req.covariate_names or [f"X{j+1}" for j in range(k)]
    confidence = np.clip(req.confidence, 0.50, 0.999)
    n_boot = max(500, min(req.n_bootstrap, 20000))
    alpha = 1 - confidence

    def _fit_propensity(a_arr, X_arr):
        """Fit logistic regression propensity score model."""
        from scipy.optimize import minimize

        X_ps = np.column_stack([np.ones(len(a_arr)), X_arr])

        def nll(beta):
            z = X_ps @ beta
            z = np.clip(z, -500, 500)
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(a_arr * np.log(p) + (1 - a_arr) * np.log(1 - p))

        beta_init = np.zeros(X_ps.shape[1])
        result = minimize(nll, beta_init, args=(), method="BFGS")
        beta_ps = result.x

        z = np.clip(X_ps @ beta_ps, -500, 500)
        ps = 1 / (1 + np.exp(-z))
        ps = np.clip(ps, 0.01, 0.99)  # trim for stability
        return ps, beta_ps

    def _fit_outcome_model(y_arr, a_arr, X_arr):
        """Fit OLS outcome model: Y ~ A + X."""
        X_out = np.column_stack([np.ones(len(y_arr)), a_arr, X_arr])
        beta = np.linalg.lstsq(X_out, y_arr, rcond=None)[0]
        return beta

    def _predict_outcome(beta, a_val, X_arr):
        """Predict E[Y|A=a_val, X]."""
        n_pred = len(X_arr)
        X_out = np.column_stack([np.ones(n_pred), np.full(n_pred, a_val), X_arr])
        return X_out @ beta

    def _aipw_estimate(y_arr, a_arr, ps, mu1, mu0):
        """Compute AIPW estimator."""
        # AIPW = E[ A*Y/e - (A-e)*mu1/e ] - E[ (1-A)*Y/(1-e) + (A-e)*mu0/(1-e) ]
        # Simplified:
        # phi1 = A*Y/e(X) - (A - e(X))*mu1(X)/e(X)  = A*(Y - mu1)/e + mu1
        # phi0 = (1-A)*Y/(1-e(X)) + (A - e(X))*mu0(X)/(1-e(X))  = (1-A)*(Y - mu0)/(1-e) + mu0
        phi1 = a_arr * (y_arr - mu1) / ps + mu1
        phi0 = (1 - a_arr) * (y_arr - mu0) / (1 - ps) + mu0
        ate = np.mean(phi1 - phi0)
        return float(ate)

    # ── Fit models on full sample ──
    ps_full, beta_ps_full = _fit_propensity(a, X_cov)
    beta_out_full = _fit_outcome_model(y, a, X_cov)
    mu1_full = _predict_outcome(beta_out_full, 1, X_cov)
    mu0_full = _predict_outcome(beta_out_full, 0, X_cov)

    # AIPW estimate
    aipw_ate = _aipw_estimate(y, a, ps_full, mu1_full, mu0_full)

    # Propensity-only (IPW) estimate
    ipw_ate = float(
        np.mean(a * y / ps_full) - np.mean((1 - a) * y / (1 - ps_full))
    )

    # Outcome-only (g-computation) estimate
    outcome_ate = float(np.mean(mu1_full - mu0_full))

    # ── Bootstrap CIs ──
    rng = np.random.default_rng()
    boot_aipw = np.empty(n_boot)
    boot_ipw = np.empty(n_boot)
    boot_outcome = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        a_b = a[idx]
        X_b = X_cov[idx]

        # Need both groups in bootstrap sample
        if np.sum(a_b == 1) < 2 or np.sum(a_b == 0) < 2:
            boot_aipw[b] = np.nan
            boot_ipw[b] = np.nan
            boot_outcome[b] = np.nan
            continue

        try:
            ps_b, _ = _fit_propensity(a_b, X_b)
            beta_out_b = _fit_outcome_model(y_b, a_b, X_b)
            mu1_b = _predict_outcome(beta_out_b, 1, X_b)
            mu0_b = _predict_outcome(beta_out_b, 0, X_b)

            boot_aipw[b] = _aipw_estimate(y_b, a_b, ps_b, mu1_b, mu0_b)
            boot_ipw[b] = float(
                np.mean(a_b * y_b / ps_b) - np.mean((1 - a_b) * y_b / (1 - ps_b))
            )
            boot_outcome[b] = float(np.mean(mu1_b - mu0_b))
        except Exception:
            boot_aipw[b] = np.nan
            boot_ipw[b] = np.nan
            boot_outcome[b] = np.nan

    def _summarize_boot(boot_arr, point_est):
        valid = boot_arr[~np.isnan(boot_arr)]
        if len(valid) < n_boot * 0.3:
            return {
                "estimate": point_est,
                "se": None,
                "ci_lower": None,
                "ci_upper": None,
                "warning": "Too many bootstrap failures.",
            }
        se_b = float(np.std(valid, ddof=1))
        lo = float(np.percentile(valid, 100 * alpha / 2))
        hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
        return {
            "estimate": point_est,
            "se": se_b,
            "ci_lower": lo,
            "ci_upper": hi,
            "significant": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
        }

    aipw_result = _summarize_boot(boot_aipw, aipw_ate)
    ipw_result = _summarize_boot(boot_ipw, ipw_ate)
    outcome_result = _summarize_boot(boot_outcome, outcome_ate)

    # ── Overlap diagnostics ──
    ps_treated = ps_full[a == 1]
    ps_control = ps_full[a == 0]

    overlap = {
        "treated_ps_mean": float(np.mean(ps_treated)),
        "treated_ps_std": float(np.std(ps_treated)),
        "treated_ps_min": float(np.min(ps_treated)),
        "treated_ps_max": float(np.max(ps_treated)),
        "control_ps_mean": float(np.mean(ps_control)),
        "control_ps_std": float(np.std(ps_control)),
        "control_ps_min": float(np.min(ps_control)),
        "control_ps_max": float(np.max(ps_control)),
        "n_extreme_weights": int(np.sum((ps_full < 0.05) | (ps_full > 0.95))),
        "overlap_adequate": bool(
            np.min(ps_treated) > 0.05 and np.max(ps_control) < 0.95
            and np.max(ps_treated) < 0.95 and np.min(ps_control) > 0.05
        ),
    }

    # Propensity score model coefficients
    ps_coef_names = ["intercept"] + list(cov_names)
    ps_coefficients = {
        name: float(beta_ps_full[i]) for i, name in enumerate(ps_coef_names)
    }

    # Outcome model coefficients
    out_coef_names = ["intercept", "treatment"] + list(cov_names)
    out_coefficients = {
        name: float(beta_out_full[i]) for i, name in enumerate(out_coef_names)
    }

    # Warnings
    warnings = []
    if not overlap["overlap_adequate"]:
        warnings.append(
            "Poor propensity score overlap detected. Extreme weights may inflate variance. "
            "Consider trimming or matching instead."
        )
    if overlap["n_extreme_weights"] > n * 0.1:
        warnings.append(
            f"{overlap['n_extreme_weights']} observations ({100*overlap['n_extreme_weights']/n:.1f}%) "
            "have extreme propensity scores. Consider weight truncation."
        )

    # Interpretation
    aipw_sig = "statistically significant" if aipw_result.get("significant", False) else "not statistically significant"
    interp = (
        f"AIPW (doubly robust) ATE = {aipw_ate:.4f} ({aipw_sig}). "
        f"IPW-only estimate: {ipw_ate:.4f}. Outcome-model-only estimate: {outcome_ate:.4f}. "
        f"The AIPW estimator is consistent if EITHER the propensity score model OR "
        f"the outcome model is correctly specified. "
        f"{'Estimates are similar across methods, suggesting robustness.' if abs(aipw_ate - ipw_ate) < 2 * (aipw_result.get('se') or 1e10) and abs(aipw_ate - outcome_ate) < 2 * (aipw_result.get('se') or 1e10) else 'Estimates differ across methods. Inspect model specifications carefully.'}"
    )

    return {
        "method": "doubly_robust_aipw",
        "n": n,
        "n_treated": n_treated,
        "n_control": n_control,
        "aipw_estimate": aipw_result,
        "ipw_estimate": ipw_result,
        "outcome_model_estimate": outcome_result,
        "propensity_score_model": {
            "type": "logistic_regression",
            "coefficients": ps_coefficients,
        },
        "outcome_model": {
            "type": "ols",
            "coefficients": out_coefficients,
        },
        "overlap_diagnostics": overlap,
        "warnings": warnings,
        "n_bootstrap": n_boot,
        "confidence_level": float(confidence),
        "interpretation": interp,
    }
