"""Q-Methodology — systematic study of subjectivity.

Implements by-person factor analysis (inverted factor technique) on Q-sort
data, varimax rotation, exemplar flagging, factor array construction,
and distinguishing/consensus statement identification.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class QAnalyzeRequest(BaseModel):
    """Full Q-methodology analysis."""
    q_sorts: list[list[float]]  # (participants × statements) — forced-distribution rankings
    participant_labels: list[str] | None = None
    statement_labels: list[str] | None = None
    n_factors: int = 3
    extraction: str = "pca"  # "pca" or "centroid"
    rotation: str = "varimax"  # "varimax" or "none"
    flagging_threshold: float | None = None  # auto-computed if None


class QCorrelateRequest(BaseModel):
    """Compute person-by-person correlation matrix from Q-sorts."""
    q_sorts: list[list[float]]  # (participants × statements)
    participant_labels: list[str] | None = None


class QInterpretRequest(BaseModel):
    """Interpret factor arrays — rank statements, find distinguishing/consensus."""
    factor_arrays: dict[str, list[float]]  # factor_name -> z-scores per statement
    statement_labels: list[str]
    significance_level: float = 0.05
    n_sorts: int  # Q-sorts loading on each factor (used when n_loading is not given)
    n_loading: dict[str, int] | None = None  # factor_name -> number of flagged sorts


# ── Helpers ───────────────────────────────────────────────────────────

def _person_correlation_matrix(q_sorts: np.ndarray) -> np.ndarray:
    """Compute correlation matrix between persons (columns = statements).

    Q-methodology correlates persons, not variables — the inverted
    factor analysis technique.
    """
    N, S = q_sorts.shape
    R = np.corrcoef(q_sorts)  # (N, N) person-by-person correlations
    return R


def _extract_pca(R: np.ndarray, n_factors: int) -> np.ndarray:
    """Extract factors via PCA on the person correlation matrix."""
    eigvals, eigvecs = np.linalg.eigh(R)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take top n_factors
    loadings = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0))
    return loadings


def _extract_centroid(R: np.ndarray, n_factors: int) -> np.ndarray:
    """Extract factors via centroid method (Thurstone's approach).

    Iteratively extracts factors by reflecting variables to maximize
    column sums, then computing the centroid factor.
    """
    N = R.shape[0]
    residual = R.copy()
    loadings = np.zeros((N, n_factors))

    for f in range(n_factors):
        # Reflect: flip signs to make all column sums positive
        for _ in range(100):  # max iterations for reflection
            col_sums = residual.sum(axis=0)
            neg_mask = col_sums < 0
            if not np.any(neg_mask):
                break
            residual[neg_mask, :] *= -1
            residual[:, neg_mask] *= -1

        # Centroid factor = normalized column sums
        col_sums = residual.sum(axis=0)
        total = col_sums.sum()
        if total <= 0:
            break
        factor = col_sums / np.sqrt(abs(total))
        loadings[:, f] = factor

        # Deflate: remove this factor's contribution
        residual -= np.outer(factor, factor)

    return loadings


def _varimax_rotation(loadings: np.ndarray, max_iter: int = 1000, eps: float = 1e-5) -> np.ndarray:
    """Varimax with Kaiser normalisation, the algorithm of R's stats::varimax."""
    x = loadings.copy()
    p, nc = x.shape
    if nc < 2:
        return x
    sc = np.sqrt((x ** 2).sum(axis=1))
    sc[sc == 0] = 1.0
    x = x / sc[:, None]
    TT = np.eye(nc)
    d = 0.0
    for _ in range(max_iter):
        z = x @ TT
        B = x.T @ (z ** 3 - z @ np.diag((z ** 2).sum(axis=0)) / p)
        u, sv, vt = np.linalg.svd(B)
        TT = u @ vt
        d_past, d = d, sv.sum()
        if d < d_past * (1 + eps):
            break
    return (x @ TT) * sc[:, None]


def _compute_factor_arrays(q_sorts: np.ndarray, loadings: np.ndarray,
                           flags: np.ndarray, n_factors: int) -> tuple:
    """Factor z-scores and arrays from the flagged Q-sorts (Brown, 1980; qmethod::qzscores).

    Each flagged sort is weighted by w = f / (1 - f^2) with its signed loading f (so a
    negatively loading sort enters reversed); the weighted sum of raw sorts is
    standardised. The factor array maps the z-score ranks onto the forced distribution.
    """
    N, S = q_sorts.shape
    factor_z_scores = np.full((n_factors, S), np.nan)
    factor_arrays = np.full((n_factors, S), np.nan)
    template = np.sort(q_sorts[0])  # forced distribution (identical for every sort)
    for f in range(n_factors):
        flagged = np.where(flags[:, f])[0]
        if len(flagged) == 0:
            continue
        lf = loadings[flagged, f]
        w = lf / (1 - lf ** 2)
        wsum = w @ q_sorts[flagged]
        z = (wsum - wsum.mean()) / wsum.std(ddof=1)
        factor_z_scores[f] = z
        order = np.argsort(z, kind="stable")  # ascending z
        factor_arrays[f, order] = template
    return factor_arrays, factor_z_scores


def _reliability(n_loading: int) -> tuple[float, float]:
    """Composite reliability 0.8n / (1 + 0.8(n - 1)) and SE of factor z-scores sqrt(1 - r)."""
    if n_loading <= 0:
        return float("nan"), float("nan")
    rel = 0.8 * n_loading / (1 + (n_loading - 1) * 0.8)
    return rel, float(np.sqrt(1 - rel))


def _validate_sorts(q_sorts):
    if not q_sorts or len({len(r) for r in q_sorts}) != 1:
        raise HTTPException(status_code=422, detail="Every Q-sort must rank the same statements (equal-length rows).")
    Q = np.array(q_sorts, dtype=float)
    if not np.all(np.isfinite(Q)):
        raise HTTPException(status_code=422, detail="Q-sort values must be numbers.")
    if np.any(Q.std(axis=1) == 0):
        raise HTTPException(status_code=422, detail="A Q-sort with all statements ranked equally cannot be correlated.")
    return Q


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/analyze")
def q_analyze(req: QAnalyzeRequest):
    """Full Q-methodology analysis: factor extraction, rotation, flagging,
    factor arrays, distinguishing and consensus statements.
    """
    Q = _validate_sorts(req.q_sorts)
    N, S = Q.shape  # N participants, S statements

    if N < 3:
        raise HTTPException(status_code=422, detail="Need at least 3 participants")
    if S < 5:
        raise HTTPException(status_code=422, detail="Need at least 5 statements")
    if req.n_factors < 1 or req.n_factors >= N:
        raise HTTPException(status_code=422, detail=f"n_factors must be between 1 and {N - 1}")
    if req.rotation not in ("varimax", "none"):
        raise HTTPException(status_code=422, detail="rotation must be 'varimax' or 'none'")

    p_labels = req.participant_labels or [f"P{i + 1}" for i in range(N)]
    s_labels = req.statement_labels or [f"S{i + 1}" for i in range(S)]

    if len(p_labels) != N:
        raise HTTPException(status_code=422, detail=f"Expected {N} participant labels, got {len(p_labels)}")
    if len(s_labels) != S:
        raise HTTPException(status_code=422, detail=f"Expected {S} statement labels, got {len(s_labels)}")

    # Step 1: Person correlation matrix
    R = _person_correlation_matrix(Q)

    # Step 2: Factor extraction
    if req.extraction == "pca":
        raw_loadings = _extract_pca(R, req.n_factors)
    elif req.extraction == "centroid":
        raw_loadings = _extract_centroid(R, req.n_factors)
    else:
        raise HTTPException(status_code=422, detail="extraction must be 'pca' or 'centroid'")

    # Step 3: Rotation
    if req.rotation == "varimax" and req.n_factors > 1:
        loadings = _varimax_rotation(raw_loadings)
    elif req.rotation == "none":
        loadings = raw_loadings
    else:
        loadings = raw_loadings

    # Order factors by sum of squared loadings and orient each so its loadings sum positive
    # (psych::principal's convention, which R qmethod uses)
    order = np.argsort(-(loadings ** 2).sum(axis=0), kind="stable")
    loadings = loadings[:, order]
    loadings = loadings * np.where(loadings.sum(axis=0) < 0, -1.0, 1.0)

    # Step 4: Flagging (qmethod::qflag): |loading| > 1.96 / sqrt(S) and the squared loading
    # exceeds the sum of the squared loadings on the other factors. Negative loadings can flag.
    threshold = req.flagging_threshold if req.flagging_threshold is not None else 1.96 / np.sqrt(S)
    flags = np.zeros((N, req.n_factors), dtype=bool)
    for i in range(N):
        sq = loadings[i] ** 2
        for f in range(req.n_factors):
            if abs(loadings[i, f]) > threshold and sq[f] > sq.sum() - sq[f]:
                flags[i, f] = True

    # Step 5: Factor z-scores and arrays
    factor_arrays, factor_z = _compute_factor_arrays(Q, loadings, flags, req.n_factors)
    n_load = flags.sum(axis=0)
    rels, ses = zip(*[_reliability(int(n)) for n in n_load])

    # Step 6: Distinguishing / consensus statements (qmethod::qdc): a difference between two
    # factors is significant at p < .05 when it exceeds 1.96 * SED, SED = sqrt(SE_a^2 + SE_b^2).
    valid = [f for f in range(req.n_factors) if n_load[f] > 0]
    statements = []
    for s_idx in range(S):
        sig = {}
        for a_i, f1 in enumerate(valid):
            for f2 in valid[a_i + 1:]:
                sed = float(np.sqrt(ses[f1] ** 2 + ses[f2] ** 2))
                diff = abs(factor_z[f1, s_idx] - factor_z[f2, s_idx])
                sig[(f1, f2)] = sig[(f2, f1)] = bool(diff > 1.96 * sed)
        if len(valid) < 2:
            category = "not tested (fewer than 2 factors with flagged sorts)"
        elif all(sig.values()):
            category = "distinguishes all"
        elif not any(sig.values()):
            category = "consensus"
        else:
            only = [f for f in valid if all(sig[(f, g)] for g in valid if g != f)
                    and not any(sig[(g, h)] for g in valid for h in valid if g != h and f not in (g, h))]
            category = f"distinguishes factor {only[0] + 1} only" if only else "partly distinguishing"
        statements.append({
            "statement": s_labels[s_idx], "statement_index": s_idx, "category": category,
            "z_scores": [None if np.isnan(v) else round(float(v), 4) for v in factor_z[:, s_idx]],
            "array_positions": [None if np.isnan(v) else float(v) for v in factor_arrays[:, s_idx]],
        })

    distinguishing = [{"statement": st["statement"], "statement_index": st["statement_index"],
                       "factor": "all" if st["category"] == "distinguishes all" else int(st["category"].split()[2]),
                       "category": st["category"]}
                      for st in statements if st["category"].startswith("distinguishes")]
    consensus = [{"statement": st["statement"], "statement_index": st["statement_index"], "z_scores": st["z_scores"]}
                 for st in statements if st["category"] == "consensus"]

    eigvals = np.linalg.eigvalsh(R)[::-1]
    total_var = np.sum(eigvals)
    eigenvalue_table = [{"factor": f + 1, "eigenvalue": round(float(eigvals[f]), 4),
                         "variance_explained": round(float(eigvals[f] / total_var * 100), 2),
                         "cumulative": round(float(np.sum(eigvals[:f + 1]) / total_var * 100), 2)}
                        for f in range(min(req.n_factors, len(eigvals)))]

    loadings_table = [{"participant": p_labels[i],
                       "loadings": [round(float(v), 4) for v in loadings[i]],
                       "flagged_on": (int(np.where(flags[i])[0][0]) + 1) if flags[i].any() else None,
                       "communality": round(float((loadings[i] ** 2).sum()), 4)} for i in range(N)]
    factor_arrays_output = [{
        "factor": f + 1,
        "n_exemplars": int(n_load[f]),
        "exemplars": [p_labels[i] for i in range(N) if flags[i, f]],
        "reliability": None if np.isnan(rels[f]) else round(float(rels[f]), 4),
        "se": None if np.isnan(ses[f]) else round(float(ses[f]), 4),
        "z_scores": [None if np.isnan(v) else round(float(v), 4) for v in factor_z[f]],
        "array": [None if np.isnan(v) else float(v) for v in factor_arrays[f]],
    } for f in range(req.n_factors)]

    return {
        "analysis": "q_methodology",
        "n_participants": N,
        "n_statements": S,
        "n_factors": req.n_factors,
        "extraction_method": req.extraction,
        "rotation": req.rotation,
        "flagging_threshold": round(float(threshold), 4),
        "eigenvalues": eigenvalue_table,
        "loadings": loadings_table,
        "factor_arrays": factor_arrays_output,
        "statements": statements,
        "distinguishing_statements": distinguishing,
        "consensus_statements": consensus,
        "statement_labels": s_labels,
        "method_note": ("Flagging: |loading| > 1.96/sqrt(statements) and dominant (Brown, 1980). Factor scores: weighted "
                        "by f/(1-f^2). Differences tested at p < .05 with SEs from factor reliability (as R qmethod)."),
    }


@router.post("/correlate")
def q_correlate(req: QCorrelateRequest):
    """Compute person-by-person correlation matrix from Q-sort data.

    This is the inverted factor analysis approach: correlations are between
    persons (how similarly they sorted statements), not between variables.
    """
    Q = _validate_sorts(req.q_sorts)
    N, S = Q.shape

    if N < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 participants")

    p_labels = req.participant_labels or [f"P{i + 1}" for i in range(N)]
    if len(p_labels) != N:
        raise HTTPException(status_code=422, detail=f"Expected {N} participant labels, got {len(p_labels)}")

    R = _person_correlation_matrix(Q)

    # Summary statistics
    off_diag = R[np.triu_indices(N, k=1)]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append({
                "participant_1": p_labels[i],
                "participant_2": p_labels[j],
                "r": round(float(R[i, j]), 4),
            })

    # Sort by correlation
    pairs.sort(key=lambda x: x["r"], reverse=True)

    return {
        "analysis": "q_correlation_matrix",
        "n_participants": N,
        "n_statements": S,
        "correlation_matrix": [[round(float(R[i, j]), 4) for j in range(N)] for i in range(N)],
        "labels": p_labels,
        "summary": {
            "mean_r": round(float(np.mean(off_diag)), 4),
            "sd_r": round(float(np.std(off_diag, ddof=1)), 4) if len(off_diag) > 1 else None,
            "min_r": round(float(np.min(off_diag)), 4),
            "max_r": round(float(np.max(off_diag)), 4),
            "median_r": round(float(np.median(off_diag)), 4),
        },
        "pairwise": pairs,
        "interpretation": "Higher correlation = more similar subjective viewpoints between participants",
    }


@router.post("/interpret")
def q_interpret(req: QInterpretRequest):
    """Interpret factor arrays: rank statements per factor, identify
    highest/lowest rated, and pairwise distinguishing statements.
    """
    factor_names = list(req.factor_arrays.keys())
    n_factors = len(factor_names)
    S = len(req.statement_labels)

    if n_factors < 1:
        raise HTTPException(status_code=422, detail="Need at least 1 factor array")

    # Validate lengths
    for fname, z_scores in req.factor_arrays.items():
        if len(z_scores) != S:
            raise HTTPException(
                status_code=422,
                detail=f"Factor '{fname}' has {len(z_scores)} z-scores but {S} statement labels",
            )

    z_matrix = np.array([req.factor_arrays[f] for f in factor_names], dtype=float)  # (n_factors, S)

    # SE of each factor's z-scores from its reliability (Brown, 1980); SED for a pair = sqrt(SE_a^2 + SE_b^2)
    n_load = {f: (req.n_loading or {}).get(f, req.n_sorts) for f in factor_names}
    if any(v < 1 for v in n_load.values()):
        raise HTTPException(status_code=422, detail="Each factor needs at least one loading Q-sort.")
    se_f = {f: _reliability(n_load[f])[1] for f in factor_names}
    z_crit = sp_stats.norm.ppf(1 - req.significance_level / 2)

    # Per-factor rankings
    factor_rankings = {}
    for f_idx, fname in enumerate(factor_names):
        z_scores = z_matrix[f_idx]
        ranked_indices = np.argsort(-z_scores)  # descending

        ranked_statements = []
        for rank, s_idx in enumerate(ranked_indices):
            ranked_statements.append({
                "rank": rank + 1,
                "statement": req.statement_labels[s_idx],
                "z_score": round(float(z_scores[s_idx]), 4),
            })

        # Top and bottom
        n_extremes = min(5, S // 2)
        highest = ranked_statements[:n_extremes]
        lowest = ranked_statements[-n_extremes:]

        factor_rankings[fname] = {
            "full_ranking": ranked_statements,
            "highest_rated": highest,
            "lowest_rated": lowest,
        }

    # Pairwise distinguishing statements
    pairwise_distinguishing = {}
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            f1, f2 = factor_names[i], factor_names[j]
            key = f"{f1}_vs_{f2}"
            diffs = z_matrix[i] - z_matrix[j]
            se_diff = float(np.sqrt(se_f[f1] ** 2 + se_f[f2] ** 2))

            distinguishing = []
            for s_idx in range(S):
                diff = float(diffs[s_idx])
                z_test = abs(diff) / se_diff if se_diff > 0 else 0.0
                p_val = 2 * (sp_stats.norm.sf(z_test))

                if p_val < req.significance_level:
                    distinguishing.append({
                        "statement": req.statement_labels[s_idx],
                        "z_score_f1": round(float(z_matrix[i, s_idx]), 4),
                        "z_score_f2": round(float(z_matrix[j, s_idx]), 4),
                        "difference": round(diff, 4),
                        "z_test": round(float(z_test), 4),
                        "p_value": round(float(p_val), 4),
                        "higher_in": f1 if diff > 0 else f2,
                    })

            # Sort by absolute difference
            distinguishing.sort(key=lambda x: abs(x["difference"]), reverse=True)

            pairwise_distinguishing[key] = {
                "n_distinguishing": len(distinguishing),
                "statements": distinguishing,
            }

    # Consensus statements (non-significant across ALL pairs)
    consensus = []
    for s_idx in range(S):
        is_consensus = True
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                diff = abs(z_matrix[i, s_idx] - z_matrix[j, s_idx])
                se_diff = float(np.sqrt(se_f[factor_names[i]] ** 2 + se_f[factor_names[j]] ** 2))
                z_test = diff / se_diff if se_diff > 0 else 0.0
                if 2 * (sp_stats.norm.sf(z_test)) < req.significance_level:
                    is_consensus = False
                    break
            if not is_consensus:
                break

        if is_consensus:
            consensus.append({
                "statement": req.statement_labels[s_idx],
                "z_scores": {factor_names[f]: round(float(z_matrix[f, s_idx]), 4)
                             for f in range(n_factors)},
                "average_z": round(float(np.mean(z_matrix[:, s_idx])), 4),
            })

    return {
        "analysis": "q_interpretation",
        "n_factors": n_factors,
        "n_statements": S,
        "significance_level": req.significance_level,
        "factor_rankings": factor_rankings,
        "pairwise_distinguishing": pairwise_distinguishing,
        "consensus_statements": consensus,
        "n_consensus": len(consensus),
    }
