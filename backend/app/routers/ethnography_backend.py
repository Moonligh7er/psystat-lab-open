"""Ethnography — qualitative & ethnographic research methods."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS
from itertools import combinations
from collections import Counter

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class CulturalConsensusRequest(BaseModel):
    responses: list[list[int]]          # rows=informants, cols=questions
    question_labels: list[str] | None = None
    informant_labels: list[str] | None = None
    model: str = "formal"               # "formal" (binary) or "informal" (continuous)


class IntercoderRequest(BaseModel):
    coder_assignments: list[list[int | str]]  # rows=items, cols=coders
    code_labels: list[str] | None = None


class ThematicFrequencyRequest(BaseModel):
    codes: list[list[str]]              # each inner list = codes for one document
    document_labels: list[str] | None = None
    group_variable: list[str] | None = None


class ContentAnalysisRequest(BaseModel):
    texts: list[str]
    categories: list[list[str]]
    word_counts: list[int] | None = None


class PileSortRequest(BaseModel):
    items: list[str]
    sorts: list[list[list[int]]]        # each sort = list of piles, each pile = list of item indices


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/cultural-consensus")
def cultural_consensus(req: CulturalConsensusRequest):
    """Cultural Consensus Model (Romney, Weller & Batchelder, 1986)."""
    R = np.array(req.responses, dtype=float)
    n_informants, n_questions = R.shape

    if n_informants < 3:
        raise HTTPException(400, "Need at least 3 informants.")
    if n_questions < 3:
        raise HTTPException(400, "Need at least 3 questions.")

    labels_q = req.question_labels or [f"Q{i+1}" for i in range(n_questions)]
    labels_i = req.informant_labels or [f"Informant_{i+1}" for i in range(n_informants)]

    if len(labels_q) != n_questions:
        raise HTTPException(400, "question_labels length must match number of questions.")
    if len(labels_i) != n_informants:
        raise HTTPException(400, "informant_labels length must match number of informants.")

    # ── Build informant-by-informant agreement matrix ──
    if req.model == "formal":
        # Binary match proportion
        agree = np.zeros((n_informants, n_informants))
        for i in range(n_informants):
            for j in range(n_informants):
                agree[i, j] = np.mean(R[i] == R[j])
    else:
        # Continuous: Pearson correlation
        agree = np.corrcoef(R)
        # Replace NaN diagonal / off-diagonal with 0
        agree = np.nan_to_num(agree, nan=0.0)

    # ── Eigendecomposition ──
    eigenvalues, eigenvectors = np.linalg.eigh(agree)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues_pos = np.maximum(eigenvalues, 0)  # clip negatives
    ratio_1_2 = float(eigenvalues_pos[0] / eigenvalues_pos[1]) if eigenvalues_pos[1] > 1e-10 else float("inf")
    consensus_exists = ratio_1_2 >= 3.0

    # ── Competence scores (first eigenvector, sign-corrected) ──
    competence = eigenvectors[:, 0].copy()
    if np.sum(competence) < 0:
        competence = -competence
    # Normalize to [0, 1] range
    comp_min, comp_max = competence.min(), competence.max()
    if comp_max - comp_min > 1e-10:
        competence_norm = (competence - comp_min) / (comp_max - comp_min)
    else:
        competence_norm = np.ones(n_informants) / n_informants

    # ── Answer key (competence-weighted majority) ──
    answer_key = []
    item_difficulty = []
    if req.model == "formal":
        unique_vals = sorted(set(R.flatten().astype(int)))
        for q in range(n_questions):
            best_val, best_score = None, -1
            for val in unique_vals:
                score = np.sum(competence_norm[R[:, q] == val])
                if score > best_score:
                    best_score = score
                    best_val = val
            answer_key.append(int(best_val))
            # Difficulty = proportion who got it "wrong"
            correct_prop = np.mean(R[:, q] == best_val)
            item_difficulty.append(float(1 - correct_prop))
    else:
        # Continuous: competence-weighted mean
        for q in range(n_questions):
            weights = competence_norm / (competence_norm.sum() + 1e-10)
            weighted_answer = np.sum(weights * R[:, q])
            answer_key.append(float(round(weighted_answer, 4)))
            # Difficulty: std of responses (higher = harder / less agreement)
            item_difficulty.append(float(np.std(R[:, q])))

    # ── Fit: proportion of variance explained by first factor ──
    total_var = np.sum(eigenvalues_pos)
    var_explained_1 = float(eigenvalues_pos[0] / total_var) if total_var > 0 else 0.0
    var_explained_2 = float(eigenvalues_pos[1] / total_var) if total_var > 0 else 0.0

    # ── Interpretation ──
    if consensus_exists:
        interp = (
            f"Cultural consensus EXISTS. The 1st-to-2nd eigenvalue ratio is {ratio_1_2:.2f} "
            f"(threshold: 3.0). The first factor explains {var_explained_1*100:.1f}% of variance, "
            f"indicating a shared cultural model among informants."
        )
    else:
        interp = (
            f"Cultural consensus is NOT supported. The 1st-to-2nd eigenvalue ratio is {ratio_1_2:.2f} "
            f"(below 3.0 threshold). This suggests multiple cultural models or insufficient agreement."
        )

    return {
        "consensus_exists": consensus_exists,
        "eigenvalue_ratio_1_2": ratio_1_2,
        "eigenvalues": [float(e) for e in eigenvalues_pos[:min(5, len(eigenvalues_pos))]],
        "variance_explained": {
            "factor_1": var_explained_1,
            "factor_2": var_explained_2,
        },
        "competence_scores": [
            {"informant": labels_i[i], "competence": float(round(competence_norm[i], 4))}
            for i in range(n_informants)
        ],
        "answer_key": [
            {"question": labels_q[q], "answer": answer_key[q], "difficulty": item_difficulty[q]}
            for q in range(n_questions)
        ],
        "fit": {
            "variance_explained_factor1": var_explained_1,
            "all_competence_positive": bool(np.all(competence_norm > 0)),
            "mean_competence": float(np.mean(competence_norm)),
        },
        "n_informants": n_informants,
        "n_questions": n_questions,
        "model": req.model,
        "interpretation": interp,
    }


@router.post("/intercoder-reliability")
def intercoder_reliability(req: IntercoderRequest):
    """Inter-coder reliability: percent agreement, kappa, Krippendorff's alpha."""
    data = req.coder_assignments  # rows=items, cols=coders
    if not data or not data[0]:
        raise HTTPException(400, "coder_assignments must be a non-empty 2D array.")

    n_items = len(data)
    n_coders = len(data[0])
    for row in data:
        if len(row) != n_coders:
            raise HTTPException(400, "All items must have the same number of coder assignments.")

    if n_coders < 2:
        raise HTTPException(400, "Need at least 2 coders.")
    if n_items < 2:
        raise HTTPException(400, "Need at least 2 items.")

    # Convert everything to strings for uniformity
    mat = [[str(cell) for cell in row] for row in data]
    all_codes = sorted(set(c for row in mat for c in row))
    code_labels = req.code_labels or all_codes

    # ── Percent agreement ──
    agreements = sum(1 for row in mat if len(set(row)) == 1)
    pct_agreement = agreements / n_items

    results: dict = {
        "n_items": n_items,
        "n_coders": n_coders,
        "percent_agreement": float(round(pct_agreement, 4)),
    }

    # ── Cohen's kappa (2 coders) or Fleiss' kappa (3+) ──
    if n_coders == 2:
        results["cohens_kappa"] = _cohens_kappa(mat, all_codes)
        results["scotts_pi"] = _scotts_pi(mat, all_codes)
        results["confusion_matrix"] = _confusion_matrix(mat, all_codes)
    else:
        results["fleiss_kappa"] = _fleiss_kappa(mat, all_codes)
        results["cohens_kappa"] = None
        results["scotts_pi"] = None

    # ── Krippendorff's alpha ──
    results["krippendorffs_alpha"] = _krippendorff_alpha(mat, all_codes)

    # ── Per-category agreement ──
    per_cat = {}
    for code in all_codes:
        relevant = [row for row in mat if code in row]
        if relevant:
            agreed = sum(1 for row in relevant if all(c == code for c in row))
            per_cat[code] = float(round(agreed / len(relevant), 4))
        else:
            per_cat[code] = None
    results["per_category_agreement"] = per_cat

    # ── Interpretation ──
    # Use best available kappa-type metric
    kappa = results.get("cohens_kappa") or results.get("fleiss_kappa") or 0
    results["interpretation"] = _interpret_kappa(kappa)
    results["code_labels"] = code_labels

    return results


@router.post("/thematic-frequency")
def thematic_frequency(req: ThematicFrequencyRequest):
    """Thematic frequency analysis for coded qualitative data."""
    if not req.codes:
        raise HTTPException(400, "codes must be a non-empty list.")

    n_docs = len(req.codes)
    doc_labels = req.document_labels or [f"Doc_{i+1}" for i in range(n_docs)]

    if req.document_labels and len(req.document_labels) != n_docs:
        raise HTTPException(400, "document_labels length must match number of code lists.")

    all_codes_flat = [c for doc in req.codes for c in doc]
    if not all_codes_flat:
        raise HTTPException(400, "No codes found in any document.")

    unique_codes = sorted(set(all_codes_flat))
    total_count = len(all_codes_flat)

    # ── Frequency table ──
    code_counts = Counter(all_codes_flat)
    freq_table = [
        {"code": code, "count": code_counts[code], "percentage": float(round(code_counts[code] / total_count * 100, 2))}
        for code in sorted(code_counts, key=code_counts.get, reverse=True)
    ]

    # ── Co-occurrence matrix ──
    code_idx = {c: i for i, c in enumerate(unique_codes)}
    n_codes = len(unique_codes)
    cooccur = np.zeros((n_codes, n_codes), dtype=int)
    for doc in req.codes:
        doc_unique = list(set(doc))
        for i in range(len(doc_unique)):
            for j in range(i + 1, len(doc_unique)):
                ci, cj = code_idx[doc_unique[i]], code_idx[doc_unique[j]]
                cooccur[ci, cj] += 1
                cooccur[cj, ci] += 1

    # Top code pairs
    pairs = []
    for i in range(n_codes):
        for j in range(i + 1, n_codes):
            if cooccur[i, j] > 0:
                pairs.append({
                    "code_a": unique_codes[i],
                    "code_b": unique_codes[j],
                    "co_occurrences": int(cooccur[i, j]),
                })
    pairs.sort(key=lambda x: x["co_occurrences"], reverse=True)

    # ── Code density per document ──
    density = [
        {"document": doc_labels[i], "n_codes": len(req.codes[i]), "unique_codes": len(set(req.codes[i]))}
        for i in range(n_docs)
    ]

    # ── Network data for visualization ──
    network_nodes = [{"id": c, "frequency": code_counts[c]} for c in unique_codes]
    network_edges = [{"source": p["code_a"], "target": p["code_b"], "weight": p["co_occurrences"]} for p in pairs]

    result: dict = {
        "n_documents": n_docs,
        "n_unique_codes": n_codes,
        "total_code_instances": total_count,
        "frequency_table": freq_table,
        "co_occurrence_matrix": {
            "labels": unique_codes,
            "matrix": cooccur.tolist(),
        },
        "top_code_pairs": pairs[:20],
        "code_density": density,
        "network_data": {"nodes": network_nodes, "edges": network_edges},
    }

    # ── Group comparison (optional) ──
    if req.group_variable is not None:
        if len(req.group_variable) != n_docs:
            raise HTTPException(400, "group_variable length must match number of documents.")

        groups = sorted(set(req.group_variable))
        group_freq: dict[str, dict[str, int]] = {g: Counter() for g in groups}
        for i, doc in enumerate(req.codes):
            g = req.group_variable[i]
            for code in doc:
                group_freq[g][code] += 1

        # Frequency by group table
        freq_by_group = {
            g: [{"code": c, "count": group_freq[g].get(c, 0)} for c in unique_codes]
            for g in groups
        }

        # Chi-square test per code (across groups)
        chi_sq_results = []
        for code in unique_codes:
            observed = [group_freq[g].get(code, 0) for g in groups]
            total_in_groups = [sum(group_freq[g].values()) for g in groups]
            # Expected proportions based on overall total
            total_obs = sum(observed)
            total_all = sum(total_in_groups)
            if total_all == 0 or total_obs == 0:
                continue
            expected = [(total_obs * t / total_all) for t in total_in_groups]
            if all(e > 0 for e in expected):
                chi2, p = sp_stats.chisquare(observed, f_exp=expected)
                chi_sq_results.append({
                    "code": code,
                    "chi_square": float(round(chi2, 4)),
                    "p_value": float(round(p, 4)),
                    "significant": p < 0.05,
                    "observed": {g: observed[i] for i, g in enumerate(groups)},
                })

        result["group_comparison"] = {
            "groups": groups,
            "frequency_by_group": freq_by_group,
            "chi_square_tests": chi_sq_results,
        }

    return result


@router.post("/content-analysis")
def content_analysis(req: ContentAnalysisRequest):
    """Quantitative content analysis metrics."""
    if not req.texts:
        raise HTTPException(400, "texts must be non-empty.")
    if len(req.texts) != len(req.categories):
        raise HTTPException(400, "texts and categories must have the same length.")

    n_texts = len(req.texts)

    if req.word_counts is not None and len(req.word_counts) != n_texts:
        raise HTTPException(400, "word_counts length must match texts length.")

    all_cats = [c for cats in req.categories for c in cats]
    if not all_cats:
        raise HTTPException(400, "No categories found.")

    unique_cats = sorted(set(all_cats))
    cat_counts = Counter(all_cats)
    total_cats = len(all_cats)

    # ── Category frequency distribution ──
    freq_dist = [
        {"category": cat, "count": cat_counts[cat], "proportion": float(round(cat_counts[cat] / total_cats, 4))}
        for cat in sorted(cat_counts, key=cat_counts.get, reverse=True)
    ]

    # ── Category proportions with 95% CI (Wilson interval) ──
    proportions_ci = []
    for cat in unique_cats:
        p = cat_counts[cat] / total_cats
        n = total_cats
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        proportions_ci.append({
            "category": cat,
            "proportion": float(round(p, 4)),
            "ci_lower": float(round(max(0, center - margin), 4)),
            "ci_upper": float(round(min(1, center + margin), 4)),
        })

    # ── Categories per text ──
    cats_per_text = [len(cats) for cats in req.categories]
    unique_cats_per_text = [len(set(cats)) for cats in req.categories]
    avg_cats = float(np.mean(cats_per_text))

    # ── Richness index (type-token ratio analogue) ──
    # TTR analogue: unique categories / total category assignments per text
    richness_per_text = []
    for i in range(n_texts):
        total_c = len(req.categories[i])
        unique_c = len(set(req.categories[i]))
        richness_per_text.append(float(round(unique_c / total_c, 4)) if total_c > 0 else 0.0)
    avg_richness = float(np.mean(richness_per_text)) if richness_per_text else 0.0

    # ── Word count statistics ──
    word_stats = None
    category_density = None
    if req.word_counts is not None:
        wc = np.array(req.word_counts, dtype=float)
        if np.any(wc <= 0):
            raise HTTPException(400, "word_counts must all be positive.")
        word_stats = {
            "mean": float(round(np.mean(wc), 2)),
            "median": float(round(np.median(wc), 2)),
            "sd": float(round(np.std(wc, ddof=1), 2)) if len(wc) > 1 else 0.0,
            "min": int(np.min(wc)),
            "max": int(np.max(wc)),
            "total": int(np.sum(wc)),
        }
        # Category density: categories per 100 words
        density_vals = [(len(req.categories[i]) / wc[i]) * 100 for i in range(n_texts)]
        category_density = {
            "per_text": [float(round(d, 2)) for d in density_vals],
            "mean": float(round(np.mean(density_vals), 2)),
            "sd": float(round(np.std(density_vals, ddof=1), 2)) if len(density_vals) > 1 else 0.0,
        }

    return {
        "n_texts": n_texts,
        "n_unique_categories": len(unique_cats),
        "total_category_assignments": total_cats,
        "frequency_distribution": freq_dist,
        "proportions_with_ci": proportions_ci,
        "categories_per_text": {
            "mean": avg_cats,
            "sd": float(round(np.std(cats_per_text, ddof=1), 4)) if n_texts > 1 else 0.0,
            "min": int(min(cats_per_text)),
            "max": int(max(cats_per_text)),
            "per_text": cats_per_text,
        },
        "richness_index": {
            "per_text": richness_per_text,
            "mean": avg_richness,
            "description": "Ratio of unique categories to total assignments per text (1.0 = all unique).",
        },
        "word_count_statistics": word_stats,
        "category_density": category_density,
    }


@router.post("/pile-sort")
def pile_sort(req: PileSortRequest):
    """Free pile sort analysis — co-occurrence, MDS, hierarchical clustering."""
    n_items = len(req.items)
    if n_items < 3:
        raise HTTPException(400, "Need at least 3 items.")
    if not req.sorts:
        raise HTTPException(400, "Need at least 1 sort.")

    # Validate item indices
    for s_idx, sort in enumerate(req.sorts):
        all_indices = [idx for pile in sort for idx in pile]
        if sorted(set(all_indices)) != list(range(n_items)):
            raise HTTPException(
                400,
                f"Sort {s_idx} must include each item index (0 to {n_items-1}) exactly once."
            )

    n_sorts = len(req.sorts)

    # ── Co-occurrence / similarity matrix ──
    cooccur = np.zeros((n_items, n_items), dtype=float)
    for sort in req.sorts:
        for pile in sort:
            for i in range(len(pile)):
                for j in range(i + 1, len(pile)):
                    cooccur[pile[i], pile[j]] += 1
                    cooccur[pile[j], pile[i]] += 1

    similarity = cooccur / n_sorts  # proportion of times sorted together
    np.fill_diagonal(similarity, 1.0)

    # ── Distance matrix ──
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    # Ensure symmetry and non-negative
    distance = np.maximum(distance, 0.0)
    distance = (distance + distance.T) / 2

    # ── MDS (2D) ──
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
        coords = mds.fit_transform(distance)
        stress = float(mds.stress_)
        mds_points = [
            {"item": req.items[i], "x": float(round(coords[i, 0], 4)), "y": float(round(coords[i, 1], 4))}
            for i in range(n_items)
        ]
    except Exception as e:
        raise HTTPException(500, f"MDS computation failed: {str(e)}")

    # ── Hierarchical clustering ──
    condensed = squareform(distance)
    Z = linkage(condensed, method="average")

    # Cluster assignments at various cut levels
    cluster_solutions = {}
    max_k = min(n_items - 1, 10)
    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        clusters: dict[int, list[str]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(req.items[i])
        cluster_solutions[f"k={k}"] = clusters

    # Dendrogram linkage data (for frontend rendering)
    dendrogram_data = [
        {
            "cluster_1": int(row[0]),
            "cluster_2": int(row[1]),
            "distance": float(round(row[2], 4)),
            "n_items": int(row[3]),
        }
        for row in Z
    ]

    return {
        "n_items": n_items,
        "n_sorts": n_sorts,
        "similarity_matrix": {
            "labels": req.items,
            "matrix": [[float(round(v, 4)) for v in row] for row in similarity],
        },
        "mds": {
            "coordinates": mds_points,
            "stress": stress,
            "stress_interpretation": _interpret_mds_stress(stress),
        },
        "hierarchical_clustering": {
            "linkage": dendrogram_data,
            "method": "average",
            "cluster_solutions": cluster_solutions,
        },
    }


# ── Helper functions ──────────────────────────────────────────────────

def _cohens_kappa(mat: list[list[str]], codes: list[str]) -> float:
    """Cohen's kappa for 2 coders."""
    n = len(mat)
    code_idx = {c: i for i, c in enumerate(codes)}
    k = len(codes)

    # Build contingency table
    table = np.zeros((k, k), dtype=int)
    for row in mat:
        table[code_idx[row[0]], code_idx[row[1]]] += 1

    po = np.trace(table) / n  # observed agreement
    row_sums = table.sum(axis=1) / n
    col_sums = table.sum(axis=0) / n
    pe = np.sum(row_sums * col_sums)  # expected agreement

    if pe >= 1.0:
        return 1.0
    return float(round((po - pe) / (1 - pe), 4))


def _scotts_pi(mat: list[list[str]], codes: list[str]) -> float:
    """Scott's pi for 2 coders."""
    n = len(mat)
    code_idx = {c: i for i, c in enumerate(codes)}
    k = len(codes)

    table = np.zeros((k, k), dtype=int)
    for row in mat:
        table[code_idx[row[0]], code_idx[row[1]]] += 1

    po = np.trace(table) / n

    # Joint marginals (average of both coders)
    joint = (table.sum(axis=0) + table.sum(axis=1)) / (2 * n)
    pe = np.sum(joint ** 2)

    if pe >= 1.0:
        return 1.0
    return float(round((po - pe) / (1 - pe), 4))


def _fleiss_kappa(mat: list[list[str]], codes: list[str]) -> float:
    """Fleiss' kappa for 3+ coders."""
    n_items = len(mat)
    n_coders = len(mat[0])
    code_idx = {c: i for i, c in enumerate(codes)}
    k = len(codes)

    # Build category count matrix: n_items x k
    counts = np.zeros((n_items, k), dtype=int)
    for i, row in enumerate(mat):
        for cell in row:
            counts[i, code_idx[cell]] += 1

    # Proportion each category was used
    p_j = counts.sum(axis=0) / (n_items * n_coders)

    # Per-item agreement
    P_i = (np.sum(counts ** 2, axis=1) - n_coders) / (n_coders * (n_coders - 1))
    P_bar = np.mean(P_i)

    Pe = np.sum(p_j ** 2)

    if Pe >= 1.0:
        return 1.0
    return float(round((P_bar - Pe) / (1 - Pe), 4))


def _krippendorff_alpha(mat: list[list[str]], codes: list[str]) -> float:
    """Krippendorff's alpha (nominal level) for any number of coders."""
    n_items = len(mat)
    n_coders = len(mat[0])
    code_idx = {c: i for i, c in enumerate(codes)}
    k = len(codes)

    # Observed disagreement
    D_o = 0.0
    total_pairs = 0
    for row in mat:
        # All coder pairs within this item
        for a, b in combinations(row, 2):
            D_o += (0 if a == b else 1)
            total_pairs += 1

    if total_pairs == 0:
        return 1.0
    D_o /= total_pairs

    # Expected disagreement (based on marginal frequencies)
    all_values = [c for row in mat for c in row]
    total_vals = len(all_values)
    val_counts = Counter(all_values)

    D_e = 0.0
    for c1 in codes:
        for c2 in codes:
            if c1 != c2:
                D_e += val_counts.get(c1, 0) * val_counts.get(c2, 0)
    D_e /= (total_vals * (total_vals - 1)) if total_vals > 1 else 1

    if D_e == 0:
        return 1.0
    return float(round(1 - D_o / D_e, 4))


def _confusion_matrix(mat: list[list[str]], codes: list[str]) -> dict:
    """Confusion matrix for 2 coders."""
    code_idx = {c: i for i, c in enumerate(codes)}
    k = len(codes)
    table = np.zeros((k, k), dtype=int)
    for row in mat:
        table[code_idx[row[0]], code_idx[row[1]]] += 1
    return {
        "labels": codes,
        "matrix": table.tolist(),
    }


def _interpret_kappa(kappa: float) -> str:
    """Interpret kappa using Landis & Koch (1977) benchmarks."""
    if kappa < 0:
        return "Poor agreement (less than chance)"
    elif kappa < 0.21:
        return "Slight agreement"
    elif kappa < 0.41:
        return "Fair agreement"
    elif kappa < 0.61:
        return "Moderate agreement"
    elif kappa < 0.81:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def _interpret_mds_stress(stress: float) -> str:
    """Interpret MDS stress using Kruskal's guidelines."""
    if stress < 0.025:
        return "Excellent fit"
    elif stress < 0.05:
        return "Good fit"
    elif stress < 0.1:
        return "Fair fit"
    elif stress < 0.2:
        return "Poor fit"
    else:
        return "Unacceptable fit — interpret with caution"
