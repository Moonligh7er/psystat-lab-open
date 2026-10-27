"""Generalizability Theory — G-study variance component estimation, D-study
optimization, and universe score estimation (Cronbach, Gleser, Nanda & Rajaratnam).

Implements variance component estimation via expected mean squares (EMS) for
crossed person-by-item (p x i) and person-by-item-by-occasion (p x i x o)
designs, D-study coefficient computation for relative and absolute decisions,
and universe score confidence intervals.

For these balanced designs the EMS (ANOVA) estimates equal REML estimates whenever none
is negative (checked against lme4 in tests/test_generalizability.py). Negative estimates
are set to zero (Cronbach et al., 1972; Brennan, 2001) and reported as such.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class GStudyRequest(BaseModel):
    """G-study: estimate variance components from observed scores."""
    scores: list[list[float]]  # person x condition matrix (persons in rows)
    design: str = "p_x_i"  # "p_x_i" or "p_x_i_x_o"
    facet_structure: dict | None = None  # for p_x_i_x_o: {"n_items": int, "n_occasions": int}
    facet_names: list[str] | None = None


class DStudyRequest(BaseModel):
    """D-study: optimize measurement design using variance components."""
    variance_components: dict  # from G-study output: {"person": float, "item": float, ...}
    design: str = "p_x_i"  # "p_x_i" or "p_x_i_x_o"
    proposed_counts: dict  # {"n_items": [1,2,...,10]} or {"n_items": [...], "n_occasions": [...]}
    facet_names: list[str] | None = None


class UniverseScoreRequest(BaseModel):
    """Estimate universe scores and confidence intervals."""
    scores: list[list[float]]  # person x condition matrix
    variance_components: dict  # from G-study
    design: str = "p_x_i"
    confidence: float = 0.95


# ── Helper Functions ──────────────────────────────────────────────────

def _validate_score_matrix(scores: list[list[float]], min_persons: int = 3,
                           min_conditions: int = 2):
    """Validate score matrix dimensions."""
    n_persons = len(scores)
    if n_persons < min_persons:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_persons} persons, got {n_persons}.",
        )

    if not all(np.isfinite(v) for row in scores for v in row):
        raise HTTPException(status_code=422, detail="Scores must be finite numbers (no missing values).")
    lengths = [len(row) for row in scores]
    if len(set(lengths)) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"All persons must have the same number of scores. Got lengths: {set(lengths)}.",
        )

    n_conditions = lengths[0]
    if n_conditions < min_conditions:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_conditions} conditions, got {n_conditions}.",
        )

    return n_persons, n_conditions


def _anova_p_x_i(X: np.ndarray) -> dict:
    """Compute ANOVA decomposition for a fully crossed p x i design.

    Returns sum of squares, degrees of freedom, mean squares, and
    estimated variance components via expected mean squares.
    """
    n_p, n_i = X.shape
    grand_mean = np.mean(X)

    # Person means and item means
    person_means = np.mean(X, axis=1)
    item_means = np.mean(X, axis=0)

    # Sum of squares
    ss_persons = n_i * np.sum((person_means - grand_mean) ** 2)
    ss_items = n_p * np.sum((item_means - grand_mean) ** 2)
    ss_residual = np.sum((X - person_means[:, None] - item_means[None, :] + grand_mean) ** 2)
    ss_total = np.sum((X - grand_mean) ** 2)

    # Degrees of freedom
    df_persons = n_p - 1
    df_items = n_i - 1
    df_residual = (n_p - 1) * (n_i - 1)

    # Mean squares
    ms_persons = ss_persons / df_persons if df_persons > 0 else 0.0
    ms_items = ss_items / df_items if df_items > 0 else 0.0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0.0

    # Variance components via EMS
    # E(MS_p) = sigma2_pi + n_i * sigma2_p
    # E(MS_i) = sigma2_pi + n_p * sigma2_i
    # E(MS_pi) = sigma2_pi
    raw = {
        "person": (ms_persons - ms_residual) / n_i,
        "item": (ms_items - ms_residual) / n_p,
        "person_x_item": ms_residual,
    }
    comp = {k: max(0.0, v) for k, v in raw.items()}
    sigma2_p, sigma2_i, sigma2_pi = comp["person"], comp["item"], comp["person_x_item"]

    total_var = sigma2_p + sigma2_i + sigma2_pi

    return {
        "anova_table": {
            "persons": {"ss": float(ss_persons), "df": df_persons, "ms": float(ms_persons)},
            "items": {"ss": float(ss_items), "df": df_items, "ms": float(ms_items)},
            "residual_pi": {"ss": float(ss_residual), "df": df_residual, "ms": float(ms_residual)},
            "total": {"ss": float(ss_total), "df": n_p * n_i - 1},
        },
        "variance_components": {k: float(v) for k, v in comp.items()},
        "variance_components_raw": {k: float(v) for k, v in raw.items()},
        "negative_components_set_to_zero": [k for k, v in raw.items() if v < 0],
        "percent_variance": {k: (float(v / total_var * 100) if total_var > 0 else 0.0) for k, v in comp.items()},
        "total_variance": float(total_var),
        "n_persons": n_p,
        "n_items": n_i,
    }


def _anova_p_x_i_x_o(X_3d: np.ndarray) -> dict:
    """Compute ANOVA decomposition for a fully crossed p x i x o design.

    X_3d has shape (n_persons, n_items, n_occasions).
    Returns variance components for all main effects and interactions.
    """
    n_p, n_i, n_o = X_3d.shape
    grand_mean = np.mean(X_3d)

    # Marginal means
    mean_p = np.mean(X_3d, axis=(1, 2))  # person means
    mean_i = np.mean(X_3d, axis=(0, 2))  # item means
    mean_o = np.mean(X_3d, axis=(0, 1))  # occasion means
    mean_pi = np.mean(X_3d, axis=2)       # person x item means
    mean_po = np.mean(X_3d, axis=1)       # person x occasion means
    mean_io = np.mean(X_3d, axis=0)       # item x occasion means

    # Sum of squares
    ss_p = n_i * n_o * np.sum((mean_p - grand_mean) ** 2)
    ss_i = n_p * n_o * np.sum((mean_i - grand_mean) ** 2)
    ss_o = n_p * n_i * np.sum((mean_o - grand_mean) ** 2)

    ss_pi = n_o * np.sum((mean_pi - mean_p[:, None] - mean_i[None, :] + grand_mean) ** 2)
    ss_po = n_i * np.sum((mean_po - mean_p[:, None] - mean_o[None, :] + grand_mean) ** 2)
    ss_io = n_p * np.sum((mean_io - mean_i[:, None] - mean_o[None, :] + grand_mean) ** 2)

    ss_total = np.sum((X_3d - grand_mean) ** 2)
    ss_residual = ss_total - ss_p - ss_i - ss_o - ss_pi - ss_po - ss_io

    # Degrees of freedom
    df_p = n_p - 1
    df_i = n_i - 1
    df_o = n_o - 1
    df_pi = (n_p - 1) * (n_i - 1)
    df_po = (n_p - 1) * (n_o - 1)
    df_io = (n_i - 1) * (n_o - 1)
    df_res = (n_p - 1) * (n_i - 1) * (n_o - 1)

    # Mean squares
    ms_p = ss_p / df_p if df_p > 0 else 0.0
    ms_i = ss_i / df_i if df_i > 0 else 0.0
    ms_o = ss_o / df_o if df_o > 0 else 0.0
    ms_pi = ss_pi / df_pi if df_pi > 0 else 0.0
    ms_po = ss_po / df_po if df_po > 0 else 0.0
    ms_io = ss_io / df_io if df_io > 0 else 0.0
    ms_res = ss_residual / df_res if df_res > 0 else 0.0

    # Variance components via EMS for fully crossed design
    raw = {
        "person": (ms_p - ms_pi - ms_po + ms_res) / (n_i * n_o),
        "item": (ms_i - ms_pi - ms_io + ms_res) / (n_p * n_o),
        "occasion": (ms_o - ms_po - ms_io + ms_res) / (n_p * n_i),
        "person_x_item": (ms_pi - ms_res) / n_o,
        "person_x_occasion": (ms_po - ms_res) / n_i,
        "item_x_occasion": (ms_io - ms_res) / n_p,
        "residual": ms_res,
    }
    components = {k: max(0.0, v) for k, v in raw.items()}

    total_var = sum(components.values())

    anova_table = {
        "persons": {"ss": float(ss_p), "df": df_p, "ms": float(ms_p)},
        "items": {"ss": float(ss_i), "df": df_i, "ms": float(ms_i)},
        "occasions": {"ss": float(ss_o), "df": df_o, "ms": float(ms_o)},
        "person_x_item": {"ss": float(ss_pi), "df": df_pi, "ms": float(ms_pi)},
        "person_x_occasion": {"ss": float(ss_po), "df": df_po, "ms": float(ms_po)},
        "item_x_occasion": {"ss": float(ss_io), "df": df_io, "ms": float(ms_io)},
        "residual": {"ss": float(ss_residual), "df": df_res, "ms": float(ms_res)},
        "total": {"ss": float(ss_total), "df": n_p * n_i * n_o - 1},
    }

    percent_var = {k: (float(v / total_var * 100) if total_var > 0 else 0.0) for k, v in components.items()}

    return {
        "anova_table": anova_table,
        "variance_components": {k: float(v) for k, v in components.items()},
        "variance_components_raw": {k: float(v) for k, v in raw.items()},
        "negative_components_set_to_zero": [k for k, v in raw.items() if v < 0],
        "percent_variance": percent_var,
        "total_variance": float(total_var),
        "n_persons": n_p,
        "n_items": n_i,
        "n_occasions": n_o,
    }


def _compute_g_coefficients_p_x_i(sigma2_p: float, sigma2_i: float,
                                    sigma2_pi: float, n_i: int) -> dict:
    """Compute generalizability and dependability coefficients for p x i design."""
    # Relative error variance (for ranking persons)
    sigma2_rel = sigma2_pi / n_i

    # Absolute error variance (for interpreting scores on a scale)
    sigma2_abs = (sigma2_i + sigma2_pi) / n_i

    # Generalizability coefficient (Ep^2) — for relative decisions
    g_coeff = sigma2_p / (sigma2_p + sigma2_rel) if (sigma2_p + sigma2_rel) > 0 else 0.0

    # Dependability coefficient (Phi) — for absolute decisions
    phi_coeff = sigma2_p / (sigma2_p + sigma2_abs) if (sigma2_p + sigma2_abs) > 0 else 0.0

    # SEM
    sem_relative = float(np.sqrt(sigma2_rel))
    sem_absolute = float(np.sqrt(sigma2_abs))

    return {
        "relative_error_variance": float(sigma2_rel),
        "absolute_error_variance": float(sigma2_abs),
        "generalizability_coefficient": float(g_coeff),
        "dependability_coefficient": float(phi_coeff),
        "sem_relative": sem_relative,
        "sem_absolute": sem_absolute,
    }


def _compute_g_coefficients_p_x_i_x_o(components: dict,
                                        n_i: int, n_o: int) -> dict:
    """Compute generalizability and dependability coefficients for p x i x o design."""
    sigma2_p = components["person"]
    sigma2_i = components["item"]
    sigma2_o = components["occasion"]
    sigma2_pi = components["person_x_item"]
    sigma2_po = components["person_x_occasion"]
    sigma2_io = components["item_x_occasion"]
    sigma2_pio = components["residual"]

    # Relative error variance
    sigma2_rel = sigma2_pi / n_i + sigma2_po / n_o + sigma2_pio / (n_i * n_o)

    # Absolute error variance
    sigma2_abs = (sigma2_i / n_i + sigma2_o / n_o + sigma2_pi / n_i +
                  sigma2_po / n_o + sigma2_io / (n_i * n_o) + sigma2_pio / (n_i * n_o))

    g_coeff = sigma2_p / (sigma2_p + sigma2_rel) if (sigma2_p + sigma2_rel) > 0 else 0.0
    phi_coeff = sigma2_p / (sigma2_p + sigma2_abs) if (sigma2_p + sigma2_abs) > 0 else 0.0

    sem_relative = float(np.sqrt(sigma2_rel))
    sem_absolute = float(np.sqrt(sigma2_abs))

    return {
        "relative_error_variance": float(sigma2_rel),
        "absolute_error_variance": float(sigma2_abs),
        "generalizability_coefficient": float(g_coeff),
        "dependability_coefficient": float(phi_coeff),
        "sem_relative": sem_relative,
        "sem_absolute": sem_absolute,
    }


# ── 1. G-Study ────────────────────────────────────────────────────────

@router.post("/g-study")
def g_study(req: GStudyRequest):
    """Estimate variance components via expected mean squares (EMS).

    Supports fully crossed p x i and p x i x o designs. Decomposes total
    variance into person, facet, and interaction components.
    """
    if req.design not in ("p_x_i", "p_x_i_x_o"):
        raise HTTPException(status_code=422, detail="design must be 'p_x_i' or 'p_x_i_x_o'.")

    if req.design == "p_x_i":
        n_persons, n_conditions = _validate_score_matrix(req.scores)
        X = np.array(req.scores)
        result = _anova_p_x_i(X)

        # Compute coefficients for current design
        vc = result["variance_components"]
        coeffs = _compute_g_coefficients_p_x_i(
            vc["person"], vc["item"], vc["person_x_item"], n_conditions
        )
        result["coefficients"] = coeffs

        facet_label = (req.facet_names[0] if req.facet_names and len(req.facet_names) >= 1
                       else "items")

        result["interpretation"] = (
            f"G-study for {n_persons} persons x {n_conditions} {facet_label} (crossed design). "
            f"Person variance = {vc['person']:.4f} ({result['percent_variance']['person']:.1f}%), "
            f"Item variance = {vc['item']:.4f} ({result['percent_variance']['item']:.1f}%), "
            f"Residual (p x i) = {vc['person_x_item']:.4f} ({result['percent_variance']['person_x_item']:.1f}%). "
            f"Ep^2 = {coeffs['generalizability_coefficient']:.3f}, "
            f"Phi = {coeffs['dependability_coefficient']:.3f}."
        )

    else:  # p_x_i_x_o
        fs = req.facet_structure or {}
        n_occasions = fs.get("n_occasions")
        n_cols = len(req.scores[0]) if req.scores else 0
        n_items = fs.get("n_items") or (n_cols // n_occasions if n_occasions else None)
        if not n_occasions or not n_items or n_items * n_occasions != n_cols:
            raise HTTPException(
                status_code=422,
                detail="For p x i x o, give the number of occasions (facet_structure.n_occasions). Columns must be "
                       "ordered occasion by occasion: all items at occasion 1, then all items at occasion 2, ...",
            )

        n_persons = len(req.scores)
        expected_cols = n_items * n_occasions
        for row in req.scores:
            if len(row) != expected_cols:
                raise HTTPException(
                    status_code=422,
                    detail=f"Each person must have {expected_cols} scores "
                           f"({n_items} items x {n_occasions} occasions). Got {len(row)}.",
                )

        _validate_score_matrix(req.scores)
        if n_persons < 3:
            raise HTTPException(status_code=422, detail="Need at least 3 persons.")
        if n_items < 2:
            raise HTTPException(status_code=422, detail="Need at least 2 items.")
        if n_occasions < 2:
            raise HTTPException(status_code=422, detail="Need at least 2 occasions.")

        # Reshape: scores are in item-major order within each occasion
        X_flat = np.array(req.scores)  # (n_persons, n_items * n_occasions)
        X_3d = X_flat.reshape(n_persons, n_occasions, n_items).transpose(0, 2, 1)
        # Now shape is (n_persons, n_items, n_occasions)

        result = _anova_p_x_i_x_o(X_3d)

        vc = result["variance_components"]
        coeffs = _compute_g_coefficients_p_x_i_x_o(
            {k: float(v) for k, v in vc.items()}, n_items, n_occasions
        )
        result["coefficients"] = coeffs

        result["interpretation"] = (
            f"G-study for {n_persons} persons x {n_items} items x {n_occasions} occasions "
            f"(fully crossed design). "
            f"Person variance = {vc['person']:.4f} ({result['percent_variance']['person']:.1f}%). "
            f"Largest facet contribution: "
            + max(
                [(k, v) for k, v in result["percent_variance"].items() if k != "person"],
                key=lambda x: x[1],
            ).__repr__()
            + f". Ep^2 = {coeffs['generalizability_coefficient']:.3f}, "
            f"Phi = {coeffs['dependability_coefficient']:.3f}."
        )

    return result


# ── 2. D-Study ────────────────────────────────────────────────────────

@router.post("/d-study")
def d_study(req: DStudyRequest):
    """Optimize measurement design using variance components from G-study.

    Computes generalizability (Ep^2) and dependability (Phi) coefficients
    across different numbers of conditions per facet.
    """
    if req.design not in ("p_x_i", "p_x_i_x_o"):
        raise HTTPException(status_code=422, detail="design must be 'p_x_i' or 'p_x_i_x_o'.")

    vc = req.variance_components

    if req.design == "p_x_i":
        required_keys = {"person", "item", "person_x_item"}
        if not required_keys.issubset(vc.keys()):
            raise HTTPException(
                status_code=422,
                detail=f"variance_components must include: {required_keys}. Got: {set(vc.keys())}.",
            )

        if "n_items" not in req.proposed_counts:
            raise HTTPException(
                status_code=422,
                detail="proposed_counts must include 'n_items' for p_x_i design.",
            )

        n_items_list = req.proposed_counts["n_items"]
        if not isinstance(n_items_list, list) or len(n_items_list) == 0:
            raise HTTPException(status_code=422, detail="n_items must be a non-empty list of integers.")

        table = []
        for ni in n_items_list:
            if ni < 1:
                continue
            coeffs = _compute_g_coefficients_p_x_i(
                vc["person"], vc["item"], vc["person_x_item"], ni
            )
            table.append({
                "n_items": ni,
                **coeffs,
            })

        # Find minimum items for Ep^2 >= 0.80
        min_for_80 = None
        for row in table:
            if row["generalizability_coefficient"] >= 0.80:
                min_for_80 = row["n_items"]
                break

        return {
            "design": req.design,
            "variance_components": vc,
            "d_study_table": table,
            "minimum_items_for_ep2_80": min_for_80,
            "interpretation": (
                f"D-study for p x i design across {len(n_items_list)} item counts. "
                + (f"Minimum {min_for_80} items needed for Ep^2 >= 0.80. "
                   if min_for_80 else
                   "Ep^2 >= 0.80 not achievable within the proposed range. ")
                + f"Current Ep^2 range: [{table[0]['generalizability_coefficient']:.3f}, "
                  f"{table[-1]['generalizability_coefficient']:.3f}]."
            ),
        }

    else:  # p_x_i_x_o
        required_keys = {"person", "item", "occasion", "person_x_item",
                         "person_x_occasion", "item_x_occasion", "residual"}
        if not required_keys.issubset(vc.keys()):
            raise HTTPException(
                status_code=422,
                detail=f"variance_components must include: {required_keys}.",
            )

        n_items_list = req.proposed_counts.get("n_items", [])
        n_occasions_list = req.proposed_counts.get("n_occasions", [])

        if not n_items_list or not n_occasions_list:
            raise HTTPException(
                status_code=422,
                detail="proposed_counts must include 'n_items' and 'n_occasions' for p_x_i_x_o.",
            )

        table = []
        for ni in n_items_list:
            for no in n_occasions_list:
                if ni < 1 or no < 1:
                    continue
                coeffs = _compute_g_coefficients_p_x_i_x_o(vc, ni, no)
                table.append({
                    "n_items": ni,
                    "n_occasions": no,
                    **coeffs,
                })

        # Find optimal configuration for Ep^2 >= 0.80
        optimal = None
        for row in sorted(table, key=lambda r: r["n_items"] * r["n_occasions"]):
            if row["generalizability_coefficient"] >= 0.80:
                optimal = {"n_items": row["n_items"], "n_occasions": row["n_occasions"]}
                break

        return {
            "design": req.design,
            "variance_components": vc,
            "d_study_table": table,
            "optimal_for_ep2_80": optimal,
            "interpretation": (
                f"D-study for p x i x o design across {len(table)} configurations. "
                + (f"Most efficient for Ep^2 >= 0.80: {optimal['n_items']} items x {optimal['n_occasions']} occasions. "
                   if optimal else
                   "Ep^2 >= 0.80 not achievable within the proposed range. ")
                + f"Ep^2 range: [{min(r['generalizability_coefficient'] for r in table):.3f}, "
                  f"{max(r['generalizability_coefficient'] for r in table):.3f}]."
            ),
        }


# ── 3. Universe Score Estimation ──────────────────────────────────────

@router.post("/universeScore")
def universe_score(req: UniverseScoreRequest):
    """Estimate universe scores and confidence intervals for each person.

    The universe score is the expected value of a person's scores across
    all admissible conditions in the universe of generalization.
    """
    if req.design not in ("p_x_i",):
        raise HTTPException(
            status_code=422,
            detail="Universe score estimation currently supports 'p_x_i' design only.",
        )

    n_persons, n_items = _validate_score_matrix(req.scores)
    X = np.array(req.scores)

    # Get variance components
    vc_result = _anova_p_x_i(X)
    vc = vc_result["variance_components"]

    # Override with provided variance components if available
    sigma2_p = req.variance_components.get("person", vc["person"])
    sigma2_i = req.variance_components.get("item", vc["item"])
    sigma2_pi = req.variance_components.get("person_x_item", vc["person_x_item"])

    # Grand mean and person means
    grand_mean = float(np.mean(X))
    person_means = np.mean(X, axis=1)

    # Reliability-like shrinkage factor
    # Universe score estimate = grand_mean + reliability * (person_mean - grand_mean)
    sigma2_error = sigma2_pi / n_items
    reliability = sigma2_p / (sigma2_p + sigma2_error) if (sigma2_p + sigma2_error) > 0 else 0.0

    # Universe score estimates (Kelley's regressed scores)
    universe_scores = grand_mean + reliability * (person_means - grand_mean)

    # SEM for universe score
    sem = float(np.sqrt(sigma2_p * (1 - reliability))) if reliability < 1 else 0.0

    # Confidence intervals
    alpha = 1 - req.confidence
    z_crit = float(sp_stats.norm.ppf(1 - alpha / 2))

    person_results = []
    for i in range(n_persons):
        ci_lower = float(universe_scores[i] - z_crit * sem)
        ci_upper = float(universe_scores[i] + z_crit * sem)
        person_results.append({
            "person": i + 1,
            "observed_mean": round(float(person_means[i]), 4),
            "universe_score": round(float(universe_scores[i]), 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "shrinkage": round(float(person_means[i] - universe_scores[i]), 4),
        })

    # Coefficients
    coeffs = _compute_g_coefficients_p_x_i(sigma2_p, sigma2_i, sigma2_pi, n_items)

    return {
        "n_persons": n_persons,
        "n_items": n_items,
        "grand_mean": round(grand_mean, 4),
        "reliability": round(reliability, 4),
        "sem": round(sem, 4),
        "confidence": req.confidence,
        "coefficients": coeffs,
        "person_scores": person_results,
        "interpretation": (
            f"Universe scores estimated for {n_persons} persons across {n_items} items. "
            f"Shrinkage factor (reliability) = {reliability:.3f}. "
            f"Scores regressed toward grand mean ({grand_mean:.2f}) by factor of "
            f"{1 - reliability:.3f}. SEM = {sem:.3f}. "
            f"Persons with extreme observed means show the most shrinkage toward the grand mean."
        ),
    }
