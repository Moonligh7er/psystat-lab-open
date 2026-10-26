"""Correspondence Analysis — simple CA, multiple CA (MCA), and supplementary
point projection for categorical data analysis.

Implements correspondence analysis via SVD on standardized residuals,
multiple correspondence analysis on indicator/Burt matrices, and
projection of supplementary rows/columns. Uses only numpy, scipy,
and standard library.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class SimpleCARequest(BaseModel):
    """Simple Correspondence Analysis on a contingency table."""
    table: list[list[float]]  # r x c contingency table (counts)
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None
    n_dimensions: int = 2


class MCARequest(BaseModel):
    """Multiple Correspondence Analysis."""
    data: list[list[str | int]]  # n x p matrix of categorical responses
    variable_names: list[str] | None = None
    n_dimensions: int = 2
    use_burt: bool = False  # if True, compute from Burt matrix


class SupplementaryRequest(BaseModel):
    """Project supplementary rows/columns onto existing CA space."""
    original_table: list[list[float]]  # original contingency table
    supplementary_rows: list[list[float]] | None = None  # new rows to project
    supplementary_cols: list[list[float]] | None = None  # new columns to project
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None
    sup_row_labels: list[str] | None = None
    sup_col_labels: list[str] | None = None
    n_dimensions: int = 2


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/simple")
def simple_ca(req: SimpleCARequest):
    """Simple Correspondence Analysis on a two-way contingency table.

    Decomposes the association between row and column categories via SVD
    on standardized residuals. Returns coordinates for biplot display,
    inertia decomposition, and chi-square contributions.
    """
    N = np.array(req.table, dtype=float)
    if N.ndim != 2:
        raise HTTPException(422, "Table must be 2-dimensional.")
    r, c = N.shape
    if r < 2 or c < 2:
        raise HTTPException(422, "Table must have at least 2 rows and 2 columns.")
    if not np.all(np.isfinite(N)) or np.any(N < 0):
        raise HTTPException(422, "All table values must be non-negative numbers.")

    grand_total = N.sum()
    if grand_total == 0:
        raise HTTPException(422, "Table is all zeros.")

    n_dim = min(req.n_dimensions, min(r, c) - 1)
    if n_dim < 1:
        n_dim = 1

    row_labels = req.row_labels or [f"Row{i+1}" for i in range(r)]
    col_labels = req.col_labels or [f"Col{j+1}" for j in range(c)]

    if len(row_labels) != r:
        raise HTTPException(422, f"row_labels length ({len(row_labels)}) must match table rows ({r}).")
    if len(col_labels) != c:
        raise HTTPException(422, f"col_labels length ({len(col_labels)}) must match table columns ({c}).")

    # Correspondence matrix (proportions)
    P = N / grand_total
    row_masses = P.sum(axis=1)  # r_i
    col_masses = P.sum(axis=0)  # c_j

    # Check for zero margins
    if np.any(row_masses == 0) or np.any(col_masses == 0):
        raise HTTPException(422, "Table has rows or columns that sum to zero. Remove them before analysis.")

    # Row and column profiles
    row_profiles = N / N.sum(axis=1, keepdims=True)
    col_profiles = N / N.sum(axis=0, keepdims=True)

    # Standardized residuals matrix
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(row_masses))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(col_masses))
    S = Dr_inv_sqrt @ (P - np.outer(row_masses, col_masses)) @ Dc_inv_sqrt

    # SVD
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)

    # Total inertia = sum of squared singular values
    total_inertia = float(np.sum(sigma ** 2))
    chi_square = total_inertia * grand_total

    # Eigenvalues (squared singular values)
    eigenvalues = sigma ** 2
    pct_explained = eigenvalues / total_inertia * 100 if total_inertia > 0 else np.zeros_like(eigenvalues)

    # Principal coordinates (standard coordinates * singular values)
    # Row principal coordinates: Dr^{-1/2} U Sigma
    row_coords = Dr_inv_sqrt @ U[:, :n_dim] * sigma[:n_dim]
    # Column principal coordinates: Dc^{-1/2} V Sigma
    col_coords = Dc_inv_sqrt @ Vt[:n_dim].T * sigma[:n_dim]

    # Standard coordinates (for asymmetric biplot)
    row_std_coords = Dr_inv_sqrt @ U[:, :n_dim]
    col_std_coords = Dc_inv_sqrt @ Vt[:n_dim].T

    # Chi-square contributions (how much each cell contributes to chi-square)
    expected = np.outer(row_masses, col_masses) * grand_total
    chi_sq_contrib = np.where(expected > 0, (N - expected) ** 2 / expected, 0.0)

    # Row contributions to inertia of each dimension
    row_contributions = np.zeros((r, n_dim))
    for d in range(n_dim):
        if eigenvalues[d] > 0:
            row_contributions[:, d] = row_masses * row_coords[:, d] ** 2 / eigenvalues[d]

    # Column contributions to inertia of each dimension
    col_contributions = np.zeros((c, n_dim))
    for d in range(n_dim):
        if eigenvalues[d] > 0:
            col_contributions[:, d] = col_masses * col_coords[:, d] ** 2 / eigenvalues[d]

    # Quality of representation (cos2): share of each point's squared chi-square
    # distance to the centroid (over all dimensions) captured by each retained dimension
    k_all = int(np.sum(sigma > 1e-12))
    row_full = Dr_inv_sqrt @ U[:, :k_all] * sigma[:k_all]
    col_full = Dc_inv_sqrt @ Vt[:k_all].T * sigma[:k_all]
    row_dist2 = np.sum(row_full ** 2, axis=1)
    col_dist2 = np.sum(col_full ** 2, axis=1)
    row_cos2 = np.where(row_dist2[:, None] > 0, row_coords ** 2 / np.where(row_dist2 > 0, row_dist2, 1)[:, None], 0.0)
    col_cos2 = np.where(col_dist2[:, None] > 0, col_coords ** 2 / np.where(col_dist2 > 0, col_dist2, 1)[:, None], 0.0)

    # Build response
    rows = []
    for i in range(r):
        rows.append({
            "label": row_labels[i],
            "mass": float(row_masses[i]),
            "coordinates": row_coords[i].tolist(),
            "standard_coordinates": row_std_coords[i].tolist(),
            "contributions": row_contributions[i].tolist(),
            "cos2": row_cos2[i].tolist(),
            "profile": row_profiles[i].tolist(),
        })

    cols = []
    for j in range(c):
        cols.append({
            "label": col_labels[j],
            "mass": float(col_masses[j]),
            "coordinates": col_coords[j].tolist(),
            "standard_coordinates": col_std_coords[j].tolist(),
            "contributions": col_contributions[j].tolist(),
            "cos2": col_cos2[j].tolist(),
            "profile": col_profiles[:, j].tolist(),
        })

    dimensions = []
    for d in range(min(int(np.sum(eigenvalues > 1e-12)), n_dim + 2)):
        dimensions.append({
            "dimension": d + 1,
            "eigenvalue": float(eigenvalues[d]),
            "inertia": float(eigenvalues[d]),
            "percent_explained": float(pct_explained[d]),
            "cumulative_percent": float(np.sum(pct_explained[:d + 1])),
        })

    # Chi-square test
    df = (r - 1) * (c - 1)
    from scipy import stats as sp_stats
    chi_sq_p = float(sp_stats.chi2.sf(chi_square, df)) if df > 0 else 1.0

    return {
        "row_coordinates": rows,
        "column_coordinates": cols,
        "dimensions": dimensions,
        "total_inertia": float(total_inertia),
        "chi_square": float(chi_square),
        "chi_square_df": df,
        "chi_square_p_value": chi_sq_p,
        "coordinates_type": "principal (row and column); standard coordinates also given per point",
        "n_dimensions_returned": n_dim,
        "chi_square_contributions": chi_sq_contrib.tolist(),
        "interpretation": (
            f"Total inertia = {total_inertia:.4f} (chi² = {chi_square:.2f}, "
            f"df = {df}, p = {chi_sq_p:.4f}). "
            f"{'Significant association between row and column categories.' if chi_sq_p < 0.05 else 'No significant association detected.'} "
            f"Dimension 1 explains {pct_explained[0]:.1f}% of inertia"
            f"{f', Dimension 2 explains {pct_explained[1]:.1f}%' if n_dim >= 2 and len(pct_explained) > 1 else ''}. "
            f"Points close together on the biplot have similar profiles."
        ),
    }


@router.post("/multiple")
def multiple_ca(req: MCARequest):
    """Multiple Correspondence Analysis (MCA) for multiple categorical variables.

    Analyzes patterns of association among several categorical variables
    simultaneously. Computes via SVD on the indicator matrix.
    """
    data = req.data
    n = len(data)
    if n < 3:
        raise HTTPException(422, "Need at least 3 observations.")

    p = len(data[0])
    if p < 2:
        raise HTTPException(422, "Need at least 2 variables.")

    var_names = req.variable_names or [f"Var{j+1}" for j in range(p)]
    if len(var_names) != p:
        raise HTTPException(422, f"variable_names length ({len(var_names)}) must match number of variables ({p}).")

    if any(len(row) != p for row in data):
        raise HTTPException(422, "Every observation must have a value for each variable.")

    # Build indicator matrix
    categories_per_var = []
    cat_labels = []
    for j in range(p):
        cats = sorted(set(str(data[i][j]) for i in range(n)))
        categories_per_var.append(cats)
        for cat in cats:
            cat_labels.append(f"{var_names[j]}:{cat}")

    total_cats = len(cat_labels)
    Z = np.zeros((n, total_cats))
    col_idx = 0
    for j in range(p):
        cats = categories_per_var[j]
        for i in range(n):
            val = str(data[i][j])
            k = cats.index(val)
            Z[i, col_idx + k] = 1.0
        col_idx += len(cats)

    n_dim = min(req.n_dimensions, total_cats - p)
    if n_dim < 1:
        raise HTTPException(422, "Each variable needs at least 2 categories.")

    # Indicator-matrix CA (standard MCA). With J categories and p variables the total
    # inertia is (J - p) / p. Burt-matrix eigenvalues are the squares of these.
    grand_total = Z.sum()
    P_z = Z / grand_total
    row_masses = P_z.sum(axis=1)
    col_masses = P_z.sum(axis=0)
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(row_masses))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(col_masses))
    S = Dr_inv_sqrt @ (P_z - np.outer(row_masses, col_masses)) @ Dc_inv_sqrt
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    k_nt = min(total_cats - p, len(sigma))
    lam = sigma[:k_nt] ** 2                      # indicator eigenvalues
    total_inertia = float((total_cats - p) / p)

    if req.use_burt:
        eigenvalues = lam ** 2
        scale = sigma[:n_dim] ** 2
        total_reported = float(np.sum(eigenvalues))
    else:
        eigenvalues = lam
        scale = sigma[:n_dim]
        total_reported = total_inertia
    cat_coords = Dc_inv_sqrt @ Vt[:n_dim].T * scale   # principal coordinates
    obj_coords = Dr_inv_sqrt @ U[:, :n_dim] * sigma[:n_dim]

    # Benzecri-adjusted eigenvalues and Greenacre's adjusted percentages (as ca::mjca lambda="adjusted")
    adj = np.where(lam > 1 / p, ((p / (p - 1)) * (lam - 1 / p)) ** 2, 0.0) if p > 1 else np.zeros_like(lam)
    greenacre_total = (p / (p - 1)) * (np.sum(lam ** 2) - (total_cats - p) / p ** 2) if p > 1 else 0.0
    object_scores = [{"index": i, "coordinates": obj_coords[i].tolist()} for i in range(n)]

    # Category coordinates
    pct_explained = eigenvalues / total_reported * 100 if total_reported > 0 else np.zeros_like(eigenvalues)
    category_results = []
    for j in range(total_cats):
        category_results.append({
            "label": cat_labels[j],
            "coordinates": cat_coords[j, :n_dim].tolist(),
        })

    # Discrimination measures (squared correlation ratios of each variable with each dimension),
    # from indicator principal coordinates: sum_k (n_k / n) * coord_k^2
    ind_coords = Dc_inv_sqrt @ Vt[:n_dim].T * sigma[:n_dim]
    discrimination = {}
    col_idx = 0
    for j in range(p):
        n_cats = len(categories_per_var[j])
        var_masses = Z[:, col_idx:col_idx + n_cats].sum(axis=0) / n
        disc = (var_masses[:, None] * ind_coords[col_idx:col_idx + n_cats] ** 2).sum(axis=0)
        discrimination[var_names[j]] = [float(v) for v in disc]
        col_idx += n_cats

    dimensions = []
    for d in range(min(n_dim, len(eigenvalues))):
        dimensions.append({
            "dimension": d + 1,
            "eigenvalue": float(eigenvalues[d]),
            "inertia": float(eigenvalues[d]),
            "percent_explained": float(pct_explained[d]),
            "cumulative_percent": float(np.sum(pct_explained[:d + 1])),
            "adjusted_eigenvalue": float(adj[d]),
            "adjusted_percent": float(adj[d] / greenacre_total * 100) if greenacre_total > 0 else None,
        })

    return {
        "object_scores": object_scores,
        "category_coordinates": category_results,
        "dimensions": dimensions,
        "total_inertia": float(total_reported),
        "total_inertia_indicator": total_inertia,
        "discrimination_measures": discrimination,
        "n_observations": n,
        "n_variables": p,
        "n_categories": total_cats,
        "n_dimensions_returned": n_dim,
        "method": "burt" if req.use_burt else "indicator",
        "interpretation": (
            f"MCA extracted {n_dim} dimensions from {p} variables with {total_cats} total categories. "
            f"Total inertia = {total_inertia:.4f}. "
            f"Dimension 1 accounts for {pct_explained[0]:.1f}% of the (indicator) inertia; "
            f"indicator-matrix percentages understate fit, so the Benzécri/Greenacre adjusted percentage "
            f"({dimensions[0]['adjusted_percent'] or 0:.1f}% for dimension 1) is usually more informative. "
            f"Variables with high discrimination measures on a dimension are most important for that dimension. "
            f"Category points close together are associated with similar response patterns."
        ),
    }


@router.post("/supplementary")
def supplementary_projection(req: SupplementaryRequest):
    """Project supplementary rows/columns onto an existing CA solution.

    Supplementary points are positioned in the existing space without
    influencing the original solution. Useful for adding new categories
    or observations to an existing analysis.
    """
    N = np.array(req.original_table, dtype=float)
    if N.ndim != 2:
        raise HTTPException(422, "original_table must be 2-dimensional.")
    r, c = N.shape
    if r < 2 or c < 2:
        raise HTTPException(422, "Table must have at least 2 rows and 2 columns.")

    if req.supplementary_rows is None and req.supplementary_cols is None:
        raise HTTPException(422, "Provide supplementary_rows and/or supplementary_cols.")

    if not np.all(np.isfinite(N)) or np.any(N < 0):
        raise HTTPException(422, "All table values must be non-negative numbers.")
    grand_total = N.sum()
    if grand_total == 0:
        raise HTTPException(422, "Original table is all zeros.")

    n_dim = min(req.n_dimensions, min(r, c) - 1)
    if n_dim < 1:
        n_dim = 1

    row_labels = req.row_labels or [f"Row{i+1}" for i in range(r)]
    col_labels = req.col_labels or [f"Col{j+1}" for j in range(c)]

    # Recompute CA on original table
    P = N / grand_total
    row_masses = P.sum(axis=1)
    col_masses = P.sum(axis=0)

    if np.any(row_masses == 0) or np.any(col_masses == 0):
        raise HTTPException(422, "Original table has zero-sum rows or columns.")

    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(row_masses))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(col_masses))
    S = Dr_inv_sqrt @ (P - np.outer(row_masses, col_masses)) @ Dc_inv_sqrt

    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    eigenvalues = sigma ** 2
    total_inertia = float(np.sum(eigenvalues))

    # Column standard coordinates (for projecting supplementary rows)
    col_std_coords = Dc_inv_sqrt @ Vt[:n_dim].T  # c x n_dim

    # Row standard coordinates (for projecting supplementary columns)
    row_std_coords = Dr_inv_sqrt @ U[:, :n_dim]  # r x n_dim

    result = {
        "n_dimensions": n_dim,
        "total_inertia": float(total_inertia),
        "eigenvalues": eigenvalues[:n_dim].tolist(),
    }

    # Project supplementary rows
    if req.supplementary_rows is not None:
        sup_rows = np.array(req.supplementary_rows, dtype=float)
        if sup_rows.ndim == 1:
            sup_rows = sup_rows.reshape(1, -1)
        if sup_rows.shape[1] != c:
            raise HTTPException(
                422,
                f"Supplementary rows must have {c} columns to match original table."
            )

        n_sup_r = sup_rows.shape[0]
        if np.any(sup_rows < 0):
            raise HTTPException(422, "Supplementary rows must be non-negative counts.")
        sup_row_labels = req.sup_row_labels or [f"SupRow{i+1}" for i in range(n_sup_r)]
        if len(sup_row_labels) != n_sup_r:
            raise HTTPException(422, f"sup_row_labels must have {n_sup_r} entries.")

        sup_row_coords = []
        for i in range(n_sup_r):
            row_total = sup_rows[i].sum()
            if row_total == 0:
                coords = np.zeros(n_dim)
            else:
                profile = sup_rows[i] / row_total
                # Transition formula: principal coordinate = profile @ column standard coordinates
                coords = np.zeros(n_dim)
                for d in range(n_dim):
                    if sigma[d] > 0:
                        coords[d] = float(profile @ col_std_coords[:, d])

            sup_row_coords.append({
                "label": sup_row_labels[i],
                "coordinates": coords.tolist(),
                "profile": (sup_rows[i] / max(row_total, 1e-10)).tolist(),
            })

        result["supplementary_rows"] = sup_row_coords

    # Project supplementary columns
    if req.supplementary_cols is not None:
        sup_cols = np.array(req.supplementary_cols, dtype=float)
        if sup_cols.ndim == 1:
            sup_cols = sup_cols.reshape(-1, 1)
        if sup_cols.shape[0] != r:
            raise HTTPException(
                422,
                f"Supplementary columns must have {r} rows to match original table."
            )

        n_sup_c = sup_cols.shape[1]
        if np.any(sup_cols < 0):
            raise HTTPException(422, "Supplementary columns must be non-negative counts.")
        sup_col_labels = req.sup_col_labels or [f"SupCol{j+1}" for j in range(n_sup_c)]
        if len(sup_col_labels) != n_sup_c:
            raise HTTPException(422, f"sup_col_labels must have {n_sup_c} entries.")

        sup_col_coords = []
        for j in range(n_sup_c):
            col_total = sup_cols[:, j].sum()
            if col_total == 0:
                coords = np.zeros(n_dim)
            else:
                profile = sup_cols[:, j] / col_total
                coords = np.zeros(n_dim)
                for d in range(n_dim):
                    if sigma[d] > 0:
                        coords[d] = float(profile @ row_std_coords[:, d])

            sup_col_coords.append({
                "label": sup_col_labels[j],
                "coordinates": coords.tolist(),
                "profile": (sup_cols[:, j] / max(col_total, 1e-10)).tolist(),
            })

        result["supplementary_columns"] = sup_col_coords

    result["interpretation"] = (
        "Supplementary points are projected into the existing CA space without "
        "affecting the original solution. Their positions show where they fall "
        "relative to the active row and column categories. Points near active "
        "categories have similar profiles."
    )

    return result
