"""Advanced ANOVA — Mixed ANOVA, MANOVA, Discriminant Analysis, Repeated Measures."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.linalg import eig, inv

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class MixedAnovaRequest(BaseModel):
    """Mixed-design ANOVA (between × within)."""
    data: list[list[float]]          # rows=subjects, cols=within-subject conditions
    groups: list[int | str]          # between-subjects group for each subject
    condition_labels: list[str] | None = None
    group_labels: list[str] | None = None
    correction: str = "greenhouse-geisser"  # "greenhouse-geisser", "huynh-feldt", "none"


class ManovaRequest(BaseModel):
    """Multivariate ANOVA."""
    dependent_variables: list[list[float]]  # each inner list is one DV (all subjects)
    groups: list[int | str]                  # group membership
    dv_names: list[str] | None = None


class DiscriminantRequest(BaseModel):
    """Linear Discriminant Analysis."""
    predictors: list[list[float]]  # each inner list is one predictor variable
    groups: list[int | str]        # group membership
    predictor_names: list[str] | None = None
    classification: bool = True


class RepeatedMeasuresRequest(BaseModel):
    """Full repeated measures ANOVA with sphericity correction."""
    data: list[list[float]]           # rows=subjects, cols=conditions
    condition_labels: list[str] | None = None
    correction: str = "greenhouse-geisser"  # "greenhouse-geisser", "huynh-feldt", "none"


# ── Helper Functions ──────────────────────────────────────────────────

def _helmert(k):
    """Orthonormal Helmert contrasts (k x k-1)."""
    C = np.zeros((k, k - 1))
    for j in range(k - 1):
        C[:j + 1, j] = 1.0
        C[j + 1, j] = -(j + 1)
        C[:, j] /= np.sqrt(np.sum(C[:, j] ** 2))
    return C


def _contrast_cov(data: np.ndarray, groups=None):
    """Covariance of orthonormal within-subject contrasts.

    With between-subject groups the covariance is pooled within groups (error SSCP / (N - g)),
    as in R's mlm-based mauchly.test and afex.
    Returns (S, residual df, number of groups).
    """
    n, k = data.shape
    Y = data @ _helmert(k)
    if groups is None:
        return np.cov(Y, rowvar=False).reshape(k - 1, k - 1), n - 1, 1
    groups = np.asarray(groups)
    E = np.zeros((k - 1, k - 1))
    levels = list(dict.fromkeys(groups.tolist()))
    for g in levels:
        Yg = Y[groups == g]
        D = Yg - Yg.mean(axis=0)
        E += D.T @ D
    df = n - len(levels)
    return E / df, df, len(levels)


def _mauchly_test(data: np.ndarray, groups=None):
    """Mauchly's test of sphericity (W, chi-square with Box's correction, p) as R's mauchly.test."""
    n, k = data.shape
    if k < 3:
        return {"W": 1.0, "chi_square": 0.0, "df": 0, "p": 1.0}
    S, nu, _ = _contrast_cov(data, groups)
    p = k - 1
    W = np.linalg.det(S) / (np.trace(S) / p) ** p

    # Chi-square approximation with Box's second-order correction, as R's mauchly.test
    rho = 1 - (2 * p * p + p + 2) / (6 * p * nu)
    w2 = (p + 2) * (p - 1) * (p - 2) * (2 * p ** 3 + 6 * p ** 2 + 3 * p + 2) / (288 * (nu * p * rho) ** 2)
    chi_sq = -nu * rho * np.log(max(W, 1e-300))
    df_chi = p * (p + 1) // 2 - 1
    pr1 = sp_stats.chi2.sf(chi_sq, df_chi)
    pr2 = sp_stats.chi2.sf(chi_sq, df_chi + 4)
    return {"W": float(W), "chi_square": float(chi_sq), "df": int(df_chi), "p": float(pr1 + w2 * (pr2 - pr1))}


def _epsilon_corrections(data: np.ndarray, groups=None):
    """Greenhouse-Geisser and Huynh-Feldt epsilon (HF with Lecoutre's 1991 correction)."""
    n, k = data.shape
    if k < 3:
        return {"greenhouse_geisser": 1.0, "huynh_feldt": 1.0}
    S, nu, g = _contrast_cov(data, groups)
    p = k - 1
    gg = float(np.clip(np.trace(S) ** 2 / (p * np.trace(S @ S)), 1.0 / p, 1.0))
    # N = n subjects, g groups: [(N - g + 1) p e - 2] / [p (N - g - p e)]
    hf = ((n - g + 1) * p * gg - 2) / (p * (n - g - p * gg))
    return {"greenhouse_geisser": gg, "huynh_feldt": float(np.clip(hf, gg, 1.0))}


def _lda_scaling(vecs: np.ndarray, W: np.ndarray, df_within: int) -> np.ndarray:
    """Scale discriminant eigenvectors so each function has unit pooled within-group
    variance (a' S_w a = 1, S_w = W / df), as MASS::lda's `scaling`. Signs are arbitrary;
    each column is oriented so its largest-magnitude coefficient is positive."""
    S_w = W / df_within
    out = np.empty_like(vecs)
    for j in range(vecs.shape[1]):
        a = vecs[:, j]
        a = a / np.sqrt(a @ S_w @ a)
        if a[np.argmax(np.abs(a))] < 0:
            a = -a
        out[:, j] = a
    return out


def _partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    total = ss_effect + ss_error
    return float(ss_effect / total) if total > 0 else 0.0


# ── 1. Mixed ANOVA ───────────────────────────────────────────────────

@router.post("/mixed-anova")
def mixed_anova(req: MixedAnovaRequest):
    """Mixed-design ANOVA: one between-subjects × one within-subjects factor (Type III SS, as afex).

    Sphericity (Mauchly) and the GG/HF epsilons use the contrast covariance pooled within groups.
    Simple effects of the within factor use a separate RM-ANOVA (own error term) in each group.
    """
    try:
        data = np.array(req.data, dtype=float)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid data format.")

    groups = np.array(req.groups)
    n_subjects, k = data.shape

    if len(groups) != n_subjects:
        raise HTTPException(400, "Length of groups must match number of rows in data.")
    if n_subjects < 3:
        raise HTTPException(400, "Need at least 3 subjects.")
    if k < 2:
        raise HTTPException(400, "Need at least 2 within-subject conditions.")

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise HTTPException(400, "Need at least 2 between-subject groups.")

    cond_labels = req.condition_labels or [f"Cond_{i+1}" for i in range(k)]
    grp_labels = req.group_labels or [str(g) for g in unique_groups]

    if len(cond_labels) != k:
        raise HTTPException(400, "condition_labels length must match number of columns.")
    if len(grp_labels) != n_groups:
        raise HTTPException(400, f"group_labels must have {n_groups} entries (one per distinct group).")
    if any(len(np.where(groups == g)[0]) < 2 for g in unique_groups):
        raise HTTPException(400, "Each between-subjects group needs at least 2 subjects.")
    if req.correction.lower() not in ("greenhouse-geisser", "huynh-feldt", "none"):
        raise HTTPException(400, "correction must be greenhouse-geisser, huynh-feldt or none.")

    # Grand mean
    grand_mean = np.mean(data)

    # Group indices and sizes
    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    group_ns = {g: len(idx) for g, idx in group_indices.items()}
    N = n_subjects

    # ── Subject means (across conditions) ──
    subject_means = np.mean(data, axis=1)

    # ── Group means (across conditions) ──
    group_means_overall = {g: np.mean(data[idx]) for g, idx in group_indices.items()}

    # ── Condition means (across all subjects) ──
    condition_means = np.mean(data, axis=0)

    # ── Cell means ──
    cell_means = {}
    for gi, g in enumerate(unique_groups):
        for ci in range(k):
            cell_means[(grp_labels[gi], cond_labels[ci])] = float(np.mean(data[group_indices[g], ci]))

    # ── Between-subjects effect (Type III) ──
    ss_between = k * sum(
        group_ns[g] * (group_means_overall[g] - grand_mean) ** 2
        for g in unique_groups
    )
    df_between = n_groups - 1

    # SS subjects within groups (error for between)
    ss_subjects_within = k * sum(
        (subject_means[s] - group_means_overall[groups[s]]) ** 2
        for s in range(N)
    )
    df_subjects_within = N - n_groups

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_subjects_within = ss_subjects_within / df_subjects_within if df_subjects_within > 0 else 0
    f_between = ms_between / ms_subjects_within if ms_subjects_within > 0 else 0
    p_between = sp_stats.f.sf(f_between, df_between, df_subjects_within)

    # ── Within-subjects effect (Type III: unweighted mean of the group means) ──
    # For each orthonormal contrast c, mu_c = mean over groups of the group mean of
    # Y_c, with Var = sigma^2 * sum(1/n_g) / g^2; SS = sum_c mu_c^2 / (sum(1/n_g) / g^2).
    # Equals N * sum((condition_means - grand_mean)^2) when groups are equal in size.
    Yc = data @ _helmert(k)
    group_contrast_means = np.array([Yc[group_indices[g]].mean(axis=0) for g in unique_groups])
    harmonic_term = np.sum([1.0 / group_ns[g] for g in unique_groups]) / n_groups ** 2
    ss_within = float(np.sum(group_contrast_means.mean(axis=0) ** 2) / harmonic_term)
    df_within = k - 1

    # ── Interaction (group × condition) ──
    ss_interaction = 0.0
    for gi, g in enumerate(unique_groups):
        for ci in range(k):
            cell_m = np.mean(data[group_indices[g], ci])
            ss_interaction += group_ns[g] * (
                cell_m - group_means_overall[g] - condition_means[ci] + grand_mean
            ) ** 2
    df_interaction = (n_groups - 1) * (k - 1)

    # ── Within-subjects error ──
    ss_within_error = 0.0
    for s in range(N):
        g = groups[s]
        gi = np.where(unique_groups == g)[0][0]
        group_cond_means = np.array([np.mean(data[group_indices[g], ci]) for ci in range(k)])
        for ci in range(k):
            ss_within_error += (
                data[s, ci] - subject_means[s] - group_cond_means[ci] + group_means_overall[g]
            ) ** 2
    df_within_error = (N - n_groups) * (k - 1)

    ms_within = ss_within / df_within if df_within > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_within_error = ss_within_error / df_within_error if df_within_error > 0 else 0

    f_within = ms_within / ms_within_error if ms_within_error > 0 else 0
    p_within = sp_stats.f.sf(f_within, df_within, df_within_error)

    f_interaction = ms_interaction / ms_within_error if ms_within_error > 0 else 0
    p_interaction = sp_stats.f.sf(f_interaction, df_interaction, df_within_error)

    # ── Sphericity & epsilon (pooled within groups) ──
    sphericity = _mauchly_test(data, groups)
    epsilons = _epsilon_corrections(data, groups)

    # ── Corrected results ──
    correction = req.correction.lower()
    corrected_within = None
    corrected_interaction = None
    if correction != "none" and k >= 3:
        eps = epsilons["greenhouse_geisser"] if correction == "greenhouse-geisser" else epsilons["huynh_feldt"]

        adj_df_within = df_within * eps
        adj_df_within_error_w = df_within_error * eps
        adj_p_within = sp_stats.f.sf(f_within, adj_df_within, adj_df_within_error_w)

        adj_df_interaction = df_interaction * eps
        adj_df_within_error_i = df_within_error * eps
        adj_p_interaction = sp_stats.f.sf(f_interaction, adj_df_interaction, adj_df_within_error_i)

        corrected_within = {
            "correction": correction,
            "epsilon": float(eps),
            "df": float(round(adj_df_within, 3)),
            "df_error": float(round(adj_df_within_error_w, 3)),
            "F": float(f_within),
            "p": float(adj_p_within),
        }
        corrected_interaction = {
            "correction": correction,
            "epsilon": float(eps),
            "df": float(round(adj_df_interaction, 3)),
            "df_error": float(round(adj_df_within_error_i, 3)),
            "F": float(f_interaction),
            "p": float(adj_p_interaction),
        }

    # ── Effect sizes ──
    eta_between = _partial_eta_squared(ss_between, ss_subjects_within)
    eta_within = _partial_eta_squared(ss_within, ss_within_error)
    eta_interaction = _partial_eta_squared(ss_interaction, ss_within_error)

    # ── Cell means table ──
    cell_means_table = []
    for gi, g in enumerate(unique_groups):
        for ci in range(k):
            vals = data[group_indices[g], ci]
            cell_means_table.append({
                "group": grp_labels[gi],
                "condition": cond_labels[ci],
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            })

    # ── Marginal means ──
    marginal_group = [
        {"group": grp_labels[gi], "mean": float(group_means_overall[g]),
         "sd": float(np.std(data[group_indices[g]], ddof=1)),
         "n": int(group_ns[g])}
        for gi, g in enumerate(unique_groups)
    ]
    marginal_condition = [
        {"condition": cond_labels[ci], "mean": float(condition_means[ci]),
         "sd": float(np.std(data[:, ci], ddof=1)),
         "n": int(N)}
        for ci in range(k)
    ]

    # ── Post-hoc: simple effects of within factor at each group level ──
    simple_effects = []
    for gi, g in enumerate(unique_groups):
        group_data = data[group_indices[g]]
        ng = group_data.shape[0]
        if ng < 2:
            continue

        # One-way RM ANOVA within this group
        group_cond_means = np.mean(group_data, axis=0)
        group_subj_means = np.mean(group_data, axis=1)
        group_grand = np.mean(group_data)

        ss_cond = ng * np.sum((group_cond_means - group_grand) ** 2)
        ss_error_rm = np.sum(
            (group_data - group_cond_means[None, :] - group_subj_means[:, None] + group_grand) ** 2
        )
        df_cond = k - 1
        df_err = (ng - 1) * (k - 1)

        ms_c = ss_cond / df_cond if df_cond > 0 else 0
        ms_e = ss_error_rm / df_err if df_err > 0 else 0
        f_val = ms_c / ms_e if ms_e > 0 else 0
        p_val = sp_stats.f.sf(f_val, df_cond, df_err) if df_cond > 0 and df_err > 0 else 1.0

        simple_effects.append({
            "group": grp_labels[gi],
            "SS": float(ss_cond),
            "df": int(df_cond),
            "MS": float(ms_c),
            "F": float(f_val),
            "p": float(p_val),
            "partial_eta_squared": _partial_eta_squared(ss_cond, ss_error_rm),
        })

    return {
        "between_subjects": {
            "SS": float(ss_between),
            "df": int(df_between),
            "df_error": int(df_subjects_within),
            "MS": float(ms_between),
            "F": float(f_between),
            "p": float(p_between),
            "partial_eta_squared": eta_between,
        },
        "within_subjects": {
            "SS": float(ss_within),
            "df": int(df_within),
            "df_error": int(df_within_error),
            "MS": float(ms_within),
            "F": float(f_within),
            "p": float(p_within),
            "partial_eta_squared": eta_within,
            "epsilon_gg": epsilons["greenhouse_geisser"],
            "epsilon_hf": epsilons["huynh_feldt"],
        },
        "interaction": {
            "SS": float(ss_interaction),
            "df": int(df_interaction),
            "df_error": int(df_within_error),
            "MS": float(ms_interaction),
            "F": float(f_interaction),
            "p": float(p_interaction),
            "partial_eta_squared": eta_interaction,
        },
        "corrected_within": corrected_within,
        "corrected_interaction": corrected_interaction,
        "sphericity_test": sphericity,
        "cell_means": cell_means_table,
        "marginal_means_group": marginal_group,
        "marginal_means_condition": marginal_condition,
        "simple_effects_within_at_group": simple_effects,
    }


# ── 2. MANOVA ────────────────────────────────────────────────────────

@router.post("/manova")
def manova(req: ManovaRequest):
    """Multivariate ANOVA with four test statistics and Box's M."""
    dvs = [np.array(dv, dtype=float) for dv in req.dependent_variables]
    groups = np.array(req.groups)
    n_total = len(groups)
    p = len(dvs)  # number of DVs

    if p < 2:
        raise HTTPException(400, "MANOVA requires at least 2 dependent variables.")
    for i, dv in enumerate(dvs):
        if len(dv) != n_total:
            raise HTTPException(400, f"DV {i} length ({len(dv)}) doesn't match groups length ({n_total}).")

    unique_groups = np.unique(groups)
    g = len(unique_groups)
    if g < 2:
        raise HTTPException(400, "Need at least 2 groups.")
    if n_total <= p + g:
        raise HTTPException(400, "Not enough observations for MANOVA.")

    dv_names = req.dv_names or [f"DV_{i+1}" for i in range(p)]
    if len(dv_names) != p:
        raise HTTPException(400, "dv_names length must match number of dependent variables.")

    # Build data matrix (n × p)
    Y = np.column_stack(dvs)
    grand_mean = np.mean(Y, axis=0)

    # Group indices
    group_idx = {grp: np.where(groups == grp)[0] for grp in unique_groups}
    group_means = {grp: np.mean(Y[idx], axis=0) for grp, idx in group_idx.items()}
    group_ns = {grp: len(idx) for grp, idx in group_idx.items()}

    # ── Hypothesis (H) and Error (E) SSCP matrices ──
    H = np.zeros((p, p))
    for grp in unique_groups:
        diff = (group_means[grp] - grand_mean).reshape(-1, 1)
        H += group_ns[grp] * (diff @ diff.T)

    E = np.zeros((p, p))
    for grp in unique_groups:
        Y_grp = Y[group_idx[grp]]
        centered = Y_grp - group_means[grp]
        E += centered.T @ centered

    # Total SSCP
    T_mat = H + E

    df_h = g - 1
    df_e = n_total - g

    try:
        E_inv = inv(E)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Error SSCP matrix is singular — check for multicollinearity or insufficient data.")

    # Eigenvalues of E_inv @ H
    HE_inv = E_inv @ H
    eigenvalues_complex = np.linalg.eigvals(HE_inv)
    eigenvalues = np.sort(np.real(eigenvalues_complex))[::-1]

    s = min(p, df_h)
    m = (abs(p - df_h) - 1) / 2
    n_param = (df_e - p - 1) / 2

    # ── Pillai's Trace ──
    pillai_V = float(np.sum(eigenvalues / (1 + eigenvalues)))
    # F-approx for Pillai
    pillai_df1 = s * max(p, df_h)  # This is a simplified approximation
    pillai_df2 = s * (df_e - p + s)
    if pillai_df2 > 0 and (s * (df_e - p + s) - pillai_V * s * max(p, df_h)) != 0:
        pillai_F = (pillai_V / s) * (pillai_df2 / max(p, df_h)) / ((1 - pillai_V / s) if (1 - pillai_V / s) > 0 else 1e-10)
    else:
        pillai_F = 0.0
    # More standard approximation
    pillai_df1 = p * df_h
    pillai_df2 = s * (df_e - p + s)
    if pillai_df2 > 0:
        pillai_F = (pillai_V / s) / ((1 - pillai_V / s) if s > 0 and pillai_V < s else 1e-10) * (pillai_df2 / pillai_df1) if pillai_df1 > 0 else 0.0
    else:
        pillai_F = 0.0
    pillai_p = sp_stats.f.sf(pillai_F, pillai_df1, pillai_df2) if pillai_df1 > 0 and pillai_df2 > 0 else 1.0

    # ── Wilks' Lambda ──
    wilks_lambda = float(np.prod(1 / (1 + eigenvalues)))
    # Rao's F approximation
    t_val = np.sqrt((p ** 2 * df_h ** 2 - 4) / (p ** 2 + df_h ** 2 - 5)) if (p ** 2 + df_h ** 2 - 5) > 0 else 1
    wilks_df1 = p * df_h
    wilks_df2 = (df_e - (p - df_h + 1) / 2) * t_val - (p * df_h - 2) / 2
    if wilks_lambda > 0 and t_val > 0:
        wilks_F = ((1 - wilks_lambda ** (1 / t_val)) / (wilks_lambda ** (1 / t_val))) * (wilks_df2 / wilks_df1) if wilks_df1 > 0 else 0.0
    else:
        wilks_F = 0.0
    wilks_p = sp_stats.f.sf(wilks_F, wilks_df1, max(wilks_df2, 1)) if wilks_df1 > 0 and wilks_df2 > 0 else 1.0

    # ── Hotelling-Lawley Trace ──
    hl_trace = float(np.sum(eigenvalues))
    hl_df1 = p * df_h
    hl_df2 = s * (df_e - p - 1) + 2
    b = (p + 2 * n_param) * (df_h + 2 * n_param) / (2 * (2 * n_param + 1) * (2 * n_param - 1)) if n_param > 0.5 else 1
    hl_F = (hl_trace / s) * (hl_df2 / hl_df1) if hl_df1 > 0 else 0.0
    hl_p = sp_stats.f.sf(hl_F, hl_df1, max(hl_df2, 1)) if hl_df1 > 0 and hl_df2 > 0 else 1.0

    # ── Roy's Largest Root ──
    roy_theta = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
    roy_df1 = max(p, df_h)
    roy_df2 = df_e - roy_df1 + df_h
    roy_F = roy_theta * (roy_df2 / roy_df1) if roy_df1 > 0 else 0.0
    roy_p = sp_stats.f.sf(roy_F, roy_df1, max(roy_df2, 1)) if roy_df1 > 0 and roy_df2 > 0 else 1.0

    multivariate_tests = {
        "pillai_trace": {"value": pillai_V, "F": float(pillai_F), "df1": float(pillai_df1), "df2": float(pillai_df2), "p": float(pillai_p)},
        "wilks_lambda": {"value": float(wilks_lambda), "F": float(wilks_F), "df1": float(wilks_df1), "df2": float(wilks_df2), "p": float(wilks_p)},
        "hotelling_lawley": {"value": float(hl_trace), "F": float(hl_F), "df1": float(hl_df1), "df2": float(hl_df2), "p": float(hl_p)},
        "roys_largest_root": {"value": float(roy_theta), "F": float(roy_F), "df1": float(roy_df1), "df2": float(roy_df2), "p": float(roy_p)},
    }

    # ── Univariate ANOVAs ──
    univariate = []
    for vi in range(p):
        y = Y[:, vi]
        gm = np.mean(y)
        ss_b = sum(group_ns[grp] * (np.mean(y[group_idx[grp]]) - gm) ** 2 for grp in unique_groups)
        ss_w = sum(np.sum((y[group_idx[grp]] - np.mean(y[group_idx[grp]])) ** 2) for grp in unique_groups)
        df_b = g - 1
        df_w = n_total - g
        ms_b = ss_b / df_b if df_b > 0 else 0
        ms_w = ss_w / df_w if df_w > 0 else 0
        f_val = ms_b / ms_w if ms_w > 0 else 0
        p_val = sp_stats.f.sf(f_val, df_b, df_w) if df_b > 0 and df_w > 0 else 1.0
        eta_sq = ss_b / (ss_b + ss_w) if (ss_b + ss_w) > 0 else 0

        univariate.append({
            "variable": dv_names[vi],
            "SS_between": float(ss_b),
            "SS_within": float(ss_w),
            "df_between": int(df_b),
            "df_within": int(df_w),
            "MS_between": float(ms_b),
            "MS_within": float(ms_w),
            "F": float(f_val),
            "p": float(p_val),
            "eta_squared": float(eta_sq),
        })

    # ── Descriptive statistics ──
    descriptives = []
    for vi in range(p):
        for grp in unique_groups:
            vals = Y[group_idx[grp], vi]
            descriptives.append({
                "variable": dv_names[vi],
                "group": str(grp),
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

    # ── Box's M test ──
    box_m_result = _box_m_test(Y, groups, unique_groups, group_idx, group_ns, p, g, n_total)

    # ── Discriminant function coefficients ──
    try:
        E_inv_H = E_inv @ H
        eig_vals, eig_vecs = np.linalg.eig(E_inv_H)
        order = np.argsort(np.real(eig_vals))[::-1]
        eig_vals = np.real(eig_vals[order])
        eig_vecs = np.real(eig_vecs[:, order])

        n_functions = min(g - 1, p)
        eig_vecs = _lda_scaling(eig_vecs[:, :n_functions], E, df_e)
        disc_coefficients = []
        for fi in range(n_functions):
            coeffs = {dv_names[vi]: float(eig_vecs[vi, fi]) for vi in range(p)}
            disc_coefficients.append({
                "function": fi + 1,
                "eigenvalue": float(eig_vals[fi]),
                "coefficients": coeffs,
            })
    except Exception:
        disc_coefficients = []

    return {
        "multivariate_tests": multivariate_tests,
        "univariate_anovas": univariate,
        "descriptives": descriptives,
        "box_m_test": box_m_result,
        "discriminant_coefficients": disc_coefficients,
    }


def _box_m_test(Y, groups, unique_groups, group_idx, group_ns, p, g, n_total):
    """Box's M test for equality of covariance matrices."""
    try:
        cov_matrices = {}
        for grp in unique_groups:
            Y_grp = Y[group_idx[grp]]
            if len(Y_grp) > 1:
                cov_matrices[grp] = np.cov(Y_grp, rowvar=False)
            else:
                return {"M": None, "F_approx": None, "p": None, "note": "Not enough data for Box's M test."}

        # Pooled covariance
        S_pooled = np.zeros((p, p))
        for grp in unique_groups:
            S_pooled += (group_ns[grp] - 1) * cov_matrices[grp]
        S_pooled /= (n_total - g)

        # M statistic
        M = 0.0
        for grp in unique_groups:
            n_i = group_ns[grp]
            if n_i > 1:
                det_Si = np.linalg.det(cov_matrices[grp])
                det_Sp = np.linalg.det(S_pooled)
                if det_Si > 0 and det_Sp > 0:
                    M += (n_i - 1) * (np.log(det_Sp) - np.log(det_Si))

        # Correction factor
        sum_inv = sum(1.0 / (group_ns[grp] - 1) for grp in unique_groups if group_ns[grp] > 1)
        c1 = (sum_inv - 1.0 / (n_total - g)) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (g - 1))

        df = p * (p + 1) * (g - 1) // 2
        chi_sq = M * (1 - c1)
        p_value = sp_stats.chi2.sf(chi_sq, df) if df > 0 else 1.0

        return {
            "M": float(M),
            "chi_square": float(chi_sq),
            "df": int(df),
            "p": float(p_value),
        }
    except Exception as e:
        return {"M": None, "F_approx": None, "p": None, "note": f"Box's M test failed: {str(e)}"}


# ── 3. Discriminant Analysis ─────────────────────────────────────────

@router.post("/discriminant")
def discriminant_analysis(req: DiscriminantRequest):
    """Linear discriminant analysis: canonical functions (scaled as MASS::lda), standardized
    coefficients, pooled within-groups structure matrix, Bartlett's tests, and classification
    with resubstitution and leave-one-out accuracy."""
    predictors = [np.array(pred, dtype=float) for pred in req.predictors]
    groups = np.array(req.groups)
    n_total = len(groups)
    p = len(predictors)

    if p < 1:
        raise HTTPException(400, "Need at least 1 predictor.")
    for i, pred in enumerate(predictors):
        if len(pred) != n_total:
            raise HTTPException(400, f"Predictor {i} length ({len(pred)}) doesn't match groups length ({n_total}).")

    unique_groups = np.unique(groups)
    g = len(unique_groups)
    if g < 2:
        raise HTTPException(400, "Need at least 2 groups.")
    if n_total <= p + g:
        raise HTTPException(400, "Not enough observations.")

    pred_names = req.predictor_names or [f"Pred_{i+1}" for i in range(p)]
    if len(pred_names) != p:
        raise HTTPException(400, "predictor_names length must match number of predictors.")

    X = np.column_stack(predictors)
    grand_mean = np.mean(X, axis=0)

    group_idx = {grp: np.where(groups == grp)[0] for grp in unique_groups}
    group_means = {grp: np.mean(X[idx], axis=0) for grp, idx in group_idx.items()}
    group_ns = {grp: len(idx) for grp, idx in group_idx.items()}

    # Prior probabilities
    priors = {str(grp): float(group_ns[grp] / n_total) for grp in unique_groups}

    # Between-groups and within-groups SSCP
    H = np.zeros((p, p))
    for grp in unique_groups:
        diff = (group_means[grp] - grand_mean).reshape(-1, 1)
        H += group_ns[grp] * (diff @ diff.T)

    W = np.zeros((p, p))
    for grp in unique_groups:
        X_grp = X[group_idx[grp]]
        centered = X_grp - group_means[grp]
        W += centered.T @ centered

    try:
        W_inv = inv(W)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Within-groups SSCP matrix is singular.")

    # Eigendecomposition of W_inv @ H
    eig_vals_complex, eig_vecs_complex = np.linalg.eig(W_inv @ H)
    eig_vals = np.real(eig_vals_complex)
    eig_vecs = np.real(eig_vecs_complex)

    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]

    n_functions = min(g - 1, p)
    eig_vals = eig_vals[:n_functions]
    eig_vecs = _lda_scaling(eig_vecs[:, :n_functions], W, n_total - g)

    # ── Discriminant functions ──
    total_eig = np.sum(eig_vals[eig_vals > 0])
    canonical_corrs = np.sqrt(eig_vals / (1 + eig_vals))

    functions = []
    for fi in range(n_functions):
        functions.append({
            "function": fi + 1,
            "eigenvalue": float(eig_vals[fi]),
            "variance_explained_pct": float(eig_vals[fi] / total_eig * 100) if total_eig > 0 else 0.0,
            "canonical_correlation": float(canonical_corrs[fi]),
        })

    # ── Unstandardized coefficients ──
    unstandardized = []
    for fi in range(n_functions):
        coeffs = {pred_names[vi]: float(eig_vecs[vi, fi]) for vi in range(p)}
        unstandardized.append({"function": fi + 1, "coefficients": coeffs})

    # ── Standardized coefficients ──
    # Standardize by within-group SD
    W_diag = np.diag(W) / (n_total - g)
    within_sds = np.sqrt(W_diag)

    standardized = []
    for fi in range(n_functions):
        coeffs = {pred_names[vi]: float(eig_vecs[vi, fi] * within_sds[vi]) for vi in range(p)}
        standardized.append({"function": fi + 1, "coefficients": coeffs})

    # ── Structure matrix: pooled within-groups correlations of predictors with functions ──
    # r_if = (S_w a_f)_i / sd_i, since a_f' S_w a_f = 1 after scaling
    S_w = W / (n_total - g)
    loadings = S_w @ eig_vecs / within_sds[:, None]
    structure_matrix = {
        pred_names[vi]: {f"Function_{fi+1}": float(loadings[vi, fi]) for fi in range(n_functions)}
        for vi in range(p)
    }

    # ── Wilks' Lambda for each function ──
    wilks_tests = []
    for fi in range(n_functions):
        # Wilks' Lambda for functions fi through end
        remaining_eig = eig_vals[fi:]
        wl = float(np.prod(1 / (1 + remaining_eig)))
        # Chi-square test
        df_chi = (p - fi) * (g - 1 - fi)
        chi_sq = -(n_total - 1 - (p + g) / 2) * np.log(max(wl, 1e-300))
        p_val = sp_stats.chi2.sf(chi_sq, df_chi) if df_chi > 0 else 1.0

        wilks_tests.append({
            "functions_tested": f"{fi+1} through {n_functions}",
            "wilks_lambda": float(wl),
            "chi_square": float(chi_sq),
            "df": int(df_chi),
            "p": float(p_val),
        })

    # ── Group centroids ──
    centroids = []
    for grp in unique_groups:
        scores = X[group_idx[grp]] @ eig_vecs
        centroid = {f"Function_{fi+1}": float(np.mean(scores[:, fi])) for fi in range(n_functions)}
        centroids.append({"group": str(grp), **centroid})

    # ── Classification ──
    classification_result = None
    if req.classification:
        classification_result = _classify_lda(
            X, groups, unique_groups, group_idx, group_means, group_ns, W, n_total, g, p
        )

    return {
        "discriminant_functions": functions,
        "unstandardized_coefficients": unstandardized,
        "standardized_coefficients": standardized,
        "structure_matrix": structure_matrix,
        "wilks_lambda_tests": wilks_tests,
        "group_centroids": centroids,
        "classification": classification_result,
        "prior_probabilities": priors,
    }


def _classify_lda(X, groups, unique_groups, group_idx, group_means, group_ns, W, n_total, g, p):
    """Classification with confusion matrix and LOO cross-validation."""
    # Pooled covariance
    S_pooled = W / (n_total - g)

    try:
        S_inv = inv(S_pooled)
    except np.linalg.LinAlgError:
        return {"error": "Cannot invert pooled covariance matrix."}

    priors = {grp: group_ns[grp] / n_total for grp in unique_groups}

    # ── Resubstitution classification ──
    predictions = []
    for i in range(n_total):
        x = X[i]
        best_grp = None
        best_score = -np.inf
        for grp in unique_groups:
            mu = group_means[grp]
            score = -0.5 * (x - mu) @ S_inv @ (x - mu) + np.log(priors[grp])
            if score > best_score:
                best_score = score
                best_grp = grp
        predictions.append(best_grp)

    predictions = np.array(predictions)

    # Confusion matrix
    confusion = {}
    for actual in unique_groups:
        row = {}
        for predicted in unique_groups:
            row[str(predicted)] = int(np.sum((groups == actual) & (predictions == predicted)))
        confusion[str(actual)] = row

    overall_accuracy = float(np.mean(predictions == groups))
    per_group_accuracy = {}
    for grp in unique_groups:
        mask = groups == grp
        per_group_accuracy[str(grp)] = float(np.mean(predictions[mask] == groups[mask])) if np.sum(mask) > 0 else 0.0

    # ── Leave-one-out cross-validation ──
    loo_correct = 0
    for i in range(n_total):
        X_loo = np.delete(X, i, axis=0)
        groups_loo = np.delete(groups, i)

        # Recompute group means and pooled covariance without observation i
        loo_group_idx = {grp: np.where(groups_loo == grp)[0] for grp in unique_groups}
        loo_group_means = {grp: np.mean(X_loo[idx], axis=0) if len(idx) > 0 else group_means[grp]
                          for grp, idx in loo_group_idx.items()}
        loo_group_ns = {grp: len(idx) for grp, idx in loo_group_idx.items()}

        W_loo = np.zeros((p, p))
        for grp in unique_groups:
            if loo_group_ns[grp] > 0:
                X_grp = X_loo[loo_group_idx[grp]]
                centered = X_grp - loo_group_means[grp]
                W_loo += centered.T @ centered

        n_loo = n_total - 1
        S_loo = W_loo / (n_loo - g)

        try:
            S_loo_inv = inv(S_loo)
        except np.linalg.LinAlgError:
            continue

        loo_priors = {grp: loo_group_ns[grp] / n_loo for grp in unique_groups}

        x = X[i]
        best_grp = None
        best_score = -np.inf
        for grp in unique_groups:
            mu = loo_group_means[grp]
            score = -0.5 * (x - mu) @ S_loo_inv @ (x - mu) + np.log(max(loo_priors[grp], 1e-300))
            if score > best_score:
                best_score = score
                best_grp = grp

        if best_grp == groups[i]:
            loo_correct += 1

    loo_accuracy = float(loo_correct / n_total)

    return {
        "confusion_matrix": confusion,
        "overall_accuracy": overall_accuracy,
        "per_group_accuracy": per_group_accuracy,
        "cross_validated_accuracy_loo": loo_accuracy,
        "n_classified": int(n_total),
    }


# ── 4. Repeated Measures ANOVA ───────────────────────────────────────

@router.post("/repeated-measures-anova")
def repeated_measures_anova(req: RepeatedMeasuresRequest):
    """Full repeated measures ANOVA with sphericity correction and pairwise comparisons."""
    try:
        data = np.array(req.data, dtype=float)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid data format.")

    n, k = data.shape

    if n < 3:
        raise HTTPException(400, "Need at least 3 subjects.")
    if k < 2:
        raise HTTPException(400, "Need at least 2 conditions.")

    cond_labels = req.condition_labels or [f"Cond_{i+1}" for i in range(k)]
    if len(cond_labels) != k:
        raise HTTPException(400, "condition_labels length must match number of columns.")

    grand_mean = np.mean(data)
    subject_means = np.mean(data, axis=1)
    condition_means = np.mean(data, axis=0)

    # ── SS decomposition ──
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_conditions = n * np.sum((condition_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_conditions

    df_conditions = k - 1
    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)

    ms_conditions = ss_conditions / df_conditions if df_conditions > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    f_val = ms_conditions / ms_error if ms_error > 0 else 0
    p_val = sp_stats.f.sf(f_val, df_conditions, df_error)

    partial_eta_sq = _partial_eta_squared(ss_conditions, ss_error)

    # ── Sphericity ──
    sphericity = _mauchly_test(data)
    epsilons = _epsilon_corrections(data)

    # ── Corrected results ──
    correction = req.correction.lower()
    corrected = None
    if correction != "none" and k >= 3:
        eps = epsilons["greenhouse_geisser"] if correction == "greenhouse-geisser" else epsilons["huynh_feldt"]
        adj_df_cond = df_conditions * eps
        adj_df_error = df_error * eps
        adj_p = sp_stats.f.sf(f_val, adj_df_cond, adj_df_error)

        corrected = {
            "correction": correction,
            "epsilon": float(eps),
            "df_conditions": float(round(adj_df_cond, 3)),
            "df_error": float(round(adj_df_error, 3)),
            "F": float(f_val),
            "p": float(adj_p),
        }

    # ── Descriptives per condition ──
    descriptives = []
    for ci in range(k):
        vals = data[:, ci]
        descriptives.append({
            "condition": cond_labels[ci],
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "se": float(np.std(vals, ddof=1) / np.sqrt(n)),
            "n": int(n),
        })

    # ── Pairwise comparisons (Bonferroni-corrected paired t-tests) ──
    n_comparisons = k * (k - 1) // 2
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = data[:, i] - data[:, j]
            t_stat, p_raw = sp_stats.ttest_rel(data[:, i], data[:, j])
            p_bonf = min(p_raw * n_comparisons, 1.0)

            mean_diff = float(np.mean(diff))
            se_diff = float(np.std(diff, ddof=1) / np.sqrt(n))
            # 95% CI
            t_crit = sp_stats.t.ppf(0.975, n - 1)
            ci_low = mean_diff - t_crit * se_diff
            ci_high = mean_diff + t_crit * se_diff

            # Cohen's d for paired
            cohens_d = mean_diff / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

            pairwise.append({
                "comparison": f"{cond_labels[i]} vs {cond_labels[j]}",
                "mean_difference": mean_diff,
                "se": se_diff,
                "t": float(t_stat),
                "df": int(n - 1),
                "p_uncorrected": float(p_raw),
                "p_bonferroni": float(p_bonf),
                "ci_95": [float(ci_low), float(ci_high)],
                "cohens_d": float(cohens_d),
            })

    return {
        "anova": {
            "SS_conditions": float(ss_conditions),
            "SS_error": float(ss_error),
            "SS_subjects": float(ss_subjects),
            "df_conditions": int(df_conditions),
            "df_error": int(df_error),
            "df_subjects": int(df_subjects),
            "MS_conditions": float(ms_conditions),
            "MS_error": float(ms_error),
            "F": float(f_val),
            "p": float(p_val),
            "partial_eta_squared": partial_eta_sq,
        },
        "sphericity_test": sphericity,
        "epsilon": epsilons,
        "corrected": corrected,
        "descriptives": descriptives,
        "pairwise_comparisons": pairwise,
    }
