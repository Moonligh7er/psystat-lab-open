"""Conjoint Analysis — choice-based conjoint via conditional logit, preference
prediction, and attribute importance analysis for decision research.

Implements conditional logit estimation via maximum likelihood (Newton-Raphson),
choice share prediction for new profiles, and attribute importance computation
with bootstrap confidence intervals. Uses only numpy and scipy.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class ConjointAnalyzeRequest(BaseModel):
    """Choice-based conjoint analysis via conditional logit."""
    profiles: list[list[dict]]  # [task][alternative] -> {attr: level, ...}
    choices: list[int]  # selected alternative index (0-based) per task
    respondent_ids: list[int] | None = None  # respondent ID per task (for clustered SE)
    attribute_names: list[str] | None = None


class PredictRequest(BaseModel):
    """Predict choice shares for new profiles."""
    profiles: list[dict]  # each profile: {attr: level, ...}
    utilities: dict  # {attr: {level: utility, ...}, ...} from /analyze


class ImportanceRequest(BaseModel):
    """Attribute importance analysis with bootstrap CI."""
    profiles: list[list[dict]]  # same as analyze
    choices: list[int]
    respondent_ids: list[int] | None = None
    attribute_names: list[str] | None = None
    n_bootstrap: int = 1000
    confidence: float = 0.95


# ── Helper Functions ──────────────────────────────────────────────────

def _validate_conjoint_inputs(profiles, choices):
    """Validate conjoint input data."""
    n_tasks = len(profiles)
    if n_tasks < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 5 choice tasks, got {n_tasks}.",
        )
    if len(choices) != n_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"choices length ({len(choices)}) must match profiles length ({n_tasks}).",
        )

    for t, task in enumerate(profiles):
        if len(task) < 2:
            raise HTTPException(
                status_code=422,
                detail=f"Task {t} must have at least 2 alternatives.",
            )
        if choices[t] < 0 or choices[t] >= len(task):
            raise HTTPException(
                status_code=422,
                detail=f"Choice index {choices[t]} out of range for task {t} with {len(task)} alternatives.",
            )


def _extract_attribute_levels(profiles: list[list[dict]]) -> tuple[list[str], dict]:
    """Extract all unique attribute-level combinations from profiles.

    Returns:
        attributes: sorted list of attribute names
        levels_per_attr: {attr: sorted list of unique levels}
    """
    levels_per_attr = {}
    for task in profiles:
        for alt in task:
            for attr, level in alt.items():
                if attr not in levels_per_attr:
                    levels_per_attr[attr] = set()
                levels_per_attr[attr].add(level)

    attributes = sorted(levels_per_attr.keys())
    levels_per_attr = {attr: sorted(levels_per_attr[attr], key=str)
                       for attr in attributes}

    return attributes, levels_per_attr


def _build_design_matrix(profiles: list[list[dict]], attributes: list[str],
                         levels_per_attr: dict,
                         reference_levels: dict) -> tuple[list[np.ndarray], list[str]]:
    """Build dummy-coded design matrices for each choice task.

    Uses effects coding: reference level omitted (coded as 0), other levels
    get indicator columns.

    Returns:
        X_tasks: list of design matrices, one per task (n_alternatives x n_params)
        param_names: list of parameter names
    """
    # Build parameter name list
    param_names = []
    param_map = {}  # (attr, level) -> column index
    col = 0
    for attr in attributes:
        levels = levels_per_attr[attr]
        ref = reference_levels[attr]
        for level in levels:
            if str(level) == str(ref):
                continue
            param_names.append(f"{attr}:{level}")
            param_map[(attr, str(level))] = col
            col += 1

    n_params = len(param_names)

    X_tasks = []
    for task in profiles:
        n_alts = len(task)
        X = np.zeros((n_alts, n_params))
        for a, alt in enumerate(task):
            for attr in attributes:
                level = str(alt.get(attr, reference_levels[attr]))
                if (attr, level) in param_map:
                    X[a, param_map[(attr, level)]] = 1.0
        X_tasks.append(X)

    return X_tasks, param_names


def _conditional_logit_ll(beta: np.ndarray, X_tasks: list[np.ndarray],
                          choices: list[int]) -> float:
    """Negative log-likelihood for conditional logit model."""
    ll = 0.0
    for t, X in enumerate(X_tasks):
        utilities = X @ beta
        # Log-sum-exp for numerical stability
        max_u = np.max(utilities)
        log_denom = max_u + np.log(np.sum(np.exp(utilities - max_u)))
        ll += utilities[choices[t]] - log_denom
    return -ll


def _conditional_logit_gradient(beta: np.ndarray, X_tasks: list[np.ndarray],
                                 choices: list[int]) -> np.ndarray:
    """Gradient of the negative log-likelihood."""
    grad = np.zeros_like(beta)
    for t, X in enumerate(X_tasks):
        utilities = X @ beta
        max_u = np.max(utilities)
        exp_u = np.exp(utilities - max_u)
        probs = exp_u / np.sum(exp_u)
        # Gradient: sum over alternatives of (indicator - prob) * X
        grad -= X[choices[t]] - X.T @ probs
    return grad


def _fit_conditional_logit(X_tasks: list[np.ndarray], choices: list[int],
                           n_params: int) -> tuple[np.ndarray, float, float]:
    """Fit conditional logit via maximum likelihood.

    Returns (beta, log_likelihood, null_log_likelihood).
    """
    beta0 = np.zeros(n_params)

    result = minimize(
        _conditional_logit_ll, beta0,
        args=(X_tasks, choices),
        jac=_conditional_logit_gradient,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=f"Conditional logit did not converge: {result.message}",
        )

    beta = result.x
    ll = -result.fun

    # Null model log-likelihood (equal probabilities)
    null_ll = 0.0
    for task in X_tasks:
        null_ll += -np.log(len(task))

    return beta, ll, null_ll


def _compute_standard_errors(beta: np.ndarray, X_tasks: list[np.ndarray],
                              choices: list[int]) -> np.ndarray:
    """Compute standard errors via numerical Hessian (finite differences)."""
    n_params = len(beta)
    hessian = np.zeros((n_params, n_params))
    eps = 1e-5

    for i in range(n_params):
        for j in range(i, n_params):
            # Second partial derivative via finite differences
            beta_pp = beta.copy()
            beta_pp[i] += eps
            beta_pp[j] += eps

            beta_pm = beta.copy()
            beta_pm[i] += eps
            beta_pm[j] -= eps

            beta_mp = beta.copy()
            beta_mp[i] -= eps
            beta_mp[j] += eps

            beta_mm = beta.copy()
            beta_mm[i] -= eps
            beta_mm[j] -= eps

            h = (_conditional_logit_ll(beta_pp, X_tasks, choices) -
                 _conditional_logit_ll(beta_pm, X_tasks, choices) -
                 _conditional_logit_ll(beta_mp, X_tasks, choices) +
                 _conditional_logit_ll(beta_mm, X_tasks, choices)) / (4 * eps * eps)

            hessian[i, j] = h
            hessian[j, i] = h

    # Standard errors from inverse Hessian
    try:
        cov_matrix = np.linalg.inv(hessian)
        se = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(n_params, float("nan"))

    return se


def _compute_part_worths(beta: np.ndarray, param_names: list[str],
                         attributes: list[str], levels_per_attr: dict,
                         reference_levels: dict,
                         se: np.ndarray | None = None) -> dict:
    """Convert estimated coefficients to part-worth utility table.

    Returns {attr: {level: {"utility": float, "se": float}, ...}, ...}
    """
    # Build lookup from param_names
    param_lookup = {}
    for i, name in enumerate(param_names):
        attr, level = name.split(":", 1)
        param_lookup[(attr, level)] = i

    part_worths = {}
    for attr in attributes:
        part_worths[attr] = {}
        ref = reference_levels[attr]
        # Reference level has utility 0
        part_worths[attr][str(ref)] = {
            "utility": 0.0,
            "se": 0.0,
            "is_reference": True,
        }
        for level in levels_per_attr[attr]:
            level_str = str(level)
            if level_str == str(ref):
                continue
            idx = param_lookup.get((attr, level_str))
            if idx is not None:
                part_worths[attr][level_str] = {
                    "utility": round(float(beta[idx]), 4),
                    "se": round(float(se[idx]), 4) if se is not None else None,
                    "is_reference": False,
                }

    return part_worths


def _compute_importance(part_worths: dict) -> dict:
    """Compute attribute importance from part-worth utilities.

    Importance = range of utilities for an attribute / sum of all ranges.
    """
    ranges = {}
    for attr, levels in part_worths.items():
        utilities = [v["utility"] for v in levels.values()]
        ranges[attr] = max(utilities) - min(utilities)

    total_range = sum(ranges.values())
    importance = {}
    for attr, rng in ranges.items():
        importance[attr] = {
            "range": round(rng, 4),
            "importance_pct": round(rng / total_range * 100, 2) if total_range > 0 else 0.0,
        }

    return importance


# ── 1. Conjoint Analysis ─────────────────────────────────────────────

@router.post("/analyze")
def conjoint_analyze(req: ConjointAnalyzeRequest):
    """Fit choice-based conjoint model via conditional logit.

    Takes choice task profiles and selected alternatives, fits a conditional
    logit model via maximum likelihood, and returns part-worth utilities,
    attribute importance, and model fit statistics.
    """
    _validate_conjoint_inputs(req.profiles, req.choices)

    attributes, levels_per_attr = _extract_attribute_levels(req.profiles)

    if req.attribute_names:
        if set(req.attribute_names) != set(attributes):
            raise HTTPException(
                status_code=422,
                detail=f"attribute_names {req.attribute_names} don't match attributes found in data: {attributes}.",
            )

    # Reference levels: first level alphabetically for each attribute
    reference_levels = {attr: levels_per_attr[attr][0] for attr in attributes}

    # Build design matrices
    X_tasks, param_names = _build_design_matrix(
        req.profiles, attributes, levels_per_attr, reference_levels
    )

    n_params = len(param_names)
    if n_params == 0:
        raise HTTPException(
            status_code=422,
            detail="No estimable parameters. Each attribute needs at least 2 levels.",
        )

    # Fit model
    beta, ll, null_ll = _fit_conditional_logit(X_tasks, req.choices, n_params)

    # Standard errors
    se = _compute_standard_errors(beta, X_tasks, req.choices)

    # McFadden pseudo-R^2
    pseudo_r2 = 1 - (ll / null_ll) if null_ll != 0 else 0.0

    # Part-worth utilities
    part_worths = _compute_part_worths(
        beta, param_names, attributes, levels_per_attr, reference_levels, se
    )

    # Attribute importance
    importance = _compute_importance(part_worths)

    # z-tests for each parameter
    coefficients = []
    for i, name in enumerate(param_names):
        z_val = float(beta[i] / se[i]) if se[i] > 1e-12 else 0.0
        p_val = float(2 * (1 - _norm_cdf(abs(z_val))))
        coefficients.append({
            "parameter": name,
            "estimate": round(float(beta[i]), 4),
            "se": round(float(se[i]), 4),
            "z": round(z_val, 4),
            "p_value": round(p_val, 4),
            "significant_05": p_val < 0.05,
        })

    # Best/worst levels per attribute
    preferences = {}
    for attr, levels in part_worths.items():
        sorted_levels = sorted(levels.items(), key=lambda x: x[1]["utility"], reverse=True)
        preferences[attr] = {
            "most_preferred": sorted_levels[0][0],
            "least_preferred": sorted_levels[-1][0],
        }

    n_tasks = len(req.profiles)

    return {
        "n_tasks": n_tasks,
        "n_parameters": n_params,
        "attributes": attributes,
        "levels_per_attribute": {a: [str(l) for l in levels_per_attr[a]] for a in attributes},
        "reference_levels": {a: str(v) for a, v in reference_levels.items()},
        "coefficients": coefficients,
        "part_worth_utilities": part_worths,
        "attribute_importance": importance,
        "preferences": preferences,
        "model_fit": {
            "log_likelihood": round(ll, 4),
            "null_log_likelihood": round(null_ll, 4),
            "mcfadden_pseudo_r2": round(pseudo_r2, 4),
            "n_tasks": n_tasks,
            "n_parameters": n_params,
        },
        "interpretation": (
            f"Conditional logit fit on {n_tasks} choice tasks with {len(attributes)} attributes. "
            f"McFadden pseudo-R^2 = {pseudo_r2:.3f}. "
            f"Most important attribute: "
            + max(importance.items(), key=lambda x: x[1]["importance_pct"])[0]
            + f" ({max(importance.items(), key=lambda x: x[1]['importance_pct'])[1]['importance_pct']:.1f}%). "
            + " ".join(
                f"{attr}: prefer '{preferences[attr]['most_preferred']}' "
                f"over '{preferences[attr]['least_preferred']}'."
                for attr in attributes
            )
        ),
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using scipy."""
    from scipy import stats as sp_stats
    return float(sp_stats.norm.cdf(x))


# ── 2. Predict Choice Shares ─────────────────────────────────────────

@router.post("/predict")
def predict_choice_shares(req: PredictRequest):
    """Predict choice probabilities for a set of profiles given estimated utilities.

    Uses the multinomial logit choice rule: P(j) = exp(V_j) / sum(exp(V_k)).
    """
    if len(req.profiles) < 2:
        raise HTTPException(
            status_code=422,
            detail="Need at least 2 profiles for choice share prediction.",
        )

    utilities_dict = req.utilities

    # Compute systematic utility for each profile
    v = []
    for i, profile in enumerate(req.profiles):
        utility = 0.0
        for attr, level in profile.items():
            level_str = str(level)
            if attr not in utilities_dict:
                raise HTTPException(
                    status_code=422,
                    detail=f"Attribute '{attr}' in profile {i} not found in utilities.",
                )
            if level_str not in utilities_dict[attr]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Level '{level_str}' for attribute '{attr}' not found in utilities.",
                )
            level_data = utilities_dict[attr][level_str]
            # Handle both dict (with "utility" key) and plain float
            if isinstance(level_data, dict):
                utility += level_data.get("utility", 0.0)
            else:
                utility += float(level_data)
        v.append(utility)

    v = np.array(v)

    # Choice probabilities via softmax
    max_v = np.max(v)
    exp_v = np.exp(v - max_v)
    probs = exp_v / np.sum(exp_v)

    predictions = []
    for i, profile in enumerate(req.profiles):
        predictions.append({
            "profile_index": i,
            "profile": profile,
            "utility": round(float(v[i]), 4),
            "choice_probability": round(float(probs[i]), 4),
            "choice_share_pct": round(float(probs[i] * 100), 2),
        })

    # Sort by probability
    predictions_sorted = sorted(predictions, key=lambda x: x["choice_probability"], reverse=True)

    return {
        "n_profiles": len(req.profiles),
        "predictions": predictions,
        "ranking": [p["profile_index"] for p in predictions_sorted],
        "most_preferred": predictions_sorted[0]["profile_index"],
        "interpretation": (
            f"Choice share predictions for {len(req.profiles)} profiles. "
            f"Profile {predictions_sorted[0]['profile_index']} has highest predicted share "
            f"({predictions_sorted[0]['choice_share_pct']:.1f}%). "
            f"Profile {predictions_sorted[-1]['profile_index']} has lowest share "
            f"({predictions_sorted[-1]['choice_share_pct']:.1f}%). "
            f"Shares sum to 100% under the IIA (independence of irrelevant alternatives) assumption."
        ),
    }


# ── 3. Attribute Importance with Bootstrap CI ─────────────────────────

@router.post("/importance")
def attribute_importance(req: ImportanceRequest):
    """Compute attribute importance with bootstrap confidence intervals.

    Importance is the range of part-worth utilities for each attribute
    divided by the sum of all ranges, expressed as a percentage.
    """
    _validate_conjoint_inputs(req.profiles, req.choices)

    attributes, levels_per_attr = _extract_attribute_levels(req.profiles)
    reference_levels = {attr: levels_per_attr[attr][0] for attr in attributes}

    X_tasks, param_names = _build_design_matrix(
        req.profiles, attributes, levels_per_attr, reference_levels
    )
    n_params = len(param_names)

    if n_params == 0:
        raise HTTPException(
            status_code=422,
            detail="No estimable parameters.",
        )

    # Fit full model
    beta, ll, null_ll = _fit_conditional_logit(X_tasks, req.choices, n_params)
    se = _compute_standard_errors(beta, X_tasks, req.choices)

    part_worths = _compute_part_worths(
        beta, param_names, attributes, levels_per_attr, reference_levels, se
    )
    point_importance = _compute_importance(part_worths)

    # Bootstrap for confidence intervals
    rng = np.random.default_rng(42)
    n_tasks = len(req.profiles)
    boot_importance = {attr: [] for attr in attributes}

    # Group tasks by respondent if respondent_ids provided (cluster bootstrap)
    if req.respondent_ids is not None:
        if len(req.respondent_ids) != n_tasks:
            raise HTTPException(
                status_code=422,
                detail="respondent_ids length must match number of tasks.",
            )
        respondent_map = {}
        for t, rid in enumerate(req.respondent_ids):
            if rid not in respondent_map:
                respondent_map[rid] = []
            respondent_map[rid].append(t)
        respondent_keys = list(respondent_map.keys())
        use_cluster = True
    else:
        use_cluster = False

    for b in range(req.n_bootstrap):
        try:
            if use_cluster:
                # Resample respondents
                boot_resp = rng.choice(respondent_keys, size=len(respondent_keys), replace=True)
                boot_indices = []
                for r in boot_resp:
                    boot_indices.extend(respondent_map[r])
            else:
                boot_indices = rng.choice(n_tasks, size=n_tasks, replace=True)

            boot_X = [X_tasks[i] for i in boot_indices]
            boot_choices = [req.choices[i] for i in boot_indices]

            boot_beta, _, _ = _fit_conditional_logit(boot_X, boot_choices, n_params)
            boot_pw = _compute_part_worths(
                boot_beta, param_names, attributes, levels_per_attr, reference_levels
            )
            boot_imp = _compute_importance(boot_pw)

            for attr in attributes:
                boot_importance[attr].append(boot_imp[attr]["importance_pct"])
        except Exception:
            continue  # skip failed bootstrap iterations

    # Compute CIs
    alpha = 1 - req.confidence
    importance_results = []
    for attr in attributes:
        boot_vals = np.array(boot_importance[attr])
        n_boot_valid = len(boot_vals)

        result_entry = {
            "attribute": attr,
            "importance_pct": point_importance[attr]["importance_pct"],
            "range": point_importance[attr]["range"],
        }

        if n_boot_valid >= 50:
            ci_lower = float(np.percentile(boot_vals, 100 * alpha / 2))
            ci_upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
            boot_se = float(np.std(boot_vals, ddof=1))
            result_entry["ci_lower"] = round(ci_lower, 2)
            result_entry["ci_upper"] = round(ci_upper, 2)
            result_entry["se"] = round(boot_se, 2)
            result_entry["n_valid_bootstraps"] = n_boot_valid
        else:
            result_entry["ci_lower"] = None
            result_entry["ci_upper"] = None
            result_entry["se"] = None
            result_entry["n_valid_bootstraps"] = n_boot_valid
            result_entry["warning"] = "Too few valid bootstrap samples for reliable CI."

        importance_results.append(result_entry)

    # Sort by importance
    importance_results = sorted(importance_results, key=lambda x: x["importance_pct"], reverse=True)

    return {
        "n_tasks": n_tasks,
        "n_attributes": len(attributes),
        "confidence": req.confidence,
        "importance": importance_results,
        "model_fit": {
            "log_likelihood": round(ll, 4),
            "mcfadden_pseudo_r2": round(1 - ll / null_ll if null_ll != 0 else 0.0, 4),
        },
        "interpretation": (
            f"Attribute importance analysis across {len(attributes)} attributes "
            f"({n_tasks} tasks, {req.n_bootstrap} bootstrap iterations). "
            f"Most important: {importance_results[0]['attribute']} "
            f"({importance_results[0]['importance_pct']:.1f}%"
            + (f", {req.confidence*100:.0f}% CI: [{importance_results[0].get('ci_lower', '?')}, "
               f"{importance_results[0].get('ci_upper', '?')}]"
               if importance_results[0].get("ci_lower") is not None else "")
            + f"). Least important: {importance_results[-1]['attribute']} "
              f"({importance_results[-1]['importance_pct']:.1f}%)."
        ),
    }
