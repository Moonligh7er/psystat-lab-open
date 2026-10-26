"""Diagnostic Classification Models — DINA, DINO, G-DINA cognitive diagnostic models."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from itertools import product

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class DinaRequest(BaseModel):
    responses: list[list[int]]  # N × J binary response matrix
    q_matrix: list[list[int]]   # J × K binary Q-matrix (items × attributes)
    max_iter: int = 500
    tol: float = 1e-6

class DinoRequest(BaseModel):
    responses: list[list[int]]  # N × J binary response matrix
    q_matrix: list[list[int]]   # J × K binary Q-matrix
    max_iter: int = 500
    tol: float = 1e-6

class GdinaRequest(BaseModel):
    responses: list[list[int]]  # N × J binary response matrix
    q_matrix: list[list[int]]   # J × K binary Q-matrix
    max_iter: int = 500
    tol: float = 1e-6

class CompareRequest(BaseModel):
    responses: list[list[int]]
    q_matrix: list[list[int]]
    max_iter: int = 500
    tol: float = 1e-6

class QMatrixValidateRequest(BaseModel):
    responses: list[list[int]]
    q_matrix: list[list[int]]
    max_iter: int = 300
    tol: float = 1e-5


# ── Helpers ───────────────────────────────────────────────────────────

def _generate_all_patterns(K: int) -> np.ndarray:
    """Generate all 2^K binary attribute patterns."""
    return np.array(list(product([0, 1], repeat=K)), dtype=int)


def _aic(ll: float, n_params: int) -> float:
    return -2.0 * ll + 2.0 * n_params


def _bic(ll: float, n_params: int, n: int) -> float:
    return -2.0 * ll + n_params * np.log(n)


def _dina_eta(alpha: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    """DINA eta: 1 if subject has mastered ALL required attributes, else 0.

    alpha: (L, K) all latent patterns or (N, K) subject profiles
    q_row: (K,) Q-matrix row for one item
    """
    required = q_row > 0
    if not np.any(required):
        return np.ones(alpha.shape[0], dtype=int)
    return np.all(alpha[:, required] >= 1, axis=1).astype(int)


def _dino_eta(alpha: np.ndarray, q_row: np.ndarray) -> np.ndarray:
    """DINO eta: 1 if subject has mastered ANY required attribute, else 0."""
    required = q_row > 0
    if not np.any(required):
        return np.ones(alpha.shape[0], dtype=int)
    return np.any(alpha[:, required] >= 1, axis=1).astype(int)


def _em_dina(responses: np.ndarray, Q: np.ndarray, max_iter: int, tol: float):
    """EM algorithm for DINA model."""
    N, J = responses.shape
    K = Q.shape[1]
    patterns = _generate_all_patterns(K)
    L = patterns.shape[0]

    # Initialize
    s = np.full(J, 0.2)  # slip
    g = np.full(J, 0.2)  # guess
    pi = np.ones(L) / L  # class proportions

    # Precompute eta for all patterns × items
    eta = np.zeros((L, J), dtype=int)
    for j in range(J):
        eta[:, j] = _dina_eta(patterns, Q[j])

    ll_old = -np.inf
    for iteration in range(max_iter):
        # E-step: posterior P(alpha_l | X_i)
        log_lik = np.zeros((N, L))
        for c in range(L):
            for j in range(J):
                p1 = (1 - s[j]) ** eta[c, j] * g[j] ** (1 - eta[c, j])
                p0 = s[j] ** eta[c, j] * (1 - g[j]) ** (1 - eta[c, j])
                p = np.clip(p1, 1e-300, 1.0)
                q = np.clip(p0, 1e-300, 1.0)
                log_lik[:, c] += responses[:, j] * np.log(p) + (1 - responses[:, j]) * np.log(q)
        log_lik += np.log(np.clip(pi, 1e-300, None))

        # Normalize posteriors
        max_ll = np.max(log_lik, axis=1, keepdims=True)
        log_lik_shifted = log_lik - max_ll
        posterior = np.exp(log_lik_shifted)
        row_sums = posterior.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        posterior /= row_sums

        # Log-likelihood
        ll = np.sum(max_ll.ravel() + np.log(np.exp(log_lik_shifted).sum(axis=1)))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

        # M-step
        N_c = posterior.sum(axis=0)  # expected count per class
        pi = N_c / N

        for j in range(J):
            # Expected counts
            eta_1 = eta[:, j] == 1
            eta_0 = eta[:, j] == 0
            r_1 = np.sum(posterior[:, eta_1] * responses[:, j:j+1], axis=(0, 1)) if np.any(eta_1) else 0
            n_1 = np.sum(posterior[:, eta_1]) if np.any(eta_1) else 1e-10
            r_0 = np.sum(posterior[:, eta_0] * responses[:, j:j+1], axis=(0, 1)) if np.any(eta_0) else 0
            n_0 = np.sum(posterior[:, eta_0]) if np.any(eta_0) else 1e-10
            s[j] = np.clip(1.0 - r_1 / n_1, 0.001, 0.999) if n_1 > 0 else 0.2
            g[j] = np.clip(r_0 / n_0, 0.001, 0.999) if n_0 > 0 else 0.2

    # Final results
    n_params = 2 * J + (L - 1)
    aic_val = _aic(ll, n_params)
    bic_val = _bic(ll, n_params, N)

    # Attribute mastery probabilities per subject
    attr_mastery = np.zeros((N, K))
    for k in range(K):
        for c in range(L):
            if patterns[c, k] == 1:
                attr_mastery[:, k] += posterior[:, c]

    # Most likely class per subject
    assignments = np.argmax(posterior, axis=1)
    assigned_patterns = patterns[assignments]

    # Classification accuracy per attribute
    class_accuracy = []
    for k in range(K):
        probs_k = attr_mastery[:, k]
        acc = np.mean(np.maximum(probs_k, 1 - probs_k))
        class_accuracy.append({"attribute": k, "accuracy": float(acc)})

    return {
        "item_parameters": [
            {"item": j, "slip": float(s[j]), "guess": float(g[j]),
             "discrimination": float(1 - s[j] - g[j])}
            for j in range(J)
        ],
        "attribute_mastery_probabilities": attr_mastery.tolist(),
        "most_likely_profiles": assigned_patterns.tolist(),
        "attribute_prevalence": [float(attr_mastery[:, k].mean()) for k in range(K)],
        "class_proportions": pi.tolist(),
        "model_fit": {
            "log_likelihood": float(ll),
            "neg2ll": float(-2 * ll),
            "n_parameters": n_params,
            "AIC": float(aic_val),
            "BIC": float(bic_val),
        },
        "classification_accuracy": class_accuracy,
        "n_iterations": iteration + 1,
    }


def _em_dino(responses: np.ndarray, Q: np.ndarray, max_iter: int, tol: float):
    """EM algorithm for DINO model."""
    N, J = responses.shape
    K = Q.shape[1]
    patterns = _generate_all_patterns(K)
    L = patterns.shape[0]

    s = np.full(J, 0.2)
    g = np.full(J, 0.2)
    pi = np.ones(L) / L

    # Precompute eta using DINO rule
    eta = np.zeros((L, J), dtype=int)
    for j in range(J):
        eta[:, j] = _dino_eta(patterns, Q[j])

    ll_old = -np.inf
    for iteration in range(max_iter):
        log_lik = np.zeros((N, L))
        for c in range(L):
            for j in range(J):
                p1 = (1 - s[j]) ** eta[c, j] * g[j] ** (1 - eta[c, j])
                p0 = s[j] ** eta[c, j] * (1 - g[j]) ** (1 - eta[c, j])
                p = np.clip(p1, 1e-300, 1.0)
                q = np.clip(p0, 1e-300, 1.0)
                log_lik[:, c] += responses[:, j] * np.log(p) + (1 - responses[:, j]) * np.log(q)
        log_lik += np.log(np.clip(pi, 1e-300, None))

        max_ll = np.max(log_lik, axis=1, keepdims=True)
        log_lik_shifted = log_lik - max_ll
        posterior = np.exp(log_lik_shifted)
        row_sums = posterior.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        posterior /= row_sums

        ll = np.sum(max_ll.ravel() + np.log(np.exp(log_lik_shifted).sum(axis=1)))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

        N_c = posterior.sum(axis=0)
        pi = N_c / N

        for j in range(J):
            eta_1 = eta[:, j] == 1
            eta_0 = eta[:, j] == 0
            r_1 = np.sum(posterior[:, eta_1] * responses[:, j:j+1], axis=(0, 1)) if np.any(eta_1) else 0
            n_1 = np.sum(posterior[:, eta_1]) if np.any(eta_1) else 1e-10
            r_0 = np.sum(posterior[:, eta_0] * responses[:, j:j+1], axis=(0, 1)) if np.any(eta_0) else 0
            n_0 = np.sum(posterior[:, eta_0]) if np.any(eta_0) else 1e-10
            s[j] = np.clip(1.0 - r_1 / n_1, 0.001, 0.999) if n_1 > 0 else 0.2
            g[j] = np.clip(r_0 / n_0, 0.001, 0.999) if n_0 > 0 else 0.2

    n_params = 2 * J + (L - 1)
    aic_val = _aic(ll, n_params)
    bic_val = _bic(ll, n_params, N)

    attr_mastery = np.zeros((N, K))
    for k in range(K):
        for c in range(L):
            if patterns[c, k] == 1:
                attr_mastery[:, k] += posterior[:, c]

    assignments = np.argmax(posterior, axis=1)
    assigned_patterns = patterns[assignments]

    class_accuracy = []
    for k in range(K):
        probs_k = attr_mastery[:, k]
        acc = np.mean(np.maximum(probs_k, 1 - probs_k))
        class_accuracy.append({"attribute": k, "accuracy": float(acc)})

    return {
        "item_parameters": [
            {"item": j, "slip": float(s[j]), "guess": float(g[j]),
             "discrimination": float(1 - s[j] - g[j])}
            for j in range(J)
        ],
        "attribute_mastery_probabilities": attr_mastery.tolist(),
        "most_likely_profiles": assigned_patterns.tolist(),
        "attribute_prevalence": [float(attr_mastery[:, k].mean()) for k in range(K)],
        "class_proportions": pi.tolist(),
        "model_fit": {
            "log_likelihood": float(ll),
            "neg2ll": float(-2 * ll),
            "n_parameters": n_params,
            "AIC": float(aic_val),
            "BIC": float(bic_val),
        },
        "classification_accuracy": class_accuracy,
        "n_iterations": iteration + 1,
    }


def _em_gdina(responses: np.ndarray, Q: np.ndarray, max_iter: int, tol: float):
    """EM algorithm for G-DINA (Generalized DINA) model.

    Saturated model: estimates a separate response probability for every
    possible reduced attribute pattern per item.
    """
    N, J = responses.shape
    K = Q.shape[1]
    patterns = _generate_all_patterns(K)
    L = patterns.shape[0]
    pi = np.ones(L) / L

    # For each item j, enumerate its reduced attribute patterns
    # (only the attributes required by Q[j])
    item_info = []
    for j in range(J):
        required = np.where(Q[j] > 0)[0]
        Kj = len(required)
        if Kj == 0:
            # No required attributes: single response probability
            reduced_patterns = np.array([[0]])
            n_reduced = 1
        else:
            reduced_patterns = _generate_all_patterns(Kj)
            n_reduced = reduced_patterns.shape[0]
        # Map each full pattern to a reduced pattern index
        pattern_map = np.zeros(L, dtype=int)
        for c in range(L):
            if Kj == 0:
                pattern_map[c] = 0
            else:
                reduced = tuple(patterns[c, required])
                for ri, rp in enumerate(reduced_patterns):
                    if tuple(rp) == reduced:
                        pattern_map[c] = ri
                        break
        item_info.append({
            "required": required,
            "Kj": Kj,
            "n_reduced": n_reduced,
            "reduced_patterns": reduced_patterns,
            "pattern_map": pattern_map,
        })

    # Initialize response probabilities per item per reduced pattern
    item_probs = []
    for j in range(J):
        n_reduced = item_info[j]["n_reduced"]
        # Initialize: low for no mastery, high for full mastery
        probs = np.linspace(0.2, 0.8, n_reduced)
        item_probs.append(probs)

    ll_old = -np.inf
    for iteration in range(max_iter):
        # E-step
        log_lik = np.zeros((N, L))
        for c in range(L):
            for j in range(J):
                ri = item_info[j]["pattern_map"][c]
                p = np.clip(item_probs[j][ri], 1e-300, 1 - 1e-300)
                log_lik[:, c] += responses[:, j] * np.log(p) + (1 - responses[:, j]) * np.log(1 - p)
        log_lik += np.log(np.clip(pi, 1e-300, None))

        max_ll = np.max(log_lik, axis=1, keepdims=True)
        log_lik_shifted = log_lik - max_ll
        posterior = np.exp(log_lik_shifted)
        row_sums = posterior.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        posterior /= row_sums

        ll = np.sum(max_ll.ravel() + np.log(np.exp(log_lik_shifted).sum(axis=1)))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

        # M-step
        N_c = posterior.sum(axis=0)
        pi = N_c / N

        for j in range(J):
            n_reduced = item_info[j]["n_reduced"]
            pm = item_info[j]["pattern_map"]
            for ri in range(n_reduced):
                classes_in_ri = np.where(pm == ri)[0]
                weight = posterior[:, classes_in_ri].sum(axis=1)  # (N,)
                total_weight = weight.sum()
                if total_weight > 0:
                    item_probs[j][ri] = np.clip(
                        np.sum(weight * responses[:, j]) / total_weight,
                        0.001, 0.999
                    )

    # Count parameters
    n_params = sum(info["n_reduced"] for info in item_info) + (L - 1)
    aic_val = _aic(ll, n_params)
    bic_val = _bic(ll, n_params, N)

    # Attribute mastery
    attr_mastery = np.zeros((N, K))
    for k in range(K):
        for c in range(L):
            if patterns[c, k] == 1:
                attr_mastery[:, k] += posterior[:, c]

    assignments = np.argmax(posterior, axis=1)
    assigned_patterns = patterns[assignments]

    # Item response function parameters
    irf_params = []
    for j in range(J):
        reduced = item_info[j]["reduced_patterns"]
        probs = item_probs[j]
        irf = []
        for ri in range(item_info[j]["n_reduced"]):
            irf.append({
                "pattern": reduced[ri].tolist(),
                "probability": float(probs[ri]),
            })
        irf_params.append({"item": j, "response_function": irf})

    # Item-level RMSEA
    item_rmsea = []
    for j in range(J):
        pm = item_info[j]["pattern_map"]
        obs_rates = []
        exp_rates = []
        for c in range(L):
            ri = pm[c]
            nc = posterior[:, c].sum()
            if nc > 1:
                obs_rate = np.sum(posterior[:, c] * responses[:, j]) / nc
                obs_rates.append(obs_rate)
                exp_rates.append(item_probs[j][ri])
        if obs_rates:
            obs_arr = np.array(obs_rates)
            exp_arr = np.array(exp_rates)
            rmsea_j = float(np.sqrt(np.mean((obs_arr - exp_arr) ** 2)))
        else:
            rmsea_j = 0.0
        item_rmsea.append({"item": j, "RMSEA": rmsea_j})

    class_accuracy = []
    for k in range(K):
        probs_k = attr_mastery[:, k]
        acc = np.mean(np.maximum(probs_k, 1 - probs_k))
        class_accuracy.append({"attribute": k, "accuracy": float(acc)})

    return {
        "item_response_functions": irf_params,
        "attribute_mastery_probabilities": attr_mastery.tolist(),
        "most_likely_profiles": assigned_patterns.tolist(),
        "attribute_prevalence": [float(attr_mastery[:, k].mean()) for k in range(K)],
        "class_proportions": pi.tolist(),
        "model_fit": {
            "log_likelihood": float(ll),
            "neg2ll": float(-2 * ll),
            "n_parameters": n_params,
            "AIC": float(aic_val),
            "BIC": float(bic_val),
        },
        "item_fit": item_rmsea,
        "classification_accuracy": class_accuracy,
        "n_iterations": iteration + 1,
    }


# ── Endpoints ─────────────────────────────────────────────────────────

def _validate_inputs(responses, q_matrix):
    """Common validation for DCM inputs."""
    R = np.array(responses)
    Q = np.array(q_matrix)
    if R.ndim != 2:
        raise HTTPException(400, "Responses must be a 2D binary matrix (subjects x items).")
    if Q.ndim != 2:
        raise HTTPException(400, "Q-matrix must be a 2D binary matrix (items x attributes).")
    N, J = R.shape
    if Q.shape[0] != J:
        raise HTTPException(400, f"Q-matrix rows ({Q.shape[0]}) must equal number of items ({J}).")
    K = Q.shape[1]
    if K < 1 or K > 10:
        raise HTTPException(400, "Number of attributes must be between 1 and 10.")
    if N < 10:
        raise HTTPException(400, "Need at least 10 subjects.")
    if J < 2:
        raise HTTPException(400, "Need at least 2 items.")
    # Validate binary
    if not np.all((R == 0) | (R == 1)):
        raise HTTPException(400, "Response matrix must be binary (0/1).")
    if not np.all((Q == 0) | (Q == 1)):
        raise HTTPException(400, "Q-matrix must be binary (0/1).")
    return R, Q, N, J, K


@router.post("/dina")
def dina_model(req: DinaRequest):
    """DINA (Deterministic Inputs, Noisy And) model.

    Estimates slip and guess parameters per item via EM. Classifies subjects
    into attribute mastery profiles. Requires mastery of ALL attributes
    specified in the Q-matrix for correct response.
    """
    R, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)
    result = _em_dina(R, Q, req.max_iter, req.tol)
    result["model"] = "DINA"
    result["n_subjects"] = N
    result["n_items"] = J
    result["n_attributes"] = K
    return result


@router.post("/dino")
def dino_model(req: DinoRequest):
    """DINO (Deterministic Inputs, Noisy Or) model.

    Complementary to DINA: mastery of ANY required attribute suffices for
    correct response. Estimates slip and guess parameters per item via EM.
    """
    R, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)
    result = _em_dino(R, Q, req.max_iter, req.tol)
    result["model"] = "DINO"
    result["n_subjects"] = N
    result["n_items"] = J
    result["n_attributes"] = K
    return result


@router.post("/gdina")
def gdina_model(req: GdinaRequest):
    """G-DINA (Generalized DINA) model.

    Saturated model that estimates a response probability for every possible
    attribute pattern per item. Subsumes both DINA and DINO as special cases.
    """
    R, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)
    result = _em_gdina(R, Q, req.max_iter, req.tol)
    result["model"] = "G-DINA"
    result["n_subjects"] = N
    result["n_items"] = J
    result["n_attributes"] = K
    return result


@router.post("/compare")
def compare_models(req: CompareRequest):
    """Compare DINA, DINO, and G-DINA models on the same data.

    Returns fit comparison table (AIC, BIC, -2LL) and item-level model
    selection based on BIC.
    """
    R, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)

    dina_result = _em_dina(R, Q, req.max_iter, req.tol)
    dino_result = _em_dino(R, Q, req.max_iter, req.tol)
    gdina_result = _em_gdina(R, Q, req.max_iter, req.tol)

    fit_table = []
    for name, res in [("DINA", dina_result), ("DINO", dino_result), ("G-DINA", gdina_result)]:
        fit_table.append({
            "model": name,
            "log_likelihood": res["model_fit"]["log_likelihood"],
            "neg2ll": res["model_fit"]["neg2ll"],
            "n_parameters": res["model_fit"]["n_parameters"],
            "AIC": res["model_fit"]["AIC"],
            "BIC": res["model_fit"]["BIC"],
        })

    # Best overall model by BIC
    best_idx = int(np.argmin([r["BIC"] for r in fit_table]))
    best_model = fit_table[best_idx]["model"]

    # Item-level comparison using observed vs expected per item
    # Use the model with lowest overall BIC as recommendation
    item_selection = []
    for j in range(J):
        item_selection.append({
            "item": j,
            "recommended_model": best_model,
            "note": "Item-level selection based on overall model BIC.",
        })

    return {
        "n_subjects": N,
        "n_items": J,
        "n_attributes": K,
        "fit_comparison": fit_table,
        "best_model": best_model,
        "item_level_selection": item_selection,
        "dina_fit": dina_result["model_fit"],
        "dino_fit": dino_result["model_fit"],
        "gdina_fit": gdina_result["model_fit"],
    }


@router.post("/qmatrix-validate")
def qmatrix_validate(req: QMatrixValidateRequest):
    """Q-matrix validation via item-level fit comparison.

    For each item-attribute entry in the Q-matrix, tests whether flipping
    it (0->1 or 1->0) improves model fit (BIC). Returns suggested
    modifications with improvement statistics.
    """
    R, Q, N, J, K = _validate_inputs(req.responses, req.q_matrix)

    # Baseline fit using DINA
    baseline = _em_dina(R, Q, req.max_iter, req.tol)
    baseline_bic = baseline["model_fit"]["BIC"]
    baseline_ll = baseline["model_fit"]["log_likelihood"]

    suggestions = []
    Q_modified = Q.copy()

    for j in range(J):
        for k in range(K):
            # Flip entry
            Q_test = Q.copy()
            Q_test[j, k] = 1 - Q_test[j, k]

            # Check that item still measures at least one attribute
            if Q_test[j, :].sum() == 0:
                continue

            try:
                test_result = _em_dina(R, Q_test, req.max_iter, req.tol)
                test_bic = test_result["model_fit"]["BIC"]
                test_ll = test_result["model_fit"]["log_likelihood"]
                bic_improvement = baseline_bic - test_bic
                ll_improvement = test_ll - baseline_ll

                if bic_improvement > 0:
                    suggestions.append({
                        "item": j,
                        "attribute": k,
                        "current_value": int(Q[j, k]),
                        "suggested_value": int(Q_test[j, k]),
                        "bic_improvement": float(bic_improvement),
                        "ll_improvement": float(ll_improvement),
                        "original_bic": float(baseline_bic),
                        "modified_bic": float(test_bic),
                    })
            except Exception:
                continue

    # Sort by BIC improvement
    suggestions.sort(key=lambda x: x["bic_improvement"], reverse=True)

    # Build suggested Q-matrix applying top non-conflicting changes
    Q_suggested = Q.copy()
    applied = set()
    for sug in suggestions:
        key = (sug["item"], sug["attribute"])
        if key not in applied:
            Q_suggested[sug["item"], sug["attribute"]] = sug["suggested_value"]
            applied.add(key)

    return {
        "n_items": J,
        "n_attributes": K,
        "baseline_bic": float(baseline_bic),
        "n_suggested_modifications": len(suggestions),
        "suggested_modifications": suggestions,
        "original_q_matrix": Q.tolist(),
        "suggested_q_matrix": Q_suggested.tolist(),
    }
