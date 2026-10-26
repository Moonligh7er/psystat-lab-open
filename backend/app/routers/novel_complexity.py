"""
Model Complexity Navigator
==========================
An honest implementation of the "Natural Resting Point" idea.

Fits a ladder of nested models (mean only, linear, then polynomials of
increasing degree) and reports AIC, AICc, BIC, adjusted R² and k-fold
cross-validated RMSE for each, so you can see where extra complexity stops
paying off. This is standard model selection presented as one view; it is
not a new method.

References: Akaike (1974); Hurvich & Tsai (1989) for AICc; Schwarz (1978);
Kass & Raftery (1995) for interpreting BIC differences.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as scipy_stats

router = APIRouter()

warnings.filterwarnings("ignore")


# ── Request / Response ────────────────────────────────────────────────

class ComplexityRequest(BaseModel):
    x: List[List[float]]  # n_samples × n_features
    y: List[float]
    variable_names: Optional[List[str]] = None
    max_poly_degree: int = 5
    cv_folds: int = 5


class ModelStep(BaseModel):
    name: str
    complexity_level: int  # 0 = simplest
    n_parameters: int
    r_squared: float
    adj_r_squared: float
    aic: float
    aicc: Optional[float] = None  # None when n - k - 1 <= 0
    bic: float
    rmse: float
    cv_rmse: Optional[float] = None
    delta_aic: Optional[float] = None  # relative to best
    delta_aicc: Optional[float] = None
    delta_bic: Optional[float] = None
    marginal_gain: Optional[float] = None  # improvement over previous step


class ComplexityResponse(BaseModel):
    n_observations: int
    n_features: int
    complexity_path: List[ModelStep]
    resting_point: ModelStep  # the recommended model
    resting_point_reason: str
    simplest_adequate: ModelStep  # simplest model with ΔBIC < 2
    best_fit: ModelStep  # lowest CV error
    overfitting_detected: bool
    overfitting_starts_at: Optional[str] = None
    interpretation: str
    diminishing_returns_at: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────

# Gaussian likelihood criteria. k counts the regression coefficients plus the
# error variance; the additive constant n·(log 2π + 1) is dropped, which does
# not affect differences between models.

def _loglik_term(n: int, rss: float) -> float:
    return n * np.log(max(rss, 1e-300) / n)


def _aic(n: int, k: int, rss: float) -> float:
    return _loglik_term(n, rss) + 2 * k


def _aicc(n: int, k: int, rss: float) -> Optional[float]:
    if n - k - 1 <= 0:
        return None
    return _aic(n, k, rss) + 2 * k * (k + 1) / (n - k - 1)


def _bic(n: int, k: int, rss: float) -> float:
    return _loglik_term(n, rss) + k * np.log(n)


def _adj_r2(r2: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def _fit_ols(X: np.ndarray, y: np.ndarray):
    """Fit OLS and return coefficients, predictions, RSS."""
    # Add intercept
    ones = np.ones((X.shape[0], 1))
    Xa = np.hstack([ones, X]) if X.ndim == 2 else np.column_stack([ones, X])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(Xa, y, rcond=None)
        pred = Xa @ beta
        rss = float(np.sum((y - pred) ** 2))
    except Exception:
        pred = np.full_like(y, np.mean(y))
        rss = float(np.sum((y - pred) ** 2))
        beta = np.array([np.mean(y)])
    return beta, pred, rss


def _folds(n: int, k: int) -> list:
    """The same k folds for every model, so their CV errors are comparable."""
    k = max(2, min(k, n))
    idx = np.random.default_rng(42).permutation(n)
    return np.array_split(idx, k)


def _cv_rmse_mean(y: np.ndarray, folds: list) -> float:
    """K-fold CV RMSE of the intercept-only model (predict the training mean)."""
    errors = []
    for fold in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        errors.extend((y[fold] - y[mask].mean()) ** 2)
    return float(np.sqrt(np.mean(errors)))


def _cv_rmse(X: np.ndarray, y: np.ndarray, folds: list) -> float:
    """K-fold cross-validation RMSE."""
    n = len(y)

    errors = []
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[~mask], y[~mask]

        if len(X_train) < 2 or len(X_test) < 1:
            continue

        ones_tr = np.ones((X_train.shape[0], 1))
        ones_te = np.ones((X_test.shape[0], 1))
        Xa_tr = np.hstack([ones_tr, X_train]) if X_train.ndim == 2 else np.column_stack([ones_tr, X_train])
        Xa_te = np.hstack([ones_te, X_test]) if X_test.ndim == 2 else np.column_stack([ones_te, X_test])

        try:
            beta = np.linalg.lstsq(Xa_tr, y_train, rcond=None)[0]
            pred = Xa_te @ beta
            errors.extend((y_test - pred) ** 2)
        except Exception:
            errors.extend((y_test - np.mean(y_train)) ** 2)

    return float(np.sqrt(np.mean(errors))) if errors else float('inf')


def _poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Powers of each predictor up to `degree`, plus pairwise products when there
    are several predictors. Predictors are standardised first so high powers stay
    numerically well conditioned (this does not change the fitted values)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_features = X.shape[1]
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)

    cols = [X]
    for d in range(2, degree + 1):
        cols.append(X ** d)

    # Add interactions for degree >= 2 if multiple features
    if degree >= 2 and n_features > 1:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                cols.append((X[:, i] * X[:, j]).reshape(-1, 1))

    return np.hstack(cols)


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/navigate")
async def navigate_complexity(req: ComplexityRequest):
    X = np.array(req.x)
    y = np.array(req.y)
    n = len(y)

    if n < 5:
        raise HTTPException(422, "Need at least 5 observations.")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != n:
        raise HTTPException(422, "X and y must have the same number of observations.")

    n_feat = X.shape[1]
    var_names = req.variable_names or [f"x{i+1}" for i in range(n_feat)]
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss == 0:
        raise HTTPException(422, "Target variable has zero variance.")

    folds = _folds(n, req.cv_folds)
    steps: list[ModelStep] = []

    def step(name, level, n_coef, rss, cv):
        """n_coef counts coefficients incl. the intercept; the criteria add 1 for σ²."""
        r2 = 1 - rss / tss
        aicc = _aicc(n, n_coef + 1, rss)
        return ModelStep(
            name=name, complexity_level=level, n_parameters=n_coef,
            r_squared=round(max(0.0, r2), 6),
            adj_r_squared=round(_adj_r2(r2, n, n_coef - 1), 6),
            aic=round(_aic(n, n_coef + 1, rss), 4),
            aicc=round(aicc, 4) if aicc is not None else None,
            bic=round(_bic(n, n_coef + 1, rss), 4),
            rmse=round(float(np.sqrt(rss / n)), 6),
            cv_rmse=round(cv, 6),
        )

    # Step 0: intercept only (predict the mean)
    steps.append(step("Intercept only (mean)", 0, 1, tss, _cv_rmse_mean(y, folds)))

    # Step 1: linear regression
    _, _, rss_lin = _fit_ols(X, y)
    steps.append(step(f"Linear ({' + '.join(var_names)})", 1, n_feat + 1, rss_lin, _cv_rmse(X, y, folds)))

    # Steps 2+: Polynomial models of increasing degree
    max_degree = min(req.max_poly_degree, max(2, (n - 2) // (n_feat + 1)))
    for degree in range(2, max_degree + 1):
        X_poly = _poly_features(X, degree)
        k_poly = X_poly.shape[1] + 1

        # Stop when the model would leave fewer than 3 residual degrees of freedom
        if k_poly >= n - 2:
            break

        _, _, rss_poly = _fit_ols(X_poly, y)
        name = f"Polynomial (degree {degree})" + (" + pairwise products" if n_feat > 1 else "")
        steps.append(step(name, degree, k_poly, rss_poly, _cv_rmse(X_poly, y, folds)))

    # Compute deltas and marginal gains
    best_aic = min(s.aic for s in steps)
    best_bic = min(s.bic for s in steps)
    aiccs = [s.aicc for s in steps if s.aicc is not None]
    best_aicc = min(aiccs) if aiccs else None
    for i, s in enumerate(steps):
        s.delta_aic = round(s.aic - best_aic, 4)
        s.delta_bic = round(s.bic - best_bic, 4)
        if s.aicc is not None and best_aicc is not None:
            s.delta_aicc = round(s.aicc - best_aicc, 4)
        if i > 0:
            prev = steps[i - 1]
            s.marginal_gain = round(s.adj_r_squared - prev.adj_r_squared, 6)

    # Find resting point: lowest BIC
    resting = min(steps, key=lambda s: s.bic)
    resting_reason = (
        f"BIC selects '{resting.name}' (BIC = {resting.bic:.1f}). "
        f"This model explains {resting.adj_r_squared:.1%} of variance (adjusted) "
        f"with {resting.n_parameters} coefficients. Every other model on the ladder has a higher BIC."
    )

    # Simplest adequate: ΔBIC < 2 is "not worth more than a bare mention" (Kass & Raftery, 1995)
    adequate = [s for s in steps if s.delta_bic is not None and s.delta_bic < 2]
    simplest = min(adequate, key=lambda s: s.complexity_level) if adequate else steps[0]

    # Best fit by CV
    cv_steps = [s for s in steps if s.cv_rmse is not None]
    best_cv = min(cv_steps, key=lambda s: s.cv_rmse) if cv_steps else steps[0]

    # Overfitting signal: CV error rises from one step to the next while training error falls
    # (training error never rises along a nested ladder, so this is "CV error rises").
    overfitting = False
    overfit_name = None
    for i in range(1, len(steps)):
        if (steps[i].cv_rmse is not None and steps[i - 1].cv_rmse is not None
                and steps[i].cv_rmse > steps[i - 1].cv_rmse
                and steps[i].rmse < steps[i - 1].rmse):
            overfitting = True
            overfit_name = steps[i].name
            break

    # Diminishing returns: first step whose adjusted R² gain is below 0.01 (one percentage point)
    diminishing = None
    for s in steps:
        if s.marginal_gain is not None and s.marginal_gain < 0.01 and s.complexity_level >= 1:
            diminishing = s.name
            break

    # Build interpretation
    interpretation = (
        f"Your data has {n} observations and {n_feat} predictor(s). "
        f"We fitted {len(steps)} models from simplest (just the mean) to most complex "
        f"(degree-{max_degree} polynomial). "
    )
    if resting.complexity_level == 0:
        interpretation += (
            "By BIC, no model improves on simply predicting the mean. "
            "This is not proof of no relationship, but the data give no support for one on this ladder."
        )
    elif resting.complexity_level == 1:
        interpretation += (
            f"By BIC, the linear model is preferred: it explains {resting.adj_r_squared:.1%} "
            f"of variance (adjusted) and polynomial terms do not improve BIC."
        )
    else:
        interpretation += (
            f"By BIC, the degree-{resting.complexity_level} polynomial is preferred, "
            f"suggesting curvature. Polynomial fits are unreliable outside the range of the data, "
            f"and with small samples the choice can change with a few observations."
        )
    if overfitting:
        interpretation += f" Overfitting detected starting at '{overfit_name}'."

    return ComplexityResponse(
        n_observations=n,
        n_features=n_feat,
        complexity_path=steps,
        resting_point=resting,
        resting_point_reason=resting_reason,
        simplest_adequate=simplest,
        best_fit=best_cv,
        overfitting_detected=overfitting,
        overfitting_starts_at=overfit_name,
        interpretation=interpretation,
        diminishing_returns_at=diminishing,
    )
