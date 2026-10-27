"""Robust fitting for statsmodels linear mixed models.

statsmodels' default optimizer (BFGS) can stop short of the (RE)ML optimum and report
non-convergence, and L-BFGS occasionally returns a non-finite likelihood, especially when a
variance component is near zero. lme4 uses derivative-free optimizers (bobyqa,
Nelder-Mead) for the same reason. fit_mixedlm tries several optimizers and keeps the fit
with the highest finite (restricted) log-likelihood.
"""

import warnings

import numpy as np

METHODS = ("bfgs", "nm", "powell", "lbfgs")


def fit_mixedlm(model, reml: bool = True, **kwargs):
    best, errors = None, []
    for method in METHODS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(reml=reml, method=method, maxiter=5000, **kwargs)
        except Exception as exc:  # singular matrices etc. for one optimizer
            errors.append(exc)
            continue
        if not np.isfinite(res.llf) or not np.all(np.isfinite(res.fe_params)):
            continue
        if best is None or res.llf > best.llf + 1e-9:
            best = res
    if best is None:
        raise errors[0] if errors else RuntimeError("Mixed model did not converge with any optimizer.")
    return best


def fixed_effect_se(result):
    """(SE, covariance) of the fixed effects: (X' V^-1 X)^-1 at the estimated variance
    components, as lme4's vcov(). statsmodels' bse_fe comes from a numerical Hessian of all
    parameters and drifts from this when a variance component is near zero. Models with
    extra variance components (vc_formula) keep statsmodels' values."""
    import pandas as pd

    model = result.model
    if getattr(model, "k_vc", 0):
        k = len(result.fe_params)
        return result.bse_fe, result.cov_params().iloc[:k, :k]
    X, Z = model.exog, model.exog_re
    G = np.asarray(result.cov_re)
    XtVX = np.zeros((X.shape[1], X.shape[1]))
    for idx in model.row_indices.values():
        Zi, Xi = Z[idx], X[idx]
        V = Zi @ G @ Zi.T + result.scale * np.eye(len(idx))
        XtVX += Xi.T @ np.linalg.solve(V, Xi)
    cov = np.linalg.inv(XtVX)
    return pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=result.fe_params.index), \
        pd.DataFrame(cov, index=result.fe_params.index, columns=result.fe_params.index)
