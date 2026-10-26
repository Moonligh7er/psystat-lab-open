"""MICE — Multiple Imputation by Chained Equations."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

router = APIRouter()


class MICERequest(BaseModel):
    data: list[dict]  # rows as dicts (like JSON records)
    n_imputations: int = 5
    max_iterations: int = 10
    seed: int | None = None


def _mice_impute(df: pd.DataFrame, n_imputations: int, max_iterations: int, seed: int | None) -> list[pd.DataFrame]:
    """Run MICE algorithm: iterative conditional imputation via chained equations."""
    rng = np.random.RandomState(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise HTTPException(400, "No numeric columns found for imputation.")

    # Identify columns with missing values
    missing_cols = [c for c in numeric_cols if df[c].isna().any()]
    if not missing_cols:
        return [df.copy() for _ in range(n_imputations)]

    imputed_datasets = []

    for m in range(n_imputations):
        imp = df.copy()
        # Initial fill: mean + noise
        for col in missing_cols:
            mask = imp[col].isna()
            mean = imp[col].mean()
            std = imp[col].std()
            n_miss = mask.sum()
            imp.loc[mask, col] = mean + rng.normal(0, std * 0.1, n_miss) if std > 0 else mean

        # Iterative chained equations
        for iteration in range(max_iterations):
            for col in missing_cols:
                mask = df[col].isna()
                if mask.sum() == 0:
                    continue

                # Predictors: all other numeric columns
                predictors = [c for c in numeric_cols if c != col]
                if not predictors:
                    continue

                # Observed data for this column
                obs_idx = ~mask
                X_obs = imp.loc[obs_idx, predictors].values
                y_obs = imp.loc[obs_idx, col].values

                # Fit linear regression with noise (Bayesian bootstrap)
                if len(X_obs) < 3 or len(predictors) == 0:
                    continue

                # Add intercept
                X_obs_c = np.column_stack([np.ones(len(X_obs)), X_obs])

                try:
                    # OLS estimate
                    beta_hat = np.linalg.lstsq(X_obs_c, y_obs, rcond=None)[0]
                    y_pred_obs = X_obs_c @ beta_hat
                    residuals = y_obs - y_pred_obs
                    sigma2 = np.var(residuals, ddof=len(beta_hat))

                    # Draw sigma from posterior (scaled inverse chi-squared)
                    df_resid = max(len(y_obs) - len(beta_hat), 1)
                    sigma2_draw = sigma2 * df_resid / rng.chisquare(df_resid)
                    sigma_draw = np.sqrt(max(sigma2_draw, 1e-10))

                    # Draw beta from posterior
                    try:
                        cov = sigma2_draw * np.linalg.inv(X_obs_c.T @ X_obs_c)
                        beta_draw = rng.multivariate_normal(beta_hat, cov)
                    except np.linalg.LinAlgError:
                        beta_draw = beta_hat

                    # Predict missing values + add noise
                    X_miss = imp.loc[mask, predictors].values
                    X_miss_c = np.column_stack([np.ones(len(X_miss)), X_miss])
                    y_pred = X_miss_c @ beta_draw + rng.normal(0, sigma_draw, mask.sum())
                    imp.loc[mask, col] = y_pred

                except (np.linalg.LinAlgError, ValueError):
                    # Fallback: predictive mean matching
                    mean = imp.loc[obs_idx, col].mean()
                    std = imp.loc[obs_idx, col].std()
                    imp.loc[mask, col] = mean + rng.normal(0, std * 0.1, mask.sum()) if std > 0 else mean

        imputed_datasets.append(imp)

    return imputed_datasets


def _pool_estimates(imputed_datasets: list[pd.DataFrame], col: str) -> dict:
    """Pool estimates across imputations using Rubin's rules."""
    means = [df[col].mean() for df in imputed_datasets]
    variances = [df[col].var() for df in imputed_datasets]

    m = len(imputed_datasets)
    pooled_mean = np.mean(means)
    within_var = np.mean(variances)
    between_var = np.var(means, ddof=1)
    total_var = within_var + between_var * (1 + 1/m)
    pooled_se = np.sqrt(total_var)

    # Fraction of missing information
    fmi = (between_var * (1 + 1/m)) / total_var if total_var > 0 else 0

    return {
        "pooled_mean": float(pooled_mean),
        "pooled_se": float(pooled_se),
        "within_variance": float(within_var),
        "between_variance": float(between_var),
        "total_variance": float(total_var),
        "fraction_missing_info": float(fmi),
    }


@router.post("/impute")
def mice_impute(req: MICERequest):
    """Run MICE and return pooled estimates + imputed datasets."""
    if not req.data:
        raise HTTPException(400, "Data cannot be empty.")
    if req.n_imputations < 2:
        raise HTTPException(400, "Need at least 2 imputations.")
    if req.n_imputations > 50:
        raise HTTPException(400, "Maximum 50 imputations.")

    df = pd.DataFrame(req.data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [c for c in numeric_cols if df[c].isna().any()]

    if not missing_cols:
        return {
            "message": "No missing values found in numeric columns.",
            "n_imputations": 0,
            "imputed_data": df.to_dict(orient="records"),
            "pooled_estimates": {},
        }

    # Missing data pattern
    patterns = []
    for col in missing_cols:
        n_miss = int(df[col].isna().sum())
        patterns.append({
            "column": col,
            "n_missing": n_miss,
            "pct_missing": round(n_miss / len(df) * 100, 1),
        })

    imputed_datasets = _mice_impute(df, req.n_imputations, req.max_iterations, req.seed)

    # Pool estimates for each column with missing data
    pooled = {}
    for col in missing_cols:
        pooled[col] = _pool_estimates(imputed_datasets, col)

    # Convergence diagnostic: check variance of means across iterations
    convergence = {}
    for col in missing_cols:
        imp_means = [float(d[col].mean()) for d in imputed_datasets]
        convergence[col] = {
            "imputation_means": imp_means,
            "range": float(max(imp_means) - min(imp_means)),
            "cv": float(np.std(imp_means) / np.mean(imp_means)) if np.mean(imp_means) != 0 else 0,
        }

    # Return first imputation as primary + all pooled estimates
    return {
        "n_imputations": req.n_imputations,
        "max_iterations": req.max_iterations,
        "missing_pattern": patterns,
        "pooled_estimates": pooled,
        "convergence": convergence,
        "imputed_data": imputed_datasets[0].to_dict(orient="records"),
        "all_imputations_summary": {
            col: {
                "means": [float(d[col].mean()) for d in imputed_datasets],
                "stds": [float(d[col].std()) for d in imputed_datasets],
            }
            for col in missing_cols
        },
    }
