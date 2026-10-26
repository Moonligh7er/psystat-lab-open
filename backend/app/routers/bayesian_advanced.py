"""Advanced Bayesian — MCMC sampling, posterior analysis, model comparison."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


class MCMCRequest(BaseModel):
    data: list[float]
    model: str = "normal"  # normal, poisson, binomial
    prior_mean: float = 0.0
    prior_std: float = 10.0
    n_samples: int = 5000
    n_chains: int = 4
    burn_in: int = 1000
    seed: int | None = None


class BayesModelCompareRequest(BaseModel):
    data: list[float]
    models: list[str] = ["normal", "t", "lognormal"]


class PosteriorPredictiveRequest(BaseModel):
    data: list[float]
    model: str = "normal"
    n_samples: int = 1000
    n_simulations: int = 500
    seed: int | None = None


def _mcmc_normal(data: np.ndarray, prior_mean: float, prior_std: float,
                 n_samples: int, burn_in: int, rng: np.random.RandomState) -> dict:
    """Metropolis-Hastings MCMC for normal model with unknown mean and variance."""
    n = len(data)
    data_mean = data.mean()
    data_var = data.var()

    # Initialize
    mu = data_mean
    sigma2 = data_var
    samples_mu = []
    samples_sigma = []
    accepted = 0

    for i in range(n_samples + burn_in):
        # Propose new mu
        mu_prop = rng.normal(mu, 0.5)
        # Log posterior for mu (normal likelihood + normal prior)
        log_lik_curr = -0.5 * n * np.log(sigma2) - 0.5 * np.sum((data - mu)**2) / sigma2
        log_prior_curr = -0.5 * ((mu - prior_mean) / prior_std)**2
        log_lik_prop = -0.5 * n * np.log(sigma2) - 0.5 * np.sum((data - mu_prop)**2) / sigma2
        log_prior_prop = -0.5 * ((mu_prop - prior_mean) / prior_std)**2

        log_ratio = (log_lik_prop + log_prior_prop) - (log_lik_curr + log_prior_curr)
        if np.log(rng.uniform()) < log_ratio:
            mu = mu_prop
            if i >= burn_in:
                accepted += 1

        # Gibbs step for sigma2 (conjugate inverse-gamma)
        alpha_post = n / 2
        beta_post = 0.5 * np.sum((data - mu)**2)
        sigma2 = 1.0 / rng.gamma(alpha_post, 1.0 / max(beta_post, 1e-10))

        if i >= burn_in:
            samples_mu.append(mu)
            samples_sigma.append(np.sqrt(sigma2))

    return {
        "mu_samples": samples_mu,
        "sigma_samples": samples_sigma,
        "acceptance_rate": accepted / n_samples,
    }


def _mcmc_poisson(data: np.ndarray, prior_mean: float, prior_std: float,
                  n_samples: int, burn_in: int, rng: np.random.RandomState) -> dict:
    """MCMC for Poisson model with gamma prior on rate."""
    n = len(data)
    # Gamma prior: shape=prior_mean^2/prior_std^2, rate=prior_mean/prior_std^2
    alpha_prior = max((prior_mean**2) / (prior_std**2), 1.0)
    beta_prior = max(prior_mean / (prior_std**2), 0.1)

    # Conjugate posterior: Gamma(alpha_prior + sum(data), beta_prior + n)
    alpha_post = alpha_prior + data.sum()
    beta_post = beta_prior + n

    samples_lambda = rng.gamma(alpha_post, 1.0 / beta_post, n_samples).tolist()

    return {
        "lambda_samples": samples_lambda,
        "acceptance_rate": 1.0,  # conjugate = direct sampling
    }


def _compute_diagnostics(samples: list[float], n_chains: int) -> dict:
    """Compute MCMC diagnostics: ESS, R-hat, autocorrelation."""
    arr = np.array(samples)
    n = len(arr)

    # Effective sample size (using autocorrelation)
    mean = arr.mean()
    var = arr.var()
    if var == 0:
        return {"ess": n, "rhat": 1.0, "autocorrelation": [1.0]}

    max_lag = min(100, n // 2)
    autocorr = []
    for lag in range(max_lag):
        c = np.mean((arr[:n-lag] - mean) * (arr[lag:] - mean)) / var
        autocorr.append(float(c))
        if c < 0.05:
            break

    # ESS approximation
    ess = n / (1 + 2 * sum(a for a in autocorr[1:] if a > 0))
    ess = max(1, min(ess, n))

    # R-hat (split-chain)
    if n >= 4:
        half = n // 2
        chain1 = arr[:half]
        chain2 = arr[half:2*half]
        w = (chain1.var() + chain2.var()) / 2
        b = half * np.var([chain1.mean(), chain2.mean()])
        var_hat = ((half - 1) / half) * w + (1 / half) * b
        rhat = np.sqrt(var_hat / w) if w > 0 else 1.0
    else:
        rhat = 1.0

    return {
        "ess": float(ess),
        "rhat": float(rhat),
        "autocorrelation": autocorr[:20],
    }


def _hdi(samples: list[float], credible_mass: float = 0.95) -> tuple[float, float]:
    """Highest Density Interval."""
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    ci_size = int(np.ceil(credible_mass * n))
    ci_widths = sorted_samples[ci_size:] - sorted_samples[:n - ci_size]
    best = np.argmin(ci_widths)
    return float(sorted_samples[best]), float(sorted_samples[best + ci_size])


@router.post("/mcmc")
def run_mcmc(req: MCMCRequest):
    """Run MCMC sampling and return posterior samples with diagnostics."""
    data = np.array(req.data)
    if len(data) < 2:
        raise HTTPException(400, "Need at least 2 observations.")

    rng = np.random.RandomState(req.seed)

    # Run multiple chains
    all_chains = []
    for chain_id in range(req.n_chains):
        chain_rng = np.random.RandomState(rng.randint(0, 2**31) + chain_id)

        if req.model == "normal":
            result = _mcmc_normal(data, req.prior_mean, req.prior_std, req.n_samples, req.burn_in, chain_rng)
            all_chains.append(result)
        elif req.model == "poisson":
            result = _mcmc_poisson(data, req.prior_mean, req.prior_std, req.n_samples, req.burn_in, chain_rng)
            all_chains.append(result)
        else:
            raise HTTPException(400, f"Unsupported model: {req.model}")

    # Combine chains for primary parameter
    if req.model == "normal":
        all_mu = []
        all_sigma = []
        for c in all_chains:
            all_mu.extend(c["mu_samples"])
            all_sigma.extend(c["sigma_samples"])

        mu_diag = _compute_diagnostics(all_mu, req.n_chains)
        sigma_diag = _compute_diagnostics(all_sigma, req.n_chains)
        mu_hdi = _hdi(all_mu)
        sigma_hdi = _hdi(all_sigma)

        # Posterior histogram bins for plotting
        mu_arr = np.array(all_mu)
        hist_counts, hist_edges = np.histogram(mu_arr, bins=50)

        return {
            "model": req.model,
            "n_chains": req.n_chains,
            "n_samples_per_chain": req.n_samples,
            "parameters": {
                "mu": {
                    "mean": float(np.mean(all_mu)),
                    "median": float(np.median(all_mu)),
                    "std": float(np.std(all_mu)),
                    "hdi_95": {"lower": mu_hdi[0], "upper": mu_hdi[1]},
                    "diagnostics": mu_diag,
                    "histogram": {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()},
                },
                "sigma": {
                    "mean": float(np.mean(all_sigma)),
                    "median": float(np.median(all_sigma)),
                    "std": float(np.std(all_sigma)),
                    "hdi_95": {"lower": _hdi(all_sigma)[0], "upper": _hdi(all_sigma)[1]},
                    "diagnostics": sigma_diag,
                },
            },
            "convergence": {
                "all_rhat_ok": mu_diag["rhat"] < 1.1 and sigma_diag["rhat"] < 1.1,
                "min_ess": min(mu_diag["ess"], sigma_diag["ess"]),
            },
            "chain_traces": {
                "mu": [c["mu_samples"][:200] for c in all_chains],  # first 200 per chain for trace plot
            },
        }
    else:  # poisson
        all_lambda = []
        for c in all_chains:
            all_lambda.extend(c["lambda_samples"])

        lam_diag = _compute_diagnostics(all_lambda, req.n_chains)
        lam_hdi = _hdi(all_lambda)
        hist_counts, hist_edges = np.histogram(all_lambda, bins=50)

        return {
            "model": req.model,
            "n_chains": req.n_chains,
            "n_samples_per_chain": req.n_samples,
            "parameters": {
                "lambda": {
                    "mean": float(np.mean(all_lambda)),
                    "median": float(np.median(all_lambda)),
                    "std": float(np.std(all_lambda)),
                    "hdi_95": {"lower": lam_hdi[0], "upper": lam_hdi[1]},
                    "diagnostics": lam_diag,
                    "histogram": {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()},
                },
            },
            "convergence": {
                "all_rhat_ok": lam_diag["rhat"] < 1.1,
                "min_ess": lam_diag["ess"],
            },
        }


@router.post("/model-compare")
def compare_models(req: BayesModelCompareRequest):
    """Compare models using BIC/AIC approximation to Bayes factors."""
    data = np.array(req.data)
    n = len(data)
    if n < 3:
        raise HTTPException(400, "Need at least 3 observations.")

    results = []
    for model_name in req.models:
        try:
            if model_name == "normal":
                mu, sigma = sp_stats.norm.fit(data)
                ll = sp_stats.norm.logpdf(data, mu, sigma).sum()
                k = 2
            elif model_name == "t":
                df_t, loc, scale = sp_stats.t.fit(data)
                ll = sp_stats.t.logpdf(data, df_t, loc, scale).sum()
                k = 3
            elif model_name == "lognormal":
                if np.any(data <= 0):
                    results.append({"model": model_name, "error": "Lognormal requires positive data."})
                    continue
                shape, loc, scale = sp_stats.lognorm.fit(data, floc=0)
                ll = sp_stats.lognorm.logpdf(data, shape, loc, scale).sum()
                k = 2
            elif model_name == "exponential":
                if np.any(data < 0):
                    results.append({"model": model_name, "error": "Exponential requires non-negative data."})
                    continue
                loc, scale = sp_stats.expon.fit(data)
                ll = sp_stats.expon.logpdf(data, loc, scale).sum()
                k = 2
            elif model_name == "gamma":
                if np.any(data <= 0):
                    results.append({"model": model_name, "error": "Gamma requires positive data."})
                    continue
                a, loc, scale = sp_stats.gamma.fit(data, floc=0)
                ll = sp_stats.gamma.logpdf(data, a, loc, scale).sum()
                k = 2
            else:
                results.append({"model": model_name, "error": f"Unknown model: {model_name}"})
                continue

            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            results.append({
                "model": model_name,
                "log_likelihood": float(ll),
                "n_parameters": k,
                "aic": float(aic),
                "bic": float(bic),
            })
        except Exception as e:
            results.append({"model": model_name, "error": str(e)})

    # Rank by BIC (lower = better)
    valid = [r for r in results if "error" not in r]
    if valid:
        min_bic = min(r["bic"] for r in valid)
        for r in valid:
            r["delta_bic"] = r["bic"] - min_bic
            # BIC weight (Schwarz weight)
            r["bic_weight"] = float(np.exp(-0.5 * r["delta_bic"]))

        total_weight = sum(r.get("bic_weight", 0) for r in valid)
        for r in valid:
            r["bic_weight"] = r["bic_weight"] / total_weight if total_weight > 0 else 0

        best = min(valid, key=lambda r: r["bic"])
        best["best"] = True

    return {
        "n_observations": n,
        "models": results,
        "best_model": best["model"] if valid else None,
        "interpretation": f"The {best['model']} model provides the best fit (BIC = {best['bic']:.1f})." if valid else "No valid models.",
    }


@router.post("/posterior-predictive")
def posterior_predictive_check(req: PosteriorPredictiveRequest):
    """Generate posterior predictive distribution for model checking."""
    data = np.array(req.data)
    if len(data) < 2:
        raise HTTPException(400, "Need at least 2 observations.")

    rng = np.random.RandomState(req.seed)

    if req.model == "normal":
        # Sample from posterior (conjugate normal-inverse-gamma)
        n = len(data)
        x_bar = data.mean()
        s2 = data.var()

        simulated = []
        test_stats = {"mean": [], "std": [], "min": [], "max": []}

        for _ in range(req.n_simulations):
            # Draw sigma from posterior
            sigma2 = s2 * (n - 1) / rng.chisquare(n - 1)
            sigma = np.sqrt(sigma2)
            # Draw mu from posterior
            mu = rng.normal(x_bar, sigma / np.sqrt(n))
            # Generate replicated data
            y_rep = rng.normal(mu, sigma, req.n_samples)

            test_stats["mean"].append(float(y_rep.mean()))
            test_stats["std"].append(float(y_rep.std()))
            test_stats["min"].append(float(y_rep.min()))
            test_stats["max"].append(float(y_rep.max()))

        # Bayesian p-values
        observed_stats = {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }

        p_values = {}
        for stat_name in test_stats:
            obs = observed_stats[stat_name]
            p = np.mean(np.array(test_stats[stat_name]) >= obs)
            p_values[stat_name] = {
                "observed": obs,
                "posterior_mean": float(np.mean(test_stats[stat_name])),
                "posterior_std": float(np.std(test_stats[stat_name])),
                "bayesian_p": float(p),
                "adequate": 0.05 < p < 0.95,
            }

        all_adequate = all(v["adequate"] for v in p_values.values())

        return {
            "model": req.model,
            "n_simulations": req.n_simulations,
            "test_statistics": test_stats,
            "bayesian_p_values": p_values,
            "model_adequate": all_adequate,
            "interpretation": "Model fits the data well (all Bayesian p-values between 0.05 and 0.95)." if all_adequate
                else "Model may not adequately capture some features of the data.",
        }
    else:
        raise HTTPException(400, f"Posterior predictive check not yet implemented for model: {req.model}")
