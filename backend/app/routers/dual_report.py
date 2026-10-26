from fastapi import APIRouter, HTTPException
import numpy as np
from scipy import optimize, stats
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.routers import novel_dialectical as D

router = APIRouter(tags=["dual-report"])

# ── Models ──────────────────────────────────────────────────────────────────

class DualTestRequest(BaseModel):
    test_type: str  # "t-test", "paired-t", "anova", "correlation", "regression"
    data: Dict[str, List[float]]
    params: Dict[str, Any] = {}  # correlation: {"kappa": 1.0}; regression: {"dv": ...}
    prior_scale: float = 0.707  # Cauchy r, used by the t-tests only

class FrequentistResult(BaseModel):
    test_name: str; statistic: float; df: Optional[float] = None; p_value: float
    ci_lower: float; ci_upper: float; effect_size: float; effect_size_type: str
    effect_ci: List[float]; significant: bool; conclusion: str

class BayesianResult(BaseModel):
    bf10: float; bf01: float; log_bf10: float; evidence_category: str; favors: str
    # Posterior summaries exist only where a real posterior is computed
    # (t-tests: delta under the Cauchy prior; correlation: rho under the
    # stretched-beta prior). ANOVA/regression use the BIC approximation, which
    # yields no posterior, so these are None there.
    posterior_mean: Optional[float] = None; posterior_sd: Optional[float] = None
    credible_interval: Optional[List[float]] = None
    prior_scale: Optional[float] = None  # Cauchy r (t-tests) or kappa (correlation); None for BIC
    prior_description: str; bf_method: str; conclusion: str

class ComparisonResult(BaseModel):
    agree: bool; frequentist_conclusion: str; bayesian_conclusion: str
    integrated_summary: str; apa_text: str

class DualTestResponse(BaseModel):
    test_type: str; frequentist: FrequentistResult
    bayesian: BayesianResult; comparison: ComparisonResult

class APAReportRequest(BaseModel):
    dual_result: DualTestResponse

class APAReportResponse(BaseModel):
    apa_paragraph: str

# ── Helpers ─────────────────────────────────────────────────────────────────

def _evidence_category(bf10: float) -> str:
    """Strength of evidence for whichever hypothesis BF favours (Lee & Wagenmakers, 2013)."""
    bf = max(bf10, 1.0 / bf10) if bf10 > 0 else float("inf")
    if bf > 100: return "Extreme"
    if bf > 30: return "Very Strong"
    if bf > 10: return "Strong"
    if bf > 3: return "Moderate"
    return "Anecdotal"

def _bf_bic(bic_null: float, bic_alt: float) -> float:
    """BIC approximation to BF10 (Wagenmakers, 2007): exp((BIC0 - BIC1) / 2).
    Implies a unit-information prior; no user prior scale enters."""
    diff = max(min((bic_null - bic_alt) / 2.0, 500), -500)
    return float(np.exp(diff))

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else 0.0

def _ncf_eta2_ci(f_stat: float, df1: float, df2: float, n: int, level: float = 0.95) -> List[float]:
    """CI for eta^2 / R^2 by inverting the noncentral F for its noncentrality
    lambda, then mapping lambda -> lambda / (lambda + N) (Steiger, 2004).
    Exact for fixed-effects ANOVA; approximate for regression with random predictors."""
    tail = (1 - level) / 2
    def lam_for(p):
        # The F cdf decreases in lambda; if lambda = 0 already gives cdf <= p, the bound is 0.
        f = lambda lam: stats.ncf.cdf(f_stat, df1, df2, lam) - p
        if f(0.0) <= 0:
            return 0.0
        hi = max(10.0, f_stat * df1 * 4)
        while f(hi) > 0:
            hi *= 2
        return optimize.brentq(f, 0.0, hi)
    lam_lo, lam_hi = lam_for(1 - tail), lam_for(tail)
    return [lam_lo / (lam_lo + n), lam_hi / (lam_hi + n)]

def _p(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"

def _bf(bf: float) -> str:
    return f"{bf:.2f}" if 0.01 <= bf < 1000 else f"{bf:.2e}"

def _freq(name, stat, df, p, ci, es, es_type, es_ci, conclusion):
    return FrequentistResult(test_name=name, statistic=round(stat, 6), df=df,
        p_value=float(p), ci_lower=round(ci[0], 6), ci_upper=round(ci[1], 6),
        effect_size=round(es, 6), effect_size_type=es_type,
        effect_ci=[round(v, 6) for v in es_ci], significant=p < 0.05, conclusion=conclusion)

_T_PRIOR = "JZS: Cauchy(0, r) prior on delta (Rouder et al., 2009)"
_BIC_PRIOR = "BIC approximation (Wagenmakers, 2007), unit-information prior; prior_scale not used"

def _t_bayes(t, n_eff, df, r):
    bf10 = D.jzs_bf10_t(t, n_eff, df, r)
    mean, sd, lo, hi = D.posterior_delta(t, n_eff, df, r)
    return dict(bf10=bf10, post=(mean, sd, [lo, hi]), prior_scale=r,
                prior_description=f"{_T_PRIOR}, r = {r}", bf_method="JZS (numerical integration)")

def _bic_bayes(bic_null, bic_alt):
    return dict(bf10=_bf_bic(bic_null, bic_alt), post=None, prior_scale=None,
                prior_description=_BIC_PRIOR, bf_method="BIC approximation")

# ── Test runners ────────────────────────────────────────────────────────────

def _run_ttest(data, params, r):
    keys = list(data.keys()); x, y = np.array(data[keys[0]], float), np.array(data[keys[1]], float)
    n1, n2 = len(x), len(y)
    t_stat, p = stats.ttest_ind(x, y); df = n1+n2-2; d = _cohens_d(x, y)
    n_eff = n1*n2/(n1+n2)
    # Pooled SE, consistent with the Student (equal-variance) t-test above.
    sp2 = ((n1-1)*np.var(x, ddof=1) + (n2-1)*np.var(y, ddof=1)) / df
    diff = np.mean(x)-np.mean(y); se = np.sqrt(sp2*(1/n1 + 1/n2)); tc = stats.t.ppf(0.975, df)
    ci = [float(diff-tc*se), float(diff+tc*se)]
    freq = _freq("Independent Samples t-test", float(t_stat), float(df), float(p),
                  ci, d, "Cohen's d", D._d_ci(float(t_stat), df, n_eff),
                  f"t({df}) = {t_stat:.3f}, {_p(p)}, d = {d:.3f}")
    return freq, _t_bayes(float(t_stat), n_eff, df, r)

def _run_paired_t(data, params, r):
    keys = list(data.keys()); x, y = np.array(data[keys[0]], float), np.array(data[keys[1]], float)
    diffs = x - y; t_stat, p = stats.ttest_rel(x, y); n = len(x); df = n-1
    sd = np.std(diffs, ddof=1); d = float(np.mean(diffs)/sd) if sd > 0 else 0.0
    se = sd/np.sqrt(n); m = np.mean(diffs); tc = stats.t.ppf(0.975, df)
    ci = [float(m-tc*se), float(m+tc*se)]
    freq = _freq("Paired Samples t-test", float(t_stat), float(df), float(p),
                  ci, d, "Cohen's d (paired)", D._d_ci(float(t_stat), df, n),
                  f"t({df}) = {t_stat:.3f}, {_p(p)}, d = {d:.3f}")
    return freq, _t_bayes(float(t_stat), n, df, r)

def _run_anova(data, params, r):
    groups = [np.array(v, float) for v in data.values()]
    f_stat, p = stats.f_oneway(*groups); k = len(groups)
    ns = [len(g) for g in groups]; N = sum(ns); df_b, df_w = k-1, N-k
    eta2 = float(f_stat*df_b / (f_stat*df_b + df_w))
    ci = _ncf_eta2_ci(float(f_stat), df_b, df_w, N)
    freq = _freq("One-Way ANOVA", float(f_stat), float(df_b), float(p),
                  ci, eta2, "eta-squared", ci,
                  f"F({df_b}, {df_w}) = {f_stat:.3f}, {_p(p)}, eta2 = {eta2:.3f}")
    grand = np.concatenate(groups); ss_tot = np.sum((grand-np.mean(grand))**2)
    ss_w = sum(np.sum((g-np.mean(g))**2) for g in groups)
    return freq, _bic_bayes(N*np.log(ss_tot/N)+np.log(N), N*np.log(ss_w/N)+k*np.log(N))

def _run_correlation(data, params, r_prior):
    keys = list(data.keys()); x, y = np.array(data[keys[0]], float), np.array(data[keys[1]], float)
    r_val, p = stats.pearsonr(x, y); n = len(x)
    se = 1/np.sqrt(n-3) if n > 3 else 0.1; z_r = np.arctanh(min(max(r_val, -0.9999), 0.9999))
    ci = [float(np.tanh(z_r-1.96*se)), float(np.tanh(z_r+1.96*se))]
    freq = _freq("Pearson Correlation", float(r_val), float(n-2), float(p),
                  ci, float(r_val), "Pearson r", ci,
                  f"r({n-2}) = {r_val:.3f}, {_p(p)}")
    # The correlation prior is a stretched Beta(1/kappa, 1/kappa) on rho, not a
    # Cauchy on delta, so prior_scale (a Cauchy r) does not apply here.
    kappa = float(params.get("kappa", 1.0))
    bf10 = D.ly_bf10_r(float(r_val), n, kappa)
    mean, sd, lo, hi = D.posterior_rho(float(r_val), n, kappa)
    desc = (f"stretched Beta(1/kappa, 1/kappa) prior on rho, kappa = {kappa:g}"
            f"{' (uniform)' if kappa == 1 else ''} (Ly et al., 2016)")
    return freq, dict(bf10=bf10, post=(mean, sd, [lo, hi]), prior_scale=kappa,
                      prior_description=desc, bf_method="Ly, Verhagen & Wagenmakers (2016)")

def _run_regression(data, params, r):
    dv = params.get("dv", list(data.keys())[-1])
    ivs = [k for k in data.keys() if k != dv]
    y = np.array(data[dv], float); X = np.column_stack([np.array(data[iv], float) for iv in ivs])
    n, k = X.shape; X_c = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_c, y, rcond=None)[0]; y_hat = X_c @ beta
    ss_res = np.sum((y-y_hat)**2); ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1-ss_res/ss_tot if ss_tot > 0 else 0
    f_stat = (r2/k)/((1-r2)/(n-k-1)) if (1-r2) > 0 and n > k+1 else 0
    p = stats.f.sf(f_stat, k, n-k-1) if f_stat > 0 else 1.0
    ci = _ncf_eta2_ci(float(f_stat), k, n-k-1, n) if f_stat > 0 else [0.0, 0.0]
    freq = _freq("Multiple Regression (F-test)", float(f_stat), float(k), float(p),
                  ci, float(r2), "R-squared", ci,
                  f"F({k}, {n-k-1}) = {f_stat:.3f}, {_p(p)}, R2 = {r2:.3f}")
    return freq, _bic_bayes(n*np.log(ss_tot/n)+np.log(n), n*np.log(ss_res/n)+(k+1)*np.log(n))

def _post_label(test_type: str) -> str:
    return "rho" if test_type == "correlation" else "delta"

_RUNNERS = {"t-test": _run_ttest, "paired-t": _run_paired_t, "anova": _run_anova,
            "correlation": _run_correlation, "regression": _run_regression}

# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/dual-test", response_model=DualTestResponse)
async def dual_test(req: DualTestRequest):
    """Run both frequentist and Bayesian analyses on the same test."""
    runner = _RUNNERS.get(req.test_type)
    if runner is None:
        raise HTTPException(422, f"Unsupported test_type: {req.test_type}. Choose from {list(_RUNNERS.keys())}")
    need = {"t-test": 2, "paired-t": 2, "correlation": 2, "anova": 2, "regression": 2}[req.test_type]
    if len(req.data) < need or any(len(v) < 3 for v in req.data.values()):
        raise HTTPException(422, f"{req.test_type} needs at least {need} variables/groups with 3 or more values each.")
    if req.test_type in ("paired-t", "correlation", "regression") and len({len(v) for v in req.data.values()}) > 1:
        raise HTTPException(422, "All variables must have the same number of values for this test.")
    freq, b = runner(req.data, req.params, req.prior_scale)
    bf10 = b["bf10"]
    bf01 = 1.0/bf10 if bf10 > 0 else 1e300
    log_bf10 = float(np.log(bf10)) if bf10 > 0 else -700
    ev_cat = _evidence_category(bf10); favors = "H1" if bf10 >= 1 else "H0"
    post_mean = post_sd = cri = None
    if b["post"] is not None:
        post_mean, post_sd, cri = b["post"]
        post_mean, post_sd = round(post_mean, 6), round(post_sd, 6)
        cri = [round(v, 6) for v in cri]  # equal-tailed 95% credible interval
    bay_conclusion = f"BF10 = {_bf(bf10)}, {ev_cat.lower()} evidence for {favors}"
    bayes = BayesianResult(bf10=round(bf10, 6), bf01=round(bf01, 6), log_bf10=round(log_bf10, 6),
        evidence_category=ev_cat, favors=favors, posterior_mean=post_mean,
        posterior_sd=post_sd, credible_interval=cri, prior_scale=b["prior_scale"],
        prior_description=b["prior_description"], bf_method=b["bf_method"],
        conclusion=bay_conclusion)
    freq_dir = "significant" if freq.significant else "non-significant"
    bay_dir = "H1" if bf10 > 1 else "H0"
    agree = (freq.significant and bf10 > 3) or (not freq.significant and bf10 < 1/3)
    if agree:
        direction = "for" if freq.significant else "against"
        summary = (f"Both frequentist ({_p(freq.p_value)}) and Bayesian (BF10 = {_bf(bf10)}) "
                   f"analyses provide converging evidence {direction} the effect.")
    else:
        summary = (f"Frequentist analysis yielded {_p(freq.p_value)} ({freq_dir}), "
                   f"while Bayesian gave BF10 = {_bf(bf10)} ({ev_cat.lower()} for {bay_dir}), "
                   f"suggesting the paradigms diverge.")
    apa = (f"{freq.conclusion}. Bayesian analysis ({bayes.prior_description}) yielded "
           f"BF10 = {_bf(bf10)}, indicating {ev_cat.lower()} evidence "
           f"{'for H1' if favors == 'H1' else 'for H0'}"
           + (f" (posterior mean {_post_label(req.test_type)} = {post_mean:.3f}, "
              f"95% CrI [{cri[0]:.3f}, {cri[1]:.3f}])" if cri is not None else "")
           + ".")
    comparison = ComparisonResult(agree=agree,
        frequentist_conclusion=f"The effect is {freq_dir} at alpha = .05 ({_p(freq.p_value)}).",
        bayesian_conclusion=bay_conclusion, integrated_summary=summary, apa_text=apa)
    return DualTestResponse(test_type=req.test_type, frequentist=freq,
                            bayesian=bayes, comparison=comparison)

@router.post("/dual-report-apa", response_model=APAReportResponse)
async def dual_report_apa(req: APAReportRequest):
    """Generate formatted APA paragraph per Keysers et al. (2020) recommendations."""
    f, b, c = req.dual_result.frequentist, req.dual_result.bayesian, req.dual_result.comparison
    bayes_line = (f"A complementary Bayesian analysis ({b.prior_description}) following "
                  f"Keysers, Gazzola, and Wagenmakers (2020) yielded BF10 = {_bf(b.bf10)} "
                  f"(BF01 = {_bf(b.bf01)}), constituting {b.evidence_category.lower()} evidence "
                  f"{'in favor of' if b.favors == 'H1' else 'against'} the alternative hypothesis.")
    post_line = ""
    if b.credible_interval is not None and b.posterior_mean is not None:
        post_line = (f" The posterior mean of {_post_label(req.dual_result.test_type)} was "
                     f"{b.posterior_mean:.3f} (SD = {b.posterior_sd:.3f}), 95% CrI "
                     f"[{b.credible_interval[0]:.3f}, {b.credible_interval[1]:.3f}].")
    integ = ("Both approaches converge, strengthening confidence in the result." if c.agree
             else "The paradigms diverge, suggesting caution in interpretation.")
    return APAReportResponse(apa_paragraph=f"{f.conclusion} {bayes_line}{post_line} {integ}")
