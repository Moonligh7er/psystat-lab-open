"""
Frequentist–Bayesian Comparison (formerly "Dialectical Inference")
=================================================================
Runs the same comparison through a classical test and a default Bayesian
test, then reports where the two frameworks agree and where they don't.

This is not a new method. Reporting both and classifying their agreement is
established practice: Wetzels et al. (2011) classified 855 published t-tests
exactly this way, and JASP reports both side by side. What this tool adds is
convenience: the exact computations, a prior-sensitivity sweep, and a plain
summary of the agreement.

Computations (all exact up to numerical integration):
  * t-tests: JZS Bayes factor (Rouder et al., 2009), Cauchy prior on the
    standardised effect delta with scale r. Posterior for delta under the
    same prior, from the noncentral-t likelihood of the observed t.
  * Correlation: default Bayes factor of Ly, Verhagen & Wagenmakers (2016),
    stretched-beta prior on rho with width kappa (kappa = 1 is uniform).
    Posterior for rho under the same prior, from the exact sampling
    distribution of r (Hotelling, 1953).
  * Frequentist CIs: Cohen's d CI by inverting the noncentral t
    distribution; correlation CI by Fisher's z transformation.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import integrate, optimize, special
from scipy import stats as scipy_stats

router = APIRouter()

DEFAULT_T_SCALES = [0.354, 0.5, 0.707, 1.0, 1.414]
DEFAULT_KAPPAS = [0.25, 0.5, 1.0, 1.5, 2.0]


# ── Request / Response ────────────────────────────────────────────────

class DialecticalRequest(BaseModel):
    group1: List[float]
    group2: Optional[List[float]] = None  # None for one-sample tests
    test_type: str = "two_sample"  # two_sample, paired, one_sample, correlation
    alpha: float = 0.05
    # Cauchy scales r for t-tests, or stretched-beta widths kappa for a
    # correlation. Defaults depend on the test type when omitted.
    prior_scales: Optional[List[float]] = None
    mu0: float = 0  # null value for one-sample


class FrameworkResult(BaseModel):
    test_name: str
    statistic: float
    df: float
    p_value: float
    effect_label: str  # "d" or "r"
    effect_size: float
    ci_lower: float  # 95% CI for the effect size
    ci_upper: float
    conclusion: str


class BayesianResult(BaseModel):
    bf10: float
    bf01: float
    prior: str
    evidence_category: str
    posterior_mean: float  # on the effect-size scale (delta or rho)
    posterior_sd: float
    credible_lower: float  # 95% equal-tailed credible interval
    credible_upper: float
    conclusion: str
    robustness: dict  # BF10 at each prior setting


class SensitivityResult(BaseModel):
    prior_scale: float
    bf10: float
    evidence_category: str
    agrees_with_frequentist: bool


class SynthesisResult(BaseModel):
    agreement_type: str  # "full_agreement", "partial_agreement", "disagreement"
    agreement_detail: str
    synthesized_claim: str
    direction_agreed: bool
    magnitude_agreed: bool
    significance_agreed: bool
    boundary_conditions: List[str]
    confidence_in_synthesis: float  # share of the three checks that agree (0, 1/3, 2/3, 1)


class DialecticalResponse(BaseModel):
    frequentist: FrameworkResult
    bayesian: BayesianResult
    synthesis: SynthesisResult
    sensitivity: List[SensitivityResult]
    prior_parameter: str  # what prior_scale means in `sensitivity`
    interpretation: str


# ── Bayes factors ─────────────────────────────────────────────────────

def _evidence_category(bf: float) -> str:
    # Lee & Wagenmakers (2013) labels, adapted from Jeffreys (1961).
    if bf < 1/100: return "Extreme evidence for H0"
    if bf < 1/30: return "Very strong evidence for H0"
    if bf < 1/10: return "Strong evidence for H0"
    if bf < 1/3: return "Moderate evidence for H0"
    if bf < 1: return "Anecdotal evidence for H0"
    if bf < 3: return "Anecdotal evidence for H1"
    if bf < 10: return "Moderate evidence for H1"
    if bf < 30: return "Strong evidence for H1"
    if bf < 100: return "Very strong evidence for H1"
    return "Extreme evidence for H1"


BF_CAP = 1e300  # Bayes factors beyond this are reported as 1e300 (or 1e-300)


def _exp_capped(log_x: float) -> float:
    return math.exp(max(min(log_x, math.log(BF_CAP)), -math.log(BF_CAP)))


def _integrate_log(logf) -> float:
    """log ∫_0^∞ exp(logf(g)) dg, rescaled around the integrand's peak so very
    large or very small Bayes factors don't overflow."""
    xs = np.geomspace(1e-8, 1e8, 600)
    peak_x = float(xs[int(np.argmax([logf(float(x)) for x in xs]))])
    peak = logf(peak_x)
    f = lambda g: math.exp(logf(g) - peak)
    val = (integrate.quad(f, 0, peak_x, limit=200)[0]
           + integrate.quad(f, peak_x, np.inf, limit=200)[0])
    return peak + math.log(val)


def jzs_bf10_t(t: float, n_eff: float, df: float, r: float) -> float:
    """JZS Bayes factor for a t statistic (Rouder et al., 2009, eq. 1).

    delta ~ Cauchy(0, r), written as delta | g ~ N(0, g r^2), g ~ InvGamma(1/2, 1/2).
    n_eff = n (one-sample / paired) or n1*n2/(n1+n2) (two-sample)."""
    log_null = -(df + 1) / 2 * math.log1p(t * t / df)

    def logf(g):
        a = 1 + n_eff * g * r * r
        return (-0.5 * math.log(a)
                - (df + 1) / 2 * math.log1p(t * t / (a * df))
                - 0.5 * math.log(2 * math.pi) - 1.5 * math.log(g) - 1 / (2 * g)
                - log_null)

    return _exp_capped(_integrate_log(logf))


RHO_GRID = np.linspace(-1 + 1e-9, 1 - 1e-9, 20001)


def _log_stretched_beta(rho: np.ndarray, kappa: float) -> np.ndarray:
    """Normalised log-density of the stretched Beta(1/kappa, 1/kappa) prior on (-1, 1)."""
    a = 1 / kappa
    return (a - 1) * (np.log1p(rho) + np.log1p(-rho)) - (2 * a - 1) * math.log(2) - special.betaln(a, a)


def ly_bf10_r(r: float, n: int, kappa: float) -> float:
    """Default correlation Bayes factor of Ly, Verhagen & Wagenmakers (2016):
    BF10 = ∫ p(r | rho) / p(r | 0) · prior(rho) drho with a stretched
    Beta(1/kappa, 1/kappa) prior. Integrated numerically in log space so it
    stays finite for large n (the closed-form hypergeometric overflows)."""
    log_ratio = _log_lik_rho(RHO_GRID, r, n) - _log_lik_rho(np.array([0.0]), r, n)[0]
    log_integrand = log_ratio + _log_stretched_beta(RHO_GRID, kappa)
    peak = float(log_integrand.max())
    val = float(integrate.trapezoid(np.exp(log_integrand - peak), RHO_GRID))
    return _exp_capped(peak + math.log(val))


# ── Posteriors ────────────────────────────────────────────────────────

def _grid_summary(grid: np.ndarray, log_post: np.ndarray):
    """Mean, SD and 95% equal-tailed interval of a density known up to a constant on a grid."""
    w = np.exp(log_post - log_post.max())
    cdf = integrate.cumulative_trapezoid(w, grid, initial=0)
    total = cdf[-1]
    w, cdf = w / total, cdf / total
    mean = float(integrate.trapezoid(grid * w, grid))
    sd = float(math.sqrt(max(0.0, integrate.trapezoid((grid - mean) ** 2 * w, grid))))
    lo = float(np.interp(0.025, cdf, grid))
    hi = float(np.interp(0.975, cdf, grid))
    return mean, sd, lo, hi


def nct_logpdf(t: float, df: float, ncp: np.ndarray, n_nodes: int = 2500) -> np.ndarray:
    """log density of the noncentral t at t for each ncp, up to a constant in ncp.

    Uses t = (Z + ncp) / S with S = sqrt(V/df), V ~ chi2(df):
        f(t; ncp) = ∫ phi(t·s − ncp) · s · f_S(s) ds,
    integrated by the trapezoid rule in log space on an s-grid that spans
    both the bulk of f_S and the peak of phi (s = ncp/t). scipy's
    nct.logpdf (1.14) returns NaN/+inf in the tails, which corrupts a
    posterior grid."""
    ncp = np.asarray(ncp, dtype=float)
    q_lo, q_hi = np.sqrt(scipy_stats.chi2.ppf([1e-15, 1 - 1e-15], df) / df)
    s_hi = q_hi
    if t != 0:
        peaks = ncp / t
        s_hi = max(s_hi, float(peaks.max()) + 8 / abs(t))
    s = np.linspace(max(q_lo * 0.5, 1e-12), s_hi, n_nodes)
    log_fs = scipy_stats.chi2.logpdf(df * s * s, df) + np.log(2 * df * s)  # density of S
    z = t * s[None, :] - ncp[:, None]
    log_w = np.log(np.gradient(s))  # trapezoid-equivalent weights on a uniform grid
    return special.logsumexp(-0.5 * z * z + (np.log(s) + log_fs + log_w)[None, :], axis=1)


def posterior_delta(t: float, n_eff: float, df: float, r: float):
    """Posterior of the standardised effect delta under a Cauchy(0, r) prior,
    using the noncentral-t likelihood t | delta ~ t_df(ncp = delta * sqrt(n_eff))."""
    d = t / math.sqrt(n_eff)
    se = math.sqrt(1 / n_eff + d * d / (2 * df))
    # Span the likelihood (±12 SE) and the prior's centre, so shrinkage toward 0 is captured.
    grid = np.linspace(min(d - 12 * se, -3 * r), max(d + 12 * se, 3 * r), 1201)
    log_lik = nct_logpdf(t, df, grid * math.sqrt(n_eff))
    log_prior = scipy_stats.cauchy.logpdf(grid, 0, r)
    return _grid_summary(grid, log_lik + log_prior)


def _log_lik_rho(rho: np.ndarray, r: float, n: int) -> np.ndarray:
    """Exact log-likelihood of rho given sample r (Hotelling, 1953), up to a constant."""
    return ((n - 1) / 2 * np.log1p(-rho * rho)
            - (n - 1.5) * np.log1p(-rho * r)
            + np.log(special.hyp2f1(0.5, 0.5, n - 0.5, (1 + rho * r) / 2)))


def posterior_rho(r: float, n: int, kappa: float):
    """Posterior of rho under the same stretched-beta prior as the Bayes factor."""
    return _grid_summary(RHO_GRID, _log_lik_rho(RHO_GRID, r, n) + _log_stretched_beta(RHO_GRID, kappa))


# ── Frequentist side ──────────────────────────────────────────────────

def _d_ci(t: float, df: float, n_eff: float, level: float = 0.95):
    """CI for Cohen's d by inverting the noncentral t (Steiger & Fouladi, 1997)."""
    tail = (1 - level) / 2
    def ncp_for(p):
        f = lambda ncp: scipy_stats.nct.cdf(t, df, ncp) - p
        lo, hi = t - 10 - abs(t), t + 10 + abs(t)
        return optimize.brentq(f, lo, hi)
    return ncp_for(1 - tail) / math.sqrt(n_eff), ncp_for(tail) / math.sqrt(n_eff)


def _cohen_label(es: float, label: str) -> str:
    a = abs(es)
    cuts = (0.2, 0.5, 0.8) if label == "d" else (0.1, 0.3, 0.5)  # Cohen (1988)
    if a < cuts[0]: return "negligible"
    if a < cuts[1]: return "small"
    if a < cuts[2]: return "medium"
    return "large"


def _design(req: DialecticalRequest):
    """Returns (test name, statistic, df, p, effect label, effect, CI, n_eff, n)."""
    g1 = np.asarray(req.group1, dtype=float)

    if req.test_type == "correlation":
        g2 = np.asarray(req.group2, dtype=float)
        if len(g1) != len(g2):
            raise HTTPException(422, "Correlation requires equal-length variables.")
        n = len(g1)
        if n < 4:
            raise HTTPException(422, "Correlation needs at least 4 pairs.")
        r, p = scipy_stats.pearsonr(g1, g2)
        r = float(r)
        if abs(r) >= 1:
            raise HTTPException(422, "The correlation is exactly ±1; the tests are undefined.")
        t = r * math.sqrt((n - 2) / (1 - r * r))
        z, se_z = math.atanh(r), 1 / math.sqrt(n - 3)
        ci = (math.tanh(z - 1.96 * se_z), math.tanh(z + 1.96 * se_z))
        return "Pearson correlation", t, n - 2, float(p), "r", r, ci, None, n

    if req.test_type in ("one_sample", "paired"):
        if req.test_type == "paired":
            g2 = np.asarray(req.group2, dtype=float)
            if len(g1) != len(g2):
                raise HTTPException(422, "Paired test requires equal-length groups.")
            x, name = g1 - g2, "Paired t-test"
            t, p = scipy_stats.ttest_rel(g1, g2)
        else:
            x, name = g1 - req.mu0, "One-sample t-test"
            t, p = scipy_stats.ttest_1samp(g1, req.mu0)
        n = len(x)
        if np.std(x, ddof=1) == 0:
            raise HTTPException(422, "The differences have zero variance; the test is undefined.")
        n_eff, df = float(n), n - 1
    else:
        g2 = np.asarray(req.group2, dtype=float)
        name = "Independent-samples t-test (Student)"
        t, p = scipy_stats.ttest_ind(g1, g2)  # pooled variance, matching the JZS model
        n1, n2 = len(g1), len(g2)
        if np.var(g1, ddof=1) + np.var(g2, ddof=1) == 0:
            raise HTTPException(422, "Both groups have zero variance; the test is undefined.")
        n, n_eff, df = n1 + n2, n1 * n2 / (n1 + n2), n1 + n2 - 2

    t, p = float(t), float(p)
    d = t / math.sqrt(n_eff)  # equals Cohen's d (pooled SD for two samples)
    return name, t, df, p, "d", d, _d_ci(t, df, n_eff), n_eff, n


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/analyze")
async def dialectical_inference(req: DialecticalRequest):
    if req.test_type not in ("two_sample", "paired", "one_sample", "correlation"):
        raise HTTPException(422, "test_type must be two_sample, paired, one_sample or correlation.")
    if len(req.group1) < 3:
        raise HTTPException(422, "Need at least 3 observations in group 1.")
    if req.test_type != "one_sample" and (not req.group2 or len(req.group2) < 3):
        raise HTTPException(422, "Need at least 3 observations in group 2.")
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")

    name, t, df, p, label, es, ci, n_eff, n = _design(req)
    is_corr = label == "r"
    scales = req.prior_scales or (DEFAULT_KAPPAS if is_corr else DEFAULT_T_SCALES)
    if any(s <= 0 for s in scales):
        raise HTTPException(422, "Prior scales must be positive.")
    default_scale = 1.0 if is_corr else 0.707

    def bf_at(s: float) -> float:
        return ly_bf10_r(es, n, s) if is_corr else jzs_bf10_t(t, n_eff, df, s)

    freq_sig = p < req.alpha
    freq = FrameworkResult(
        test_name=name, statistic=round(t, 6), df=round(df, 4), p_value=p,
        effect_label=label, effect_size=round(es, 6),
        ci_lower=round(ci[0], 6), ci_upper=round(ci[1], 6),
        conclusion=(f"{'Significant' if freq_sig else 'Not significant'} at α = {req.alpha} "
                    f"(p = {p:.4g}). {label} = {es:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]."),
    )

    bf10 = bf_at(default_scale)
    bf01 = 1 / bf10 if bf10 > 0 else float("inf")
    if is_corr:
        post = posterior_rho(es, n, default_scale)
        prior_desc = "Stretched Beta(1, 1) on ρ (uniform), Ly et al. (2016)"
    else:
        post = posterior_delta(t, n_eff, df, default_scale)
        prior_desc = "Cauchy(0, 0.707) on δ (JZS), Rouder et al. (2009)"
    robustness = {str(s): round(bf_at(s), 6) for s in scales}
    ev = _evidence_category(bf10)
    bayes = BayesianResult(
        bf10=bf10, bf01=bf01, prior=prior_desc, evidence_category=ev,
        posterior_mean=round(post[0], 6), posterior_sd=round(post[1], 6),
        credible_lower=round(post[2], 6), credible_upper=round(post[3], 6),
        conclusion=(f"{ev}. BF₁₀ = {bf10:.4g}: the data are {bf10:.3g}× as likely under H1 as under H0."
                    if bf10 >= 1 else
                    f"{ev}. BF₀₁ = {bf01:.4g}: the data are {bf01:.3g}× as likely under H0 as under H1."),
        robustness=robustness,
    )

    sensitivity = []
    for s in scales:
        bf = robustness[str(s)]
        sensitivity.append(SensitivityResult(
            prior_scale=s, bf10=bf, evidence_category=_evidence_category(bf),
            agrees_with_frequentist=(freq_sig and bf > 3) or (not freq_sig and bf < 1 / 3),
        ))

    synthesis = _synthesize(freq, bayes, req.alpha, is_corr)
    prior_param = "κ (stretched-beta width)" if is_corr else "r (Cauchy scale)"
    interpretation = (
        f"Frequentist: {freq.conclusion}\n\n"
        f"Bayesian ({prior_desc}): {bayes.conclusion}\n\n"
        f"Comparison: {synthesis.synthesized_claim}"
    )
    return DialecticalResponse(
        frequentist=freq, bayesian=bayes, synthesis=synthesis,
        sensitivity=sensitivity, prior_parameter=prior_param, interpretation=interpretation,
    )


def _synthesize(freq: FrameworkResult, bayes: BayesianResult, alpha: float, is_corr: bool) -> SynthesisResult:
    freq_sig = freq.p_value < alpha
    h1 = bayes.bf10 > 3
    h0 = bayes.bf10 < 1 / 3

    def sign(x): return (x > 0) - (x < 0)
    direction_agreed = sign(freq.effect_size) == sign(bayes.posterior_mean)
    magnitude_agreed = (min(freq.ci_upper, bayes.credible_upper)
                        > max(freq.ci_lower, bayes.credible_lower))

    if freq_sig and h1:
        kind, ok = "full_agreement", True
        detail = "The test rejects H0 and the Bayes factor favours H1 (BF₁₀ > 3)."
    elif not freq_sig and h0:
        kind, ok = "full_agreement", True
        detail = "The test does not reject H0 and the Bayes factor favours H0 (BF₁₀ < 1/3)."
    elif freq_sig and h0:
        kind, ok = "disagreement", False
        detail = ("The test rejects H0, yet the Bayes factor favours H0. This is Lindley's paradox: "
                  "with large samples a small p-value can sit alongside evidence for the null.")
    elif not freq_sig and h1:
        kind, ok = "disagreement", False
        detail = ("The test does not reject H0, yet the Bayes factor favours H1. Check the prior "
                  "sensitivity below; this is unusual and often depends on the prior scale.")
    elif freq_sig:
        kind, ok = "partial_agreement", False
        detail = ("The test rejects H0 but the Bayes factor is inconclusive (1/3 ≤ BF₁₀ ≤ 3). "
                  "p-values just under α often correspond to weak Bayesian evidence.")
    else:
        kind, ok = "partial_agreement", False
        detail = ("Neither framework gives clear evidence: the test is not significant and the "
                  "Bayes factor is inconclusive. More data would be needed to decide.")

    boundaries = [
        (f"The test stops being significant for α below p = {freq.p_value:.4g}." if freq_sig
         else f"The test would be significant only for α above p = {freq.p_value:.4g}.")
    ]
    favour_h1 = sorted(float(k) for k, v in bayes.robustness.items() if v > 3)
    favour_h0 = sorted(float(k) for k, v in bayes.robustness.items() if v < 1 / 3)
    param = "κ" if is_corr else "r"
    n_scales = len(bayes.robustness)
    if favour_h1 and favour_h0:
        boundaries.append(f"Prior-sensitive: BF₁₀ > 3 at {param} = {', '.join(map(str, favour_h1))} "
                          f"but BF₁₀ < 1/3 at {param} = {', '.join(map(str, favour_h0))}.")
    elif len(favour_h1) == n_scales:
        boundaries.append(f"BF₁₀ > 3 at every prior setting tested ({param} = {min(favour_h1)}–{max(favour_h1)}).")
    elif len(favour_h0) == n_scales:
        boundaries.append(f"BF₁₀ < 1/3 at every prior setting tested ({param} = {min(favour_h0)}–{max(favour_h0)}).")
    elif favour_h1 or favour_h0:
        which = favour_h1 or favour_h0
        side = "H1 (BF₁₀ > 3)" if favour_h1 else "H0 (BF₁₀ < 1/3)"
        boundaries.append(f"The Bayes factor favours {side} only at {param} = {', '.join(map(str, which))}; "
                          f"it is inconclusive at the other settings.")
    else:
        boundaries.append("The Bayes factor is inconclusive at every prior setting tested.")

    label = freq.effect_label
    size = _cohen_label(freq.effect_size, label)
    if kind == "full_agreement" and freq_sig and direction_agreed:
        claim = (f"Both frameworks support a {'positive' if freq.effect_size > 0 else 'negative'} effect "
                 f"({label} = {freq.effect_size:.3f}, a {size} effect by Cohen's conventions; "
                 f"BF₁₀ = {bayes.bf10:.3g}).")
    elif kind == "full_agreement" and not freq_sig:
        claim = (f"Both frameworks point away from an effect: p = {freq.p_value:.4g} and BF₀₁ = {bayes.bf01:.3g}. "
                 f"The Bayes factor favours H0 over this particular alternative; it does not show the effect is exactly zero.")
    elif kind == "partial_agreement":
        claim = (f"The frameworks partly agree. p = {freq.p_value:.4g}, BF₁₀ = {bayes.bf10:.3g}. "
                 f"The evidence is suggestive, not decisive.")
    else:
        claim = (f"The frameworks disagree: p = {freq.p_value:.4g}, BF₁₀ = {bayes.bf10:.3g}. "
                 f"The conclusion depends on which framework (and prior) you adopt.")

    checks = [ok, direction_agreed, magnitude_agreed]
    return SynthesisResult(
        agreement_type=kind, agreement_detail=detail, synthesized_claim=claim,
        direction_agreed=direction_agreed, magnitude_agreed=magnitude_agreed,
        significance_agreed=ok, boundary_conditions=boundaries,
        confidence_in_synthesis=round(sum(checks) / 3, 4),
    )
