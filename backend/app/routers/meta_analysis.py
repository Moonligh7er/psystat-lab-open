"""Meta-Analysis — effect size conversion, fixed/random-effects pooling,
publication bias, moderator analysis, and cumulative meta-analysis.

Inverse-variance meta-analysis following metafor (Viechtbauer, 2010):
  - tau² by DerSimonian–Laird or REML (Fisher scoring, Viechtbauer 2005)
  - Q-profile confidence interval for tau² (Viechtbauer, 2007)
  - Knapp–Hartung adjustment, Higgins et al. (2009) prediction interval
  - Egger regression test, Begg & Mazumdar rank test, Duval & Tweedie L0
    trim-and-fill, Rosenthal / Orwin fail-safe N
  - subgroup analysis (fixed or mixed effects) and meta-regression with a
    residual-heterogeneity moment estimator
Reference values in tests/test_meta_analysis.py come from metafor 5.2.1.
"""

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class EffectSizeRequest(BaseModel):
    """Convert between effect size types from raw statistics."""
    means: list[float] | None = None        # [M1, M2]
    sds: list[float] | None = None          # [SD1, SD2]
    ns: list[int] | None = None             # [n1, n2]
    t_value: float | None = None
    f_value: float | None = None            # one-way F with df_num=1
    r_value: float | None = None
    odds_ratio: float | None = None
    log_odds_ratio: float | None = None
    proportions: list[float] | None = None  # [p1, p2]  (for 2x2)
    chi_square: float | None = None         # chi-sq with df=1
    total_n: int | None = None              # fallback total N
    df_error: int | None = None             # denominator df for F


class FixedEffectsRequest(BaseModel):
    effect_sizes: list[float]
    standard_errors: list[float] | None = None
    confidence_intervals: list[list[float]] | None = None  # [[lo, hi], ...] assumed 95%
    study_labels: list[str] | None = None
    confidence_level: float = 0.95


class RandomEffectsRequest(BaseModel):
    effect_sizes: list[float]
    standard_errors: list[float] | None = None
    confidence_intervals: list[list[float]] | None = None
    study_labels: list[str] | None = None
    confidence_level: float = 0.95
    use_reml: bool = False
    knapp_hartung: bool = False
    leave_one_out: bool = True


class PublicationBiasRequest(BaseModel):
    effect_sizes: list[float]
    standard_errors: list[float]
    study_labels: list[str] | None = None
    alpha: float = 0.05
    orwin_target: float | None = None  # criterion ("trivial") effect for Orwin's fail-safe N


class ModeratorRequest(BaseModel):
    effect_sizes: list[float]
    standard_errors: list[float]
    moderators: list[list[float]] | None = None       # continuous, one row per study
    moderator_names: list[str] | None = None
    categorical_moderator: list[str] | None = None     # one categorical grouping
    categorical_moderator_name: str | None = None
    confidence_level: float = 0.95
    mixed_effects: bool = True  # random effects within subgroups / residual tau² in meta-regression


class CumulativeRequest(BaseModel):
    effect_sizes: list[float]
    standard_errors: list[float]
    sort_values: list[float]            # e.g. publication year
    study_labels: list[str] | None = None
    confidence_level: float = 0.95
    method: str = "random"              # "fixed" or "random"


# ── Helpers ───────────────────────────────────────────────────────────

def _r(v, d=6):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), d)


def _zcrit(cl):
    if not 0 < cl < 1:
        raise HTTPException(422, "confidence_level must be between 0 and 1.")
    return float(sp_stats.norm.ppf(1 - (1 - cl) / 2))


def _labels(labels, k):
    if labels is None:
        return [f"Study {i + 1}" for i in range(k)]
    if len(labels) != k:
        raise HTTPException(422, f"study_labels has {len(labels)} entries but there are {k} effect sizes.")
    return list(labels)


def _arrays(effect_sizes, standard_errors, min_k=2, what="studies"):
    es = np.asarray(effect_sizes, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    k = len(es)
    if k < min_k:
        raise HTTPException(422, f"At least {min_k} {what} are required.")
    if len(se) != k:
        raise HTTPException(422, "effect_sizes and standard_errors must have the same length.")
    if not (np.all(np.isfinite(es)) and np.all(np.isfinite(se))):
        raise HTTPException(422, "Effect sizes and standard errors must be finite numbers.")
    if np.any(se <= 0):
        raise HTTPException(422, "All standard errors must be positive.")
    return es, se, k


def _resolve_ses(effect_sizes, standard_errors, confidence_intervals) -> np.ndarray:
    """Standard errors given directly, or derived from 95% CIs as (hi - lo) / (2 * 1.96)."""
    k = len(effect_sizes)
    if standard_errors is not None:
        ses = np.array(standard_errors, dtype=float)
        if len(ses) != k:
            raise HTTPException(422, "effect_sizes and standard_errors must have the same length.")
        return ses
    if confidence_intervals is not None:
        cis = np.array(confidence_intervals, dtype=float)
        if cis.shape != (k, 2):
            raise HTTPException(422, "confidence_intervals must be a list of [lo, hi] pairs matching effect_sizes length.")
        return (cis[:, 1] - cis[:, 0]) / (2 * sp_stats.norm.ppf(0.975))
    raise HTTPException(422, "Provide either standard_errors or confidence_intervals.")


def _fe(es, se):
    w = 1.0 / se ** 2
    mu = float(np.sum(w * es) / np.sum(w))
    return mu, float(1.0 / math.sqrt(np.sum(w))), w


def _q(es, se, tau2=0.0):
    """Generalised Q statistic at a given tau²."""
    w = 1.0 / (se ** 2 + tau2)
    mu = np.sum(w * es) / np.sum(w)
    return float(np.sum(w * (es - mu) ** 2))


def _tau2_dl(es, se):
    k = len(es)
    _, _, w = _fe(es, se)
    Q = _q(es, se)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    return max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0


def _tau2_reml(es, se, max_iter=200, tol=1e-10):
    """REML tau² by Fisher scoring with step halving (as metafor's rma.uni).

    Score = (y'PPy - tr(P)) / 2 and information tr(PP) / 2, with
    P = W - W 1 1' W / sum(w). For the univariate model y'PPy = sum(w² r²),
    tr(P) = sum(w) - sum(w²)/sum(w), tr(PP) = sum(w²) - 2 sum(w³)/sum(w) + (sum(w²)/sum(w))².
    """
    vi = se ** 2
    tau2 = _tau2_dl(es, se)
    for _ in range(max_iter):
        w = 1.0 / (vi + tau2)
        sw, sw2, sw3 = np.sum(w), np.sum(w ** 2), np.sum(w ** 3)
        mu = np.sum(w * es) / sw
        r = es - mu
        num = np.sum(w ** 2 * r ** 2) - (sw - sw2 / sw)
        den = sw2 - 2 * sw3 / sw + (sw2 / sw) ** 2
        step = num / den
        new = tau2 + step
        while new < 0:  # step halving keeps tau² in the parameter space
            step /= 2
            new = tau2 + step
            if abs(step) < tol:
                new = 0.0
                break
        if abs(new - tau2) < tol:
            return float(new)
        tau2 = new
    return float(tau2)


def _tau2(es, se, method):
    return _tau2_reml(es, se) if method == "REML" else _tau2_dl(es, se)


def _tau2_qprofile_ci(es, se, cl=0.95):
    """Q-profile CI for tau² (Viechtbauer, 2007): solve Q(tau²) = chi²_{k-1} quantiles."""
    k = len(es)
    if k < 2:
        return None, None
    lo_q = sp_stats.chi2.ppf(1 - (1 - cl) / 2, k - 1)   # upper quantile -> lower bound
    hi_q = sp_stats.chi2.ppf((1 - cl) / 2, k - 1)       # lower quantile -> upper bound

    def bound(target):
        if _q(es, se, 0.0) <= target:
            return 0.0
        upper = max(1.0, 10 * float(np.var(es)))
        while _q(es, se, upper) > target:
            upper *= 4
            if upper > 1e8:
                return None
        return float(brentq(lambda t: _q(es, se, t) - target, 0.0, upper, xtol=1e-12))

    return bound(lo_q), bound(hi_q)


def _typical_v(se):
    """Typical within-study variance (Higgins & Thompson, 2002)."""
    w = 1.0 / se ** 2
    k = len(se)
    return float((k - 1) * np.sum(w) / (np.sum(w) ** 2 - np.sum(w ** 2)))


def _i2_h2(tau2, se):
    s2 = _typical_v(se)
    i2 = 100 * tau2 / (tau2 + s2)
    return i2, (tau2 + s2) / s2


def _pool_re(es, se, tau2, cl=0.95, knha=False):
    k = len(es)
    w = 1.0 / (se ** 2 + tau2)
    mu = float(np.sum(w * es) / np.sum(w))
    se_mu = float(1.0 / math.sqrt(np.sum(w)))
    if knha and k > 1:
        q = float(np.sum(w * (es - mu) ** 2) / (k - 1))
        se_mu = se_mu * math.sqrt(q)
        crit = float(sp_stats.t.ppf(1 - (1 - cl) / 2, k - 1))
        stat = mu / se_mu if se_mu > 0 else float("nan")
        p = float(2 * sp_stats.t.sf(abs(stat), k - 1))
        df = k - 1
    else:
        crit = _zcrit(cl)
        stat = mu / se_mu
        p = float(2 * sp_stats.norm.sf(abs(stat)))
        df = None
    return {"mu": mu, "se": se_mu, "ci": (mu - crit * se_mu, mu + crit * se_mu), "stat": stat, "p": p, "df": df, "w": w}


def _heterogeneity(es, se, tau2=None, tau2_label="DL"):
    """Cochran's Q and I²/H². I² uses the reported tau² and the typical within-study variance."""
    k = len(es)
    Q = _q(es, se)
    df = k - 1
    Qp = float(sp_stats.chi2.sf(Q, df)) if df > 0 else 1.0
    t2_dl = _tau2_dl(es, se)
    t2 = t2_dl if tau2 is None else tau2
    i2, h2 = _i2_h2(t2, se) if k > 1 else (0.0, 1.0)
    return {
        "Q": _r(Q), "Q_df": df, "Q_p": _r(Qp, 8),
        "I_squared": _r(i2, 2), "H_squared": _r(h2, 4),
        "tau_squared": _r(t2), "tau": _r(math.sqrt(t2)), "tau_squared_method": tau2_label,
        "tau_squared_DL": _r(t2_dl), "tau_DL": _r(math.sqrt(t2_dl)),
    }


def _prediction_interval(mu, tau2, se_mu, k, cl=0.95):
    """Higgins, Thompson & Spiegelhalter (2009): mu ± t_{k-2} sqrt(tau² + SE²)."""
    if k < 3:
        return None
    t = float(sp_stats.t.ppf(1 - (1 - cl) / 2, k - 2))
    half = t * math.sqrt(tau2 + se_mu ** 2)
    return [_r(mu - half), _r(mu + half)]


def _forest(es, se, w, labels, cl):
    z = _zcrit(cl)
    pct = w / np.sum(w) * 100
    return [{"label": labels[i], "es": _r(es[i]), "se": _r(se[i]),
             "ci": [_r(es[i] - z * se[i]), _r(es[i] + z * se[i])], "weight_pct": _r(pct[i], 2)} for i in range(len(es))]


def _interpret_i2(i2: float) -> str:
    if i2 < 25:
        return "Low heterogeneity (I² < 25%)."
    if i2 < 50:
        return "Low-to-moderate heterogeneity (25% ≤ I² < 50%)."
    if i2 < 75:
        return "Moderate-to-substantial heterogeneity (50% ≤ I² < 75%): consider exploring moderators."
    return "Substantial heterogeneity (I² ≥ 75%): the average effect may describe few studies well; explore moderators."


def _interpret_pooled(pooled: float, p: float) -> str:
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    direction = "positive" if pooled > 0 else "negative" if pooled < 0 else "zero"
    return f"The pooled effect is {direction} (ES = {pooled:.4f}, p = {p:.4f}), {sig} at α = .05."


# ── Endpoint 1: Effect Size Conversion ────────────────────────────────

@router.post("/effect-sizes")
async def convert_effect_sizes(req: EffectSizeRequest):
    """Convert summary statistics to d, g, r and (log) odds ratio with SEs and 95% CIs.

    Formulas follow Borenstein, Hedges, Higgins & Rothstein (2009), ch. 4–7.
    Conversions from r, chi-square, F and odds ratios assume two groups; d from
    chi-square or F is non-directional (reported as positive). When only total N
    is supplied, equal group sizes are assumed and the response says so.
    """
    notes = []
    n1 = n2 = n_total = None
    if req.ns is not None:
        if len(req.ns) != 2 or min(req.ns) < 2:
            raise HTTPException(422, "ns must be two group sizes, each at least 2.")
        n1, n2 = int(req.ns[0]), int(req.ns[1])
        n_total = n1 + n2
    elif req.total_n is not None:
        if req.total_n < 4:
            raise HTTPException(422, "total_n must be at least 4.")
        n_total = int(req.total_n)
        n1, n2 = n_total / 2, n_total / 2
        notes.append("Group sizes not given: equal groups (N/2 each) assumed.")

    d = var_d = None
    source = None

    if req.means is not None or req.sds is not None:
        if req.means is None or req.sds is None or len(req.means) != 2 or len(req.sds) != 2:
            raise HTTPException(422, "Provide two means and two SDs.")
        if n1 is None:
            raise HTTPException(422, "Means and SDs need group sizes (ns).")
        if min(req.sds) <= 0:
            raise HTTPException(422, "SDs must be positive.")
        (m1, m2), (s1, s2) = req.means, req.sds
        sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        d = (m1 - m2) / sp
        var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
        source = "means/SDs"
    elif req.t_value is not None:
        if n1 is None:
            raise HTTPException(422, "A t value needs the group sizes (ns) or total_n.")
        d = req.t_value * math.sqrt((n1 + n2) / (n1 * n2))
        var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
        source = "independent-samples t"
    elif req.f_value is not None:
        if req.f_value < 0:
            raise HTTPException(422, "F cannot be negative.")
        if n1 is None:
            raise HTTPException(422, "An F value needs the group sizes (ns) or total_n.")
        d = math.sqrt(req.f_value) * math.sqrt((n1 + n2) / (n1 * n2))
        var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
        source = "F (1 numerator df)"
        notes.append("d from F is non-directional (shown as positive).")
    elif req.chi_square is not None:
        if req.chi_square < 0 or n_total is None:
            raise HTTPException(422, "A chi-square needs a non-negative value and the total N.")
        phi = math.sqrt(req.chi_square / n_total)
        if phi >= 1:
            raise HTTPException(422, "phi = sqrt(chi²/N) must be below 1; check chi-square and N.")
        r_phi = phi
        d = 2 * r_phi / math.sqrt(1 - r_phi ** 2)
        var_d = 4 / ((n_total - 1) * (1 - r_phi ** 2))  # via Var(r) = (1-r²)²/(n-1)
        source = "chi-square (df = 1) via phi"
        notes.append("d from chi-square is non-directional (shown as positive).")
    elif req.r_value is not None:
        r = req.r_value
        if abs(r) >= 1:
            raise HTTPException(422, "r must be strictly between -1 and 1.")
        d = 2 * r / math.sqrt(1 - r ** 2)
        if n_total is not None:
            var_d = 4 / ((n_total - 1) * (1 - r ** 2))  # Borenstein et al. (2009) eq. 7.10
        else:
            notes.append("No sample size given, so standard errors and CIs cannot be computed.")
        source = "correlation r"
    elif req.odds_ratio is not None or req.log_odds_ratio is not None:
        if req.odds_ratio is not None and req.odds_ratio <= 0:
            raise HTTPException(422, "Odds ratio must be positive.")
        log_or = math.log(req.odds_ratio) if req.odds_ratio is not None else req.log_odds_ratio
        d = log_or * math.sqrt(3) / math.pi  # Hasselblad & Hedges (1995)
        notes.append("Standard errors need the 2x2 cell counts (or proportions with group sizes); not computed from an odds ratio alone.")
        source = "odds ratio (logistic conversion)"
    elif req.proportions is not None:
        if len(req.proportions) != 2 or not all(0 < p < 1 for p in req.proportions):
            raise HTTPException(422, "Provide two proportions strictly between 0 and 1.")
        p1, p2 = req.proportions
        log_or = math.log((p1 / (1 - p1)) / (p2 / (1 - p2)))
        d = log_or * math.sqrt(3) / math.pi
        if n1 is not None:
            var_log_or = 1 / (n1 * p1 * (1 - p1)) + 1 / (n2 * p2 * (1 - p2))
            var_d = var_log_or * 3 / math.pi ** 2
        else:
            notes.append("Group sizes are needed for standard errors.")
        source = "two proportions (logistic conversion)"
    else:
        raise HTTPException(422, "Provide one of: means/SDs/ns, t_value, f_value, chi_square, r_value, odds_ratio, log_odds_ratio, proportions.")

    z = sp_stats.norm.ppf(0.975)
    se_d = math.sqrt(var_d) if var_d is not None else None

    # Hedges' g: J = 1 - 3 / (4 df - 1), df = n1 + n2 - 2
    J = 1 - 3 / (4 * (n_total - 2) - 1) if n_total is not None and n_total > 3 else None
    g = d * J if J is not None else None
    se_g = se_d * J if se_d is not None and J is not None else None

    # r from d: r = d / sqrt(d² + a), a = (n1+n2)² / (n1 n2) (= 4 for equal groups)
    a = (n1 + n2) ** 2 / (n1 * n2) if n1 is not None else 4.0
    r_val = d / math.sqrt(d ** 2 + a)
    se_r = math.sqrt(a ** 2 * var_d / (d ** 2 + a) ** 3) if var_d is not None else None

    log_or_val = d * math.pi / math.sqrt(3)
    se_lor = se_d * math.pi / math.sqrt(3) if se_d is not None else None

    def ci(est, s):
        return None if s is None else [_r(est - z * s), _r(est + z * s)]

    return {
        "source": source,
        "cohens_d": {"estimate": _r(d), "se": _r(se_d), "ci_95": ci(d, se_d)},
        "hedges_g": {"estimate": _r(g), "se": _r(se_g), "ci_95": ci(g, se_g) if g is not None else None,
                     "correction_factor_J": _r(J)},
        "r": {"estimate": _r(r_val), "se": _r(se_r), "ci_95": ci(r_val, se_r)},
        "log_odds_ratio": {"estimate": _r(log_or_val), "se": _r(se_lor), "ci_95": ci(log_or_val, se_lor)},
        "odds_ratio": {"estimate": _r(math.exp(log_or_val)),
                       "ci_95": None if se_lor is None else [_r(math.exp(log_or_val - z * se_lor)), _r(math.exp(log_or_val + z * se_lor))]},
        "sample_sizes": {"n1": n1, "n2": n2, "total": n_total},
        "notes": notes,
    }


# ── Endpoint 2: Fixed-Effect Meta-Analysis ────────────────────────────

@router.post("/fixed-effects")
async def fixed_effects_meta(req: FixedEffectsRequest):
    """Inverse-variance fixed-effect (common-effect) meta-analysis with heterogeneity statistics."""
    se = _resolve_ses(req.effect_sizes, req.standard_errors, req.confidence_intervals)
    es, se, k = _arrays(req.effect_sizes, se)
    labels = _labels(req.study_labels, k)
    cl, z = req.confidence_level, _zcrit(req.confidence_level)

    mu, se_mu, w = _fe(es, se)
    zval = mu / se_mu
    p = float(2 * sp_stats.norm.sf(abs(zval)))
    het = _heterogeneity(es, se)

    cumulative = []
    for j in range(1, k + 1):
        m, s, _ = _fe(es[:j], se[:j])
        cumulative.append({"study": labels[j - 1], "cumulative_es": _r(m), "cumulative_ci": [_r(m - z * s), _r(m + z * s)]})

    return {
        "model": "fixed-effect",
        "k": k,
        "pooled_effect": _r(mu),
        "pooled_se": _r(se_mu),
        "ci": [_r(mu - z * se_mu), _r(mu + z * se_mu)],
        "z": _r(zval, 4),
        "p": _r(p, 8),
        "heterogeneity": het,
        # A fixed-effect model assumes one true effect, so there is no prediction interval.
        "prediction_interval": None,
        "forest_data": _forest(es, se, w, labels, cl),
        "cumulative_data": cumulative,
        "interpretation": _interpret_pooled(mu, p) + " " + _interpret_i2(het["I_squared"])
        + (" With this much heterogeneity a random-effects model is usually more appropriate." if het["I_squared"] >= 50 else ""),
    }


# ── Endpoint 3: Random-Effects Meta-Analysis ──────────────────────────

@router.post("/random-effects")
async def random_effects_meta(req: RandomEffectsRequest):
    """Random-effects meta-analysis (DerSimonian–Laird or REML tau²), optional Knapp–Hartung."""
    se = _resolve_ses(req.effect_sizes, req.standard_errors, req.confidence_intervals)
    es, se, k = _arrays(req.effect_sizes, se)
    labels = _labels(req.study_labels, k)
    cl = req.confidence_level
    method = "REML" if req.use_reml else "DL"

    tau2 = _tau2(es, se, method)
    fit = _pool_re(es, se, tau2, cl, req.knapp_hartung and k > 2)
    het = _heterogeneity(es, se, tau2, method)
    t2_lo, t2_hi = _tau2_qprofile_ci(es, se, cl)
    i2_ci = [_r(_i2_h2(t, se)[0], 2) if t is not None else None for t in (t2_lo, t2_hi)]

    loo = None
    if req.leave_one_out and k >= 3:
        loo = []
        for j in range(k):
            e_j, s_j = np.delete(es, j), np.delete(se, j)
            t_j = _tau2(e_j, s_j, method)
            f_j = _pool_re(e_j, s_j, t_j, cl, req.knapp_hartung and k - 1 > 2)
            loo.append({"omitted": labels[j], "pooled_es": _r(f_j["mu"]), "ci": [_r(f_j["ci"][0]), _r(f_j["ci"][1])],
                        "p": _r(f_j["p"], 8), "tau_squared": _r(t_j)})

    stat_name = "t" if fit["df"] is not None else "z"
    return {
        "model": "random-effects",
        "method": "REML" if method == "REML" else "DerSimonian-Laird",
        "knapp_hartung": bool(fit["df"] is not None),
        "k": k,
        "pooled_effect": _r(fit["mu"]),
        "pooled_se": _r(fit["se"]),
        "ci": [_r(fit["ci"][0]), _r(fit["ci"][1])],
        stat_name: _r(fit["stat"], 4),
        "test_df": fit["df"],
        "p": _r(fit["p"], 8),
        "heterogeneity": het,
        "tau_squared": _r(tau2),
        "tau_squared_DL": het["tau_squared_DL"],
        "tau_squared_REML": _r(tau2) if method == "REML" else None,
        "tau_squared_CI": [_r(t2_lo), _r(t2_hi)],
        "tau_squared_CI_method": "Q-profile (Viechtbauer, 2007)",
        "I_squared_CI": i2_ci,
        "prediction_interval": _prediction_interval(fit["mu"], tau2, fit["se"], k, cl),
        "prediction_interval_method": "Higgins et al. (2009): t with k − 2 df",
        "forest_data": _forest(es, se, fit["w"], labels, cl),
        "leave_one_out": loo,
        "interpretation": (
            _interpret_pooled(fit["mu"], fit["p"]) + " " + _interpret_i2(het["I_squared"])
            + (" Knapp–Hartung adjustment applied (t-based test and CI)." if fit["df"] is not None else "")
        ),
    }


# ── Endpoint 4: Publication Bias ──────────────────────────────────────

def _trim_and_fill(es, se, side):
    """Duval & Tweedie (2000) trim-and-fill with the L0 estimator, fixed-effect pooling.

    `side` is where studies are assumed missing. Implementation mirrors metafor's
    trimfill(): flip so missing studies are on the left, trim the k0 largest,
    re-estimate, recount L0 from signed ranks of all deviations, iterate.
    """
    k = len(es)
    y = -es if side == "right" else es.copy()
    order = np.argsort(y, kind="mergesort")
    y, s = y[order], se[order]
    k0, prev, mu = 0, -1, None
    for _ in range(100):
        if k0 == prev:
            break
        prev = k0
        mu, _, _ = _fe(y[:k - k0], s[:k - k0])
        dev = y - mu
        ranks = sp_stats.rankdata(np.abs(dev), method="ordinal")
        Sr = float(np.sum(ranks[dev > 0]))
        k0 = max(0, int(round((4 * Sr - k * (k + 1)) / (2 * k - 1))))
        k0 = min(k0, k - 1)
    filled_y = np.concatenate([y, 2 * mu - y[k - k0:]]) if k0 > 0 else y
    filled_s = np.concatenate([s, s[k - k0:]]) if k0 > 0 else s
    if side == "right":
        filled_y = -filled_y
    imputed = [{"es": _r(v), "se": _r(t)} for v, t in zip(filled_y[k:], filled_s[k:])]
    return k0, filled_y, filled_s, imputed


@router.post("/publication-bias")
async def publication_bias(req: PublicationBiasRequest):
    """Small-study effects / publication bias: Egger, Begg & Mazumdar, trim-and-fill, fail-safe N.

    These detect funnel-plot asymmetry, which can arise from causes other than
    publication bias (true heterogeneity, chance); none of them proves bias.
    """
    es, se, k = _arrays(req.effect_sizes, req.standard_errors, min_k=3)
    labels = _labels(req.study_labels, k)
    alpha = req.alpha
    vi = se ** 2

    # Egger (1997): regress y/se on 1/se; the intercept tests asymmetry (t, k-2 df).
    # Equivalent to metafor regtest(model="lm", predictor="sei").
    X = np.column_stack([np.ones(k), 1 / se])
    yz = es / se
    beta, *_ = np.linalg.lstsq(X, yz, rcond=None)
    resid = yz - X @ beta
    mse = float(resid @ resid / (k - 2)) if k > 2 else float("nan")
    cov = mse * np.linalg.inv(X.T @ X)
    b0, se_b0 = float(beta[0]), float(math.sqrt(cov[0, 0]))
    t_egg = b0 / se_b0
    p_egg = float(2 * sp_stats.t.sf(abs(t_egg), k - 2))
    egger = {"intercept": _r(b0), "intercept_se": _r(se_b0), "t": _r(t_egg, 4), "df": k - 2,
             "p": _r(p_egg, 8), "significant": bool(p_egg < alpha)}

    # Begg & Mazumdar (1994): Kendall's tau between standardised deviates and variances.
    mu_fe, se_fe, _ = _fe(es, se)
    v_star = vi - se_fe ** 2
    dev = (es - mu_fe) / np.sqrt(v_star)
    tau_b, p_b = sp_stats.kendalltau(dev, vi)
    begg = {"kendalls_tau": _r(tau_b), "p": _r(p_b, 8), "significant": bool(p_b < alpha)}

    # Side of the funnel with missing studies: from the slope of y on se (weights 1/v);
    # a positive slope means small studies show larger effects, so studies are missing on the left.
    W = np.diag(1 / vi)
    Xs = np.column_stack([np.ones(k), se])
    slope = float(np.linalg.solve(Xs.T @ W @ Xs, Xs.T @ W @ es)[1])
    side = "left" if slope > 0 else "right"
    k0, fy, fs, imputed = _trim_and_fill(es, se, side)
    mu_f, se_f, _ = _fe(fy, fs)
    z = sp_stats.norm.ppf(0.975)
    trim_fill = {
        "imputed_studies": k0, "side": side, "estimator": "L0, fixed-effect",
        "adjusted_es": _r(mu_f), "adjusted_se": _r(se_f), "adjusted_ci_95": [_r(mu_f - z * se_f), _r(mu_f + z * se_f)],
        "imputed_data": imputed,
    }

    # Rosenthal (1979) fail-safe N with Stouffer's sum of z and a one-tailed alpha.
    z_vals = es / se
    z_alpha = sp_stats.norm.ppf(1 - alpha)
    rosenthal_N = max(0, int(math.ceil((np.sum(z_vals) / z_alpha) ** 2 - k)))  # rounded up, as metafor::fsn
    # Orwin (1983): studies averaging zero needed to bring the unweighted mean to the criterion.
    mean_es = float(np.mean(es))
    orwin_N = None
    if req.orwin_target is not None and req.orwin_target != 0 and abs(mean_es) > abs(req.orwin_target):
        orwin_N = int(math.ceil(k * (mean_es - req.orwin_target) / req.orwin_target))
    failsafe = {
        "rosenthal_N": rosenthal_N,
        "rosenthal_interpretation": f"{rosenthal_N} unpublished null studies would be needed to raise the combined one-tailed p above α = {alpha}."
        + (" This exceeds Rosenthal's 5k + 10 benchmark." if rosenthal_N > 5 * k + 10 else ""),
        "orwin_N": orwin_N,
        "note": "Fail-safe N ignores effect size and between-study variance; treat it as a rough indication only.",
    }

    se_range = np.linspace(0.0, float(np.max(se) * 1.1), 50)
    funnel_data = {
        "studies": [{"label": labels[i], "es": _r(es[i]), "se": _r(se[i])} for i in range(k)],
        "pooled_es": _r(mu_fe),
        "pseudo_ci": {"se_values": [_r(s) for s in se_range],
                      "lower": [_r(mu_fe - z * s) for s in se_range], "upper": [_r(mu_fe + z * s) for s in se_range]},
    }
    contours = {lab: {"lower": [_r(-zc * s) for s in se_range], "upper": [_r(zc * s) for s in se_range]}
                for lab, zc in (("p_01", sp_stats.norm.ppf(0.995)), ("p_05", z), ("p_10", sp_stats.norm.ppf(0.95)))}

    flags = []
    if egger["significant"]:
        flags.append("Egger's test indicates funnel-plot asymmetry")
    if begg["significant"]:
        flags.append("Begg's rank test indicates asymmetry")
    if k0 > 0:
        flags.append(f"trim-and-fill imputes {k0} stud{'y' if k0 == 1 else 'ies'} on the {side}")
    interp = ("Possible small-study effects: " + "; ".join(flags) + ". Asymmetry can also reflect true heterogeneity."
              if flags else "No clear evidence of funnel-plot asymmetry from these tests (they have low power with few studies).")
    if k < 10:
        interp += " With fewer than 10 studies these tests are not recommended (Sterne et al., 2011)."

    return {
        "k": k,
        "egger": egger,
        "begg": begg,
        "trim_and_fill": trim_fill,
        "failsafe_N": failsafe,
        "funnel_data": funnel_data,
        "contour_funnel_data": {"studies": funnel_data["studies"], "se_values": funnel_data["pseudo_ci"]["se_values"], "contours": contours},
        "interpretation": interp,
    }


# ── Endpoint 5: Moderator Analysis ────────────────────────────────────

def _wls(X, y, w):
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    b = cov @ XtW @ y
    return b, cov


@router.post("/moderator-analysis")
async def moderator_analysis(req: ModeratorRequest):
    """Subgroup analysis (categorical moderator) and meta-regression (continuous moderators)."""
    es, se, k = _arrays(req.effect_sizes, req.standard_errors, min_k=3)
    vi = se ** 2
    cl, z = req.confidence_level, _zcrit(req.confidence_level)
    results = {}

    if req.categorical_moderator is not None:
        cat = req.categorical_moderator
        if len(cat) != k:
            raise HTTPException(422, "categorical_moderator length must match effect_sizes length.")
        groups = sorted(set(cat))
        if len(groups) < 2:
            raise HTTPException(422, "The categorical moderator needs at least 2 groups.")
        subgroup_results, ests, ses = [], [], []
        Q_within = 0.0
        for grp in groups:
            idx = np.array([i for i in range(k) if cat[i] == grp])
            e_g, s_g = es[idx], se[idx]
            k_g = len(idx)
            tau2_g = _tau2_dl(e_g, s_g) if (req.mixed_effects and k_g >= 2) else 0.0
            fit = _pool_re(e_g, s_g, tau2_g, cl)
            Q_g = _q(e_g, s_g) if k_g >= 2 else 0.0
            Q_within += Q_g
            ests.append(fit["mu"])
            ses.append(fit["se"])
            subgroup_results.append({
                "group": grp, "k": int(k_g), "pooled_es": _r(fit["mu"]), "se": _r(fit["se"]),
                "ci": [_r(fit["ci"][0]), _r(fit["ci"][1])], "z": _r(fit["stat"], 4), "p": _r(fit["p"], 8),
                "tau_squared": _r(tau2_g),
                "heterogeneity": _heterogeneity(e_g, s_g, tau2_g if req.mixed_effects else None) if k_g >= 2 else None,
            })
        ests, ses = np.array(ests), np.array(ses)
        if req.mixed_effects:
            # Test of subgroup differences on the subgroup estimates and their RE standard errors
            # (Borenstein et al., 2009, ch. 19; as in RevMan / metafor with separate tau²).
            Q_between = _q(ests, ses)
            model = "mixed-effects (separate DL tau² within subgroups)"
        else:
            Q_between = _q(es, se) - Q_within
            model = "fixed-effect"
        df_b = len(groups) - 1
        p_b = float(sp_stats.chi2.sf(max(0.0, Q_between), df_b))
        results["subgroup_analysis"] = {
            "moderator_name": req.categorical_moderator_name or "group",
            "n_groups": len(groups),
            "subgroups": subgroup_results,
            "Q_between": _r(Q_between), "Q_between_df": df_b, "Q_between_p": _r(p_b, 8),
            "Q_within": _r(Q_within), "model": model,
            "interpretation": (f"Test for subgroup differences: Q({df_b}) = {Q_between:.2f}, p = {p_b:.4f}. "
                               + ("Effects differ between subgroups." if p_b < 0.05 else "No significant difference between subgroups.")
                               + " Subgroup comparisons are observational, not randomised."),
        }

    if req.moderators is not None:
        mods = np.array(req.moderators, dtype=float)
        if mods.ndim == 1:
            mods = mods.reshape(-1, 1)
        if mods.ndim != 2 or mods.shape[0] != k:
            raise HTTPException(422, "moderators must have one row per study.")
        p_mod = mods.shape[1]
        names = req.moderator_names if req.moderator_names and len(req.moderator_names) == p_mod else [f"Mod_{j + 1}" for j in range(p_mod)]
        X = np.column_stack([np.ones(k), mods])
        p = X.shape[1]
        if k <= p:
            raise HTTPException(422, "Too few studies for this many moderators.")
        if np.linalg.matrix_rank(X) < p:
            raise HTTPException(422, "Moderators are collinear (or constant); the model is not identified.")

        # Residual heterogeneity test QE with fixed-effect weights (df k - p)
        w_fe = 1 / vi
        b_fe, cov_fe = _wls(X, es, w_fe)
        QE = float(np.sum(w_fe * (es - X @ b_fe) ** 2))
        QE_p = float(sp_stats.chi2.sf(QE, k - p))

        if req.mixed_effects:
            # Residual tau², method of moments (DerSimonian–Laird generalised to meta-regression)
            P = np.diag(w_fe) - (w_fe[:, None] * X) @ cov_fe @ (X.T * w_fe)
            tau2_res = max(0.0, (QE - (k - p)) / float(np.trace(P)))
            w = 1 / (vi + tau2_res)
            tau2_tot = _tau2_dl(es, se)
            R2 = max(0.0, (tau2_tot - tau2_res) / tau2_tot) if tau2_tot > 0 else 0.0
        else:
            tau2_res, w, R2 = 0.0, w_fe, None
        b, cov = _wls(X, es, w)
        se_b = np.sqrt(np.diag(cov))
        zb = b / se_b
        pb = 2 * sp_stats.norm.sf(np.abs(zb))
        # Omnibus Wald test of the moderators (excluding the intercept)
        bm, Vm = b[1:], cov[1:, 1:]
        QM = float(bm @ np.linalg.solve(Vm, bm))
        QM_p = float(sp_stats.chi2.sf(QM, p - 1))

        coefs = [{"term": "intercept" if j == 0 else names[j - 1], "estimate": _r(b[j]), "se": _r(se_b[j]),
                  "z": _r(zb[j], 4), "p": _r(pb[j], 8), "ci": [_r(b[j] - z * se_b[j]), _r(b[j] + z * se_b[j])]}
                 for j in range(p)]
        x1 = np.linspace(float(mods[:, 0].min()), float(mods[:, 0].max()), 30)
        others = mods[:, 1:].mean(axis=0) if p_mod > 1 else np.array([])
        line = [{"x": _r(x), "y": _r(b[0] + b[1] * x + (float(b[2:] @ others) if p_mod > 1 else 0.0))} for x in x1]
        results["meta_regression"] = {
            "model": "mixed-effects (method-of-moments residual tau²)" if req.mixed_effects else "fixed-effect",
            "tau_squared": _r(tau2_res),
            "coefficients": coefs,
            "Q_model": _r(QM), "Q_model_df": p - 1, "Q_model_p": _r(QM_p, 8),
            "Q_residual": _r(QE), "Q_residual_df": k - p, "Q_residual_p": _r(QE_p, 8),
            "R_squared_analog": _r(R2, 4) if R2 is not None else None,
            "bubble_plot_data": [{"moderator_value": _r(mods[i, 0]), "es": _r(es[i]), "weight": _r(w[i])} for i in range(k)],
            "regression_line": line,
            "interpretation": (f"Omnibus test of moderators: QM({p - 1}) = {QM:.2f}, p = {QM_p:.4f}. "
                               + (f"R² = {R2:.1%} of the between-study variance is accounted for. " if R2 is not None else "")
                               + f"Residual heterogeneity: QE({k - p}) = {QE:.2f}, p = {QE_p:.4f}."),
        }

    if not results:
        raise HTTPException(422, "Provide at least one moderator (continuous or categorical).")
    return results


# ── Endpoint 6: Cumulative Meta-Analysis ──────────────────────────────

@router.post("/cumulative")
async def cumulative_meta(req: CumulativeRequest):
    """Cumulative meta-analysis: pool studies one at a time in sort order."""
    es, se, k = _arrays(req.effect_sizes, req.standard_errors)
    sv = np.asarray(req.sort_values, dtype=float)
    if len(sv) != k:
        raise HTTPException(422, "sort_values must have one value per study.")
    if req.method not in ("fixed", "random"):
        raise HTTPException(422, "method must be 'fixed' or 'random'.")
    labels = _labels(req.study_labels, k)
    cl = req.confidence_level

    order = np.argsort(sv, kind="mergesort")
    es_s, se_s, sv_s = es[order], se[order], sv[order]
    labels_s = [labels[i] for i in order]
    steps = []
    for j in range(1, k + 1):
        e, s = es_s[:j], se_s[:j]
        tau2 = _tau2_dl(e, s) if (req.method == "random" and j >= 2) else 0.0
        fit = _pool_re(e, s, tau2, cl)
        i2 = _i2_h2(tau2, s)[0] if j >= 2 else 0.0
        steps.append({"step": j, "study_added": labels_s[j - 1], "sort_value": _r(sv_s[j - 1], 4), "cumulative_k": j,
                      "pooled_es": _r(fit["mu"]), "se": _r(fit["se"]), "ci": [_r(fit["ci"][0]), _r(fit["ci"][1])],
                      "z": _r(fit["stat"], 4), "p": _r(fit["p"], 8), "tau_squared": _r(tau2), "I_squared": _r(i2, 2)})

    return {
        "method": req.method,
        "k": k,
        "steps": steps,
        "final_pooled_es": steps[-1]["pooled_es"],
        "final_ci": steps[-1]["ci"],
        "interpretation": (f"Cumulative {req.method}-effects meta-analysis across {k} studies; final pooled ES = "
                           f"{steps[-1]['pooled_es']:.4f} {steps[-1]['ci']}. Look for the point at which the estimate and its CI stop changing."),
    }
