from fastapi import APIRouter
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(tags=["Assumption Gate"])

N_RESAMPLE = 2000  # permutation / bootstrap / Monte Carlo resamples


def _mc_p(count: int, b: int = N_RESAMPLE) -> float:
    """Resampling p-value (count + 1) / (B + 1), which is never exactly 0 (Phipson & Smyth, 2010)."""
    return (count + 1) / (b + 1)


def _percentage_bend(x, y, beta: float = 0.2):
    """Percentage bend correlation (Wilcox, 1994) with its t-test (df = n − 2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    m = int(np.floor((1 - beta) * n))

    def bent(v):
        med = np.median(v)
        omega = np.sort(np.abs(v - med))[m - 1]
        psi = (v - med) / omega
        i1, i2 = int(np.sum(psi < -1)), int(np.sum(psi > 1))
        s = np.sum(v[(psi >= -1) & (psi <= 1)])
        phi = (omega * (i2 - i1) + s) / (n - i1 - i2)
        return np.clip((v - phi) / omega, -1, 1)

    a, b = bent(x), bent(y)
    r = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    t = r * np.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else np.inf
    return r, float(2 * stats.t.sf(abs(t), n - 2))

class AssumptionGateRequest(BaseModel):
    test_type: str  # "t-test","anova","correlation","regression","chi-square","paired-t","repeated-anova","manova"
    data: Dict[str, List[float]]
    params: Dict[str, Any] = {}

class RunWithAlternativesRequest(AssumptionGateRequest):
    run_alternatives: bool = True

# ── helpers ──────────────────────────────────────────────────────────────────

def _asmp(name, test_used, stat, p, passed, severity, detail):
    return {"name": name, "test_used": test_used, "statistic": stat,
            "p_value": p, "passed": passed, "severity": severity, "detail": detail}

def _alt(name, reason, relaxed, trade):
    return {"test_name": name, "reason": reason, "assumptions_relaxed": relaxed, "trade_off": trade}

def _normality(vals, label, alpha=0.05):
    if len(vals) < 3:
        return _asmp(f"Normality ({label})", "Shapiro-Wilk", None, None, False, "violation",
                     f"Too few obs (n={len(vals)}).")
    w, p = stats.shapiro(vals[:5000])
    ok = p > alpha
    return _asmp(f"Normality ({label})", "Shapiro-Wilk", round(float(w), 6), round(float(p), 6),
                 ok, "info" if ok else ("warning" if p > 0.01 else "violation"),
                 f"{'Passed' if ok else 'Failed'}: W={w:.4f}, p={p:.4f}.")

def _levene(groups, label=""):
    s, p = stats.levene(*groups); ok = p > 0.05
    nm = f"Homogeneity of variance{' (' + label + ')' if label else ''}"
    return _asmp(nm, "Levene's test", round(float(s), 6), round(float(p), 6),
                 ok, "info" if ok else "warning", f"{'Passed' if ok else 'Failed'}: F={s:.4f}, p={p:.4f}.")

def _outliers(vals, label, k=3.0):
    med = np.median(vals); mad = np.median(np.abs(vals - med)) * 1.4826
    n_out = int(np.sum(vals != med)) if mad == 0 else int(np.sum(np.abs(vals - med) / mad > k))
    return _asmp(f"Outliers ({label})", "MAD method (k=3)", float(n_out), None,
                 n_out == 0, "info" if n_out == 0 else "warning", f"{n_out} outlier(s) in {label}.")

def _vif(X):
    Xa = np.column_stack([np.ones(X.shape[0]), X]); vifs = []
    for i in range(1, Xa.shape[1]):
        m = [j for j in range(Xa.shape[1]) if j != i]
        r2 = 1 - np.sum((Xa[:, i] - Xa[:, m] @ np.linalg.lstsq(Xa[:, m], Xa[:, i], rcond=None)[0]) ** 2) / np.sum((Xa[:, i] - Xa[:, i].mean()) ** 2)
        vifs.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return np.array(vifs)

def _bp_test(res, X):
    n = len(res); Xc = np.column_stack([np.ones(n), X])
    sq = res ** 2 / (np.sum(res ** 2) / n)
    fitted = Xc @ np.linalg.lstsq(Xc, sq, rcond=None)[0]
    lm = np.sum((fitted - sq.mean()) ** 2) / 2
    return float(lm), float(stats.chi2.sf(lm, Xc.shape[1] - 1))

def _cohens_d(a, b):
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / sp) if sp else 0.0

def _eta_sq(groups):
    g = np.concatenate(groups); ss_t = np.sum((g - g.mean()) ** 2)
    return float(sum(len(x)*(x.mean()-g.mean())**2 for x in groups) / ss_t) if ss_t else 0.0

def _get_groups(data, params):
    keys = sorted(k for k in data if k.lower().startswith("group"))
    if not keys: keys = sorted(data.keys())
    return [np.array(data[k], dtype=float) for k in keys]

# ── checkers ─────────────────────────────────────────────────────────────────

def _ck_ttest(data, params):
    groups = _get_groups(data, params)
    if len(groups) < 2: return [], "Need at least 2 groups.", []
    a, b = groups[0], groups[1]
    asmps = [_normality(a, "Group 1"), _normality(b, "Group 2"), _levene([a, b])]
    n_ok = min(len(a), len(b)) >= 30
    asmps.append(_asmp("Sample size", "n ≥ 30 per group (rule of thumb)", float(min(len(a), len(b))), None,
                       n_ok, "info" if n_ok else "warning",
                       f"Sizes: {len(a)}, {len(b)}. " + ("Large enough that mild non-normality matters little."
                       if n_ok else "Small: normality matters more, and these checks have little power to detect violations.")))
    alts, nf, vf = [], not asmps[0]["passed"] or not asmps[1]["passed"], not asmps[2]["passed"]
    if vf: alts.append(_alt("Welch's t-test", "Robust to unequal variances", ["Homogeneity of variance"], "Slightly less power when assumptions met."))
    if nf:
        alts += [_alt("Mann-Whitney U", "Non-parametric", ["Normality"], "Less power for normal data."),
                 _alt("Permutation test", "Distribution-free", ["Normality", "Homogeneity of variance"], "Computationally heavier."),
                 _alt("Bootstrap CI", "Non-parametric CI", ["Normality"], "No single p-value.")]
    rob = ("With 30 or more per group, the t-test is usually robust to mild non-normality; "
           "large samples also make normality tests flag trivial departures.") if n_ok and nf else ""
    return asmps, rob, alts

def _ck_anova(data, params):
    groups = _get_groups(data, params)
    asmps = [_normality(g, f"Group {i+1}") for i, g in enumerate(groups)]
    asmps.append(_levene(groups))
    sizes = [len(g) for g in groups]; ratio = max(sizes) / max(min(sizes), 1)
    asmps.append(_asmp("Group size balance", "Max/min ratio", round(ratio, 2), None,
                       ratio < 1.5, "info" if ratio < 1.5 else "warning",
                       f"Ratio {ratio:.2f}. {'Balanced' if ratio < 1.5 else 'Imbalanced'}."))
    alts = []
    if not asmps[-2]["passed"]: alts.append(_alt("Welch's ANOVA", "Robust to heteroscedasticity", ["Homogeneity of variance"], "Slightly less power."))
    if any(not a["passed"] for a in asmps[:-2]):
        alts += [_alt("Kruskal-Wallis", "Non-parametric", ["Normality"], "Tests whether one group tends to have larger values (stochastic dominance), not means."),
                 _alt("Bootstrap F-test", "Resampling-based", ["Normality", "Homogeneity of variance"], "Computationally intensive.")]
    return asmps, "", alts

def _ck_corr(data, params):
    keys = sorted(data.keys())[:2]
    x, y = np.array(data[keys[0]], dtype=float), np.array(data[keys[1]], dtype=float)
    asmps = [_normality(x, keys[0]), _normality(y, keys[1])]
    slope, intercept = np.polyfit(x, y, 1); res = y - (slope * x + intercept)
    res = res[np.argsort(x, kind="stable")]  # runs test needs residuals in order of x
    signs = res >= 0; runs = 1 + np.sum(signs[1:] != signs[:-1])
    n1, n2 = int(signs.sum()), int((~signs).sum())
    if n1 > 0 and n2 > 0:
        mu = 1 + 2*n1*n2/(n1+n2); var = (2*n1*n2*(2*n1*n2-n1-n2))/((n1+n2)**2*(n1+n2-1))
        z = (runs - mu) / np.sqrt(var) if var > 0 else 0; p_r = 2*(stats.norm.sf(abs(z)))
    else: z, p_r = 0, 1.0
    asmps.append(_asmp("Linearity", "Runs test on residuals", round(float(z), 4), round(float(p_r), 4),
                       p_r > 0.05, "info" if p_r > 0.05 else "warning",
                       "Linear pattern" if p_r > 0.05 else "Non-linear pattern suspected."))
    asmps += [_outliers(x, keys[0]), _outliers(y, keys[1])]
    alts = []
    if not asmps[0]["passed"] or not asmps[1]["passed"] or not asmps[2]["passed"]:
        alts = [_alt("Spearman rho", "Rank-based", ["Normality", "Linearity"], "Measures monotonic rather than linear association."),
                _alt("Kendall tau", "Robust rank correlation", ["Normality", "Linearity"], "More conservative."),
                _alt("Percentage bend correlation", "Robust to outliers", ["Outliers", "Normality"], "Less commonly reported.")]
    return asmps, "", alts

def _ck_regression(data, params):
    ivs = params.get("ivs", [k for k in sorted(data.keys()) if k != params.get("dv", sorted(data.keys())[-1])])
    dv = params.get("dv", sorted(data.keys())[-1])
    y = np.array(data[dv], dtype=float)
    X = np.column_stack([np.array(data[k], dtype=float) for k in ivs])
    Xc = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]; res = y - Xc @ beta
    asmps = [_normality(res, "residuals")]
    bp_s, bp_p = _bp_test(res, X)
    asmps.append(_asmp("Homoscedasticity", "Breusch-Pagan", round(bp_s, 4), round(bp_p, 4),
                       bp_p > 0.05, "info" if bp_p > 0.05 else "warning",
                       "Constant variance" if bp_p > 0.05 else "Heteroscedasticity detected."))
    if X.shape[1] > 1:
        mv = float(np.max(_vif(X)))
        asmps.append(_asmp("Multicollinearity", "VIF", round(mv, 2), None,
                           mv < 10, "info" if mv < 10 else "violation",
                           f"Max VIF={mv:.2f}. {'OK' if mv < 10 else 'High multicollinearity'}."))
    H = Xc @ np.linalg.inv(Xc.T @ Xc) @ Xc.T; h = np.diag(H)
    mse = np.sum(res**2) / (len(y) - Xc.shape[1])
    cooks = res**2 * h / (Xc.shape[1] * mse * (1-h)**2 + 1e-12); thr = 4/len(y)
    ni = int(np.sum(cooks > thr))
    asmps.append(_asmp("Influential points", f"Cook's D > 4/n ({thr:.4f})", float(ni), None,
                       ni == 0, "info" if ni == 0 else "warning", f"{ni} influential point(s)."))
    alts = []
    if bp_p <= 0.05: alts.append(_alt("OLS with HC3 robust standard errors", "Valid tests under heteroscedasticity", ["Homoscedasticity"], "Wider intervals; same coefficients."))
    if not asmps[0]["passed"] or ni > 0:
        alts.append(_alt("Robust regression (Huber)", "Downweights outliers", ["Normality of residuals", "Influential points"], "Less efficient under ideal conditions."))
    return asmps, "", alts

def _ck_chisq(data, params):
    obs = np.array(list(data.values()), dtype=float)
    if obs.ndim == 1: exp = np.full_like(obs, obs.mean())
    else:
        rs, cs = obs.sum(axis=1, keepdims=True), obs.sum(axis=0, keepdims=True)
        exp = rs * cs / obs.sum()
    me = float(exp.min()); ep = me >= 5
    asmps = [_asmp("Expected freq >= 5", "Min expected count", round(me, 2), None, ep,
                   "info" if ep else "violation", f"Min={me:.2f}. {'OK' if ep else 'Too low'}.")]
    tn = float(obs.sum()); np_ = tn >= 20
    asmps.append(_asmp("Sample size", "Total N >= 20", tn, None, np_, "info" if np_ else "warning", f"N={int(tn)}."))
    alts = []
    if not ep:
        if obs.ndim == 2 and obs.shape == (2, 2):
            alts.append(_alt("Fisher's exact test", "Exact for 2x2", ["Expected freq >= 5"], "Only 2x2."))
        alts.append(_alt("Monte Carlo chi-square", "Simulated p-value", ["Expected freq >= 5"], "Stochastic."))
    return asmps, "", alts

def _ck_paired(data, params):
    keys = sorted(data.keys())[:2]
    d = np.array(data[keys[0]], dtype=float) - np.array(data[keys[1]], dtype=float)
    asmps = [_normality(d, "differences"), _outliers(d, "differences")]
    alts = []
    if not asmps[0]["passed"]: alts.append(_alt("Wilcoxon signed-rank", "Non-parametric paired", ["Normality of differences"], "Tests median diff."))
    return asmps, "", alts

def _ck_rep_anova(data, params):
    keys = sorted(data.keys()); lvls = [np.array(data[k], dtype=float) for k in keys]
    asmps = [_normality(lv, f"Level {i+1}") for i, lv in enumerate(lvls)]
    if len(lvls) >= 2:
        pv = [np.var(lvls[i]-lvls[j], ddof=1) for i in range(len(lvls)) for j in range(i+1, len(lvls))]
        r = max(pv) / max(min(pv), 1e-12)
        asmps.append(_asmp("Sphericity (approx)", "Pairwise variance ratio", round(r, 4), None,
                           r < 2, "info" if r < 2 else "warning",
                           f"Ratio={r:.2f}. {'OK' if r < 2 else 'Use Greenhouse-Geisser'}."))
    return asmps, "", []

def _ck_manova(data, params):
    keys = sorted(data.keys()); mat = np.column_stack([np.array(data[k], dtype=float) for k in keys])
    n, p = mat.shape; mean = mat.mean(axis=0)
    S_ml = np.cov(mat, rowvar=False) * (n - 1) / n  # Mardia (1970) uses the ML covariance
    S_inv = np.linalg.pinv(S_ml); D = (mat - mean) @ S_inv @ (mat - mean).T
    b1p = float(np.sum(D**3)) / n**2; sk_s = n*b1p/6; sk_df = p*(p+1)*(p+2)/6
    sk_p = float(stats.chi2.sf(sk_s, sk_df)); sk_ok = sk_p > 0.05
    asmps = [_asmp("MV normality (skewness)", "Mardia's skewness", round(sk_s, 4), round(sk_p, 4),
                   sk_ok, "info" if sk_ok else "warning", "Passed" if sk_ok else "MV skewness detected.")]
    b2p = float(np.sum(np.diag(D)**2)) / n; ke = p*(p+2); kse = np.sqrt(8*p*(p+2)/n)
    kz = (b2p - ke) / kse if kse > 0 else 0; kp = float(2*(stats.norm.sf(abs(kz)))); kk = kp > 0.05
    asmps.append(_asmp("MV normality (kurtosis)", "Mardia's kurtosis", round(float(kz), 4), round(kp, 4),
                       kk, "info" if kk else "warning", "Passed" if kk else "MV kurtosis detected."))
    return asmps, "", []

CHECKERS = {"t-test": _ck_ttest, "anova": _ck_anova, "correlation": _ck_corr,
            "regression": _ck_regression, "chi-square": _ck_chisq, "paired-t": _ck_paired,
            "repeated-anova": _ck_rep_anova, "manova": _ck_manova}

def _run_check(req):
    ck = CHECKERS.get(req.test_type)
    if not ck:
        return {"test_type": req.test_type, "overall_pass": False, "assumptions": [],
                "recommendation": f"Unknown test type: {req.test_type}", "alternatives": [], "robustness_note": ""}
    asmps, rob, alts = ck(req.data, req.params)
    ok = all(a["passed"] for a in asmps)
    return {"test_type": req.test_type, "overall_pass": ok, "assumptions": asmps,
            "recommendation": f"Proceed with {req.test_type}" if ok else "Consider alternatives",
            "alternatives": alts, "robustness_note": rob}

# ── alternative runners ──────────────────────────────────────────────────────

def _alt_r(test, stat, pv, es, match=None):
    return {"test": test, "statistic": stat, "p_value": pv, "effect_size": es, "conclusion_matches": match}

def _run_ttest_alts(data, params):
    a, b = _get_groups(data, params)[0], _get_groups(data, params)[1]
    ts, tp = stats.ttest_ind(a, b, equal_var=True); d = round(_cohens_d(a, b), 4)
    primary = {"test": "Independent t-test", "statistic": round(float(ts), 4), "p_value": round(float(tp), 6), "effect_size": d}
    sig = tp < 0.05; alts = []
    wt, wp = stats.ttest_ind(a, b, equal_var=False)
    alts.append(_alt_r("Welch's t-test", round(float(wt), 4), round(float(wp), 6), d, (wp<0.05)==sig))
    us, up = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 2 * float(us) / (len(a) * len(b)) - 1  # rank-biserial, positive when group 1 tends to be larger
    alts.append(_alt_r("Mann-Whitney U (effect: rank-biserial r)", round(float(us), 4), round(float(up), 6), round(rb, 4), (up<0.05)==sig))
    rng = np.random.default_rng(42); comb = np.concatenate([a, b]); obs = abs(float(a.mean()-b.mean()))
    ct = 0
    for _ in range(N_RESAMPLE):
        perm = rng.permutation(comb)  # one shuffle splits into both groups
        ct += abs(perm[:len(a)].mean() - perm[len(a):].mean()) >= obs
    pp = _mc_p(int(ct))
    alts.append(_alt_r(f"Permutation test ({N_RESAMPLE})", round(float(a.mean()-b.mean()), 4), round(pp, 6), d, (pp<0.05)==sig))
    diffs = [rng.choice(a, len(a), replace=True).mean()-rng.choice(b, len(b), replace=True).mean() for _ in range(N_RESAMPLE)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boot = _alt_r("Bootstrap 95% CI of the mean difference", round(float(a.mean()-b.mean()), 4), None, d, (not(lo<=0<=hi))==sig)
    boot["ci"] = [round(float(lo), 4), round(float(hi), 4)]
    alts.append(boot)
    return primary, alts

def _run_anova_alts(data, params):
    groups = _get_groups(data, params)
    fs, fp = stats.f_oneway(*groups); es = round(_eta_sq(groups), 4)
    primary = {"test": "One-way ANOVA", "statistic": round(float(fs), 4), "p_value": round(float(fp), 6), "effect_size": es}
    sig = fp < 0.05; alts = []
    try:
        from statsmodels.stats.oneway import anova_oneway
        w = anova_oneway(groups, use_var="unequal", welch_correction=True)
        alts.append(_alt_r("Welch's ANOVA", round(float(w.statistic), 4), round(float(w.pvalue), 6), es, (w.pvalue<0.05)==sig))
    except Exception: pass
    ks, kp = stats.kruskal(*groups)
    alts.append(_alt_r("Kruskal-Wallis", round(float(ks), 4), round(float(kp), 6), es, (kp<0.05)==sig))
    # Bootstrap under H0: centre each group on the grand mean before resampling
    rng = np.random.default_rng(42); grand = np.concatenate(groups).mean()
    centred = [g - g.mean() + grand for g in groups]
    bfs = np.array([stats.f_oneway(*[rng.choice(g, len(g), replace=True) for g in centred])[0] for _ in range(N_RESAMPLE)])
    bp = _mc_p(int(np.sum(bfs >= fs)))
    alts.append(_alt_r(f"Bootstrap F-test (null-centred, {N_RESAMPLE})", round(float(fs), 4), round(bp, 6), es, (bp<0.05)==sig))
    return primary, alts

def _run_corr_alts(data, params):
    keys = sorted(data.keys())[:2]
    x, y = np.array(data[keys[0]], dtype=float), np.array(data[keys[1]], dtype=float)
    r, p = stats.pearsonr(x, y); r2 = round(float(r**2), 4)
    primary = {"test": "Pearson r", "statistic": round(float(r), 4), "p_value": round(float(p), 6), "effect_size": r2}
    sig = p < 0.05; alts = []
    rs, rsp = stats.spearmanr(x, y)
    alts.append(_alt_r("Spearman rho", round(float(rs), 4), round(float(rsp), 6), round(float(rs**2), 4), (rsp<0.05)==sig))
    kt, ktp = stats.kendalltau(x, y)
    alts.append(_alt_r("Kendall tau", round(float(kt), 4), round(float(ktp), 6), round(float(kt**2), 4), (ktp<0.05)==sig))
    try:
        pb, pbp = _percentage_bend(x, y, 0.2)
        alts.append(_alt_r("Percentage bend correlation (β = .2)", round(pb, 4), round(pbp, 6), round(pb**2, 4), (pbp<0.05)==sig))
    except Exception: pass
    return primary, alts

def _run_reg_alts(data, params):
    ivs = params.get("ivs", [k for k in sorted(data.keys()) if k != params.get("dv", sorted(data.keys())[-1])])
    dv = params.get("dv", sorted(data.keys())[-1])
    y = np.array(data[dv], dtype=float)
    X = np.column_stack([np.array(data[k], dtype=float) for k in ivs])
    Xc = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(Xc, y, rcond=None)[0]; res = y - Xc @ b
    ss_r, ss_t = np.sum(res**2), np.sum((y-y.mean())**2)
    r2 = 1 - ss_r/ss_t if ss_t else 0
    fs = ((ss_t-ss_r)/X.shape[1])/(ss_r/(len(y)-X.shape[1]-1)) if ss_r else 0
    fp = stats.f.sf(fs, X.shape[1], len(y)-X.shape[1]-1)
    primary = {"test": "OLS Regression", "statistic": round(float(fs), 4), "p_value": round(float(fp), 6), "effect_size": round(float(r2), 4)}
    alts = []; sig = fp < 0.05
    import statsmodels.api as sm
    try:
        hc3 = sm.OLS(y, Xc).fit(cov_type="HC3")
        wald = hc3.f_test(np.eye(Xc.shape[1])[1:])  # all slopes = 0, with HC3 covariance
        wf, wp = float(np.squeeze(wald.fvalue)), float(np.squeeze(wald.pvalue))
        alts.append(_alt_r("OLS with HC3 robust SEs (Wald F)", round(wf, 4), round(wp, 6), round(float(r2), 4), (wp<0.05)==sig))
    except Exception: pass
    try:
        rlm = sm.RLM(y, Xc, M=sm.robust.norms.HuberT()).fit()
        h = _alt_r("Robust regression (Huber M-estimator)", None, None, None, None)
        h["coefficients"] = [round(float(b_), 4) for b_ in rlm.params]
        h["p_values"] = [round(float(p_), 6) for p_ in rlm.pvalues]  # asymptotic z-tests
        alts.append(h)
    except Exception: pass
    return primary, alts

def _run_chi_alts(data, params):
    obs = np.array(list(data.values()), dtype=float)
    if obs.ndim == 1:
        cs, cp = stats.chisquare(obs)
        primary = {"test": "Chi-square GOF", "statistic": round(float(cs), 4), "p_value": round(float(cp), 6),
                   "effect_size": round(float(np.sqrt(cs/obs.sum())), 4)}
    else:
        cs, cp, _, _ = stats.chi2_contingency(obs); k = min(obs.shape)
        cv = np.sqrt(cs/(obs.sum()*(k-1))) if k > 1 else 0
        primary = {"test": "Chi-square test", "statistic": round(float(cs), 4), "p_value": round(float(cp), 6), "effect_size": round(float(cv), 4)}
    sig = cp < 0.05; alts = []; rng = np.random.default_rng(42)
    if obs.ndim == 2 and obs.shape == (2, 2):
        _, fp = stats.fisher_exact(obs.astype(int))
        alts.append(_alt_r("Fisher's exact", None, round(float(fp), 6), None, (fp<0.05)==sig))
    if obs.ndim == 2:
        rs, csl = obs.sum(axis=1), obs.sum(axis=0); tot = obs.sum()
        # Rows simulated with fixed row totals under independence (column margins vary); same Yates setting as the primary test
        mc = sum(stats.chi2_contingency(np.array([rng.multinomial(int(rs[i]), csl/tot) for i in range(obs.shape[0])])+1e-12)[0] >= cs for _ in range(N_RESAMPLE))
    else:
        exp = np.full_like(obs, obs.mean())
        mc = sum(stats.chisquare(rng.multinomial(int(obs.sum()), exp/exp.sum()))[0] >= cs for _ in range(N_RESAMPLE))
    mp = _mc_p(int(mc))
    alts.append(_alt_r("Monte Carlo chi-square", round(float(cs), 4), round(mp, 6), primary["effect_size"], (mp<0.05)==sig))
    return primary, alts

ALT_RUNNERS = {"t-test": _run_ttest_alts, "anova": _run_anova_alts, "correlation": _run_corr_alts,
               "regression": _run_reg_alts, "chi-square": _run_chi_alts}

# ── endpoints ────────────────────────────────────────────────────────────────

@router.post("/check")
async def check_assumptions(req: AssumptionGateRequest):
    return _run_check(req)

@router.post("/run-with-alternatives")
async def run_with_alternatives(req: RunWithAlternativesRequest):
    result = _run_check(req)
    runner = ALT_RUNNERS.get(req.test_type)
    if runner and req.run_alternatives:
        primary, alt_results = runner(req.data, req.params)
        result["primary_result"] = primary
        result["alternative_results"] = alt_results
        conc = [a.get("conclusion_matches") for a in alt_results if a.get("conclusion_matches") is not None]
        if conc and all(conc): result["convergence"] = "All methods agree"
        elif conc: result["convergence"] = "Results diverge — interpret with caution"
        else: result["convergence"] = "Not all methods provide p-values for comparison"
    else:
        result.update({"primary_result": None, "alternative_results": [], "convergence": "No alternative runner for this test type"})
    return result
