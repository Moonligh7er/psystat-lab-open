"""APA 7th Edition formatter — generates publication-ready text from statistical results."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import math

router = APIRouter()


def _fmt(val: float, decimals: int = 2) -> str:
    """Format number, strip leading zero for values bounded by ±1."""
    s = f"{val:.{decimals}f}"
    if -1 < val < 1 and val != 0:
        s = s.lstrip("0").lstrip("-0")
        if val < 0:
            s = "-" + s
    return s


def _p_fmt(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    return f"p = {_fmt(p, 3)}"


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _eta_label(eta: float) -> str:
    if eta < 0.01:
        return "negligible"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"


# ── Request Models ────────────────────────────────────────────────────

class APARequest(BaseModel):
    test_type: str
    results: dict
    alpha: float = 0.05


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/format")
def format_apa(req: APARequest):
    r = req.results
    tt = req.test_type.lower().replace("-", "").replace("_", "").replace(" ", "")
    lines = []
    interpretation = ""

    if tt in ("ttest", "independentttest"):
        t = r.get("t_statistic", 0)
        df = r.get("df", 0)
        p = r.get("p_value", 1)
        d = r.get("cohens_d", 0)
        m1 = r.get("group1_mean", 0)
        m2 = r.get("group2_mean", 0)
        sd1 = r.get("group1_std", 0)
        sd2 = r.get("group2_std", 0)
        n1 = r.get("group1_n", 0)
        n2 = r.get("group2_n", 0)

        lines.append(f"Group 1 (M = {m1:.2f}, SD = {sd1:.2f}, n = {n1}) vs. Group 2 (M = {m2:.2f}, SD = {sd2:.2f}, n = {n2}).")
        lines.append(f"An independent-samples t-test revealed a {'statistically significant' if p < req.alpha else 'non-significant'} difference, t({df}) = {t:.2f}, {_p_fmt(p)}, d = {_fmt(d)}.")
        lines.append(f"The effect size was {_effect_label(d)} (Cohen's d = {_fmt(d)}).")

        lev_p = r.get("levene_p")
        if lev_p is not None:
            if lev_p < 0.05:
                lines.append(f"Note: Levene's test indicated unequal variances, F = {r.get('levene_statistic', 0):.2f}, {_p_fmt(lev_p)}. Consider Welch's t-test.")
            else:
                lines.append(f"Levene's test confirmed homogeneity of variances, F = {r.get('levene_statistic', 0):.2f}, {_p_fmt(lev_p)}.")

        interpretation = "significant" if p < req.alpha else "non-significant"

    elif tt in ("pairedttest", "paired"):
        t = r.get("t_statistic", 0)
        df = r.get("df", 0)
        p = r.get("p_value", 1)
        d = r.get("cohens_d", 0)
        md = r.get("mean_difference", 0)
        n = r.get("n_pairs", 0)

        lines.append(f"A paired-samples t-test was conducted on {n} paired observations.")
        lines.append(f"Results indicated a {'statistically significant' if p < req.alpha else 'non-significant'} mean difference (M_diff = {md:.2f}), t({df}) = {t:.2f}, {_p_fmt(p)}, d = {_fmt(d)}.")
        lines.append(f"The effect size was {_effect_label(d)} (Cohen's d = {_fmt(d)}).")
        interpretation = "significant" if p < req.alpha else "non-significant"

    elif tt in ("anova", "onewayanova"):
        f_val = r.get("f_statistic", 0)
        p = r.get("p_value", 1)
        eta = r.get("eta_squared", 0)
        dfb = r.get("df_between", 0)
        dfw = r.get("df_within", 0)
        k = len(r.get("group_means", []))

        lines.append(f"A one-way ANOVA was conducted across {k} groups.")
        lines.append(f"There was a {'statistically significant' if p < req.alpha else 'non-significant'} effect, F({dfb}, {dfw}) = {f_val:.2f}, {_p_fmt(p)}, η² = {_fmt(eta)}.")
        lines.append(f"The effect size was {_eta_label(eta)} (η² = {_fmt(eta)}).")

        lev_p = r.get("levene_p")
        if lev_p is not None and lev_p < 0.05:
            lines.append(f"Levene's test was significant ({_p_fmt(lev_p)}), suggesting unequal variances. Consider Welch's ANOVA or non-parametric alternatives.")

        tukey = r.get("tukey_pairwise", [])
        sig_pairs = [t for t in tukey if t.get("significant")]
        if sig_pairs:
            pairs_str = ", ".join(f"Group {t['group_i']+1} vs. {t['group_j']+1} ({_p_fmt(t['p_value'])})" for t in sig_pairs)
            lines.append(f"Post-hoc Tukey HSD tests revealed significant pairwise differences: {pairs_str}.")

        interpretation = "significant" if p < req.alpha else "non-significant"

    elif tt in ("chisquare", "chi2"):
        chi2 = r.get("chi2_statistic", 0)
        p = r.get("p_value", 1)
        df = r.get("df", 0)
        v = r.get("cramers_v", 0)

        lines.append(f"A chi-square test of independence was performed.")
        lines.append(f"The association was {'statistically significant' if p < req.alpha else 'not statistically significant'}, χ²({df}) = {chi2:.2f}, {_p_fmt(p)}, V = {_fmt(v)}.")
        lines.append(f"The effect size (Cramér's V = {_fmt(v)}) suggests a {'small' if v < 0.3 else 'medium' if v < 0.5 else 'large'} association.")
        interpretation = "significant" if p < req.alpha else "non-significant"

    elif tt in ("regression", "linearregression"):
        r2 = r.get("r_squared", 0)
        adj_r2 = r.get("adj_r_squared", 0)
        f_val = r.get("f_statistic", 0)
        f_p = r.get("f_p_value", 1)
        dfm = r.get("df_model", 0)
        dfr = r.get("df_resid", 0)
        n = r.get("n_observations", 0)

        lines.append(f"A {'simple' if dfm == 1 else 'multiple'} linear regression was conducted (N = {n}).")
        lines.append(f"The overall model was {'statistically significant' if f_p < req.alpha else 'non-significant'}, F({dfm}, {dfr}) = {f_val:.2f}, {_p_fmt(f_p)}, R² = {_fmt(r2)}, adjusted R² = {_fmt(adj_r2)}.")
        lines.append(f"The model explained {r2*100:.1f}% of the variance in the outcome variable.")

        coeffs = r.get("coefficients", [])
        p_vals = r.get("p_values", [])
        t_vals = r.get("t_values", [])
        ses = r.get("std_errors", [])
        for i in range(1, len(coeffs)):
            b = coeffs[i]
            se = ses[i] if i < len(ses) else 0
            t = t_vals[i] if i < len(t_vals) else 0
            p = p_vals[i] if i < len(p_vals) else 1
            lines.append(f"  Predictor {i}: B = {b:.3f}, SE = {se:.3f}, t = {t:.2f}, {_p_fmt(p)}.")

        interpretation = "significant" if f_p < req.alpha else "non-significant"

    elif tt in ("mannwhitney", "mannwhitneyu", "wilcoxon", "kruskalwallis", "friedman"):
        stat = r.get("statistic", 0)
        p = r.get("p_value", 1)
        es = r.get("effect_size", 0)
        n = r.get("n_total", r.get("n", 0))

        test_names = {
            "mannwhitney": "Mann-Whitney U",
            "mannwhitneyu": "Mann-Whitney U",
            "wilcoxon": "Wilcoxon signed-rank",
            "kruskalwallis": "Kruskal-Wallis H",
            "friedman": "Friedman",
        }
        name = test_names.get(tt, tt)
        stat_sym = "U" if "mann" in tt else "W" if "wilcox" in tt else "H" if "kruskal" in tt else "χ²"

        lines.append(f"A {name} test was conducted (N = {n}).")
        lines.append(f"The test was {'statistically significant' if p < req.alpha else 'not statistically significant'}, {stat_sym} = {stat:.2f}, {_p_fmt(p)}.")
        if es:
            lines.append(f"Effect size: r = {_fmt(es)}.")
        interpretation = "significant" if p < req.alpha else "non-significant"

    elif tt in ("correlation", "correlationmatrix", "spearman", "kendall", "pearson"):
        r_val = r.get("r", r.get("correlation", 0))
        p = r.get("p_value", 1)
        n = r.get("n", 0)
        method = r.get("method", "Pearson").capitalize()

        lines.append(f"A {method} correlation was computed (N = {n}).")
        lines.append(f"There was a {'statistically significant' if p < req.alpha else 'non-significant'} {'positive' if r_val > 0 else 'negative'} correlation, r = {_fmt(r_val)}, {_p_fmt(p)}.")

        ar = abs(r_val)
        strength = "weak" if ar < 0.3 else "moderate" if ar < 0.7 else "strong"
        lines.append(f"The correlation strength was {strength} (r = {_fmt(r_val)}).")
        interpretation = "significant" if p < req.alpha else "non-significant"

    elif tt in ("metaanalysis", "meta"):
        pooled = r.get("pooled_effect", 0)
        ci_lo = r.get("ci_lower", 0)
        ci_hi = r.get("ci_upper", 0)
        q = r.get("Q", 0)
        q_p = r.get("Q_p", 1)
        i2 = r.get("I_squared", 0)
        k = r.get("k_studies", 0)

        lines.append(f"A meta-analysis was conducted across {k} studies.")
        lines.append(f"The pooled effect size was {pooled:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}].")
        het = "low" if i2 < 25 else "moderate" if i2 < 75 else "high"
        lines.append(f"Heterogeneity was {het}, Q({k-1}) = {q:.2f}, {_p_fmt(q_p)}, I² = {i2:.1f}%.")
        interpretation = f"pooled effect {'significant' if (ci_lo > 0 or ci_hi < 0) else 'includes zero'}"

    elif tt in ("mediation",):
        indirect = r.get("indirect_effect", 0)
        sobel_z = r.get("sobel_z", 0)
        sobel_p = r.get("sobel_p", 1)
        total = r.get("total_effect", 0)
        direct = r.get("direct_effect", 0)
        proportion = r.get("proportion_mediated", 0)

        lines.append(f"A mediation analysis was conducted (Baron & Kenny approach).")
        lines.append(f"The total effect was c = {total:.3f}, the direct effect was c' = {direct:.3f}, and the indirect effect was ab = {indirect:.3f}.")
        lines.append(f"The Sobel test was {'significant' if sobel_p < req.alpha else 'non-significant'}, z = {sobel_z:.2f}, {_p_fmt(sobel_p)}.")
        if proportion:
            lines.append(f"The mediator accounted for {proportion*100:.1f}% of the total effect.")
        interpretation = "mediation supported" if sobel_p < req.alpha else "mediation not supported"

    else:
        lines.append(f"APA formatting for '{req.test_type}' is not yet supported. Raw results returned.")
        interpretation = "unknown"

    return {
        "apa_text": " ".join(lines),
        "apa_lines": lines,
        "interpretation": interpretation,
        "test_type": req.test_type,
    }
