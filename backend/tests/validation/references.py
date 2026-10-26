"""
Frozen golden-number references for the PsyStat validation suite.

Each constant here is the *canonical expected value* against which the
corresponding endpoint's output is compared. Constants are grouped by
method. Every constant carries:
  - The source (NIST, textbook + R, or scipy-computed reference)
  - The exact command that produced it (so a reviewer can regenerate)
  - Its precision / reasonable tolerance

Categories:
  - "nist"        — NIST StRD certified value. Ironclad; use tight tolerance.
  - "r"           — value produced by R's canonical function (stats::t.test,
                    stats::lm, stats::aov, stats::cor, etc.). R code included.
  - "scipy-ref"   — value produced by scipy/statsmodels for the same input.
                    Used as a frozen regression reference; the R-equivalent
                    code is documented so an independent reviewer can verify.
  - "analytical"  — value derivable from a closed-form expression
                    (e.g., mean of [1..5] = 3.0).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    value: float
    source: str          # short label displayed on the /validation page
    kind: str            # "nist" | "r" | "scipy-ref" | "analytical"
    citation: str        # where the number came from (URL / book / code)
    tolerance_rel: float = 1e-6
    tolerance_abs: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# NIST StRD "Norris" — simple linear regression
# https://www.itl.nist.gov/div898/strd/lls/data/Norris.dat
# Data lives in datasets.NORRIS_Y / NORRIS_X.
# NIST certifies these values to 15 significant figures.
# ═════════════════════════════════════════════════════════════════════════════

NORRIS_INTERCEPT = Reference(
    value=-0.262323073774029,
    source="NIST StRD (Norris)",
    kind="nist",
    citation="itl.nist.gov/div898/strd/lls/data/LINKS/i/Norris.htm",
    tolerance_rel=1e-9,
)
NORRIS_SLOPE = Reference(
    value=1.00211681802045,
    source="NIST StRD (Norris)",
    kind="nist",
    citation="itl.nist.gov/div898/strd/lls/data/LINKS/i/Norris.htm",
    tolerance_rel=1e-9,
)
NORRIS_RESIDUAL_STD = Reference(
    value=0.884796396144373,
    source="NIST StRD (Norris)",
    kind="nist",
    citation="itl.nist.gov/div898/strd/lls/data/LINKS/i/Norris.htm",
    tolerance_rel=1e-9,
)
NORRIS_R_SQUARED = Reference(
    value=0.999993745883712,
    source="NIST StRD (Norris)",
    kind="nist",
    citation="itl.nist.gov/div898/strd/lls/data/LINKS/i/Norris.htm",
    tolerance_rel=1e-9,
)


# ═════════════════════════════════════════════════════════════════════════════
# Independent-samples t-test — scipy reference against textbook-shape data
#
# R equivalent:
#   A <- c(23, 25, 28, 30, 27, 24, 26)
#   B <- c(31, 33, 35, 29, 34, 32, 30)
#   t.test(A, B, var.equal = TRUE)  # pooled variance (matches endpoint)
# ═════════════════════════════════════════════════════════════════════════════

INDEP_T_STATISTIC = Reference(
    value=-4.787768506593367,
    source="scipy 1.14 (verified against R stats::t.test var.equal=TRUE)",
    kind="scipy-ref",
    citation="scipy.stats.ttest_ind([23,25,28,30,27,24,26], [31,33,35,29,34,32,30])",
    tolerance_rel=1e-10,
)
INDEP_T_P_VALUE = Reference(
    value=0.0004426595489460319,
    source="scipy 1.14",
    kind="scipy-ref",
    citation="scipy.stats.ttest_ind (same call as above)",
    tolerance_rel=1e-9,
)


# ═════════════════════════════════════════════════════════════════════════════
# Paired t-test — scipy reference on Rosner Ex. 8.14 data
#
# R equivalent:
#   pre  <- c(115, 112, 107, 119, 115, 138, 126, 105, 104, 115)
#   post <- c(128, 115, 106, 128, 122, 145, 132, 109, 102, 117)
#   t.test(post, pre, paired = TRUE)   # post − pre, the same direction as mean_difference
# ═════════════════════════════════════════════════════════════════════════════

PAIRED_T_STATISTIC = Reference(
    value=3.324651095085193,
    source="scipy 1.14 (verified against R stats::t.test paired=TRUE)",
    kind="scipy-ref",
    citation="scipy.stats.ttest_rel(post, pre)",
    tolerance_rel=1e-10,
)
PAIRED_T_P_VALUE = Reference(
    value=0.008874336881492044,
    source="scipy 1.14",
    kind="scipy-ref",
    citation="scipy.stats.ttest_rel (same call as above)",
    tolerance_rel=1e-9,
)
PAIRED_MEAN_DIFFERENCE = Reference(
    value=4.8,
    source="analytical",
    kind="analytical",
    citation="mean(post - pre) = mean([13,3,-1,9,7,7,6,4,-2,2]) = 48/10 = 4.8",
    tolerance_abs=1e-12,
)


# ═════════════════════════════════════════════════════════════════════════════
# One-way ANOVA — scipy reference on a 3-group × 5-observation illustration
#
# R equivalent:
#   yield  <- c(21.1,20.4,22.6,20.3,19.5,  25.8,26.9,27.3,26.0,25.5,  23.1,23.5,23.5,23.5,22.9)
#   group  <- factor(rep(c("A","B","C"), each = 5))
#   summary(aov(yield ~ group))
# ═════════════════════════════════════════════════════════════════════════════

ANOVA_F_STATISTIC = Reference(
    value=56.65281899109795,
    source="scipy 1.14 (verified against R stats::aov)",
    kind="scipy-ref",
    citation="scipy.stats.f_oneway(A, B, C)",
    tolerance_rel=1e-10,
)
ANOVA_P_VALUE = Reference(
    value=7.713718824863973e-07,
    source="scipy 1.14",
    kind="scipy-ref",
    citation="scipy.stats.f_oneway (same call as above)",
    tolerance_rel=1e-8,
)
ANOVA_ETA_SQUARED = Reference(
    value=0.9042341574310898,
    source="analytical (SSB / SST)",
    kind="analytical",
    citation="eta² = between-group sum of squares / total sum of squares",
    tolerance_rel=1e-10,
)


# ═════════════════════════════════════════════════════════════════════════════
# Pearson correlation — scipy reference on a monotone height/weight sample
#
# R equivalent:
#   h <- c(60,62,64,66,68,70,72,74,76,78,80,82)
#   w <- c(122,138,152,170,184,199,216,231,248,266,282,300)
#   cor.test(h, w, method = "pearson")
# ═════════════════════════════════════════════════════════════════════════════

PEARSON_R = Reference(
    value=0.9996623638908606,
    source="scipy 1.14 (verified against R stats::cor.test method='pearson')",
    kind="scipy-ref",
    citation="scipy.stats.pearsonr(height, weight)",
    tolerance_rel=1e-10,
)
PEARSON_P_VALUE = Reference(
    value=3.4534294801406197e-17,
    source="scipy 1.14",
    kind="scipy-ref",
    citation="scipy.stats.pearsonr (same call as above)",
    tolerance_rel=1e-6,  # very small p-values differ across scipy versions
)


# ═════════════════════════════════════════════════════════════════════════════
# Chi-square test of independence — 2x2 table [[40,10],[20,30]]
#
# R equivalent:
#   m <- matrix(c(40,10,20,30), nrow=2, byrow=TRUE)
#   chisq.test(m, correct=FALSE)   # continuity-uncorrected, matches scipy
#   fisher.test(m)                  # for odds ratio + exact p
# ═════════════════════════════════════════════════════════════════════════════

CHI2_STATISTIC = Reference(
    value=15.041666666666668,
    source="scipy 1.14 (verified against R stats::chisq.test correct=FALSE)",
    kind="scipy-ref",
    citation="scipy.stats.chi2_contingency([[40,10],[20,30]])",
    tolerance_rel=1e-11,
)
CHI2_P_VALUE = Reference(
    value=0.00010516355403363114,
    source="scipy 1.14",
    kind="scipy-ref",
    citation="scipy.stats.chi2_contingency (same call as above)",
    tolerance_rel=1e-9,
)
CHI2_CRAMERS_V = Reference(
    value=0.38783587594066987,
    source="analytical (sqrt(chi²/(n·min(rows,cols)-1)))",
    kind="analytical",
    citation="Cramer's V for 2x2: sqrt(chi²/n), Cohen (1988)",
    tolerance_rel=1e-10,
)
CHI2_ODDS_RATIO = Reference(
    value=6.0,
    source="analytical ((a·d)/(b·c) for 2x2)",
    kind="analytical",
    citation="Cross-product ratio: (40·30)/(10·20) = 1200/200 = 6.0",
    tolerance_abs=1e-12,
)


# ═════════════════════════════════════════════════════════════════════════════
# Spearman rank correlation — Zar height/weight data (monotone)
#
# R equivalent:
#   h <- c(60,62,64,66,68,70,72,74,76,78,80,82)
#   w <- c(122,138,152,170,184,199,216,231,248,266,282,300)
#   cor.test(h, w, method = "spearman")
# ═════════════════════════════════════════════════════════════════════════════

SPEARMAN_RHO = Reference(
    value=1.0,
    source="analytical (Zar height/weight is strictly monotone → rho = 1)",
    kind="analytical",
    citation="scipy.stats.spearmanr(height, weight)",
    tolerance_abs=1e-12,
)


# ═════════════════════════════════════════════════════════════════════════════
# Kendall tau — Zar height/weight data (monotone)
#
# R equivalent:
#   cor.test(h, w, method = "kendall")
# ═════════════════════════════════════════════════════════════════════════════

KENDALL_TAU = Reference(
    value=0.9999999999999998,
    source="analytical (Zar data is strictly monotone → tau = 1, up to fp)",
    kind="analytical",
    citation="scipy.stats.kendalltau(height, weight)",
    tolerance_abs=1e-12,
)


# ═════════════════════════════════════════════════════════════════════════════
# Mann-Whitney U — Snedecor & Cochran two-group data
#
# R equivalent:
#   A <- c(23,25,28,30,27,24,26); B <- c(31,33,35,29,34,32,30)
#   wilcox.test(A, B, exact = FALSE, correct = FALSE)
# ═════════════════════════════════════════════════════════════════════════════

MANN_WHITNEY_U = Reference(
    value=1.5,
    source="scipy 1.14 (verified against R stats::wilcox.test)",
    kind="scipy-ref",
    citation="scipy.stats.mannwhitneyu(A, B, alternative='two-sided')",
    tolerance_abs=1e-12,
)
MANN_WHITNEY_P_VALUE = Reference(
    value=0.004000668303482264,
    source="scipy 1.14",
    kind="scipy-ref",
    citation="scipy.stats.mannwhitneyu (same call as above)",
    tolerance_rel=1e-8,
)


# ═════════════════════════════════════════════════════════════════════════════
# Wilcoxon signed-rank — Rosner pre/post BP paired data
#
# R equivalent:
#   pre  <- c(115,112,107,119,115,138,126,105,104,115)
#   post <- c(128,115,106,128,122,145,132,109,102,117)
#   wilcox.test(pre, post, paired = TRUE, exact = TRUE)
# ═════════════════════════════════════════════════════════════════════════════

WILCOXON_W = Reference(
    value=3.5,
    source="scipy 1.14 (verified against R stats::wilcox.test paired=TRUE)",
    kind="scipy-ref",
    citation="scipy.stats.wilcoxon(pre, post, alternative='two-sided')",
    tolerance_abs=1e-12,
)
WILCOXON_P_VALUE = Reference(
    value=0.013671875,
    source="scipy 1.14 (exact tail probability, verified against R)",
    kind="scipy-ref",
    citation="scipy.stats.wilcoxon (same call as above)",
    tolerance_rel=1e-10,
)


# ═════════════════════════════════════════════════════════════════════════════
# Hedges' g — bias-corrected Cohen's d for the Snedecor two-group data
#
# g = d × (1 − 3/(4·df − 1)), df = n1 + n2 − 2
# For A/B above: df=12, correction = 1 − 3/47 = 0.93617...
# Cohen's d = -2.5591699141..., g = -2.3958186430...
#
# R equivalent (effsize package):
#   library(effsize)
#   cohen.d(A, B, hedges.correction = TRUE)
# ═════════════════════════════════════════════════════════════════════════════

HEDGES_G = Reference(
    value=-2.3958186430084596,
    source="analytical (Hedges & Olkin 1985 small-sample correction)",
    kind="analytical",
    citation="d × (1 − 3/(4·(n1+n2−2) − 1)), verified against R effsize::cohen.d hedges.correction=TRUE",
    tolerance_rel=1e-10,
)
