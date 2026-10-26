"""Research Narrative Builder — generates APA-style method and results sections
from a sequence of statistical analyses.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Models ───────────────────────────────────────────────────────────────

class AnalysisEntry(BaseModel):
    module: str
    test_type: str
    results: Dict[str, Any]
    order: int
    section: str  # hypothesis_test, assumption_check, effect_size, sensitivity, exploratory

class NarrativeRequest(BaseModel):
    analyses: List[AnalysisEntry]

class NarrativeResponse(BaseModel):
    method_section: str
    results_section: str
    method_sentences: List[str]  # the same text, one sentence per item
    results_sentences: List[str]
    individual_sentences: List[str]
    apa_tables: List[str]

# ── Helpers ──────────────────────────────────────────────────────────────

def _first(r: Dict[str, Any], *keys):
    """First key present with a non-None value (unlike `a or b`, keeps legitimate zeros)."""
    for k in keys:
        if r.get(k) is not None:
            return r[k]
    return None


def _fmt_p(p: float) -> str:
    """Format p-value in APA style (no leading zero; never rounds across .05)."""
    if p is None:
        return "p = NA"
    p = float(p)
    if p < .001:
        return "p < .001"
    if p > .999:
        return "p > .999"
    s = f"{p:.3f}"
    if (p < .05) != (float(s) < .05):  # e.g. .0496 would round to .050
        s = f"{p:.4f}"
    return "p = " + s.lstrip("0")

def _fmt(v: float, decimals: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{v:.{decimals}f}"

def _is_sig(p: Optional[float], alpha: float = .05) -> bool:
    if p is None:
        return False
    return p < alpha

def _sig_word(p: Optional[float]) -> str:
    return "significant" if _is_sig(p) else "not significant"

SECTION_ORDER = [
    "assumption_check",
    "hypothesis_test",
    "effect_size",
    "sensitivity",
    "exploratory",
]

SECTION_LABELS = {
    "assumption_check": "Assumption Checks",
    "hypothesis_test": "Hypothesis Tests",
    "effect_size": "Effect Sizes",
    "sensitivity": "Sensitivity Analyses",
    "exploratory": "Exploratory Analyses",
}

SECTION_TRANSITIONS = {
    "assumption_check": "Prior to hypothesis testing, distributional assumptions were examined.",
    "hypothesis_test": "The primary analyses were then conducted.",
    "effect_size": "Effect sizes were computed to quantify the magnitude of the observed effects.",
    "sensitivity": "Sensitivity analyses were performed to evaluate the robustness of the findings.",
    "exploratory": "Additional exploratory analyses were conducted.",
}

# ── APA sentence generators ─────────────────────────────────────────────

def _apa_ttest(r: Dict[str, Any]) -> str:
    t = _first(r, "t_statistic", "t")
    df = r.get("df")
    p = _first(r, "p_value", "p")
    d = _first(r, "cohens_d", "d")
    sig = _sig_word(p)
    base = f"An independent-samples t-test {'revealed a significant' if _is_sig(p) else 'did not reveal a significant'} difference, t({_fmt(df, 0) if df else '?'}) = {_fmt(t)}, {_fmt_p(p)}"
    if d is not None:
        base += f", d = {_fmt(d)}"
    return base + "."

def _apa_paired_ttest(r: Dict[str, Any]) -> str:
    t = _first(r, "t_statistic", "t")
    df = r.get("df")
    p = _first(r, "p_value", "p")
    d = _first(r, "cohens_d", "d")
    sig = _sig_word(p)
    base = f"A paired-samples t-test {'indicated a significant' if _is_sig(p) else 'did not indicate a significant'} difference, t({_fmt(df, 0) if df else '?'}) = {_fmt(t)}, {_fmt_p(p)}"
    if d is not None:
        base += f", d = {_fmt(d)}"
    return base + "."

def _apa_anova(r: Dict[str, Any]) -> str:
    f_val = _first(r, "f_statistic", "F", "f")
    df1 = _first(r, "df1", "df_between")
    df2 = _first(r, "df2", "df_within")
    p = _first(r, "p_value", "p")
    eta = _first(r, "eta_squared", "eta_sq")
    sig = _sig_word(p)
    base = f"A one-way ANOVA {'showed a significant' if _is_sig(p) else 'did not show a significant'} main effect, F({_fmt(df1, 0) if df1 else '?'}, {_fmt(df2, 0) if df2 else '?'}) = {_fmt(f_val)}, {_fmt_p(p)}"
    if eta is not None:
        base += f", \u03b7\u00b2 = {_fmt(eta).replace('0.', '.', 1)}"
    return base + "."

def _apa_correlation(r: Dict[str, Any]) -> str:
    rv = _first(r, "r", "correlation")
    p = _first(r, "p_value", "p")
    n = _first(r, "n", "N")
    df = _first(r, "df")
    sig = _sig_word(p)
    direction = "positive" if (rv or 0) > 0 else "negative"
    base = (f"A Pearson correlation indicated a significant {direction} relationship, r" if _is_sig(p)
            else "A Pearson correlation did not indicate a significant relationship, r")
    if df is not None:
        base += f"({int(df)})"
    elif n is not None:
        base += f"({int(n) - 2})"  # APA reports r with its degrees of freedom, n − 2
    base += f" = {_fmt(rv).replace('0.', '.', 1)}, {_fmt_p(p)}"  # APA: no leading zero for r
    return base + "."

def _apa_regression(r: Dict[str, Any]) -> str:
    f_val = _first(r, "f_statistic", "F", "f")
    df1 = _first(r, "df1", "df_model")
    df2 = _first(r, "df2", "df_resid")
    p = _first(r, "p_value", "p", "f_p", "f_p_value")
    r_sq = _first(r, "r_squared", "R2", "r2")
    sig = _sig_word(p)
    base = f"A linear regression was {sig}, F({_fmt(df1, 0) if df1 else '?'}, {_fmt(df2, 0) if df2 else '?'}) = {_fmt(f_val)}, {_fmt_p(p)}"
    if r_sq is not None:
        base += f", R\u00b2 = {_fmt(r_sq).replace('0.', '.', 1)}"
    return base + "."

def _apa_chi_square(r: Dict[str, Any]) -> str:
    chi = _first(r, "chi_square", "chi2", "statistic")
    df = r.get("df")
    p = _first(r, "p_value", "p")
    v = _first(r, "cramers_v", "V", "v")
    sig = _sig_word(p)
    base = f"A chi-square test of independence was {sig}, \u03c7\u00b2({_fmt(df, 0) if df else '?'}) = {_fmt(chi)}, {_fmt_p(p)}"
    if v is not None:
        base += f", V = {_fmt(v).replace('0.', '.', 1)}"
    return base + "."

def _apa_mann_whitney(r: Dict[str, Any]) -> str:
    u = _first(r, "U", "statistic")
    p = _first(r, "p_value", "p")
    sig = _sig_word(p)
    rr = _first(r, "rank_biserial", "r")
    base = f"A Mann-Whitney U test was {sig}, U = {_fmt(u)}, {_fmt_p(p)}"
    if rr is not None:
        base += f", r = {_fmt(rr)}"
    return base + "."

def _apa_wilcoxon(r: Dict[str, Any]) -> str:
    w = _first(r, "W", "statistic")
    p = _first(r, "p_value", "p")
    sig = _sig_word(p)
    base = f"A Wilcoxon signed-rank test was {sig}, W = {_fmt(w)}, {_fmt_p(p)}"
    return base + "."

def _apa_kruskal(r: Dict[str, Any]) -> str:
    h = _first(r, "H", "statistic")
    df = r.get("df")
    p = _first(r, "p_value", "p")
    sig = _sig_word(p)
    base = f"A Kruskal-Wallis test was {sig}, H({_fmt(df, 0) if df else '?'}) = {_fmt(h)}, {_fmt_p(p)}"
    return base + "."

def _apa_shapiro(r: Dict[str, Any]) -> str:
    w = _first(r, "W", "statistic")
    p = _first(r, "p_value", "p")
    verdict = ("indicating no significant departure from normality" if not _is_sig(p)
               else "indicating a significant departure from normality")
    return f"A Shapiro-Wilk test indicated W = {_fmt(w)}, {_fmt_p(p)}, {verdict}."

def _apa_levene(r: Dict[str, Any]) -> str:
    f_val = _first(r, "F", "statistic")
    p = _first(r, "p_value", "p")
    verdict = ("indicating no significant difference in variances" if not _is_sig(p)
               else "indicating unequal variances")
    return f"Levene's test was {_sig_word(p)}, F = {_fmt(f_val)}, {_fmt_p(p)}, {verdict}."

def _apa_generic(test_type: str, r: Dict[str, Any]) -> str:
    stat = _first(r, "statistic", "t", "F", "chi2")
    p = _first(r, "p_value", "p")
    sig = _sig_word(p)
    base = f"The {test_type} was {sig}"
    if stat is not None:
        base += f", statistic = {_fmt(stat)}"
    if p is not None:
        base += f", {_fmt_p(p)}"
    return base + "."


APA_GENERATORS = {
    "t-test": _apa_ttest,
    "ttest": _apa_ttest,
    "independent_ttest": _apa_ttest,
    "independent-samples t-test": _apa_ttest,
    "paired_ttest": _apa_paired_ttest,
    "paired": _apa_paired_ttest,
    "paired-t": _apa_paired_ttest,
    "paired t-test": _apa_paired_ttest,
    "anova": _apa_anova,
    "one-way anova": _apa_anova,
    "oneway_anova": _apa_anova,
    "correlation": _apa_correlation,
    "pearson": _apa_correlation,
    "pearson_correlation": _apa_correlation,
    "regression": _apa_regression,
    "linear_regression": _apa_regression,
    "chi-square": _apa_chi_square,
    "chi_square": _apa_chi_square,
    "mann-whitney": _apa_mann_whitney,
    "mann_whitney": _apa_mann_whitney,
    "wilcoxon": _apa_wilcoxon,
    "kruskal-wallis": _apa_kruskal,
    "kruskal_wallis": _apa_kruskal,
    "shapiro-wilk": _apa_shapiro,
    "shapiro_wilk": _apa_shapiro,
    "shapiro": _apa_shapiro,
    "levene": _apa_levene,
    "levenes": _apa_levene,
}


def _generate_sentence(entry: AnalysisEntry) -> str:
    key = entry.test_type.lower().strip()
    gen = APA_GENERATORS.get(key)
    if gen:
        return gen(entry.results)
    return _apa_generic(entry.test_type, entry.results)


def _build_apa_table(entries: List[AnalysisEntry]) -> List[str]:
    """Build simple APA-style result summary tables."""
    tables: List[str] = []
    # Group hypothesis tests into a summary table
    hyp = [e for e in entries if e.section == "hypothesis_test"]
    if hyp:
        header = f"{'Test':<30} {'Statistic':<15} {'p':<10} {'Effect Size':<12}"
        sep = "-" * len(header)
        rows = [header, sep]
        for e in hyp:
            r = e.results
            stat = _first(r, "t_statistic", "f_statistic", "chi_square", "statistic", "t", "F")
            p = _first(r, "p_value", "p")
            es = _first(r, "cohens_d", "eta_squared", "r", "cramers_v", "r_squared")
            rows.append(f"{e.test_type:<30} {str(round(float(stat), 3)) if stat is not None else 'NA':<15} {_fmt_p(p).replace('p = ', '').replace('p ', '') if p is not None else 'NA':<10} {str(round(float(es), 3)) if es is not None else 'NA':<12}")
        tables.append("\n".join(rows))
    return tables


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.post("/generate", response_model=NarrativeResponse)
async def generate_narrative(req: NarrativeRequest):
    if not req.analyses:
        raise HTTPException(status_code=400, detail="At least one analysis entry is required.")

    sorted_entries = sorted(req.analyses, key=lambda e: e.order)

    # Generate individual sentences
    individual_sentences: List[str] = []
    for entry in sorted_entries:
        individual_sentences.append(_generate_sentence(entry))

    # Group by section
    sections: Dict[str, List[AnalysisEntry]] = {}
    for entry in sorted_entries:
        sections.setdefault(entry.section, []).append(entry)

    # Build method section
    modules_used = list({e.module for e in sorted_entries})
    tests_used = list({e.test_type for e in sorted_entries})
    method_parts = [
        f"Statistical analyses were conducted using the following modules: {', '.join(modules_used)}.",
        f"The analytic strategy included {', '.join(tests_used)}.",
    ]
    if "assumption_check" in sections:
        assumption_tests = [e.test_type for e in sections["assumption_check"]]
        method_parts.append(
            f"Distributional assumptions were evaluated using {', '.join(assumption_tests)}."
        )
    method_parts.append(
        f"An alpha level of .05 was used for all statistical tests unless otherwise noted."
    )
    method_section = " ".join(method_parts)

    # Build results section
    results_parts: List[str] = []
    for sec_key in SECTION_ORDER:
        if sec_key not in sections:
            continue
        transition = SECTION_TRANSITIONS.get(sec_key, "")
        if transition:
            results_parts.append(transition)
        for entry in sections[sec_key]:
            results_parts.append(_generate_sentence(entry))
        results_parts.append("")  # blank line between sections

    results_section = " ".join(p for p in results_parts if p)
    results_sentences = [p for p in results_parts if p]

    # Build APA tables
    apa_tables = _build_apa_table(sorted_entries)

    return NarrativeResponse(
        method_section=method_section,
        results_section=results_section,
        method_sentences=method_parts,
        results_sentences=results_sentences,
        individual_sentences=individual_sentences,
        apa_tables=apa_tables,
    )
