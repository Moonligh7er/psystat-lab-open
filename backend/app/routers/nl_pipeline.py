"""Natural language → analysis pipeline — parses user queries and routes to
the appropriate PsyStat statistical module / endpoint."""

from __future__ import annotations

import math
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class NLQueryRequest(BaseModel):
    query: str
    available_variables: list[str] | None = None
    data_summary: dict | None = None  # {col: {type, n, mean, sd, ...}}


class AnalysisAlternative(BaseModel):
    analysis_type: str
    reason: str


class ParseQueryResponse(BaseModel):
    analysis_type: str
    confidence: float
    parameters: dict
    explanation: str
    alternatives: list[AnalysisAlternative]
    module: str
    endpoint: str


class ColumnProfile(BaseModel):
    name: str
    type: str  # "continuous", "categorical", "ordinal"
    n_unique: int | None = None
    n_missing: int | None = None
    mean: float | None = None
    sd: float | None = None


class DataProfileRequest(BaseModel):
    columns: list[ColumnProfile]
    n_rows: int
    research_goal: str | None = None


class AnalysisSuggestion(BaseModel):
    analysis: str
    reason: str
    module: str
    endpoint: str
    priority: int  # 1 = highest


class DataIssue(BaseModel):
    column: str
    issue: str
    recommendation: str


class SuggestAnalysisResponse(BaseModel):
    suggestions: list[AnalysisSuggestion]
    data_issues: list[DataIssue]
    power_note: str


# ---------------------------------------------------------------------------
# Analysis catalogue — maps analysis types to modules + endpoints
# ---------------------------------------------------------------------------

_ANALYSIS_REGISTRY: dict[str, dict] = {
    "t_test": {
        "module": "stats_studio",
        "endpoint": "/stats/t-test",
        "description": "Independent-samples t-test comparing two group means",
    },
    "paired_t_test": {
        "module": "stats_studio",
        "endpoint": "/stats/paired-t-test",
        "description": "Paired-samples t-test for repeated measures on two occasions",
    },
    "anova": {
        "module": "stats_studio",
        "endpoint": "/stats/anova",
        "description": "One-way ANOVA comparing means across 3+ groups",
    },
    "correlation": {
        "module": "stats_studio",
        "endpoint": "/stats/correlation",
        "description": "Bivariate correlation (Pearson/Spearman)",
    },
    "regression": {
        "module": "stats_studio",
        "endpoint": "/stats/regression",
        "description": "Multiple linear regression",
    },
    "chi_square": {
        "module": "stats_studio",
        "endpoint": "/stats/chi-square",
        "description": "Chi-square test of independence for categorical variables",
    },
    "mixed_anova": {
        "module": "advanced_stats",
        "endpoint": "/advanced/mixed-anova",
        "description": "Mixed-design ANOVA with between- and within-subjects factors",
    },
    "hlm": {
        "module": "hlm",
        "endpoint": "/hlm/fit",
        "description": "Hierarchical linear model (multilevel / mixed-effects)",
    },
    "mediation": {
        "module": "advanced_stats",
        "endpoint": "/advanced/mediation",
        "description": "Mediation analysis (Baron & Kenny / bootstrapped indirect effect)",
    },
    "moderation": {
        "module": "advanced_stats",
        "endpoint": "/advanced/moderation",
        "description": "Moderation analysis (interaction effect)",
    },
    "sem": {
        "module": "sem",
        "endpoint": "/sem/fit",
        "description": "Structural equation modeling",
    },
    "factor_analysis": {
        "module": "psychometrics",
        "endpoint": "/psychometrics/efa",
        "description": "Exploratory factor analysis for scale development",
    },
    "reliability": {
        "module": "psychometrics",
        "endpoint": "/psychometrics/reliability",
        "description": "Internal consistency reliability (Cronbach's alpha, omega)",
    },
    "bayesian_t_test": {
        "module": "bayesian",
        "endpoint": "/bayesian/t-test",
        "description": "Bayesian t-test with Bayes factor",
    },
    "bayesian_regression": {
        "module": "bayesian",
        "endpoint": "/bayesian/regression",
        "description": "Bayesian linear regression with credible intervals",
    },
    "survival": {
        "module": "medical",
        "endpoint": "/medical/survival",
        "description": "Survival / time-to-event analysis (Kaplan-Meier, Cox)",
    },
    "nonparametric": {
        "module": "nonparametric",
        "endpoint": "/nonparametric/mann-whitney",
        "description": "Non-parametric group comparison (Mann-Whitney / Kruskal-Wallis)",
    },
    "logistic_regression": {
        "module": "advanced_stats",
        "endpoint": "/advanced/logistic",
        "description": "Logistic regression for binary/categorical outcomes",
    },
    "descriptive": {
        "module": "stats_studio",
        "endpoint": "/stats/descriptive",
        "description": "Descriptive statistics summary",
    },
}


# ---------------------------------------------------------------------------
# Keyword pattern rules  (priority order — first match wins base score)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[dict] = [
    # Mediation / moderation (check before generic regression)
    {
        "patterns": [r"\bmediat", r"\bindirect\s+effect", r"\bsobel"],
        "analysis": "mediation",
        "base_confidence": 0.85,
    },
    {
        "patterns": [r"\bmoderat", r"\binteraction\b(?!.*anova)"],
        "analysis": "moderation",
        "base_confidence": 0.80,
    },
    # HLM / multilevel
    {
        "patterns": [r"\bnested\b", r"\bmultilevel\b", r"\bhierarchical\s+linear",
                     r"\bmixed.?effect", r"\brandom\s+intercept", r"\brandom\s+slope",
                     r"\bhlm\b", r"\bicc\b"],
        "analysis": "hlm",
        "base_confidence": 0.85,
    },
    # SEM
    {
        "patterns": [r"\bstructural\s+equation", r"\bsem\b", r"\bpath\s+(model|analysis)",
                     r"\blatent\s+variable"],
        "analysis": "sem",
        "base_confidence": 0.85,
    },
    # Survival
    {
        "patterns": [r"\bsurvival\b", r"\btime.to.event", r"\bkaplan", r"\bcox\b",
                     r"\bhazard\b", r"\bcensored\b"],
        "analysis": "survival",
        "base_confidence": 0.85,
    },
    # Psychometrics
    {
        "patterns": [r"\bfactor\s+structure", r"\bscale\s+(development|validation)",
                     r"\breliabilit", r"\bcronbach", r"\bomega\b",
                     r"\bexploratory\s+factor", r"\befa\b", r"\bcfa\b"],
        "analysis": "factor_analysis",
        "base_confidence": 0.82,
    },
    {
        "patterns": [r"\breliabilit", r"\bcronbach", r"\binternal\s+consisten",
                     r"\bomega\b", r"\bsplit.half"],
        "analysis": "reliability",
        "base_confidence": 0.82,
    },
    # Bayesian
    {
        "patterns": [r"\bbayesian\b", r"\bbayes\s+factor", r"\bprior\b.*\bposterior\b",
                     r"\bposterior\b", r"\bcredible\s+interval"],
        "analysis": "bayesian_t_test",  # refined later based on context
        "base_confidence": 0.78,
    },
    # Chi-square
    {
        "patterns": [r"\bchi.?square", r"\bcontingency", r"\bcrosstab",
                     r"\bindependence\s+test", r"\bcategorical.*categori"],
        "analysis": "chi_square",
        "base_confidence": 0.85,
    },
    # Logistic regression
    {
        "patterns": [r"\blogistic\b", r"\bbinary\s+outcome", r"\bodds\s+ratio",
                     r"\blog.?reg"],
        "analysis": "logistic_regression",
        "base_confidence": 0.85,
    },
    # ANOVA
    {
        "patterns": [r"\banova\b", r"\banalysis\s+of\s+variance"],
        "analysis": "anova",
        "base_confidence": 0.88,
    },
    {
        "patterns": [r"\bmixed.?anova", r"\bsplit.?plot", r"\bwithin.*between",
                     r"\bbetween.*within", r"\brepeated.*group"],
        "analysis": "mixed_anova",
        "base_confidence": 0.85,
    },
    # Paired t-test
    {
        "patterns": [r"\bpaired\b", r"\bbefore\s+(and\s+)?after",
                     r"\bpre.?post\b", r"\brepeated\s+measure.*two",
                     r"\bwithin.?subject.*compar"],
        "analysis": "paired_t_test",
        "base_confidence": 0.82,
    },
    # T-test (generic group comparison)
    {
        "patterns": [r"\bt.?test\b", r"\bcompar.*two\s+(group|mean|sample)",
                     r"\bdifference\s+between\s+(two\s+)?(group|mean)",
                     r"\bgroup\s+difference"],
        "analysis": "t_test",
        "base_confidence": 0.82,
    },
    # Broader comparison phrases → t-test / anova decided later
    {
        "patterns": [r"\bcompar", r"\bdifference\s+between", r"\bbetween\s+group"],
        "analysis": "t_test",
        "base_confidence": 0.65,
    },
    # Regression
    {
        "patterns": [r"\bregression\b", r"\bpredict", r"\bmodel\b.*\boutcome",
                     r"\bexplain\s+variance", r"\bcoefficient"],
        "analysis": "regression",
        "base_confidence": 0.80,
    },
    # Correlation
    {
        "patterns": [r"\bcorrelat", r"\brelationship\b", r"\bassociat",
                     r"\bpearson\b", r"\bspearman\b"],
        "analysis": "correlation",
        "base_confidence": 0.78,
    },
    # Non-parametric
    {
        "patterns": [r"\bnon.?parametric", r"\bmann.?whitney", r"\bkruskal",
                     r"\bwilcoxon\b", r"\brank"],
        "analysis": "nonparametric",
        "base_confidence": 0.80,
    },
    # Descriptive (fallback)
    {
        "patterns": [r"\bdescriptive", r"\bsummar", r"\bmean.*\bsd\b",
                     r"\bfrequenc"],
        "analysis": "descriptive",
        "base_confidence": 0.70,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_query(query: str) -> tuple[str, float, list[AnalysisAlternative]]:
    """Return (analysis_type, confidence, alternatives) from keyword rules."""
    q = query.lower()
    matches: list[tuple[str, float]] = []

    for rule in _KEYWORD_RULES:
        for pat in rule["patterns"]:
            if re.search(pat, q):
                matches.append((rule["analysis"], rule["base_confidence"]))
                break  # one match per rule is enough

    if not matches:
        return "descriptive", 0.30, [
            AnalysisAlternative(
                analysis_type="correlation",
                reason="Could not determine intent — correlation is a common exploratory step",
            )
        ]

    # Deduplicate, keep highest confidence per analysis
    best: dict[str, float] = {}
    for atype, conf in matches:
        if atype not in best or conf > best[atype]:
            best[atype] = conf

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    top_analysis, top_conf = ranked[0]

    alternatives = [
        AnalysisAlternative(
            analysis_type=atype,
            reason=_ANALYSIS_REGISTRY.get(atype, {}).get("description", ""),
        )
        for atype, _ in ranked[1:4]
    ]

    return top_analysis, top_conf, alternatives


def _extract_variables(query: str, available: list[str] | None) -> dict:
    """Best-effort extraction of variable roles from the query text."""
    params: dict = {
        "dependent": None,
        "independent": [],
        "grouping": None,
        "covariates": [],
        "paired": False,
        "alternative": "two-sided",
    }

    q = query.lower()

    # Detect paired
    if re.search(r"\bpaired\b|\bbefore.*after\b|\bpre.?post\b|\brepeated\b|\bwithin.subject", q):
        params["paired"] = True

    # Detect directionality
    if re.search(r"\bgreater\b|\bhigher\b|\bmore\b|\bincrease", q):
        params["alternative"] = "greater"
    elif re.search(r"\bless\b|\blower\b|\bfewer\b|\bdecrease", q):
        params["alternative"] = "less"

    if not available:
        return params

    # Try to match variable names mentioned in the query
    available_lower = {v.lower(): v for v in available}
    mentioned = []
    for vlow, vorig in available_lower.items():
        # Escape the variable name for safe regex, allow partial match
        if re.search(re.escape(vlow), q):
            mentioned.append(vorig)

    # Heuristic assignment: first mentioned → DV, rest → IV
    if len(mentioned) >= 1:
        params["dependent"] = mentioned[0]
    if len(mentioned) >= 2:
        params["independent"] = mentioned[1:] if len(mentioned) > 2 else [mentioned[1]]

    # Look for explicit "on" / "by" / "predict" structure
    on_match = re.search(r"(\w+)\s+on\s+(\w+)", q)
    if on_match:
        dv_cand = on_match.group(1)
        iv_cand = on_match.group(2)
        if dv_cand in available_lower:
            params["dependent"] = available_lower[dv_cand]
        if iv_cand in available_lower:
            params["independent"] = [available_lower[iv_cand]]

    by_match = re.search(r"by\s+(\w+)", q)
    if by_match:
        g = by_match.group(1)
        if g in available_lower:
            params["grouping"] = available_lower[g]

    return params


def _refine_bayesian(query: str, base: str) -> str:
    """If Bayesian was detected, decide sub-type."""
    q = query.lower()
    if re.search(r"\bregress|\bpredict|\bmodel", q):
        return "bayesian_regression"
    return "bayesian_t_test"


def _refine_comparison(query: str, base: str, available: list[str] | None,
                       data_summary: dict | None) -> str:
    """Decide between t-test and ANOVA for generic comparison queries."""
    q = query.lower()

    # Explicit mentions
    if re.search(r"\bthree\b|\bfour\b|\bfive\b|\bmultiple\s+group|\b3\+?\s+group|\b[3-9]\s+group", q):
        return "anova"
    if re.search(r"\btwo\s+group|\b2\s+group", q):
        return "t_test"

    # Check data summary for grouping variable n_unique
    if data_summary:
        for col, info in data_summary.items():
            if isinstance(info, dict) and info.get("type") in ("categorical", "nominal"):
                n_unique = info.get("n_unique", 0)
                if n_unique > 2:
                    return "anova"

    return base


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/parse-query", response_model=ParseQueryResponse)
def parse_query(req: NLQueryRequest):
    """Parse a natural-language research question into a routable analysis
    specification."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    analysis_type, confidence, alternatives = _match_query(req.query)

    # Refinements
    if analysis_type.startswith("bayesian"):
        analysis_type = _refine_bayesian(req.query, analysis_type)

    if analysis_type in ("t_test", "anova"):
        analysis_type = _refine_comparison(
            req.query, analysis_type, req.available_variables, req.data_summary
        )

    # Boost confidence when variable names are provided and matched
    params = _extract_variables(req.query, req.available_variables)
    if params["dependent"]:
        confidence = min(confidence + 0.05, 1.0)
    if params["independent"]:
        confidence = min(confidence + 0.05, 1.0)

    registry_entry = _ANALYSIS_REGISTRY.get(analysis_type, _ANALYSIS_REGISTRY["descriptive"])

    return ParseQueryResponse(
        analysis_type=analysis_type,
        confidence=round(confidence, 2),
        parameters=params,
        explanation=registry_entry["description"],
        alternatives=alternatives,
        module=registry_entry["module"],
        endpoint=registry_entry["endpoint"],
    )


# ---------------------------------------------------------------------------
# POST /suggest-analysis
# ---------------------------------------------------------------------------

def _assess_power(n: int, analyses: list[str]) -> str:
    """Quick heuristic power note."""
    notes: list[str] = []
    if n < 20:
        notes.append(
            f"With N={n}, most analyses will be severely underpowered. "
            "Consider collecting more data or using Bayesian approaches."
        )
    elif n < 50:
        notes.append(
            f"N={n} may be adequate for large effects but underpowered for "
            "small-to-medium effects in most tests."
        )
    elif n < 200:
        notes.append(
            f"N={n} provides reasonable power for medium effects in common tests."
        )
    else:
        notes.append(
            f"N={n} provides good statistical power for most planned analyses."
        )

    if "hlm" in analyses and n < 50:
        notes.append("HLM typically requires 30+ clusters with 5+ observations each.")
    if "sem" in analyses and n < 200:
        notes.append("SEM generally needs N >= 200 for stable parameter estimates.")
    if "factor_analysis" in analyses and n < 100:
        notes.append("Factor analysis is most reliable with N >= 5-10 per item (min ~100).")

    return " ".join(notes)


@router.post("/suggest-analysis", response_model=SuggestAnalysisResponse)
def suggest_analysis(req: DataProfileRequest):
    """Given a data profile, suggest appropriate analyses and flag data issues."""
    if not req.columns:
        raise HTTPException(status_code=400, detail="columns list must not be empty")
    if req.n_rows < 1:
        raise HTTPException(status_code=400, detail="n_rows must be at least 1")

    continuous_cols = [c for c in req.columns if c.type == "continuous"]
    categorical_cols = [c for c in req.columns if c.type in ("categorical", "ordinal")]

    suggestions: list[AnalysisSuggestion] = []
    data_issues: list[DataIssue] = []
    priority_counter = 1

    # ---- Data issues scan ----
    for col in req.columns:
        if col.n_missing is not None and col.n_missing > 0:
            pct = (col.n_missing / req.n_rows) * 100
            if pct > 50:
                data_issues.append(DataIssue(
                    column=col.name,
                    issue=f"{pct:.0f}% missing values",
                    recommendation="Consider dropping this variable or using multiple imputation (MICE).",
                ))
            elif pct > 5:
                data_issues.append(DataIssue(
                    column=col.name,
                    issue=f"{pct:.1f}% missing values",
                    recommendation="Assess missingness mechanism (MCAR/MAR/MNAR). Consider multiple imputation.",
                ))

        if col.type == "continuous" and col.n_unique is not None and col.n_unique <= 5:
            data_issues.append(DataIssue(
                column=col.name,
                issue=f"Marked as continuous but only {col.n_unique} unique values",
                recommendation="Verify this variable is truly continuous; it may be ordinal or categorical.",
            ))

        if col.type == "categorical" and col.n_unique is not None and col.n_unique > 20:
            data_issues.append(DataIssue(
                column=col.name,
                issue=f"{col.n_unique} unique categories — very high cardinality",
                recommendation="Consider recoding or binning categories before analysis.",
            ))

    # ---- Always suggest descriptives ----
    suggestions.append(AnalysisSuggestion(
        analysis="descriptive",
        reason="Start with descriptive statistics and data visualization",
        module="stats_studio",
        endpoint="/stats/descriptive",
        priority=priority_counter,
    ))
    priority_counter += 1

    # ---- Correlation if 2+ continuous ----
    if len(continuous_cols) >= 2:
        suggestions.append(AnalysisSuggestion(
            analysis="correlation",
            reason=f"{len(continuous_cols)} continuous variables detected — examine bivariate relationships",
            module="stats_studio",
            endpoint="/stats/correlation",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- Group comparison if categorical + continuous ----
    for cat in categorical_cols:
        for cont in continuous_cols:
            n_groups = cat.n_unique or 2
            if n_groups == 2:
                suggestions.append(AnalysisSuggestion(
                    analysis="t_test",
                    reason=f"Compare {cont.name} across 2 groups of {cat.name}",
                    module="stats_studio",
                    endpoint="/stats/t-test",
                    priority=priority_counter,
                ))
            elif n_groups <= 10:
                suggestions.append(AnalysisSuggestion(
                    analysis="anova",
                    reason=f"Compare {cont.name} across {n_groups} groups of {cat.name}",
                    module="stats_studio",
                    endpoint="/stats/anova",
                    priority=priority_counter,
                ))
            priority_counter += 1
            if priority_counter > 8:  # limit output
                break
        if priority_counter > 8:
            break

    # ---- Chi-square if 2+ categorical ----
    if len(categorical_cols) >= 2:
        suggestions.append(AnalysisSuggestion(
            analysis="chi_square",
            reason=f"Test independence between categorical variables ({categorical_cols[0].name}, {categorical_cols[1].name})",
            module="stats_studio",
            endpoint="/stats/chi-square",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- Regression if multiple continuous ----
    if len(continuous_cols) >= 2:
        suggestions.append(AnalysisSuggestion(
            analysis="regression",
            reason="Multiple continuous variables available for regression modeling",
            module="stats_studio",
            endpoint="/stats/regression",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- Logistic regression if binary outcome + predictors ----
    binary_cats = [c for c in categorical_cols if c.n_unique == 2]
    if binary_cats and (len(continuous_cols) >= 1 or len(categorical_cols) >= 2):
        suggestions.append(AnalysisSuggestion(
            analysis="logistic_regression",
            reason=f"Binary outcome ({binary_cats[0].name}) with available predictors",
            module="advanced_stats",
            endpoint="/advanced/logistic",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- Factor analysis if many continuous (scale-like) ----
    if len(continuous_cols) >= 5:
        suggestions.append(AnalysisSuggestion(
            analysis="factor_analysis",
            reason=f"{len(continuous_cols)} continuous variables — consider exploratory factor analysis",
            module="psychometrics",
            endpoint="/psychometrics/efa",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- Research goal refinement ----
    if req.research_goal:
        goal = req.research_goal.lower()
        goal_type, _, _ = _match_query(goal)
        reg = _ANALYSIS_REGISTRY.get(goal_type)
        if reg and not any(s.analysis == goal_type for s in suggestions):
            suggestions.insert(1, AnalysisSuggestion(
                analysis=goal_type,
                reason=f"Matches stated research goal: {req.research_goal}",
                module=reg["module"],
                endpoint=reg["endpoint"],
                priority=1,
            ))

    analysis_types = [s.analysis for s in suggestions]
    power_note = _assess_power(req.n_rows, analysis_types)

    return SuggestAnalysisResponse(
        suggestions=suggestions,
        data_issues=data_issues,
        power_note=power_note,
    )
