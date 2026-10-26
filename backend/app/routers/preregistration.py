"""
Pre-Registration Document Generator
Generates structured pre-registration documents (OSF, AsPredicted, Markdown)
from study design parameters.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

router = APIRouter(tags=["preregistration"])

# ── Models ──────────────────────────────────────────────────────────────────

class Hypothesis(BaseModel):
    id: str
    text: str
    direction: str  # "two-tailed", "greater", "less"
    variables: Dict[str, str]  # {"iv": "treatment", "dv": "anxiety_score"}

class AnalysisPlan(BaseModel):
    hypothesis_id: str
    test_type: str
    alpha: float = 0.05
    correction: Optional[str] = None
    covariates: List[str] = []
    assumptions_to_check: List[str] = []
    alternative_if_violated: str = ""

class ExclusionCriterion(BaseModel):
    description: str
    decision_rule: str

class PowerAnalysisResult(BaseModel):
    test_type: str
    effect_size: float
    effect_size_type: str
    alpha: float
    power: float
    required_n: int
    rationale: str

class PreRegistrationRequest(BaseModel):
    title: str
    authors: List[str]
    date: Optional[str] = None
    format: str = "osf"

    research_questions: List[str]
    hypotheses: List[Hypothesis]
    study_type: str
    design_description: str

    target_n: int
    power_analyses: List[PowerAnalysisResult] = []
    power_analysis_text: str = ""  # free-text power analysis / sample-size justification
    population: str
    recruitment: str
    exclusion_criteria: List[ExclusionCriterion]
    stopping_rule: str

    measures: List[Dict[str, str]]

    analysis_plans: List[AnalysisPlan]
    exploratory_analyses: List[str] = []

    data_exclusion_handling: str
    missing_data_plan: str
    outlier_plan: str

class SectionOut(BaseModel):
    title: str
    content: str

class ChecklistItem(BaseModel):
    item: str
    complete: bool
    note: str

class PreRegistrationResponse(BaseModel):
    document: str
    format: str
    word_count: int
    sections: List[SectionOut]
    created_at: str
    checklist: List[ChecklistItem]

# ── Helpers ─────────────────────────────────────────────────────────────────

def _direction_label(d: str) -> str:
    return {"two-tailed": "two-tailed", "greater": "one-tailed (greater)",
            "less": "one-tailed (less)"}.get(d, d)

def _build_checklist(req: PreRegistrationRequest) -> List[ChecklistItem]:
    items: List[ChecklistItem] = []

    def chk(item: str, ok: bool, note: str = ""):
        items.append(ChecklistItem(item=item, complete=ok, note=note))

    chk("Title provided", bool(req.title), "" if req.title else "Missing study title")
    chk("At least one author", len(req.authors) > 0,
        "" if req.authors else "No authors listed")
    chk("Research questions specified", len(req.research_questions) > 0,
        "" if req.research_questions else "No research questions")
    chk("Hypotheses specified", len(req.hypotheses) > 0,
        "" if req.hypotheses else "No hypotheses")
    chk("Design description provided", bool(req.design_description),
        "" if req.design_description else "Missing design description")
    chk("Target N specified", req.target_n > 0,
        "" if req.target_n > 0 else "Target N is 0 or missing")
    chk("Stopping rule defined", bool(req.stopping_rule),
        "" if req.stopping_rule else "No stopping rule specified")
    chk("Population described", bool(req.population),
        "" if req.population else "No population description")
    chk("Recruitment plan", bool(req.recruitment),
        "" if req.recruitment else "No recruitment plan")
    chk("At least one measure", len(req.measures) > 0,
        "" if req.measures else "No measures listed")
    chk("Analysis plans provided", len(req.analysis_plans) > 0,
        "" if req.analysis_plans else "No analysis plans")
    chk("Missing data plan", bool(req.missing_data_plan),
        "" if req.missing_data_plan else "No missing data plan")
    chk("Outlier plan", bool(req.outlier_plan),
        "" if req.outlier_plan else "No outlier handling plan")
    has_power = len(req.power_analyses) > 0 or bool(req.power_analysis_text.strip())
    chk("Power analysis provided", has_power, "" if has_power else "No power analysis")

    # Check every hypothesis has a matching analysis plan
    hyp_ids = {h.id for h in req.hypotheses}
    covered = {a.hypothesis_id for a in req.analysis_plans}
    uncovered = hyp_ids - covered
    chk("All hypotheses have analysis plans", len(uncovered) == 0,
        "" if not uncovered else f"Missing analysis plan for: {', '.join(sorted(uncovered))}")

    # Check every hypothesis has variables defined
    empty_vars = [h.id for h in req.hypotheses if not h.variables]
    chk("All hypotheses have variables", len(empty_vars) == 0,
        "" if not empty_vars else f"No variables for: {', '.join(empty_vars)}")

    # Exclusion criteria present
    chk("Exclusion criteria defined", len(req.exclusion_criteria) > 0,
        "" if req.exclusion_criteria else "No exclusion criteria specified")

    return items

# ── OSF Format ──────────────────────────────────────────────────────────────

def _generate_osf(req: PreRegistrationRequest) -> List[SectionOut]:
    sections: List[SectionOut] = []
    date_str = req.date or datetime.utcnow().strftime("%Y-%m-%d")

    # 1 Study Information
    lines = [f"Title: {req.title}",
             f"Authors: {', '.join(req.authors)}",
             f"Date: {date_str}", "",
             "Research Questions:"]
    for i, rq in enumerate(req.research_questions, 1):
        lines.append(f"  {i}. {rq}")
    lines += ["", "Hypotheses:"]
    for h in req.hypotheses:
        lines.append(f"  {h.id}: {h.text} [{_direction_label(h.direction)}]")
        if h.variables:
            for role, name in h.variables.items():
                lines.append(f"    - {role.upper()}: {name}")
    sections.append(SectionOut(title="Study Information", content="\n".join(lines)))

    # 2 Design Plan
    lines = [f"Study type: {req.study_type}", "",
             "Design:", req.design_description]
    sections.append(SectionOut(title="Design Plan", content="\n".join(lines)))

    # 3 Sampling Plan
    lines = [f"Target sample size: N = {req.target_n}",
             f"Population: {req.population}",
             f"Recruitment: {req.recruitment}",
             f"Stopping rule: {req.stopping_rule}", "",
             "Power Analyses:"]
    for pa in req.power_analyses:
        lines.append(f"  - Test: {pa.test_type}")
        lines.append(f"    Effect size: {pa.effect_size_type} = {pa.effect_size}")
        lines.append(f"    Alpha: {pa.alpha}, Power: {pa.power}")
        lines.append(f"    Required N: {pa.required_n}")
        lines.append(f"    Rationale: {pa.rationale}")
        lines.append("")
    if req.power_analysis_text.strip():
        lines.append(f"  {req.power_analysis_text.strip()}")
    sections.append(SectionOut(title="Sampling Plan", content="\n".join(lines)))

    # 4 Variables
    ivs, dvs, covs, constructs = set(), set(), set(), set()
    for h in req.hypotheses:
        for role, name in h.variables.items():
            if role.lower() == "iv":
                ivs.add(name)
            elif role.lower() == "dv":
                dvs.add(name)
    for ap in req.analysis_plans:
        for c in ap.covariates:
            covs.add(c)
    for m in req.measures:
        if "construct" in m:
            constructs.add(m["construct"])
    lines = [f"Independent variables: {', '.join(sorted(ivs)) or 'N/A'}",
             f"Dependent variables: {', '.join(sorted(dvs)) or 'N/A'}",
             f"Covariates: {', '.join(sorted(covs)) or 'None'}",
             f"Measured constructs: {', '.join(sorted(constructs)) or 'N/A'}", "",
             "Measures:"]
    for m in req.measures:
        parts = [f"{m.get('name', 'Unnamed')}"]
        if "construct" in m:
            parts.append(f"construct={m['construct']}")
        if "reliability" in m:
            parts.append(f"reliability: {m['reliability']}")
        lines.append(f"  - {', '.join(parts)}")
    sections.append(SectionOut(title="Variables", content="\n".join(lines)))

    # 5 Analysis Plan
    lines = []
    for ap in req.analysis_plans:
        lines.append(f"Hypothesis {ap.hypothesis_id}:")
        lines.append(f"  Statistical test: {ap.test_type}")
        lines.append(f"  Alpha level: {ap.alpha}")
        if ap.correction:
            lines.append(f"  Multiple-comparison correction: {ap.correction}")
        if ap.covariates:
            lines.append(f"  Covariates: {', '.join(ap.covariates)}")
        if ap.assumptions_to_check:
            lines.append(f"  Assumptions to verify: {', '.join(ap.assumptions_to_check)}")
        if ap.alternative_if_violated:
            lines.append(f"  If assumptions violated: {ap.alternative_if_violated}")
        lines.append("")
    sections.append(SectionOut(title="Analysis Plan", content="\n".join(lines)))

    # 6 Exclusion Criteria
    lines = []
    for ec in req.exclusion_criteria:
        lines.append(f"  - {ec.description}")
        lines.append(f"    Rule: {ec.decision_rule}")
    lines.append("")
    lines.append(f"Data exclusion handling: {req.data_exclusion_handling}")
    sections.append(SectionOut(title="Exclusion Criteria", content="\n".join(lines)))

    # 7 Data Handling
    lines = [f"Missing data: {req.missing_data_plan}",
             f"Outliers: {req.outlier_plan}"]
    sections.append(SectionOut(title="Data Handling", content="\n".join(lines)))

    # 8 Exploratory Analyses
    if req.exploratory_analyses:
        lines = ["The following analyses are exploratory and not confirmatory:"]
        for i, ea in enumerate(req.exploratory_analyses, 1):
            lines.append(f"  {i}. {ea}")
    else:
        lines = ["No exploratory analyses planned."]
    sections.append(SectionOut(title="Exploratory Analyses", content="\n".join(lines)))

    # 9 Timeline
    lines = [f"Pre-registration date: {date_str}",
             f"Stopping rule: {req.stopping_rule}"]
    sections.append(SectionOut(title="Timeline", content="\n".join(lines)))

    return sections

# ── AsPredicted Format ──────────────────────────────────────────────────────

def _generate_aspredicted(req: PreRegistrationRequest) -> List[SectionOut]:
    sections: List[SectionOut] = []

    # Q1
    sections.append(SectionOut(
        title="1) Have any data been collected for this study already?",
        content="No, no data have been collected for this study yet."
    ))

    # Q2
    hyp_text = "; ".join(f"{h.id}: {h.text}" for h in req.hypotheses)
    sections.append(SectionOut(
        title="2) What's the main question being asked or hypothesis being tested?",
        content=f"Research questions: {'; '.join(req.research_questions)}\n\n"
                f"Hypotheses: {hyp_text}"
    ))

    # Q3
    dvs = set()
    for h in req.hypotheses:
        for role, name in h.variables.items():
            if role.lower() == "dv":
                dvs.add(name)
    measure_lines = [f"- {m.get('name', 'Unnamed')} ({m.get('construct', '')})"
                     for m in req.measures]
    sections.append(SectionOut(
        title="3) Describe the key dependent variable(s) or measures.",
        content=f"Dependent variables: {', '.join(sorted(dvs)) or 'N/A'}\n\n"
                f"Measures:\n" + "\n".join(measure_lines)
    ))

    # Q4
    sections.append(SectionOut(
        title="4) How many and which conditions will participants be assigned to?",
        content=f"Study type: {req.study_type}\n\n{req.design_description}"
    ))

    # Q5
    analysis_lines = []
    for ap in req.analysis_plans:
        detail = f"{ap.hypothesis_id}: {ap.test_type}, alpha={ap.alpha}"
        if ap.correction:
            detail += f", correction={ap.correction}"
        analysis_lines.append(detail)
    sections.append(SectionOut(
        title="5) Specify exactly which analyses you will conduct to examine "
              "the main question/hypothesis.",
        content="\n".join(analysis_lines)
    ))

    # Q6
    if req.exploratory_analyses:
        content = "\n".join(f"- {ea}" for ea in req.exploratory_analyses)
    else:
        content = "No secondary/exploratory analyses planned."
    sections.append(SectionOut(
        title="6) Any secondary analyses?",
        content=content
    ))

    # Q7
    pa_lines = [f"- {pa.test_type}: required N={pa.required_n} "
                f"({pa.effect_size_type}={pa.effect_size}, power={pa.power})"
                for pa in req.power_analyses]
    sections.append(SectionOut(
        title="7) How many observations will be collected or what will "
              "determine sample size?",
        content=f"Target N: {req.target_n}\n"
                f"Stopping rule: {req.stopping_rule}\n\n"
                f"Power analyses:\n" + "\n".join(pa_lines + ([req.power_analysis_text.strip()] if req.power_analysis_text.strip() else []))
    ))

    # Q8
    extra = []
    if req.exclusion_criteria:
        extra.append("Exclusion criteria:")
        for ec in req.exclusion_criteria:
            extra.append(f"  - {ec.description} ({ec.decision_rule})")
    extra.append(f"\nMissing data plan: {req.missing_data_plan}")
    extra.append(f"Outlier plan: {req.outlier_plan}")
    sections.append(SectionOut(
        title="8) Anything else you would like to pre-register?",
        content="\n".join(extra)
    ))

    return sections

# ── Markdown Format ─────────────────────────────────────────────────────────

def _generate_markdown(req: PreRegistrationRequest) -> List[SectionOut]:
    sections: List[SectionOut] = []
    date_str = req.date or datetime.utcnow().strftime("%Y-%m-%d")

    # Header
    header = (f"# Pre-Registration: {req.title}\n\n"
              f"**Authors:** {', '.join(req.authors)}  \n"
              f"**Date:** {date_str}  \n"
              f"**Study type:** {req.study_type}\n")
    sections.append(SectionOut(title="Header", content=header))

    # Research Questions & Hypotheses
    lines = ["## Research Questions\n"]
    for i, rq in enumerate(req.research_questions, 1):
        lines.append(f"{i}. {rq}")
    lines += ["\n## Hypotheses\n"]
    for h in req.hypotheses:
        lines.append(f"**{h.id}:** {h.text} *[{_direction_label(h.direction)}]*  ")
        if h.variables:
            vars_str = ", ".join(f"{r.upper()}={n}" for r, n in h.variables.items())
            lines.append(f"Variables: {vars_str}\n")
    sections.append(SectionOut(title="Research Questions & Hypotheses",
                               content="\n".join(lines)))

    # Design
    sections.append(SectionOut(title="Design",
                               content=f"## Design\n\n{req.design_description}"))

    # Sample & Power
    lines = ["## Sampling Plan\n",
             f"- **Target N:** {req.target_n}",
             f"- **Population:** {req.population}",
             f"- **Recruitment:** {req.recruitment}",
             f"- **Stopping rule:** {req.stopping_rule}\n",
             "### Power Analyses\n",
             "| Test | Effect Size | Alpha | Power | Required N | Rationale |",
             "|------|------------|-------|-------|-----------|-----------|"]
    for pa in req.power_analyses:
        lines.append(f"| {pa.test_type} | {pa.effect_size_type}={pa.effect_size} "
                     f"| {pa.alpha} | {pa.power} | {pa.required_n} "
                     f"| {pa.rationale} |")
    if req.power_analysis_text.strip():
        lines += ["", req.power_analysis_text.strip()]
    sections.append(SectionOut(title="Sampling Plan", content="\n".join(lines)))

    # Measures
    lines = ["## Measures\n",
             "| Measure | Construct | Reliability |",
             "|---------|-----------|-------------|"]
    for m in req.measures:
        lines.append(f"| {m.get('name', '')} | {m.get('construct', '')} "
                     f"| {m.get('reliability', '')} |")
    sections.append(SectionOut(title="Measures", content="\n".join(lines)))

    # Analysis Plan
    lines = ["## Analysis Plan\n"]
    for ap in req.analysis_plans:
        lines.append(f"### {ap.hypothesis_id}\n")
        lines.append(f"- **Test:** {ap.test_type}")
        lines.append(f"- **Alpha:** {ap.alpha}")
        if ap.correction:
            lines.append(f"- **Correction:** {ap.correction}")
        if ap.covariates:
            lines.append(f"- **Covariates:** {', '.join(ap.covariates)}")
        if ap.assumptions_to_check:
            lines.append(f"- **Assumptions:** {', '.join(ap.assumptions_to_check)}")
        if ap.alternative_if_violated:
            lines.append(f"- **If violated:** {ap.alternative_if_violated}")
        lines.append("")
    sections.append(SectionOut(title="Analysis Plan", content="\n".join(lines)))

    # Exclusion Criteria
    lines = ["## Exclusion Criteria\n"]
    for ec in req.exclusion_criteria:
        lines.append(f"- **{ec.description}** — {ec.decision_rule}")
    lines += ["", f"Data exclusion handling: {req.data_exclusion_handling}"]
    sections.append(SectionOut(title="Exclusion Criteria",
                               content="\n".join(lines)))

    # Data Handling
    lines = ["## Data Handling\n",
             f"**Missing data:** {req.missing_data_plan}\n",
             f"**Outliers:** {req.outlier_plan}"]
    sections.append(SectionOut(title="Data Handling", content="\n".join(lines)))

    # Exploratory
    lines = ["## Exploratory Analyses\n",
             "*These analyses are exploratory and not confirmatory.*\n"]
    if req.exploratory_analyses:
        for i, ea in enumerate(req.exploratory_analyses, 1):
            lines.append(f"{i}. {ea}")
    else:
        lines.append("None planned.")
    sections.append(SectionOut(title="Exploratory Analyses",
                               content="\n".join(lines)))

    return sections

# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/generate", response_model=PreRegistrationResponse)
async def generate_preregistration(req: PreRegistrationRequest):
    """Generate a structured pre-registration document."""
    fmt = req.format.lower()
    if fmt not in ("osf", "aspredicted", "markdown"):
        raise HTTPException(status_code=400,
                            detail=f"Unsupported format '{req.format}'. "
                                   "Use 'osf', 'aspredicted', or 'markdown'.")

    generators = {
        "osf": _generate_osf,
        "aspredicted": _generate_aspredicted,
        "markdown": _generate_markdown,
    }
    sections = generators[fmt](req)

    if fmt == "markdown":
        document = "\n\n".join(s.content for s in sections)
    else:
        parts = []
        for s in sections:
            parts.append(f"{'=' * 60}")
            parts.append(s.title.upper())
            parts.append(f"{'=' * 60}")
            parts.append(s.content)
            parts.append("")
        document = "\n".join(parts)

    checklist = _build_checklist(req)
    word_count = len(document.split())

    return PreRegistrationResponse(
        document=document,
        format=fmt,
        word_count=word_count,
        sections=sections,
        created_at=datetime.utcnow().isoformat(),
        checklist=checklist,
    )


@router.post("/checklist", response_model=List[ChecklistItem])
async def preregistration_checklist(req: PreRegistrationRequest):
    """Validate completeness of a pre-registration and return a checklist."""
    return _build_checklist(req)
