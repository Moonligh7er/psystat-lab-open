"""Contradiction Detector — Finds inconsistencies across multiple analyses.

Identifies direction conflicts, significance disagreements, suppression
effects, and Simpson's paradox patterns when the same variables appear
in different analyses.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any
from itertools import combinations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

router = APIRouter()


# ── Pydantic Models ───────────────────────────────────────────────────────

class AnalysisEntry(BaseModel):
    module: str  # e.g. "Stats Studio", "Regression"
    test_type: str  # e.g. "pearson", "regression", "ttest"
    variables: List[str]  # variable names involved
    results: Dict[str, Any]  # must include: direction (+1/-1), effect_size, p_value, significant


class ContradictionRequest(BaseModel):
    analyses: List[AnalysisEntry]


class ContradictionItem(BaseModel):
    analysis_pair: List[str]  # [module1_test, module2_test]
    shared_variables: List[str]
    type: str  # "direction", "significance", "suppression", "simpson", "magnitude"
    severity: str  # "critical", "warning", "note"
    explanation: str
    recommendation: str


class SeverityBreakdown(BaseModel):
    critical: int = 0
    warning: int = 0
    note: int = 0


class ContradictionSummary(BaseModel):
    total_checked: int
    contradictions_found: int
    severity_breakdown: SeverityBreakdown


class ContradictionResponse(BaseModel):
    contradictions: List[ContradictionItem]
    summary: ContradictionSummary


# ── Helpers ───────────────────────────────────────────────────────────────

# Effect sizes for these tests are never negative (eta², R², Cramér's V, chi-square, F),
# so their sign says nothing about direction.
UNSIGNED_TESTS = ("anova", "chi", "regression_r2", "r2", "r_squared", "kruskal", "f_test", "manova", "cramer")


def _get_direction(results: Dict[str, Any], test_type: str = "") -> Optional[int]:
    """Direction as +1 / −1 from an explicit 'direction', or from a signed effect size. None if unknown."""
    d = results.get("direction")
    if d is not None:
        if isinstance(d, str):
            s = d.strip().lower()
            if s in ("positive", "pos", "+", "1"):
                return 1
            if s in ("negative", "neg", "-", "-1"):
                return -1
            return None
        return None if float(d) == 0 else (1 if float(d) > 0 else -1)
    if any(u in test_type.lower() for u in UNSIGNED_TESTS):
        return None
    es = results.get("effect_size")
    if es is not None and float(es) != 0:
        return 1 if float(es) > 0 else -1
    return None


def _get_es(results: Dict[str, Any]) -> Optional[float]:
    es = results.get("effect_size")
    if es is not None:
        v = float(es)
        if not (math.isnan(v) or math.isinf(v)):
            return v
    return None


def _get_sig(results: Dict[str, Any]) -> Optional[bool]:
    s = results.get("significant")
    if s is not None:
        return bool(s)
    p = results.get("p_value")
    if p is not None:
        return float(p) < 0.05
    return None


def _label(a: AnalysisEntry) -> str:
    return f"{a.module} ({a.test_type})"


def _shared_vars(a: AnalysisEntry, b: AnalysisEntry) -> List[str]:
    return sorted(set(a.variables) & set(b.variables))


# ── Core Detection ────────────────────────────────────────────────────────

def _detect_contradictions(analyses: List[AnalysisEntry]) -> List[ContradictionItem]:
    contradictions: List[ContradictionItem] = []

    for i, j in combinations(range(len(analyses)), 2):
        a, b = analyses[i], analyses[j]
        shared = _shared_vars(a, b)
        if not shared:
            continue

        label_a = _label(a)
        label_b = _label(b)
        dir_a = _get_direction(a.results, a.test_type)
        dir_b = _get_direction(b.results, b.test_type)
        es_a = _get_es(a.results)
        es_b = _get_es(b.results)
        sig_a = _get_sig(a.results)
        sig_b = _get_sig(b.results)

        # 1. Direction contradiction
        if dir_a is not None and dir_b is not None and dir_a != dir_b:
            # Check for suppression pattern: bivariate vs multivariate
            is_suppression = False
            a_multivariate = len(a.variables) > 2
            b_multivariate = len(b.variables) > 2

            if a_multivariate != b_multivariate:
                is_suppression = True
                contradictions.append(ContradictionItem(
                    analysis_pair=[label_a, label_b],
                    shared_variables=shared,
                    type="suppression",
                    severity="critical",
                    explanation=(
                        f"The relationship between {', '.join(shared)} reverses direction when additional "
                        f"variables are controlled for. This may be a suppression or confounding effect: the bivariate analysis shows "
                        f"{'positive' if (dir_a if not a_multivariate else dir_b) > 0 else 'negative'} "
                        f"but the multivariate analysis shows the opposite."
                    ),
                    recommendation=(
                        "Investigate which covariates cause the reversal. This often indicates confounding. "
                        "Report both results and discuss the theoretical implications of the suppression."
                    ),
                ))

            if not is_suppression:
                # Check for Simpson's paradox: if one analysis is grouped/stratified
                # Only when an analysis is explicitly marked as stratified/grouped
                a_has_groups = bool(a.results.get("grouped", False))
                b_has_groups = bool(b.results.get("grouped", False))

                if a_has_groups != b_has_groups:
                    contradictions.append(ContradictionItem(
                        analysis_pair=[label_a, label_b],
                        shared_variables=shared,
                        type="simpson",
                        severity="critical",
                        explanation=(
                            f"Possible Simpson's paradox: the direction of the relationship between "
                            f"{', '.join(shared)} reverses when data is aggregated vs. stratified. "
                            f"The aggregated analysis shows {'positive' if (dir_a if not a_has_groups else dir_b) > 0 else 'negative'} "
                            f"while the stratified analysis shows the opposite."
                        ),
                        recommendation=(
                            "This is a well-known statistical paradox. The stratified analysis is usually more "
                            "appropriate. Identify the lurking variable that causes the reversal and report both results."
                        ),
                    ))
                else:
                    contradictions.append(ContradictionItem(
                        analysis_pair=[label_a, label_b],
                        shared_variables=shared,
                        type="direction",
                        severity="critical",
                        explanation=(
                            f"Direction conflict: '{label_a}' shows a "
                            f"{'positive' if dir_a > 0 else 'negative'} effect, but '{label_b}' shows a "
                            f"{'positive' if dir_b > 0 else 'negative'} effect for the same variables "
                            f"({', '.join(shared)})."
                        ),
                        recommendation=(
                            "Investigate why these analyses disagree. Check for differences in variable coding, "
                            "sample subsets, outlier handling, or model specification."
                        ),
                    ))

        # 2. Significance contradiction
        if sig_a is not None and sig_b is not None and sig_a != sig_b:
            # Only flag if directions agree (direction conflict already flagged above)
            if dir_a is None or dir_b is None or dir_a == dir_b:
                contradictions.append(ContradictionItem(
                    analysis_pair=[label_a, label_b],
                    shared_variables=shared,
                    type="significance",
                    severity="warning",
                    explanation=(
                        f"Significance disagreement: '{label_a}' finds "
                        f"{'significant' if sig_a else 'non-significant'} results, but '{label_b}' finds "
                        f"{'significant' if sig_b else 'non-significant'} results for {', '.join(shared)}."
                    ),
                    recommendation=(
                        "Different tests have different assumptions and power. The disagreement may reflect "
                        "assumption violations, different sample sizes, or the effect being near the significance boundary. "
                        "Report both results and focus on effect sizes rather than p-values."
                    ),
                ))

        # 3. Effect size discrepancy (same direction, but very different magnitudes)
        if (es_a is not None and es_b is not None
                and dir_a is not None and dir_b is not None and dir_a == dir_b):
            abs_a, abs_b = abs(es_a), abs(es_b)
            if max(abs_a, abs_b) > 0:
                ratio = min(abs_a, abs_b) / max(abs_a, abs_b)
                if ratio < 0.33 and max(abs_a, abs_b) > 0.1:
                    contradictions.append(ContradictionItem(
                        analysis_pair=[label_a, label_b],
                        shared_variables=shared,
                        type="magnitude",
                        severity="note",
                        explanation=(
                            f"Large magnitude discrepancy: effect sizes for {', '.join(shared)} differ substantially "
                            f"(|ES| = {abs_a:.3f} vs. {abs_b:.3f}). While both are in the same direction, "
                            f"the practical interpretation could differ."
                        ),
                        recommendation=(
                            "Different methods can yield different effect size metrics that are not directly comparable. "
                            "Ensure you are comparing like with like. If both are on the same scale, investigate "
                            "what drives the discrepancy."
                        ),
                    ))

    return contradictions


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=ContradictionResponse)
async def detect_contradictions(req: ContradictionRequest):
    try:
        if len(req.analyses) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 analyses to check for contradictions.")

        contradictions = _detect_contradictions(req.analyses)

        # Count pairs checked (only pairs sharing variables)
        total_pairs = 0
        for i, j in combinations(range(len(req.analyses)), 2):
            if _shared_vars(req.analyses[i], req.analyses[j]):
                total_pairs += 1

        breakdown = SeverityBreakdown(
            critical=sum(1 for c in contradictions if c.severity == "critical"),
            warning=sum(1 for c in contradictions if c.severity == "warning"),
            note=sum(1 for c in contradictions if c.severity == "note"),
        )

        return ContradictionResponse(
            contradictions=contradictions,
            summary=ContradictionSummary(
                total_checked=total_pairs,
                contradictions_found=len(contradictions),
                severity_breakdown=breakdown,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
