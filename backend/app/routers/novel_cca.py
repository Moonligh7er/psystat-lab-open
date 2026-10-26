"""
Specification Robustness Summary (formerly "Convergent Core Analysis")
=====================================================================
Summarises the output of a multiverse / specification-curve analysis:
how consistent the direction is, how spread out the estimates are, how
often they reach significance, and which analytic decisions move them.

This is not a new method. It packages descriptive summaries already used in
the literature: sign stability and significance rates (Young & Holsteen,
2017; Simonsohn, Simmons & Nelson, 2020) and the influence of each modelling
decision (Young & Holsteen's "model influence"; Steegen et al., 2016).

What it does NOT do: formal inference. Specifications share the same data,
so "95% of specifications agree" is not 95% independent confirmations.
For a test, use the specification-curve permutation test (Simonsohn et al.,
2020), which needs the raw data rather than a list of estimates.
"""

from __future__ import annotations

from typing import List, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

CAVEAT = (
    "Specifications reuse the same data, so these percentages describe the set of "
    "analyses you supplied; they are not independent replications and carry no p-value. "
    "For formal inference use the specification-curve permutation test (Simonsohn et al., 2020)."
)


# ── Request / Response ────────────────────────────────────────────────

class Specification(BaseModel):
    effect_size: float
    p_value: float
    decisions: dict | None = None  # e.g. {"outlier": "IQR", "missing": "listwise"}


class CCARequest(BaseModel):
    specifications: List[Specification]
    alpha: float = 0.05
    core_threshold: float = 0.95  # share of specs that must share a sign
    # Which scale the effect sizes are on. Size words ("small", "medium") are
    # only given for r and d, using Cohen's (1988) conventions.
    effect_metric: Literal["r", "d", "other"] = "other"


class CCAResponse(BaseModel):
    n_specifications: int

    # Direction
    direction_positive_pct: float
    direction_negative_pct: float
    direction_consistent: bool
    dominant_direction: str  # "positive" | "negative" | "mixed"

    # Magnitude
    median_effect: float
    iqr_lower: float
    iqr_upper: float
    trimmed_mean_90: float  # mean after removing 5% of specifications from each tail
    magnitude_range: list  # [min, max]

    # Significance
    pct_significant: float
    pct_significant_after_trim: float  # among specs left after the same 5%/5% trim by effect size

    # Summary
    core_claim: str
    core_direction: str
    core_magnitude_lower: float  # = IQR bounds
    core_magnitude_upper: float
    core_confidence: float  # share of specs that are significant AND in the dominant direction
    caveat: str

    # Decision influence — which decisions move the estimate
    boundary_conditions: list  # [{decision, flip_rate, spread, group_medians, detail}]

    # Agreement at several thresholds
    robustness_thresholds: list  # [{threshold, holds, detail}]


# ── Helpers ───────────────────────────────────────────────────────────

def _trim_index(es: np.ndarray, pct: float = 0.05) -> np.ndarray:
    """Indices of specifications left after dropping the floor(pct*n) smallest and
    largest effect sizes (nothing is dropped when floor(pct*n) is 0)."""
    n = len(es)
    k = int(n * pct)
    order = np.argsort(es, kind="stable")
    return order[k: n - k]


def _size_word(es: float, metric: str) -> str | None:
    if metric == "other":
        return None
    a = abs(es)
    cuts = (0.2, 0.5, 0.8) if metric == "d" else (0.1, 0.3, 0.5)  # Cohen (1988)
    return "negligible" if a < cuts[0] else "small" if a < cuts[1] else "medium" if a < cuts[2] else "large"


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _boundary_analysis(specs: List[Specification], dominant: int) -> list:
    """For each decision, the median estimate at each of its levels."""
    keys: set[str] = set()
    for s in specs:
        if s.decisions:
            keys.update(s.decisions.keys())

    out = []
    for key in sorted(keys):
        groups: dict[str, list[float]] = {}
        for s in specs:
            val = (s.decisions or {}).get(key, "(not specified)")
            groups.setdefault(str(val), []).append(s.effect_size)
        if len(groups) < 2:
            continue

        medians = {v: float(np.median(es)) for v, es in groups.items()}
        flipped = [v for v, m in medians.items() if dominant != 0 and _sign(m) == -dominant]
        spread = max(medians.values()) - min(medians.values())
        detail = f"Median estimate ranges over {spread:.3f} across '{key}' choices. "
        detail += (f"With {', '.join(repr(v) for v in flipped)}, the median estimate has the opposite sign."
                   if flipped else "Every choice gives a median estimate with the dominant sign.")
        out.append({
            "decision": key,
            "flip_rate": round(len(flipped) / len(medians), 4),  # share of levels whose median flips sign
            "spread": round(spread, 6),
            "group_medians": {k: round(v, 6) for k, v in medians.items()},
            "group_sizes": {k: len(v) for k, v in groups.items()},
            "detail": detail,
        })
    out.sort(key=lambda b: b["spread"], reverse=True)
    return out


# ── Endpoint ──────────────────────────────────────────────────────────

@router.post("/analyze", response_model=CCAResponse)
async def convergent_core_analysis(req: CCARequest):
    specs = req.specifications
    n = len(specs)
    if n < 3:
        raise HTTPException(422, "Need at least 3 specifications.")
    if not 0 < req.alpha < 1:
        raise HTTPException(422, "alpha must be between 0 and 1.")
    if not 0.5 < req.core_threshold <= 1:
        raise HTTPException(422, "core_threshold must be above 0.5 and at most 1.")
    es = np.array([s.effect_size for s in specs], dtype=float)
    pv = np.array([s.p_value for s in specs], dtype=float)
    if not (np.all(np.isfinite(es)) and np.all((pv >= 0) & (pv <= 1))):
        raise HTTPException(422, "Effect sizes must be finite and p-values between 0 and 1.")

    pos, neg = float(np.mean(es > 0)), float(np.mean(es < 0))
    dominant = 1 if pos > neg else -1 if neg > pos else 0
    dom_word = {1: "positive", -1: "negative", 0: "mixed"}[dominant]
    dom_share = max(pos, neg)
    direction_consistent = dominant != 0 and dom_share >= req.core_threshold

    median_es = float(np.median(es))
    q1, q3 = float(np.percentile(es, 25)), float(np.percentile(es, 75))
    keep = _trim_index(es)
    trimmed_mean = float(np.mean(es[keep]))

    sig = pv < req.alpha
    pct_sig = float(np.mean(sig))
    pct_sig_trim = float(np.mean(sig[keep]))
    sig_and_dominant = float(np.mean(sig & (np.sign(es) == dominant))) if dominant else 0.0

    robustness = []
    for thresh in (0.90, 0.95, 0.99):
        d_ok = dominant != 0 and dom_share >= thresh
        s_ok = sig_and_dominant >= thresh
        robustness.append({
            "threshold": thresh,
            "direction_holds": d_ok,
            "significance_holds": s_ok,
            "holds": d_ok and s_ok,
            "direction_pct": round(dom_share, 4),
            "significance_pct": round(sig_and_dominant, 4),
            "detail": (f"At {thresh:.0%}: {dom_share:.1%} of specifications share the dominant sign; "
                       f"{sig_and_dominant:.1%} are significant in that direction."),
        })

    size = _size_word(median_es, req.effect_metric)
    size_txt = f"{size} (Cohen's conventions for {req.effect_metric}), " if size else ""
    if direction_consistent and sig_and_dominant >= 0.8:
        claim = (f"Across the {n} specifications supplied, {dom_share:.0%} give a {dom_word} estimate and "
                 f"{sig_and_dominant:.0%} are significant in that direction. Median estimate {median_es:.3f}, "
                 f"{size_txt}middle half of estimates [{q1:.3f}, {q3:.3f}].")
    elif direction_consistent:
        claim = (f"The direction is consistent ({dom_share:.0%} {dom_word}), but only {sig_and_dominant:.0%} "
                 f"of specifications are significant in that direction, so whether the result is "
                 f"significant depends on analytic choices. Median estimate {median_es:.3f}, {size_txt}"
                 f"middle half [{q1:.3f}, {q3:.3f}].")
    else:
        claim = (f"The direction is not consistent: {pos:.0%} of specifications are positive and {neg:.0%} "
                 f"negative (threshold {req.core_threshold:.0%}). The sign of the result depends on analytic choices.")

    boundaries = _boundary_analysis(specs, dominant)
    flipping = [b["decision"] for b in boundaries if b["flip_rate"] > 0]
    if flipping:
        claim += f" Choices about {', '.join(flipping[:3])} can reverse the sign of the median estimate."

    return CCAResponse(
        n_specifications=n,
        direction_positive_pct=round(pos, 4),
        direction_negative_pct=round(neg, 4),
        direction_consistent=direction_consistent,
        dominant_direction=dom_word,
        median_effect=round(median_es, 6),
        iqr_lower=round(q1, 6),
        iqr_upper=round(q3, 6),
        trimmed_mean_90=round(trimmed_mean, 6),
        magnitude_range=[round(float(es.min()), 6), round(float(es.max()), 6)],
        pct_significant=round(pct_sig, 4),
        pct_significant_after_trim=round(pct_sig_trim, 4),
        core_claim=claim,
        core_direction=dom_word,
        core_magnitude_lower=round(q1, 6),
        core_magnitude_upper=round(q3, 6),
        core_confidence=round(sig_and_dominant, 4),
        caveat=CAVEAT,
        boundary_conditions=boundaries,
        robustness_thresholds=robustness,
    )


@router.post("/from-multiverse")
async def cca_from_multiverse(req: dict):
    """Accepts raw multiverse output and converts it into the request format."""
    specs_raw = req.get("specifications", [])
    specifications = [
        Specification(
            effect_size=s.get("effect_size", 0),
            p_value=s.get("p_value", 1),
            decisions=s.get("decisions"),
        )
        for s in specs_raw
    ]
    cca_req = CCARequest(
        specifications=specifications,
        alpha=req.get("alpha", 0.05),
        core_threshold=req.get("core_threshold", 0.95),
        effect_metric=req.get("effect_metric", "other"),
    )
    return await convergent_core_analysis(cca_req)
