"""Interpretation Confidence — positive predictive value of a significant result.

Ioannidis (2005), without the bias term:

    PPV = power · prior / (power · prior + alpha · (1 − prior))

PPV is the share of results reaching p < alpha that reflect true effects, given
the prior probability that a tested hypothesis is true and the study's power.

Power must be power for a plausible TRUE effect size, chosen in advance. Power
computed from the observed effect ("post-hoc" or "observed" power) is a
function of the p-value and says nothing new (Hoenig & Heisey, 2001), so it is
not used here.

The calculation uses the alpha threshold, not the exact p-value; a result with
p = .001 and one with p = .049 get the same PPV.
"""

from __future__ import annotations

import math
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import stats as sp_stats

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────

class PPVRequest(BaseModel):
    power: Optional[float] = Field(default=None, ge=0, le=1)  # power for an assumed true effect
    # Alternatively: an assumed true effect (Cohen's d, two independent groups) and the total N
    assumed_effect_size: Optional[float] = None
    n: Optional[int] = None
    prior_probability: float = Field(default=0.5, ge=0.001, le=0.999)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    p_value: Optional[float] = None  # accepted for context only; PPV uses the alpha threshold
    effect_size: Optional[float] = None  # deprecated alias of assumed_effect_size


class PriorSensitivity(BaseModel):
    prior: float
    ppv: float
    fpr: float


class PowerSensitivity(BaseModel):
    power: float
    ppv: float
    fpr: float


class PPVResponse(BaseModel):
    ppv: float
    false_positive_risk: float  # 1 − PPV
    power_used: float
    power_source: str  # "given" or "computed for the assumed effect size"
    post_hoc_power: Optional[float] = None  # kept for compatibility; always None
    prior_sensitivity: List[PriorSensitivity]
    power_sensitivity: List[PowerSensitivity]
    interpretation: str
    confidence_level: str  # "high" (≥ .85), "moderate" (≥ .60), "low" (≥ .30), "very_low"


# ── Helpers ───────────────────────────────────────────────────────────────

def _power_two_sample(d: float, n: int, alpha: float) -> float:
    """Two-sided power, two independent groups of n/2 each (noncentral t)."""
    df = n - 2
    ncp = abs(d) * math.sqrt(n / 4)
    crit = sp_stats.t.ppf(1 - alpha / 2, df)
    return float(sp_stats.nct.sf(crit, df, ncp) + sp_stats.nct.cdf(-crit, df, ncp))


def _ppv(power: float, prior: float, alpha: float) -> float:
    den = power * prior + alpha * (1 - prior)
    return power * prior / den if den > 0 else 0.0


def _level(ppv: float) -> str:
    return "high" if ppv >= 0.85 else "moderate" if ppv >= 0.60 else "low" if ppv >= 0.30 else "very_low"


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=PPVResponse)
async def compute_ppv(req: PPVRequest):
    alpha, prior = req.alpha, req.prior_probability
    assumed = req.assumed_effect_size if req.assumed_effect_size is not None else req.effect_size

    if req.power is not None:
        power, source = float(req.power), "given"
    elif assumed is not None and req.n is not None:
        if req.n < 4:
            raise HTTPException(422, "n must be at least 4.")
        power = _power_two_sample(assumed, req.n, alpha)
        source = "computed for the assumed effect size (two independent groups, equal n)"
    else:
        raise HTTPException(422, "Provide 'power', or an assumed true effect size (Cohen's d) and the total n.")

    ppv = _ppv(power, prior, alpha)
    prior_sens = [PriorSensitivity(prior=p, ppv=round(_ppv(power, p, alpha), 4), fpr=round(1 - _ppv(power, p, alpha), 4))
                  for p in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)]
    power_sens = [PowerSensitivity(power=w, ppv=round(_ppv(w, prior, alpha), 4), fpr=round(1 - _ppv(w, prior, alpha), 4))
                  for w in (0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99)]

    parts = [
        f"With power {power:.2f}, a prior probability of {prior:.2f} that the hypothesis is true, and α = {alpha}, "
        f"{ppv:.0%} of results reaching p < {alpha} would reflect true effects and {1 - ppv:.0%} would be false positives.",
    ]
    if ppv < 0.5:
        parts.append("Under these assumptions most significant results would be false positives. Higher power, a "
                     "stricter α, or a better-supported hypothesis would change that.")
    elif ppv < 0.85:
        parts.append("A substantial share of significant results would be false positives; replication would "
                     "strengthen the conclusion.")
    else:
        parts.append("Under these assumptions significant results would usually reflect true effects.")
    parts.append("This assumes no bias (no p-hacking or selective reporting), which Ioannidis (2005) shows lowers "
                 "PPV further. The figure depends heavily on the prior, so check the table across priors.")
    if req.p_value is not None:
        parts.append(f"Your p-value ({req.p_value:.4g}) is not used beyond being below or above α.")

    return PPVResponse(
        ppv=round(ppv, 4), false_positive_risk=round(1 - ppv, 4), power_used=round(power, 4),
        power_source=source, post_hoc_power=None, prior_sensitivity=prior_sens, power_sensitivity=power_sens,
        interpretation=" ".join(parts), confidence_level=_level(ppv),
    )
