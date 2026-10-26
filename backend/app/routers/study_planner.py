"""Study Planner — power analysis, sample size calculation, study design helpers."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from statsmodels.stats.power import (
    TTestIndPower, TTestPower, FTestAnovaPower,
    GofChisquarePower, NormalIndPower,
)

router = APIRouter()


class SampleSizeRequest(BaseModel):
    test_type: str  # "t-test-ind", "t-test-paired", "anova", "chi-square", "correlation"
    effect_size: float
    alpha: float = 0.05
    power: float = 0.80
    n_groups: int = 2          # for ANOVA
    n_levels: int = 2          # for chi-square (df)
    alternative: str = "two-sided"


class PowerCurveRequest(BaseModel):
    test_type: str = "t-test-ind"
    effect_sizes: list[float] = [0.2, 0.5, 0.8]
    alpha: float = 0.05
    sample_sizes: list[int] = [10, 20, 30, 50, 80, 100, 150, 200]


class EffectSizeConverterRequest(BaseModel):
    value: float
    from_type: str  # "d", "r", "eta_squared", "f", "odds_ratio"
    to_type: str


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/sample-size")
def calculate_sample_size(req: SampleSizeRequest):
    """Calculate required sample size for various test types."""
    if req.effect_size <= 0:
        raise HTTPException(400, "Effect size must be > 0.")
    if not (0 < req.alpha < 1) or not (0 < req.power < 1):
        raise HTTPException(400, "Alpha and power must be between 0 and 1.")

    if req.test_type == "t-test-ind":
        analysis = TTestIndPower()
        n = analysis.solve_power(
            effect_size=req.effect_size, alpha=req.alpha,
            power=req.power, alternative=req.alternative,
        )
        return {
            "n_per_group": int(np.ceil(n)),
            "total_n": int(np.ceil(n)) * 2,
            "test": "Independent samples t-test",
            "effect_size": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
        }

    elif req.test_type == "t-test-paired":
        analysis = TTestPower()
        n = analysis.solve_power(
            effect_size=req.effect_size, alpha=req.alpha,
            power=req.power, alternative=req.alternative,
        )
        return {
            "n_pairs": int(np.ceil(n)),
            "total_n": int(np.ceil(n)),
            "test": "Paired samples t-test",
            "effect_size": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
        }

    elif req.test_type == "anova":
        analysis = FTestAnovaPower()
        n = analysis.solve_power(
            effect_size=req.effect_size, alpha=req.alpha,
            power=req.power, k_groups=req.n_groups,
        )
        return {
            "n_per_group": int(np.ceil(n)),
            "total_n": int(np.ceil(n)) * req.n_groups,
            "test": f"One-way ANOVA ({req.n_groups} groups)",
            "effect_size_f": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
        }

    elif req.test_type == "chi-square":
        analysis = GofChisquarePower()
        n = analysis.solve_power(
            effect_size=req.effect_size, alpha=req.alpha,
            power=req.power, n_bins=req.n_levels,
        )
        return {
            "total_n": int(np.ceil(n)),
            "test": "Chi-square test",
            "effect_size_w": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
        }

    elif req.test_type == "correlation":
        # Use Fisher's z transform approach
        z = 0.5 * np.log((1 + req.effect_size) / (1 - req.effect_size)) if abs(req.effect_size) < 1 else req.effect_size
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - req.alpha / 2)
        z_beta = norm.ppf(req.power)
        n = int(np.ceil(((z_alpha + z_beta) / z) ** 2 + 3))
        return {
            "total_n": n,
            "test": "Pearson correlation",
            "effect_size_r": req.effect_size,
            "alpha": req.alpha,
            "power": req.power,
        }

    else:
        raise HTTPException(400, f"Unknown test_type: {req.test_type}")


@router.post("/power-curve")
def power_curve(req: PowerCurveRequest):
    """Generate power curves across sample sizes for given effect sizes."""
    if req.test_type == "t-test-ind":
        analysis = TTestIndPower()
    elif req.test_type == "t-test-paired":
        analysis = TTestPower()
    else:
        raise HTTPException(400, "Power curve currently supports t-test-ind and t-test-paired.")

    curves = {}
    for es in req.effect_sizes:
        powers = []
        for n in req.sample_sizes:
            try:
                p = analysis.power(effect_size=es, nobs=n, alpha=req.alpha)
                powers.append({"n": n, "power": float(p)})
            except Exception:
                powers.append({"n": n, "power": None})
        curves[str(es)] = powers

    return {
        "effect_sizes": req.effect_sizes,
        "sample_sizes": req.sample_sizes,
        "alpha": req.alpha,
        "curves": curves,
    }


@router.post("/effect-size-convert")
def convert_effect_size(req: EffectSizeConverterRequest):
    """Convert between effect size measures."""
    d = None
    if req.from_type == "d":
        d = req.value
    elif req.from_type == "r":
        d = 2 * req.value / np.sqrt(1 - req.value**2) if abs(req.value) < 1 else None
    elif req.from_type == "eta_squared":
        f = np.sqrt(req.value / (1 - req.value)) if req.value < 1 else None
        d = 2 * f if f is not None else None
    elif req.from_type == "f":
        d = 2 * req.value
    elif req.from_type == "odds_ratio":
        d = np.log(req.value) * np.sqrt(3) / np.pi if req.value > 0 else None
    else:
        raise HTTPException(400, f"Unknown from_type: {req.from_type}")

    if d is None:
        raise HTTPException(400, "Could not convert — check input value range.")

    # Convert d to target
    result = {}
    if req.to_type == "d":
        result["value"] = float(d)
    elif req.to_type == "r":
        result["value"] = float(d / np.sqrt(d**2 + 4))
    elif req.to_type == "eta_squared":
        f = d / 2
        result["value"] = float(f**2 / (f**2 + 1))
    elif req.to_type == "f":
        result["value"] = float(d / 2)
    elif req.to_type == "odds_ratio":
        result["value"] = float(np.exp(d * np.pi / np.sqrt(3)))
    else:
        raise HTTPException(400, f"Unknown to_type: {req.to_type}")

    result["from"] = req.from_type
    result["to"] = req.to_type
    result["original_value"] = req.value
    result["interpretation"] = _interpret_d(d)
    return result


def _interpret_d(d: float) -> str:
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "Large effect"
    elif abs_d >= 0.5:
        return "Medium effect"
    elif abs_d >= 0.2:
        return "Small effect"
    else:
        return "Negligible effect"
