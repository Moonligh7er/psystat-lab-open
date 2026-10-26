"""Effect Size — calculation, conversion, interpretation, CIs, NNT, and power.

Textbook-accurate effect size computations with LaTeX formulas,
confidence intervals via non-central distributions and analytic methods,
and field-specific interpretation benchmarks.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
import math
from typing import Optional

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class FromStatisticsRequest(BaseModel):
    """Calculate effect sizes from summary statistics."""
    stat_type: str  # "means", "t", "F", "chi_square", "r", "odds_ratio",
                    # "proportions", "correlation", "r_squared"
    # Two-group means
    mean1: float | None = None
    mean2: float | None = None
    sd1: float | None = None
    sd2: float | None = None
    n1: int | None = None
    n2: int | None = None
    # t-value
    t_value: float | None = None
    df: float | None = None
    # F-value
    f_value: float | None = None
    df1: float | None = None
    df2: float | None = None
    # Chi-square
    chi_square: float | None = None
    N: int | None = None
    # Correlation
    r: float | None = None
    # Odds ratio
    odds_ratio: float | None = None
    # Proportions
    p1: float | None = None
    p2: float | None = None
    # Regression
    r_squared: float | None = None
    n: int | None = None
    k: int | None = None  # number of predictors
    alpha: float = 0.05


class ConvertRequest(BaseModel):
    """Convert between effect size types."""
    from_type: str  # d, g, r, OR, log_OR, eta_sq, omega_sq, f, f_sq, w, h, CLES
    from_value: float
    to_type: str
    n1: int | None = None
    n2: int | None = None
    df: float | None = None
    N: int | None = None


class InterpretRequest(BaseModel):
    """Interpret effect size magnitude."""
    effect_size_type: str
    value: float
    context: str | None = None  # e.g., "clinical_psychology", "education",
                                # "social_psychology", "medicine"


class ConfidenceIntervalRequest(BaseModel):
    """Confidence interval for an effect size."""
    effect_size_type: str  # d, g, r, eta_sq, OR, log_OR, f, w, h
    value: float
    n: int | None = None
    n1: int | None = None
    n2: int | None = None
    alpha: float = 0.05


class NNTRequest(BaseModel):
    """Number needed to treat and clinical metrics."""
    effect_size_type: str  # d, r, OR
    value: float
    base_rate: float  # control group event rate (0-1)


class PowerFromEffectRequest(BaseModel):
    """Power analysis from effect size."""
    effect_size_type: str  # d, r, f, f_sq, w, eta_sq
    value: float
    n: int | None = None
    n1: int | None = None
    n2: int | None = None
    alpha: float = 0.05
    test_type: str = "two_sample_t"  # two_sample_t, one_sample_t, paired_t,
                                      # correlation, anova, chi_square, regression


# ── Helpers ───────────────────────────────────────────────────────────

def _pooled_sd(sd1: float, sd2: float, n1: int, n2: int) -> float:
    """Compute pooled standard deviation (Cohen's formula)."""
    return math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))


def _hedges_correction(df: float) -> float:
    """Hedges' g correction factor J (exact via gamma function)."""
    # J = Gamma(df/2) / (sqrt(df/2) * Gamma((df-1)/2))
    # Approximation: J ≈ 1 - 3/(4*df - 1)
    return 1 - 3 / (4 * df - 1)


def _se_d(d: float, n1: int, n2: int) -> float:
    """Standard error of Cohen's d (Hedges & Olkin, 1985)."""
    return math.sqrt(n1 + n2) / math.sqrt(n1 * n2) * math.sqrt(1 + d**2 * n1 * n2 / (2 * (n1 + n2)))


def _se_r(r: float, n: int) -> float:
    """Standard error of r using Fisher z-transform."""
    if n <= 3:
        return float('inf')
    return 1 / math.sqrt(n - 3)


def _fisher_z(r: float) -> float:
    """Fisher z-transformation of r."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * math.log((1 + r) / (1 - r))


def _fisher_z_inv(z: float) -> float:
    """Inverse Fisher z-transformation."""
    return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)


def _d_to_r(d: float, n1: int | None = None, n2: int | None = None) -> float:
    """Convert Cohen's d to point-biserial r."""
    if n1 is not None and n2 is not None:
        a = (n1 + n2)**2 / (n1 * n2)
    else:
        a = 4  # equal groups assumption
    return d / math.sqrt(d**2 + a)


def _r_to_d(r: float, n1: int | None = None, n2: int | None = None) -> float:
    """Convert point-biserial r to Cohen's d."""
    if n1 is not None and n2 is not None:
        a = (n1 + n2)**2 / (n1 * n2)
    else:
        a = 4
    return r * math.sqrt(a) / math.sqrt(1 - r**2) if abs(r) < 1 else float('inf') * np.sign(r)


def _d_to_or(d: float) -> float:
    """Convert Cohen's d to odds ratio (logistic approximation, Borenstein 2009)."""
    return math.exp(d * math.pi / math.sqrt(3))


def _or_to_d(odds_ratio: float) -> float:
    """Convert odds ratio to Cohen's d."""
    if odds_ratio <= 0:
        raise ValueError("Odds ratio must be positive")
    return math.log(odds_ratio) * math.sqrt(3) / math.pi


def _d_to_cles(d: float) -> float:
    """Common language effect size (probability of superiority) from d."""
    return float(sp_stats.norm.cdf(d / math.sqrt(2)))


def _d_to_u3(d: float) -> float:
    """Cohen's U3 from d — % of treatment group above control median."""
    return float(sp_stats.norm.cdf(d))


def _d_to_overlap(d: float) -> float:
    """Overlap coefficient (OVL) from d."""
    return float(2 * sp_stats.norm.cdf(-abs(d) / 2))


def _eta_sq_to_f(eta_sq: float) -> float:
    """Convert eta-squared to Cohen's f."""
    if eta_sq >= 1:
        return float('inf')
    return math.sqrt(eta_sq / (1 - eta_sq))


def _f_to_eta_sq(f: float) -> float:
    """Convert Cohen's f to eta-squared."""
    return f**2 / (1 + f**2)


def _f_to_d(f: float) -> float:
    """Convert Cohen's f to d (2-group case)."""
    return 2 * f


def _d_to_f(d: float) -> float:
    """Convert d to Cohen's f (2-group case)."""
    return abs(d) / 2


def _d_to_h(d: float) -> float:
    """Approximate conversion from d to Cohen's h via proportions."""
    # Not a direct conversion; d ≈ h for small effects
    return d


def _ci_d_nct(d: float, n1: int, n2: int, alpha: float = 0.05) -> dict:
    """CI for d using non-central t distribution."""
    df = n1 + n2 - 2
    ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
    se = _se_d(d, n1, n2)
    try:
        t_lower = sp_stats.nct.ppf(alpha / 2, df, ncp)
        t_upper = sp_stats.nct.ppf(1 - alpha / 2, df, ncp)
        d_lower = t_lower * math.sqrt((n1 + n2) / (n1 * n2))
        d_upper = t_upper * math.sqrt((n1 + n2) / (n1 * n2))
    except Exception:
        # Fallback to normal approximation
        z = sp_stats.norm.ppf(1 - alpha / 2)
        d_lower = d - z * se
        d_upper = d + z * se
    return {"lower": d_lower, "upper": d_upper, "se": se, "method": "non-central t"}


def _ci_r_fisher(r: float, n: int, alpha: float = 0.05) -> dict:
    """CI for r using Fisher z-transform."""
    z = _fisher_z(r)
    se_z = _se_r(r, n)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z
    r_lower = _fisher_z_inv(z_lower)
    r_upper = _fisher_z_inv(z_upper)
    return {"lower": r_lower, "upper": r_upper, "se": se_z, "method": "Fisher z-transform"}


# ── Conversion Graph ─────────────────────────────────────────────────

def _convert_direct(from_type: str, from_value: float, to_type: str,
                    n1: int | None = None, n2: int | None = None,
                    df: float | None = None, N: int | None = None) -> dict | None:
    """Try a direct conversion. Returns dict with value, formula, notes or None."""
    ft, tt = from_type.lower(), to_type.lower()
    v = from_value
    result = None

    if ft == tt:
        return {"value": v, "formula": f"{ft} = {ft}", "notes": "Identity conversion."}

    # ── d-based conversions ──
    if ft == "d":
        if tt == "g":
            total_n = (n1 or 30) + (n2 or 30) - 2
            j = _hedges_correction(total_n)
            result = {"value": v * j,
                      "formula": r"g = d \cdot J = d \cdot \left(1 - \frac{3}{4\,df - 1}\right)",
                      "notes": f"Hedges correction J = {j:.6f}, df = {total_n}"}
        elif tt == "r":
            result = {"value": _d_to_r(v, n1, n2),
                      "formula": r"r = \frac{d}{\sqrt{d^2 + a}}, \; a = \frac{(n_1+n_2)^2}{n_1 n_2}",
                      "notes": "Point-biserial r." + (" Equal groups assumed." if n1 is None else "")}
        elif tt == "or":
            result = {"value": _d_to_or(v),
                      "formula": r"OR = e^{d\,\pi / \sqrt{3}}",
                      "notes": "Logistic approximation (Borenstein et al., 2009)."}
        elif tt == "log_or":
            result = {"value": v * math.pi / math.sqrt(3),
                      "formula": r"\ln(OR) = d\,\pi / \sqrt{3}",
                      "notes": "Logistic approximation."}
        elif tt == "eta_sq":
            f_val = _d_to_f(v)
            result = {"value": _f_to_eta_sq(f_val),
                      "formula": r"\eta^2 = \frac{d^2/4}{1 + d^2/4}",
                      "notes": "Via f = d/2 (2-group case)."}
        elif tt == "f":
            result = {"value": _d_to_f(v),
                      "formula": r"f = |d| / 2",
                      "notes": "For two-group comparison."}
        elif tt == "f_sq":
            result = {"value": (_d_to_f(v))**2,
                      "formula": r"f^2 = d^2 / 4",
                      "notes": "For two-group comparison."}
        elif tt == "cles":
            result = {"value": _d_to_cles(v),
                      "formula": r"CLES = \Phi\!\left(\frac{d}{\sqrt{2}}\right)",
                      "notes": "Probability of superiority."}
        elif tt == "w":
            # d -> f -> w (w ≈ f for 2-group)
            result = {"value": _d_to_f(v),
                      "formula": r"w \approx f = |d| / 2",
                      "notes": "Approximate; w = f for balanced 2×2 designs."}
        elif tt == "h":
            result = {"value": v,
                      "formula": r"h \approx d",
                      "notes": "Approximate equivalence for small-to-moderate effects."}
        elif tt == "omega_sq":
            f_val = _d_to_f(v)
            eta = _f_to_eta_sq(f_val)
            total_n = (n1 or 30) + (n2 or 30)
            k_groups = 2
            omega = (eta * (total_n - k_groups) - (k_groups - 1) * (1 - eta)) / (total_n - 1)
            omega = max(0, omega)
            result = {"value": omega,
                      "formula": r"\omega^2 = \frac{\eta^2(N - k) - (k-1)(1-\eta^2)}{N - 1}",
                      "notes": "Bias-corrected; assumes k=2 groups."}

    # ── g-based conversions ──
    elif ft == "g":
        if tt == "d":
            total_n = (n1 or 30) + (n2 or 30) - 2
            j = _hedges_correction(total_n)
            result = {"value": v / j if j != 0 else v,
                      "formula": r"d = g / J",
                      "notes": f"Inverse Hedges correction, J = {j:.6f}"}
        else:
            # Convert g -> d -> target
            total_n = (n1 or 30) + (n2 or 30) - 2
            j = _hedges_correction(total_n)
            d_val = v / j if j != 0 else v
            return _convert_direct("d", d_val, to_type, n1, n2, df, N)

    # ── r-based conversions ──
    elif ft == "r":
        if tt == "d":
            result = {"value": _r_to_d(v, n1, n2),
                      "formula": r"d = \frac{r\sqrt{a}}{\sqrt{1 - r^2}}",
                      "notes": "Point-biserial r to d."}
        elif tt == "r_squared":
            result = {"value": v**2,
                      "formula": r"r^2 = r \times r",
                      "notes": "Coefficient of determination."}
        elif tt == "fisher_z":
            result = {"value": _fisher_z(v),
                      "formula": r"z_r = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right)",
                      "notes": "Fisher z-transformation."}
        elif tt == "or":
            d_val = _r_to_d(v, n1, n2)
            result = {"value": _d_to_or(d_val),
                      "formula": r"OR = e^{\pi\,d / \sqrt{3}}, \; d = r\sqrt{a}/\sqrt{1-r^2}",
                      "notes": "Via d (logistic approximation)."}
        else:
            # r -> d -> target
            d_val = _r_to_d(v, n1, n2)
            sub = _convert_direct("d", d_val, to_type, n1, n2, df, N)
            if sub:
                sub["notes"] = "Converted via r → d → target. " + sub.get("notes", "")
            return sub

    # ── OR-based conversions ──
    elif ft == "or":
        if v <= 0:
            raise ValueError("Odds ratio must be positive")
        if tt == "d":
            result = {"value": _or_to_d(v),
                      "formula": r"d = \frac{\ln(OR)\,\sqrt{3}}{\pi}",
                      "notes": "Logistic approximation."}
        elif tt == "log_or":
            result = {"value": math.log(v),
                      "formula": r"\ln(OR) = \log_e(OR)",
                      "notes": "Natural logarithm of odds ratio."}
        elif tt == "r":
            d_val = _or_to_d(v)
            result = {"value": _d_to_r(d_val, n1, n2),
                      "formula": r"r = d / \sqrt{d^2 + a}, \; d = \ln(OR)\sqrt{3}/\pi",
                      "notes": "Via d."}
        else:
            d_val = _or_to_d(v)
            sub = _convert_direct("d", d_val, to_type, n1, n2, df, N)
            if sub:
                sub["notes"] = "Converted via OR → d → target. " + sub.get("notes", "")
            return sub

    # ── log_OR-based conversions ──
    elif ft == "log_or":
        or_val = math.exp(v)
        return _convert_direct("or", or_val, to_type, n1, n2, df, N)

    # ── eta_sq-based conversions ──
    elif ft == "eta_sq":
        if tt == "f":
            result = {"value": _eta_sq_to_f(v),
                      "formula": r"f = \sqrt{\frac{\eta^2}{1 - \eta^2}}",
                      "notes": ""}
        elif tt == "d":
            f_val = _eta_sq_to_f(v)
            result = {"value": _f_to_d(f_val),
                      "formula": r"d = 2f = 2\sqrt{\frac{\eta^2}{1 - \eta^2}}",
                      "notes": "2-group case only."}
        elif tt == "f_sq":
            result = {"value": v / (1 - v) if v < 1 else float('inf'),
                      "formula": r"f^2 = \frac{\eta^2}{1 - \eta^2}",
                      "notes": ""}
        else:
            f_val = _eta_sq_to_f(v)
            d_val = _f_to_d(f_val)
            sub = _convert_direct("d", d_val, to_type, n1, n2, df, N)
            if sub:
                sub["notes"] = "Converted via η² → f → d → target. " + sub.get("notes", "")
            return sub

    # ── omega_sq-based conversions ──
    elif ft == "omega_sq":
        # omega_sq ≈ eta_sq - bias; for conversion, treat similarly
        if tt == "eta_sq":
            total_n = N or ((n1 or 30) + (n2 or 30))
            k_groups = 2
            # Reverse: eta = (omega*(N-1) + (k-1)) / (N - k + omega*(N-1) + (k-1))
            # Simplified for k=2
            eta = (v * (total_n - 1) + 1) / (total_n - 2 + v * (total_n - 1) + 1)
            result = {"value": eta,
                      "formula": r"\eta^2 \approx \frac{\omega^2(N-1)+(k-1)}{N - k + \omega^2(N-1)+(k-1)}",
                      "notes": "Approximate inverse; assumes k=2."}
        else:
            # omega -> eta -> f -> d -> target
            total_n = N or ((n1 or 30) + (n2 or 30))
            eta_result = _convert_direct("omega_sq", v, "eta_sq", n1, n2, df, total_n)
            if eta_result:
                return _convert_direct("eta_sq", eta_result["value"], to_type, n1, n2, df, N)
            return None

    # ── f-based conversions ──
    elif ft == "f":
        if tt == "eta_sq":
            result = {"value": _f_to_eta_sq(v),
                      "formula": r"\eta^2 = \frac{f^2}{1 + f^2}",
                      "notes": ""}
        elif tt == "d":
            result = {"value": _f_to_d(v),
                      "formula": r"d = 2f",
                      "notes": "2-group case."}
        elif tt == "f_sq":
            result = {"value": v**2,
                      "formula": r"f^2 = f \times f",
                      "notes": ""}
        else:
            d_val = _f_to_d(v)
            sub = _convert_direct("d", d_val, to_type, n1, n2, df, N)
            if sub:
                sub["notes"] = "Converted via f → d → target. " + sub.get("notes", "")
            return sub

    # ── f_sq-based conversions ──
    elif ft == "f_sq":
        f_val = math.sqrt(max(0, v))
        sub = _convert_direct("f", f_val, to_type, n1, n2, df, N)
        if sub:
            sub["notes"] = "Converted via f² → f → target. " + sub.get("notes", "")
        return sub

    # ── w-based conversions ──
    elif ft == "w":
        # w ≈ f for balanced designs
        sub = _convert_direct("f", v, to_type, n1, n2, df, N)
        if sub:
            sub["notes"] = "w treated as f (equivalent for balanced designs). " + sub.get("notes", "")
        return sub

    # ── h-based conversions ──
    elif ft == "h":
        # h → d approximate
        sub = _convert_direct("d", v, to_type, n1, n2, df, N)
        if sub:
            sub["notes"] = "h ≈ d (approximate). " + sub.get("notes", "")
        return sub

    # ── CLES-based conversions ──
    elif ft == "cles":
        # Inverse: d = sqrt(2) * Phi_inv(CLES)
        if not (0 < v < 1):
            raise ValueError("CLES must be between 0 and 1 (exclusive)")
        d_val = math.sqrt(2) * float(sp_stats.norm.ppf(v))
        sub = _convert_direct("d", d_val, to_type, n1, n2, df, N)
        if sub:
            sub["notes"] = f"Converted via CLES → d = {d_val:.4f} → target. " + sub.get("notes", "")
            sub["formula"] = r"d = \sqrt{2}\,\Phi^{-1}(CLES), \;" + sub.get("formula", "")
        return sub

    # ── fisher_z-based ──
    elif ft == "fisher_z":
        r_val = _fisher_z_inv(v)
        sub = _convert_direct("r", r_val, to_type, n1, n2, df, N)
        if sub:
            sub["notes"] = "Converted via Fisher z → r → target. " + sub.get("notes", "")
        return sub

    return result


# ── Benchmarks ───────────────────────────────────────────────────────

COHEN_BENCHMARKS = {
    "d": {"small": 0.2, "medium": 0.5, "large": 0.8},
    "g": {"small": 0.2, "medium": 0.5, "large": 0.8},
    "r": {"small": 0.1, "medium": 0.3, "large": 0.5},
    "f": {"small": 0.1, "medium": 0.25, "large": 0.4},
    "f_sq": {"small": 0.02, "medium": 0.15, "large": 0.35},
    "eta_sq": {"small": 0.01, "medium": 0.06, "large": 0.14},
    "partial_eta_sq": {"small": 0.01, "medium": 0.06, "large": 0.14},
    "omega_sq": {"small": 0.01, "medium": 0.06, "large": 0.14},
    "w": {"small": 0.1, "medium": 0.3, "large": 0.5},
    "h": {"small": 0.2, "medium": 0.5, "large": 0.8},
    "OR": {"small": 1.5, "medium": 2.5, "large": 4.3},
    "CLES": {"small": 0.56, "medium": 0.64, "large": 0.71},
}

SAWILOWSKY_BENCHMARKS = {
    "d": {"very_small": 0.01, "small": 0.2, "medium": 0.5,
           "large": 0.8, "very_large": 1.2, "huge": 2.0},
    "g": {"very_small": 0.01, "small": 0.2, "medium": 0.5,
           "large": 0.8, "very_large": 1.2, "huge": 2.0},
    "r": {"very_small": 0.05, "small": 0.1, "medium": 0.3,
           "large": 0.5, "very_large": 0.7, "huge": 0.9},
}

FIELD_NORMS = {
    "social_psychology": {
        "d": {"mean": 0.43, "median": 0.36,
              "source": "Richard, Bond, & Stokes-Zoota (2003)"},
        "r": {"mean": 0.21, "median": 0.18,
              "source": "Richard, Bond, & Stokes-Zoota (2003)"},
    },
    "clinical_psychology": {
        "d": {"mean": 0.68, "median": 0.58,
              "source": "Lipsey & Wilson (1993), psychotherapy outcomes"},
        "r": {"mean": 0.32, "median": 0.28,
              "source": "Meyer et al. (2001)"},
    },
    "education": {
        "d": {"mean": 0.40, "median": 0.33,
              "source": "Hattie (2009) synthesis of meta-analyses"},
        "r": {"mean": 0.20, "median": 0.16,
              "source": "Hattie (2009)"},
    },
    "medicine": {
        "d": {"mean": 0.35, "median": 0.27,
              "source": "Leucht et al. (2012), pharmacological interventions"},
        "OR": {"mean": 1.86, "median": 1.52,
               "source": "Leucht et al. (2012)"},
    },
    "organizational_psychology": {
        "d": {"mean": 0.45, "median": 0.38,
              "source": "Schmidt & Hunter (2015)"},
        "r": {"mean": 0.22, "median": 0.19,
              "source": "Schmidt & Hunter (2015)"},
    },
}


# ── Endpoint 1: /from-statistics ─────────────────────────────────────

@router.post("/from-statistics")
async def from_statistics(req: FromStatisticsRequest):
    """Calculate effect sizes from summary statistics."""
    try:
        st = req.stat_type.lower()

        # ── Two-group means ──
        if st == "means":
            for field in ("mean1", "mean2", "sd1", "sd2", "n1", "n2"):
                if getattr(req, field) is None:
                    raise HTTPException(status_code=422,
                                        detail=f"'{field}' is required for stat_type='means'")
            m1, m2 = req.mean1, req.mean2
            s1, s2 = req.sd1, req.sd2
            n1, n2 = req.n1, req.n2
            if s1 <= 0 or s2 <= 0:
                raise HTTPException(status_code=422, detail="Standard deviations must be positive")
            if n1 < 2 or n2 < 2:
                raise HTTPException(status_code=422, detail="Each group must have n >= 2")

            sp = _pooled_sd(s1, s2, n1, n2)
            d = (m1 - m2) / sp
            df = n1 + n2 - 2
            j = _hedges_correction(df)
            g = d * j
            glass_delta = (m1 - m2) / s2  # control group SD
            r = _d_to_r(d, n1, n2)
            odds_ratio = _d_to_or(d)
            cles = _d_to_cles(d)

            se_d = _se_d(d, n1, n2)
            se_g = se_d * j
            ci_d = _ci_d_nct(d, n1, n2, req.alpha)
            ci_g = {"lower": ci_d["lower"] * j, "upper": ci_d["upper"] * j,
                    "se": se_g, "method": "non-central t (corrected)"}

            total_n = n1 + n2
            ci_r = _ci_r_fisher(r, total_n, req.alpha)

            se_glass = math.sqrt((n1 + n2) / (n1 * n2) + glass_delta**2 / (2 * (n2 - 1)))

            return {
                "cohens_d": {"value": d, "se": se_d, "ci": ci_d,
                             "formula": r"d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}"},
                "hedges_g": {"value": g, "se": se_g, "ci": ci_g,
                             "formula": r"g = d \cdot J, \; J = 1 - \frac{3}{4\,df - 1}"},
                "glass_delta": {"value": glass_delta, "se": se_glass,
                                "formula": r"\Delta = \frac{\bar{X}_1 - \bar{X}_2}{s_2}"},
                "r": {"value": r, "ci": ci_r,
                       "formula": r"r = \frac{d}{\sqrt{d^2 + a}}"},
                "odds_ratio": {"value": odds_ratio,
                               "formula": r"OR = e^{d\pi / \sqrt{3}}"},
                "cles": {"value": cles,
                         "formula": r"CLES = \Phi\!\left(\frac{d}{\sqrt{2}}\right)"},
                "pooled_sd": sp,
                "df": df,
            }

        # ── t-value ──
        elif st == "t":
            if req.t_value is None or req.df is None:
                raise HTTPException(status_code=422,
                                    detail="'t_value' and 'df' required for stat_type='t'")
            t = req.t_value
            df = req.df
            n1 = req.n1
            n2 = req.n2

            if n1 is not None and n2 is not None:
                d = t * math.sqrt((n1 + n2) / (n1 * n2))
                se_d_val = _se_d(d, n1, n2)
                ci_d = _ci_d_nct(d, n1, n2, req.alpha)
            else:
                # Approximate: assume equal groups, df = n1+n2-2
                n_approx = int(df / 2) + 1
                d = 2 * t / math.sqrt(df)
                se_d_val = _se_d(d, n_approx, n_approx)
                n1, n2 = n_approx, n_approx
                ci_d = _ci_d_nct(d, n1, n2, req.alpha)

            j = _hedges_correction(df)
            g = d * j
            r = _d_to_r(d, n1, n2)
            ci_r = _ci_r_fisher(r, n1 + n2, req.alpha)

            return {
                "cohens_d": {"value": d, "se": se_d_val, "ci": ci_d,
                             "formula": r"d = t\sqrt{\frac{n_1 + n_2}{n_1 n_2}}"},
                "hedges_g": {"value": g, "se": se_d_val * j,
                             "formula": r"g = d \cdot J"},
                "r": {"value": r, "ci": ci_r,
                       "formula": r"r = \frac{d}{\sqrt{d^2 + a}}"},
            }

        # ── F-value ──
        elif st == "f":
            if req.f_value is None or req.df1 is None or req.df2 is None:
                raise HTTPException(status_code=422,
                                    detail="'f_value', 'df1', 'df2' required for stat_type='F'")
            F = req.f_value
            df1 = req.df1
            df2 = req.df2

            eta_sq = (F * df1) / (F * df1 + df2)
            partial_eta_sq = eta_sq  # identical for one-way designs
            N_total = df1 + df2 + 1
            omega_sq = (F * df1 - df1) / (F * df1 + df2 + 1)
            omega_sq = max(0, omega_sq)
            epsilon_sq = (F - 1) * df1 / (F * df1 + df2)
            epsilon_sq = max(0, epsilon_sq)

            results = {
                "eta_squared": {"value": eta_sq,
                                "formula": r"\eta^2 = \frac{F \cdot df_1}{F \cdot df_1 + df_2}"},
                "partial_eta_squared": {"value": partial_eta_sq,
                                        "formula": r"\eta_p^2 = \frac{F \cdot df_1}{F \cdot df_1 + df_2}"},
                "omega_squared": {"value": omega_sq,
                                  "formula": r"\omega^2 = \frac{(F - 1)\,df_1}{F\,df_1 + df_2 + 1}"},
                "epsilon_squared": {"value": epsilon_sq,
                                    "formula": r"\epsilon^2 = \frac{(F - 1)\,df_1}{F\,df_1 + df_2}"},
            }

            # For 2-group case, also compute d
            if df1 == 1:
                t_equiv = math.sqrt(F)
                n1 = req.n1
                n2 = req.n2
                if n1 and n2:
                    d = t_equiv * math.sqrt((n1 + n2) / (n1 * n2))
                else:
                    d = 2 * t_equiv / math.sqrt(df2)
                    n1 = int(df2 / 2) + 1
                    n2 = n1
                results["cohens_d"] = {
                    "value": d,
                    "se": _se_d(d, n1, n2),
                    "ci": _ci_d_nct(d, n1, n2, req.alpha),
                    "formula": r"d = 2\sqrt{F / df_2} \text{ (2-group case)}",
                }

            # CI for eta-squared via non-central F
            try:
                ncp_lower = 0
                ncp_upper = max(F * df1 * 3, 1)  # initial bracket
                # Find non-centrality parameter bounds
                from scipy.optimize import brentq

                def _ncf_cdf_target(ncp, p_target):
                    return sp_stats.ncf.cdf(F, df1, df2, ncp) - p_target

                # Lower bound
                try:
                    ncp_lo = brentq(_ncf_cdf_target, 0, ncp_upper, args=(1 - req.alpha / 2,))
                except ValueError:
                    ncp_lo = 0
                # Upper bound
                try:
                    ncp_hi = brentq(_ncf_cdf_target, 0, ncp_upper * 2, args=(req.alpha / 2,))
                except ValueError:
                    ncp_hi = ncp_upper
                eta_lower = ncp_lo / (ncp_lo + N_total)
                eta_upper = ncp_hi / (ncp_hi + N_total)
                results["eta_squared"]["ci"] = {
                    "lower": max(0, eta_lower), "upper": min(1, eta_upper),
                    "method": "non-central F"
                }
            except Exception:
                pass

            return results

        # ── Chi-square ──
        elif st == "chi_square":
            if req.chi_square is None or req.N is None:
                raise HTTPException(status_code=422,
                                    detail="'chi_square' and 'N' required for stat_type='chi_square'")
            chi2 = req.chi_square
            N = req.N
            df_val = req.df if req.df is not None else 1

            phi = math.sqrt(chi2 / N)
            # Cramer's V: k = min(rows, cols); df = (r-1)(c-1)
            # For df=1, V = phi. For general case, k = df + 1 (assuming square table)
            k = df_val + 1  # min dimension
            cramers_v = math.sqrt(chi2 / (N * df_val)) if df_val > 0 else phi
            w = math.sqrt(chi2 / N)

            se_w = math.sqrt((1 + w**2) / N)  # approximate
            z_crit = sp_stats.norm.ppf(1 - req.alpha / 2)

            return {
                "phi": {"value": phi,
                        "formula": r"\phi = \sqrt{\chi^2 / N}"},
                "cramers_v": {"value": cramers_v,
                              "formula": r"V = \sqrt{\frac{\chi^2}{N \cdot df}}",
                              "notes": f"df = {df_val}, min dimension = {k}"},
                "cohens_w": {"value": w, "se": se_w,
                             "ci": {"lower": max(0, w - z_crit * se_w),
                                    "upper": w + z_crit * se_w,
                                    "method": "normal approximation"},
                             "formula": r"w = \sqrt{\chi^2 / N}"},
            }

        # ── Correlation r ──
        elif st == "r" or st == "correlation":
            if req.r is None:
                raise HTTPException(status_code=422,
                                    detail="'r' required for stat_type='r'")
            r = req.r
            if abs(r) > 1:
                raise HTTPException(status_code=422, detail="r must be between -1 and 1")
            N = req.N or req.n

            r_sq = r**2
            fz = _fisher_z(r)
            d = _r_to_d(r, req.n1, req.n2)

            results = {
                "r": {"value": r},
                "r_squared": {"value": r_sq,
                              "formula": r"r^2 = r \times r"},
                "fisher_z": {"value": fz,
                             "formula": r"z_r = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right)"},
                "cohens_d": {"value": d,
                             "formula": r"d = \frac{r\sqrt{a}}{\sqrt{1-r^2}}"},
            }

            if N is not None and N > 3:
                ci_r = _ci_r_fisher(r, N, req.alpha)
                results["r"]["ci"] = ci_r
                results["r"]["se"] = 1 / math.sqrt(N - 3)

                se_d = _se_d(d, req.n1 or N // 2, req.n2 or (N - N // 2))
                results["cohens_d"]["se"] = se_d

            if req.N is not None:
                or_val = _d_to_or(d)
                results["odds_ratio"] = {"value": or_val,
                                         "formula": r"OR = e^{d\pi/\sqrt{3}}"}

            return results

        # ── Odds ratio ──
        elif st == "odds_ratio":
            if req.odds_ratio is None:
                raise HTTPException(status_code=422,
                                    detail="'odds_ratio' required for stat_type='odds_ratio'")
            OR = req.odds_ratio
            if OR <= 0:
                raise HTTPException(status_code=422, detail="Odds ratio must be positive")

            log_or = math.log(OR)
            d = _or_to_d(OR)
            r = _d_to_r(d, req.n1, req.n2)

            results = {
                "odds_ratio": {"value": OR},
                "log_odds_ratio": {"value": log_or,
                                   "formula": r"\ln(OR)"},
                "cohens_d": {"value": d,
                             "formula": r"d = \frac{\ln(OR)\,\sqrt{3}}{\pi}"},
                "r": {"value": r,
                       "formula": r"r = d / \sqrt{d^2 + a}"},
            }

            if req.n1 is not None and req.n2 is not None:
                se_log_or = math.sqrt(1/max(1, req.n1) + 1/max(1, req.n2))
                z_crit = sp_stats.norm.ppf(1 - req.alpha / 2)
                results["log_odds_ratio"]["se"] = se_log_or
                results["log_odds_ratio"]["ci"] = {
                    "lower": log_or - z_crit * se_log_or,
                    "upper": log_or + z_crit * se_log_or,
                    "method": "delta method (approximate)"
                }
                results["odds_ratio"]["ci"] = {
                    "lower": math.exp(log_or - z_crit * se_log_or),
                    "upper": math.exp(log_or + z_crit * se_log_or),
                    "method": "delta method"
                }

            return results

        # ── Proportions ──
        elif st == "proportions":
            for field in ("p1", "p2", "n1", "n2"):
                if getattr(req, field) is None:
                    raise HTTPException(status_code=422,
                                        detail=f"'{field}' required for stat_type='proportions'")
            p1, p2 = req.p1, req.p2
            n1, n2 = req.n1, req.n2
            if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
                raise HTTPException(status_code=422, detail="Proportions must be between 0 and 1")

            rd = p1 - p2  # risk difference
            rr = p1 / p2 if p2 > 0 else float('inf')  # risk ratio

            # Odds ratio
            o1 = p1 / (1 - p1) if p1 < 1 else float('inf')
            o2 = p2 / (1 - p2) if p2 < 1 else float('inf')
            OR = o1 / o2 if o2 > 0 else float('inf')

            # NNT
            nnt = 1 / abs(rd) if rd != 0 else float('inf')

            # Cohen's h
            h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

            # SEs
            se_rd = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            se_h = math.sqrt(1 / n1 + 1 / n2)  # approximate
            z_crit = sp_stats.norm.ppf(1 - req.alpha / 2)

            se_log_rr = math.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2)) if p1 > 0 and p2 > 0 else float('inf')
            se_log_or = math.sqrt(1 / (n1 * p1 * (1 - p1)) + 1 / (n2 * p2 * (1 - p2))) if 0 < p1 < 1 and 0 < p2 < 1 else float('inf')

            return {
                "risk_difference": {
                    "value": rd, "se": se_rd,
                    "ci": {"lower": rd - z_crit * se_rd, "upper": rd + z_crit * se_rd,
                           "method": "Wald"},
                    "formula": r"RD = p_1 - p_2",
                },
                "risk_ratio": {
                    "value": rr,
                    "ci": {"lower": math.exp(math.log(rr) - z_crit * se_log_rr),
                           "upper": math.exp(math.log(rr) + z_crit * se_log_rr),
                           "method": "log-transform"} if rr != float('inf') and rr > 0 else None,
                    "formula": r"RR = p_1 / p_2",
                },
                "odds_ratio": {
                    "value": OR,
                    "ci": {"lower": math.exp(math.log(OR) - z_crit * se_log_or),
                           "upper": math.exp(math.log(OR) + z_crit * se_log_or),
                           "method": "log-transform"} if OR not in (0, float('inf')) else None,
                    "formula": r"OR = \frac{p_1/(1-p_1)}{p_2/(1-p_2)}",
                },
                "nnt": {
                    "value": nnt,
                    "formula": r"NNT = 1 / |RD|",
                    "notes": "Positive = NNT (benefit), negative = NNH (harm)" if rd != 0 else "",
                },
                "cohens_h": {
                    "value": h, "se": se_h,
                    "ci": {"lower": h - z_crit * se_h, "upper": h + z_crit * se_h,
                           "method": "normal approximation"},
                    "formula": r"h = 2\arcsin(\sqrt{p_1}) - 2\arcsin(\sqrt{p_2})",
                },
            }

        # ── R-squared (regression) ──
        elif st == "r_squared":
            if req.r_squared is None or req.n is None or req.k is None:
                raise HTTPException(status_code=422,
                                    detail="'r_squared', 'n', 'k' required for stat_type='r_squared'")
            R2 = req.r_squared
            n = req.n
            k = req.k
            if not (0 <= R2 <= 1):
                raise HTTPException(status_code=422, detail="R² must be between 0 and 1")
            if n <= k + 1:
                raise HTTPException(status_code=422, detail="n must be greater than k + 1")

            f_sq = R2 / (1 - R2) if R2 < 1 else float('inf')
            adj_r2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)

            # SE of R² (approximate, Olkin & Finn 1995)
            se_r2 = math.sqrt(4 * R2 * (1 - R2)**2 * (n - k - 1)**2
                              / ((n**2 - 1) * (n + 3))) if n > 3 else None

            return {
                "f_squared": {"value": f_sq,
                              "formula": r"f^2 = \frac{R^2}{1 - R^2}"},
                "adjusted_r_squared": {"value": adj_r2,
                                       "formula": r"R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}"},
                "r_squared": {"value": R2, "se": se_r2},
                "cohens_f": {"value": math.sqrt(f_sq) if f_sq != float('inf') else float('inf'),
                             "formula": r"f = \sqrt{f^2}"},
            }

        else:
            raise HTTPException(status_code=422,
                                detail=f"Unknown stat_type: '{req.stat_type}'. Supported: means, t, F, "
                                       f"chi_square, r, correlation, odds_ratio, proportions, r_squared")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation error: {str(e)}")


# ── Endpoint 2: /convert ─────────────────────────────────────────────

@router.post("/convert")
async def convert(req: ConvertRequest):
    """Convert between any two effect size types."""
    try:
        result = _convert_direct(req.from_type, req.from_value, req.to_type,
                                 req.n1, req.n2, req.df, req.N)
        if result is None:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot convert from '{req.from_type}' to '{req.to_type}'. "
                       f"Supported types: d, g, r, r_squared, fisher_z, OR, log_OR, "
                       f"eta_sq, omega_sq, f, f_sq, w, h, CLES")

        # Compute SE and CI when possible
        se = None
        ci = None
        total_n = None
        if req.n1 is not None and req.n2 is not None:
            total_n = req.n1 + req.n2
        elif req.N is not None:
            total_n = req.N

        tt = req.to_type.lower()
        val = result["value"]

        if tt == "d" and req.n1 and req.n2:
            se = _se_d(val, req.n1, req.n2)
            ci = _ci_d_nct(val, req.n1, req.n2)
        elif tt == "g" and req.n1 and req.n2:
            j = _hedges_correction(req.n1 + req.n2 - 2)
            se = _se_d(val / j, req.n1, req.n2) * j
        elif tt == "r" and total_n and total_n > 3:
            ci = _ci_r_fisher(val, total_n)
            se = ci["se"]
        elif tt == "or" and req.n1 and req.n2:
            d_approx = _or_to_d(val) if val > 0 else 0
            se_d_approx = _se_d(d_approx, req.n1, req.n2)
            se_log_or = se_d_approx * math.pi / math.sqrt(3)
            se = se_log_or * val  # delta method
            z_crit = sp_stats.norm.ppf(0.975)
            log_v = math.log(val)
            ci = {"lower": math.exp(log_v - z_crit * se_log_or),
                  "upper": math.exp(log_v + z_crit * se_log_or),
                  "method": "delta method"}

        if se is not None:
            result["se"] = se
        if ci is not None:
            result["ci"] = ci

        result["from_type"] = req.from_type
        result["from_value"] = req.from_value
        result["to_type"] = req.to_type

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")


# ── Endpoint 3: /interpret ───────────────────────────────────────────

@router.post("/interpret")
async def interpret(req: InterpretRequest):
    """Interpret effect size magnitude with benchmarks."""
    try:
        es_type = req.effect_size_type.lower()
        val = abs(req.value)  # magnitude interpretation uses absolute value
        raw_val = req.value

        # ── Cohen's benchmarks ──
        cohen_key = es_type
        # Normalize key
        if cohen_key in ("hedges_g", "g"):
            cohen_key = "g"
        elif cohen_key in ("cohens_d", "d"):
            cohen_key = "d"
        elif cohen_key in ("or", "odds_ratio"):
            cohen_key = "OR"
            # For OR, benchmarks are on raw scale (>1)
            val = max(raw_val, 1 / raw_val) if raw_val > 0 else raw_val
        elif cohen_key == "cles":
            cohen_key = "CLES"
            val = raw_val  # CLES is always positive, interpreted as-is

        cohen = COHEN_BENCHMARKS.get(cohen_key, {})
        sawilowsky = SAWILOWSKY_BENCHMARKS.get(cohen_key, {})

        # Determine magnitude label (Sawilowsky if available, else Cohen)
        magnitude = "negligible"
        if sawilowsky:
            thresholds = sorted(sawilowsky.items(), key=lambda x: x[1])
            for label, thresh in thresholds:
                if val >= thresh:
                    magnitude = label
        elif cohen:
            if val >= cohen.get("large", float('inf')):
                magnitude = "large"
            elif val >= cohen.get("medium", float('inf')):
                magnitude = "medium"
            elif val >= cohen.get("small", float('inf')):
                magnitude = "small"

        response = {
            "effect_size_type": req.effect_size_type,
            "value": req.value,
            "absolute_value": val,
            "magnitude": magnitude,
            "cohen_benchmarks": cohen if cohen else None,
        }

        if sawilowsky:
            response["sawilowsky_benchmarks"] = sawilowsky

        # Percentile relative to field norms
        if req.context and req.context in FIELD_NORMS:
            field_data = FIELD_NORMS[req.context]
            norm_key = cohen_key
            if norm_key in field_data:
                field_info = field_data[norm_key]
                mean_es = field_info["mean"]
                # Approximate percentile assuming effect sizes are roughly
                # half-normal distributed around 0 with SD ≈ mean/0.8
                sd_approx = mean_es / 0.8 if mean_es > 0 else 1
                percentile = float(sp_stats.norm.cdf(val, loc=0, scale=sd_approx) * 100)
                response["field_norms"] = {
                    "field": req.context,
                    "mean_effect": mean_es,
                    "median_effect": field_info.get("median"),
                    "source": field_info.get("source"),
                    "percentile_in_field": round(percentile, 1),
                    "comparison": (
                        f"Your effect ({val:.3f}) is "
                        + ("above" if val > mean_es else "below" if val < mean_es else "at")
                        + f" the average effect in {req.context.replace('_', ' ')} "
                        + f"(M = {mean_es:.3f})"
                    ),
                }

        # Practical significance note
        practical_notes = []
        if cohen_key in ("d", "g"):
            u3 = _d_to_u3(val)
            cles = _d_to_cles(val)
            ovl = _d_to_overlap(val)
            practical_notes.append(
                f"U3: {u3:.1%} of treatment group exceeds control median")
            practical_notes.append(
                f"CLES: {cles:.1%} probability of superiority")
            practical_notes.append(
                f"Overlap: {ovl:.1%} of distributions overlap")
        elif cohen_key == "r":
            r_sq = val**2
            practical_notes.append(
                f"r² = {r_sq:.4f} — {r_sq:.1%} of variance explained")
        elif cohen_key == "eta_sq" or cohen_key == "partial_eta_sq":
            practical_notes.append(f"{val:.1%} of variance explained")
        elif cohen_key == "OR":
            practical_notes.append(
                f"The odds of the outcome are {val:.2f}x "
                + ("higher" if raw_val > 1 else "lower")
                + " in the treatment group")

        response["practical_significance"] = practical_notes

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretation error: {str(e)}")


# ── Endpoint 4: /confidence-interval ─────────────────────────────────

@router.post("/confidence-interval")
async def confidence_interval(req: ConfidenceIntervalRequest):
    """Compute CI for any effect size using appropriate method."""
    try:
        es = req.effect_size_type.lower()
        val = req.value
        alpha = req.alpha
        n1 = req.n1
        n2 = req.n2
        n = req.n

        if n is None and n1 is not None and n2 is not None:
            n = n1 + n2
        if n1 is None and n is not None:
            n1 = n // 2
            n2 = n - n1

        if n is None or n < 4:
            raise HTTPException(status_code=422,
                                detail="Sample size (n or n1+n2) required and must be >= 4")

        z_crit = sp_stats.norm.ppf(1 - alpha / 2)

        # ── d or g ──
        if es in ("d", "cohens_d", "g", "hedges_g"):
            ci = _ci_d_nct(val, n1, n2, alpha)
            ci["effect_size_type"] = es
            ci["value"] = val
            ci["formula"] = r"\text{Non-central } t: \; \lambda = d\sqrt{\frac{n_1 n_2}{n_1+n_2}}"
            return ci

        # ── r ──
        elif es in ("r", "correlation"):
            ci = _ci_r_fisher(val, n, alpha)
            ci["effect_size_type"] = es
            ci["value"] = val
            ci["formula"] = r"z_r \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{n-3}}"
            return ci

        # ── eta-squared ──
        elif es in ("eta_sq", "eta_squared", "partial_eta_sq"):
            # Use non-central F approach
            # Need to recover F and dfs; approximate from eta and N
            # eta² = F*df1 / (F*df1 + df2), assume df1=1 for simplicity
            df1 = 1
            df2 = n - 2
            if val >= 1 or val <= 0:
                raise HTTPException(status_code=422,
                                    detail="eta² must be strictly between 0 and 1 for CI")
            F_obs = val * df2 / (df1 * (1 - val))

            from scipy.optimize import brentq

            def _ncf_target(ncp, p_target):
                return sp_stats.ncf.cdf(F_obs, df1, df2, ncp) - p_target

            try:
                ncp_lo = brentq(_ncf_target, 0, max(F_obs * df1 * 5, 50),
                                args=(1 - alpha / 2,))
            except ValueError:
                ncp_lo = 0
            try:
                ncp_hi = brentq(_ncf_target, 0, max(F_obs * df1 * 10, 100),
                                args=(alpha / 2,))
            except ValueError:
                ncp_hi = F_obs * df1 * 3

            eta_lower = ncp_lo / (ncp_lo + n)
            eta_upper = ncp_hi / (ncp_hi + n)

            return {
                "effect_size_type": es,
                "value": val,
                "lower": max(0, eta_lower),
                "upper": min(1, eta_upper),
                "method": "non-central F",
                "formula": r"\text{Invert non-central } F \text{ distribution for } \lambda",
            }

        # ── OR ──
        elif es in ("or", "odds_ratio"):
            if val <= 0:
                raise HTTPException(status_code=422, detail="OR must be positive")
            log_or = math.log(val)
            # SE via delta method (approximate)
            se_log = math.sqrt(1 / max(1, n1) + 1 / max(1, n2)
                               + 1 / max(1, n1) + 1 / max(1, n2)) / math.sqrt(2)
            # Better: if we had cell counts. Use approximate based on d.
            d_equiv = _or_to_d(val)
            se_d = _se_d(d_equiv, n1, n2)
            se_log = se_d * math.pi / math.sqrt(3)
            lower = math.exp(log_or - z_crit * se_log)
            upper = math.exp(log_or + z_crit * se_log)
            return {
                "effect_size_type": es,
                "value": val,
                "lower": lower,
                "upper": upper,
                "se": se_log,
                "method": "delta method (via d conversion)",
                "formula": r"\ln(OR) \pm z_{\alpha/2} \cdot SE_{\ln(OR)}",
            }

        # ── log OR ──
        elif es in ("log_or", "log_odds_ratio"):
            d_equiv = val * math.sqrt(3) / math.pi
            se_d = _se_d(d_equiv, n1, n2)
            se_log = se_d * math.pi / math.sqrt(3)
            return {
                "effect_size_type": es,
                "value": val,
                "lower": val - z_crit * se_log,
                "upper": val + z_crit * se_log,
                "se": se_log,
                "method": "delta method",
                "formula": r"\ln(OR) \pm z_{\alpha/2} \cdot SE",
            }

        # ── Cohen's f ──
        elif es == "f":
            eta = _f_to_eta_sq(val)
            eta_ci = await confidence_interval(ConfidenceIntervalRequest(
                effect_size_type="eta_sq", value=eta, n=n, n1=n1, n2=n2, alpha=alpha
            ))
            f_lower = _eta_sq_to_f(max(0, eta_ci["lower"]))
            f_upper = _eta_sq_to_f(min(0.999, eta_ci["upper"]))
            return {
                "effect_size_type": es,
                "value": val,
                "lower": f_lower,
                "upper": f_upper,
                "method": "via eta² non-central F",
                "formula": r"f = \sqrt{\eta^2 / (1-\eta^2)}",
            }

        # ── Cohen's w ──
        elif es == "w":
            se_w = math.sqrt((1 + val**2) / n)
            return {
                "effect_size_type": es,
                "value": val,
                "lower": max(0, val - z_crit * se_w),
                "upper": val + z_crit * se_w,
                "se": se_w,
                "method": "normal approximation",
                "formula": r"w \pm z_{\alpha/2}\sqrt{\frac{1+w^2}{N}}",
            }

        # ── Cohen's h ──
        elif es == "h":
            se_h = math.sqrt(1 / n1 + 1 / n2) if n1 and n2 else math.sqrt(4 / n)
            return {
                "effect_size_type": es,
                "value": val,
                "lower": val - z_crit * se_h,
                "upper": val + z_crit * se_h,
                "se": se_h,
                "method": "normal approximation",
                "formula": r"h \pm z_{\alpha/2}\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}",
            }

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported effect size type '{es}' for CI. "
                       f"Supported: d, g, r, eta_sq, OR, log_OR, f, w, h")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CI computation error: {str(e)}")


# ── Endpoint 5: /nnt ─────────────────────────────────────────────────

@router.post("/nnt")
async def nnt(req: NNTRequest):
    """Number needed to treat and related clinical metrics from effect size."""
    try:
        es = req.effect_size_type.lower()
        val = req.value
        base_rate = req.base_rate

        if not (0 < base_rate < 1):
            raise HTTPException(status_code=422,
                                detail="base_rate must be between 0 and 1 (exclusive)")

        # Convert everything to d first
        if es in ("d", "cohens_d"):
            d = val
        elif es in ("r", "correlation"):
            d = _r_to_d(val)
        elif es in ("or", "odds_ratio"):
            if val <= 0:
                raise HTTPException(status_code=422, detail="OR must be positive")
            d = _or_to_d(val)
        else:
            raise HTTPException(status_code=422,
                                detail=f"Unsupported type '{es}'. Use d, r, or OR.")

        # ── NNT via Kraemer & Kupfer (2006) ──
        # Treatment event rate: Phi(Phi_inv(CER) + d)
        cer = base_rate
        z_cer = float(sp_stats.norm.ppf(cer))
        ter = float(sp_stats.norm.cdf(z_cer + d))

        ard = ter - cer  # absolute risk difference
        nnt_val = 1 / abs(ard) if ard != 0 else float('inf')

        # Also compute NNT from OR directly
        if es in ("or", "odds_ratio"):
            # Exact NNT from OR: NNT = 1 / (CER*(OR-1)/(1 + CER*(OR-1)) - CER + CER)
            # Simplified: TER = OR*CER / (1 - CER + OR*CER)
            OR = val
            ter_or = OR * cer / (1 - cer + OR * cer)
            ard_or = ter_or - cer
            nnt_from_or = 1 / abs(ard_or) if ard_or != 0 else float('inf')
        else:
            nnt_from_or = None

        # ── CLES ──
        cles = _d_to_cles(d)

        # ── Probability of superiority ──
        prob_sup = cles  # same as CLES

        # ── U3 ──
        u3 = _d_to_u3(d)

        # ── OVL (overlap coefficient) ──
        ovl = _d_to_overlap(d)

        # Interpretation
        nnt_label = "NNT" if ard > 0 else "NNH"

        result = {
            "d": d,
            "base_rate_control": cer,
            "treatment_event_rate": ter,
            "absolute_risk_difference": ard,
            "nnt": {
                "value": nnt_val,
                "label": nnt_label,
                "method": "Kraemer & Kupfer (2006)",
                "formula": r"NNT = \frac{1}{|\Phi(\Phi^{-1}(CER) + d) - CER|}",
                "interpretation": (
                    f"On average, {nnt_val:.1f} patients must be treated for "
                    f"one additional {'success' if ard > 0 else 'adverse event'}"
                ) if nnt_val != float('inf') else "No difference in event rates",
            },
            "cles": {
                "value": cles,
                "formula": r"CLES = \Phi\!\left(\frac{d}{\sqrt{2}}\right)",
                "interpretation": f"{cles:.1%} probability that a randomly selected "
                                  f"treated individual scores better than a control",
            },
            "probability_of_superiority": {
                "value": prob_sup,
                "interpretation": f"P(X_T > X_C) = {prob_sup:.4f}",
            },
            "u3": {
                "value": u3,
                "formula": r"U_3 = \Phi(d)",
                "interpretation": f"{u3:.1%} of the treatment group exceeds "
                                  f"the control group median",
            },
            "overlap_coefficient": {
                "value": ovl,
                "formula": r"OVL = 2\Phi(-|d|/2)",
                "interpretation": f"{ovl:.1%} overlap between the two distributions",
            },
        }

        if nnt_from_or is not None:
            result["nnt_from_or_exact"] = {
                "value": nnt_from_or,
                "method": "Exact logistic (Bender & Blettner, 2002)",
                "formula": r"NNT = \frac{1}{\frac{OR \cdot CER}{1 - CER + OR \cdot CER} - CER}",
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NNT computation error: {str(e)}")


# ── Endpoint 6: /power-from-effect ───────────────────────────────────

@router.post("/power-from-effect")
async def power_from_effect(req: PowerFromEffectRequest):
    """Quick power lookup from effect size."""
    try:
        es = req.effect_size_type.lower()
        val = req.value
        alpha = req.alpha
        test = req.test_type.lower()

        n = req.n
        n1 = req.n1
        n2 = req.n2
        if n is None and n1 is not None and n2 is not None:
            n = n1 + n2
        if n1 is None and n is not None:
            n1 = n // 2
            n2 = n - n1

        # Convert to the native effect size for the test
        d = None
        f_val = None
        r_val = None
        w_val = None

        if es in ("d", "cohens_d", "g", "hedges_g"):
            d = val
            f_val = abs(d) / 2
            r_val = _d_to_r(d)
        elif es in ("r", "correlation"):
            r_val = val
            d = _r_to_d(val)
            f_val = abs(d) / 2
        elif es in ("f", "cohens_f"):
            f_val = val
            d = _f_to_d(val)
            r_val = _d_to_r(d)
        elif es in ("f_sq", "f_squared"):
            f_val = math.sqrt(max(0, val))
            d = _f_to_d(f_val)
        elif es in ("w", "cohens_w"):
            w_val = val
            f_val = val  # w ≈ f
        elif es in ("eta_sq", "eta_squared"):
            f_val = _eta_sq_to_f(val)
            d = _f_to_d(f_val)
        else:
            raise HTTPException(status_code=422,
                                detail=f"Unsupported effect size type '{es}'. "
                                       f"Use d, g, r, f, f_sq, w, or eta_sq.")

        power = None
        n_for_80 = None
        n_for_95 = None
        method_desc = ""

        # ── Two-sample t-test ──
        if test in ("two_sample_t", "independent_t", "t_test"):
            if n1 is None or n2 is None:
                raise HTTPException(status_code=422,
                                    detail="n1 and n2 (or n) required for two-sample t-test")
            df_val = n1 + n2 - 2
            ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df_val)
            power = 1 - sp_stats.nct.cdf(t_crit, df_val, ncp) + sp_stats.nct.cdf(-t_crit, df_val, ncp)
            method_desc = "Non-central t distribution"

            # Find n for 80% and 95% power (equal groups)
            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(4, 100000):
                    n_half = n_try
                    df_try = 2 * n_half - 2
                    ncp_try = d * math.sqrt(n_half / 2)
                    t_crit_try = sp_stats.t.ppf(1 - alpha / 2, df_try)
                    pwr = 1 - sp_stats.nct.cdf(t_crit_try, df_try, ncp_try) + \
                          sp_stats.nct.cdf(-t_crit_try, df_try, ncp_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_half * 2
                        else:
                            n_for_95 = n_half * 2
                        break

        # ── One-sample t-test ──
        elif test in ("one_sample_t",):
            if n is None:
                raise HTTPException(status_code=422, detail="n required for one-sample t-test")
            df_val = n - 1
            ncp = d * math.sqrt(n)
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df_val)
            power = 1 - sp_stats.nct.cdf(t_crit, df_val, ncp) + sp_stats.nct.cdf(-t_crit, df_val, ncp)
            method_desc = "Non-central t distribution (one-sample)"

            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(4, 100000):
                    df_try = n_try - 1
                    ncp_try = d * math.sqrt(n_try)
                    t_crit_try = sp_stats.t.ppf(1 - alpha / 2, df_try)
                    pwr = 1 - sp_stats.nct.cdf(t_crit_try, df_try, ncp_try) + \
                          sp_stats.nct.cdf(-t_crit_try, df_try, ncp_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_try
                        else:
                            n_for_95 = n_try
                        break

        # ── Paired t-test ──
        elif test in ("paired_t", "paired"):
            if n is None:
                raise HTTPException(status_code=422, detail="n required for paired t-test")
            df_val = n - 1
            ncp = d * math.sqrt(n)
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df_val)
            power = 1 - sp_stats.nct.cdf(t_crit, df_val, ncp) + sp_stats.nct.cdf(-t_crit, df_val, ncp)
            method_desc = "Non-central t distribution (paired)"

            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(4, 100000):
                    df_try = n_try - 1
                    ncp_try = d * math.sqrt(n_try)
                    t_crit_try = sp_stats.t.ppf(1 - alpha / 2, df_try)
                    pwr = 1 - sp_stats.nct.cdf(t_crit_try, df_try, ncp_try) + \
                          sp_stats.nct.cdf(-t_crit_try, df_try, ncp_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_try
                        else:
                            n_for_95 = n_try
                        break

        # ── Correlation ──
        elif test in ("correlation", "pearson", "r"):
            if n is None:
                raise HTTPException(status_code=422, detail="n required for correlation test")
            if r_val is None:
                r_val = _d_to_r(d)
            z_r = _fisher_z(r_val)
            se_z = 1 / math.sqrt(n - 3)
            z_crit = sp_stats.norm.ppf(1 - alpha / 2)
            power = sp_stats.norm.sf(z_crit - z_r / se_z) + sp_stats.norm.cdf(-z_crit - z_r / se_z)
            method_desc = "Fisher z-transform"

            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(5, 100000):
                    se_try = 1 / math.sqrt(n_try - 3)
                    pwr = sp_stats.norm.sf(z_crit - z_r / se_try) + \
                          sp_stats.norm.cdf(-z_crit - z_r / se_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_try
                        else:
                            n_for_95 = n_try
                        break

        # ── ANOVA (one-way) ──
        elif test in ("anova", "one_way_anova", "f_test"):
            if n is None:
                raise HTTPException(status_code=422, detail="n (total) required for ANOVA")
            if f_val is None:
                raise HTTPException(status_code=422, detail="Cannot compute f for ANOVA power")
            # Assume k=2 groups unless df provided
            k = 2
            df1 = k - 1
            df2 = n - k
            ncp = f_val**2 * n
            f_crit = sp_stats.f.ppf(1 - alpha, df1, df2)
            power = 1 - sp_stats.ncf.cdf(f_crit, df1, df2, ncp)
            method_desc = "Non-central F distribution"

            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(2 * k + 1, 100000):
                    df2_try = n_try - k
                    ncp_try = f_val**2 * n_try
                    f_crit_try = sp_stats.f.ppf(1 - alpha, df1, df2_try)
                    pwr = 1 - sp_stats.ncf.cdf(f_crit_try, df1, df2_try, ncp_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_try
                        else:
                            n_for_95 = n_try
                        break

        # ── Chi-square ──
        elif test in ("chi_square", "chi2", "chi_squared"):
            if n is None:
                raise HTTPException(status_code=422, detail="n required for chi-square power")
            w = w_val if w_val is not None else (f_val or abs(d) / 2)
            df_chi = 1  # default; could parameterize
            ncp = w**2 * n
            chi_crit = sp_stats.chi2.ppf(1 - alpha, df_chi)
            power = 1 - sp_stats.ncx2.cdf(chi_crit, df_chi, ncp)
            method_desc = "Non-central chi-square distribution"

            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(5, 100000):
                    ncp_try = w**2 * n_try
                    pwr = 1 - sp_stats.ncx2.cdf(chi_crit, df_chi, ncp_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_try
                        else:
                            n_for_95 = n_try
                        break

        # ── Regression ──
        elif test in ("regression", "multiple_regression", "linear_regression"):
            if n is None:
                raise HTTPException(status_code=422, detail="n required for regression power")
            f_sq = f_val**2 if f_val else val
            k_pred = 1  # default predictors
            df1 = k_pred
            df2 = n - k_pred - 1
            ncp = f_sq * n
            f_crit = sp_stats.f.ppf(1 - alpha, df1, df2)
            power = 1 - sp_stats.ncf.cdf(f_crit, df1, df2, ncp)
            method_desc = "Non-central F distribution (regression)"

            for target_power, label in [(0.80, "n_for_80"), (0.95, "n_for_95")]:
                for n_try in range(k_pred + 3, 100000):
                    df2_try = n_try - k_pred - 1
                    ncp_try = f_sq * n_try
                    f_crit_try = sp_stats.f.ppf(1 - alpha, df1, df2_try)
                    pwr = 1 - sp_stats.ncf.cdf(f_crit_try, df1, df2_try, ncp_try)
                    if pwr >= target_power:
                        if label == "n_for_80":
                            n_for_80 = n_try
                        else:
                            n_for_95 = n_try
                        break

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown test_type '{req.test_type}'. Supported: two_sample_t, "
                       f"one_sample_t, paired_t, correlation, anova, chi_square, regression")

        return {
            "effect_size_type": req.effect_size_type,
            "effect_size_value": val,
            "test_type": req.test_type,
            "n": n,
            "n1": n1,
            "n2": n2,
            "alpha": alpha,
            "power": power,
            "required_n_for_80_power": n_for_80,
            "required_n_for_95_power": n_for_95,
            "method": method_desc,
            "notes": (
                f"Power = {power:.4f} ({power:.1%}). "
                + (f"Need total N = {n_for_80} for 80% power, "
                   f"N = {n_for_95} for 95% power."
                   if n_for_80 and n_for_95 else "")
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Power computation error: {str(e)}")
