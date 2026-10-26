"""Signal Detection Theory — d-prime, criterion, ROC analysis, and
multi-condition comparisons for perception, memory, and clinical research.

Implements SDT metrics (d', c, beta, A', B''D) with Hautus (1995) log-linear
correction, ROC curve construction from confidence ratings, z-ROC analysis
for unequal variance SDT, and multi-condition comparison tables.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class SDTAnalyzeRequest(BaseModel):
    """Basic SDT analysis from frequency counts."""
    hits: int
    misses: int
    false_alarms: int
    correct_rejections: int


class ROCRequest(BaseModel):
    """ROC analysis from confidence ratings."""
    signal_ratings: list[int]  # confidence ratings for signal-present trials
    noise_ratings: list[int]  # confidence ratings for signal-absent trials
    n_levels: int | None = None  # number of confidence levels (auto-detected if None)


class MultiConditionRequest(BaseModel):
    """SDT analysis across multiple conditions."""
    conditions: list[dict]  # each: {"name": str, "hits": int, "misses": int, "false_alarms": int, "correct_rejections": int}


# ── Helper Functions ──────────────────────────────────────────────────

def _validate_counts(hits: int, misses: int, fa: int, cr: int):
    """Validate that all counts are non-negative and sufficient."""
    if any(c < 0 for c in [hits, misses, fa, cr]):
        raise HTTPException(
            status_code=422,
            detail="All frequency counts must be non-negative.",
        )
    n_signal = hits + misses
    n_noise = fa + cr
    if n_signal == 0:
        raise HTTPException(status_code=422, detail="No signal trials (hits + misses = 0).")
    if n_noise == 0:
        raise HTTPException(status_code=422, detail="No noise trials (false_alarms + correct_rejections = 0).")


def _log_linear_correction(hits: int, misses: int, fa: int, cr: int) -> tuple[float, float]:
    """Apply Hautus (1995) log-linear correction for extreme hit/FA rates.

    Adds 0.5 to each cell count and 1 to each marginal total.
    This avoids infinite z-scores when hit rate = 1.0 or FA rate = 0.0.

    Returns (corrected_hit_rate, corrected_fa_rate).
    """
    n_signal = hits + misses
    n_noise = fa + cr
    hit_rate = (hits + 0.5) / (n_signal + 1)
    fa_rate = (fa + 0.5) / (n_noise + 1)
    return hit_rate, fa_rate


def _compute_sdt_metrics(hits: int, misses: int, fa: int, cr: int) -> dict:
    """Compute all SDT metrics from frequency counts.

    Returns d-prime, criterion c, beta, A-prime, and B''D.
    """
    n_signal = hits + misses
    n_noise = fa + cr

    # Raw rates
    raw_hr = hits / n_signal
    raw_far = fa / n_noise

    # Log-linear corrected rates (Hautus 1995)
    hr, far = _log_linear_correction(hits, misses, fa, cr)

    # z-scores
    z_hr = float(sp_stats.norm.ppf(hr))
    z_far = float(sp_stats.norm.ppf(far))

    # d-prime (sensitivity)
    d_prime = z_hr - z_far

    # Criterion c (response bias — negative = liberal, positive = conservative)
    c = -0.5 * (z_hr + z_far)

    # Beta (likelihood ratio criterion)
    beta = np.exp(-c * d_prime) if abs(d_prime) < 20 else float("inf")

    # A-prime (non-parametric sensitivity)
    if raw_hr >= raw_far:
        a_prime = 0.5 + ((raw_hr - raw_far) * (1 + raw_hr - raw_far)) / \
                  (4 * raw_hr * (1 - raw_far)) if (raw_hr * (1 - raw_far)) > 0 else 0.5
    else:
        a_prime = 0.5 - ((raw_far - raw_hr) * (1 + raw_far - raw_hr)) / \
                  (4 * raw_far * (1 - raw_hr)) if (raw_far * (1 - raw_hr)) > 0 else 0.5

    # B''D (non-parametric bias — Donaldson 1992)
    if raw_hr == raw_far:
        b_prime_d = 0.0
    else:
        b_prime_d = ((1 - raw_hr) * (1 - raw_far) - raw_hr * raw_far) / \
                    ((1 - raw_hr) * (1 - raw_far) + raw_hr * raw_far) \
                    if ((1 - raw_hr) * (1 - raw_far) + raw_hr * raw_far) > 0 else 0.0

    # Interpretation
    if abs(d_prime) < 0.5:
        sensitivity_interp = "very low sensitivity (near chance)"
    elif abs(d_prime) < 1.0:
        sensitivity_interp = "low sensitivity"
    elif abs(d_prime) < 2.0:
        sensitivity_interp = "moderate sensitivity"
    elif abs(d_prime) < 3.0:
        sensitivity_interp = "high sensitivity"
    else:
        sensitivity_interp = "very high sensitivity"

    if c < -0.5:
        bias_interp = "strongly liberal (tendency to say 'yes')"
    elif c < -0.1:
        bias_interp = "liberal (slight tendency to say 'yes')"
    elif c <= 0.1:
        bias_interp = "unbiased (no systematic tendency)"
    elif c <= 0.5:
        bias_interp = "conservative (slight tendency to say 'no')"
    else:
        bias_interp = "strongly conservative (tendency to say 'no')"

    return {
        "hit_rate": round(raw_hr, 4),
        "false_alarm_rate": round(raw_far, 4),
        "corrected_hit_rate": round(hr, 4),
        "corrected_false_alarm_rate": round(far, 4),
        "z_hit_rate": round(z_hr, 4),
        "z_false_alarm_rate": round(z_far, 4),
        "d_prime": round(d_prime, 4),
        "criterion_c": round(c, 4),
        "beta": round(float(beta), 4),
        "a_prime": round(a_prime, 4),
        "b_double_prime_d": round(b_prime_d, 4),
        "n_signal_trials": n_signal,
        "n_noise_trials": n_noise,
        "sensitivity_interpretation": sensitivity_interp,
        "bias_interpretation": bias_interp,
    }


# ── 1. Basic SDT Analysis ────────────────────────────────────────────

@router.post("/analyze")
def sdt_analyze(req: SDTAnalyzeRequest):
    """Compute signal detection theory metrics from frequency counts.

    Takes hits, misses, false alarms, and correct rejections. Applies
    Hautus (1995) log-linear correction for extreme rates. Returns
    d-prime, criterion c, beta, A-prime, and B''D with interpretations.
    """
    _validate_counts(req.hits, req.misses, req.false_alarms, req.correct_rejections)

    metrics = _compute_sdt_metrics(req.hits, req.misses, req.false_alarms, req.correct_rejections)

    n_total = req.hits + req.misses + req.false_alarms + req.correct_rejections
    accuracy = (req.hits + req.correct_rejections) / n_total

    return {
        "counts": {
            "hits": req.hits,
            "misses": req.misses,
            "false_alarms": req.false_alarms,
            "correct_rejections": req.correct_rejections,
            "total_trials": n_total,
            "overall_accuracy": round(accuracy, 4),
        },
        **metrics,
        "interpretation": (
            f"d' = {metrics['d_prime']:.3f} ({metrics['sensitivity_interpretation']}). "
            f"Criterion c = {metrics['criterion_c']:.3f} ({metrics['bias_interpretation']}). "
            f"A' = {metrics['a_prime']:.3f} (non-parametric sensitivity, 0.5 = chance, 1.0 = perfect). "
            f"Log-linear correction applied (Hautus, 1995) to avoid infinite z-scores."
        ),
    }


# ── 2. ROC Analysis ──────────────────────────────────────────────────

@router.post("/roc")
def roc_analysis(req: ROCRequest):
    """Construct ROC curve from confidence ratings.

    Takes confidence ratings for signal-present and signal-absent trials,
    constructs empirical ROC points, computes Az (area under ROC), and
    fits z-ROC for unequal-variance SDT assessment.
    """
    if not req.signal_ratings:
        raise HTTPException(status_code=422, detail="signal_ratings must not be empty.")
    if not req.noise_ratings:
        raise HTTPException(status_code=422, detail="noise_ratings must not be empty.")

    signal = np.array(req.signal_ratings)
    noise = np.array(req.noise_ratings)

    # Determine number of confidence levels
    all_ratings = np.concatenate([signal, noise])
    min_rating = int(np.min(all_ratings))
    max_rating = int(np.max(all_ratings))
    n_levels = req.n_levels or (max_rating - min_rating + 1)

    if n_levels < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 confidence levels for ROC analysis.")

    n_signal = len(signal)
    n_noise = len(noise)

    # Compute ROC points at each criterion threshold
    roc_points = [{"false_alarm_rate": 0.0, "hit_rate": 0.0, "criterion": max_rating + 1}]

    for criterion in range(max_rating, min_rating - 1, -1):
        hr = float(np.mean(signal >= criterion))
        far = float(np.mean(noise >= criterion))
        roc_points.append({
            "criterion": criterion,
            "hit_rate": round(hr, 4),
            "false_alarm_rate": round(far, 4),
        })

    # Ensure we end at (1, 1)
    roc_points.append({"false_alarm_rate": 1.0, "hit_rate": 1.0, "criterion": min_rating - 1})

    # Remove duplicates, sort by FAR
    seen = set()
    unique_points = []
    for pt in roc_points:
        key = (pt["false_alarm_rate"], pt["hit_rate"])
        if key not in seen:
            seen.add(key)
            unique_points.append(pt)
    roc_points = sorted(unique_points, key=lambda x: x["false_alarm_rate"])

    # Az (area under ROC) via trapezoidal rule
    fars = [p["false_alarm_rate"] for p in roc_points]
    hrs = [p["hit_rate"] for p in roc_points]
    az = float(np.trapz(hrs, fars))

    # z-ROC analysis (for unequal-variance SDT)
    # Use interior points (exclude 0,0 and 1,1)
    interior_points = [p for p in roc_points
                       if 0 < p["false_alarm_rate"] < 1 and 0 < p["hit_rate"] < 1]

    z_roc_data = []
    z_roc_fit = None
    da = None

    if len(interior_points) >= 2:
        z_fars = []
        z_hrs = []
        for pt in interior_points:
            z_far = float(sp_stats.norm.ppf(pt["false_alarm_rate"]))
            z_hr = float(sp_stats.norm.ppf(pt["hit_rate"]))
            z_fars.append(z_far)
            z_hrs.append(z_hr)
            z_roc_data.append({
                "z_false_alarm_rate": round(z_far, 4),
                "z_hit_rate": round(z_hr, 4),
            })

        z_fars_arr = np.array(z_fars)
        z_hrs_arr = np.array(z_hrs)

        # Fit linear z-ROC: z(HR) = intercept + slope * z(FAR)
        if len(z_fars) >= 2:
            # Simple OLS
            n_pts = len(z_fars_arr)
            mean_x = np.mean(z_fars_arr)
            mean_y = np.mean(z_hrs_arr)
            ss_xx = np.sum((z_fars_arr - mean_x) ** 2)

            if ss_xx > 1e-12:
                slope = float(np.sum((z_fars_arr - mean_x) * (z_hrs_arr - mean_y)) / ss_xx)
                intercept = float(mean_y - slope * mean_x)

                # Predicted values and residuals
                predicted = intercept + slope * z_fars_arr
                residuals = z_hrs_arr - predicted
                ss_res = float(np.sum(residuals ** 2))
                ss_tot = float(np.sum((z_hrs_arr - mean_y) ** 2))
                r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

                # da (unequal variance d-prime)
                da = intercept * np.sqrt(2 / (1 + slope ** 2))

                z_roc_fit = {
                    "intercept": round(intercept, 4),
                    "slope": round(slope, 4),
                    "r_squared": round(r_squared, 4),
                    "da": round(float(da), 4),
                    "equal_variance": abs(slope - 1.0) < 0.15,
                    "slope_interpretation": (
                        "Slope near 1.0 supports equal-variance SDT assumption."
                        if abs(slope - 1.0) < 0.15
                        else f"Slope = {slope:.2f} deviates from 1.0, suggesting unequal "
                             f"signal/noise variance (ratio ~{1/slope:.2f})."
                    ),
                }

    return {
        "n_signal_trials": n_signal,
        "n_noise_trials": n_noise,
        "n_confidence_levels": n_levels,
        "roc_points": roc_points,
        "az": round(az, 4),
        "z_roc_data": z_roc_data,
        "z_roc_fit": z_roc_fit,
        "interpretation": (
            f"ROC analysis on {n_signal} signal and {n_noise} noise trials "
            f"with {n_levels} confidence levels. "
            f"Az = {az:.3f} (0.5 = chance, 1.0 = perfect discrimination). "
            + (f"z-ROC slope = {z_roc_fit['slope']:.2f} (da = {z_roc_fit['da']:.3f}). "
               f"{z_roc_fit['slope_interpretation']}"
               if z_roc_fit else
               "Insufficient interior ROC points for z-ROC analysis.")
        ),
    }


# ── 3. Multi-Condition SDT Comparison ─────────────────────────────────

@router.post("/multi-condition")
def multi_condition_sdt(req: MultiConditionRequest):
    """Run SDT analysis across multiple experimental conditions.

    Computes d-prime, c, and other metrics for each condition and provides
    pairwise d-prime differences with standard errors for comparison.
    """
    if len(req.conditions) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 conditions for comparison.")

    # Validate each condition
    for i, cond in enumerate(req.conditions):
        required_keys = {"hits", "misses", "false_alarms", "correct_rejections"}
        if not required_keys.issubset(cond.keys()):
            raise HTTPException(
                status_code=422,
                detail=f"Condition {i} missing required keys: {required_keys - cond.keys()}",
            )
        _validate_counts(cond["hits"], cond["misses"], cond["false_alarms"], cond["correct_rejections"])

    # Compute SDT metrics for each condition
    condition_results = []
    for cond in req.conditions:
        name = cond.get("name", f"Condition {len(condition_results) + 1}")
        metrics = _compute_sdt_metrics(
            cond["hits"], cond["misses"],
            cond["false_alarms"], cond["correct_rejections"],
        )
        metrics["condition"] = name
        condition_results.append(metrics)

    # Pairwise d-prime differences with SE
    pairwise = []
    n_conditions = len(condition_results)
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            ci = condition_results[i]
            cj = condition_results[j]

            d_diff = ci["d_prime"] - cj["d_prime"]

            # SE of d-prime difference (Macmillan & Creelman, 2005)
            # SE(d') approx sqrt(HR(1-HR)/n_signal + FAR(1-FAR)/n_noise) for each
            n_sig_i = ci["n_signal_trials"]
            n_noi_i = ci["n_noise_trials"]
            hr_i = ci["corrected_hit_rate"]
            far_i = ci["corrected_false_alarm_rate"]

            n_sig_j = cj["n_signal_trials"]
            n_noi_j = cj["n_noise_trials"]
            hr_j = cj["corrected_hit_rate"]
            far_j = cj["corrected_false_alarm_rate"]

            # Variance of z-scores via delta method
            pdf_hr_i = sp_stats.norm.pdf(sp_stats.norm.ppf(hr_i))
            pdf_far_i = sp_stats.norm.pdf(sp_stats.norm.ppf(far_i))
            pdf_hr_j = sp_stats.norm.pdf(sp_stats.norm.ppf(hr_j))
            pdf_far_j = sp_stats.norm.pdf(sp_stats.norm.ppf(far_j))

            var_d_i = (hr_i * (1 - hr_i)) / (n_sig_i * pdf_hr_i ** 2 + 1e-12) + \
                      (far_i * (1 - far_i)) / (n_noi_i * pdf_far_i ** 2 + 1e-12)
            var_d_j = (hr_j * (1 - hr_j)) / (n_sig_j * pdf_hr_j ** 2 + 1e-12) + \
                      (far_j * (1 - far_j)) / (n_noi_j * pdf_far_j ** 2 + 1e-12)

            se_diff = float(np.sqrt(var_d_i + var_d_j))
            z_test = d_diff / se_diff if se_diff > 1e-12 else 0.0
            p_value = float(2 * (sp_stats.norm.sf(abs(z_test))))

            pairwise.append({
                "condition_1": ci["condition"],
                "condition_2": cj["condition"],
                "d_prime_1": ci["d_prime"],
                "d_prime_2": cj["d_prime"],
                "difference": round(d_diff, 4),
                "se": round(se_diff, 4),
                "z": round(z_test, 4),
                "p_value": round(p_value, 4),
                "significant_05": p_value < 0.05,
            })

    # Summary table
    summary_table = []
    for cr in condition_results:
        summary_table.append({
            "condition": cr["condition"],
            "d_prime": cr["d_prime"],
            "criterion_c": cr["criterion_c"],
            "beta": cr["beta"],
            "a_prime": cr["a_prime"],
            "hit_rate": cr["hit_rate"],
            "false_alarm_rate": cr["false_alarm_rate"],
        })

    # Overall interpretation
    best_cond = max(condition_results, key=lambda x: x["d_prime"])
    worst_cond = min(condition_results, key=lambda x: x["d_prime"])
    n_sig_pairs = sum(1 for p in pairwise if p["significant_05"])

    return {
        "n_conditions": n_conditions,
        "condition_results": condition_results,
        "summary_table": summary_table,
        "pairwise_comparisons": pairwise,
        "n_significant_pairs": n_sig_pairs,
        "interpretation": (
            f"SDT analysis across {n_conditions} conditions. "
            f"Highest sensitivity: {best_cond['condition']} (d' = {best_cond['d_prime']:.3f}). "
            f"Lowest sensitivity: {worst_cond['condition']} (d' = {worst_cond['d_prime']:.3f}). "
            f"{n_sig_pairs}/{len(pairwise)} pairwise d' differences significant at p < .05."
        ),
    }
