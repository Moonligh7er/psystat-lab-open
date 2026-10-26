"""Experience Sampling Method / Ecological Momentary Assessment Analysis.

Provides descriptive decomposition, lagged analysis (emotional inertia),
affective dynamics (MSSD, PAC), and compliance/missingness diagnostics
for intensive longitudinal data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class EMAObservation(BaseModel):
    """A single EMA observation."""
    participant_id: str
    timestamp: float  # Unix epoch or ordinal
    variables: dict[str, float]


class EMADescriptiveRequest(BaseModel):
    """Descriptive statistics for EMA data."""
    observations: list[EMAObservation]
    variable_names: list[str] | None = None


class EMALaggedRequest(BaseModel):
    """Lagged analysis for EMA data."""
    observations: list[EMAObservation]
    variables: list[str]
    lag: int = 1


class EMADynamicsRequest(BaseModel):
    """Affective dynamics analysis."""
    observations: list[EMAObservation]
    variables: list[str]


class EMAComplianceRequest(BaseModel):
    """Compliance and missingness analysis."""
    observations: list[EMAObservation]
    expected_per_day: int = 5
    total_days: int | None = None
    variables: list[str] | None = None


# ── Helpers ───────────────────────────────────────────────────────────

def _group_by_participant(observations: list[EMAObservation]) -> dict[str, list[EMAObservation]]:
    """Group observations by participant, sorted by timestamp."""
    groups: dict[str, list[EMAObservation]] = {}
    for obs in observations:
        groups.setdefault(obs.participant_id, []).append(obs)
    for pid in groups:
        groups[pid].sort(key=lambda o: o.timestamp)
    return groups


def _extract_variable(obs_list: list[EMAObservation], var: str) -> np.ndarray:
    """Extract a variable's values from a list of observations, skipping missing."""
    vals = [o.variables[var] for o in obs_list if var in o.variables]
    return np.array(vals, dtype=float)


def _compute_icc1(groups: dict[str, np.ndarray]) -> dict:
    """Compute ICC(1) via one-way random effects ANOVA decomposition.

    Returns ICC(1), between-person variance, within-person variance.
    """
    all_values = []
    group_arrays = []
    for pid, vals in groups.items():
        if len(vals) > 0:
            all_values.extend(vals.tolist())
            group_arrays.append(vals)

    if len(group_arrays) < 2:
        return {"icc1": None, "var_between": None, "var_within": None,
                "detail": "Need at least 2 participants"}

    grand_mean = np.mean(all_values)
    k = len(group_arrays)
    n_total = len(all_values)
    ns = [len(g) for g in group_arrays]
    n_mean = n_total / k  # average group size

    # Sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_arrays)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in group_arrays)

    df_between = k - 1
    df_within = n_total - k

    if df_between == 0 or df_within == 0:
        return {"icc1": None, "var_between": None, "var_within": None,
                "detail": "Insufficient degrees of freedom"}

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Variance components
    var_within = ms_within
    var_between = max(0.0, (ms_between - ms_within) / n_mean)

    total_var = var_between + var_within
    icc1 = var_between / total_var if total_var > 0 else 0.0

    return {
        "icc1": round(float(icc1), 4),
        "var_between": round(float(var_between), 4),
        "var_within": round(float(var_within), 4),
        "ms_between": round(float(ms_between), 4),
        "ms_within": round(float(ms_within), 4),
    }


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/descriptive")
def ema_descriptive(req: EMADescriptiveRequest):
    """Compute person-level means, within-person SDs, ICC(1), and
    between/within variance decomposition for each variable.
    """
    if len(req.observations) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 observations")

    groups = _group_by_participant(req.observations)

    # Determine variables to analyze
    all_vars: set[str] = set()
    for obs in req.observations:
        all_vars.update(obs.variables.keys())
    if req.variable_names:
        target_vars = [v for v in req.variable_names if v in all_vars]
    else:
        target_vars = sorted(all_vars)

    if not target_vars:
        raise HTTPException(status_code=422, detail="No valid variables found")

    variable_results = {}
    for var in target_vars:
        # Person-level values
        person_groups: dict[str, np.ndarray] = {}
        person_stats = []
        for pid, obs_list in groups.items():
            vals = _extract_variable(obs_list, var)
            if len(vals) > 0:
                person_groups[pid] = vals
                person_stats.append({
                    "participant_id": pid,
                    "n": int(len(vals)),
                    "mean": round(float(np.mean(vals)), 4),
                    "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                })

        # Grand descriptives
        all_vals = np.concatenate(list(person_groups.values())) if person_groups else np.array([])
        if len(all_vals) == 0:
            variable_results[var] = {"detail": "No data for this variable"}
            continue

        # Within-person SD (average of person-level SDs)
        person_sds = [float(np.std(v, ddof=1)) for v in person_groups.values() if len(v) > 1]
        avg_within_sd = float(np.mean(person_sds)) if person_sds else None

        # ICC(1) and variance decomposition
        icc_result = _compute_icc1(person_groups)

        variable_results[var] = {
            "grand_mean": round(float(np.mean(all_vals)), 4),
            "grand_sd": round(float(np.std(all_vals, ddof=1)), 4) if len(all_vals) > 1 else None,
            "avg_within_person_sd": round(avg_within_sd, 4) if avg_within_sd is not None else None,
            "n_participants": len(person_groups),
            "n_observations": int(len(all_vals)),
            "icc": icc_result,
            "person_level": person_stats,
        }

    return {
        "analysis": "ema_descriptive",
        "n_participants": len(groups),
        "n_observations": len(req.observations),
        "variables": variable_results,
    }


@router.post("/lagged")
def ema_lagged(req: EMALaggedRequest):
    """Compute lag-1 autocorrelations (emotional inertia), cross-lagged
    correlations, and pooled estimates across participants.
    """
    if len(req.observations) < 3:
        raise HTTPException(status_code=422, detail="Need at least 3 observations for lagged analysis")
    if req.lag < 1:
        raise HTTPException(status_code=422, detail="Lag must be >= 1")

    groups = _group_by_participant(req.observations)
    lag = req.lag
    variables = req.variables

    # Validate variables exist
    all_vars: set[str] = set()
    for obs in req.observations:
        all_vars.update(obs.variables.keys())
    for v in variables:
        if v not in all_vars:
            raise HTTPException(status_code=422, detail=f"Variable '{v}' not found in data")

    # Autocorrelations (inertia) per person
    autocorr_results: dict[str, list] = {v: [] for v in variables}
    cross_lagged_results: dict[str, list] = {}

    # Cross-lag pairs
    cross_pairs = [(v1, v2) for i, v1 in enumerate(variables)
                   for v2 in variables[i + 1:]]
    for pair in cross_pairs:
        key = f"{pair[0]}_->_{pair[1]}"
        cross_lagged_results[key] = []
        key_rev = f"{pair[1]}_->_{pair[0]}"
        cross_lagged_results[key_rev] = []

    for pid, obs_list in groups.items():
        if len(obs_list) <= lag:
            continue

        # Extract variable time series for this person
        var_series: dict[str, np.ndarray] = {}
        for v in variables:
            var_series[v] = np.array(
                [o.variables.get(v, np.nan) for o in obs_list], dtype=float
            )

        # Autocorrelations
        for v in variables:
            series = var_series[v]
            valid_mask = ~np.isnan(series[:-lag]) & ~np.isnan(series[lag:])
            x = series[:-lag][valid_mask]
            y = series[lag:][valid_mask]
            if len(x) > 2:
                r, p = sp_stats.pearsonr(x, y)
                autocorr_results[v].append({
                    "participant_id": pid,
                    "r": round(float(r), 4),
                    "p": round(float(p), 4),
                    "n_pairs": int(len(x)),
                })

        # Cross-lagged correlations
        for v1, v2 in cross_pairs:
            s1 = var_series[v1]
            s2 = var_series[v2]

            # v1(t) -> v2(t+lag)
            valid = ~np.isnan(s1[:-lag]) & ~np.isnan(s2[lag:])
            x, y = s1[:-lag][valid], s2[lag:][valid]
            if len(x) > 2:
                r, p = sp_stats.pearsonr(x, y)
                cross_lagged_results[f"{v1}_->_{v2}"].append({
                    "participant_id": pid,
                    "r": round(float(r), 4),
                    "p": round(float(p), 4),
                    "n_pairs": int(len(x)),
                })

            # v2(t) -> v1(t+lag)
            valid = ~np.isnan(s2[:-lag]) & ~np.isnan(s1[lag:])
            x, y = s2[:-lag][valid], s1[lag:][valid]
            if len(x) > 2:
                r, p = sp_stats.pearsonr(x, y)
                cross_lagged_results[f"{v2}_->_{v1}"].append({
                    "participant_id": pid,
                    "r": round(float(r), 4),
                    "p": round(float(p), 4),
                    "n_pairs": int(len(x)),
                })

    # Pooled estimates (Fisher z-transform)
    def _pool_correlations(person_results: list) -> dict:
        if not person_results:
            return {"pooled_r": None, "pooled_p": None, "k": 0}
        rs = [pr["r"] for pr in person_results]
        ns = [pr["n_pairs"] for pr in person_results]
        zs = [np.arctanh(r) if abs(r) < 1.0 else np.sign(r) * 3.0 for r in rs]
        weights = [n - 3 for n in ns]
        total_w = sum(w for w in weights if w > 0)
        if total_w <= 0:
            return {"pooled_r": None, "pooled_p": None, "k": len(rs)}
        z_bar = sum(z * max(w, 0) for z, w in zip(zs, weights)) / total_w
        se_z = 1.0 / np.sqrt(total_w)
        z_test = z_bar / se_z if se_z > 0 else 0.0
        p_val = 2 * (sp_stats.norm.sf(abs(z_test)))
        pooled_r = float(np.tanh(z_bar))
        return {
            "pooled_r": round(pooled_r, 4),
            "pooled_z": round(float(z_bar), 4),
            "se_z": round(float(se_z), 4),
            "pooled_p": round(float(p_val), 4),
            "k": len(rs),
        }

    pooled_auto = {v: _pool_correlations(autocorr_results[v]) for v in variables}
    pooled_cross = {k: _pool_correlations(v) for k, v in cross_lagged_results.items()}

    return {
        "analysis": "ema_lagged",
        "lag": lag,
        "autocorrelations": {
            "person_level": autocorr_results,
            "pooled": pooled_auto,
        },
        "cross_lagged": {
            "person_level": cross_lagged_results,
            "pooled": pooled_cross,
        },
        "interpretation": {
            "autocorrelation": "Higher lag-1 autocorrelation = greater emotional inertia (slower recovery)",
            "cross_lagged": "Cross-lagged r indicates predictive relationship from variable at time t to another at time t+lag",
        },
    }


@router.post("/dynamics")
def ema_dynamics(req: EMADynamicsRequest):
    """Compute affective dynamics: MSSD, probability of acute change (PAC),
    within-person variability, and person-level time trends.
    """
    if len(req.observations) < 3:
        raise HTTPException(status_code=422, detail="Need at least 3 observations")

    groups = _group_by_participant(req.observations)
    variables = req.variables

    # Validate variables
    all_vars: set[str] = set()
    for obs in req.observations:
        all_vars.update(obs.variables.keys())
    for v in variables:
        if v not in all_vars:
            raise HTTPException(status_code=422, detail=f"Variable '{v}' not found")

    results_by_var: dict[str, dict] = {}

    for var in variables:
        person_dynamics = []

        for pid, obs_list in groups.items():
            vals = np.array(
                [o.variables.get(var, np.nan) for o in obs_list], dtype=float
            )
            timestamps = np.array([o.timestamp for o in obs_list], dtype=float)

            # Remove NaN
            valid = ~np.isnan(vals)
            vals = vals[valid]
            timestamps = timestamps[valid]

            if len(vals) < 3:
                continue

            # MSSD — Mean Squared Successive Differences
            diffs = np.diff(vals)
            mssd = float(np.mean(diffs ** 2))

            # PAC — Probability of Acute Change
            # Acute change = successive difference > 1 SD of the person's distribution
            person_sd = float(np.std(vals, ddof=1))
            if person_sd > 0:
                pac = float(np.mean(np.abs(diffs) > person_sd))
            else:
                pac = 0.0

            # Within-person variability metrics
            wp_sd = person_sd
            wp_range = float(np.max(vals) - np.min(vals))
            wp_iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))

            # Time trend (linear slope)
            if len(timestamps) > 2 and np.std(timestamps) > 0:
                slope, intercept, r_val, p_val, se = sp_stats.linregress(timestamps, vals)
                time_trend = {
                    "slope": round(float(slope), 6),
                    "r_squared": round(float(r_val ** 2), 4),
                    "p": round(float(p_val), 4),
                    "direction": "increasing" if slope > 0 else "decreasing",
                }
            else:
                time_trend = {"slope": None, "detail": "Insufficient data for trend"}

            person_dynamics.append({
                "participant_id": pid,
                "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4),
                "sd": round(wp_sd, 4),
                "mssd": round(mssd, 4),
                "rmssd": round(float(np.sqrt(mssd)), 4),
                "pac": round(pac, 4),
                "range": round(wp_range, 4),
                "iqr": round(wp_iqr, 4),
                "time_trend": time_trend,
            })

        # Aggregate across persons
        if person_dynamics:
            mssds = [p["mssd"] for p in person_dynamics]
            pacs = [p["pac"] for p in person_dynamics]
            sds = [p["sd"] for p in person_dynamics]

            aggregate = {
                "mean_mssd": round(float(np.mean(mssds)), 4),
                "sd_mssd": round(float(np.std(mssds, ddof=1)), 4) if len(mssds) > 1 else None,
                "mean_pac": round(float(np.mean(pacs)), 4),
                "mean_within_sd": round(float(np.mean(sds)), 4),
                "n_participants": len(person_dynamics),
            }
        else:
            aggregate = {"detail": "No valid data"}

        results_by_var[var] = {
            "person_level": person_dynamics,
            "aggregate": aggregate,
        }

    return {
        "analysis": "ema_dynamics",
        "variables": results_by_var,
        "interpretation": {
            "mssd": "Mean squared successive differences — captures instability (higher = more volatile)",
            "pac": "Probability of acute change — proportion of successive changes exceeding 1 within-person SD",
            "time_trend": "Linear slope over time — positive = increasing trend, negative = decreasing",
        },
    }


@router.post("/compliance")
def ema_compliance(req: EMAComplianceRequest):
    """Analyze response rates, time-of-day patterns, and systematic
    missingness in EMA data.
    """
    if len(req.observations) < 1:
        raise HTTPException(status_code=422, detail="Need at least 1 observation")

    groups = _group_by_participant(req.observations)

    # Determine time range
    all_timestamps = [o.timestamp for o in req.observations]
    t_min, t_max = min(all_timestamps), max(all_timestamps)
    total_span_days = (t_max - t_min) / 86400.0 if t_max > t_min else 1.0
    total_days = req.total_days if req.total_days else max(1, int(np.ceil(total_span_days)))
    expected_total = req.expected_per_day * total_days

    # Per-participant compliance
    participant_compliance = []
    for pid, obs_list in groups.items():
        n_responses = len(obs_list)
        response_rate = n_responses / expected_total if expected_total > 0 else 0.0

        # Time-of-day distribution (hour bins from timestamps)
        hours = [(o.timestamp % 86400) / 3600.0 for o in obs_list]
        hour_bins = [0] * 24
        for h in hours:
            hour_bins[int(h) % 24] += 1

        # Inter-response intervals
        ts_sorted = sorted(o.timestamp for o in obs_list)
        if len(ts_sorted) > 1:
            intervals = np.diff(ts_sorted)
            mean_interval = float(np.mean(intervals))
            sd_interval = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
            cv_interval = sd_interval / mean_interval if mean_interval > 0 else 0.0
        else:
            mean_interval = None
            sd_interval = None
            cv_interval = None

        participant_compliance.append({
            "participant_id": pid,
            "n_responses": n_responses,
            "response_rate": round(float(response_rate), 4),
            "hour_distribution": hour_bins,
            "mean_interval_sec": round(mean_interval, 2) if mean_interval is not None else None,
            "sd_interval_sec": round(sd_interval, 2) if sd_interval is not None else None,
            "cv_interval": round(cv_interval, 4) if cv_interval is not None else None,
        })

    # Aggregate time-of-day pattern
    agg_hours = [0] * 24
    for pc in participant_compliance:
        for h in range(24):
            agg_hours[h] += pc["hour_distribution"][h]

    # Systematic missingness: check if variables are differentially missing
    variable_missingness: dict[str, dict] = {}
    if req.variables:
        for var in req.variables:
            present = 0
            missing = 0
            for obs in req.observations:
                if var in obs.variables and not np.isnan(obs.variables[var]):
                    present += 1
                else:
                    missing += 1
            total = present + missing
            variable_missingness[var] = {
                "present": present,
                "missing": missing,
                "missing_rate": round(missing / total, 4) if total > 0 else 0.0,
            }

    # Overall stats
    response_rates = [pc["response_rate"] for pc in participant_compliance]
    overall = {
        "n_participants": len(groups),
        "total_observations": len(req.observations),
        "expected_per_participant": expected_total,
        "mean_response_rate": round(float(np.mean(response_rates)), 4),
        "sd_response_rate": round(float(np.std(response_rates, ddof=1)), 4) if len(response_rates) > 1 else None,
        "min_response_rate": round(float(np.min(response_rates)), 4),
        "max_response_rate": round(float(np.max(response_rates)), 4),
        "aggregate_hour_distribution": agg_hours,
        "peak_hour": int(np.argmax(agg_hours)),
    }

    # Systematic missingness test: chi-square across participants
    obs_counts = [pc["n_responses"] for pc in participant_compliance]
    if len(obs_counts) > 1 and sum(obs_counts) > 0:
        expected_uniform = np.full(len(obs_counts), np.mean(obs_counts))
        chi2, p_uniform = sp_stats.chisquare(obs_counts, f_exp=expected_uniform)
        systematic_test = {
            "test": "chi_square_uniformity",
            "chi2": round(float(chi2), 4),
            "p": round(float(p_uniform), 4),
            "significant": float(p_uniform) < 0.05,
            "interpretation": "Significant = response rates differ systematically across participants",
        }
    else:
        systematic_test = {"detail": "Insufficient participants for systematic test"}

    return {
        "analysis": "ema_compliance",
        "overall": overall,
        "systematic_missingness_test": systematic_test,
        "variable_missingness": variable_missingness if variable_missingness else None,
        "participant_level": participant_compliance,
    }
