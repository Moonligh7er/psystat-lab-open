"""Criminology module — Geographic profiling, spatial clustering, temporal analysis,
recidivism survival analysis, and propensity score matching."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from collections import Counter
from typing import Optional

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class Incident(BaseModel):
    lat: float
    lng: float
    type: Optional[str] = None
    date: Optional[str] = None
    hour: Optional[int] = None
    day_of_week: Optional[str] = None


class GeographicProfileRequest(BaseModel):
    incidents: list[Incident]
    grid_resolution: int = 50
    buffer_zone: float = 0.5
    f: float = 1.2
    g: float = 1.2


class SpatialClusterRequest(BaseModel):
    points: list[list[float]]
    n_clusters: int = 3


class TemporalIncident(BaseModel):
    date: Optional[str] = None
    hour: Optional[int] = None
    day_of_week: Optional[str] = None
    type: Optional[str] = None


class TemporalAnalysisRequest(BaseModel):
    incidents: list[TemporalIncident]


class RecidivismSurvivalRequest(BaseModel):
    time_to_event: list[float]
    event_occurred: list[int]
    risk_factors: Optional[list[list[float]]] = None
    factor_names: Optional[list[str]] = None


class PropensityScoreRequest(BaseModel):
    treatment: list[int]
    covariates: list[list[float]]
    covariate_names: Optional[list[str]] = None
    outcome: list[float]


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/geographic-profile")
def geographic_profile(req: GeographicProfileRequest):
    """Rossmo's formula for geographic profiling of serial incidents."""
    if len(req.incidents) < 2:
        raise HTTPException(400, "At least 2 incidents required for geographic profiling.")
    if req.grid_resolution < 5 or req.grid_resolution > 200:
        raise HTTPException(400, "grid_resolution must be between 5 and 200.")

    coords = np.array([[inc.lat, inc.lng] for inc in req.incidents])
    n = len(coords)
    f, g = req.f, req.g
    B = req.buffer_zone  # buffer zone in km

    # Center of minimum distance (geometric median via Weiszfeld)
    center = _geometric_median(coords)

    # Build grid around incident bounding box
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lng_min, lng_max = coords[:, 1].min(), coords[:, 1].max()
    lat_pad = (lat_max - lat_min) * 0.3 + 0.01
    lng_pad = (lng_max - lng_min) * 0.3 + 0.01

    lat_grid = np.linspace(lat_min - lat_pad, lat_max + lat_pad, req.grid_resolution)
    lng_grid = np.linspace(lng_min - lng_pad, lng_max + lng_pad, req.grid_resolution)

    # Compute Rossmo probability surface
    prob_surface = np.zeros((req.grid_resolution, req.grid_resolution))

    for i, lat in enumerate(lat_grid):
        for j, lng in enumerate(lng_grid):
            total = 0.0
            for k in range(n):
                d = _haversine_km(lat, lng, coords[k, 0], coords[k, 1])
                d = max(d, 0.01)  # avoid division by zero
                if d > B:
                    total += 1.0 / (d ** f)
                else:
                    total += (B ** (g - f)) / ((2 * B - d) ** g)
            prob_surface[i, j] = total

    # Normalize to probabilities
    total_sum = prob_surface.sum()
    if total_sum > 0:
        prob_surface /= total_sum

    # Extract top hotspots (cells above 95th percentile)
    threshold = np.percentile(prob_surface, 95)
    hotspots = []
    for i in range(req.grid_resolution):
        for j in range(req.grid_resolution):
            if prob_surface[i, j] >= threshold:
                hotspots.append({
                    "lat": float(lat_grid[i]),
                    "lng": float(lng_grid[j]),
                    "probability": float(prob_surface[i, j]),
                })

    # Sort by probability descending, limit to top 50
    hotspots.sort(key=lambda h: h["probability"], reverse=True)
    hotspots = hotspots[:50]

    return {
        "n_incidents": n,
        "grid_resolution": req.grid_resolution,
        "buffer_zone_km": B,
        "center_of_minimum_distance": {
            "lat": float(center[0]),
            "lng": float(center[1]),
        },
        "hotspots": hotspots,
    }


@router.post("/spatial-cluster")
def spatial_cluster(req: SpatialClusterRequest):
    """K-Means spatial clustering using scipy/numpy."""
    points = np.array(req.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise HTTPException(400, "Points must be a list of [lat, lng] pairs.")
    n = points.shape[0]
    k = req.n_clusters
    if k < 1:
        raise HTTPException(400, "n_clusters must be at least 1.")
    if k > n:
        raise HTTPException(400, "n_clusters cannot exceed the number of points.")

    # K-Means++ initialization and Lloyd's algorithm
    labels, centers, inertia = _kmeans(points, k, max_iter=100)

    clusters = []
    for c_id in range(k):
        mask = labels == c_id
        cluster_points = points[mask]
        n_pts = int(mask.sum())
        if n_pts > 0:
            c_center = centers[c_id]
            # Compute spread as mean distance from center in km
            dists = np.array([
                _haversine_km(p[0], p[1], c_center[0], c_center[1])
                for p in cluster_points
            ])
            spread = float(dists.mean()) if n_pts > 1 else 0.0
        else:
            c_center = centers[c_id]
            spread = 0.0

        clusters.append({
            "id": c_id,
            "n_points": n_pts,
            "center": {"lat": float(c_center[0]), "lng": float(c_center[1])},
            "spread_km": round(spread, 4),
        })

    return {
        "n_clusters": k,
        "inertia": float(inertia),
        "labels": [int(l) for l in labels],
        "clusters": clusters,
    }


@router.post("/temporal-analysis")
def temporal_analysis(req: TemporalAnalysisRequest):
    """Temporal pattern analysis of incident data."""
    if len(req.incidents) < 1:
        raise HTTPException(400, "At least 1 incident required.")

    n = len(req.incidents)
    hours = [inc.hour for inc in req.incidents if inc.hour is not None]
    days = [inc.day_of_week for inc in req.incidents if inc.day_of_week is not None]
    types = [inc.type for inc in req.incidents if inc.type is not None]

    # Hour distribution (24 bins)
    hour_distribution = [0] * 24
    for h in hours:
        if 0 <= h < 24:
            hour_distribution[h] += 1

    # Day distribution
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_counts = Counter(days)
    day_distribution = {d: day_counts.get(d, 0) for d in day_order}
    # Also include any non-standard day keys present in the data
    for d in day_counts:
        if d not in day_distribution:
            day_distribution[d] = day_counts[d]

    # Type counts
    type_counts = dict(Counter(types))

    # Peak hour
    peak_hour = int(np.argmax(hour_distribution)) if hours else None

    # Peak day
    peak_day = max(day_distribution, key=day_distribution.get) if days else None

    # Most common type
    most_common_type = max(type_counts, key=type_counts.get) if types else None

    return {
        "n_incidents": n,
        "peak_hour": peak_hour,
        "peak_day": peak_day,
        "most_common_type": most_common_type,
        "day_distribution": day_distribution,
        "hour_distribution": hour_distribution,
        "type_counts": type_counts,
    }


@router.post("/recidivism-survival")
def recidivism_survival(req: RecidivismSurvivalRequest):
    """Kaplan-Meier survival analysis for recidivism data with optional
    Cox-proportional-hazards-style risk factor analysis."""
    times = np.array(req.time_to_event, dtype=float)
    events = np.array(req.event_occurred, dtype=int)

    if len(times) != len(events):
        raise HTTPException(400, "time_to_event and event_occurred must have the same length.")
    n = len(times)
    if n < 2:
        raise HTTPException(400, "At least 2 observations required.")
    if not np.all((events == 0) | (events == 1)):
        raise HTTPException(400, "event_occurred must contain only 0 (censored) or 1 (event).")
    if np.any(times < 0):
        raise HTTPException(400, "time_to_event values must be non-negative.")

    # ── Kaplan-Meier estimator ──
    unique_times = np.sort(np.unique(times[events == 1]))
    survival_prob = 1.0
    km_curve = [{"time": 0.0, "survival": 1.0, "n_at_risk": n, "n_events": 0}]

    for t in unique_times:
        n_at_risk = int(np.sum(times >= t))
        n_events_t = int(np.sum((times == t) & (events == 1)))
        if n_at_risk > 0:
            survival_prob *= (1 - n_events_t / n_at_risk)
        km_curve.append({
            "time": float(t),
            "survival": float(survival_prob),
            "n_at_risk": n_at_risk,
            "n_events": n_events_t,
        })

    # Median survival time (time at which survival drops to 0.5)
    median_time = None
    for point in km_curve:
        if point["survival"] <= 0.5:
            median_time = point["time"]
            break

    # Overall recidivism rate
    recidivism_rate = float(events.sum() / n)

    # ── Risk factor analysis (Cox-like univariate) ──
    risk_factor_analysis = []
    if req.risk_factors is not None:
        factors = np.array(req.risk_factors, dtype=float)
        if factors.shape[0] != n:
            raise HTTPException(400, "risk_factors must have the same number of rows as observations.")
        n_factors = factors.shape[1]
        names = req.factor_names or [f"factor_{i}" for i in range(n_factors)]
        if len(names) != n_factors:
            raise HTTPException(400, "factor_names length must match number of columns in risk_factors.")

        for i in range(n_factors):
            x = factors[:, i]
            # Univariate Cox-like analysis using exponential model
            # log-rank test: split at median
            median_x = np.median(x)
            high = x >= median_x
            low = ~high

            if high.sum() < 1 or low.sum() < 1:
                risk_factor_analysis.append({
                    "name": names[i],
                    "hazard_ratio": None,
                    "p_value": None,
                    "coefficient": None,
                    "significant": False,
                    "direction": "insufficient data",
                })
                continue

            # Log-rank test
            chi2, p_val = _log_rank_test(
                times[high], events[high],
                times[low], events[low],
            )

            # Approximate hazard ratio from event rates
            rate_high = events[high].sum() / max(times[high].sum(), 1e-10)
            rate_low = events[low].sum() / max(times[low].sum(), 1e-10)
            hr = rate_high / max(rate_low, 1e-10)

            # Point-biserial correlation as coefficient proxy
            coef = float(np.corrcoef(x, events)[0, 1])

            risk_factor_analysis.append({
                "name": names[i],
                "hazard_ratio": float(round(hr, 4)),
                "p_value": float(round(p_val, 6)),
                "coefficient": float(round(coef, 4)),
                "significant": bool(p_val < 0.05),
                "direction": "increases risk" if hr > 1 else "decreases risk",
            })

    return {
        "n_observations": n,
        "n_events": int(events.sum()),
        "n_censored": int((events == 0).sum()),
        "recidivism_rate": round(recidivism_rate, 4),
        "median_time_to_recidivism": median_time,
        "survival_curve": km_curve,
        "risk_factor_analysis": risk_factor_analysis,
    }


@router.post("/propensity-score")
def propensity_score(req: PropensityScoreRequest):
    """Propensity score matching for causal inference in observational studies."""
    treatment = np.array(req.treatment, dtype=int)
    covariates = np.array(req.covariates, dtype=float)
    outcome = np.array(req.outcome, dtype=float)
    n = len(treatment)

    if covariates.shape[0] != n:
        raise HTTPException(400, "covariates must have the same number of rows as treatment.")
    if len(outcome) != n:
        raise HTTPException(400, "outcome must have the same length as treatment.")
    if not np.all((treatment == 0) | (treatment == 1)):
        raise HTTPException(400, "treatment must contain only 0 (control) or 1 (treated).")

    n_treated = int(treatment.sum())
    n_control = int((treatment == 0).sum())
    if n_treated < 1 or n_control < 1:
        raise HTTPException(400, "Need at least 1 treated and 1 control observation.")

    n_covariates = covariates.shape[1]
    cov_names = req.covariate_names or [f"covariate_{i}" for i in range(n_covariates)]
    if len(cov_names) != n_covariates:
        raise HTTPException(400, "covariate_names length must match number of covariate columns.")

    # ── Estimate propensity scores via logistic regression (Newton-Raphson) ──
    p_scores = _logistic_propensity(covariates, treatment)

    # ── Nearest-neighbor matching (1:1 without replacement) ──
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    matched_pairs = []
    used_controls = set()

    # Sort treated by propensity score for greedy matching
    treated_sorted = treated_idx[np.argsort(p_scores[treated_idx])]

    for t_idx in treated_sorted:
        best_dist = float("inf")
        best_c = None
        for c_idx in control_idx:
            if c_idx in used_controls:
                continue
            dist = abs(p_scores[t_idx] - p_scores[c_idx])
            if dist < best_dist:
                best_dist = dist
                best_c = c_idx
        if best_c is not None:
            matched_pairs.append({
                "treated_index": int(t_idx),
                "control_index": int(best_c),
                "treated_pscore": float(round(p_scores[t_idx], 6)),
                "control_pscore": float(round(p_scores[best_c], 6)),
                "distance": float(round(best_dist, 6)),
            })
            used_controls.add(best_c)

    n_matched = len(matched_pairs)
    if n_matched < 1:
        raise HTTPException(400, "No matched pairs could be formed.")

    # ── ATT (Average Treatment Effect on the Treated) ──
    matched_treated_outcomes = np.array([outcome[p["treated_index"]] for p in matched_pairs])
    matched_control_outcomes = np.array([outcome[p["control_index"]] for p in matched_pairs])
    att = float(np.mean(matched_treated_outcomes - matched_control_outcomes))

    # Standard error via paired differences
    diffs = matched_treated_outcomes - matched_control_outcomes
    att_se = float(np.std(diffs, ddof=1) / np.sqrt(n_matched)) if n_matched > 1 else 0.0
    att_t = att / att_se if att_se > 0 else 0.0
    att_p = float(2 * (sp_stats.t.sf(abs(att_t), df=max(n_matched - 1, 1))))

    # ── Balance statistics (standardized mean differences) ──
    matched_t_idx = [p["treated_index"] for p in matched_pairs]
    matched_c_idx = [p["control_index"] for p in matched_pairs]

    balance_stats = []
    for j in range(n_covariates):
        # Before matching
        mean_t_before = float(covariates[treated_idx, j].mean())
        mean_c_before = float(covariates[control_idx, j].mean())
        pooled_std = np.sqrt(
            (covariates[treated_idx, j].var() + covariates[control_idx, j].var()) / 2
        )
        smd_before = (mean_t_before - mean_c_before) / pooled_std if pooled_std > 0 else 0.0

        # After matching
        mean_t_after = float(covariates[matched_t_idx, j].mean())
        mean_c_after = float(covariates[matched_c_idx, j].mean())
        pooled_std_after = np.sqrt(
            (covariates[matched_t_idx, j].var() + covariates[matched_c_idx, j].var()) / 2
        )
        smd_after = (mean_t_after - mean_c_after) / pooled_std_after if pooled_std_after > 0 else 0.0

        balance_stats.append({
            "covariate": cov_names[j],
            "mean_treated_before": round(mean_t_before, 4),
            "mean_control_before": round(mean_c_before, 4),
            "smd_before": round(smd_before, 4),
            "mean_treated_after": round(mean_t_after, 4),
            "mean_control_after": round(mean_c_after, 4),
            "smd_after": round(smd_after, 4),
            "balanced": bool(abs(smd_after) < 0.1),
        })

    # Effect size (Cohen's d on matched outcomes)
    pooled_sd_outcome = np.sqrt(
        (matched_treated_outcomes.var(ddof=1) + matched_control_outcomes.var(ddof=1)) / 2
    ) if n_matched > 1 else 1.0
    effect_size = att / pooled_sd_outcome if pooled_sd_outcome > 0 else 0.0

    return {
        "n_treated": n_treated,
        "n_control": n_control,
        "n_matched_pairs": n_matched,
        "att": round(att, 6),
        "att_se": round(att_se, 6),
        "att_p_value": round(att_p, 6),
        "effect_size": round(float(effect_size), 4),
        "matched_pairs": matched_pairs,
        "balance_statistics": balance_stats,
    }


# ── Helper Functions ──────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance between two lat/lng points in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _geometric_median(points: np.ndarray, tol: float = 1e-7, max_iter: int = 200) -> np.ndarray:
    """Weiszfeld's algorithm for the geometric median."""
    y = points.mean(axis=0).copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(points - y, axis=1)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        y_new = (points * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(y_new - y) < tol:
            break
        y = y_new
    return y


def _kmeans(points: np.ndarray, k: int, max_iter: int = 100):
    """K-Means clustering with K-Means++ initialization."""
    n = points.shape[0]
    rng = np.random.default_rng(42)

    # K-Means++ init
    centers = np.empty((k, points.shape[1]))
    centers[0] = points[rng.integers(n)]
    for c in range(1, k):
        dists = np.min(cdist(points, centers[:c]), axis=1)
        probs = dists ** 2
        probs /= probs.sum()
        centers[c] = points[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = cdist(points, centers)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = points[mask].mean(axis=0)

    # Inertia
    inertia = sum(
        np.sum((points[labels == c] - centers[c]) ** 2)
        for c in range(k)
    )
    return labels, centers, float(inertia)


def _log_rank_test(
    times1: np.ndarray, events1: np.ndarray,
    times2: np.ndarray, events2: np.ndarray,
) -> tuple[float, float]:
    """Log-rank test comparing two survival curves."""
    all_times = np.sort(np.unique(np.concatenate([
        times1[events1 == 1], times2[events2 == 1]
    ])))

    observed_1 = 0.0
    expected_1 = 0.0
    variance = 0.0

    for t in all_times:
        n1 = np.sum(times1 >= t)
        n2 = np.sum(times2 >= t)
        n_total = n1 + n2
        if n_total == 0:
            continue

        d1 = np.sum((times1 == t) & (events1 == 1))
        d2 = np.sum((times2 == t) & (events2 == 1))
        d_total = d1 + d2

        e1 = d_total * n1 / n_total
        observed_1 += d1
        expected_1 += e1

        if n_total > 1:
            variance += (d_total * n1 * n2 * (n_total - d_total)) / (n_total ** 2 * (n_total - 1))

    if variance <= 0:
        return 0.0, 1.0

    chi2 = (observed_1 - expected_1) ** 2 / variance
    p_value = float(sp_stats.chi2.sf(chi2, df=1))
    return float(chi2), p_value


def _logistic_propensity(X: np.ndarray, y: np.ndarray, max_iter: int = 100, lr: float = 0.1) -> np.ndarray:
    """Simple logistic regression via gradient descent to estimate propensity scores."""
    n, p = X.shape
    # Standardize
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds

    # Add intercept
    X_aug = np.column_stack([np.ones(n), X_std])
    beta = np.zeros(p + 1)

    for _ in range(max_iter):
        z = X_aug @ beta
        z = np.clip(z, -20, 20)
        prob = 1.0 / (1.0 + np.exp(-z))
        grad = X_aug.T @ (prob - y) / n
        beta -= lr * grad

    z = X_aug @ beta
    z = np.clip(z, -20, 20)
    scores = 1.0 / (1.0 + np.exp(-z))
    # Clip to avoid exact 0/1
    return np.clip(scores, 0.001, 0.999)
