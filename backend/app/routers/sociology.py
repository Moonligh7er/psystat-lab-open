"""Sociology — weighted surveys, event history, social mobility, inequality, segregation."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats
import pandas as pd

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class SurveyWeightedRequest(BaseModel):
    values: list[float]
    weights: list[float]
    groups: list[int | str] | None = None
    strata: list[int | str] | None = None


class EventHistoryRequest(BaseModel):
    durations: list[float]
    events: list[int]  # 1=event, 0=censored
    covariates: list[list[float]] | None = None
    covariate_names: list[str] | None = None
    time_intervals: list[float] | None = None


class SocialMobilityRequest(BaseModel):
    origin: list[int | str]
    destination: list[int | str]
    weights: list[float] | None = None


class InequalityRequest(BaseModel):
    values: list[float]
    weights: list[float] | None = None
    groups: list[str] | None = None


class SegregationRequest(BaseModel):
    group_a_counts: list[int]
    group_b_counts: list[int]
    unit_labels: list[str] | None = None


# ── Helpers ───────────────────────────────────────────────────────────

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    wm = np.average(values, weights=weights)
    return float(np.average((values - wm) ** 2, weights=weights))


def _weighted_se(values: np.ndarray, weights: np.ndarray) -> float:
    """Linearization-based SE for weighted mean."""
    n = len(values)
    wm = np.average(values, weights=weights)
    w_sum = np.sum(weights)
    resid = values - wm
    var_est = np.sum(weights ** 2 * resid ** 2) / (w_sum ** 2)
    # finite-population-style correction
    correction = n / (n - 1) if n > 1 else 1.0
    return float(np.sqrt(var_est * correction))


def _design_effect(weights: np.ndarray) -> float:
    """Kish's DEFF = n * sum(w^2) / (sum(w))^2."""
    n = len(weights)
    return float(n * np.sum(weights ** 2) / np.sum(weights) ** 2)


def _gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute the Gini coefficient, optionally weighted."""
    if weights is None:
        weights = np.ones(len(values))
    sorted_idx = np.argsort(values)
    v = values[sorted_idx]
    w = weights[sorted_idx]
    cum_w = np.cumsum(w)
    cum_vw = np.cumsum(v * w)
    total_vw = cum_vw[-1]
    total_w = cum_w[-1]
    # Trapezoidal Lorenz area
    lorenz_area = np.sum(w[1:] * (cum_vw[:-1] + cum_vw[1:])) / (2.0 * total_w * total_vw)
    return float(1.0 - 2.0 * lorenz_area)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.post("/survey-weighted")
def survey_weighted(req: SurveyWeightedRequest):
    """Weighted survey analysis with optional grouping and stratification."""
    values = np.array(req.values, dtype=float)
    weights = np.array(req.weights, dtype=float)

    if len(values) != len(weights):
        raise HTTPException(400, "values and weights must have the same length.")
    if len(values) < 2:
        raise HTTPException(400, "Need at least 2 observations.")
    if np.any(weights <= 0):
        raise HTTPException(400, "All weights must be positive.")

    wm = _weighted_mean(values, weights)
    wv = _weighted_variance(values, weights)
    se = _weighted_se(values, weights)
    deff = _design_effect(weights)
    n_eff = float(len(values) / deff)
    ci_lower = float(wm - 1.96 * se)
    ci_upper = float(wm + 1.96 * se)

    result: dict = {
        "weighted_mean": round(wm, 6),
        "weighted_se": round(se, 6),
        "ci_95": [round(ci_lower, 6), round(ci_upper, 6)],
        "weighted_variance": round(wv, 6),
        "design_effect": round(deff, 4),
        "effective_sample_size": round(n_eff, 2),
        "n": len(values),
    }

    # ── Group analysis ────────────────────────────────────────────────
    if req.groups is not None:
        groups = np.array(req.groups)
        if len(groups) != len(values):
            raise HTTPException(400, "groups must have the same length as values.")
        unique_groups = np.unique(groups)
        group_stats = {}
        for g in unique_groups:
            mask = groups == g
            gv = values[mask]
            gw = weights[mask]
            gm = _weighted_mean(gv, gw)
            gse = _weighted_se(gv, gw)
            group_stats[str(g)] = {
                "weighted_mean": round(gm, 6),
                "weighted_se": round(gse, 6),
                "ci_95": [round(gm - 1.96 * gse, 6), round(gm + 1.96 * gse, 6)],
                "n": int(np.sum(mask)),
            }
        result["group_estimates"] = group_stats

        # Weighted chi-square test of independence (Rao-Scott style approximation)
        # Bin values into quartiles for chi-square
        try:
            quartiles = np.percentile(values, [25, 50, 75])
            binned = np.digitize(values, quartiles)
            obs_table = pd.crosstab(
                pd.Series(groups, name="group"),
                pd.Series(binned, name="bin"),
            )
            chi2, p, dof, expected = sp_stats.chi2_contingency(obs_table.values)
            # Rao-Scott correction: divide chi2 by average DEFF
            avg_deff = deff
            chi2_adj = chi2 / avg_deff
            p_adj = float(sp_stats.chi2.sf(chi2_adj, dof))
            result["weighted_chi_square"] = {
                "chi2_unadjusted": round(float(chi2), 4),
                "chi2_rao_scott": round(float(chi2_adj), 4),
                "dof": int(dof),
                "p_value": round(p_adj, 6),
            }
        except Exception:
            result["weighted_chi_square"] = {"error": "Could not compute weighted chi-square."}

    # ── Stratified estimates ──────────────────────────────────────────
    if req.strata is not None:
        strata = np.array(req.strata)
        if len(strata) != len(values):
            raise HTTPException(400, "strata must have the same length as values.")
        unique_strata = np.unique(strata)
        strat_estimates = {}
        combined_wm = 0.0
        total_weight = 0.0
        for s in unique_strata:
            mask = strata == s
            sv = values[mask]
            sw = weights[mask]
            sm = _weighted_mean(sv, sw)
            sse = _weighted_se(sv, sw)
            sw_sum = float(np.sum(sw))
            strat_estimates[str(s)] = {
                "weighted_mean": round(sm, 6),
                "weighted_se": round(sse, 6),
                "ci_95": [round(sm - 1.96 * sse, 6), round(sm + 1.96 * sse, 6)],
                "n": int(np.sum(mask)),
                "weight_sum": round(sw_sum, 4),
            }
            combined_wm += sm * sw_sum
            total_weight += sw_sum
        combined_wm /= total_weight
        result["stratified_estimates"] = strat_estimates
        result["combined_stratified_mean"] = round(float(combined_wm), 6)

    return result


@router.post("/event-history")
def event_history(req: EventHistoryRequest):
    """Event history / discrete-time survival analysis."""
    durations = np.array(req.durations, dtype=float)
    events = np.array(req.events, dtype=int)

    if len(durations) != len(events):
        raise HTTPException(400, "durations and events must have the same length.")
    if len(durations) < 3:
        raise HTTPException(400, "Need at least 3 observations.")
    if not set(np.unique(events)).issubset({0, 1}):
        raise HTTPException(400, "events must contain only 0 (censored) or 1 (event).")
    if np.any(durations < 0):
        raise HTTPException(400, "durations must be non-negative.")

    # ── Build life table ──────────────────────────────────────────────
    if req.time_intervals is not None:
        breaks = sorted(set([0.0] + list(req.time_intervals)))
    else:
        unique_times = np.unique(durations)
        if len(unique_times) <= 10:
            breaks = sorted(set([0.0] + unique_times.tolist()))
        else:
            breaks = np.linspace(0, float(np.max(durations)), min(11, len(unique_times) + 1)).tolist()

    life_table = []
    n_total = len(durations)
    n_remaining = n_total
    cum_survival = 1.0

    for i in range(len(breaks) - 1):
        t_start = breaks[i]
        t_end = breaks[i + 1]
        # Subjects at risk at interval start
        at_risk = int(np.sum(durations >= t_start))
        # Events in this interval
        n_events_interval = int(np.sum((durations >= t_start) & (durations < t_end) & (events == 1)))
        # Censored in this interval
        n_censored_interval = int(np.sum((durations >= t_start) & (durations < t_end) & (events == 0)))
        # Adjusted at risk (actuarial method)
        adjusted_risk = at_risk - n_censored_interval / 2.0
        hazard = float(n_events_interval / adjusted_risk) if adjusted_risk > 0 else 0.0
        interval_survival = 1.0 - hazard
        cum_survival *= interval_survival
        # Greenwood SE
        if adjusted_risk > 0 and interval_survival > 0:
            var_component = n_events_interval / (adjusted_risk * (adjusted_risk - n_events_interval)) if adjusted_risk > n_events_interval else 0.0
        else:
            var_component = 0.0
        se = float(cum_survival * np.sqrt(var_component)) if var_component >= 0 else 0.0

        life_table.append({
            "interval": f"[{round(t_start, 2)}, {round(t_end, 2)})",
            "n_at_risk": at_risk,
            "n_events": n_events_interval,
            "n_censored": n_censored_interval,
            "hazard": round(hazard, 6),
            "survival": round(cum_survival, 6),
            "se": round(se, 6),
        })

    # Median survival: first interval where survival <= 0.5
    median_survival = None
    for row in life_table:
        if row["survival"] <= 0.5:
            # Parse interval start
            interval_str = row["interval"]
            median_survival = float(interval_str.split(",")[0].strip("["))
            break

    result: dict = {
        "life_table": life_table,
        "median_survival": median_survival,
        "total_events": int(np.sum(events)),
        "total_censored": int(np.sum(events == 0)),
        "n": len(durations),
    }

    # Hazard rates by interval
    result["hazard_by_interval"] = [
        {"interval": row["interval"], "hazard": row["hazard"]} for row in life_table
    ]

    # ── Covariates: discrete-time logistic regression ─────────────────
    if req.covariates is not None:
        cov = np.array(req.covariates, dtype=float)
        if cov.shape[0] != len(durations):
            raise HTTPException(400, "covariates rows must match the number of observations.")
        n_covariates = cov.shape[1]
        cov_names = req.covariate_names or [f"X{i+1}" for i in range(n_covariates)]
        if len(cov_names) != n_covariates:
            raise HTTPException(400, "covariate_names length must match number of covariates.")

        # Expand to person-period format
        person_ids = []
        period_ids = []
        y_pp = []
        cov_pp = []

        for j, (dur, ev) in enumerate(zip(durations, events)):
            for i in range(len(breaks) - 1):
                t_start = breaks[i]
                t_end = breaks[i + 1]
                if dur >= t_start:
                    person_ids.append(j)
                    period_ids.append(i)
                    cov_pp.append(cov[j].tolist())
                    if dur < t_end and ev == 1:
                        y_pp.append(1)
                    else:
                        y_pp.append(0)
                if dur < t_end:
                    break

        y_pp = np.array(y_pp, dtype=float)
        cov_pp = np.array(cov_pp, dtype=float)
        period_ids_arr = np.array(period_ids)

        # Add period dummies
        n_periods = len(breaks) - 1
        if n_periods > 1:
            period_dummies = np.zeros((len(y_pp), n_periods - 1))
            for k in range(1, n_periods):
                period_dummies[period_ids_arr == k, k - 1] = 1.0
            X_full = np.column_stack([cov_pp, period_dummies])
        else:
            X_full = cov_pp

        # Fit logistic regression via statsmodels
        try:
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X_full)
            model = sm.Logit(y_pp, X_with_const)
            fit = model.fit(disp=0, maxiter=100)

            covariate_results = []
            # Extract only covariate coefficients (skip intercept at index 0)
            for k in range(n_covariates):
                idx = k + 1  # +1 for intercept
                coef = float(fit.params[idx])
                se_coef = float(fit.bse[idx])
                odds_ratio = float(np.exp(coef))
                ci_low = float(np.exp(coef - 1.96 * se_coef))
                ci_high = float(np.exp(coef + 1.96 * se_coef))
                p_val = float(fit.pvalues[idx])
                covariate_results.append({
                    "name": cov_names[k],
                    "coefficient": round(coef, 6),
                    "odds_ratio": round(odds_ratio, 4),
                    "or_ci_95": [round(ci_low, 4), round(ci_high, 4)],
                    "se": round(se_coef, 6),
                    "z": round(float(fit.tvalues[idx]), 4),
                    "p_value": round(p_val, 6),
                })
            result["discrete_time_logistic"] = {
                "covariates": covariate_results,
                "aic": round(float(fit.aic), 2),
                "bic": round(float(fit.bic), 2),
                "pseudo_r2": round(float(fit.prsquared), 4),
                "n_person_periods": len(y_pp),
            }
        except Exception as e:
            result["discrete_time_logistic"] = {"error": f"Model fitting failed: {str(e)}"}

        # ── Log-rank test (using first covariate as binary group) ─────
        first_cov = cov[:, 0]
        unique_vals = np.unique(first_cov)
        if len(unique_vals) == 2:
            try:
                g1_mask = first_cov == unique_vals[0]
                g2_mask = first_cov == unique_vals[1]
                # Manual log-rank statistic
                all_event_times = np.unique(durations[events == 1])
                observed_1 = 0.0
                expected_1 = 0.0
                var_sum = 0.0
                for t in all_event_times:
                    d = np.sum((durations == t) & (events == 1))
                    n_risk = np.sum(durations >= t)
                    d1 = np.sum((durations == t) & (events == 1) & g1_mask)
                    n1 = np.sum((durations >= t) & g1_mask)
                    n2 = np.sum((durations >= t) & g2_mask)
                    e1 = n1 * d / n_risk if n_risk > 0 else 0
                    observed_1 += d1
                    expected_1 += e1
                    if n_risk > 1:
                        var_sum += (n1 * n2 * d * (n_risk - d)) / (n_risk ** 2 * (n_risk - 1))
                if var_sum > 0:
                    log_rank_stat = float((observed_1 - expected_1) ** 2 / var_sum)
                    log_rank_p = float(sp_stats.chi2.sf(log_rank_stat, 1))
                    result["log_rank_test"] = {
                        "chi2": round(log_rank_stat, 4),
                        "dof": 1,
                        "p_value": round(log_rank_p, 6),
                        "groups": [str(unique_vals[0]), str(unique_vals[1])],
                    }
            except Exception:
                pass

    return result


@router.post("/social-mobility")
def social_mobility(req: SocialMobilityRequest):
    """Social mobility analysis — intergenerational mobility tables."""
    origin = np.array(req.origin)
    destination = np.array(req.destination)

    if len(origin) != len(destination):
        raise HTTPException(400, "origin and destination must have the same length.")
    if len(origin) < 5:
        raise HTTPException(400, "Need at least 5 observations.")

    weights = np.array(req.weights, dtype=float) if req.weights is not None else np.ones(len(origin))
    if len(weights) != len(origin):
        raise HTTPException(400, "weights must have the same length as origin/destination.")

    # Build categories (sorted union)
    all_cats = sorted(set(origin.tolist()) | set(destination.tolist()), key=str)
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    k = len(all_cats)

    # Build weighted mobility table
    table = np.zeros((k, k), dtype=float)
    for o, d, w in zip(origin, destination, weights):
        table[cat_to_idx[o], cat_to_idx[d]] += w

    total = table.sum()
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)

    # Row percentages (outflow)
    row_pct = np.zeros_like(table)
    for i in range(k):
        if row_totals[i] > 0:
            row_pct[i] = table[i] / row_totals[i] * 100

    # Column percentages (inflow)
    col_pct = np.zeros_like(table)
    for j in range(k):
        if col_totals[j] > 0:
            col_pct[:, j] = table[:, j] / col_totals[j] * 100

    # Mobility rates (assume categories are ordinal in sorted order)
    diagonal = sum(table[i, i] for i in range(k))
    immobility_rate = float(diagonal / total) if total > 0 else 0.0
    total_mobility_rate = 1.0 - immobility_rate

    upward = 0.0
    downward = 0.0
    for i in range(k):
        for j in range(k):
            if j > i:
                upward += table[i, j]
            elif j < i:
                downward += table[i, j]
    upward_rate = float(upward / total) if total > 0 else 0.0
    downward_rate = float(downward / total) if total > 0 else 0.0

    # Odds ratios for adjacent category pairs
    odds_ratios = []
    for i in range(k - 1):
        for j in range(k - 1):
            a, b = table[i, j], table[i, j + 1]
            c, d_ = table[i + 1, j], table[i + 1, j + 1]
            if b > 0 and c > 0:
                or_val = float((a * d_) / (b * c))
                odds_ratios.append({
                    "origin_pair": [str(all_cats[i]), str(all_cats[i + 1])],
                    "destination_pair": [str(all_cats[j]), str(all_cats[j + 1])],
                    "odds_ratio": round(or_val, 4),
                    "log_odds_ratio": round(float(np.log(or_val)) if or_val > 0 else float("nan"), 4),
                })

    # Chi-square test of independence
    chi2, p, dof, expected = sp_stats.chi2_contingency(table)

    # Glass's social distance index: mean absolute distance from diagonal
    glass_total = 0.0
    glass_weight = 0.0
    for i in range(k):
        for j in range(k):
            glass_total += table[i, j] * abs(i - j)
            glass_weight += table[i, j]
    glass_index = float(glass_total / glass_weight) if glass_weight > 0 else 0.0

    # Dissimilarity index (per origin category vs marginal destination distribution)
    dest_pct_marginal = col_totals / total if total > 0 else np.zeros(k)
    dissimilarity_by_origin = {}
    for i in range(k):
        if row_totals[i] > 0:
            origin_dest_pct = table[i] / row_totals[i]
            d_idx = float(0.5 * np.sum(np.abs(origin_dest_pct - dest_pct_marginal)))
            dissimilarity_by_origin[str(all_cats[i])] = round(d_idx, 4)

    # Format table
    cat_labels = [str(c) for c in all_cats]
    mobility_table = {
        "categories": cat_labels,
        "counts": table.tolist(),
        "row_percentages": np.round(row_pct, 2).tolist(),
        "column_percentages": np.round(col_pct, 2).tolist(),
        "row_totals": row_totals.tolist(),
        "column_totals": col_totals.tolist(),
        "total": float(total),
    }

    return {
        "mobility_table": mobility_table,
        "total_mobility_rate": round(total_mobility_rate, 4),
        "upward_mobility_rate": round(upward_rate, 4),
        "downward_mobility_rate": round(downward_rate, 4),
        "immobility_rate": round(immobility_rate, 4),
        "odds_ratios": odds_ratios,
        "chi_square_test": {
            "chi2": round(float(chi2), 4),
            "dof": int(dof),
            "p_value": round(float(p), 6),
        },
        "glass_social_distance_index": round(glass_index, 4),
        "dissimilarity_index_by_origin": dissimilarity_by_origin,
        "n": len(origin),
    }


@router.post("/inequality-indices")
def inequality_indices(req: InequalityRequest):
    """Compute inequality and concentration indices."""
    values = np.array(req.values, dtype=float)

    if len(values) < 3:
        raise HTTPException(400, "Need at least 3 observations.")
    if np.any(values < 0):
        raise HTTPException(400, "Values must be non-negative for inequality indices.")

    weights = np.array(req.weights, dtype=float) if req.weights is not None else np.ones(len(values))
    if len(weights) != len(values):
        raise HTTPException(400, "weights must have the same length as values.")

    total_w = np.sum(weights)
    w_mean = np.average(values, weights=weights)

    if w_mean == 0:
        raise HTTPException(400, "Mean of values is zero — cannot compute inequality indices.")

    # ── Gini coefficient with bootstrap CI ────────────────────────────
    gini_point = _gini(values, weights)
    rng = np.random.default_rng(42)
    n = len(values)
    n_boot = 1000
    gini_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        gini_boot[b] = _gini(values[idx], weights[idx])
    gini_ci = [round(float(np.percentile(gini_boot, 2.5)), 4),
               round(float(np.percentile(gini_boot, 97.5)), 4)]
    gini_se = round(float(np.std(gini_boot, ddof=1)), 4)

    # ── Theil indices ─────────────────────────────────────────────────
    # Theil T (GE(1))
    v_pos = values.copy()
    w_pos = weights.copy()
    # Filter out zero values for log computation
    pos_mask = v_pos > 0
    if np.sum(pos_mask) < 2:
        theil_t = float("nan")
        theil_l = float("nan")
    else:
        vp = v_pos[pos_mask]
        wp = w_pos[pos_mask]
        wp_sum = np.sum(wp)
        mu = np.average(vp, weights=wp)
        # Theil T = (1/N) * sum(w_i * (y_i/mu) * ln(y_i/mu))
        ratio = vp / mu
        theil_t = float(np.sum(wp * ratio * np.log(ratio)) / wp_sum)
        # Theil L = (1/N) * sum(w_i * ln(mu/y_i))
        theil_l = float(np.sum(wp * np.log(mu / vp)) / wp_sum)

    # ── Atkinson index ────────────────────────────────────────────────
    atkinson = {}
    for eps in [0.5, 1.0, 2.0]:
        if np.sum(pos_mask) < 2:
            atkinson[str(eps)] = float("nan")
            continue
        vp = values[pos_mask]
        wp = weights[pos_mask]
        mu = np.average(vp, weights=wp)
        wp_sum = np.sum(wp)
        if eps == 1.0:
            # Geometric mean
            log_mean = np.sum(wp * np.log(vp)) / wp_sum
            atkinson["1.0"] = round(float(1.0 - np.exp(log_mean) / mu), 6)
        else:
            power = 1.0 - eps
            gen_mean = (np.sum(wp * vp ** power) / wp_sum) ** (1.0 / power)
            atkinson[str(eps)] = round(float(1.0 - gen_mean / mu), 6)

    # ── Palma ratio (top 10% / bottom 40%) ────────────────────────────
    sorted_idx = np.argsort(values)
    sv = values[sorted_idx]
    sw = weights[sorted_idx]
    cum_w = np.cumsum(sw)
    cum_w_pct = cum_w / cum_w[-1]
    bottom_40_mask = cum_w_pct <= 0.40
    top_10_mask = cum_w_pct > 0.90
    bottom_40_income = float(np.sum(sv[bottom_40_mask] * sw[bottom_40_mask]))
    top_10_income = float(np.sum(sv[top_10_mask] * sw[top_10_mask]))
    palma = round(float(top_10_income / bottom_40_income), 4) if bottom_40_income > 0 else float("inf")

    # ── Percentile ratios ─────────────────────────────────────────────
    def _weighted_percentile(vals_sorted, weights_sorted, pct):
        cum = np.cumsum(weights_sorted)
        target = pct / 100.0 * cum[-1]
        idx = np.searchsorted(cum, target)
        idx = min(idx, len(vals_sorted) - 1)
        return float(vals_sorted[idx])

    p10 = _weighted_percentile(sv, sw, 10)
    p50 = _weighted_percentile(sv, sw, 50)
    p90 = _weighted_percentile(sv, sw, 90)

    percentile_ratios = {
        "p90_p10": round(p90 / p10, 4) if p10 > 0 else float("inf"),
        "p90_p50": round(p90 / p50, 4) if p50 > 0 else float("inf"),
        "p50_p10": round(p50 / p10, 4) if p10 > 0 else float("inf"),
    }

    result: dict = {
        "gini": round(gini_point, 4),
        "gini_se": gini_se,
        "gini_ci_95": gini_ci,
        "theil_t": round(theil_t, 6) if not np.isnan(theil_t) else None,
        "theil_l": round(theil_l, 6) if not np.isnan(theil_l) else None,
        "atkinson": atkinson,
        "palma_ratio": palma,
        "percentile_ratios": percentile_ratios,
        "weighted_mean": round(float(w_mean), 4),
        "n": len(values),
    }

    # ── Between/within group decomposition of Theil T ─────────────────
    if req.groups is not None:
        groups = np.array(req.groups)
        if len(groups) != len(values):
            raise HTTPException(400, "groups must have the same length as values.")

        unique_groups = np.unique(groups)
        overall_mu = w_mean
        between_t = 0.0
        within_t = 0.0
        group_details = {}

        for g in unique_groups:
            mask = groups == g
            gv = values[mask]
            gw = weights[mask]
            g_mu = np.average(gv, weights=gw)
            g_share = np.sum(gv * gw) / np.sum(values * weights)  # income share
            g_pop_share = np.sum(gw) / total_w

            # Within-group Theil T
            g_pos = gv > 0
            if np.sum(g_pos) >= 2 and g_mu > 0:
                gvp = gv[g_pos]
                gwp = gw[g_pos]
                ratio = gvp / g_mu
                g_theil = float(np.sum(gwp * ratio * np.log(ratio)) / np.sum(gwp))
            else:
                g_theil = 0.0

            within_t += g_share * g_theil
            if g_mu > 0 and overall_mu > 0:
                between_t += g_share * np.log(g_mu / overall_mu)

            group_details[str(g)] = {
                "mean": round(float(g_mu), 4),
                "theil_t": round(g_theil, 6),
                "income_share": round(float(g_share), 4),
                "population_share": round(float(g_pop_share), 4),
                "n": int(np.sum(mask)),
            }

        result["theil_decomposition"] = {
            "between_group": round(float(between_t), 6),
            "within_group": round(float(within_t), 6),
            "total": round(float(between_t + within_t), 6),
            "between_share_pct": round(float(between_t / (between_t + within_t) * 100), 2) if (between_t + within_t) > 0 else 0.0,
            "group_details": group_details,
        }

    return result


@router.post("/segregation-indices")
def segregation_indices(req: SegregationRequest):
    """Compute residential/occupational segregation measures."""
    a = np.array(req.group_a_counts, dtype=float)
    b = np.array(req.group_b_counts, dtype=float)

    if len(a) != len(b):
        raise HTTPException(400, "group_a_counts and group_b_counts must have the same length.")
    if len(a) < 2:
        raise HTTPException(400, "Need at least 2 spatial units.")
    if np.any(a < 0) or np.any(b < 0):
        raise HTTPException(400, "Counts must be non-negative.")

    total_a = np.sum(a)
    total_b = np.sum(b)
    total_pop = total_a + total_b

    if total_a == 0 or total_b == 0:
        raise HTTPException(400, "Both groups must have at least some members.")

    unit_totals = a + b
    n_units = len(a)
    labels = req.unit_labels or [f"Unit_{i+1}" for i in range(n_units)]

    P = total_a / total_pop  # overall proportion of group A

    # ── Dissimilarity Index (D) ───────────────────────────────────────
    D = float(0.5 * np.sum(np.abs(a / total_a - b / total_b)))

    # ── Isolation Index (xPx) ─────────────────────────────────────────
    # Probability that a randomly drawn neighbor of a group-A member is also group A
    xPx = 0.0
    for i in range(n_units):
        if unit_totals[i] > 0:
            xPx += (a[i] / total_a) * (a[i] / unit_totals[i])
    xPx = float(xPx)

    # ── Exposure Index (xPy) ──────────────────────────────────────────
    # Probability that a randomly drawn neighbor of a group-A member is group B
    xPy = 0.0
    for i in range(n_units):
        if unit_totals[i] > 0:
            xPy += (a[i] / total_a) * (b[i] / unit_totals[i])
    xPy = float(xPy)

    # ── Information Theory Index (H / Theil) ──────────────────────────
    # Overall entropy
    E_total = 0.0
    if P > 0 and P < 1:
        E_total = -(P * np.log(P) + (1 - P) * np.log(1 - P))

    H = 0.0
    if E_total > 0:
        for i in range(n_units):
            t_i = unit_totals[i]
            if t_i > 0:
                p_i = a[i] / t_i
                if 0 < p_i < 1:
                    e_i = -(p_i * np.log(p_i) + (1 - p_i) * np.log(1 - p_i))
                else:
                    e_i = 0.0
                H += t_i * (E_total - e_i)
        H = float(H / (total_pop * E_total))

    # ── Gini Segregation Index ────────────────────────────────────────
    gini_seg = 0.0
    for i in range(n_units):
        for j in range(n_units):
            ti = unit_totals[i]
            tj = unit_totals[j]
            if ti > 0 and tj > 0:
                gini_seg += ti * tj * abs(a[i] / ti - a[j] / tj)
    gini_seg = float(gini_seg / (2 * total_pop ** 2 * P * (1 - P))) if P > 0 and P < 1 else 0.0

    return {
        "dissimilarity_index": round(D, 4),
        "dissimilarity_interpretation": _interpret_D(D),
        "isolation_index": round(xPx, 4),
        "isolation_interpretation": _interpret_isolation(xPx, P),
        "exposure_index": round(xPy, 4),
        "exposure_interpretation": f"A group-A member's average neighborhood is {round(xPy * 100, 1)}% group B.",
        "information_theory_index": round(H, 4),
        "information_theory_interpretation": _interpret_H(H),
        "gini_segregation_index": round(gini_seg, 4),
        "gini_interpretation": f"Gini segregation of {round(gini_seg, 3)} (0 = perfect integration, 1 = complete segregation).",
        "summary": {
            "n_units": n_units,
            "total_group_a": int(total_a),
            "total_group_b": int(total_b),
            "overall_proportion_a": round(float(P), 4),
        },
    }


def _interpret_D(D: float) -> str:
    if D < 0.3:
        level = "low"
    elif D < 0.6:
        level = "moderate"
    else:
        level = "high"
    return (
        f"Dissimilarity of {round(D, 3)} indicates {level} segregation. "
        f"{round(D * 100, 1)}% of group A (or B) would need to relocate for even distribution."
    )


def _interpret_isolation(xPx: float, P: float) -> str:
    return (
        f"Isolation of {round(xPx, 3)} vs. overall proportion {round(P, 3)}. "
        f"Group A members live in areas that are on average {round(xPx * 100, 1)}% group A "
        f"(compared to {round(P * 100, 1)}% overall)."
    )


def _interpret_H(H: float) -> str:
    if H < 0.1:
        level = "very low"
    elif H < 0.3:
        level = "low to moderate"
    elif H < 0.5:
        level = "moderate"
    else:
        level = "high"
    return (
        f"Information theory index of {round(H, 3)} indicates {level} segregation. "
        f"Represents the proportion of total entropy accounted for by between-unit differences."
    )
