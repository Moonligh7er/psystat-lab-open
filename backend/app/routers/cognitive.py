"""Cognitive Lab — scoring and analysis for cognitive test battery.

Scores trial-level data from Stroop, N-Back, Flanker, Go/No-Go and task-switching
tasks. RT effects use correct trials only and Welch's t-test (RT variances usually
differ between conditions). Signal-detection indices use the log-linear correction
(Hautus, 1995): H = (hits + 0.5) / (targets + 1), F = (false alarms + 0.5) / (non-targets + 1).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


class _PairedConditions(BaseModel):
    """Two conditions, each with per-trial RTs (ms) and correctness flags."""

    def _check(self, a_rt, a_ok, b_rt, b_ok, a_name, b_name):
        if len(a_rt) != len(a_ok):
            raise ValueError(f"{a_name}: {len(a_rt)} RTs but {len(a_ok)} correctness flags")
        if len(b_rt) != len(b_ok):
            raise ValueError(f"{b_name}: {len(b_rt)} RTs but {len(b_ok)} correctness flags")
        if not a_rt or not b_rt:
            raise ValueError("Each condition needs at least one trial")
        if any(r <= 0 for r in a_rt + b_rt):
            raise ValueError("Reaction times must be positive (milliseconds)")
        return self


class StroopRequest(_PairedConditions):
    congruent_rts: list[float]     # reaction times in ms
    incongruent_rts: list[float]
    congruent_correct: list[bool]
    incongruent_correct: list[bool]

    @model_validator(mode="after")
    def _v(self):
        return self._check(self.congruent_rts, self.congruent_correct, self.incongruent_rts, self.incongruent_correct, "congruent", "incongruent")


class NBackRequest(BaseModel):
    responses: list[dict]  # [{stimulus, is_target, responded, rt}, ...]


class FlankerRequest(_PairedConditions):
    congruent_rts: list[float]
    incongruent_rts: list[float]
    congruent_correct: list[bool]
    incongruent_correct: list[bool]

    @model_validator(mode="after")
    def _v(self):
        return self._check(self.congruent_rts, self.congruent_correct, self.incongruent_rts, self.incongruent_correct, "congruent", "incongruent")


class GoNoGoRequest(BaseModel):
    trials: list[dict]  # [{trial_type: "go"|"nogo", responded: bool, rt: float|null}, ...]


class TaskSwitchingRequest(_PairedConditions):
    switch_rts: list[float]
    repeat_rts: list[float]
    switch_correct: list[bool]
    repeat_correct: list[bool]

    @model_validator(mode="after")
    def _v(self):
        return self._check(self.switch_rts, self.switch_correct, self.repeat_rts, self.repeat_correct, "switch", "repeat")


class BatteryComparisonRequest(BaseModel):
    """Compare two test sessions."""
    session1: dict  # {stroop_interference, nback_dprime, flanker_effect, ...}
    session2: dict


# ── Helpers ────────────────────────────────────────────────────────────

def _rt_effect(slow_rt, slow_ok, fast_rt, fast_ok):
    """Mean RT difference (slow − fast) on correct trials with Welch's t-test.

    Returns None for quantities that cannot be computed rather than a placeholder.
    """
    slow = np.asarray(slow_rt, float)[np.asarray(slow_ok, bool)]
    fast = np.asarray(fast_rt, float)[np.asarray(fast_ok, bool)]
    out = {
        "slow_mean": float(slow.mean()) if len(slow) else None,
        "fast_mean": float(fast.mean()) if len(fast) else None,
        "slow_sd": float(slow.std(ddof=1)) if len(slow) > 1 else None,
        "fast_sd": float(fast.std(ddof=1)) if len(fast) > 1 else None,
        "effect": float(slow.mean() - fast.mean()) if len(slow) and len(fast) else None,
        "t": None, "df": None, "p": None, "n_slow": int(len(slow)), "n_fast": int(len(fast)),
    }
    if len(slow) > 1 and len(fast) > 1 and (slow.var(ddof=1) > 0 or fast.var(ddof=1) > 0):
        res = sp_stats.ttest_ind(slow, fast, equal_var=False)
        out.update(t=float(res.statistic), df=float(res.df), p=float(res.pvalue))
    return out


def _accuracy(flags):
    return float(np.mean(flags)) if len(flags) else None


def _sdt(hits, misses, false_alarms, correct_rejections):
    """d' and criterion c with the log-linear correction (Hautus, 1995)."""
    n_sig, n_noise = hits + misses, false_alarms + correct_rejections
    if n_sig == 0 or n_noise == 0:
        return None, None
    H = (hits + 0.5) / (n_sig + 1)
    F = (false_alarms + 0.5) / (n_noise + 1)
    zH, zF = sp_stats.norm.ppf(H), sp_stats.norm.ppf(F)
    return float(zH - zF), float(-0.5 * (zH + zF))


SDT_NOTE = "d′ and c use the log-linear correction (Hautus, 1995): H = (hits + 0.5)/(targets + 1), F = (FA + 0.5)/(non-targets + 1)."
RT_NOTE = "RTs from correct trials only; Welch's t-test on trial-level RTs (within one participant)."


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/stroop")
def score_stroop(req: StroopRequest):
    """Score Stroop test — interference effect, accuracy, RT analysis."""
    e = _rt_effect(req.incongruent_rts, req.incongruent_correct, req.congruent_rts, req.congruent_correct)
    return {
        "interference_ms": e["effect"],
        "congruent_mean_rt": e["fast_mean"],
        "incongruent_mean_rt": e["slow_mean"],
        "congruent_accuracy": _accuracy(req.congruent_correct),
        "incongruent_accuracy": _accuracy(req.incongruent_correct),
        "overall_accuracy": _accuracy(req.congruent_correct + req.incongruent_correct),
        "congruent_rt_std": e["fast_sd"],
        "incongruent_rt_std": e["slow_sd"],
        "n_correct_congruent": e["n_fast"],
        "n_correct_incongruent": e["n_slow"],
        "t_statistic": e["t"],
        "df": e["df"],
        "p_value": e["p"],
        "significant_interference": None if e["p"] is None else bool(e["p"] < 0.05),
        "method": RT_NOTE,
    }


@router.post("/nback")
def score_nback(req: NBackRequest):
    """Score N-Back task — d', hit rate, false alarm rate."""
    if not req.responses:
        raise HTTPException(422, "No trials supplied")
    hits = misses = false_alarms = correct_rejections = 0
    hit_rts = []
    for trial in req.responses:
        is_target = bool(trial.get("is_target", False))
        responded = bool(trial.get("responded", False))
        rt = trial.get("rt")
        if is_target and responded:
            hits += 1
            if isinstance(rt, (int, float)):
                hit_rts.append(rt)
        elif is_target:
            misses += 1
        elif responded:
            false_alarms += 1
        else:
            correct_rejections += 1

    n_targets, n_non_targets = hits + misses, false_alarms + correct_rejections
    d_prime, criterion = _sdt(hits, misses, false_alarms, correct_rejections)
    return {
        "d_prime": d_prime,
        "criterion": criterion,
        "hit_rate": hits / n_targets if n_targets else None,
        "false_alarm_rate": false_alarms / n_non_targets if n_non_targets else None,
        "accuracy": (hits + correct_rejections) / len(req.responses),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_rejections": correct_rejections,
        "mean_hit_rt": float(np.mean(hit_rts)) if hit_rts else None,
        "n_trials": len(req.responses),
        "method": SDT_NOTE,
    }


@router.post("/flanker")
def score_flanker(req: FlankerRequest):
    """Score Eriksen Flanker task — flanker effect, accuracy."""
    e = _rt_effect(req.incongruent_rts, req.incongruent_correct, req.congruent_rts, req.congruent_correct)
    return {
        "flanker_effect_ms": e["effect"],
        "congruent_mean_rt": e["fast_mean"],
        "incongruent_mean_rt": e["slow_mean"],
        "congruent_accuracy": _accuracy(req.congruent_correct),
        "incongruent_accuracy": _accuracy(req.incongruent_correct),
        "n_correct_congruent": e["n_fast"],
        "n_correct_incongruent": e["n_slow"],
        "t_statistic": e["t"],
        "df": e["df"],
        "p_value": e["p"],
        "significant_effect": None if e["p"] is None else bool(e["p"] < 0.05),
        "method": RT_NOTE,
    }


@router.post("/gonogo")
def score_gonogo(req: GoNoGoRequest):
    """Score Go/No-Go task — commission errors, omission errors, d'."""
    if not req.trials:
        raise HTTPException(422, "No trials supplied")
    commission_errors = omission_errors = correct_go = correct_nogo = 0
    go_rts = []
    for trial in req.trials:
        trial_type = trial.get("trial_type", "go")
        if trial_type not in ("go", "nogo"):
            raise HTTPException(422, f"trial_type must be 'go' or 'nogo', got {trial_type!r}")
        responded = bool(trial.get("responded", False))
        rt = trial.get("rt")
        if trial_type == "go":
            if responded:
                correct_go += 1
                if isinstance(rt, (int, float)):
                    go_rts.append(rt)
            else:
                omission_errors += 1
        elif responded:
            commission_errors += 1
        else:
            correct_nogo += 1

    n_go, n_nogo = correct_go + omission_errors, correct_nogo + commission_errors
    d_prime, criterion = _sdt(correct_go, omission_errors, commission_errors, correct_nogo)
    return {
        "commission_errors": commission_errors,
        "omission_errors": omission_errors,
        "commission_rate": commission_errors / n_nogo if n_nogo else None,
        "omission_rate": omission_errors / n_go if n_go else None,
        "d_prime": d_prime,
        "criterion": criterion,
        "mean_go_rt": float(np.mean(go_rts)) if go_rts else None,
        "go_rt_std": float(np.std(go_rts, ddof=1)) if len(go_rts) > 1 else None,
        "accuracy": (correct_go + correct_nogo) / len(req.trials),
        "n_trials": len(req.trials),
        "method": SDT_NOTE,
    }


@router.post("/task-switching")
def score_task_switching(req: TaskSwitchingRequest):
    """Score Task Switching — switch cost (switch − repeat RT on correct trials)."""
    e = _rt_effect(req.switch_rts, req.switch_correct, req.repeat_rts, req.repeat_correct)
    acc_s, acc_r = _accuracy(req.switch_correct), _accuracy(req.repeat_correct)
    return {
        "switch_cost_ms": e["effect"],
        "switch_mean_rt": e["slow_mean"],
        "repeat_mean_rt": e["fast_mean"],
        "switch_accuracy": acc_s,
        "repeat_accuracy": acc_r,
        "accuracy_switch_cost": acc_r - acc_s,
        "n_correct_switch": e["n_slow"],
        "n_correct_repeat": e["n_fast"],
        "t_statistic": e["t"],
        "df": e["df"],
        "p_value": e["p"],
        "significant_cost": None if e["p"] is None else bool(e["p"] < 0.05),
        "method": RT_NOTE + " Mixing cost needs single-task blocks, which this endpoint does not take.",
    }
