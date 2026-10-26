"""
MTMM (/api/mtmm/analyze): Campbell & Fiske (1959) criteria on a matrix whose
answers are worked out by hand, plus the exact payload the Psychometrics screen sends.
"""

import numpy as np
import pytest

from app.routers import mtmm as M

# 3 traits (A, B, C) x 2 methods (1, 2); variable order A1 B1 C1 A2 B2 C2
R = np.array([
    [1.00, 0.40, 0.30, 0.70, 0.20, 0.15],
    [0.40, 1.00, 0.35, 0.25, 0.65, 0.18],
    [0.30, 0.35, 1.00, 0.12, 0.22, 0.60],
    [0.70, 0.25, 0.12, 1.00, 0.38, 0.28],
    [0.20, 0.65, 0.22, 0.38, 1.00, 0.33],
    [0.15, 0.18, 0.60, 0.28, 0.33, 1.00],
])
NAMES = ["A1", "B1", "C1", "A2", "B2", "C2"]


def _req(matrix):
    return M.MTMMAnalyzeRequest(
        correlation_matrix=matrix.tolist(), variable_names=NAMES,
        trait_labels=["A", "B", "C"], method_labels=["1", "2"],
        variable_trait_map={n: n[0] for n in NAMES}, variable_method_map={n: n[1] for n in NAMES},
    )


@pytest.mark.validation
def test_campbell_fiske_hand_worked_example(record_validation):
    c = M.mtmm_analyze(_req(R))["campbell_fiske_criteria"]
    # Validity diagonal .70 / .65 / .60 all exceed .30
    assert c["criterion_1"]["pass"]
    # Each validity value vs the 2 HTHM values in its row and 2 in its column: 3 x 4 = 12
    assert c["criterion_2"]["pass"] and c["criterion_2"]["detail"].startswith("0/12")
    # Each validity value vs the HTMM values of its two variables: 3 x 2 x 2 = 12
    assert c["criterion_3"]["pass"] and c["criterion_3"]["detail"].startswith("0/12")
    # Pattern AB > BC > AC in M1 (.40 .35 .30), M2 (.38 .33 .28) and the heteromethod
    # block (averages .225 .20 .135): identical rank order
    assert c["criterion_4"]["pass"] and c["pattern_mean_spearman"] == 1.0
    record_validation(
        method="MTMM Campbell–Fiske criteria", endpoint="/api/mtmm/analyze",
        statistic="criteria met (hand-worked 3x2 matrix)", observed=float(c["criteria_met"]), expected=4.0,
        reference="Campbell & Fiske (1959), worked by hand", reference_kind="analytical",
        dataset="constructed 3-trait x 2-method matrix", tolerance_abs=0,
    )


def test_criterion_2_counts_only_same_block_row_and_column():
    R2 = R.copy()
    R2[0, 4] = R2[4, 0] = 0.72  # A1-B2 now exceeds validities A1-A2 (.70) and B1-B2 (.65)
    c = M.mtmm_analyze(_req(R2))["campbell_fiske_criteria"]
    assert not c["criterion_2"]["pass"] and c["criterion_2"]["detail"].startswith("2/12")


def test_criterion_3_detects_method_variance():
    R3 = R.copy()
    R3[0, 1] = R3[1, 0] = 0.80  # A1-B1 (same method) exceeds both validities involving A1 or B1
    c = M.mtmm_analyze(_req(R3))["campbell_fiske_criteria"]
    assert not c["criterion_3"]["pass"] and c["criterion_3"]["detail"].startswith("2/12")


def test_criterion_4_not_testable_with_two_traits():
    names = ["v1", "v2", "v3", "v4", "v5", "v6"]
    m = np.array([[1, .7, .55, .6, .45, .35], [.7, 1, .5, .55, .6, .3], [.55, .5, 1, .4, .35, .55],
                  [.6, .55, .4, 1, .65, .5], [.45, .6, .35, .65, 1, .45], [.35, .3, .55, .5, .45, 1]])
    req = M.MTMMAnalyzeRequest(
        correlation_matrix=m.tolist(), variable_names=names, trait_labels=["T1", "T2"], method_labels=["M1", "M2", "M3"],
        variable_trait_map=dict(zip(names, ["T1"] * 3 + ["T2"] * 3)),
        variable_method_map=dict(zip(names, ["M1", "M2", "M3"] * 2)),
    )
    c = M.mtmm_analyze(req)["campbell_fiske_criteria"]
    assert c["criterion_4"]["pass"] is None and c["criteria_testable"] == 3


def test_screen_payload_round_trip(client):
    """Payload built by Psychometrics.tsx from its example inputs; fields its result card reads."""
    matrix = [[1.00, 0.70, 0.55, 0.60, 0.45, 0.35], [0.70, 1.00, 0.50, 0.55, 0.60, 0.30],
              [0.55, 0.50, 1.00, 0.40, 0.35, 0.55], [0.60, 0.55, 0.40, 1.00, 0.65, 0.50],
              [0.45, 0.60, 0.35, 0.65, 1.00, 0.45], [0.35, 0.30, 0.55, 0.50, 0.45, 1.00]]
    mapping = [("v1", "Trait1", "Method1"), ("v2", "Trait1", "Method2"), ("v3", "Trait1", "Method3"),
               ("v4", "Trait2", "Method1"), ("v5", "Trait2", "Method2"), ("v6", "Trait2", "Method3")]
    r = client.post("/api/mtmm/analyze", json={
        "correlation_matrix": matrix,
        "variable_names": [m[0] for m in mapping],
        "trait_labels": ["Trait1", "Trait2"],
        "method_labels": ["Method1", "Method2", "Method3"],
        "variable_trait_map": {m[0]: m[1] for m in mapping},
        "variable_method_map": {m[0]: m[2] for m in mapping},
    })
    assert r.status_code == 200, r.text
    j = r.json()
    cf = j["campbell_fiske_criteria"]
    for k in ("criterion_1", "criterion_2", "criterion_3", "criterion_4"):
        assert {"pass", "description", "detail"} <= set(cf[k])
    assert {"criteria_met", "criteria_testable"} <= set(cf)
    assert all({"label", "n", "mean_r"} <= set(t) for t in j["summary"].values())
    assert all({"var1", "var2", "trait1", "method1", "method2", "r"} <= set(c) for c in j["convergent_validity_coefficients"])
    assert all({"comparison", "ratio", "avg_convergent_r", "avg_method_effect_r"} <= set(d) for d in j["discriminant_validity_ratios"])
