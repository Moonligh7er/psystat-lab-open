"""
Validation: independent-samples t-test (equal variance / pooled).

Reference values are the scipy-computed t and p for a small illustration
dataset. The R equivalent that reproduces the same numbers is documented in
`references.py` — a reviewer with R can independently verify.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_indep_t_test(client, record_validation):
    response = client.post(
        "/api/stats/t-test",
        json={"group1": ds.SC_TABLE_6_1_1_GROUP_A, "group2": ds.SC_TABLE_6_1_1_GROUP_B},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    checks = [
        ("t_statistic", data["t_statistic"], ref.INDEP_T_STATISTIC),
        ("p_value",     data["p_value"],     ref.INDEP_T_P_VALUE),
        ("hedges_g",    data["hedges_g"],    ref.HEDGES_G),
    ]
    for name, observed, r in checks:
        assert observed == pytest.approx(r.value, rel=r.tolerance_rel), (
            f"{name}: observed={observed} expected={r.value} ({r.source})"
        )
        record_validation(
            method="independent-samples t-test",
            endpoint="/api/stats/t-test",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="teaching example (7 + 7 obs)",
            tolerance_rel=r.tolerance_rel,
        )
