"""
Validation: Mann-Whitney U test on the Snedecor & Cochran two-group data.

Reference U and p are frozen from scipy 1.14; R equivalent (wilcox.test)
is documented in references.py.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_mann_whitney_u(client, record_validation):
    response = client.post(
        "/api/advanced/non-parametric",
        json={
            "test": "mann-whitney",
            "group1": ds.SC_TABLE_6_1_1_GROUP_A,
            "group2": ds.SC_TABLE_6_1_1_GROUP_B,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    checks = [
        ("U_statistic", data["U_statistic"], ref.MANN_WHITNEY_U),
        ("p_value",     data["p_value"],     ref.MANN_WHITNEY_P_VALUE),
    ]
    for name, observed, r in checks:
        if r.tolerance_abs:
            assert observed == pytest.approx(r.value, abs=r.tolerance_abs), (
                f"{name}: observed={observed} expected={r.value} ({r.source})"
            )
        else:
            assert observed == pytest.approx(r.value, rel=r.tolerance_rel), (
                f"{name}: observed={observed} expected={r.value} ({r.source})"
            )
        record_validation(
            method="Mann-Whitney U test",
            endpoint="/api/advanced/non-parametric",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="Snedecor & Cochran 7+7 obs (same as indep t-test)",
            tolerance_rel=r.tolerance_rel,
            tolerance_abs=r.tolerance_abs,
        )
