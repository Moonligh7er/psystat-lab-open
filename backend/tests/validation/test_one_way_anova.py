"""
Validation: one-way ANOVA on a 3-group × 5-observation illustration.

Reference F and p are from scipy 1.14; eta² is analytical.
R equivalent that reproduces the same F/p is documented in `references.py`.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_one_way_anova(client, record_validation):
    response = client.post(
        "/api/stats/anova",
        json={"groups": [ds.SC_TOMATO_GROUP_A, ds.SC_TOMATO_GROUP_B, ds.SC_TOMATO_GROUP_C]},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    checks = [
        ("f_statistic", data["f_statistic"], ref.ANOVA_F_STATISTIC),
        ("p_value",     data["p_value"],     ref.ANOVA_P_VALUE),
        ("eta_squared", data["eta_squared"], ref.ANOVA_ETA_SQUARED),
    ]
    for name, observed, r in checks:
        assert observed == pytest.approx(r.value, rel=r.tolerance_rel), (
            f"{name}: observed={observed} expected={r.value} ({r.source})"
        )
        record_validation(
            method="one-way ANOVA",
            endpoint="/api/stats/anova",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="3 groups × 5 obs teaching example",
            tolerance_rel=r.tolerance_rel,
        )
