"""
Validation: Wilcoxon signed-rank test on the Rosner pre/post BP data.

Reference W and p are frozen from scipy 1.14 (exact-tail probability);
R equivalent (wilcox.test paired=TRUE, exact=TRUE) is documented in
references.py.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_wilcoxon_signed_rank(client, record_validation):
    response = client.post(
        "/api/advanced/non-parametric",
        json={
            "test": "wilcoxon",
            "group1": ds.ROSNER_PRE,
            "group2": ds.ROSNER_POST,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    checks = [
        ("W_statistic", data["W_statistic"], ref.WILCOXON_W),
        ("p_value",     data["p_value"],     ref.WILCOXON_P_VALUE),
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
            method="Wilcoxon signed-rank test",
            endpoint="/api/advanced/non-parametric",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="Rosner (2016) Ex. 8.14 pre/post BP",
            tolerance_rel=r.tolerance_rel,
            tolerance_abs=r.tolerance_abs,
        )
