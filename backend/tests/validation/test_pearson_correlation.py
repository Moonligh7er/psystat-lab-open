"""
Validation: Pearson correlation via /api/advanced-stats/correlation-matrix.

Reference r and p are from scipy 1.14 on a monotone-shaped height/weight
sample. R equivalent (cor.test method="pearson") is in `references.py`.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_pearson_correlation(client, record_validation):
    response = client.post(
        "/api/advanced/correlation-matrix",
        json={
            "variables": [ds.ZAR_HEIGHT, ds.ZAR_WEIGHT],
            "labels": ["height", "weight"],
            "method": "pearson",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    r_observed = data["correlation_matrix"][0][1]
    p_observed = data["p_value_matrix"][0][1]

    checks = [
        ("r",       r_observed, ref.PEARSON_R),
        ("p_value", p_observed, ref.PEARSON_P_VALUE),
    ]
    for name, observed, r in checks:
        assert observed == pytest.approx(r.value, rel=r.tolerance_rel), (
            f"{name}: observed={observed} expected={r.value} ({r.source})"
        )
        record_validation(
            method="Pearson product-moment correlation",
            endpoint="/api/advanced/correlation-matrix",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="12-pair monotone height/weight teaching example",
            tolerance_rel=r.tolerance_rel,
        )
