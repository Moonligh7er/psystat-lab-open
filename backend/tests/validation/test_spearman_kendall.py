"""
Validation: Spearman rho and Kendall tau via /api/advanced/non-parametric.

Both statistics are computed from ranks and are analytically 1.0 (up to
floating-point precision) on the strictly-monotone Zar height/weight data.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_spearman(client, record_validation):
    response = client.post(
        "/api/advanced/non-parametric",
        json={
            "test": "spearman",
            "group1": ds.ZAR_HEIGHT,
            "group2": ds.ZAR_WEIGHT,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    r = ref.SPEARMAN_RHO
    assert data["statistic"] == pytest.approx(r.value, abs=r.tolerance_abs)
    record_validation(
        method="Spearman rank correlation",
        endpoint="/api/advanced/non-parametric",
        statistic="rho",
        observed=float(data["statistic"]),
        expected=float(r.value),
        reference=r.source,
        reference_kind=r.kind,
        citation=r.citation,
        dataset="Zar height/weight (strictly monotone)",
        tolerance_abs=r.tolerance_abs,
    )


@pytest.mark.validation
def test_kendall(client, record_validation):
    response = client.post(
        "/api/advanced/non-parametric",
        json={
            "test": "kendall",
            "group1": ds.ZAR_HEIGHT,
            "group2": ds.ZAR_WEIGHT,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    r = ref.KENDALL_TAU
    assert data["statistic"] == pytest.approx(r.value, abs=r.tolerance_abs)
    record_validation(
        method="Kendall tau",
        endpoint="/api/advanced/non-parametric",
        statistic="tau",
        observed=float(data["statistic"]),
        expected=float(r.value),
        reference=r.source,
        reference_kind=r.kind,
        citation=r.citation,
        dataset="Zar height/weight (strictly monotone)",
        tolerance_abs=r.tolerance_abs,
    )
