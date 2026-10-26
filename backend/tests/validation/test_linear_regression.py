"""
Validation: simple linear regression against NIST StRD Norris.

Norris is the canonical linear-regression benchmark. NIST certifies the
regression coefficients, residual standard deviation, and R-squared to 15
significant figures.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_norris_regression(client, record_validation):
    """Endpoint /api/stats/regression must reproduce NIST Norris certified values."""
    response = client.post(
        "/api/stats/regression",
        json={
            "x": [[x] for x in ds.NORRIS_X],
            "y": ds.NORRIS_Y,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    intercept, slope = data["coefficients"]
    residual_std = data["residual_std_error"]
    r_squared = data["r_squared"]

    checks = [
        ("intercept",     intercept,    ref.NORRIS_INTERCEPT),
        ("slope",         slope,        ref.NORRIS_SLOPE),
        ("residual_std",  residual_std, ref.NORRIS_RESIDUAL_STD),
        ("r_squared",     r_squared,    ref.NORRIS_R_SQUARED),
    ]

    for name, observed, r in checks:
        assert observed == pytest.approx(r.value, rel=r.tolerance_rel), (
            f"{name}: observed={observed} expected={r.value} ({r.source})"
        )
        record_validation(
            method="simple linear regression",
            endpoint="/api/stats/regression",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="NIST StRD: Norris",
            tolerance_rel=r.tolerance_rel,
        )
