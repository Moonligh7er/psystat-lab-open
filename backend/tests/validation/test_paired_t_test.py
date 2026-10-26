"""
Validation: paired-samples t-test (Rosner 2016 Ex. 8.14, pre/post BP).

Reference for t and p is scipy 1.14; the mean difference is analytical.
R equivalent that reproduces the numbers is in `references.py`.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_paired_t_test(client, record_validation):
    response = client.post(
        "/api/stats/paired-t-test",
        json={"before": ds.ROSNER_PRE, "after": ds.ROSNER_POST},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    checks = [
        ("t_statistic",     data["t_statistic"],     ref.PAIRED_T_STATISTIC),
        ("p_value",         data["p_value"],         ref.PAIRED_T_P_VALUE),
        ("mean_difference", data["mean_difference"], ref.PAIRED_MEAN_DIFFERENCE),
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
            method="paired-samples t-test",
            endpoint="/api/stats/paired-t-test",
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
