"""
Validation: chi-square test of independence on a 2x2 contingency table.

The odds ratio and Cramer's V are analytically closed-form for a 2x2, so
those checks are ironclad. Chi² and p come from scipy 1.14 with the R
equivalent (chisq.test correct=FALSE) documented in references.py.
"""

from __future__ import annotations

import pytest

from tests.validation import datasets as ds
from tests.validation import references as ref


@pytest.mark.validation
def test_chi_square_2x2(client, record_validation):
    response = client.post(
        "/api/stats/chi-square",
        json={"observed": ds.ROSNER_2X2_CONTINGENCY},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    checks = [
        ("chi2_statistic", data["chi2_statistic"], ref.CHI2_STATISTIC),
        ("p_value",        data["p_value"],        ref.CHI2_P_VALUE),
        ("cramers_v",      data["cramers_v"],      ref.CHI2_CRAMERS_V),
        ("odds_ratio",     data["odds_ratio"],     ref.CHI2_ODDS_RATIO),
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
            method="chi-square test of independence",
            endpoint="/api/stats/chi-square",
            statistic=name,
            observed=float(observed),
            expected=float(r.value),
            reference=r.source,
            reference_kind=r.kind,
            citation=r.citation,
            dataset="Rosner 2x2 contingency [[40,10],[20,30]]",
            tolerance_rel=r.tolerance_rel,
            tolerance_abs=r.tolerance_abs,
        )
