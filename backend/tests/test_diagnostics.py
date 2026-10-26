"""Diagnostics (/api/diagnostics/*). Grubbs reference: R outliers::grubbs.test(x, two.sided = TRUE)."""

import numpy as np
import pytest

from app.routers import diagnostics as D

X = [12, 15, 11, 19, 14, 16, 13, 18, 17, 10, 25, 9]


@pytest.mark.validation
def test_grubbs_matches_r_outliers(record_validation):
    g = D._grubbs_test(np.array(X, float))
    assert g["statistic"] == pytest.approx(2.2500192100, abs=1e-6)
    assert g["p_value"] == pytest.approx(0.118752171, abs=1e-8)
    record_validation(method="Grubbs outlier test (two-sided)", endpoint="/api/diagnostics/outlier-diagnostics",
                      statistic="p", observed=float(g["p_value"]), expected=0.118752171,
                      reference="R outliers::grubbs.test", reference_kind="r", dataset="12 values", tolerance_abs=1e-8)


def test_bad_inputs_are_422(client):
    assert client.post("/api/diagnostics/residual-diagnostics", json={"dv": "y", "ivs": ["nope"], "data": {"y": [1, 2, 3]}}).status_code == 422
    assert client.post("/api/diagnostics/qq-plot", json={"data": [1, 2, 3, 4], "distribution": "linregress"}).status_code == 422
    assert client.post("/api/diagnostics/outlier-diagnostics", json={"columns": ["z"], "data": {"a": [1, 2]}}).status_code == 422
