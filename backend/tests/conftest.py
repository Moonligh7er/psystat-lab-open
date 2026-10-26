"""
Shared pytest fixtures + the PsyStat validation reporter.

The reporter accumulates results from every test marked `@pytest.mark.validation`
and writes `backend/validation-results.json` at session end. This JSON is the
data source for the public /validation dashboard (see landing site) and for the
`methods_validated / methods_total` fields on `/api/version`.

Each validation test should attach a `validation_record` dict via the
`record_validation` fixture, so the reporter has enough metadata to render a
row in the dashboard.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Point the rate limiter at a fresh per-session SQLite DB *before* importing
# app.main, which initializes the DB at import time. Without this, the smoke
# suite's rapid TestClient requests trip the production rate limit and
# subsequent runs cascade-fail because the DB persists in /tmp.
os.environ.setdefault(
    "RATE_LIMIT_DB",
    os.path.join(tempfile.gettempdir(), f"psystat_test_rate_{os.getpid()}.db"),
)
os.environ.setdefault("RATE_LIMIT_AUTH", "10000")
os.environ.setdefault("RATE_LIMIT_UNAUTH", "10000")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient shared across the session."""
    return TestClient(app)


# ── Validation reporter ────────────────────────────────────────────────────
#
# Collects records from each validation test. Records are appended via the
# `record_validation` fixture. At session end, we write a JSON report.

_VALIDATION_RECORDS: list[dict[str, Any]] = []


@pytest.fixture
def record_validation(request):
    """
    Fixture used inside @pytest.mark.validation tests to record what was
    validated. Usage:

        def test_something(client, record_validation):
            record_validation(
                method="independent-samples t-test",
                endpoint="/api/stats/t-test",
                reference="Snedecor & Cochran (1989) Table 6.1.1",
                reference_kind="textbook",  # or "nist" | "r" | "analytical"
                dataset="tomato_yield_5x5",
                statistic="t",
                observed=2.1234,
                expected=2.1234,
                tolerance_abs=1e-10,
            )
    """

    def _record(**fields):
        record = {
            "test_id": request.node.nodeid,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        _VALIDATION_RECORDS.append(record)
        return record

    return _record


def pytest_sessionstart(session):
    _VALIDATION_RECORDS.clear()


def pytest_sessionfinish(session, exitstatus):
    """Write validation-results.json at the repo root of backend/."""
    if not _VALIDATION_RECORDS:
        return

    # Group by (method, endpoint) so /validation renders one row per method.
    by_method: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _VALIDATION_RECORDS:
        key = (r.get("method", "?"), r.get("endpoint", "?"))
        entry = by_method.setdefault(
            key,
            {
                "method": r.get("method"),
                "endpoint": r.get("endpoint"),
                "checks": [],
                "references": set(),
                "reference_kinds": set(),
                "datasets": set(),
            },
        )
        entry["checks"].append(
            {
                "statistic": r.get("statistic"),
                "observed": r.get("observed"),
                "expected": r.get("expected"),
                "tolerance_abs": r.get("tolerance_abs"),
                "tolerance_rel": r.get("tolerance_rel"),
                "dataset": r.get("dataset"),
                "test_id": r.get("test_id"),
            }
        )
        if r.get("reference"):
            entry["references"].add(r["reference"])
        if r.get("reference_kind"):
            entry["reference_kinds"].add(r["reference_kind"])
        if r.get("dataset"):
            entry["datasets"].add(r["dataset"])

    methods = []
    for (_, _), entry in by_method.items():
        entry["references"] = sorted(entry["references"])
        entry["reference_kinds"] = sorted(entry["reference_kinds"])
        entry["datasets"] = sorted(entry["datasets"])
        entry["n_checks"] = len(entry["checks"])
        methods.append(entry)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exit_status": exitstatus,
        "n_methods_validated": len(methods),
        "n_total_checks": len(_VALIDATION_RECORDS),
        "methods": sorted(methods, key=lambda m: m["method"] or ""),
    }

    out = Path(__file__).resolve().parent.parent / "validation-results.json"
    out.write_text(json.dumps(report, indent=2, default=str))
