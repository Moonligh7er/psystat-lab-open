"""Sentry crash reporting: off without a DSN, and when on it never sends research data.

A crash during an analysis must reach Sentry as an error with a stack trace, but the request
body (participants' data), query values, cookies, user identity and the values of local
variables must not be in the event.
"""
import json

import pytest
import sentry_sdk
from fastapi import APIRouter
from fastapi.testclient import TestClient

from app import monitoring
from app.main import app

SECRET_VALUE = "participant-score-98765"

CRASH_PATH = "/api/_test_sentry_crash"


async def _crash(payload: dict):
    participant_scores = payload.get("scores")  # a local that must not be reported
    raise RuntimeError(f"boom with {len(participant_scores or [])} scores")


class _Capture:
    def __init__(self):
        self.envelopes = []


@pytest.fixture
def crash_route():
    """A route that always fails, present only for the duration of one test."""
    router = APIRouter()
    router.add_api_route(CRASH_PATH, _crash, methods=["POST"])
    before = list(app.router.routes)
    app.include_router(router)
    app.openapi_schema = None
    yield
    app.router.routes[:] = before
    app.openapi_schema = None


@pytest.fixture
def sentry_events(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://public@o1.ingest.sentry.io/1")
    assert monitoring.init_sentry()
    cap = _Capture()
    # Keep the real transport (the SDK reads its DSN) but never let it send anything.
    monkeypatch.setattr(sentry_sdk.get_client().transport, "capture_envelope", cap.envelopes.append)
    yield cap
    # Back to a disabled client. The env var must go first: init() without a dsn reads it.
    monkeypatch.delenv("SENTRY_DSN")
    sentry_sdk.init()
    assert sentry_sdk.get_client().transport is None  # nothing can be sent


def _events(cap):
    out = []
    for env in cap.envelopes:
        for item in env.items:
            if item.type in ("event", "error"):
                out.append(item.payload.json)
    return out


def test_off_without_dsn(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert monitoring.init_sentry() is False


def test_crash_is_reported_without_research_data(crash_route, sentry_events):
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(CRASH_PATH + "?subject=" + SECRET_VALUE,
                    json={"scores": [SECRET_VALUE, 1, 2]},
                    headers={"Cookie": "sb=" + SECRET_VALUE})
    assert r.status_code == 500
    assert SECRET_VALUE not in r.text

    events = _events(sentry_events)
    assert events, "the crash was not reported"
    ev = events[0]
    assert ev["exception"]["values"][-1]["type"] == "RuntimeError"
    assert ev.get("tags", {}).get("error_ref")
    dumped = json.dumps(events)
    assert SECRET_VALUE not in dumped
    assert "participant_scores" not in json.dumps([
        f.get("vars", {}) for v in ev["exception"]["values"]
        for f in (v.get("stacktrace") or {}).get("frames", [])
    ])
    assert "user" not in ev
