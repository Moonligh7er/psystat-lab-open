"""Crash reporting with Sentry. Off unless SENTRY_DSN is set.

Requests here carry research data and AI conversations, so Sentry gets the error, its stack
trace and the route only: no request bodies, cookies, headers beyond the safe defaults, query
values, user identity, AI inputs/outputs or local variable values (a stack frame's locals can
hold a participant's data). No performance tracing.
"""
import os

import sentry_sdk

_OFF = {"mode": "off"}


def _scrub(event, _hint):
    """Second line of defence: drop anything request- or user-shaped that slipped through."""
    req = event.get("request")
    if isinstance(req, dict):
        for key in ("data", "cookies", "query_string", "env"):
            req.pop(key, None)
    event.pop("user", None)
    return event


def init_sentry() -> bool:
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("RAILWAY_ENVIRONMENT_NAME") or "development",
        release=os.environ.get("RAILWAY_GIT_COMMIT_SHA") or None,
        traces_sample_rate=0.0,
        max_request_body_size="never",
        data_collection={
            "user_info": False,
            "cookies": _OFF,
            "http_bodies": [],
            "url_query_params": _OFF,
            "gen_ai": {"inputs": False, "outputs": False},
            "database_query_data": False,
            "stack_frame_variables": False,
        },
        before_send=_scrub,
    )
    return True


def capture(exc: BaseException, ref: str | None = None) -> None:
    """Report an exception the app handled itself (the global 500 handler)."""
    if sentry_sdk.get_client().is_active():
        with sentry_sdk.new_scope() as scope:
            if ref:
                scope.set_tag("error_ref", ref)
            sentry_sdk.capture_exception(exc)
