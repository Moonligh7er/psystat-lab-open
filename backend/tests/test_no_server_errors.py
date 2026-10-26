"""Every POST endpoint must answer schema-shaped (often nonsensical) input with a 2xx or a
4xx explanation, never a 5xx crash. See scripts/smoke_all_endpoints.py."""

import importlib.util
import os

import pytest


@pytest.mark.smoke
def test_no_endpoint_returns_5xx():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "smoke_all_endpoints.py")
    spec = importlib.util.spec_from_file_location("smoke_all_endpoints", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bad = mod.run_sweep()
    assert not bad, "\n".join(f"{p} -> {c}: {d}" for p, c, d in bad)
