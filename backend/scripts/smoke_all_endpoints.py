"""POST a schema-shaped body to every endpoint and list any that return a 5xx.

Run from backend/: python scripts/smoke_all_endpoints.py
A 4xx is fine (the generated input may be nonsensical); a 5xx is a crash to fix.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("RATE_LIMIT_AUTH", "100000")
os.environ.setdefault("RATE_LIMIT_UNAUTH", "100000")
os.environ.setdefault("RATE_LIMIT_DB", os.path.join(os.environ.get("TEMP", "."), "fuzz_rate.db"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from fastapi.testclient import TestClient
from app.main import app

spec = app.openapi()
comps = spec["components"]["schemas"]
rng = np.random.default_rng(0)
N = 30


def resolve(s):
    while "$ref" in s:
        s = comps[s["$ref"].split("/")[-1]]
    return s


def gen(s, name="", depth=0):
    s = resolve(s)
    if "default" in s and depth > 0 and s["default"] is not None:
        return s["default"]
    if "enum" in s:
        return s["enum"][0]
    if "anyOf" in s:
        opts = [o for o in s["anyOf"] if resolve(o).get("type") != "null"]
        return gen(opts[0], name, depth) if opts else None
    t = s.get("type")
    if t == "object" or "properties" in s:
        props = s.get("properties", {})
        if not props:
            ap = s.get("additionalProperties")
            if isinstance(ap, dict) and ap:
                return {f"v{i}": gen(ap, f"v{i}", depth + 1) for i in range(3)}
            return {}
        return {k: gen(v, k, depth + 1) for k, v in props.items()}
    if t == "array":
        items = resolve(s.get("items", {}))
        if items.get("type") == "array":
            inner = resolve(items.get("items", {}))
            if inner.get("type") in ("number", "integer"):
                if "corr" in name or "matrix" in name and "data" not in name:
                    k = 4
                    A = rng.normal(size=(50, k)); R = np.corrcoef(A.T)
                    return R.round(4).tolist()
                return (rng.normal(size=(N, 4)).round(3) if inner.get("type") == "number" else rng.integers(0, 2, size=(N, 4))).tolist()
            return [gen(items, name, depth + 1) for _ in range(4)]
        if items.get("type") == "number":
            return rng.normal(10, 2, N).round(3).tolist()
        if items.get("type") == "integer":
            return rng.integers(0, 3, N).tolist()
        if items.get("type") == "string":
            return [f"{name}_{i}" for i in range(4)]
        return [gen(items, name, depth + 1) for _ in range(3)]
    if t == "number":
        lo, hi = s.get("minimum", s.get("exclusiveMinimum")), s.get("maximum", s.get("exclusiveMaximum"))
        if lo is not None and hi is not None:
            return (lo + hi) / 2
        if any(w in name for w in ("alpha", "prob", "rate", "power", "level", "prior")):
            return 0.05 if "alpha" in name else 0.5
        return 0.5
    if t == "integer":
        return s.get("minimum", 2) if s.get("minimum", 0) > 2 else 3
    if t == "boolean":
        return False
    if t == "string":
        return "test"
    return None


def run_sweep():
    """Return [(path, status, detail)] for every POST endpoint that answers with a 5xx."""
    client = TestClient(app, raise_server_exceptions=False)
    bad = []
    for path, ops in spec["paths"].items():
        op = ops.get("post")
        if not op or "{" in path:
            continue
        if any(x in path for x in ("payments", "account", "ai-assistant", "ai/", "/ai", "export", "webhook")):
            continue
        sch = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
        body = gen(sch) if sch else None
        try:
            r = client.post(path, json=body)
            code, detail = r.status_code, r.text[:200]
        except Exception as e:  # noqa
            code, detail = 599, repr(e)[:200]
        if code >= 500:
            bad.append((path, code, detail))
    return bad


if __name__ == "__main__":
    bad = run_sweep()
    print(f"{len(bad)} endpoints returned >= 500")
    for b in bad:
        print(b)
