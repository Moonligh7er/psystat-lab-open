"""Export the FastAPI OpenAPI schema so the app can generate request/response types from it.

Usage (from backend/):  python scripts/export_openapi.py ../psystat-app/types/openapi.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SUPABASE_JWT_SECRET", "schema-export")

from app.main import app  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
schema = app.openapi()
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(schema, f, indent=1, sort_keys=True, ensure_ascii=False)
    f.write("\n")
print(f"wrote {out}: {len(schema.get('paths', {}))} paths")
