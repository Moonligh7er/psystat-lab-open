import json
import os
import time
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import JSONResponse as StarletteJSONResponse

from fastapi import encoders as _fastapi_encoders

# Routers without a response_model are serialised by FastAPI's jsonable_encoder
# before the response class sees them, and it rejects numpy scalars that are not
# Python subclasses (np.bool_, np.int64, np.float32, ...). Register them once here
# so a stray numpy value can never turn a result into a 500.
_fastapi_encoders.ENCODERS_BY_TYPE.update({
    np.bool_: bool,
    **{t: int for t in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)},
    **{t: float for t in (np.float16, np.float32)},
    np.ndarray: lambda a: a.tolist(),
})


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def sanitize(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if v != v or v in (float("inf"), float("-inf")):  # NaN / ±inf are not valid JSON
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


class NumpySafeResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(sanitize(content), cls=NumpyEncoder).encode("utf-8")


from app.routers import (
    stats, psychometrics, bayesian, genomic, medical, social_graph,
    cognitive, study_planner, advanced_stats, data_cleaning,
    apa_formatter, data_import, assumptions, mice, bayesian_advanced, sem, reports,
    nonparametric, extended_stats, smart_recommend, missingness,
    criminology, biometrics, hlm, mediation, advanced_anova, sociology,
    political_science, ethnography_backend, nl_pipeline, export_engine,
    meta_analysis, power_analysis, latent_class, time_series,
    multilevel_sem, measurement_invariance, irt_advanced,
    causal_inference, text_analysis, ml_module, effect_size,
    pipeline, assumption_gate, multiverse, simulator,
    diagnostics, dual_report, preregistration,
    novel_cca, novel_complexity, novel_dialectical, novel_spf,
    ema, lta, mtmm, q_methodology,
    propensity_score, signal_detection, generalizability, conjoint,
    network_psychometrics, dcm,
    ergm, spatial_stats, correspondence,
    narrative, equivalence, sensitivity_decisions,
    reviewer2, triangulation, what_would_it_take,
    interpretation_confidence, contradiction,
)

PSYSTAT_VERSION = "0.2.0-beta.1"

app = FastAPI(
    default_response_class=NumpySafeResponse,
    title="PsyStat Lab API",
    description="AI-powered behavioral science statistics and research backend",
    version=PSYSTAT_VERSION,
)

# ── CORS — restricted to known origins ────────────────────────────────
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",") if os.environ.get("ALLOWED_ORIGINS") else [
    "https://psystat-lab.dev",
    "https://www.psystat-lab.dev",
    # Old vercel.app domain kept as a transition fallback so anything
    # cached at the old URL keeps working while DNS + 301 settle.
    # Remove once analytics show zero traffic on the vercel.app origin.
    "https://psystat-nexus.vercel.app",
    "https://www.psystat-nexus.vercel.app",
    "http://localhost:8081",
    "http://localhost:19006",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Global exception handler — guarantees CORS headers on 500 responses ──
# Without this, an unhandled exception returns no CORS header and the browser
# shows a misleading "blocked by CORS policy" message instead of the real error.
def _origin_for(request: Request) -> str:
    origin = request.headers.get("origin", "")
    if origin and origin in ALLOWED_ORIGINS:
        return origin
    return ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "*"


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    origin = _origin_for(request)
    return StarletteJSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)[:500]},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        },
    )


# ── Auth middleware — verifies Supabase JWT on protected routes ────────

# ── Rate limiter — SQLite-backed for persistence across deploys ────────
import sqlite3
import threading
import math

_db_lock = threading.Lock()
_DB_PATH = os.environ.get("RATE_LIMIT_DB", "/tmp/psystat_rate_limits.db")

def _init_rate_db():
    with _db_lock:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                key TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_key ON rate_limits(key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_ts ON rate_limits(timestamp)")
        conn.commit()
        conn.close()

_init_rate_db()

RATE_LIMIT_AUTH = int(os.environ.get("RATE_LIMIT_AUTH", "60"))       # Authenticated: 60 req/min
RATE_LIMIT_UNAUTH = int(os.environ.get("RATE_LIMIT_UNAUTH", "20"))  # Unauthenticated: 20 req/min (protect compute)
RATE_WINDOW = int(os.environ.get("RATE_WINDOW", "60"))              # seconds
MAX_REQUEST_BODY = 10 * 1024 * 1024  # 10 MB
MAX_ARRAY_ITEMS = 100_000   # Max items in any single array in request body
MAX_NESTING_DEPTH = 10      # Max JSON nesting depth


def _check_rate_limit(key: str, limit: int) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    now = time.time()
    with _db_lock:
        conn = sqlite3.connect(_DB_PATH)
        try:
            conn.execute("DELETE FROM rate_limits WHERE key = ? AND timestamp < ?", (key, now - RATE_WINDOW))
            row = conn.execute("SELECT COUNT(*) FROM rate_limits WHERE key = ?", (key,)).fetchone()
            count = row[0] if row else 0
            if count >= limit:
                return False
            conn.execute("INSERT INTO rate_limits (key, timestamp) VALUES (?, ?)", (key, now))
            conn.commit()
            if count == 0:
                conn.execute("DELETE FROM rate_limits WHERE timestamp < ?", (now - RATE_WINDOW * 2,))
                conn.commit()
        finally:
            conn.close()
    return True


def _validate_json_depth_and_arrays(obj, depth=0):
    """Recursively validate JSON: check nesting depth and array sizes."""
    if depth > MAX_NESTING_DEPTH:
        return "Request data is too deeply nested"
    if isinstance(obj, list):
        if len(obj) > MAX_ARRAY_ITEMS:
            return f"Array too large ({len(obj)} items). Maximum {MAX_ARRAY_ITEMS}."
        for item in obj:
            err = _validate_json_depth_and_arrays(item, depth + 1)
            if err:
                return err
    elif isinstance(obj, dict):
        for v in obj.values():
            err = _validate_json_depth_and_arrays(v, depth + 1)
            if err:
                return err
    return None


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path

    # Skip health checks
    if path == "/health":
        return await call_next(request)

    # ── Request size limit ──
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY:
        return StarletteJSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Maximum 10 MB."},
        )

    # ── HTTPS enforcement ──
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if forwarded_proto == "http" and os.environ.get("ENFORCE_HTTPS", ""):
        https_url = str(request.url).replace("http://", "https://", 1)
        return StarletteJSONResponse(
            status_code=301,
            content={"detail": "HTTPS required"},
            headers={"Location": https_url},
        )

    # ── Global input validation (arrays/nesting in request body) ──
    if request.method == "POST" and content_length and int(content_length) > 0:
        try:
            body_bytes = await request.body()
            if body_bytes:
                import json as _json
                try:
                    parsed = _json.loads(body_bytes)
                    err = _validate_json_depth_and_arrays(parsed)
                    if err:
                        return StarletteJSONResponse(
                            status_code=400,
                            content={"detail": err},
                        )
                except _json.JSONDecodeError:
                    pass  # Let FastAPI handle malformed JSON

                # Re-inject body so downstream can read it
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request._receive = receive
        except Exception:
            pass

    # ── Tiered rate limiting ──
    # Extract user_id directly from JWT (don't rely on request.state which
    # may not be set yet depending on middleware execution order)
    user_id = None
    auth_header = request.headers.get("authorization", "")
    is_authenticated = bool(user_id)

    if is_authenticated:
        rate_key = f"user:{user_id}"
        limit = RATE_LIMIT_AUTH
    else:
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"ip:{client_ip}"
        limit = RATE_LIMIT_UNAUTH

    if not _check_rate_limit(rate_key, limit):
        return StarletteJSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded ({limit} requests/minute). Please try again shortly."},
        )

    return await call_next(request)


app.include_router(stats.router, prefix="/api/stats", tags=["Stats Studio"])
app.include_router(psychometrics.router, prefix="/api/psychometrics", tags=["Psychometrics"])
app.include_router(bayesian.router, prefix="/api/bayesian", tags=["Bayesian Lab"])
app.include_router(genomic.router, prefix="/api/genomic", tags=["Genomics"])
app.include_router(medical.router, prefix="/api/medical", tags=["Medical Stats"])
app.include_router(social_graph.router, prefix="/api/social-graph", tags=["Social Graph"])
app.include_router(cognitive.router, prefix="/api/cognitive", tags=["Cognitive Lab"])
app.include_router(study_planner.router, prefix="/api/study-planner", tags=["Study Planner"])
app.include_router(advanced_stats.router, prefix="/api/advanced", tags=["Advanced Stats"])
app.include_router(data_cleaning.router, prefix="/api/data", tags=["Data Cleaning"])
app.include_router(apa_formatter.router, prefix="/api/apa", tags=["APA Formatter"])
app.include_router(data_import.router, prefix="/api/import", tags=["Data Import"])
app.include_router(assumptions.router, prefix="/api/assumptions", tags=["Assumption Checking"])
app.include_router(mice.router, prefix="/api/mice", tags=["MICE Imputation"])
app.include_router(bayesian_advanced.router, prefix="/api/bayesian-advanced", tags=["Bayesian Advanced"])
app.include_router(sem.router, prefix="/api/sem", tags=["SEM"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(nonparametric.router, prefix="/api/nonparametric", tags=["Non-Parametric"])
app.include_router(extended_stats.router, prefix="/api/extended-stats", tags=["Extended Stats"])
app.include_router(smart_recommend.router, prefix="/api/recommend", tags=["recommendation"])
app.include_router(missingness.router, prefix="/api/missingness", tags=["missingness"])
app.include_router(criminology.router, prefix="/api/criminology", tags=["Criminology"])
app.include_router(biometrics.router, prefix="/api/biometrics", tags=["BioMetrics"])
app.include_router(hlm.router, prefix="/api/hlm", tags=["HLM"])
app.include_router(mediation.router, prefix="/api/mediation", tags=["Mediation & Moderation"])
app.include_router(advanced_anova.router, prefix="/api/advanced-anova", tags=["Advanced ANOVA"])
app.include_router(sociology.router, prefix="/api/sociology", tags=["Sociology"])
app.include_router(political_science.router, prefix="/api/political-science", tags=["Political Science"])
app.include_router(ethnography_backend.router, prefix="/api/ethnography", tags=["Ethnography"])
app.include_router(nl_pipeline.router, prefix="/api/nl-pipeline", tags=["NL Pipeline"])
app.include_router(export_engine.router, prefix="/api/export", tags=["Export Engine"])
app.include_router(meta_analysis.router, prefix="/api/meta-analysis", tags=["Meta-Analysis"])
app.include_router(power_analysis.router, prefix="/api/power", tags=["Power Analysis"])
app.include_router(latent_class.router, prefix="/api/latent-class", tags=["Latent Class"])
app.include_router(time_series.router, prefix="/api/time-series", tags=["Time Series"])
app.include_router(multilevel_sem.router, prefix="/api/multilevel-sem", tags=["Multilevel SEM"])
app.include_router(measurement_invariance.router, prefix="/api/invariance", tags=["Measurement Invariance"])
app.include_router(irt_advanced.router, prefix="/api/irt", tags=["IRT Advanced"])
app.include_router(causal_inference.router, prefix="/api/causal", tags=["Causal Inference"])
app.include_router(text_analysis.router, prefix="/api/text-analysis", tags=["Text Analysis"])
app.include_router(ml_module.router, prefix="/api/ml", tags=["Machine Learning"])
app.include_router(effect_size.router, prefix="/api/effect-size", tags=["Effect Size"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline Builder"])
app.include_router(assumption_gate.router, prefix="/api/assumption-gate", tags=["Assumption Gate"])
app.include_router(multiverse.router, prefix="/api/multiverse", tags=["Multiverse Analysis"])
app.include_router(simulator.router, prefix="/api/simulator", tags=["Data Simulator"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["Diagnostics"])
app.include_router(dual_report.router, prefix="/api/dual-report", tags=["Dual Report"])
app.include_router(preregistration.router, prefix="/api/preregistration", tags=["Pre-Registration"])
app.include_router(novel_cca.router, prefix="/api/novel/cca", tags=["Novel: CCA"])
app.include_router(novel_complexity.router, prefix="/api/novel/complexity", tags=["Novel: Complexity Navigator"])
app.include_router(novel_dialectical.router, prefix="/api/novel/dialectical", tags=["Novel: Dialectical Inference"])
app.include_router(novel_spf.router, prefix="/api/novel/spf", tags=["Novel: Scale-Persistent Features"])
app.include_router(ema.router, prefix="/api/ema", tags=["EMA / ESM"])
app.include_router(lta.router, prefix="/api/lta", tags=["Latent Transition Analysis"])
app.include_router(mtmm.router, prefix="/api/mtmm", tags=["MTMM"])
app.include_router(q_methodology.router, prefix="/api/q-methodology", tags=["Q-Methodology"])
app.include_router(propensity_score.router, prefix="/api/propensity-score", tags=["Propensity Score Matching"])
app.include_router(signal_detection.router, prefix="/api/signal-detection", tags=["Signal Detection Theory"])
app.include_router(generalizability.router, prefix="/api/generalizability", tags=["Generalizability Theory"])
app.include_router(conjoint.router, prefix="/api/conjoint", tags=["Conjoint Analysis"])
app.include_router(network_psychometrics.router, prefix="/api/network-psychometrics", tags=["Network Psychometrics"])
app.include_router(dcm.router, prefix="/api/dcm", tags=["Diagnostic Classification Models"])
app.include_router(ergm.router, prefix="/api/ergm", tags=["ERGM"])
app.include_router(spatial_stats.router, prefix="/api/spatial", tags=["Spatial Statistics"])
app.include_router(correspondence.router, prefix="/api/correspondence", tags=["Correspondence Analysis"])
app.include_router(narrative.router, prefix="/api/narrative", tags=["Research Narrative"])
app.include_router(equivalence.router, prefix="/api/equivalence", tags=["Equivalence Testing"])
app.include_router(sensitivity_decisions.router, prefix="/api/sensitivity-decisions", tags=["Sensitivity Decisions"])
app.include_router(reviewer2.router, prefix="/api/reviewer2", tags=["Adversarial Reviewer"])
app.include_router(triangulation.router, prefix="/api/triangulation", tags=["Method Triangulation"])
app.include_router(what_would_it_take.router, prefix="/api/what-would-it-take", tags=["Sensitivity Calculator"])
app.include_router(interpretation_confidence.router, prefix="/api/interpretation-confidence", tags=["Interpretation Confidence"])
app.include_router(contradiction.router, prefix="/api/contradiction", tags=["Contradiction Detector"])


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "psystat-nexus"}


@app.get("/api/version")
def version_info():
    """
    Reports the running backend version and (if the validation suite has
    run in this deployment) a summary of numerical-accuracy coverage. This
    endpoint is what the /validation page and the in-app Citation Generator
    consume to stamp the exact version used to produce a given result.

    The validation-results.json file is written by pytest at CI time and
    baked into the container image; if it is absent (e.g. a dev machine
    that has not run the suite), the summary fields are null.
    """
    import json
    import os

    summary = {
        "version": PSYSTAT_VERSION,
        "scipy": None,
        "statsmodels": None,
        "numpy": None,
        "methods_validated": None,
        "total_checks": None,
        "validation_generated_at": None,
    }
    try:
        import scipy, statsmodels, numpy
        summary["scipy"] = scipy.__version__
        summary["statsmodels"] = statsmodels.__version__
        summary["numpy"] = numpy.__version__
    except Exception:
        pass

    # validation-results.json is written to backend/ at test time, and
    # picked up here if the CI pipeline has committed / bundled it.
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "validation-results.json"),
        os.path.join(os.path.dirname(__file__), "validation-results.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    report = json.load(f)
                summary["methods_validated"] = report.get("n_methods_validated")
                summary["total_checks"] = report.get("n_total_checks")
                summary["validation_generated_at"] = report.get("generated_at")
            except Exception:
                pass
            break
    return summary
