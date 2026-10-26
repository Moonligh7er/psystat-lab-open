# PsyStat Lab — open core

This repository holds the open parts of [PsyStat Lab](https://www.psystat-lab.dev): the statistical
implementations, the test and validation suites with their datasets and reference values, and the
marketing site. It is exported from the product's private repository, so it is updated in batches.

## What's here

| Path | Contents |
|---|---|
| `backend/app/routers/` | The statistical methods (FastAPI routers): t-tests through IRT, SEM, meta-analysis, time series, causal inference, ... |
| `backend/app/irt_engine.py` | Marginal maximum-likelihood IRT (Rasch, 2PL, 3PL, graded response, partial credit) |
| `backend/tests/` | Unit tests and the numerical validation suite (`pytest -m validation`), each check against NIST StRD, a textbook, or the R reference implementation |
| `backend/tests/validation/` | Reference values (`references.py`), datasets (`datasets.py`) and how the suite works (`README.md`) |
| `backend/validation-results.json` | The latest validation run; the [validation page](https://www.psystat-lab.dev/validation.html) renders it |
| `*.html`, `blog/` | The marketing site |

Not included (proprietary): the hosted infrastructure, the AI orchestration layer, authentication,
subscription and billing code, the database schema, and the mobile/web app.

## Run it

```bash
cd backend
python -m venv venv && venv/bin/pip install -r requirements.txt pytest httpx
venv/bin/python -m pytest -m validation      # numerical accuracy against references
venv/bin/python -m pytest                    # everything
venv/bin/uvicorn app.main:app --reload       # the statistics API at http://localhost:8000/docs
```

## Found a numerical discrepancy?

Open an issue with the method, the input data, PsyStat's output and the reference value (with the R
code or source that produced it). See `backend/tests/validation/README.md`.

## License

MIT — see `LICENSE`.
