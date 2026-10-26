# PsyStat Validation Suite

Numerical-accuracy tests for every statistical method PsyStat ships. Each test
compares the backend endpoint's output against a **canonical golden value** from
one of these sources, in order of preference:

1. **NIST StRD** — [Statistical Reference Datasets](https://www.itl.nist.gov/div898/strd/)
   published by the U.S. National Institute of Standards and Technology.
   Certified to 15 significant figures.
2. **Peer-reviewed statistical textbook** — Snedecor & Cochran, Casella &
   Berger, Cohen, Hedges & Olkin, etc. Values hand-verified against the
   text and typically also verified against R.
3. **R reference implementation** — the community-consensus implementation
   for the given method (e.g., `stats::t.test`, `stats::lm`, `psych::alpha`,
   `metafor::rma`). The R code that produced each reference value is
   documented alongside it in `references.py`.
4. **Analytical closed-form** — where the answer is exact (e.g., mean of
   [1, 2, 3, 4, 5] = 3.0).

## What the suite is for

- **In-CI protection against numerical regressions.** If a refactor changes
  `scipy` version, or someone tweaks a formula, the tests fail with a
  concrete decimal-place diff.
- **Public accountability.** Results are emitted as `validation-results.json`
  and rendered by the `/validation` page on the marketing site. Every
  validated method carries the reference it was checked against.
- **Grant and IRB defense.** A user citing PsyStat can point to the exact
  certified reference their analysis was checked against, and the CI run
  that verified it, at the exact PsyStat version they used.

## How to add a validated method

1. Pick a reference. NIST StRD if possible; otherwise textbook + R.
2. If it's a NIST dataset that's not already in `datasets.py`, bake the
   raw values in there with a citation. Do not fetch datasets over the
   network at test time — CI must be deterministic and offline-safe.
3. Bake the certified value(s) in `references.py` with the exact source.
4. Write `test_<method>.py`. Use the `client` and `record_validation`
   fixtures. Compare with `numpy.testing.assert_allclose` or
   `pytest.approx`, and call `record_validation(...)` for each numeric
   comparison so it appears on the /validation dashboard.

## Tolerance convention

- Parametric tests against NIST datasets: `rtol=1e-9, atol=1e-11` (the
  "lower level of difficulty" NIST certification precision).
- Textbook / R references: `rtol=1e-4` for values published to 4 sig figs,
  `rtol=1e-6` for values published or verified to 6+ sig figs.
- Set tolerances to the *smallest* value at which your implementation
  currently passes plus one order of magnitude of safety, so drift trips
  the test. Don't set slack tolerances just to make a test pass.

## Running the suite

```bash
cd backend
pytest -m validation                  # only the validation suite
pytest -m validation -v               # verbose per-check output
pytest                                # all tests including schema smoke tests
```

After each run, `backend/validation-results.json` is written. This is what
the `/validation` page renders.
