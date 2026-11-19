"""Request-size limits shared by the routers.

The API is public and served by a single small process, so any request whose cost grows with
a number the client chooses (resamples, iterations, grid sizes, nodes) is capped. Prefer pydantic
`Field(..., ge=, le=)` with these constants so violations become 422 automatically; use the
helpers below where the cost depends on several inputs at once (e.g. resamples x cases).

The caps are generous for research use (datasets of up to ~10k rows are typical).
"""

from fastapi import HTTPException

MAX_RESAMPLES = 10_000            # bootstrap / permutation / Monte Carlo replications
MAX_RESAMPLE_WORK = 20_000_000    # resamples x cases per request
MAX_ITER = 10_000                 # EM / optimiser / MCMC iteration counts
MAX_STARTS = 100                  # random starts
MAX_CLASSES = 20                  # latent classes / clusters / components
MAX_DRAWS = 50_000                # MCMC draws per chain
MAX_CHAINS = 8
MIN_TOL = 1e-12                   # a tolerance of 0 would never converge
MAX_PAIRWISE_N = 5_000            # n for algorithms that build n x n matrices


def too_large(message: str) -> HTTPException:
    """A 422 for a request that exceeds a size limit (the message should state the limit)."""
    return HTTPException(status_code=422, detail=message)


def check_resampling(n_resamples: int, n_cases: int, what: str = "resampling",
                     max_resamples: int = MAX_RESAMPLES, max_work: int = MAX_RESAMPLE_WORK) -> None:
    """Refuse resampling requests whose replications, or replications x cases, exceed the caps."""
    if n_resamples > max_resamples:
        raise too_large(f"Too many {what} replications ({n_resamples}); the maximum is {max_resamples:,}.")
    if n_resamples * max(int(n_cases), 1) > max_work:
        raise too_large(
            f"This {what} is too large: {n_resamples:,} replications x {n_cases:,} cases exceeds "
            f"{max_work:,}. Use fewer replications."
        )


def check_work(units: int, max_units: int, message: str) -> None:
    """Refuse a request whose computed work (`units`) exceeds `max_units`; `message` should give the limit."""
    if units > max_units:
        raise too_large(message)
