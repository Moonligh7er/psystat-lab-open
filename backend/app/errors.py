"""Client-safe error messages.

Exception text from numpy, scipy, statsmodels, sklearn, Stripe or Supabase can carry internal
details (file paths, library internals, fragments of the request, upstream responses), so it is
logged here and never returned. Clients get a plain description of what failed and, for the
common numerical failures, a hint they can act on.
"""

import logging

import numpy as np

logger = logging.getLogger("psystat.errors")

_HINTS = (
    (np.linalg.LinAlgError, "The data are singular or nearly so: look for constant, duplicated or perfectly collinear variables."),
    (ZeroDivisionError, "A quantity was zero where it must not be (for example a variable with no variance)."),
    (OverflowError, "The values are too large to compute with; consider rescaling them."),
    (MemoryError, "The analysis is too large to run; use fewer cases, variables or iterations."),
)

GENERIC_HINT = "Check for constant or redundant variables, very small groups, or missing values."


def safe_detail(what: str, exc: BaseException) -> str:
    """Log `exc` and return a message about `what` failing that is safe to show the client."""
    logger.warning("%s failed: %s: %r", what, type(exc).__name__, exc)
    for cls, hint in _HINTS:
        if isinstance(exc, cls):
            return f"{what} failed. {hint}"
    return f"{what} failed for these data. {GENERIC_HINT}"
