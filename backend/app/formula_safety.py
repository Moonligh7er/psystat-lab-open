"""Keep user text out of statsmodels/patsy formulas.

patsy evaluates every formula term as a Python expression, so a variable name such as
"__import__('os').system('...')" placed in a formula runs as code. Formulas must only ever
contain internal names generated here; user labels are mapped back onto the results.
"""

import logging
import re

logger = logging.getLogger(__name__)

_INTERNAL = re.compile(r"\b(_[pwcvo]\d+)\b")


def internal_names(labels, prefix: str) -> tuple[list[str], dict[str, str]]:
    """Internal formula names (_p0, _p1, ...) for user labels, and the map back to the labels."""
    names = [f"_{prefix}{i}" for i in range(len(labels))]
    return names, {n: str(lbl)[:120] for n, lbl in zip(names, labels)}


def relabel(obj, mapping: dict[str, str]):
    """Replace internal names with user labels in every key and string of a result."""
    if not mapping:
        return obj
    sub = lambda s: _INTERNAL.sub(lambda m: mapping.get(m.group(1), m.group(1)), s)
    if isinstance(obj, dict):
        return {(sub(k) if isinstance(k, str) else k): relabel(v, mapping) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(relabel(v, mapping) for v in obj)
    if isinstance(obj, str):
        return sub(obj)
    return obj


def model_error(what: str, exc: Exception) -> str:
    """A client-safe error message; the exception itself is logged, never returned."""
    logger.warning("%s failed: %r", what, exc)
    return f"{what} could not be fitted to these data. Check for constant or redundant predictors, very small groups, or missing values."
