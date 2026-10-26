"""
Input validation utilities for PsyStat backend.
Prevents oversized arrays, NaN/Inf values, and other malformed inputs.
"""

import math
from fastapi import HTTPException

MAX_ARRAY_LENGTH = 100_000
MAX_MATRIX_ROWS = 10_000
MAX_MATRIX_COLS = 1_000
MAX_STRING_LENGTH = 100_000


def validate_numeric_array(arr: list, name: str = "data", max_len: int = MAX_ARRAY_LENGTH) -> list[float]:
    """Validate and sanitize a numeric array."""
    if not isinstance(arr, list):
        raise HTTPException(400, f"{name} must be a list")
    if len(arr) > max_len:
        raise HTTPException(400, f"{name} too large ({len(arr)} items). Maximum {max_len}.")
    if len(arr) == 0:
        raise HTTPException(400, f"{name} cannot be empty")

    cleaned = []
    for i, v in enumerate(arr):
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                raise HTTPException(400, f"{name}[{i}] contains NaN or Infinity")
            cleaned.append(f)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{name}[{i}] is not a valid number: {v!r}")
    return cleaned


def validate_matrix(matrix: list, name: str = "matrix", max_rows: int = MAX_MATRIX_ROWS, max_cols: int = MAX_MATRIX_COLS) -> list[list[float]]:
    """Validate and sanitize a numeric matrix."""
    if not isinstance(matrix, list):
        raise HTTPException(400, f"{name} must be a list of lists")
    if len(matrix) > max_rows:
        raise HTTPException(400, f"{name} has too many rows ({len(matrix)}). Maximum {max_rows}.")
    if len(matrix) == 0:
        raise HTTPException(400, f"{name} cannot be empty")

    cleaned = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise HTTPException(400, f"{name}[{i}] must be a list")
        if len(row) > max_cols:
            raise HTTPException(400, f"{name}[{i}] has too many columns ({len(row)}). Maximum {max_cols}.")
        cleaned.append(validate_numeric_array(row, f"{name}[{i}]"))
    return cleaned


def validate_string(s: str, name: str = "input", max_len: int = MAX_STRING_LENGTH) -> str:
    """Validate string input length."""
    if not isinstance(s, str):
        raise HTTPException(400, f"{name} must be a string")
    if len(s) > max_len:
        raise HTTPException(400, f"{name} too long ({len(s)} chars). Maximum {max_len}.")
    return s


def validate_positive_int(v: int, name: str = "value", max_val: int = 1_000_000) -> int:
    """Validate a positive integer."""
    if not isinstance(v, int) or v < 1:
        raise HTTPException(400, f"{name} must be a positive integer")
    if v > max_val:
        raise HTTPException(400, f"{name} too large ({v}). Maximum {max_val}.")
    return v
