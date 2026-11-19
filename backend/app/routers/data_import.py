"""Multi-format data import — SPSS, Excel, Stata, R (RDS), CSV."""

import bz2
import gzip
import io
import json
import lzma
import os
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.errors import safe_detail

router = APIRouter()

# Size limits. Uploads are parsed in the threadpool, but one oversized file can still exhaust the
# memory of the single small server process, so every format is bounded before it is parsed.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024        # raw upload (matches the 10 MB request-body limit)
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # xlsx (a zip) and compressed RDS, after decompression
MAX_ZIP_ENTRIES = 10_000
MAX_ROWS = 100_000
MAX_COLS = 1_000
MAX_CELLS = 5_000_000                      # rows x columns
MAX_TEXT_CHARS = 5_000_000                 # pasted text
MAX_SEPARATOR_CHARS = 10

SUPPORTED = "csv, tsv, xlsx, xls, sav, dta, rds, json"
LIMITS = f"at most {MAX_ROWS:,} rows, {MAX_COLS:,} columns and {MAX_CELLS:,} cells"


class _DataTooLarge(Exception):
    pass


def _too_large(what: str = "The data") -> HTTPException:
    return HTTPException(422, f"{what} is too large to import; the limit is {LIMITS}.")


def _check_dims(n_rows: int, n_cols: int) -> None:
    if n_rows > MAX_ROWS or n_cols > MAX_COLS or n_rows * n_cols > MAX_CELLS:
        raise _DataTooLarge()


def _check_xlsx(content: bytes) -> None:
    """Refuse zip bombs and oversized sheets before pandas/openpyxl materialise them.

    The zip directory gives each member's uncompressed size (zipfile never inflates a member past
    its declared size). Sheet dimensions come from a streaming pass over the first sheet's XML:
    openpyxl pads every row to the widest one, so a single cell in column XFD would otherwise
    turn each row into 16,384 cells.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
        infos = zf.infolist()
    except (zipfile.BadZipFile, ValueError, OSError):
        raise HTTPException(422, "The file is not a valid .xlsx workbook.")
    if len(infos) > MAX_ZIP_ENTRIES or sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
        raise HTTPException(
            422, f"The workbook is too large: at most {MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB uncompressed.")

    from openpyxl import load_workbook
    from openpyxl.worksheet._reader import WorkSheetParser

    wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True, keep_links=False)
    try:
        if not wb.worksheets:
            raise HTTPException(422, "The workbook has no worksheets.")
        ws = wb.worksheets[0]  # pandas reads the first sheet by default
        max_row = max_col = 0
        with ws._get_source() as src:
            parser = WorkSheetParser(src, ws._shared_strings, data_only=True, epoch=wb.epoch,
                                     date_formats=wb._date_formats,
                                     timedelta_formats=wb._timedelta_formats)
            for idx, row in parser.parse():
                if row:
                    max_row = idx
                    max_col = max(max_col, max(cell["column"] for cell in row))
                    if max_row > MAX_ROWS + 1 or max_col > MAX_COLS:
                        raise _DataTooLarge()
        _check_dims(max(max_row - 1, 0), max_col)  # first row is the header
    finally:
        wb.close()


def _check_compressed(content: bytes) -> None:
    """Refuse gzip/bzip2/xz payloads that inflate past MAX_UNCOMPRESSED_BYTES (RDS files)."""
    if content[:2] == b"\x1f\x8b":
        stream = gzip.GzipFile(fileobj=io.BytesIO(content))
    elif content[:3] == b"BZh":
        stream = bz2.BZ2File(io.BytesIO(content))
    elif content[:6] == b"\xfd7zXZ\x00":
        stream = lzma.LZMAFile(io.BytesIO(content))
    else:
        return
    total = 0
    with stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return
            total += len(chunk)
            if total > MAX_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    422, f"The file is too large: at most {MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB uncompressed.")


def _check_json(text: str) -> None:
    """Bound the frame pd.read_json would build (a dict of dicts expands to columns x union of keys)."""
    try:
        obj = json.loads(text)
    except ValueError:
        return  # pandas reports the parse error
    if isinstance(obj, list):
        n_rows = len(obj)
        keys: set = set()
        width = 0
        for item in obj:
            if isinstance(item, dict):
                keys.update(item.keys())
                if len(keys) > MAX_COLS:
                    raise _DataTooLarge()
            elif isinstance(item, list):
                width = max(width, len(item))
        _check_dims(n_rows, max(len(keys), width, 1))
    elif isinstance(obj, dict):
        index: set = set()
        length = 1
        for value in obj.values():
            if isinstance(value, dict):
                index.update(value.keys())
                if len(index) > MAX_ROWS:
                    raise _DataTooLarge()
            elif isinstance(value, list):
                length = max(length, len(value))
        _check_dims(max(len(index), length), len(obj))


def _first_line_fields(first_line: str, sep: str, regex: bool) -> int:
    if regex:
        return len(re.split(sep, first_line.strip())) if first_line.strip() else 0
    return first_line.count(sep) + 1


def _read_delimited(text: str, sep: str, header, regex: bool = False) -> pd.DataFrame:
    """read_csv with the column count checked first and the row count bounded."""
    first_line = text.lstrip("\r\n").split("\n", 1)[0]
    if _first_line_fields(first_line, sep, regex) > MAX_COLS:
        raise _DataTooLarge()
    kwargs = {"sep": sep, "header": header, "nrows": MAX_ROWS + 1}
    if regex and sep != r"\s+":  # pandas runs "\s+" in the C engine; other regexes need python
        kwargs["engine"] = "python"
    return pd.read_csv(io.StringIO(text), **kwargs)


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _parse_upload(ext: str, content: bytes) -> pd.DataFrame:
    """Parse an uploaded file (runs in the threadpool)."""
    if ext == "csv":
        return _read_delimited(_decode(content), ",", 0)
    if ext == "tsv":
        return _read_delimited(_decode(content), "\t", 0)
    if ext == "xlsx":
        _check_xlsx(content)
        return pd.read_excel(io.BytesIO(content), engine="openpyxl", nrows=MAX_ROWS + 1)
    if ext == "xls":
        return pd.read_excel(io.BytesIO(content), engine="xlrd", nrows=MAX_ROWS + 1)
    if ext == "json":
        text = _decode(content)
        _check_json(text)
        return pd.read_json(io.StringIO(text))
    if ext == "dta":
        return pd.read_stata(io.BytesIO(content))
    if ext in ("sav", "rds"):
        if ext == "rds":
            _check_compressed(content)
        # pyreadstat / pyreadr read from a path, not a buffer
        fd, path = tempfile.mkstemp(suffix="." + ext)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
            if ext == "sav":
                import pyreadstat
                _, meta = pyreadstat.read_sav(path, metadataonly=True)
                n_rows = meta.number_rows if meta.number_rows and meta.number_rows > 0 else 0
                _check_dims(n_rows, meta.number_columns or 0)
                df, _ = pyreadstat.read_sav(path, apply_value_formats=True, row_limit=MAX_ROWS + 1)
                return df
            import pyreadr
            result = pyreadr.read_r(path)
            return list(result.values())[0]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    raise HTTPException(400, f"Unsupported file format. Supported: {SUPPORTED}.")


def _df_to_response(df: pd.DataFrame, filename: str) -> dict:
    """Convert DataFrame to a JSON-serializable response."""
    # Replace NaN/inf with None for JSON
    df = df.replace([np.inf, -np.inf], np.nan)

    # Detect variable types
    var_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            vtype = "numeric"
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            vtype = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            vtype = "datetime"
        elif pd.api.types.is_bool_dtype(df[col]):
            vtype = "boolean"
        else:
            vtype = "other"

        n_missing = int(df[col].isna().sum())
        var_info.append({
            "name": col,
            "dtype": dtype,
            "type": vtype,
            "n_missing": n_missing,
            "pct_missing": round(n_missing / len(df) * 100, 1) if len(df) > 0 else 0,
            "n_unique": int(df[col].nunique()),
        })

    # Convert to records, handling NaN
    records = json.loads(df.to_json(orient="records", date_format="iso"))

    return {
        "filename": filename,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "variable_info": var_info,
        "data": records,
        "preview": records[:10],
        "summary": {
            "total_missing": int(df.isna().sum().sum()),
            "pct_missing": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 1) if len(df) > 0 and len(df.columns) > 0 else 0,
            "numeric_columns": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
            "categorical_columns": [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])],
        }
    }


def _parse_or_422(parse, *args) -> pd.DataFrame:
    """Run a parser, mapping size violations and parse failures to client-safe 4xx errors."""
    try:
        df = parse(*args)
    except HTTPException:
        raise
    except _DataTooLarge:
        raise _too_large()
    except ImportError as exc:
        safe_detail("Import (missing reader)", exc)
        raise HTTPException(400, "This file format cannot be read on this server.")
    except MemoryError:
        raise _too_large()
    except Exception as exc:
        safe_detail("Reading the data", exc)  # logs the exception; its text is never returned
        raise HTTPException(
            422, "The data could not be parsed. Check the file format, separator and encoding.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLS or df.size > MAX_CELLS:
        raise _too_large()
    return df


@router.post("/upload")
async def import_data(file: UploadFile = File(...)):
    """Import data from SPSS (.sav), Excel (.xlsx/.xls), Stata (.dta), CSV (.csv), or TSV (.tsv)."""
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    filename = file.filename[:255]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "tsv", "xlsx", "xls", "sav", "dta", "rds", "json"):
        raise HTTPException(400, f"Unsupported file format. Supported: {SUPPORTED}.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The file is too large; the maximum upload is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    # Parsing is CPU-bound: run it in the threadpool so the event loop keeps serving requests.
    df = await run_in_threadpool(_parse_or_422, _parse_upload, ext, content)

    if df.empty:
        raise HTTPException(400, "File contains no data.")

    return await run_in_threadpool(_df_to_response, df, filename)


class ParseTextRequest(BaseModel):
    text: str = Field("", max_length=MAX_TEXT_CHARS)
    # A literal separator (multi-character separators are matched literally, never as a regex).
    separator: str = Field("auto", max_length=MAX_SEPARATOR_CHARS)
    has_header: bool = True


def _parse_text(text: str, separator: str, has_header: bool) -> pd.DataFrame:
    regex = False
    if separator == "auto":
        first_line = text.strip().split("\n")[0]
        if "\t" in first_line:
            separator = "\t"
        elif "," in first_line:
            separator = ","
        elif ";" in first_line:
            separator = ";"
        else:
            separator, regex = r"\s+", True  # our own fixed pattern
    elif len(separator) > 1:
        # pandas treats multi-character separators as regular expressions; make them literal.
        separator, regex = re.escape(separator), True
    return _read_delimited(text, separator, 0 if has_header else None, regex=regex)


@router.post("/parse-text")
def parse_text_data(payload: ParseTextRequest):
    """Parse pasted text data (CSV, TSV, or space-separated)."""
    if not payload.text.strip():
        raise HTTPException(400, "No data provided.")
    if payload.separator == "":
        raise HTTPException(422, "The separator must not be empty.")

    df = _parse_or_422(_parse_text, payload.text, payload.separator, payload.has_header)
    return _df_to_response(df, "pasted_data.csv")
