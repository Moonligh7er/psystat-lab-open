"""Multi-format data import — SPSS, Excel, Stata, R (RDS), CSV."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import pandas as pd
import numpy as np
import io
import json

router = APIRouter()


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


@router.post("/upload")
async def import_data(file: UploadFile = File(...)):
    """Import data from SPSS (.sav), Excel (.xlsx/.xls), Stata (.dta), CSV (.csv), or TSV (.tsv)."""
    if not file.filename:
        raise HTTPException(400, "No filename provided.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    content = await file.read()
    buf = io.BytesIO(content)

    try:
        if ext == "csv":
            df = pd.read_csv(buf)
        elif ext == "tsv":
            df = pd.read_csv(buf, sep="\t")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(buf, engine="openpyxl" if ext == "xlsx" else "xlrd")
        elif ext == "sav":
            df = pd.read_spss(buf)
        elif ext == "dta":
            df = pd.read_stata(buf)
        elif ext == "rds":
            # RDS requires pyreadr
            import pyreadr
            result = pyreadr.read_r(buf)
            df = list(result.values())[0]
        elif ext == "json":
            df = pd.read_json(buf)
        else:
            raise HTTPException(400, f"Unsupported file format: .{ext}. Supported: csv, tsv, xlsx, xls, sav, dta, rds, json.")
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise HTTPException(400, f"File format .{ext} requires the '{missing}' package. Please install it.")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    if df.empty:
        raise HTTPException(400, "File contains no data.")

    return _df_to_response(df, file.filename)


@router.post("/parse-text")
async def parse_text_data(payload: dict):
    """Parse pasted text data (CSV, TSV, or space-separated)."""
    text = payload.get("text", "")
    separator = payload.get("separator", "auto")
    has_header = payload.get("has_header", True)

    if not text.strip():
        raise HTTPException(400, "No data provided.")

    # Auto-detect separator
    if separator == "auto":
        first_line = text.strip().split("\n")[0]
        if "\t" in first_line:
            separator = "\t"
        elif "," in first_line:
            separator = ","
        elif ";" in first_line:
            separator = ";"
        else:
            separator = r"\s+"

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            header=0 if has_header else None,
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to parse data: {str(e)}")

    return _df_to_response(df, "pasted_data.csv")
