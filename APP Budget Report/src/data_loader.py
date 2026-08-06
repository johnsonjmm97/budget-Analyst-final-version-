"""
Loading and basic cleaning of budget spreadsheets.

This module knows nothing about Streamlit. It only deals with bytes in and
pandas DataFrames out, which keeps it easy to test and reuse.
"""

import io
import os
from typing import List, Optional

import pandas as pd

# Extensions we are willing to open. Anything else is rejected early with a
# clear message instead of failing deep inside pandas.
EXCEL_EXTENSIONS = (".xlsx", ".xlsm")
CSV_EXTENSIONS = (".csv",)
SUPPORTED_EXTENSIONS = EXCEL_EXTENSIONS + CSV_EXTENSIONS


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file is not a format we can read."""


class EmptyFileError(ValueError):
    """Raised when the file parsed correctly but contains no usable rows."""


def get_extension(filename: str) -> str:
    """Return the lowercased file extension, e.g. '.xlsx'."""
    return os.path.splitext(filename)[1].lower()


def is_supported(filename: str) -> bool:
    """True if we can read this filename's format."""
    return get_extension(filename) in SUPPORTED_EXTENSIONS


def _to_buffer(file_bytes: bytes) -> io.BytesIO:
    """
    Wrap raw bytes in an in-memory binary stream.

    We always work from bytes rather than from the uploaded file object
    directly. A file object can only be read once before its internal cursor
    reaches the end; bytes can be re-read as many times as we like (for
    example, once to list sheet names and again to load a sheet).
    """
    return io.BytesIO(file_bytes)


def list_sheet_names(file_bytes: bytes, filename: str) -> List[str]:
    """
    Return the sheet names inside an Excel workbook.

    CSV files have no concept of sheets, so an empty list is returned.
    """
    if get_extension(filename) in CSV_EXTENSIONS:
        return []

    if not is_supported(filename):
        raise UnsupportedFileTypeError(
            "Unsupported file type '{}'. Please upload one of: {}.".format(
                get_extension(filename) or "(none)", ", ".join(SUPPORTED_EXTENSIONS)
            )
        )

    # ExcelFile reads only the workbook structure, not every cell, so this is
    # cheap even for large files.
    with pd.ExcelFile(_to_buffer(file_bytes), engine="openpyxl") as workbook:
        return list(workbook.sheet_names)


def load_dataframe(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
    header_row: int = 0,
) -> pd.DataFrame:
    """
    Read an uploaded budget file into a cleaned DataFrame.

    Args:
        file_bytes:  Raw contents of the uploaded file.
        filename:    Original filename, used only to detect the format.
        sheet_name:  Which Excel sheet to read. None means the first sheet.
        header_row:  Zero-based index of the row holding the column names.
                     Budget exports often have a title row above the real
                     header, so this is exposed to the user in the sidebar.

    Raises:
        UnsupportedFileTypeError: the extension is not one we handle.
        EmptyFileError:           the file parsed but held no usable rows.
    """
    extension = get_extension(filename)

    if extension in CSV_EXTENSIONS:
        df = pd.read_csv(_to_buffer(file_bytes), header=header_row)
    elif extension in EXCEL_EXTENSIONS:
        df = pd.read_excel(
            _to_buffer(file_bytes),
            sheet_name=sheet_name if sheet_name is not None else 0,
            header=header_row,
            engine="openpyxl",
        )
    else:
        raise UnsupportedFileTypeError(
            "Unsupported file type '{}'. Please upload one of: {}.".format(
                extension or "(none)", ", ".join(SUPPORTED_EXTENSIONS)
            )
        )

    df = clean_dataframe(df)

    if df.empty:
        raise EmptyFileError(
            "The file was read successfully but contains no data rows. "
            "Check that the header row setting matches your spreadsheet."
        )

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light, non-destructive cleaning that is safe for any spreadsheet.

    Deliberately conservative: we do not rename, retype, or drop columns that
    hold real values. Budget-specific interpretation happens in a later
    milestone, once we know which columns mean what.
    """
    df = df.copy()

    # Excel exports frequently include trailing blank rows and unnamed blank
    # columns. Drop anything that is entirely empty.
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Normalise column names: force to string, collapse whitespace. Stray
    # spaces in headers are a common source of KeyError bugs later on.
    df.columns = [" ".join(str(col).split()) for col in df.columns]

    # Two columns with the same name is a real spreadsheet in the wild — a
    # genuinely repeated header, or two headers that only differed by
    # whitespace before the line above collapsed them together. pandas allows
    # it silently at this point, but every later groupby() on that name
    # crashes with "cannot insert X, already exists". Make names unique now,
    # while we still know which column is which, rather than downstream where
    # the failure is opaque and far from its cause.
    df.columns = _deduplicate_columns(df.columns)

    # dropna leaves gaps in the index (0, 3, 4, ...). Renumber so row numbers
    # shown in the UI are meaningful.
    df = df.reset_index(drop=True)

    return df


def _deduplicate_columns(columns) -> list:
    """
    Suffix repeated column names so every name is unique: "Department",
    "Department (2)", "Department (3)", ...

    Mirrors the suffixing analyzer.py already uses for computed columns
    (Variance, Variance (2), ...) — one convention for "this name was taken"
    across the codebase, so a user who has seen one recognises the other.
    """
    seen = {}
    result = []
    for name in columns:
        count = seen.get(name, 0) + 1
        seen[name] = count
        result.append(name if count == 1 else "{} ({})".format(name, count))
    return result


# Name given to the column that records which sheet each row came from.
# "Period" rather than "Sheet" because that is what these sheets almost always
# are — Q1/Q2/Q3/Q4, Jan/Feb/Mar — and because the analyser's own column
# detection recognises "Period", so the quarter charts light up automatically.
SHEET_COLUMN = "Period"


def load_combined(
    file_bytes: bytes,
    filename: str,
    sheet_header_rows,
    sheet_column: str = SHEET_COLUMN,
) -> pd.DataFrame:
    """
    Read several worksheets that share a column layout into one table.

    A workbook with a sheet per quarter holds one budget split across tabs,
    not four unrelated budgets. Read one at a time, a department's line items
    are scattered and no trend across the year can be seen. Stacked into one
    frame — with a column naming the sheet each row came from — the whole year
    becomes a single analysis.

    Args:
        file_bytes:        Raw contents of the uploaded file.
        filename:          Original filename, used to detect the format.
        sheet_header_rows: Pairs of (sheet name, header row) — each sheet may
                           have its headers on a different row.
        sheet_column:      Name for the column recording the source sheet.

    Raises:
        EmptyFileError: none of the sheets contained usable rows.
    """
    frames = []

    for sheet_name, header_row in sheet_header_rows:
        try:
            frame = load_dataframe(
                file_bytes, filename, sheet_name=sheet_name, header_row=header_row
            )
        except EmptyFileError:
            # One empty tab in a workbook is not a reason to fail the whole
            # combination — skip it and use the sheets that do have data.
            continue

        # Guard against a file that already has a column called "Period":
        # inserting a second one would create the duplicate-name crash that
        # _deduplicate_columns exists to prevent.
        name = sheet_column
        suffix = 2
        while name in frame.columns:
            name = "{} ({})".format(sheet_column, suffix)
            suffix += 1

        frame = frame.copy()
        frame.insert(0, name, str(sheet_name).strip())
        frames.append(frame)

    if not frames:
        raise EmptyFileError(
            "None of the selected sheets contained any data rows."
        )

    # sort=False keeps the column order of the first sheet rather than
    # alphabetising it, so the combined table reads like the originals.
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.reset_index(drop=True)


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """
    Compute headline facts about a DataFrame for display in the UI.

    Returns a plain dict so the UI layer decides how to present it.
    """
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "text_columns": [c for c in df.columns if c not in numeric_columns],
        "missing_values": int(df.isna().sum().sum()),
    }
