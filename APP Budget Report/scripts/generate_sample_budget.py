"""
Create sample budget workbooks for testing the app.

Run from the project root:

    python scripts/generate_sample_budget.py

Produces sample_data/sample_budget.xlsx with two sheets:

  "FY2026 Budget"  — a clean departmental budget-vs-actual table, with a blank
                     spacer row and a TOTAL row appended.
  "Messy Export"   — the same data as a realistic bad export: a title row above
                     the headers, amounts stored as text ("$1,200.00"), an
                     accounting negative in parentheses, unusual column names,
                     and a SUBTOTAL row.

The messy sheet exists to prove the analyser's currency parsing, column
detection and total-row exclusion actually work on files like the ones people
really have.
"""

import os

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sample_budget.xlsx")

CLEAN_SHEET = "FY2026 Budget"
MESSY_SHEET = "Messy Export"

ROWS = [
    # Department,        Category,          Quarter, Budgeted, Actual
    ("Marketing",        "Advertising",     "Q1", 120000,  138500),
    ("Marketing",        "Events",          "Q1",  45000,   41200),
    ("Marketing",        "Advertising",     "Q2", 120000,  151000),
    ("Marketing",        "Events",          "Q2",  45000,   52300),
    ("Engineering",      "Salaries",        "Q1", 480000,  472000),
    ("Engineering",      "Cloud Services",  "Q1",  90000,  117400),
    ("Engineering",      "Salaries",        "Q2", 480000,  495000),
    ("Engineering",      "Cloud Services",  "Q2",  90000,  131800),
    ("Operations",       "Facilities",      "Q1",  75000,   73900),
    ("Operations",       "Travel",          "Q1",  30000,   18600),
    ("Operations",       "Facilities",      "Q2",  75000,   76400),
    ("Operations",       "Travel",          "Q2",  30000,   44100),
    ("Student Services", "Scholarships",    "Q1", 250000,  250000),
    ("Student Services", "Outreach",        "Q1",  35000,   29800),
    ("Student Services", "Scholarships",    "Q2", 250000,  248000),
    ("Student Services", "Outreach",        "Q2",  35000,   38900),
]

COLUMNS = ["Department", "Category", "Quarter", "Budgeted Amount", "Actual Amount"]


def build_clean_sheet() -> pd.DataFrame:
    """The tidy version, with a blank spacer row and a TOTAL row appended."""
    df = pd.DataFrame(ROWS, columns=COLUMNS)

    # Derived column: positive means overspent.
    df["Variance"] = df["Actual Amount"] - df["Budgeted Amount"]

    blank_row = pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
    total_row = pd.DataFrame(
        [[
            "TOTAL",
            None,
            None,
            df["Budgeted Amount"].sum(),
            df["Actual Amount"].sum(),
            df["Variance"].sum(),
        ]],
        columns=df.columns,
    )

    return pd.concat([df, blank_row, total_row], ignore_index=True)


def _as_currency_text(amount) -> str:
    """Format a number the way a finance export does: '$1,200.00'."""
    if amount is None:
        return ""
    if amount < 0:
        return "(${:,.2f})".format(abs(amount))   # accounting negative
    return "${:,.2f}".format(amount)


def build_messy_sheet() -> pd.DataFrame:
    """
    The same data, exported badly.

    Different header names, amounts as text, one refunded (negative) line, and
    a SUBTOTAL row. Nothing here is exaggerated — every quirk is one this app
    will meet in a real submission.
    """
    records = []
    for department, category, quarter, budgeted, actual in ROWS:
        records.append({
            "Cost Centre": department,
            "Line Item": category,
            "Period": quarter,
            "Planned Spend": _as_currency_text(budgeted),
            "Amount Spent": _as_currency_text(actual),
        })

    # A refunded line item: negative actual spend, in accounting notation.
    records.append({
        "Cost Centre": "Operations",
        "Line Item": "Equipment Refund",
        "Period": "Q2",
        "Planned Spend": "$0.00",
        "Amount Spent": "($12,500.00)",
    })

    # A line that was budgeted but never spent — the cell is a placeholder.
    records.append({
        "Cost Centre": "Student Services",
        "Line Item": "Emergency Fund",
        "Period": "Q2",
        "Planned Spend": "$50,000.00",
        "Amount Spent": "-",
    })

    df = pd.DataFrame(records)

    subtotal = pd.DataFrame([{
        "Cost Centre": "SUBTOTAL",
        "Line Item": None,
        "Period": None,
        "Planned Spend": "$2,300,000.00",
        "Amount Spent": "$2,368,900.00",
    }])

    return pd.concat([df, subtotal], ignore_index=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean = build_clean_sheet()
    messy = build_messy_sheet()

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        clean.to_excel(writer, index=False, sheet_name=CLEAN_SHEET)

        # The messy sheet starts one row lower, leaving room for a title row
        # above the headers — exactly the layout that breaks naive readers.
        messy.to_excel(writer, index=False, sheet_name=MESSY_SHEET, startrow=1)
        writer.sheets[MESSY_SHEET].cell(
            row=1, column=1, value="FY2026 Operating Budget — Internal Draft"
        )

    print("Wrote {}".format(OUTPUT_PATH))
    print("  '{}': {} rows".format(CLEAN_SHEET, len(clean)))
    print("  '{}': {} rows (header on row 1, use header row = 1)".format(
        MESSY_SHEET, len(messy)
    ))


if __name__ == "__main__":
    main()
