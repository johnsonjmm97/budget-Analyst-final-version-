"""
AI Budget Report Analyst — Streamlit entry point.

Milestone 1: upload a budget spreadsheet, load it into pandas, and preview it.
Milestone 2: detect budget/actual columns, parse currency, exclude total rows,
             compute variance, and display KPIs and breakdowns.
Milestone 3: KPI dashboard cards and the clean summary table.
Milestone 4: Plotly charts with a table view behind each.
Milestone 5: an AI executive summary, compared against the rule-written
             baseline, plus a downloadable Markdown report.
Milestone 6: an AI chat tab that answers questions about the loaded budget.

This file is the *presentation layer* only. All data work is delegated to
modules in src/, so the UI stays readable and the logic stays testable.
"""

import hashlib
import re

import pandas as pd
import streamlit as st

from src.analyzer import (
    ACTUAL_KEYWORDS,
    BUDGET_KEYWORDS,
    DEFAULT_THRESHOLD_PCT,
    SUMMARY_ACTUAL,
    SUMMARY_BUDGET,
    SUMMARY_FLAG,
    SUMMARY_VARIANCE,
    SUMMARY_VARIANCE_PCT,
    ColumnMapping,
    analyze_budget,
    build_kpi_cards,
    detect_columns,
    format_currency,
    format_percentage,
    headline_sentence,
    score_column_name,
)
from src.ai_chat import SUGGESTED_QUESTIONS, stream_answer
from src.ai_client import (
    ChatError,
    MissingAPIKeyError,
    build_client,
    describe_usage,
    is_configured,
    was_refused,
)
from src.ai_summary import stream_summary
from src.budget_context import build_context, estimate_size
from src.report_export import build_markdown_report
from src.charts import (
    budget_vs_actual_bar,
    get_palette,
    period_variance_bar,
    spending_by_category_data,
    spending_by_category_pie,
    variance_bar,
    variance_pct_bar,
)
from src.data_loader import (
    EmptyFileError,
    UnsupportedFileTypeError,
    list_sheet_names,
    load_combined,
    load_dataframe,
    summarize_dataframe,
)

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Budget Report Analyst",
    page_icon="📊",
    layout="wide",
)

# Sentinel shown in the column-mapping dropdowns for "no column selected".
NO_COLUMN = "(none)"


def context_signature(*parts) -> str:
    """
    Build a short, stable id from whatever identifies the current dataset.

    Used to suffix widget keys. Streamlit remembers a widget's value by its
    key across reruns, which is normally what you want — but when the user
    switches to a different worksheet, the old column names no longer exist.
    Making the key depend on the dataset means switching sheets creates fresh
    widgets that fall back to auto-detection, instead of clinging to a stale
    selection that would crash or silently mis-map.
    """
    return hashlib.md5(repr(parts).encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Sidebar: file source
# ---------------------------------------------------------------------------
def render_source_sidebar():
    """
    Upload a budget file.

    Returns (file_bytes, filename), or (None, None) if nothing is uploaded.
    Reading the bytes once here — rather than passing the file object around —
    matters because Streamlit reruns this script on every interaction and a
    file stream can only be consumed once.
    """
    # Test seam. Streamlit's AppTest harness cannot drive a real file uploader,
    # so the tests inject file bytes through session state instead. Nothing in
    # the UI sets this key, so it is invisible to users.
    injected = st.session_state.get("preloaded_file")
    if injected:
        return injected["bytes"], injected["name"]

    st.sidebar.header("1. Upload your budget")

    uploaded_file = st.sidebar.file_uploader(
        "Budget spreadsheet",
        type=["xlsx", "xlsm", "csv"],
        key="upload",
        help="Excel workbook or CSV export of your budget.",
    )
    if uploaded_file is None:
        return None, None

    return uploaded_file.getvalue(), uploaded_file.name


_GENERIC_HEADER = re.compile(
    r"^(unnamed:?\s*\d+|column\s*\d+|\d+(\.\d+)?)$", re.IGNORECASE
)


def _is_generic_header(name) -> bool:
    """
    True for placeholder headers rather than real ones: "Unnamed: 3" (pandas'
    name for a blank cell), "Column1" (Excel's name for an untitled table
    column), or a bare number (a data row being read as headers).
    """
    return bool(_GENERIC_HEADER.match(str(name).strip()))


def _header_row_score(df: pd.DataFrame) -> tuple:
    """
    Higher is better. Used to pick which row actually holds the column names.

    Ranked in order of how much each signal proves, because the obvious test —
    "did we find budget and actual columns?" — is not enough on its own.
    detect_columns() falls back to picking the first two *numeric* columns when
    the headers tell it nothing, so a junk header row like
    "Column1, Column2, ..." still yields a complete mapping and would tie with
    the real headers one row below it.

    So a mapping matched by column *name* outranks one matched by position,
    and placeholder headers count against the row.
    """
    mapping = detect_columns(df)

    matched_by_name = sum(
        1
        for column, keywords in (
            (mapping.budget, BUDGET_KEYWORDS),
            (mapping.actual, ACTUAL_KEYWORDS),
        )
        if column and score_column_name(column, keywords) > 0
    )
    labels = sum(
        1 for c in (mapping.department, mapping.category, mapping.period) if c
    )
    generic = sum(1 for c in df.columns if _is_generic_header(c))

    return (int(mapping.is_complete), matched_by_name, labels, -generic)


@st.cache_data(show_spinner=False)
def detect_header_row(file_bytes: bytes, filename: str, sheet_name, max_rows: int = 6) -> int:
    """
    Try the first few rows as the header row and keep the best-scoring one.

    This exists so a title row above the real table ("FY2026 Operating
    Budget — Internal Draft"), or a junk "Column1, Column2" row, doesn't
    require anyone to notice it and set the header row by hand. Falls back to
    row 0 — the common case — if nothing scores better.

    Cached because Streamlit reruns the whole script on every interaction, and
    this reads the file up to `max_rows` times.
    """
    best_row, best_score = 0, None

    for row in range(max_rows):
        try:
            candidate = load_dataframe(
                file_bytes, filename, sheet_name=sheet_name, header_row=row
            )
        except Exception:  # noqa: BLE001 - a bad guess is expected, not a bug
            continue

        score = _header_row_score(candidate)
        if best_score is None or score > best_score:
            best_score, best_row = score, row

        # Both amount columns matched by name, no placeholder headers: this
        # row cannot be beaten, so stop reading the file.
        if best_score[1] == 2 and best_score[3] == 0:
            break

    return best_row


@st.cache_data(show_spinner=False)
def detect_combinable_sheets(file_bytes: bytes, filename: str):
    """
    Find the largest set of worksheets that share the same column layout.

    A workbook with Q1/Q2/Q3/Q4 tabs is one budget split across four sheets.
    Sheets that share a column signature are almost certainly periods of the
    same table; sheets that don't (a summary tab, a notes tab) are something
    else and are left alone.

    Returns a tuple of (sheet name, header row) pairs, empty when there is no
    group of two or more. Tuples rather than lists so the result stays
    hashable for the cache.
    """
    try:
        sheets = list_sheet_names(file_bytes, filename)
    except UnsupportedFileTypeError:
        return ()

    if len(sheets) < 2:
        return ()

    groups = {}
    for sheet in sheets:
        header_row = detect_header_row(file_bytes, filename, sheet)
        try:
            frame = load_dataframe(
                file_bytes, filename, sheet_name=sheet, header_row=header_row
            )
        except Exception:  # noqa: BLE001 - an unreadable tab just isn't a match
            continue

        # The signature is the column names themselves. Two sheets with the
        # same headers in the same order are the same table.
        signature = tuple(str(c) for c in frame.columns)
        groups.setdefault(signature, []).append((sheet, header_row))

    largest = max(groups.values(), key=len, default=[])
    return tuple(largest) if len(largest) >= 2 else ()


# ---------------------------------------------------------------------------
# Sidebar: settings (collapsed — auto-detection handles the common case)
# ---------------------------------------------------------------------------
def render_settings(file_bytes: bytes, filename: str):
    """
    Sheet, header row, column mapping, and analysis options.

    Everything here has an auto-detected default, so the sidebar shows only
    the upload box until someone actually needs to correct a guess. Opening
    "Adjust settings" is the exception, not a step everyone has to take.

    Returns (raw_df, mapping, exclude_total_rows, threshold_pct, signature),
    or (None, ...) if the file could not be read — the caller has already had
    the error shown to it and should stop.
    """
    # Scopes the sheet/header widgets to *this upload*. Without it, switching
    # to a different file would keep showing the previous file's manually
    # chosen header row — Streamlit remembers a widget's value by its key
    # across reruns, regardless of which file is now loaded.
    file_signature = context_signature(filename, len(file_bytes))

    try:
        sheets = list_sheet_names(file_bytes, filename)
    except UnsupportedFileTypeError as error:
        st.sidebar.error(str(error))
        return None, ColumnMapping(), True, DEFAULT_THRESHOLD_PCT, file_signature

    combinable = detect_combinable_sheets(file_bytes, filename)

    with st.sidebar.expander("⚙️ Adjust settings", expanded=False):
        combine = False
        if combinable:
            combined_names = [name for name, _ in combinable]
            combine = st.checkbox(
                "Combine {} matching sheets".format(len(combinable)),
                value=True,
                key="combine_sheets_{}".format(file_signature),
                help="These sheets share the same columns, so they are almost "
                     "certainly periods of one budget: {}. Combining them "
                     "analyses the whole year at once.".format(
                         ", ".join(combined_names)
                     ),
            )

        sheet_name = None
        if combine:
            # The worksheet picker is meaningless while every matching sheet
            # is being read, so it is not shown at all rather than shown and
            # silently ignored.
            st.caption("Reading: {}".format(", ".join(combined_names)))
        elif len(sheets) > 1:
            # Only shown when there is a real choice to make. A single-sheet
            # workbook — the overwhelming majority of budget files — needs no
            # control here at all.
            sheet_name = st.selectbox(
                "Worksheet",
                options=sheets,
                key="worksheet_{}".format(file_signature),
                help="Which tab of the workbook holds the budget table.",
            )
        elif sheets:
            sheet_name = sheets[0]

        header_row = 0
        if not combine:
            # Keyed by sheet too, not just the file: switching worksheets must
            # re-trigger auto-detection rather than reusing the previous
            # sheet's header row. Streamlit only applies `value=` the first
            # time a key is created — an unscoped key would silently keep the
            # first sheet's answer no matter which sheet was chosen afterwards.
            header_row = int(st.number_input(
                "Header row",
                min_value=0,
                max_value=20,
                value=detect_header_row(file_bytes, filename, sheet_name),
                step=1,
                key="header_row_{}_{}".format(file_signature, sheet_name or ""),
                help=(
                    "Row containing the column names, counting from 0. Detected "
                    "automatically — change this only if the table looks wrong."
                ),
            ))

        try:
            if combine:
                raw_df = load_combined(file_bytes, filename, combinable)
            else:
                raw_df = load_dataframe(
                    file_bytes, filename, sheet_name=sheet_name, header_row=header_row
                )
        except (UnsupportedFileTypeError, EmptyFileError) as error:
            st.error(str(error))
            return None, ColumnMapping(), True, DEFAULT_THRESHOLD_PCT, file_signature
        except Exception as error:  # noqa: BLE001 - surface any parsing failure
            st.error("Could not read the file: {}".format(error))
            st.caption("Try setting the header row above, or re-saving as .xlsx.")
            return None, ColumnMapping(), True, DEFAULT_THRESHOLD_PCT, file_signature

        detected = detect_columns(raw_df)
        signature = context_signature(
            filename, sheet_name, header_row, combine, tuple(raw_df.columns)
        )

        st.caption("Column mapping — change anything that looks wrong.")
        columns = list(raw_df.columns)

        def _select(name, label, detected_value, help_text, allow_none):
            options = ([NO_COLUMN] + columns) if allow_none else columns
            default = detected_value if detected_value in options else options[0]
            choice = st.selectbox(
                label,
                options=options,
                index=options.index(default),
                key="map_{}_{}".format(name, signature),
                help=help_text,
            )
            return None if choice == NO_COLUMN else choice

        budget = _select(
            "budget", "Budgeted amount *", detected.budget,
            "The column holding planned or approved amounts.", allow_none=False,
        )
        actual = _select(
            "actual", "Actual amount *", detected.actual,
            "The column holding amounts actually spent.", allow_none=False,
        )
        department = _select(
            "department", "Department (optional)", detected.department,
            "Used to group the report by department or cost centre.", allow_none=True,
        )
        category = _select(
            "category", "Category (optional)", detected.category,
            "Used to group the report by expense category or line item.", allow_none=True,
        )
        period = _select(
            "period", "Period (optional)", detected.period,
            "Quarter, month or other time period.", allow_none=True,
        )

        st.divider()

        threshold = st.number_input(
            "Flag threshold (% over budget)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULT_THRESHOLD_PCT,
            step=0.5,
            key="threshold_pct",
            help="Anything more than this far over budget is marked 🔴. "
                 "Everything else is 🟢.",
        )

        exclude_totals = st.checkbox(
            "Exclude TOTAL / SUBTOTAL rows",
            value=True,
            key="exclude_totals",
            help="Aggregate rows already contain the sum of the rows above them. "
                 "Leaving them in would double every figure.",
        )

    mapping = ColumnMapping(
        budget=budget,
        actual=actual,
        category=category,
        department=department,
        period=period,
    )
    return raw_df, mapping, exclude_totals, float(threshold), signature


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
def render_welcome():
    """Shown before any file is uploaded, so the app is never a blank page."""
    st.info("👈 **Upload a budget spreadsheet in the sidebar to begin.**")

    st.markdown(
        """
        ### What this app does

        | | |
        | --- | --- |
        | 📊 | Reads your spreadsheet and works out budget vs. actual |
        | 🔴 | Flags every account more than **3% over budget** |
        | 📈 | Charts spending by department, category and period |
        | ✨ | Writes an executive summary with Claude |
        | 💬 | Answers questions about the figures in plain English |

        **Supported formats:** `.xlsx`, `.xlsm`, `.csv`

        ### What your file needs

        Just three things, in any column order and under any column names:

        1. A **label** column — department, category or account name
        2. A **budgeted amount** column
        3. An **actual amount** column

        Amounts can be plain numbers or text like `$1,200.00`. TOTAL rows are
        detected and excluded automatically, and so are your budget, actual
        and department/category/period columns — the report appears as soon
        as you upload, with nothing else to fill in. If a guess looks wrong,
        **⚙️ Adjust settings** in the sidebar lets you correct it.
        """
    )


def md_safe(text: str) -> str:
    """
    Escape dollar signs before text reaches a Streamlit markdown widget.

    Streamlit renders `$...$` as LaTeX maths, so a sentence carrying two
    currency amounts — "$5,432,105 against a budget of $4,998,300" — has
    everything between the two dollar signs silently turned into italic
    symbols. Every headline sentence in this app contains at least two
    amounts, so this is not an edge case.

    Only for text handed to markdown widgets: `st.metric` values are rendered
    literally and must NOT be escaped, or a backslash appears on screen.
    """
    return text.replace("$", r"\$")


def currency_column(label: str) -> st.column_config.NumberColumn:
    """Reusable display format for money columns in a table."""
    return st.column_config.NumberColumn(label, format="dollar")


def percent_column(label: str) -> st.column_config.NumberColumn:
    """Reusable display format for percentage columns in a table."""
    return st.column_config.NumberColumn(label, format="%.1f%%")


def render_kpis(analysis):
    """
    The dashboard cards.

    Note how little this function knows: the engine hands over a list of
    ready-to-draw cards, and all the UI decides is that they go four to a row.
    Every judgement about what a number means lives in analyzer.py.
    """
    st.subheader("Key performance indicators")

    cards = build_kpi_cards(analysis)
    per_row = 4

    for start in range(0, len(cards), per_row):
        row = cards[start:start + per_row]
        columns = st.columns(per_row)
        for column, card in zip(columns, row):
            column.metric(
                label=card.label,
                value=card.value,
                delta=card.delta,
                delta_color=card.delta_color,
                help=card.help_text,
            )

    # One plain-English sentence, written by rule. In Milestone 5 this becomes
    # the baseline we hold Claude's executive summary against.
    sentence = md_safe(headline_sentence(analysis))
    if analysis.kpis["total_variance"] > 0:
        st.error(sentence)
    else:
        st.success(sentence)


def render_summary_table(analysis):
    """
    The clean summary table: only the columns a reader needs, standard
    headings, sorted worst-first, with an optional calculated TOTAL row.
    """
    st.subheader("Summary table")

    controls = st.columns([1, 1, 2])
    include_total = controls[0].checkbox(
        "Show TOTAL row", value=True, key="summary_total",
        help="Calculated from the line items below — not copied from the file.",
    )
    sort_by_variance = controls[1].checkbox(
        "Sort by variance", value=True, key="summary_sort",
        help="Off keeps the original spreadsheet order.",
    )

    table = analysis.summary_table(
        include_total=include_total, sort_by_variance=sort_by_variance
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            SUMMARY_BUDGET: currency_column(SUMMARY_BUDGET),
            SUMMARY_ACTUAL: currency_column(SUMMARY_ACTUAL),
            SUMMARY_VARIANCE: currency_column(SUMMARY_VARIANCE),
            SUMMARY_VARIANCE_PCT: percent_column(SUMMARY_VARIANCE_PCT),
        },
    )

    st.download_button(
        "Download summary table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="budget_summary.csv",
        mime="text/csv",
        key="download_summary",
    )


def active_palette():
    """
    Pick the chart palette that matches the app's current theme.

    Dark mode is a *selected* palette, not an inversion of the light one:
    the same hues, stepped for a dark surface. Flipping colours automatically
    produces saturated tones that vibrate against dark backgrounds.
    """
    try:
        theme_type = st.context.theme.type
    except Exception:  # noqa: BLE001 - older Streamlit, or no theme context
        theme_type = "light"
    return get_palette(theme_type or "light")


def render_chart_with_table(fig, table, notes, caption, table_config):
    """
    Draw a chart, its caveats, and a table view of the same numbers.

    The table is not optional politeness. A value that can only be read by
    hovering is unreachable by keyboard and by anyone using a screen reader,
    and it cannot be checked. Every chart here has a text twin.
    """
    # theme=None keeps our own palette; Streamlit's default would override the
    # colours we deliberately chose for the active theme.
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            "displaylogo": False,
            # Lasso and box-select do nothing useful on a bar or a pie; the
            # PNG download does, so it stays.
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )
    st.caption(caption)

    for note in notes:
        st.caption("ⓘ {}".format(note))

    with st.expander("Show the numbers behind this chart"):
        st.dataframe(
            table, width="stretch", hide_index=True, column_config=table_config
        )


def render_charts(analysis, group_column, palette):
    """The five visualisations, each with its table-view twin."""
    budget_col = analysis.mapping.budget
    actual_col = analysis.mapping.actual
    grouped = analysis.by_group(group_column)

    # --- Variance by group: the one that answers "who is over?" -----------
    #
    # Deliberately first. Budget-vs-actual shows two totals and leaves the
    # reader to subtract; this shows the answer, colour-coded against the
    # threshold. It is the chart a manager looks at.
    st.markdown("#### Over / under budget by {}".format(group_column.lower()))

    render_chart_with_table(
        fig=variance_bar(
            grouped, group_column, analysis.variance_column,
            SUMMARY_FLAG, palette,
        ),
        table=grouped,
        notes=[],
        caption=(
            "Bars right of zero are over budget, left are under. Red means more "
            "than {:.0f}% over — the threshold set in the sidebar.".format(
                analysis.threshold_pct
            )
        ),
        table_config={
            budget_col: currency_column("Budgeted"),
            actual_col: currency_column("Actual"),
            analysis.variance_column: currency_column("Variance"),
            analysis.variance_pct_column: percent_column("Variance %"),
        },
    )

    # --- Variance % against the threshold ---------------------------------
    st.markdown("#### Variance % vs the {:.0f}% threshold".format(
        analysis.threshold_pct
    ))

    render_chart_with_table(
        fig=variance_pct_bar(
            grouped, group_column, analysis.variance_pct_column,
            SUMMARY_FLAG, analysis.threshold_pct, palette,
        ),
        table=grouped,
        notes=[],
        caption=(
            "Percentage, not dollars. A small area far over in percentage terms "
            "is out of control even when the amount looks small next to a big "
            "department."
        ),
        table_config={
            budget_col: currency_column("Budgeted"),
            actual_col: currency_column("Actual"),
            analysis.variance_column: currency_column("Variance"),
            analysis.variance_pct_column: percent_column("Variance %"),
        },
    )

    # --- Budget vs Actual, by quarter (or whatever period was mapped) -----
    #
    # This pair is deliberately independent of the "Group by" selector above.
    # A period has a natural sequence — Q1 before Q2 before Q3 — and that
    # sequence is the whole point of looking at it; letting the selector
    # switch it to "by Department" would defeat the purpose of having it.
    period_column = analysis.mapping.period

    if not period_column:
        st.info(
            "Map a period column (e.g. Quarter) in the sidebar to see budget "
            "vs. actual and the variance broken down across periods."
        )
    else:
        period_table = analysis.by_group_ordered(period_column)

        if len(period_table) < 2:
            st.info(
                "Only one {} was found in the data, so there is nothing to "
                "compare it against yet.".format(period_column.lower())
            )
        else:
            st.markdown("#### Budget vs actual by {}".format(period_column.lower()))

            render_chart_with_table(
                fig=budget_vs_actual_bar(
                    period_table, period_column, budget_col, actual_col, palette
                ),
                table=period_table,
                notes=[],
                caption=(
                    "{} appear left to right in the order they appear in your "
                    "file. Bars share one zero baseline, so the gap between "
                    "each pair *is* the variance.".format(period_column)
                ),
                table_config={
                    budget_col: currency_column("Budgeted"),
                    actual_col: currency_column("Actual"),
                    analysis.variance_column: currency_column("Variance"),
                    analysis.variance_pct_column: percent_column("Variance %"),
                },
            )

            st.markdown("#### Difference in spending by {}".format(
                period_column.lower()
            ))

            render_chart_with_table(
                fig=period_variance_bar(
                    period_table, period_column, analysis.variance_column,
                    SUMMARY_FLAG, palette,
                ),
                table=period_table,
                notes=[],
                caption=(
                    "How far actual spending was from budget in each {}. Bars "
                    "above zero are over budget, below are under. Red means "
                    "more than {:.0f}% over.".format(
                        period_column.lower(), analysis.threshold_pct
                    )
                ),
                table_config={
                    budget_col: currency_column("Budgeted"),
                    actual_col: currency_column("Actual"),
                    analysis.variance_column: currency_column("Variance"),
                    analysis.variance_pct_column: percent_column("Variance %"),
                },
            )

    st.markdown("#### Spending by {}".format(group_column.lower()))

    pie_table, pie_notes = spending_by_category_data(analysis, group_column)
    if pie_table.empty:
        st.info(
            "No positive spending to show as a share of the whole. A pie chart "
            "needs values greater than zero."
        )
        return

    render_chart_with_table(
        fig=spending_by_category_pie(pie_table, group_column, actual_col, palette),
        table=pie_table,
        notes=pie_notes,
        caption=(
            "Share of total actual spending. Read this for proportion, not for "
            "comparing similar slices — that is what the bar chart is for."
        ),
        table_config={actual_col: currency_column("Actual")},
    )


def render_group_breakdown(analysis, palette):
    """Charts and the aggregated table, all scoped by one grouping control."""
    group_options = analysis.mapping.grouping_columns()
    if not group_options:
        st.info(
            "No department, category or period column was mapped, so grouped "
            "breakdowns and charts are unavailable. Assign one in the sidebar "
            "to enable them."
        )
        return

    st.subheader("Breakdown")

    # One control above everything it scopes. Per-chart filters would let the
    # charts drift out of sync and quietly disagree with each other.
    group_column = st.selectbox(
        "Group by", options=group_options, key="group_by",
        help="Applies to both charts and the table below.",
    )

    render_charts(analysis, group_column, palette)

    st.markdown("#### Totals by {}".format(group_column.lower()))
    grouped = analysis.by_group(group_column)
    st.dataframe(
        grouped,
        width="stretch",
        hide_index=True,
        column_config={
            analysis.mapping.budget: currency_column("Budgeted"),
            analysis.mapping.actual: currency_column("Actual"),
            analysis.variance_column: currency_column("Variance"),
            analysis.variance_pct_column: percent_column("Variance %"),
        },
    )
    st.caption("Sorted worst-first: the largest overspend appears at the top.")


def render_department_detail(analysis, palette):
    """
    Drill into one department: its own figures, chart and line items.

    The Breakdown section above compares every department at once. This
    answers the question that always follows it — "fine, so what is going on
    *inside* Sales?" — which no comparison chart can, because the detail is
    exactly what a comparison flattens away.
    """
    # Prefer the department column; fall back to whatever label column exists
    # so this still works on a file whose only label is "Account" or, as in
    # one real workbook, a misspelled "Deparment annual spending".
    label_column = analysis.mapping.department
    if not label_column:
        groups = analysis.mapping.grouping_columns()
        label_column = groups[0] if groups else None
    if not label_column:
        return

    values = analysis.values_in(label_column)
    if not values:
        return

    st.subheader("{} detail".format(label_column))

    selected = st.selectbox(
        "Select a {} to see its spending".format(label_column.lower()),
        options=values,
        key="detail_value",
    )

    totals = analysis.totals_for(label_column, selected)

    # --- Its headline figures --------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Budgeted", format_currency(totals["budget"]))
    col2.metric("Actual", format_currency(totals["actual"]))
    # Named "Over / under budget" rather than "Variance" so it cannot be
    # confused with the whole-file Variance card at the top of the page —
    # two metrics with the same label but different scopes on one screen is
    # a genuine misreading risk, not just an awkward test.
    col3.metric(
        "Over / under budget",
        format_currency(totals["variance"]),
        delta=format_percentage(totals["variance_pct"]),
        # Overspending is bad, so the usual green-for-positive is inverted.
        delta_color="inverse",
    )
    col4.metric(
        "Status",
        totals["flag"],
        help="🔴 means more than {:.0f}% over budget.".format(analysis.threshold_pct),
    )

    st.caption("{:,} line item{} · {} flagged 🔴".format(
        totals["line_items"],
        "" if totals["line_items"] == 1 else "s",
        totals["flagged_count"],
    ))

    # --- What it spent on -------------------------------------------------
    #
    # Charted against a *different* label column: showing Sales broken down
    # by Sales would be one bar restating the KPI above it.
    sub_column = next(
        (c for c in analysis.mapping.grouping_columns() if c != label_column),
        None,
    )

    if sub_column:
        sub_table = analysis.by_group_within(label_column, selected, sub_column)
        st.markdown("#### {} spending by {}".format(selected, sub_column.lower()))

        render_chart_with_table(
            fig=budget_vs_actual_bar(
                sub_table, sub_column,
                analysis.mapping.budget, analysis.mapping.actual, palette,
            ),
            table=sub_table,
            notes=[],
            caption="Where {}'s money went, budgeted against actual.".format(selected),
            table_config={
                analysis.mapping.budget: currency_column("Budgeted"),
                analysis.mapping.actual: currency_column("Actual"),
                analysis.variance_column: currency_column("Variance"),
                analysis.variance_pct_column: percent_column("Variance %"),
            },
        )

    # --- Its line items ---------------------------------------------------
    st.markdown("#### {} line items".format(selected))

    rows = analysis.rows_for(label_column, selected)
    display_columns = analysis.mapping.grouping_columns() + [
        analysis.mapping.budget,
        analysis.mapping.actual,
        analysis.variance_column,
        analysis.variance_pct_column,
        analysis.flag_column,
    ]
    # De-duplicated, because grouping_columns() already contains label_column.
    display_columns = list(dict.fromkeys(display_columns))

    st.dataframe(
        rows[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            analysis.mapping.budget: currency_column("Budgeted"),
            analysis.mapping.actual: currency_column("Actual"),
            analysis.variance_column: currency_column("Variance"),
            analysis.variance_pct_column: percent_column("Variance %"),
        },
    )


def render_extremes(analysis):
    """Side-by-side lists of the biggest overspends and underspends."""
    st.subheader("Biggest variances")

    label_columns = analysis.mapping.grouping_columns()
    display_columns = label_columns + [
        analysis.mapping.budget,
        analysis.mapping.actual,
        analysis.variance_column,
        analysis.variance_pct_column,
    ]
    config = {
        analysis.mapping.budget: currency_column("Budgeted"),
        analysis.mapping.actual: currency_column("Actual"),
        analysis.variance_column: currency_column("Variance"),
        analysis.variance_pct_column: percent_column("Variance %"),
    }

    left, right = st.columns(2)

    with left:
        st.markdown("**Over budget**")
        over = analysis.top_overspends(5)
        if over.empty:
            st.caption("No line items are over budget.")
        else:
            st.dataframe(
                over[display_columns],
                width="stretch", hide_index=True, column_config=config,
            )

    with right:
        st.markdown("**Under budget**")
        under = analysis.top_underspends(5)
        if under.empty:
            st.caption("No line items are under budget.")
        else:
            st.dataframe(
                under[display_columns],
                width="stretch", hide_index=True, column_config=config,
            )


def render_line_items(analysis):
    """
    The full working data, with a status filter.

    This is the audit view — every original column plus our computed ones —
    so it lives behind an expander. The summary table above is what a reader
    is meant to look at.
    """
    with st.expander("Full line-item detail (all columns)"):
        df = analysis.data
        statuses = sorted(df[analysis.status_column].unique().tolist())
        selected = st.multiselect(
            "Filter by status", options=statuses, default=statuses,
            key="status_filter",
        )
        filtered = (
            df[df[analysis.status_column].isin(selected)] if selected else df.iloc[0:0]
        )

        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
            column_config={
                analysis.mapping.budget: currency_column(analysis.mapping.budget),
                analysis.mapping.actual: currency_column(analysis.mapping.actual),
                analysis.variance_column: currency_column(analysis.variance_column),
                analysis.variance_pct_column: percent_column(
                    analysis.variance_pct_column
                ),
            },
        )
        st.caption("Showing {:,} of {:,} line items.".format(len(filtered), len(df)))

        # Let the user take the enriched data away with them.
        st.download_button(
            "Download full analysed data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="budget_analysis.csv",
            mime="text/csv",
            key="download_full",
        )


def render_data_quality(analysis, raw_df: pd.DataFrame, filename: str):
    """
    Everything the reader needs to trust — or challenge — the numbers above.

    Showing what we changed is not optional in financial reporting. Every row
    we dropped and every value we could not parse is disclosed here.
    """
    with st.expander("Data quality and assumptions"):
        if analysis.warnings:
            for warning in analysis.warnings:
                st.warning(warning)
        else:
            st.success("No data issues detected.")

        st.markdown(
            "**Sign convention:** variance = actual − budgeted, so a positive "
            "variance means **over budget**."
        )

        if not analysis.excluded_rows.empty:
            st.markdown("**Rows excluded as totals/subtotals:**")
            st.dataframe(
                analysis.excluded_rows, width="stretch", hide_index=True
            )

        summary = summarize_dataframe(raw_df)
        st.markdown("**Source file:** `{}` — {:,} rows × {} columns".format(
            filename, summary["rows"], summary["columns"]
        ))

        # Every example is cast to text. A column of mixed types (numbers and
        # strings) cannot be converted to Arrow, which is the format Streamlit
        # uses to send tables to the browser — it would fail to render.
        details = pd.DataFrame({
            "Column": raw_df.columns,
            "Detected type": [str(dtype) for dtype in raw_df.dtypes],
            "Non-empty values": raw_df.notna().sum().values,
            "Example value": [
                str(raw_df[col].dropna().iloc[0]) if raw_df[col].notna().any() else "—"
                for col in raw_df.columns
            ],
        })
        st.dataframe(details, width="stretch", hide_index=True)

        st.markdown("**Raw data as read from the file:**")
        st.dataframe(raw_df.head(50), width="stretch")


# ---------------------------------------------------------------------------
# AI executive summary
# ---------------------------------------------------------------------------
def render_report_download(analysis, filename: str, summary):
    """
    Offer the whole report — figures and narrative — as one Markdown file.

    The narrative travels with the numbers. A download of figures alone leaves
    the reader to do the reading themselves, which is the work the summary
    exists to save.
    """
    report = build_markdown_report(analysis, filename, summary)
    st.download_button(
        "Download full report (Markdown)",
        data=report.encode("utf-8"),
        file_name="budget_report.md",
        mime="text/markdown",
        key="download_report",
        help="Opens in any editor and pastes into Word or Google Docs.",
    )


def render_baseline_comparison(analysis, summary: str):
    """
    Put Claude's summary next to the rule-written sentence from Milestone 3.

    This comparison is the actual academic content of the project. It shows
    what generative AI *adds* — rather than just demonstrating that an API was
    called. Both statements are true and drawn from the same figures; the
    difference is that one was specified in advance and the other decides for
    itself what is worth saying.
    """
    baseline = headline_sentence(analysis)

    with st.expander("Compare with the rule-written baseline"):
        left, right = st.columns(2)

        with left:
            st.markdown("**Written by rule** (`headline_sentence()`)")
            st.info(md_safe(baseline))
            st.caption("{} words · always identical for the same figures · "
                       "no API, no cost".format(len(baseline.split())))

        with right:
            st.markdown("**Written by Claude**")
            st.success(md_safe(summary.split("\n\n")[0]) if summary else "")
            st.caption("{} words in full · varies between runs · costs tokens "
                       "and a few seconds".format(len(summary.split())))

        st.markdown(
            """
            **What the rule-written version cannot do:** decide which
            departments are worth naming, notice that a small category is
            proportionally the worst offender, judge which variances are
            material, or suggest what to investigate. Every one of those is a
            judgement about *relevance*, and relevance is what you would have
            to hand-code — a rule per question you thought to ask in advance.

            **What it does better:** it is free, instant, and identical every
            time. If a fixed sentence answers your question, the model is
            overhead. The comparison is the point: use AI where judgement is
            needed, not where a formula already works.
            """
        )


def render_ai_summary(analysis, filename: str, signature: str):
    """The AI executive summary section, with caching and a download."""
    st.subheader("AI executive summary")

    if not is_configured(getattr(st, "secrets", None)):
        st.caption(
            "Add a Claude API key to generate a written executive summary — "
            "see the **💬 AI Chat** tab for setup. The report below works "
            "without one."
        )
        render_report_download(analysis, filename, None)
        return

    # Discard a summary written about a different budget. It would otherwise
    # sit above figures it does not describe, which is worse than no summary.
    if st.session_state.get("summary_signature") != signature:
        st.session_state["summary_signature"] = signature
        st.session_state["ai_summary"] = None
        st.session_state["ai_summary_usage"] = None

    summary = st.session_state.get("ai_summary")

    if st.button(
        "✨ Regenerate summary" if summary else "✨ Generate executive summary",
        key="generate_summary",
    ):
        try:
            client = build_client(secrets=getattr(st, "secrets", None))
        except MissingAPIKeyError as error:
            st.error(str(error))
            return

        outcome = {}
        try:
            text = st.write_stream(
                stream_summary(client, build_context(analysis, filename), outcome)
            )
        except ChatError as error:
            st.error(str(error))
            return

        if was_refused(outcome):
            st.warning("Claude declined to write a summary for this data.")
            return

        # Cached in session state so it is not regenerated — and re-billed —
        # every time Streamlit reruns the script, which is on every click.
        st.session_state["ai_summary"] = text
        st.session_state["ai_summary_usage"] = describe_usage(outcome)
        st.rerun()

    if summary:
        st.markdown(md_safe(summary))
        st.caption(
            "Written by Claude from the computed figures. The numbers are "
            "calculated in pandas, not by the model."
        )
        usage = st.session_state.get("ai_summary_usage")
        if usage:
            st.caption(usage)
        render_baseline_comparison(analysis, summary)

    render_report_download(analysis, filename, summary)


# ---------------------------------------------------------------------------
# AI chat
# ---------------------------------------------------------------------------
def reset_chat_if_budget_changed(signature: str):
    """
    Clear the conversation when the underlying budget changes.

    Context is only meaningful while it matches the data. If the user uploads a
    different file or remaps a column, every earlier answer refers to numbers
    that no longer exist — keeping that history would let Claude answer a new
    question using the old budget's figures.
    """
    if st.session_state.get("chat_signature") != signature:
        st.session_state["chat_signature"] = signature
        st.session_state["chat_history"] = []


def render_api_key_help():
    """Shown when no API key is configured, so the tab is never a dead end."""
    st.info("💬 The AI chat needs a Claude API key. The report tab works without one.")

    st.markdown(
        """
        ### Setting up your key

        **1.** Get a key from [console.anthropic.com](https://console.anthropic.com)
        → *API Keys*.

        **2. Running locally** — create a file named `.env` in the project root:

        ```
        ANTHROPIC_API_KEY=sk-ant-your-key-here
        ```

        `.gitignore` already excludes `.env`, so the key stays off GitHub.

        **3. Deployed on Streamlit Cloud** — *App settings → Secrets*:

        ```toml
        ANTHROPIC_API_KEY = "sk-ant-your-key-here"
        ```

        Restart the app afterwards. **Never paste a key into the code itself** —
        anything committed to git stays in its history permanently.
        """
    )


def render_chat_message(role: str, content: str):
    """Draw one turn of the conversation."""
    with st.chat_message(role, avatar="📊" if role == "assistant" else None):
        # Claude quotes figures from the brief, so its answers carry the same
        # "$…$ reads as LaTeX" hazard as the rule-written sentence.
        st.markdown(md_safe(content))


def handle_question(client, question: str, context: str):
    """
    Send one question and stream the answer into the page.

    The whole conversation is sent on every request. The Claude API is
    stateless — it has no memory of previous calls — so "maintaining context"
    means resending the history each time, which is exactly what
    `st.session_state["chat_history"]` holds.
    """
    history = st.session_state["chat_history"]
    history.append({"role": "user", "content": question})
    render_chat_message("user", question)

    outcome = {}
    with st.chat_message("assistant", avatar="📊"):
        try:
            answer = st.write_stream(
                stream_answer(client, history, context, outcome)
            )
        except ChatError as error:
            st.error(str(error))
            history.pop()          # don't keep a question we never answered
            return

        if was_refused(outcome):
            st.warning(
                "Claude declined to answer that one. Try rephrasing it as a "
                "question about the budget figures."
            )
            history.pop()
            return

        usage = describe_usage(outcome)
        if usage:
            st.caption(usage)

    history.append({"role": "assistant", "content": answer})


def render_chat_tab(analysis, filename: str, signature: str):
    """The AI Chat tab: ask questions about the budget in plain English."""
    st.subheader("Ask about this budget")

    if not is_configured(getattr(st, "secrets", None)):
        render_api_key_help()
        return

    reset_chat_if_budget_changed(signature)
    context = build_context(analysis, filename)

    st.caption(
        "Claude sees the analysed figures — totals, breakdowns and the largest "
        "variances — not the raw spreadsheet. It answers from those numbers only."
    )

    # Replay the conversation so far. Streamlit reruns this script on every
    # interaction, so the transcript has to be redrawn from session state
    # each time rather than accumulating on screen.
    for message in st.session_state["chat_history"]:
        render_chat_message(message["role"], message["content"])

    pending_question = None

    if not st.session_state["chat_history"]:
        st.markdown("**Try one of these:**")
        for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
            if st.button(suggestion, key="suggest_{}".format(index)):
                pending_question = suggestion

    typed = st.chat_input("Ask a question about this budget")
    if typed:
        pending_question = typed

    if pending_question:
        try:
            client = build_client(secrets=getattr(st, "secrets", None))
        except MissingAPIKeyError as error:
            st.error(str(error))
            return
        handle_question(client, pending_question, context)
        st.rerun()   # redraw so the suggestion buttons disappear

    with st.expander("What exactly is sent to Claude?"):
        size = estimate_size(context)
        st.caption(
            "{:,} characters (~{:,} tokens) per question. The brief is cached "
            "between questions, so follow-ups cost far less than the first.".format(
                size["characters"], size["approx_tokens"]
            )
        )
        st.code(context, language="markdown")

    if st.session_state["chat_history"]:
        if st.button("Clear conversation"):
            st.session_state["chat_history"] = []
            st.rerun()


def describe_detected_mapping(mapping: ColumnMapping) -> str:
    """
    One line stating what the app is treating as budget/actual/labels.

    This is the payoff for hiding the mapping dropdowns: the reader still
    gets told what happened, in one sentence they can skim, rather than
    either five dropdowns to check or silence about what was assumed.
    """
    parts = ["Budget = **{}**".format(mapping.budget),
             "Actual = **{}**".format(mapping.actual)]
    groups = [g for g in (mapping.department, mapping.category, mapping.period) if g]
    if groups:
        parts.append("grouped by " + ", ".join("**{}**".format(g) for g in groups))
    return " · ".join(parts) + " — open ⚙️ Adjust settings in the sidebar to change this."


# ---------------------------------------------------------------------------
# Application flow
# ---------------------------------------------------------------------------
def main():
    st.title("📊 AI Budget Report Analyst")
    st.caption("Upload a budget spreadsheet, analyse it, and ask Claude about it.")

    file_bytes, filename = render_source_sidebar()
    if file_bytes is None:
        render_welcome()
        return

    # Sheet, header row, and column mapping are all auto-detected here.
    # Nothing else in the app runs a second read of the file — the sidebar's
    # collapsed "Adjust settings" is the only place these choices are made,
    # and raw_df is already the final, correctly-parsed table by the time it
    # comes back.
    raw_df, mapping, exclude_totals, threshold, signature = render_settings(
        file_bytes, filename
    )
    if raw_df is None:
        return   # render_settings has already shown the reader what went wrong

    # --- Analyse ----------------------------------------------------------
    try:
        analysis = analyze_budget(
            raw_df,
            mapping,
            exclude_total_rows=exclude_totals,
            threshold_pct=threshold,
        )
    except ValueError as error:
        st.error(str(error))
        st.caption("Open **⚙️ Adjust settings** in the sidebar and check the "
                   "column mapping.")
        return

    # Keep results in session state so later milestones (charts, AI summary)
    # can read them without recomputing.
    st.session_state["analysis"] = analysis
    st.session_state["budget_filename"] = filename

    st.success("Analysed **{}** — {:,} line items.".format(
        filename, analysis.kpis["line_items"]
    ))
    st.caption(describe_detected_mapping(mapping))

    palette = active_palette()

    # Two tabs: the computed report, and the conversation about it. Tabs keep
    # both one click away — the chat is meant to be read alongside the figures,
    # not instead of them.
    report_tab, chat_tab = st.tabs(["📊 Report", "💬 AI Chat"])

    with report_tab:
        render_kpis(analysis)
        st.divider()
        render_ai_summary(analysis, filename, signature)
        st.divider()
        render_summary_table(analysis)
        st.divider()
        render_group_breakdown(analysis, palette)
        st.divider()
        render_department_detail(analysis, palette)
        st.divider()
        render_extremes(analysis)
        st.divider()
        render_line_items(analysis)
        render_data_quality(analysis, raw_df, filename)

    with chat_tab:
        render_chat_tab(analysis, filename, signature)


if __name__ == "__main__":
    main()
