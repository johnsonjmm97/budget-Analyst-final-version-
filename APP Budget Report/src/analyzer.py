"""
Budget analysis engine.

Turns a raw spreadsheet DataFrame into budget insight:

    1. Guess which columns hold budgeted vs. actual amounts.
    2. Parse currency text such as "$1,200.00" or "(500)" into real numbers.
    3. Identify and exclude TOTAL / SUBTOTAL rows.
    4. Compute variance, variance %, and an over/under status per row.
    5. Roll everything up into headline KPIs and per-group breakdowns.

Like data_loader, this module never imports Streamlit. It is pure pandas, so
it can be tested from a plain script (see tests/test_analyzer.py).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name vocabulary
#
# Real budget files use wildly different headers. Rather than demanding one
# exact format, we score each column name against these keyword lists and pick
# the best match. The user can always override the guess in the UI.
# ---------------------------------------------------------------------------
BUDGET_KEYWORDS = [
    "budget", "budgeted", "planned", "plan", "forecast",
    "allocated", "allocation", "approved", "estimate", "target",
]
ACTUAL_KEYWORDS = [
    "actual", "actuals", "spent", "spend", "expense", "expenses",
    "incurred", "used", "ytd", "paid", "cost",
]
VARIANCE_KEYWORDS = ["variance", "var", "difference", "diff", "delta", "over under"]
CATEGORY_KEYWORDS = [
    "category", "account", "line item", "item", "description",
    "expense type", "type", "gl code", "gl",
]
DEPARTMENT_KEYWORDS = [
    "department", "dept", "division", "unit", "team",
    "cost center", "cost centre", "org", "organization", "function",
]
PERIOD_KEYWORDS = ["quarter", "month", "period", "date", "fiscal year", "year", "fy"]

# Row labels that mark an aggregate row rather than a real line item.
TOTAL_ROW_PATTERN = re.compile(
    r"^\s*(grand\s+|sub[\s\-]?)?(total|totals|sum|subtotal)\b",
    re.IGNORECASE,
)

# Values that mean "no number here" when they appear in an amount column.
NULL_TOKENS = {"", "-", "--", "—", "–", "n/a", "na", "nan", "none", "nil", "tbd"}

# Standard headings used in the presentation-ready summary table. The source
# file might call the budget column "Planned Spend" or "FY26 Approved"; the
# report always calls it "Budgeted", so every report reads the same way.
SUMMARY_BUDGET = "Budgeted"
SUMMARY_ACTUAL = "Actual"
SUMMARY_VARIANCE = "Variance"
SUMMARY_VARIANCE_PCT = "Variance %"
SUMMARY_STATUS = "Status"
SUMMARY_FLAG = "Flag"
TOTAL_ROW_LABEL = "TOTAL"

# How far over budget a line may go before it is flagged.
#
# Expressed as a percentage of the budget rather than an absolute amount, so a
# $3,000 overspend on a $10,000 budget (30%) is flagged while the same $3,000
# on a $2,000,000 budget (0.15%) is not. Percentage is what tells you whether
# an area is actually out of control.
DEFAULT_THRESHOLD_PCT = 3.0

FLAG_OVER = "🔴 Over"
FLAG_OK = "🟢 OK"
FLAG_UNKNOWN = "—"

# Amounts closer to zero than this are treated as exactly on budget. Guards
# against float rounding noise producing a "$0.00 over budget" status.
ZERO_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# Currency parsing
# ---------------------------------------------------------------------------
def parse_currency_value(value) -> float:
    """
    Convert a single spreadsheet cell into a float, or NaN if it isn't a number.

    Handles the formats budget exports actually contain:

        1200        -> 1200.0     already numeric
        "$1,200.50" -> 1200.5     currency symbol + thousands separators
        "(500)"     -> -500.0     accounting notation for negatives
        "1.200,50"  -> 1200.5     European separators
        "12%"       -> 12.0       stray percent sign
        "-"         -> NaN        placeholder for "no value"
    """
    # Already a number: pass straight through.
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if pd.notna(value) else np.nan
    if value is None or pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return np.nan

    # Accounting negatives are wrapped in parentheses: (1,500) means -1500.
    is_negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))

    # Strip everything that isn't a digit or a separator: currency symbols,
    # spaces, percent signs, stray letters like "USD".
    digits = re.sub(r"[^\d.,]", "", text)
    if not digits:
        return np.nan

    digits = _normalize_separators(digits)

    try:
        number = float(digits)
    except ValueError:
        return np.nan

    return -number if is_negative else number


def _normalize_separators(digits: str) -> str:
    """
    Decide which of '.' and ',' is the decimal point, and remove the other.

    The rule: whichever separator appears *last* is the decimal point, because
    thousands separators always come before the decimal point. When only one
    kind is present, a single separator followed by exactly two digits is a
    decimal point; anything else is a thousands separator.
    """
    has_dot = "." in digits
    has_comma = "," in digits

    if has_dot and has_comma:
        decimal_sep = "." if digits.rfind(".") > digits.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        digits = digits.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        parts = digits.split(",")
        # "1,50" -> decimal; "1,500" or "1,234,567" -> thousands separators.
        if len(parts) == 2 and len(parts[1]) == 2:
            digits = digits.replace(",", ".")
        else:
            digits = digits.replace(",", "")
    elif has_dot:
        parts = digits.split(".")
        # "1.234.567" can only be thousands separators.
        if len(parts) > 2:
            digits = digits.replace(".", "")

    return digits


def parse_currency_series(series: pd.Series) -> pd.Series:
    """Apply parse_currency_value across a column, returning a float Series."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(parse_currency_value).astype(float)


def numeric_ratio(series: pd.Series) -> float:
    """
    Fraction of non-empty cells in a column that parse as numbers.

    Used to tell an amount column (mostly numbers) from a label column, even
    when the amounts are stored as text.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    parsed = parse_currency_series(non_null)
    return float(parsed.notna().sum()) / float(len(non_null))


def looks_numeric(series: pd.Series, threshold: float = 0.7) -> bool:
    """True if enough of the column parses as numbers to treat it as amounts."""
    return numeric_ratio(series) >= threshold


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------
@dataclass
class ColumnMapping:
    """Which spreadsheet column plays which role in the analysis."""

    budget: Optional[str] = None
    actual: Optional[str] = None
    category: Optional[str] = None
    department: Optional[str] = None
    period: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """We can only analyse once we know where budget and actual live."""
        return bool(self.budget) and bool(self.actual)

    def grouping_columns(self) -> List[str]:
        """Label columns available for grouped breakdowns, most useful first."""
        return [c for c in (self.department, self.category, self.period) if c]


def _normalize(name: str) -> str:
    """Lowercase a header and reduce punctuation to single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split())


def score_column_name(name: str, keywords: List[str]) -> int:
    """
    Score how strongly a column name matches a keyword list.

    100 = the whole header is the keyword ("Budget")
     60 = the keyword appears as a standalone word ("Budget Amount")
     25 = the keyword appears inside a word ("FY26Budget")
      0 = no match

    Tiered scoring beats a plain substring check: given "Budget" and
    "Budgeted Amount Prior Year", the exact match should win.
    """
    normalized = _normalize(name)
    if not normalized:
        return 0

    best = 0
    words = normalized.split()
    for keyword in keywords:
        keyword = _normalize(keyword)
        if normalized == keyword:
            best = max(best, 100)
        elif keyword in words or keyword in normalized.split():
            best = max(best, 60)
        elif " " in keyword and keyword in normalized:
            best = max(best, 60)
        elif keyword in normalized:
            best = max(best, 25)
    return best


def _best_match(
    df: pd.DataFrame,
    keywords: List[str],
    exclude: List[str],
    require_numeric: bool,
) -> Optional[str]:
    """Return the highest-scoring unused column matching a keyword list."""
    best_column = None
    best_score = 0

    for column in df.columns:
        if column in exclude:
            continue
        score = score_column_name(column, keywords)
        if score == 0:
            continue
        if require_numeric and not looks_numeric(df[column]):
            continue
        if score > best_score:
            best_column, best_score = column, score

    return best_column


def detect_columns(df: pd.DataFrame) -> ColumnMapping:
    """
    Guess the role of each column in a budget table.

    Order matters. Variance columns are identified first and set aside, because
    a header like "Budget Variance" would otherwise be mistaken for the budget
    column and silently corrupt every number in the report.
    """
    reserved: List[str] = []

    # Set aside any pre-existing variance column — we recompute variance
    # ourselves rather than trusting a column we did not calculate.
    for column in df.columns:
        if score_column_name(column, VARIANCE_KEYWORDS) >= 60:
            reserved.append(column)

    budget = _best_match(df, BUDGET_KEYWORDS, reserved, require_numeric=True)
    if budget:
        reserved.append(budget)

    actual = _best_match(df, ACTUAL_KEYWORDS, reserved, require_numeric=True)
    if actual:
        reserved.append(actual)

    # Fallback: headers gave us nothing useful, so fall back on position.
    # The first two numeric columns are budget then actual far more often
    # than not, and the user can correct us in the sidebar.
    if not (budget and actual):
        numeric_columns = [
            c for c in df.columns if c not in reserved and looks_numeric(df[c])
        ]
        if not budget and numeric_columns:
            budget = numeric_columns.pop(0)
            reserved.append(budget)
        if not actual and numeric_columns:
            actual = numeric_columns.pop(0)
            reserved.append(actual)

    label_columns = [c for c in df.columns if c not in reserved]
    label_df = df[label_columns] if label_columns else df.iloc[:, :0]

    department = _best_match(label_df, DEPARTMENT_KEYWORDS, [], require_numeric=False)
    category = _best_match(
        label_df, CATEGORY_KEYWORDS, [department] if department else [], require_numeric=False
    )
    period = _best_match(
        label_df,
        PERIOD_KEYWORDS,
        [c for c in (department, category) if c],
        require_numeric=False,
    )

    # Last resort: a budget file always has *some* label column, but its
    # header may be misspelled ("Deparment") or phrased unlike anything in our
    # keyword lists ("Deparment annual spending"). Without a label there is
    # nothing to group, chart, or drill into — the report degrades to a single
    # total. Falling back to the first text column is far better than that, and
    # the user can still correct the role in the sidebar.
    if not any((department, category, period)):
        for column in label_columns:
            if not looks_numeric(df[column]):
                category = column
                break

    return ColumnMapping(
        budget=budget,
        actual=actual,
        category=category,
        department=department,
        period=period,
    )


# ---------------------------------------------------------------------------
# Total-row detection
# ---------------------------------------------------------------------------
def find_total_rows(df: pd.DataFrame, amount_columns: List[str]) -> pd.Series:
    """
    Boolean mask marking rows that are aggregates, not line items.

    Why this matters: a spreadsheet's TOTAL row already contains the sum of
    every row above it. Leaving it in doubles every KPI. This is the single
    most common way an automated budget report produces wrong numbers.

    Detection is text-based: a label cell starting with TOTAL / SUBTOTAL /
    GRAND TOTAL / SUM. We only inspect non-amount columns, so a legitimate
    department named e.g. "Total Rewards" in a label column is still caught,
    which is why the UI lets the user turn exclusion off.
    """
    label_columns = [c for c in df.columns if c not in amount_columns]
    mask = pd.Series(False, index=df.index)

    for column in label_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        matches = df[column].apply(
            lambda v: bool(TOTAL_ROW_PATTERN.match(str(v))) if pd.notna(v) else False
        )
        mask = mask | matches

    return mask


def flag_for(variance_pct: float, threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> str:
    """
    Red or green for one line, based on how far over budget it is.

    Only *overspending* is flagged. A department 20% under budget is not
    underperforming in the sense this flag means — it may be, but that is a
    different question with a different answer, and colouring it red would
    train the reader to ignore red.

    A missing percentage (budget of zero) returns a dash rather than green:
    "we could not judge this" is not the same as "this is fine".
    """
    if variance_pct is None or pd.isna(variance_pct):
        return FLAG_UNKNOWN
    return FLAG_OVER if variance_pct > threshold_pct else FLAG_OK


def flag_series(variance_pct: pd.Series,
                threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> pd.Series:
    """Apply flag_for across a column of variance percentages."""
    return variance_pct.apply(lambda value: flag_for(value, threshold_pct))


def _unique_name(base: str, used: List[str]) -> str:
    """
    Return `base`, or `base (2)`, `base (3)`... if it is already taken.

    Needed when assembling the summary table: a source file may already have a
    label column literally called "Variance" or "Status", and two columns with
    the same name in one DataFrame is a silent data-loss bug.
    """
    if base not in used:
        return base
    suffix = 2
    while "{} ({})".format(base, suffix) in used:
        suffix += 1
    return "{} ({})".format(base, suffix)


def _unique_column_name(df: pd.DataFrame, base: str) -> str:
    """
    Pick a column name that does not clash with the user's own columns.

    We never overwrite a column that came from the source file — if the file
    already has "Variance", ours becomes "Variance (calculated)" so the user
    can compare the two.
    """
    if base not in df.columns:
        return base
    candidate = "{} (calculated)".format(base)
    suffix = 2
    while candidate in df.columns:
        candidate = "{} (calculated {})".format(base, suffix)
        suffix += 1
    return candidate


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------
@dataclass
class BudgetAnalysis:
    """Everything the UI needs to render a budget report."""

    data: pd.DataFrame                      # line items, with computed columns
    excluded_rows: pd.DataFrame             # total/subtotal rows we set aside
    mapping: ColumnMapping
    kpis: Dict[str, float] = field(default_factory=dict)
    variance_column: str = "Variance"
    variance_pct_column: str = "Variance %"
    status_column: str = "Status"
    flag_column: str = "Flag"
    warnings: List[str] = field(default_factory=list)
    threshold_pct: float = DEFAULT_THRESHOLD_PCT

    def _grouped_totals(self, group_column: str,
                        frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Shared aggregation behind by_group(), by_group_ordered() and the
        per-department drill-down.

        `sort=False` on the groupby preserves the column's own row order —
        needed by the ordered variant, and harmless here since by_group()
        re-sorts the result by variance anyway.

        `frame` restricts the aggregation to a subset of rows (one
        department's line items, say) while keeping every other rule — the
        flag threshold, the safe percentage — identical to the whole-file
        version. A drill-down that computed its own totals a slightly
        different way would quietly disagree with the report above it.
        """
        source = self.data if frame is None else frame
        grouped = (
            source.groupby(group_column, dropna=False, sort=False)
            .agg({
                self.mapping.budget: "sum",
                self.mapping.actual: "sum",
                self.variance_column: "sum",
            })
            .reset_index()
        )
        grouped[self.variance_pct_column] = _safe_percentage(
            grouped[self.variance_column], grouped[self.mapping.budget]
        )
        grouped[SUMMARY_FLAG] = flag_series(
            grouped[self.variance_pct_column], self.threshold_pct
        )
        return grouped

    def by_group(self, group_column: str) -> pd.DataFrame:
        """
        Aggregate budget, actual and variance by a label column.

        Returned sorted by variance descending, so the worst overspends are at
        the top — that is what a reader wants to see first.

        The flag is computed on the *group's* combined percentage, not by
        counting flagged rows inside it. A department can contain one badly
        overspent line and still be fine overall; the group flag answers "is
        this department over?", and the line-item table answers "where".
        """
        grouped = self._grouped_totals(group_column)
        return grouped.sort_values(self.variance_column, ascending=False)

    def by_group_ordered(self, group_column: str) -> pd.DataFrame:
        """
        Same aggregation as by_group(), but keeps the column's own row order.

        Use this for anything with a natural sequence — quarters, months —
        where Q3 belongs after Q2 on the page regardless of which one
        overspent more. Sorting by variance, as by_group() does, would
        scramble that sequence.
        """
        return self._grouped_totals(group_column)

    def flagged_groups(self, group_column: str) -> pd.DataFrame:
        """Just the groups that are over the threshold, worst first."""
        grouped = self.by_group(group_column)
        return grouped[grouped[SUMMARY_FLAG] == FLAG_OVER]

    def values_in(self, group_column: str) -> List[str]:
        """
        Every distinct value in a label column, sorted, as strings.

        Strings because these populate a dropdown and are matched back against
        it: a department coded as the number 400 must survive the round trip
        to the widget and back without becoming 400.0.
        """
        values = self.data[group_column].dropna().astype(str).unique()
        return sorted(values)

    def rows_for(self, group_column: str, value: str) -> pd.DataFrame:
        """
        The line items belonging to one value of a label column.

        Compared as strings for the same reason values_in() returns them:
        the value arrives back from a dropdown as text.
        """
        return self.data[self.data[group_column].astype(str) == str(value)]

    def totals_for(self, group_column: str, value: str) -> Dict:
        """
        Headline figures for a single department (or category, or period).

        Same shape and same rules as the whole-file KPIs, so the drill-down
        and the report above it can never tell different stories.
        """
        rows = self.rows_for(group_column, value)

        budget = float(rows[self.mapping.budget].sum())
        actual = float(rows[self.mapping.actual].sum())
        variance = actual - budget
        variance_pct = (variance / budget * 100.0) if budget else np.nan

        return {
            "budget": budget,
            "actual": actual,
            "variance": variance,
            "variance_pct": variance_pct,
            "flag": flag_for(variance_pct, self.threshold_pct),
            "line_items": len(rows),
            "flagged_count": int((rows[self.flag_column] == FLAG_OVER).sum()),
        }

    def by_group_within(self, group_column: str, value: str,
                        sub_column: str) -> pd.DataFrame:
        """
        Totals by `sub_column`, restricted to one value of `group_column`.

        This is what turns "Sales is 2% over" into "…because Revenue is over
        and Travel is under" — the breakdown *inside* a department.
        """
        return self._grouped_totals(sub_column, self.rows_for(group_column, value))

    def summary_table(
        self,
        include_total: bool = False,
        sort_by_variance: bool = True,
    ) -> pd.DataFrame:
        """
        Build the presentation-ready summary table.

        This is deliberately *not* the same thing as `self.data`. `data` is the
        full working set — every original column plus our computed ones — which
        is what you want for auditing but far too noisy for a report.

        The summary table keeps only what a reader needs, in reading order:
        the label columns (department, category, period), then Budgeted,
        Actual, Variance, Variance % and Status, under standard headings.

        Args:
            include_total:    Append a calculated TOTAL row at the bottom.
            sort_by_variance: Worst overspend first. Turn off to keep the
                              original spreadsheet order.
        """
        label_columns = self.mapping.grouping_columns()

        source_to_display = [
            (self.mapping.budget, SUMMARY_BUDGET),
            (self.mapping.actual, SUMMARY_ACTUAL),
            (self.variance_column, SUMMARY_VARIANCE),
            (self.variance_pct_column, SUMMARY_VARIANCE_PCT),
            (self.flag_column, SUMMARY_FLAG),
            (self.status_column, SUMMARY_STATUS),
        ]

        frame = self.data
        if sort_by_variance:
            frame = frame.sort_values(self.variance_column, ascending=False)

        # Build the table column by column rather than renaming in place. A
        # label column could legitimately be called "Variance", and assembling
        # explicitly means such a clash cannot silently drop a column.
        table = pd.DataFrame(index=range(len(frame)))
        used: List[str] = []

        for column in label_columns:
            name = _unique_name(column, used)
            table[name] = frame[column].values
            used.append(name)

        # Remembers where each standard heading actually ended up, in case a
        # clash forced it to be renamed.
        resolved = {}
        for source, display in source_to_display:
            name = _unique_name(display, used)
            table[name] = frame[source].values
            used.append(name)
            resolved[display] = name

        if include_total and label_columns:
            table = pd.concat(
                [table, self._total_row(table, resolved)], ignore_index=True
            )

        return table

    def _total_row(self, table: pd.DataFrame, resolved: Dict[str, str]) -> pd.DataFrame:
        """
        A calculated TOTAL row for the bottom of the summary table.

        Note the distinction from the TOTAL rows we *removed* from the source
        file. Those were pre-existing aggregates that would have been counted
        twice. This one we compute ourselves from the line items that survived
        cleaning, so it is guaranteed to match the KPI cards above it.
        """
        budget_col = resolved[SUMMARY_BUDGET]
        actual_col = resolved[SUMMARY_ACTUAL]
        variance_col = resolved[SUMMARY_VARIANCE]
        pct_col = resolved[SUMMARY_VARIANCE_PCT]
        status_col = resolved[SUMMARY_STATUS]

        total_budget = float(table[budget_col].sum())
        total_actual = float(table[actual_col].sum())
        total_variance = total_actual - total_budget

        row = {column: None for column in table.columns}
        row[table.columns[0]] = TOTAL_ROW_LABEL
        row[budget_col] = total_budget
        row[actual_col] = total_actual
        row[variance_col] = total_variance
        row[pct_col] = (
            (total_variance / total_budget * 100.0) if total_budget else np.nan
        )
        row[status_col] = _variance_status(total_variance)

        return pd.DataFrame([row])

    def top_overspends(self, limit: int = 5) -> pd.DataFrame:
        """The largest over-budget line items."""
        over = self.data[self.data[self.variance_column] > ZERO_TOLERANCE]
        return over.nlargest(limit, self.variance_column)

    def top_underspends(self, limit: int = 5) -> pd.DataFrame:
        """The largest under-budget line items."""
        under = self.data[self.data[self.variance_column] < -ZERO_TOLERANCE]
        return under.nsmallest(limit, self.variance_column)


def _safe_percentage(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Percentage that returns NaN instead of raising or producing infinity
    when the denominator is zero.

    A budgeted amount of 0 with actual spending is a real scenario (unbudgeted
    spend). Dividing by it yields infinity, which would poison every chart and
    average downstream, so we mark it as "not applicable" instead.
    """
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator) * 100.0


def analyze_budget(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    exclude_total_rows: bool = True,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> BudgetAnalysis:
    """
    Run the full analysis pipeline.

    Args:
        df:                  Cleaned DataFrame from data_loader.
        mapping:             Which columns hold budget / actual / labels.
        exclude_total_rows:  Drop TOTAL-style rows before computing anything.
        threshold_pct:       How far over budget (as a % of budget) a line may
                             go before it is flagged red.

    Raises:
        ValueError: budget or actual column is missing or not in the DataFrame.
    """
    if not mapping.is_complete:
        raise ValueError(
            "Both a budgeted-amount column and an actual-amount column must be "
            "selected before the budget can be analysed."
        )
    for role, column in (("budgeted", mapping.budget), ("actual", mapping.actual)):
        if column not in df.columns:
            raise ValueError(
                "The selected {} column '{}' is not in this sheet.".format(role, column)
            )
    if mapping.budget == mapping.actual:
        raise ValueError(
            "The budgeted and actual columns must be different columns."
        )

    working = df.copy()
    warnings: List[str] = []

    # Step 1 — turn amount columns into real numbers.
    for column in (mapping.budget, mapping.actual):
        before = working[column]
        working[column] = parse_currency_series(before)
        newly_missing = int(working[column].isna().sum() - before.isna().sum())
        if newly_missing > 0:
            warnings.append(
                "{} value(s) in '{}' could not be read as a number and were "
                "treated as blank.".format(newly_missing, column)
            )

    # Step 2 — separate aggregate rows from real line items.
    amount_columns = [mapping.budget, mapping.actual]
    total_mask = find_total_rows(working, amount_columns)
    if exclude_total_rows and total_mask.any():
        excluded = working[total_mask].copy()
        working = working[~total_mask].copy()
        warnings.append(
            "Excluded {} total/subtotal row(s) so their amounts are not counted "
            "twice.".format(len(excluded))
        )
    else:
        excluded = working.iloc[0:0].copy()

    # Step 3 — drop rows with no amounts at all; they carry no information.
    blank_amounts = working[mapping.budget].isna() & working[mapping.actual].isna()
    if blank_amounts.any():
        warnings.append(
            "Ignored {} row(s) with no budgeted or actual amount.".format(
                int(blank_amounts.sum())
            )
        )
        working = working[~blank_amounts].copy()

    # Treat a missing amount on one side as zero, so a line item budgeted but
    # never spent still appears (as a full underspend) rather than vanishing.
    working[mapping.budget] = working[mapping.budget].fillna(0.0)
    working[mapping.actual] = working[mapping.actual].fillna(0.0)

    working = working.reset_index(drop=True)

    # Step 4 — compute variance.
    #
    # Sign convention: variance = actual - budget, so POSITIVE means overspent.
    # This is the convention for expense budgets, where spending more than
    # planned is the thing you want flagged.
    variance_col = _unique_column_name(working, "Variance")
    variance_pct_col = _unique_column_name(working, "Variance %")
    status_col = _unique_column_name(working, "Status")
    flag_col = _unique_column_name(working, "Flag")

    working[variance_col] = working[mapping.actual] - working[mapping.budget]
    working[variance_pct_col] = _safe_percentage(
        working[variance_col], working[mapping.budget]
    )
    working[status_col] = working[variance_col].apply(_variance_status)

    # A suffixed name means the file brought its own Variance column, which we
    # deliberately ignored. Say so: spreadsheets differ on which way round the
    # subtraction goes, and a reader comparing our figure against theirs needs
    # to know the signs may be opposite rather than assume one of them is a bug.
    if variance_col != "Variance":
        warnings.append(
            "Your file already has a 'Variance' column. It was left untouched "
            "and not used — variance is recalculated here as actual minus "
            "budgeted, so a positive number always means over budget. If your "
            "file subtracts the other way round, the two columns will have "
            "opposite signs."
        )

    # Step 5 — flag the lines that are materially over budget.
    working[flag_col] = flag_series(working[variance_pct_col], threshold_pct)

    analysis = BudgetAnalysis(
        data=working,
        excluded_rows=excluded,
        mapping=mapping,
        variance_column=variance_col,
        variance_pct_column=variance_pct_col,
        status_column=status_col,
        flag_column=flag_col,
        warnings=warnings,
        threshold_pct=threshold_pct,
    )
    analysis.kpis = compute_kpis(analysis)
    return analysis


def _variance_status(variance: float) -> str:
    """Label a single variance figure."""
    if pd.isna(variance):
        return "Unknown"
    if variance > ZERO_TOLERANCE:
        return "Over budget"
    if variance < -ZERO_TOLERANCE:
        return "Under budget"
    return "On budget"


def compute_kpis(analysis: BudgetAnalysis) -> Dict[str, float]:
    """Headline figures for the top of the report."""
    df = analysis.data
    budget_col = analysis.mapping.budget
    actual_col = analysis.mapping.actual
    variance_col = analysis.variance_column

    total_budget = float(df[budget_col].sum())
    total_actual = float(df[actual_col].sum())
    total_variance = total_actual - total_budget

    over = df[df[variance_col] > ZERO_TOLERANCE]
    under = df[df[variance_col] < -ZERO_TOLERANCE]
    flagged = df[df[analysis.flag_column] == FLAG_OVER]

    return {
        "threshold_pct": analysis.threshold_pct,
        "flagged_count": len(flagged),
        "flagged_amount": float(flagged[variance_col].sum()) if len(flagged) else 0.0,
        "total_budget": total_budget,
        "total_actual": total_actual,
        "total_variance": total_variance,
        "variance_pct": (total_variance / total_budget * 100.0) if total_budget else np.nan,
        "utilization_pct": (total_actual / total_budget * 100.0) if total_budget else np.nan,
        "line_items": len(df),
        "over_budget_count": len(over),
        "under_budget_count": len(under),
        "on_budget_count": len(df) - len(over) - len(under),
        "largest_overspend": float(over[variance_col].max()) if len(over) else 0.0,
        "largest_underspend": float(under[variance_col].min()) if len(under) else 0.0,
        "excluded_rows": len(analysis.excluded_rows),
    }


# ---------------------------------------------------------------------------
# KPI presentation
# ---------------------------------------------------------------------------
@dataclass
class KpiCard:
    """
    One dashboard card, ready to draw.

    The engine decides *what the number is and how it reads*; the UI decides
    only where to put the box. Keeping the wording here means the same cards
    could be rendered to a PDF or an email later without touching this logic —
    and it keeps app.py free of financial reasoning.
    """

    label: str
    value: str
    delta: Optional[str] = None
    delta_color: str = "normal"   # "normal" | "inverse" | "off"
    help_text: str = ""


def build_kpi_cards(analysis: "BudgetAnalysis") -> List[KpiCard]:
    """
    The four headline financial metrics, plus supporting counts.

    Card order is deliberate: budgeted, actual, the gap between them, then how
    much of the budget that gap represents. A reader should be able to stop
    after the first four and still have the whole story.
    """
    kpis = analysis.kpis
    over_budget = kpis["total_variance"] > ZERO_TOLERANCE

    return [
        KpiCard(
            label="Total budget",
            value=format_currency(kpis["total_budget"]),
            help_text="Sum of the budgeted amount across all line items.",
        ),
        KpiCard(
            label="Total actual",
            value=format_currency(kpis["total_actual"]),
            help_text="Sum of the amount actually spent across all line items.",
        ),
        KpiCard(
            label="Variance",
            value=format_currency(kpis["total_variance"]),
            delta="Over budget" if over_budget else "Under budget",
            # Streamlit paints a positive delta green. For spending that is
            # backwards, so the colour is inverted: overspending shows red.
            delta_color="inverse",
            help_text="Actual minus budget. Positive means over budget.",
        ),
        KpiCard(
            label="Variance %",
            value=format_percentage(kpis["variance_pct"]),
            delta="{:.1f}% of budget used".format(kpis["utilization_pct"])
            if pd.notna(kpis["utilization_pct"]) else None,
            delta_color="off",
            help_text="Variance as a share of the total budget.",
        ),
        KpiCard(
            label="Line items",
            value="{:,}".format(kpis["line_items"]),
            help_text="Rows analysed, after excluding totals and blank rows.",
        ),
        KpiCard(
            label="Flagged 🔴",
            value="{:,}".format(kpis["flagged_count"]),
            delta="over {:.0f}% threshold".format(kpis["threshold_pct"]),
            delta_color="off",
            help_text="Line items more than the threshold over budget. These "
                      "are the accounts to look at first.",
        ),
        KpiCard(
            label="Under budget",
            value="{:,}".format(kpis["under_budget_count"]),
            help_text="Line items that spent less than was budgeted.",
        ),
        KpiCard(
            label="Largest overspend",
            value=format_currency(kpis["largest_overspend"]),
            help_text="The single worst over-budget line item.",
        ),
    ]


def headline_sentence(analysis: "BudgetAnalysis") -> str:
    """
    One plain-English sentence describing the overall result.

    Written by rule, not by AI. It is also the baseline we will compare
    Claude's executive summary against in Milestone 5 — if the AI cannot beat
    a sentence built from an if-statement, it is not earning its place.
    """
    kpis = analysis.kpis
    direction = "over" if kpis["total_variance"] > ZERO_TOLERANCE else "under"

    return (
        "Across {:,} line items, actual spending of {} against a budget of {} "
        "leaves the organisation {} budget by {} ({}).".format(
            kpis["line_items"],
            format_currency(kpis["total_actual"]),
            format_currency(kpis["total_budget"]),
            direction,
            format_currency(abs(kpis["total_variance"])),
            format_percentage(kpis["variance_pct"]),
        )
    )


def format_currency(value: float, symbol: str = "$") -> str:
    """Render a number as currency for display. NaN becomes an em dash."""
    if value is None or pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    return "{}{}{:,.0f}".format(sign, symbol, abs(value))


def format_percentage(value: float) -> str:
    """Render a percentage for display, with an explicit + on overspend."""
    if value is None or pd.isna(value):
        return "—"
    return "{:+.1f}%".format(value)
