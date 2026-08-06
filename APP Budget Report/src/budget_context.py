"""
Turn a BudgetAnalysis into the text brief we hand to Claude.

This module is the single most important design decision in the AI milestones,
so it is worth stating plainly:

    We send Claude the FIGURES WE ALREADY COMPUTED, never the raw spreadsheet.

Why:

  * Accuracy. Language models are unreliable at arithmetic and excellent at
    explanation. pandas has already summed, grouped and compared every number
    with total precision. Asking Claude to redo that work would introduce
    errors into figures that were correct.
  * Cost and speed. A 5,000-row spreadsheet is a large prompt on every single
    turn of a conversation. A brief of the same data is a fraction of that.
  * Consistency. The chat and the on-screen report are then guaranteed to
    quote the same numbers, because they come from the same object.

The brief is also written to be *cacheable*: the content is deterministic and
contains no timestamps or random ids, so an unchanged budget produces
byte-identical text every turn and Claude's prompt cache can serve it.
"""

from typing import List

import pandas as pd

from src.analyzer import format_currency, format_percentage

# How much detail to include. Enough for the model to answer specific questions
# about real line items; small enough to stay cheap on every turn.
MAX_GROUP_ROWS = 25
MAX_LINE_ITEMS = 15


def _format_group_table(analysis, group_column: str, limit: int) -> List[str]:
    """Render one grouped breakdown as aligned text lines."""
    grouped = analysis.by_group(group_column).head(limit)

    lines = ["| {} | Budgeted | Actual | Variance | Variance % |".format(group_column)]
    for _, row in grouped.iterrows():
        lines.append("| {} | {} | {} | {} | {} |".format(
            row[group_column],
            format_currency(row[analysis.mapping.budget]),
            format_currency(row[analysis.mapping.actual]),
            format_currency(row[analysis.variance_column]),
            format_percentage(row[analysis.variance_pct_column]),
        ))
    return lines


def _format_line_items(analysis, frame: pd.DataFrame) -> List[str]:
    """Render individual line items, labelled with whatever columns exist."""
    label_columns = analysis.mapping.grouping_columns()
    lines = []

    for _, row in frame.iterrows():
        label = " / ".join(str(row[c]) for c in label_columns) or "(unlabelled)"
        lines.append("- {}: budgeted {}, actual {}, variance {} ({})".format(
            label,
            format_currency(row[analysis.mapping.budget]),
            format_currency(row[analysis.mapping.actual]),
            format_currency(row[analysis.variance_column]),
            format_percentage(row[analysis.variance_pct_column]),
        ))
    return lines


def build_context(analysis, filename: str = "") -> str:
    """
    Build the full budget brief.

    Deliberately plain text rather than JSON: it costs fewer tokens, and the
    model reads a labelled table at least as well as a nested object.
    """
    kpis = analysis.kpis
    parts: List[str] = []

    parts.append("# Budget data")
    if filename:
        parts.append("Source file: {}".format(filename))
    parts.append(
        "Sign convention: variance = actual - budgeted. A POSITIVE variance "
        "means OVER budget (more was spent than planned). A negative variance "
        "means under budget."
    )

    # --- Headline figures -------------------------------------------------
    parts.append("\n## Totals")
    parts.append("- Total budgeted: {}".format(format_currency(kpis["total_budget"])))
    parts.append("- Total actual: {}".format(format_currency(kpis["total_actual"])))
    parts.append("- Total variance: {} ({})".format(
        format_currency(kpis["total_variance"]),
        format_percentage(kpis["variance_pct"]),
    ))
    parts.append("- Budget used: {}".format(
        format_percentage(kpis["utilization_pct"]).lstrip("+")
    ))
    parts.append("- Line items analysed: {:,}".format(kpis["line_items"]))
    parts.append("- Over budget: {:,} | Under budget: {:,} | On budget: {:,}".format(
        kpis["over_budget_count"], kpis["under_budget_count"], kpis["on_budget_count"]
    ))

    # --- Grouped breakdowns ----------------------------------------------
    for group_column in analysis.mapping.grouping_columns():
        parts.append("\n## Totals by {}".format(group_column))
        parts.extend(_format_group_table(analysis, group_column, MAX_GROUP_ROWS))

    # --- Extremes ---------------------------------------------------------
    overspends = analysis.top_overspends(MAX_LINE_ITEMS)
    if not overspends.empty:
        parts.append("\n## Largest over-budget line items")
        parts.extend(_format_line_items(analysis, overspends))

    underspends = analysis.top_underspends(MAX_LINE_ITEMS)
    if not underspends.empty:
        parts.append("\n## Largest under-budget line items")
        parts.extend(_format_line_items(analysis, underspends))

    # --- Disclosure -------------------------------------------------------
    #
    # The model is told what we changed, so it can qualify its answers instead
    # of presenting cleaned figures as if they came straight from the file.
    parts.append("\n## Data quality notes")
    if analysis.warnings:
        for warning in analysis.warnings:
            parts.append("- {}".format(warning))
    else:
        parts.append("- No data issues were detected.")

    parts.append(
        "- Column mapping used: budgeted = '{}', actual = '{}'.".format(
            analysis.mapping.budget, analysis.mapping.actual
        )
    )

    return "\n".join(parts)


def estimate_size(context: str) -> dict:
    """
    Rough size of the brief, for display in the UI.

    The character count is exact; the token figure is a ~4-characters-per-token
    approximation, which is close enough to show the user roughly what each
    question costs. For an exact count the API has a token-counting endpoint.
    """
    characters = len(context)
    return {
        "characters": characters,
        "approx_tokens": characters // 4,
        "lines": context.count("\n") + 1,
    }
