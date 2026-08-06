"""
Assemble a downloadable Markdown report.

The point of the download is that the numbers and the narrative travel
together. A CSV of figures with the AI summary left behind in the browser is
half a deliverable — the reader gets the data and none of the reading of it.

Markdown rather than PDF: it opens in any editor, pastes into Word or Google
Docs with its formatting intact, and renders on GitHub. No extra dependency.
"""

from typing import Optional

from src.analyzer import format_currency, format_percentage, headline_sentence

# Kept in sync with budget_context: enough detail to stand alone, not so much
# that the report becomes the spreadsheet again.
MAX_GROUP_ROWS = 25


def _markdown_table(analysis, group_column: str, limit: int) -> str:
    """Render one grouped breakdown as a real Markdown table."""
    grouped = analysis.by_group(group_column).head(limit)

    lines = [
        "| {} | Budgeted | Actual | Variance | Variance % |".format(group_column),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in grouped.iterrows():
        lines.append("| {} | {} | {} | {} | {} |".format(
            row[group_column],
            format_currency(row[analysis.mapping.budget]),
            format_currency(row[analysis.mapping.actual]),
            format_currency(row[analysis.variance_column]),
            format_percentage(row[analysis.variance_pct_column]),
        ))
    return "\n".join(lines)


def build_markdown_report(
    analysis,
    filename: str = "",
    ai_summary: Optional[str] = None,
) -> str:
    """
    Build the full report.

    Args:
        analysis:   The BudgetAnalysis to report on.
        filename:   Source file name, for provenance.
        ai_summary: Claude's executive summary, if one has been generated.
                    When absent the report still stands on its own — the
                    figures and the rule-written headline are always included.
    """
    kpis = analysis.kpis
    parts = ["# Budget Report"]

    if filename:
        parts.append("Source file: `{}`".format(filename))

    # --- The narrative ----------------------------------------------------
    if ai_summary:
        parts.append("\n## Executive summary\n")
        parts.append(ai_summary.strip())
        # Attribution is not optional. A reader who cannot tell which sentences
        # a model wrote cannot calibrate how much to trust them.
        parts.append(
            "\n*Written by Claude from the computed figures below. "
            "Figures are calculated in pandas, not by the model.*"
        )
    else:
        parts.append("\n## Summary\n")
        parts.append(headline_sentence(analysis))

    # --- The figures ------------------------------------------------------
    parts.append("\n## Key figures\n")
    parts.append("| Metric | Value |")
    parts.append("| --- | ---: |")
    parts.append("| Total budgeted | {} |".format(format_currency(kpis["total_budget"])))
    parts.append("| Total actual | {} |".format(format_currency(kpis["total_actual"])))
    parts.append("| Variance | {} |".format(format_currency(kpis["total_variance"])))
    parts.append("| Variance % | {} |".format(format_percentage(kpis["variance_pct"])))
    parts.append("| Line items | {:,} |".format(kpis["line_items"]))
    parts.append("| Over budget | {:,} |".format(kpis["over_budget_count"]))
    parts.append("| Under budget | {:,} |".format(kpis["under_budget_count"]))

    for group_column in analysis.mapping.grouping_columns():
        parts.append("\n## Totals by {}\n".format(group_column))
        parts.append(_markdown_table(analysis, group_column, MAX_GROUP_ROWS))

    # --- Provenance -------------------------------------------------------
    parts.append("\n## Method and assumptions\n")
    parts.append(
        "- Variance = actual − budgeted. A positive variance means over budget."
    )
    parts.append("- Budgeted column: `{}`. Actual column: `{}`.".format(
        analysis.mapping.budget, analysis.mapping.actual
    ))
    for warning in analysis.warnings:
        parts.append("- {}".format(warning))
    if not analysis.warnings:
        parts.append("- No data issues were detected.")

    return "\n".join(parts) + "\n"
