"""
Tests for the chart builders.

Run from the project root:

    python tests/test_charts.py

A chart is hard to "test" in the sense of checking it looks right — that needs
eyes. What these tests do check is everything that can be wrong *underneath* a
correct-looking picture: wrong numbers, dropped categories, totals that
disagree with the KPIs, colours assigned by rank instead of identity.

Those are the failures that make a chart lie convincingly.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import (  # noqa: E402
    FLAG_OVER,
    SUMMARY_FLAG,
    ColumnMapping,
    analyze_budget,
)
from src.charts import (  # noqa: E402
    DARK,
    LIGHT,
    OTHER_LABEL,
    budget_vs_actual_bar,
    budget_vs_actual_data,
    get_palette,
    period_variance_bar,
    spending_by_category_data,
    spending_by_category_pie,
    variance_bar,
    variance_pct_bar,
)


def _analysis(rows):
    """Build an analysis from (department, budget, actual) tuples."""
    df = pd.DataFrame(rows, columns=["Department", "Budget", "Actual"])
    mapping = ColumnMapping(budget="Budget", actual="Actual", department="Department")
    return analyze_budget(df, mapping)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def test_budget_vs_actual_data_sums_and_sorts():
    analysis = _analysis([
        ("Sales", 100, 150),
        ("Sales", 100, 150),
        ("IT", 900, 800),
    ])
    table, notes = budget_vs_actual_data(analysis, "Department")

    assert list(table["Department"]) == ["IT", "Sales"]   # largest spender first
    assert list(table["Budget"]) == [900, 200]
    assert list(table["Actual"]) == [800, 300]
    assert notes == []


def test_budget_vs_actual_folds_the_tail_without_losing_money():
    """The chart total must still match the KPI cards."""
    rows = [("Dept {}".format(i), 100 * (20 - i), 100 * (20 - i)) for i in range(20)]
    analysis = _analysis(rows)
    table, notes = budget_vs_actual_data(analysis, "Department", max_categories=5)

    assert len(table) == 6                       # 5 real categories + Other
    assert table.iloc[-1]["Department"] == OTHER_LABEL
    assert table["Budget"].sum() == analysis.kpis["total_budget"]
    assert table["Actual"].sum() == analysis.kpis["total_actual"]
    assert any("combined into" in note for note in notes)


def test_pie_data_excludes_negative_spending_and_says_so():
    """
    A refund is a negative actual. A pie cannot draw a negative slice, so it
    must be removed *and disclosed* rather than silently rendered as zero.
    """
    analysis = _analysis([
        ("Marketing", 1000, 900),
        ("Refunds", 0, -500),
    ])
    table, notes = spending_by_category_data(analysis, "Department")

    assert list(table["Department"]) == ["Marketing"]
    assert any("negative" in note for note in notes)
    assert any("Refunds" in note for note in notes)


def test_pie_data_drops_zero_spend_categories():
    analysis = _analysis([("A", 100, 100), ("B", 100, 0)])
    table, _ = spending_by_category_data(analysis, "Department")
    assert list(table["Department"]) == ["A"]


def test_pie_data_caps_slices_at_six_plus_other():
    rows = [("Dept {}".format(i), 100, 100 * (20 - i)) for i in range(12)]
    analysis = _analysis(rows)
    table, notes = spending_by_category_data(analysis, "Department", max_slices=6)

    assert len(table) == 7
    assert table.iloc[-1]["Department"] == OTHER_LABEL
    assert any("six slices" in note for note in notes)


def test_pie_data_totals_match_the_kpis_when_nothing_is_dropped():
    analysis = _analysis([("A", 100, 120), ("B", 200, 180)])
    table, notes = spending_by_category_data(analysis, "Department")

    assert table["Actual"].sum() == analysis.kpis["total_actual"]
    assert notes == []


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_bar_chart_has_two_series_with_fixed_identity_colours():
    """
    Budgeted is always slot 1 and Actual always slot 2 — colour follows the
    entity, never its rank. Otherwise filtering would repaint the survivors
    and a reader who learned "blue is budget" would be misled.
    """
    analysis = _analysis([("Sales", 100, 150), ("IT", 900, 800)])
    table, _ = budget_vs_actual_data(analysis, "Department")
    fig = budget_vs_actual_bar(table, "Department", "Budget", "Actual", LIGHT)

    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["Budgeted", "Actual"]
    assert fig.data[0].marker.color == LIGHT.slot(0)
    assert fig.data[1].marker.color == LIGHT.slot(1)

    # Reversing the sort order must not change which series is which colour.
    reversed_table = table.iloc[::-1].reset_index(drop=True)
    reversed_fig = budget_vs_actual_bar(
        reversed_table, "Department", "Budget", "Actual", LIGHT
    )
    assert reversed_fig.data[0].marker.color == LIGHT.slot(0)


def test_bar_chart_plots_the_values_it_was_given():
    analysis = _analysis([("Sales", 100, 150), ("IT", 900, 800)])
    table, _ = budget_vs_actual_data(analysis, "Department")
    fig = budget_vs_actual_bar(table, "Department", "Budget", "Actual", LIGHT)

    assert list(fig.data[0].y) == [900, 200 - 100]   # IT, Sales budgeted
    assert list(fig.data[1].y) == [800, 150]         # IT, Sales actual


def test_bar_chart_goes_horizontal_for_many_categories():
    """Vertical labels become unreadable past a handful of categories."""
    few = _analysis([("A", 1, 1), ("B", 1, 1)])
    few_table, _ = budget_vs_actual_data(few, "Department")
    assert budget_vs_actual_bar(
        few_table, "Department", "Budget", "Actual", LIGHT
    ).data[0].orientation == "v"

    many = _analysis([("Dept {}".format(i), 1, 1) for i in range(10)])
    many_table, _ = budget_vs_actual_data(many, "Department")
    assert budget_vs_actual_bar(
        many_table, "Department", "Budget", "Actual", LIGHT
    ).data[0].orientation == "h"


def test_bar_chart_uses_a_single_value_axis():
    """Two y-scales on one plot invent correlations that are not in the data."""
    analysis = _analysis([("Sales", 100, 150)])
    table, _ = budget_vs_actual_data(analysis, "Department")
    fig = budget_vs_actual_bar(table, "Department", "Budget", "Actual", LIGHT)

    layout_keys = fig.layout.to_plotly_json().keys()
    assert not [key for key in layout_keys if key.startswith(("xaxis", "yaxis"))
                and key not in ("xaxis", "yaxis")]
    assert all(trace.yaxis in (None, "y") for trace in fig.data)


def test_pie_labels_every_slice_so_colour_is_never_the_only_cue():
    analysis = _analysis([("A", 100, 300), ("B", 100, 200)])
    table, _ = spending_by_category_data(analysis, "Department")
    fig = spending_by_category_pie(table, "Department", "Actual", LIGHT)

    assert fig.data[0].textinfo == "label+percent"
    assert fig.layout.showlegend is False        # labels replace the legend
    assert fig.data[0].sort is False             # our order, not Plotly's


def test_pie_paints_other_in_neutral_grey():
    """'Other' is a bucket, not a category — a hue would make it compete."""
    rows = [("Dept {}".format(i), 100, 100 * (20 - i)) for i in range(12)]
    analysis = _analysis(rows)
    table, _ = spending_by_category_data(analysis, "Department", max_slices=6)
    fig = spending_by_category_pie(table, "Department", "Actual", LIGHT)

    assert fig.data[0].marker.colors[-1] == LIGHT.neutral
    assert fig.data[0].labels[-1] == OTHER_LABEL


def test_both_themes_are_selected_not_flipped():
    assert get_palette("light") is LIGHT
    assert get_palette("dark") is DARK
    assert get_palette(None) is LIGHT            # safe default

    # Same number of slots, different steps chosen for each surface.
    assert len(LIGHT.series) == len(DARK.series) == 8
    assert LIGHT.series != DARK.series
    assert LIGHT.surface != DARK.surface


def test_palette_never_cycles_past_the_last_slot():
    """A generated 9th hue is indistinguishable from an existing one."""
    assert LIGHT.slot(99) == LIGHT.series[-1]


# ---------------------------------------------------------------------------
# The threshold charts
# ---------------------------------------------------------------------------
def test_variance_bar_colours_by_flag_not_by_size():
    """
    Red must mean "over the threshold", never "biggest bar".

    Colouring by magnitude would double-encode bar length and tell the reader
    nothing the length does not already say.
    """
    analysis = _analysis([
        ("Marketing", 100, 130),      # +30% → flagged
        ("Engineering", 1000, 1010),  # +1%  → fine
        ("Ops", 500, 400),            # under → fine
    ])
    grouped = analysis.by_group("Department")
    fig = variance_bar(grouped, "Department", analysis.variance_column,
                       SUMMARY_FLAG, LIGHT)

    # The chart reverses row order so the worst sits at the top.
    flags = list(grouped[SUMMARY_FLAG])[::-1]
    expected = [LIGHT.critical if f == FLAG_OVER else LIGHT.good for f in flags]
    assert list(fig.data[0].marker.color) == expected
    assert LIGHT.critical in expected, "Nothing was flagged — bad fixture"
    assert LIGHT.good in expected


def test_variance_bar_shows_the_zero_line():
    """Direction from zero is what makes a diverging bar readable."""
    analysis = _analysis([("A", 100, 130)])
    grouped = analysis.by_group("Department")
    fig = variance_bar(grouped, "Department", analysis.variance_column,
                       SUMMARY_FLAG, LIGHT)

    assert fig.layout.xaxis.zeroline is True
    assert fig.data[0].orientation == "h"


def test_variance_pct_chart_draws_the_threshold_line():
    """A rule the reader cannot see is a rule they have to take on trust."""
    analysis = _analysis([("A", 100, 130), ("B", 100, 101)])
    grouped = analysis.by_group("Department")
    fig = variance_pct_bar(grouped, "Department", analysis.variance_pct_column,
                           SUMMARY_FLAG, 3.0, LIGHT)

    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert lines, "No threshold line drawn"
    assert lines[0].y0 == 3.0
    assert any("3% threshold" in (a.text or "") for a in fig.layout.annotations)


def test_variance_pct_chart_skips_unmeasurable_rows():
    """A zero budget has no percentage — plotting it as 0% would be a lie."""
    analysis = _analysis([("Funded", 100, 130), ("Unbudgeted", 0, 500)])
    grouped = analysis.by_group("Department")
    fig = variance_pct_bar(grouped, "Department", analysis.variance_pct_column,
                           SUMMARY_FLAG, 3.0, LIGHT)

    assert "Unbudgeted" not in list(fig.data[0].x)
    assert "Funded" in list(fig.data[0].x)


def test_by_group_ordered_keeps_the_files_period_order():
    """
    Alphabetical ordering would put Q10 before Q2 and Apr before Jan.

    The order rows appear in the file is almost always the intended sequence,
    so a period chart must preserve it rather than re-sorting by variance
    (as by_group() correctly does for departments) or alphabetically.
    """
    df = pd.DataFrame({
        "Quarter": ["Q1", "Q1", "Q4", "Q2", "Q3"],
        "Budget": [100, 50, 100, 100, 100],
        "Actual": [110, 40, 80, 120, 130],
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", period="Quarter")
    analysis = analyze_budget(df, mapping)

    table = analysis.by_group_ordered("Quarter")
    assert list(table["Quarter"]) == ["Q1", "Q4", "Q2", "Q3"]   # file order
    assert table.iloc[0]["Budget"] == 150                        # Q1 rows summed
    assert table.iloc[0]["Actual"] == 150

    # by_group(), used for departments, still sorts worst-first — the two
    # methods must not be confused with each other.
    sorted_table = analysis.by_group("Quarter")
    assert list(sorted_table["Quarter"])[0] == "Q3"              # +30, the worst


def test_period_variance_bar_preserves_order_and_colours_by_flag():
    df = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4"],
        "Budget": [100, 100, 100, 100],
        "Actual": [130, 101, 95, 102],   # +30% flagged, rest under threshold
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", period="Quarter")
    analysis = analyze_budget(df, mapping)

    table = analysis.by_group_ordered("Quarter")
    fig = period_variance_bar(table, "Quarter", analysis.variance_column,
                              SUMMARY_FLAG, LIGHT)

    assert list(fig.data[0].x) == ["Q1", "Q2", "Q3", "Q4"]       # never reordered
    assert list(fig.data[0].y) == [30, 1, -5, 2]

    expected_colours = [LIGHT.critical, LIGHT.good, LIGHT.good, LIGHT.good]
    assert list(fig.data[0].marker.color) == expected_colours


def test_budget_vs_actual_bar_accepts_an_ordered_period_table():
    """
    The same drawing function used for departments must work unchanged when
    fed a chronologically-ordered quarter table — no re-sorting inside it.
    """
    df = pd.DataFrame({
        "Quarter": ["Q1", "Q2", "Q3", "Q4"],
        "Budget": [400, 300, 200, 100],
        "Actual": [420, 250, 260, 90],
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", period="Quarter")
    analysis = analyze_budget(df, mapping)

    table = analysis.by_group_ordered("Quarter")
    fig = budget_vs_actual_bar(table, "Quarter", "Budget", "Actual", LIGHT)

    assert list(fig.data[0].x) == ["Q1", "Q2", "Q3", "Q4"]
    assert list(fig.data[0].y) == [400, 300, 200, 100]     # budgeted, unsorted
    assert list(fig.data[1].y) == [420, 250, 260, 90]      # actual, unsorted
    assert fig.data[0].orientation == "v"                  # 4 categories: vertical


def test_status_colours_are_not_categorical_slots():
    """
    Red and green mean good/bad and must never be reused for a series.

    A reader who learns red = over-threshold must not meet red again as
    "category 8".
    """
    for palette in (LIGHT, DARK):
        assert palette.critical not in palette.series
        assert palette.good not in palette.series
    # Mode-invariant: the same two hexes clear contrast on both surfaces.
    assert LIGHT.critical == DARK.critical
    assert LIGHT.good == DARK.good


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0

    for test in tests:
        try:
            test()
            print("  PASS  {}".format(test.__name__))
        except Exception as error:  # noqa: BLE001 - report and continue
            failures += 1
            print("  FAIL  {}: {}".format(test.__name__, error))

    print("\n{} passed, {} failed".format(len(tests) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
