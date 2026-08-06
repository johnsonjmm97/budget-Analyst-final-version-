"""
Tests for the budget analysis engine.

Run from the project root:

    python tests/test_analyzer.py

Written with plain asserts and a small runner so no extra test framework is
needed. Each test builds a tiny DataFrame, runs one piece of the engine, and
checks the result — this is how you verify logic without clicking through a UI.
"""

import os
import sys

import numpy as np
import pandas as pd

# Allow "import src.analyzer" when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import (  # noqa: E402
    FLAG_OK,
    FLAG_OVER,
    FLAG_UNKNOWN,
    SUMMARY_ACTUAL,
    SUMMARY_BUDGET,
    SUMMARY_FLAG,
    SUMMARY_STATUS,
    SUMMARY_VARIANCE,
    SUMMARY_VARIANCE_PCT,
    ColumnMapping,
    flag_for,
    analyze_budget,
    build_kpi_cards,
    detect_columns,
    find_total_rows,
    headline_sentence,
    looks_numeric,
    parse_currency_value,
)


def test_parse_currency_handles_real_world_formats():
    assert parse_currency_value(1200) == 1200.0
    assert parse_currency_value("$1,200.50") == 1200.50
    assert parse_currency_value("(500)") == -500.0          # accounting negative
    assert parse_currency_value("-1,500") == -1500.0
    assert parse_currency_value("1.200,50") == 1200.50      # European format
    assert parse_currency_value("USD 3 400") == 3400.0
    assert parse_currency_value("12%") == 12.0
    assert np.isnan(parse_currency_value("-"))
    assert np.isnan(parse_currency_value("N/A"))
    assert np.isnan(parse_currency_value(None))
    assert np.isnan(parse_currency_value("no amount"))


def test_looks_numeric_distinguishes_amounts_from_labels():
    assert looks_numeric(pd.Series(["$100", "$200", "$300"]))
    assert looks_numeric(pd.Series([1, 2, 3]))
    assert not looks_numeric(pd.Series(["Marketing", "Engineering", "Operations"]))


def test_detect_columns_finds_budget_and_actual():
    df = pd.DataFrame({
        "Department": ["A", "B"],
        "Budgeted Amount": [100, 200],
        "Actual Amount": [110, 190],
    })
    mapping = detect_columns(df)
    assert mapping.budget == "Budgeted Amount"
    assert mapping.actual == "Actual Amount"
    assert mapping.department == "Department"
    assert mapping.is_complete


def test_detect_columns_ignores_existing_variance_column():
    """A 'Budget Variance' header must not be mistaken for the budget column."""
    df = pd.DataFrame({
        "Cost Center": ["A", "B"],
        "Planned": [100, 200],
        "Spent": [110, 190],
        "Budget Variance": [10, -10],
    })
    mapping = detect_columns(df)
    assert mapping.budget == "Planned"
    assert mapping.actual == "Spent"
    assert mapping.department == "Cost Center"


def test_detect_columns_falls_back_to_position():
    """Unhelpful headers still yield a usable guess."""
    df = pd.DataFrame({
        "Item": ["A", "B"],
        "Col1": [100, 200],
        "Col2": [110, 190],
    })
    mapping = detect_columns(df)
    assert mapping.budget == "Col1"
    assert mapping.actual == "Col2"


def test_find_total_rows_matches_aggregate_labels():
    df = pd.DataFrame({
        "Category": ["Travel", "Subtotal", "Rent", "GRAND TOTAL"],
        "Budget": [10, 20, 30, 60],
    })
    mask = find_total_rows(df, amount_columns=["Budget"])
    assert list(mask) == [False, True, False, True]


def test_analysis_excludes_totals_so_kpis_are_not_doubled():
    df = pd.DataFrame({
        "Category": ["Travel", "Rent", "TOTAL"],
        "Budget": [100, 200, 300],
        "Actual": [150, 180, 330],
    })
    analysis = analyze_budget(df, ColumnMapping(budget="Budget", actual="Actual"))

    assert analysis.kpis["total_budget"] == 300      # not 600
    assert analysis.kpis["total_actual"] == 330      # not 660
    assert analysis.kpis["total_variance"] == 30
    assert analysis.kpis["line_items"] == 2
    assert analysis.kpis["excluded_rows"] == 1


def test_variance_sign_and_status():
    """Positive variance means overspent."""
    df = pd.DataFrame({
        "Category": ["Over", "Under", "Exact"],
        "Budget": [100, 100, 100],
        "Actual": [120, 80, 100],
    })
    analysis = analyze_budget(df, ColumnMapping(budget="Budget", actual="Actual"))

    assert list(analysis.data[analysis.variance_column]) == [20, -20, 0]
    assert list(analysis.data[analysis.status_column]) == [
        "Over budget", "Under budget", "On budget",
    ]
    assert list(analysis.data[analysis.variance_pct_column]) == [20.0, -20.0, 0.0]


def test_analysis_parses_currency_text():
    df = pd.DataFrame({
        "Category": ["Travel", "Rent"],
        "Budget": ["$1,000.00", "$2,000.00"],
        "Actual": ["$1,250.00", "(500)"],
    })
    analysis = analyze_budget(df, ColumnMapping(budget="Budget", actual="Actual"))

    assert analysis.kpis["total_budget"] == 3000.0
    assert analysis.kpis["total_actual"] == 750.0


def test_zero_budget_does_not_produce_infinity():
    """Unbudgeted spending must not create an infinite percentage."""
    df = pd.DataFrame({
        "Category": ["Unbudgeted"],
        "Budget": [0],
        "Actual": [500],
    })
    analysis = analyze_budget(df, ColumnMapping(budget="Budget", actual="Actual"))

    assert analysis.data[analysis.variance_column].iloc[0] == 500
    assert pd.isna(analysis.data[analysis.variance_pct_column].iloc[0])


def test_existing_variance_column_is_not_overwritten():
    df = pd.DataFrame({
        "Category": ["Travel"],
        "Budget": [100],
        "Actual": [120],
        "Variance": [999],          # a wrong value from the source file
    })
    analysis = analyze_budget(df, ColumnMapping(budget="Budget", actual="Actual"))

    assert analysis.variance_column == "Variance (calculated)"
    assert analysis.data["Variance"].iloc[0] == 999
    assert analysis.data["Variance (calculated)"].iloc[0] == 20


def test_grouped_summary_sums_by_department():
    df = pd.DataFrame({
        "Department": ["Sales", "Sales", "IT"],
        "Budget": [100, 100, 500],
        "Actual": [150, 150, 400],
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", department="Department")
    analysis = analyze_budget(df, mapping)
    grouped = analysis.by_group("Department")

    sales = grouped[grouped["Department"] == "Sales"].iloc[0]
    assert sales["Budget"] == 200
    assert sales["Actual"] == 300
    assert sales[analysis.variance_column] == 100
    # Sorted worst-first: Sales (+100) above IT (-100).
    assert grouped.iloc[0]["Department"] == "Sales"


def test_incomplete_mapping_raises():
    df = pd.DataFrame({"Budget": [100], "Actual": [120]})
    for bad_mapping in (
        ColumnMapping(budget="Budget", actual=None),
        ColumnMapping(budget="Budget", actual="Missing Column"),
        ColumnMapping(budget="Budget", actual="Budget"),
    ):
        try:
            analyze_budget(df, bad_mapping)
        except ValueError:
            continue
        raise AssertionError("Expected ValueError for mapping: {}".format(bad_mapping))


# ---------------------------------------------------------------------------
# Milestone 3: KPI cards and the clean summary table
# ---------------------------------------------------------------------------
def _sample_analysis():
    """A small analysis used by several of the reporting tests."""
    df = pd.DataFrame({
        "Department": ["Sales", "IT", "HR"],
        "Budget": [1000, 2000, 500],
        "Actual": [1200, 1800, 500],
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", department="Department")
    return analyze_budget(df, mapping)


def test_kpi_cards_cover_the_four_headline_metrics():
    cards = build_kpi_cards(_sample_analysis())
    labels = [card.label for card in cards]

    for expected in ("Total budget", "Total actual", "Variance", "Variance %"):
        assert expected in labels, "Missing KPI card '{}'".format(expected)

    values = {card.label: card.value for card in cards}
    assert values["Total budget"] == "$3,500"
    assert values["Total actual"] == "$3,500"
    assert values["Variance"] == "$0"
    assert values["Variance %"] == "+0.0%"


def test_variance_card_uses_inverse_colour():
    """Overspending must show red, not green."""
    cards = {c.label: c for c in build_kpi_cards(_sample_analysis())}
    assert cards["Variance"].delta_color == "inverse"


def test_headline_sentence_states_the_result_in_english():
    df = pd.DataFrame({
        "Department": ["Sales"],
        "Budget": [1000],
        "Actual": [1250],
    })
    analysis = analyze_budget(
        df, ColumnMapping(budget="Budget", actual="Actual", department="Department")
    )
    sentence = headline_sentence(analysis)

    assert "over budget" in sentence
    assert "$250" in sentence
    assert "$1,250" in sentence
    assert "$1,000" in sentence


def test_summary_table_has_standard_headings_in_reading_order():
    table = _sample_analysis().summary_table()

    assert list(table.columns) == [
        "Department",
        SUMMARY_BUDGET,
        SUMMARY_ACTUAL,
        SUMMARY_VARIANCE,
        SUMMARY_VARIANCE_PCT,
        SUMMARY_FLAG,
        SUMMARY_STATUS,
    ]
    # Worst overspend first.
    assert table.iloc[0]["Department"] == "Sales"
    assert table.iloc[0][SUMMARY_VARIANCE] == 200


def test_summary_table_renames_source_columns_to_standard_headings():
    """A file calling it 'Planned Spend' still reports as 'Budgeted'."""
    df = pd.DataFrame({
        "Cost Centre": ["Ops"],
        "Planned Spend": [100],
        "Amount Spent": [150],
    })
    analysis = analyze_budget(df, detect_columns(df))
    table = analysis.summary_table()

    assert SUMMARY_BUDGET in table.columns
    assert "Planned Spend" not in table.columns
    assert table.iloc[0][SUMMARY_BUDGET] == 100


def test_summary_table_total_row_matches_the_kpis():
    analysis = _sample_analysis()
    table = analysis.summary_table(include_total=True)
    total = table.iloc[-1]

    assert total["Department"] == "TOTAL"
    assert total[SUMMARY_BUDGET] == analysis.kpis["total_budget"]
    assert total[SUMMARY_ACTUAL] == analysis.kpis["total_actual"]
    assert total[SUMMARY_VARIANCE] == analysis.kpis["total_variance"]
    # The total row is extra, not a replacement for a line item.
    assert len(table) == len(analysis.data) + 1


def test_summary_table_can_keep_original_order():
    table = _sample_analysis().summary_table(sort_by_variance=False)
    assert list(table["Department"]) == ["Sales", "IT", "HR"]


def test_summary_table_survives_a_column_named_variance():
    """A label column called 'Status' must not collide with our own."""
    df = pd.DataFrame({
        "Status": ["Approved", "Pending"],
        "Budget": [100, 200],
        "Actual": [150, 150],
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", category="Status")
    table = analyze_budget(df, mapping).summary_table()

    # Both survive: the source column keeps its name, ours is suffixed.
    assert "Status" in table.columns
    assert "Status (2)" in table.columns
    assert len(table.columns) == len(set(table.columns))


# ---------------------------------------------------------------------------
# The 3% flag threshold
# ---------------------------------------------------------------------------
def test_flag_marks_only_overspending_past_the_threshold():
    assert flag_for(10.0) == FLAG_OVER      # 10% over → red
    assert flag_for(3.1) == FLAG_OVER       # just past → red
    assert flag_for(3.0) == FLAG_OK         # exactly at the line → not over
    assert flag_for(2.9) == FLAG_OK
    assert flag_for(0.0) == FLAG_OK


def test_underspending_is_never_flagged_red():
    """
    Being under budget is a different question with a different answer.

    Colouring it red would train the reader to ignore red, which defeats the
    whole point of having a flag.
    """
    assert flag_for(-50.0) == FLAG_OK
    assert flag_for(-3.1) == FLAG_OK


def test_unmeasurable_variance_is_neither_red_nor_green():
    """A zero budget gives no percentage — that is not the same as 'fine'."""
    assert flag_for(float("nan")) == FLAG_UNKNOWN
    assert flag_for(None) == FLAG_UNKNOWN


def test_threshold_is_configurable():
    assert flag_for(4.0, threshold_pct=5.0) == FLAG_OK
    assert flag_for(6.0, threshold_pct=5.0) == FLAG_OVER


def test_analysis_flags_line_items_and_counts_them():
    df = pd.DataFrame({
        "Department": ["Marketing", "Engineering", "Ops"],
        "Budget": [100, 1000, 500],
        "Actual": [110, 1020, 480],          # +10%, +2%, -4%
    })
    analysis = analyze_budget(
        df, ColumnMapping(budget="Budget", actual="Actual", department="Department")
    )

    assert list(analysis.data[analysis.flag_column]) == [FLAG_OVER, FLAG_OK, FLAG_OK]
    assert analysis.kpis["flagged_count"] == 1
    assert analysis.kpis["flagged_amount"] == 10
    assert analysis.kpis["threshold_pct"] == 3.0


def test_groups_are_flagged_on_their_combined_percentage():
    """
    A department containing one bad line can still be fine overall.

    The group flag answers "is this department over?"; the line-item table
    answers "where". Flagging a group because any row inside it is flagged
    would make every large department permanently red.
    """
    df = pd.DataFrame({
        "Department": ["Sales", "Sales"],
        "Budget": [100, 10000],
        "Actual": [130, 10000],              # +30% on a tiny line, +0.3% overall
    })
    analysis = analyze_budget(
        df, ColumnMapping(budget="Budget", actual="Actual", department="Department")
    )

    assert list(analysis.data[analysis.flag_column]) == [FLAG_OVER, FLAG_OK]
    grouped = analysis.by_group("Department")
    assert grouped.iloc[0][SUMMARY_FLAG] == FLAG_OK
    assert analysis.flagged_groups("Department").empty


def test_by_group_ordered_preserves_file_order_unlike_by_group():
    df = pd.DataFrame({
        "Quarter": ["Q1", "Q4", "Q2", "Q3"],
        "Budget": [100, 100, 100, 100],
        "Actual": [110, 90, 200, 105],   # Q2 is the worst overspend
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", period="Quarter")
    analysis = analyze_budget(df, mapping)

    ordered = analysis.by_group_ordered("Quarter")
    assert list(ordered["Quarter"]) == ["Q1", "Q4", "Q2", "Q3"]   # file order

    sorted_by_variance = analysis.by_group("Quarter")
    assert sorted_by_variance.iloc[0]["Quarter"] == "Q2"          # worst first

    # Same underlying figures either way — only the row order differs.
    assert set(zip(ordered["Quarter"], ordered[SUMMARY_FLAG])) == \
        set(zip(sorted_by_variance["Quarter"], sorted_by_variance[SUMMARY_FLAG]))


def test_custom_threshold_flows_through_the_analysis():
    df = pd.DataFrame({
        "Department": ["A"],
        "Budget": [100],
        "Actual": [104],                     # +4%
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", department="Department")

    strict = analyze_budget(df, mapping, threshold_pct=3.0)
    lenient = analyze_budget(df, mapping, threshold_pct=5.0)

    assert strict.data[strict.flag_column].iloc[0] == FLAG_OVER
    assert lenient.data[lenient.flag_column].iloc[0] == FLAG_OK
    assert lenient.kpis["flagged_count"] == 0


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
