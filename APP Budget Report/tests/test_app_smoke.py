"""
Smoke test for the Streamlit app.

Run from the project root:

    python tests/test_app_smoke.py

Uses Streamlit's own AppTest harness, which executes app.py headlessly — no
browser, no clicking. It ticks the "use the sample budget file" checkbox and
then asserts the app rendered without raising and produced sensible KPIs.

This is how you catch a broken UI in one second instead of discovering it
during a demo.
"""

import json
import re
import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)


@contextmanager
def no_api_key_anywhere():
    """
    Pretend the machine has no Claude API key.

    Both the environment variable and the `.env` loader are suppressed. Once a
    developer adds a real key, a test that only pops the variable silently
    starts testing the opposite of what it claims to.
    """
    import src.ai_client as ai_client

    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    saved_loader = ai_client.load_dotenv
    ai_client.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        ai_client.load_dotenv = saved_loader
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key


def _widget(elements, label):
    """
    Find a widget by its visible label.

    Indexing into app.selectbox[0] is fragile: Streamlit lists main-area
    widgets before sidebar ones, so positions shift whenever the layout
    changes. Looking a widget up by label is stable.
    """
    for element in elements:
        if element.label == label:
            return element
    raise AssertionError(
        "No widget labelled '{}'. Available: {}".format(
            label, [e.label for e in elements]
        )
    )


SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "sample_budget.xlsx",
)


def _run_with_sample():
    """
    Start the app with the sample file already loaded.

    AppTest cannot drive a real `st.file_uploader` — there is no way to hand it
    bytes. So the app reads an optional `preloaded_file` key from session state
    as a test seam, and the tests write it here. Nothing in the UI sets that
    key, so users never see this path.
    """
    app = AppTest.from_file(APP_PATH, default_timeout=60)

    with open(SAMPLE_PATH, "rb") as handle:
        app.session_state["preloaded_file"] = {
            "bytes": handle.read(),
            "name": "sample_budget.xlsx",
        }

    app.run()
    assert not app.exception, app.exception
    return app


def test_app_starts_without_a_file():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception, app.exception
    assert any("Upload a budget spreadsheet" in i.value for i in app.info)


def test_app_analyses_the_sample_file():
    app = _run_with_sample()

    labels = [m.label for m in app.metric]
    for expected in ("Total budget", "Total actual", "Variance", "Variance %"):
        assert expected in labels, "Missing KPI '{}'. Got: {}".format(expected, labels)

    values = {m.label: m.value for m in app.metric}
    # The clean sample sheet totals $2,250,000 budgeted against $2,378,900.
    assert values["Total budget"] == "$2,250,000", values["Total budget"]
    assert values["Total actual"] == "$2,378,900", values["Total actual"]
    assert values["Variance"] == "$128,900", values["Variance"]
    assert values["Variance %"] == "+5.7%", values["Variance %"]
    assert values["Line items"] == "16", values["Line items"]


def test_summary_table_is_rendered_with_a_total_row():
    """The clean summary table must appear, and its TOTAL must match the KPIs."""
    app = _run_with_sample()

    tables = [element.value for element in app.dataframe]
    summaries = [
        t for t in tables
        if "Budgeted" in t.columns and "Variance %" in t.columns
        and "TOTAL" in t.iloc[:, 0].astype(str).values
    ]
    assert summaries, "No summary table with a TOTAL row was rendered"

    total = summaries[0].iloc[-1]
    assert total["Budgeted"] == 2250000
    assert total["Actual"] == 2378900
    assert total["Variance"] == 128900


def test_app_renders_all_six_charts():
    """
    Every visualisation must reach the page, with real traces, in the order
    they are drawn: variance by group, variance % by group, budget-vs-actual
    by quarter, difference by quarter, the pie, then the department drill-down.

    Budget-vs-actual and the difference chart are period-based and
    deliberately independent of the "Group by" selector — they must show
    Quarter, not whatever department/category happens to be selected.
    """
    app = _run_with_sample()

    charts = app.get("plotly_chart")
    assert len(charts) == 6, "Expected 6 charts, got {}".format(len(charts))

    # `.value` on a plotly element reads selection state, which these charts do
    # not use. The figure itself travels to the browser as JSON in `.spec`.
    variance, variance_pct, by_quarter, quarter_diff, pie, detail = [
        json.loads(chart.proto.spec)["data"] for chart in charts
    ]

    # The drill-down charts one department's categories, not every department.
    assert [trace["name"] for trace in detail] == ["Budgeted", "Actual"]

    assert variance[0]["type"] == "bar"
    assert variance[0]["orientation"] == "h"     # diverging, horizontal
    assert variance_pct[0]["type"] == "bar"

    assert [trace["name"] for trace in by_quarter] == ["Budgeted", "Actual"]
    assert by_quarter[0]["type"] == "bar"
    assert list(by_quarter[0]["x"]) == ["Q1", "Q2"]     # the sample's quarters

    assert quarter_diff[0]["type"] == "bar"
    assert list(quarter_diff[0]["x"]) == ["Q1", "Q2"]

    assert pie[0]["type"] == "pie"
    # The pie must show the four departments of the sample file.
    assert set(pie[0]["labels"]) == {
        "Engineering", "Student Services", "Marketing", "Operations",
    }


def test_period_charts_are_independent_of_the_group_by_selector():
    """
    Switching "Group by" to Category must not change the quarter charts.

    They are wired to the period column directly, not to the selector, so
    that Q1/Q2/Q3/Q4 always stays in its own natural sequence regardless of
    what the reader happens to be grouping the other charts by.
    """
    app = _run_with_sample()
    _widget(app.selectbox, "Group by").select("Category").run()
    assert not app.exception, app.exception

    charts = app.get("plotly_chart")
    by_quarter = json.loads(charts[2].proto.spec)["data"]
    assert list(by_quarter[0]["x"]) == ["Q1", "Q2"]


def test_missing_period_column_shows_guidance_not_an_error():
    """A file with no quarter/period mapped must degrade gracefully."""
    app = _run_with_sample()

    # Clear the Period mapping in the sidebar.
    _widget(app.selectbox, "Period (optional)").select("(none)").run()
    assert not app.exception, app.exception

    messages = [i.value for i in app.info]
    assert any("period column" in m.lower() for m in messages), messages


def test_department_dropdown_drills_into_one_department():
    """
    Picking a department must re-scope its figures, chart and line items —
    the whole point of the drill-down. If the numbers didn't change, the
    dropdown would be decoration.
    """
    app = _run_with_sample()

    picker = _widget(app.selectbox, "Select a department to see its spending")
    assert set(picker.options) == {
        "Engineering", "Student Services", "Marketing", "Operations",
    }, picker.options

    picker.select("Marketing").run()
    assert not app.exception, app.exception

    values = {m.label: m.value for m in app.metric}
    # Marketing in the sample: $330,000 budgeted, $383,000 actual.
    assert values["Budgeted"] == "$330,000", values["Budgeted"]
    assert values["Actual"] == "$383,000", values["Actual"]
    assert values["Over / under budget"] == "$53,000", values["Over / under budget"]

    # Switching departments must actually change the figures.
    picker = _widget(app.selectbox, "Select a department to see its spending")
    picker.select("Operations").run()
    values = {m.label: m.value for m in app.metric}
    assert values["Budgeted"] == "$210,000", values["Budgeted"]


def test_drilldown_variance_label_does_not_collide_with_the_headline_kpi():
    """
    Two metrics labelled "Variance" with different scopes on one page is a
    misreading waiting to happen — the drill-down uses its own wording.
    """
    app = _run_with_sample()
    labels = [m.label for m in app.metric]
    assert labels.count("Variance") == 1, labels
    assert "Over / under budget" in labels


def test_quarterly_workbook_is_combined_automatically():
    """
    A workbook with a sheet per quarter is one budget, not four. It must be
    read as a whole by default — and the summary tab, whose columns differ,
    must be left out of the combination rather than stacked into it.
    """
    import io

    import openpyxl

    workbook = openpyxl.Workbook()
    summary = workbook.active
    summary.title = "Actual vs Budget"
    summary.append(["Quarter", "Total Budget", "Total Actual"])
    summary.append(["Q1", 300, 330])

    for index, quarter in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
        sheet = workbook.create_sheet(quarter)
        sheet.append(["Category", "Department", "Budget", "Actual"])
        sheet.append(["Revenue", "Sales", 100 * index, 110 * index])
        sheet.append(["Payroll", "HR", 200 * index, 190 * index])

    buffer = io.BytesIO()
    workbook.save(buffer)

    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["preloaded_file"] = {
        "bytes": buffer.getvalue(),
        "name": "quarterly.xlsx",
    }
    app.run()
    assert not app.exception, app.exception

    combine = [c for c in app.checkbox if "Combine" in c.label]
    assert combine, [c.label for c in app.checkbox]
    assert "4" in combine[0].label, combine[0].label
    assert combine[0].value is True, "Combining should be the default"

    values = {m.label: m.value for m in app.metric}
    # 2 rows × 4 quarters — the 1-row summary sheet is not part of this.
    assert values["Line items"] == "8", values["Line items"]
    # Budget: (100+200)*(1+2+3+4) = 3000.
    assert values["Total budget"] == "$3,000", values["Total budget"]

    # Turning it off falls back to a single sheet, and the worksheet picker
    # reappears now that there is a choice to make again.
    combine[0].uncheck().run()
    assert not app.exception, app.exception
    assert _widget(app.selectbox, "Worksheet") is not None


def test_currency_in_prose_is_not_rendered_as_latex_maths():
    """
    Streamlit reads `$...$` as a LaTeX maths delimiter.

    Every headline sentence in this app contains two currency amounts, so
    without escaping, "$2,378,900 against a budget of $2,250,000" renders as
    italic symbols with the dollar signs eaten. Caught by looking at the
    running app, not by any assertion about the analysis being correct.
    """
    app = _run_with_sample()

    prose = [e.value for e in app.error] + [e.value for e in app.success]
    sentences = [p for p in prose if "against a budget of" in p]
    assert sentences, "Headline sentence not rendered"

    for sentence in sentences:
        # Every dollar sign must be backslash-escaped.
        assert "\\$" in sentence, sentence
        assert not re.search(r"(?<!\\)\$", sentence), \
            "Unescaped $ will render as LaTeX: {}".format(sentence)


def test_flag_threshold_marks_overspending_red():
    """The 3% rule must reach the KPI cards and the summary table."""
    app = _run_with_sample()

    values = {m.label: m.value for m in app.metric}
    assert "Flagged 🔴" in values, list(values)
    assert int(values["Flagged 🔴"]) > 0, "Sample data has overspends to flag"

    tables = [element.value for element in app.dataframe]
    flagged = [t for t in tables if "Flag" in t.columns]
    assert flagged, "No table carries the Flag column"

    marks = set(flagged[0]["Flag"].astype(str))
    assert any("🔴" in m for m in marks), marks
    assert any("🟢" in m for m in marks), marks


def test_app_has_report_and_chat_tabs():
    app = _run_with_sample()
    labels = [tab.label for tab in app.tabs]
    assert "📊 Report" in labels, labels
    assert "💬 AI Chat" in labels, labels


def test_chat_tab_explains_setup_when_no_api_key():
    """
    Without a key the chat tab must guide the user, not crash or sit blank.

    The key is removed from the environment for this test so the result does
    not depend on whether the machine running the tests happens to have one.
    """
    with no_api_key_anywhere():
        app = _run_with_sample()
        assert not app.exception, app.exception

        messages = [i.value for i in app.info] + [m.value for m in app.markdown]
        joined = "\n".join(messages)
        assert "Claude API key" in joined, "No setup guidance rendered"
        assert "ANTHROPIC_API_KEY" in joined
        # The report itself must still work with no key configured.
        assert "Total budget" in [m.label for m in app.metric]


def test_report_download_works_without_an_api_key():
    """The report must be a complete deliverable with no key configured."""
    with no_api_key_anywhere():
        app = _run_with_sample()
        assert not app.exception, app.exception

        labels = [b.label for b in app.get("download_button")]
        assert any("full report" in label for label in labels), labels
        # No key means no generate button, but the report still downloads.
        assert not any("Generate executive summary" in b.label for b in app.button)


def test_generate_summary_button_appears_when_a_key_is_configured():
    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-placeholder-for-tests"
    try:
        app = _run_with_sample()
        assert not app.exception, app.exception

        labels = [b.label for b in app.button]
        assert any("Generate executive summary" in label for label in labels), labels
    finally:
        if saved is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_chat_ui_appears_when_a_key_is_configured():
    """
    With a key present the chat controls render.

    A placeholder key is used and no question is asked, so this test makes no
    network call and costs nothing — it checks our wiring, not Anthropic's API.
    """
    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-placeholder-for-tests"
    try:
        app = _run_with_sample()
        assert not app.exception, app.exception

        assert len(app.chat_input) == 1, "Expected the chat input to render"

        buttons = [b.label for b in app.button]
        assert any("overspent the most" in label for label in buttons), buttons
        assert any("financial risks" in label for label in buttons), buttons
    finally:
        if saved is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_app_detects_columns_on_the_sample_file():
    app = _run_with_sample()
    selected = [s.value for s in app.selectbox]
    assert "Budgeted Amount" in selected, selected
    assert "Actual Amount" in selected, selected
    assert "Department" in selected, selected


def test_sidebar_has_nothing_but_the_uploader_and_one_settings_link():
    """
    The whole point of this milestone: uploading a file should be the only
    thing anyone has to do. Every other control — sheet, header row, column
    mapping, threshold — must exist only inside one collapsed expander, never
    loose in the sidebar.
    """
    app = _run_with_sample()
    assert not app.exception, app.exception

    assert [h.value for h in app.sidebar.get("header")] == [], \
        "Sidebar still has numbered section headers"

    expanders = [e.label for e in app.get("expander")]
    assert "⚙️ Adjust settings" in expanders


def test_header_row_auto_detects_without_the_user_setting_it():
    """
    Switching to the sheet with a title row above its real headers must
    already show the right numbers — the point is that nobody has to notice
    the title row exists, let alone count which row the headers are on.
    """
    app = _run_with_sample()

    _widget(app.selectbox, "Worksheet").select("Messy Export").run()
    assert not app.exception, app.exception

    # No manual header-row correction here — this is the auto-detected result.
    assert _widget(app.number_input, "Header row").value == 1

    values = {m.label: m.value for m in app.metric}
    assert values["Line items"] == "18", values["Line items"]
    assert values["Total budget"] == "$2,300,000", values["Total budget"]


def test_detected_mapping_is_shown_without_opening_settings():
    """A reader must be able to see what was assumed without a single click."""
    app = _run_with_sample()

    captions = [c.value for c in app.caption]
    assert any("Budget = **Budgeted Amount**" in c for c in captions), captions
    assert any("Adjust settings" in c for c in captions), captions


def test_app_handles_the_messy_sheet():
    """
    The messy sheet has a title row, text amounts and odd headers.

    Selecting it and setting the header row to 1 must still produce a report.
    """
    app = _run_with_sample()

    worksheet = _widget(app.selectbox, "Worksheet")
    assert "Messy Export" in worksheet.options, worksheet.options
    worksheet.select("Messy Export").run()

    _widget(app.number_input, "Header row").set_value(1).run()
    assert not app.exception, app.exception

    values = {m.label: m.value for m in app.metric}
    # 16 base rows + a refund line + an unspent line = 18; SUBTOTAL excluded.
    assert values["Line items"] == "18", values["Line items"]
    # $2,250,000 base + $50,000 emergency fund = $2,300,000 budgeted.
    assert values["Total budget"] == "$2,300,000", values["Total budget"]
    # $2,378,900 base - $12,500 refund + $0 unspent = $2,366,400.
    assert values["Total actual"] == "$2,366,400", values["Total actual"]

    selected = [s.value for s in app.selectbox]
    assert "Planned Spend" in selected, selected
    assert "Amount Spent" in selected, selected


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
