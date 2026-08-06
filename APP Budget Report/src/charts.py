"""
Plotly chart builders for the budget report.

Same discipline as analyzer.py: this module never imports Streamlit. Each
function returns a `plotly.graph_objects.Figure`, and the UI decides where to
put it. That means charts can be exported to a PDF or an email later without
touching a line of this file.

Design rules applied throughout (they are not decoration — each prevents a
specific way charts mislead people):

  * Colour is assigned by the job it does. Budget vs. actual are two *identities*,
    so they get categorical hues. Nothing is coloured darker-because-bigger,
    which would double-encode bar length and waste the only free channel.
  * One value axis, never two. A second y-scale invents correlations that are
    not in the data.
  * Recessive chrome: hairline gridlines one shade off the surface, no dashed
    grid, generous padding. The data should be the loudest thing on screen.
  * Every chart has a table-view twin in the UI, so no value is reachable only
    by hovering.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from src.analyzer import FLAG_OVER, SUMMARY_FLAG

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Label used when small categories are folded together.
OTHER_LABEL = "Other"


@dataclass(frozen=True)
class Palette:
    """
    Every colour a chart needs, for one theme.

    Both themes are *selected*, not computed by flipping the light one. Dark
    mode uses the same hues stepped for a dark surface — an automatic inversion
    produces colours that vibrate against dark backgrounds.
    """

    surface: str          # chart background
    text_primary: str
    text_secondary: str
    muted: str            # axis tick labels
    grid: str             # hairline gridlines
    axis: str             # baseline / axis rule
    neutral: str          # the "Other" bucket — deliberately colourless
    series: Tuple[str, ...]   # categorical slots, in fixed order

    # Status colours are reserved. They mean good/bad, never "series 3" — a
    # reader who learns that red means over-threshold must never see red used
    # for an ordinary category. They are also mode-invariant: the same two
    # hexes clear contrast on both the light and dark chart surfaces.
    good: str = "#0ca30c"
    critical: str = "#d03b3b"

    def slot(self, index: int) -> str:
        """Categorical colour by position, never cycled past the last slot."""
        return self.series[min(index, len(self.series) - 1)]


# Slot order is the colourblind-safety mechanism, not a cosmetic choice: this
# ordering is validated so that any two *adjacent* slots stay distinguishable
# under colour-vision deficiency. Reordering it silently breaks that.
LIGHT = Palette(
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    neutral="#898781",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)

DARK = Palette(
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    neutral="#898781",
    series=("#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"),
)


def get_palette(theme: str = "light") -> Palette:
    """Pick the palette for the active app theme."""
    return DARK if str(theme).lower() == "dark" else LIGHT


# ---------------------------------------------------------------------------
# Data preparation
#
# Kept separate from the drawing code so the UI can show the exact numbers
# behind each chart as a table. A chart whose values cannot be read as text
# fails accessibility, and is also impossible to check.
# ---------------------------------------------------------------------------
def _fold_into_other(
    df: pd.DataFrame,
    label_column: str,
    value_columns: List[str],
    max_rows: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Keep the largest `max_rows` categories and sum the rest into "Other".

    Folding rather than truncating matters: the chart's total still equals the
    KPI cards above it. Silently dropping the tail would make the chart
    disagree with the headline figures, and the reader would have no way to
    tell which was wrong.
    """
    if len(df) <= max_rows:
        return df, 0

    kept = df.iloc[:max_rows].copy()
    tail = df.iloc[max_rows:]

    other = {column: tail[column].sum() for column in value_columns}
    other[label_column] = OTHER_LABEL

    folded = pd.concat([kept, pd.DataFrame([other])], ignore_index=True)
    return folded, len(tail)


def budget_vs_actual_data(
    analysis,
    group_column: str,
    max_categories: int = 12,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Budget and actual totals per category, largest first.

    Returns (table, notes). Notes are plain-English caveats for the UI to show
    beneath the chart — anything the reader needs in order to trust it.
    """
    budget_col = analysis.mapping.budget
    actual_col = analysis.mapping.actual

    grouped = (
        analysis.data.groupby(group_column, dropna=False)[[budget_col, actual_col]]
        .sum()
        .reset_index()
        .sort_values(actual_col, ascending=False)
        .reset_index(drop=True)
    )

    table, folded = _fold_into_other(
        grouped, group_column, [budget_col, actual_col], max_categories
    )

    notes = []
    if folded:
        notes.append(
            "The {} smallest categories are combined into '{}' so the chart "
            "still totals to the figures above.".format(folded, OTHER_LABEL)
        )

    return table, notes


def spending_by_category_data(
    analysis,
    group_column: str,
    max_slices: int = 6,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Actual spending per category, for the composition chart.

    Two real-world problems are handled here, both of which would otherwise
    produce a nonsensical pie:

      * A pie cannot represent a negative value. Refunds and credits are
        genuinely negative actuals, so they are removed and disclosed.
      * Zero-spend categories add slices of no width but still consume a
        colour, so they are dropped too.
    """
    actual_col = analysis.mapping.actual

    grouped = (
        analysis.data.groupby(group_column, dropna=False)[actual_col]
        .sum()
        .reset_index()
        .sort_values(actual_col, ascending=False)
        .reset_index(drop=True)
    )

    notes = []

    negative = grouped[grouped[actual_col] < 0]
    if not negative.empty:
        notes.append(
            "{} categor{} with negative net spending (refunds or credits) "
            "cannot be shown as a share of a whole and {} excluded: {}.".format(
                len(negative),
                "y" if len(negative) == 1 else "ies",
                "is" if len(negative) == 1 else "are",
                ", ".join(str(v) for v in negative[group_column]),
            )
        )

    positive = grouped[grouped[actual_col] > 0].reset_index(drop=True)

    table, folded = _fold_into_other(
        positive, group_column, [actual_col], max_slices
    )

    if folded:
        notes.append(
            "The {} smallest categories are combined into '{}'. A pie stops "
            "being readable past about six slices.".format(folded, OTHER_LABEL)
        )

    return table, notes


# ---------------------------------------------------------------------------
# Shared chart chrome
# ---------------------------------------------------------------------------
def _apply_base_layout(fig: go.Figure, palette: Palette, height: int) -> go.Figure:
    """Styling every chart shares, so they read as one family."""
    fig.update_layout(
        template="none",
        height=height,
        paper_bgcolor=palette.surface,
        plot_bgcolor=palette.surface,
        font=dict(family=FONT_FAMILY, size=13, color=palette.text_secondary),
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color=palette.text_secondary),
        ),
        hoverlabel=dict(
            bgcolor=palette.surface,
            bordercolor=palette.axis,
            font=dict(family=FONT_FAMILY, size=13, color=palette.text_primary),
        ),
    )
    return fig


def _style_value_axis(fig: go.Figure, palette: Palette, axis: str) -> None:
    """
    The axis carrying money: hairline gridlines, currency ticks, no zero line.

    Solid hairlines only. A dashed grid reads as "projection" or "threshold"
    when it is just a grid.
    """
    settings = dict(
        showgrid=True,
        gridcolor=palette.grid,
        gridwidth=1,
        griddash="solid",
        zeroline=False,
        showline=False,
        tickprefix="$",
        tickformat=",.0f",
        tickfont=dict(color=palette.muted, size=12),
        # Without this, the tight base margin clips "$2,000,000" down to "0".
        # The container must always be sized to include its axis band.
        automargin=True,
        title=None,
    )
    if axis == "y":
        fig.update_yaxes(**settings)
    else:
        fig.update_xaxes(**settings)


def _style_category_axis(fig: go.Figure, palette: Palette, axis: str) -> None:
    """The axis carrying names: a baseline rule, no grid."""
    settings = dict(
        showgrid=False,
        showline=True,
        linecolor=palette.axis,
        linewidth=1,
        zeroline=False,
        ticks="outside",
        tickcolor=palette.axis,
        ticklen=4,
        tickfont=dict(color=palette.text_secondary, size=12),
        automargin=True,
        title=None,
    )
    if axis == "y":
        fig.update_yaxes(**settings)
    else:
        fig.update_xaxes(**settings)


# ---------------------------------------------------------------------------
# Chart 1 — Budget vs Actual
# ---------------------------------------------------------------------------
def budget_vs_actual_bar(
    table: pd.DataFrame,
    group_column: str,
    budget_column: str,
    actual_column: str,
    palette: Optional[Palette] = None,
) -> go.Figure:
    """
    Grouped bars comparing budgeted against actual, per category.

    Why a grouped bar and not something else: the reader's job is to compare
    two *distinct series* within each category, and bars anchored to a common
    zero baseline are the one form where length maps directly to amount. The
    gap between each pair of bars is the variance, made visible without any
    arithmetic.

    Orientation flips to horizontal past six categories, where vertical labels
    would otherwise be rotated and become hard to read.
    """
    palette = palette or LIGHT
    horizontal = len(table) > 6

    labels = table[group_column].astype(str).tolist()
    series = [
        ("Budgeted", table[budget_column], palette.slot(0)),
        ("Actual", table[actual_column], palette.slot(1)),
    ]

    fig = go.Figure()
    for name, values, colour in series:
        # Horizontal bars read bottom-to-top, so the order is reversed to keep
        # the largest category at the top where the eye starts.
        plotted_labels = labels[::-1] if horizontal else labels
        plotted_values = values.tolist()[::-1] if horizontal else values.tolist()

        fig.add_trace(go.Bar(
            name=name,
            x=plotted_values if horizontal else plotted_labels,
            y=plotted_labels if horizontal else plotted_values,
            orientation="h" if horizontal else "v",
            marker=dict(color=colour),
            hovertemplate=(
                "<b>%{" + ("y" if horizontal else "x") + "}</b><br>"
                + name + ": $%{" + ("x" if horizontal else "y") + ":,.0f}"
                + "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        # Thin marks with breathing room. Wide saturated blocks read loud and
        # make the chart feel heavier than the data it carries.
        bargap=0.45,        # space between categories
        bargroupgap=0.06,   # the small surface gap between paired bars
        barcornerradius=4,  # softened data-ends, not full pills
    )

    if horizontal:
        _style_value_axis(fig, palette, "x")
        _style_category_axis(fig, palette, "y")
    else:
        _style_value_axis(fig, palette, "y")
        _style_category_axis(fig, palette, "x")

    height = max(360, 40 * len(table) + 140) if horizontal else 420
    return _apply_base_layout(fig, palette, height)


# ---------------------------------------------------------------------------
# Chart 2 — Spending by Category
# ---------------------------------------------------------------------------
def spending_by_category_pie(
    table: pd.DataFrame,
    group_column: str,
    actual_column: str,
    palette: Optional[Palette] = None,
) -> go.Figure:
    """
    Composition of actual spending, as a share of the whole.

    A pie answers exactly one question — "roughly what share of the money went
    where?" — and answers it at a glance. It is a poor tool for comparing
    similar values, because the eye judges angle badly; that is what the bar
    chart above it is for. Kept to six slices plus "Other" for that reason.

    Every slice carries its own name and percentage, so identity never depends
    on matching a colour to a legend.
    """
    palette = palette or LIGHT

    labels = table[group_column].astype(str).tolist()
    values = table[actual_column].tolist()

    # "Other" is deliberately colourless: it is a bucket, not a category, and
    # giving it a hue would make it compete with the real ones.
    colours = [
        palette.neutral if label == OTHER_LABEL else palette.slot(index)
        for index, label in enumerate(labels)
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        sort=False,             # keep our largest-first order
        direction="clockwise",
        textinfo="label+percent",
        # One decimal place. Plotly's default gives "8.95%", which implies a
        # precision the underlying estimate does not have.
        texttemplate="%{label}<br>%{percent:.1%}",
        textposition="outside",
        automargin=True,
        insidetextfont=dict(color=palette.surface),
        outsidetextfont=dict(color=palette.text_secondary, size=12),
        marker=dict(
            colors=colours,
            # A ring in the surface colour separates slices as a *gap*, rather
            # than drawing a visible border around each one.
            line=dict(color=palette.surface, width=2),
        ),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
    ))

    # One series with direct labels on every slice: a legend box would repeat
    # information already on the chart.
    fig.update_layout(showlegend=False)

    return _apply_base_layout(fig, palette, 460)


# ---------------------------------------------------------------------------
# Chart 3 — Variance by group (which areas are over, and by how much)
# ---------------------------------------------------------------------------
def variance_bar(
    table: pd.DataFrame,
    group_column: str,
    variance_column: str,
    flag_column: str,
    palette: Optional[Palette] = None,
) -> go.Figure:
    """
    Horizontal bars showing the money each area is over or under budget.

    This is the diverging form: bars run right of zero when over budget and
    left when under, so the *sign* is read from direction and the *size* from
    length. The budget-vs-actual chart shows two totals and leaves the reader
    to subtract; this one shows the subtraction.

    Red means over the threshold. Colour is not the only cue — position
    relative to the zero line already says over or under, and the flag column
    in the table beside it says so in words.
    """
    palette = palette or LIGHT

    # Reversed so the largest overspend sits at the top, where the eye starts.
    frame = table.iloc[::-1]
    colours = [
        palette.critical if flag == FLAG_OVER else palette.good
        for flag in frame[flag_column]
    ]

    fig = go.Figure(go.Bar(
        x=frame[variance_column].tolist(),
        y=frame[group_column].astype(str).tolist(),
        orientation="h",
        marker=dict(color=colours),
        hovertemplate="<b>%{y}</b><br>Variance: $%{x:,.0f}<extra></extra>",
    ))

    fig.update_layout(bargap=0.45, barcornerradius=4, showlegend=False)

    # The zero line is the reference the whole chart is read against, so unlike
    # the other charts it is drawn — one shade darker than the grid.
    _style_value_axis(fig, palette, "x")
    fig.update_xaxes(zeroline=True, zerolinecolor=palette.axis, zerolinewidth=1)
    _style_category_axis(fig, palette, "y")

    height = max(320, 40 * len(table) + 120)
    return _apply_base_layout(fig, palette, height)


# ---------------------------------------------------------------------------
# Chart 4 — Variance % against the threshold
# ---------------------------------------------------------------------------
def variance_pct_bar(
    table: pd.DataFrame,
    group_column: str,
    variance_pct_column: str,
    flag_column: str,
    threshold_pct: float,
    palette: Optional[Palette] = None,
) -> go.Figure:
    """
    Variance as a percentage, with the threshold drawn as a line.

    Percentage answers a different question from amount, and the two often
    disagree: a small department 30% over is out of control, while a large one
    1% over is noise even if the dollar figure is larger. This chart makes the
    threshold rule visible — every bar crossing the line is flagged, and the
    reader can see how close the others are to crossing it.
    """
    palette = palette or LIGHT

    frame = table.dropna(subset=[variance_pct_column])
    frame = frame.sort_values(variance_pct_column, ascending=False)

    colours = [
        palette.critical if flag == FLAG_OVER else palette.good
        for flag in frame[flag_column]
    ]

    fig = go.Figure(go.Bar(
        x=frame[group_column].astype(str).tolist(),
        y=frame[variance_pct_column].tolist(),
        marker=dict(color=colours),
        hovertemplate="<b>%{x}</b><br>%{y:+.1f}% vs budget<extra></extra>",
    ))

    # The threshold itself, labelled. A rule the reader cannot see is a rule
    # they have to take on trust.
    fig.add_hline(
        y=threshold_pct,
        line=dict(color=palette.critical, width=1, dash="dot"),
        annotation_text="{:.0f}% threshold".format(threshold_pct),
        # Right-anchored: at "top left" the label sits over the axis band and
        # gets clipped by the margin.
        annotation_position="top right",
        annotation_font=dict(color=palette.critical, size=12),
    )

    fig.update_layout(bargap=0.45, barcornerradius=4, showlegend=False)

    _style_category_axis(fig, palette, "x")
    fig.update_yaxes(
        showgrid=True, gridcolor=palette.grid, gridwidth=1, griddash="solid",
        zeroline=True, zerolinecolor=palette.axis, zerolinewidth=1,
        showline=False, ticksuffix="%", tickformat=".0f",
        tickfont=dict(color=palette.muted, size=12), automargin=True, title=None,
    )

    return _apply_base_layout(fig, palette, 420)


# ---------------------------------------------------------------------------
# Chart 5 — The dollar difference between actual and budgeted, per period
# ---------------------------------------------------------------------------
def period_variance_bar(
    table: pd.DataFrame,
    period_column: str,
    variance_column: str,
    flag_column: str,
    palette: Optional[Palette] = None,
) -> go.Figure:
    """
    Vertical bars showing the dollar variance for each period, left to right
    in the period's own order — Q1, Q2, Q3, Q4, not sorted by which one
    overspent most.

    This is the period twin of variance_bar(), which is used for departments
    and deliberately sorts worst-first. A period axis has a sequence a reader
    already knows; re-sorting it would fight that expectation instead of using
    it. Colour still follows the same rule everywhere else in the app: red
    means past the threshold, never "the bigger bar".
    """
    palette = palette or LIGHT

    colours = [
        palette.critical if flag == FLAG_OVER else palette.good
        for flag in table[flag_column]
    ]

    fig = go.Figure(go.Bar(
        x=table[period_column].astype(str).tolist(),
        y=table[variance_column].tolist(),
        marker=dict(color=colours),
        hovertemplate="<b>%{x}</b><br>Variance: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(bargap=0.45, barcornerradius=4, showlegend=False)

    _style_category_axis(fig, palette, "x")
    _style_value_axis(fig, palette, "y")
    fig.update_yaxes(zeroline=True, zerolinecolor=palette.axis, zerolinewidth=1)

    return _apply_base_layout(fig, palette, 400)
