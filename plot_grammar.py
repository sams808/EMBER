"""
plot_grammar.py — the L2 declarative plot grammar (framework-agnostic core).

One spec shape replaces the per-workspace hardcoded plot dispatch:

    PlotSpec(mark="bar", y="WasteSiteId", x="wt_pct", hue="WastePhase",
             title=..., log_x=True)
    render_plot(panel, df, spec)          # qt side, in plot_grammar_render

Specs are plain data — JSON-serializable via to_dict()/from_dict(), hashable,
diffable — which is what makes saved perspectives, guided templates, and
(Stage 3) .ember project files possible. The legacy plot_helpers renderers
are frozen: kept so existing perspectives render, but no new code targets
them.

House rules the grammar owns centrally (so no view can drift):
- Horizontal bars for categorical rankings (the app-wide convention);
  `y` is the category column, `x` the value column.
- Cardinality guardrails: a spec that would draw more than the guardrail
  limits (bars, facets, heatmap cells) is trimmed loudly, never silently.
- Unit-honest labels: the axis label for a value column comes from the
  engine's basis (VALUE_AXIS_LABELS), not from whatever the caller types.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from matrix_science import log10_safe
from plot_helpers import (
    ACCENT_COLOR,
    BASE_COLOR,
    bar_figure_height,
    composable,
    heatmap_ytick_step,
    make_unique_plot_labels,
    numeric_plot_series,
    positive_value_frame,
)
from query_engine import VALUE_AXIS_LABELS

# Marks implemented by the Stage-2 renderer, each with the label a UI combo
# shows for it. The registry is deliberately one mapping, not scattered
# string literals: validation, UI combos, and the renderer dispatch all
# read these names, so a mark can't exist in the grammar but be
# untranslatable in a workspace (or vice versa).
MARK_LABELS: Dict[str, str] = {
    "bar": "Bar",
    "stackedbar": "Stacked bar",
    "heatmap": "Heatmap",
    "scatter": "Scatter",
    "box": "Box (distribution)",
}
MARKS: Tuple[str, ...] = tuple(MARK_LABELS)
MARK_BY_LABEL: Dict[str, str] = {label: mark for mark, label in MARK_LABELS.items()}

PALETTES: Tuple[str, ...] = ("okabe-ito", "viridis", "single")

# Cardinality guardrails (blueprint critic requirement): trim loudly instead
# of drawing an unreadable figure or hanging on a 500-facet request.
MAX_BARS = 60
MAX_HUE_LEVELS = 12
MAX_FACETS = 12
MAX_HEATMAP_CELLS = 4000

# House plot colors. ACCENT_COLOR/BASE_COLOR come from plot_helpers so the
# grammar and the legacy renderers can never drift into two different
# "house accents" on one screen.
BAR_COLOR = ACCENT_COLOR
SCATTER_COLOR = "#4C78A8"


@dataclass(frozen=True)
class PlotSpec:
    """A declarative plot description. Frozen: derive variants with
    `dataclasses.replace` (re-exported as `spec_replace`), never mutate."""

    mark: str = "bar"
    x: str = ""                      # value column (bar/box) or x dimension (scatter/heatmap value)
    y: str = ""                      # category column (bar/box/heatmap rows) or y value (scatter)
    hue: Optional[str] = None        # series/color dimension (grouped/stacked bars, heatmap columns)
    facet: Optional[str] = None      # small-multiples dimension
    log_x: bool = False
    log_y: bool = False
    errorbars: Optional[str] = None  # column holding symmetric absolute errors for x (bar) / y (scatter)
    top_n: int = 30                  # bar/box: keep the top N categories by value; <=0 means "all (guardrailed)"
    sort: str = "value"              # "value" | "label" | "none" (bar/box category ordering)
    title: str = ""
    subtitle: str = ""               # the global scope stamp lands here
    xlabel: str = ""                 # empty -> derived from VALUE_AXIS_LABELS when x is an engine value column
    ylabel: str = ""
    palette: str = "okabe-ito"
    annotations: Tuple[Dict, ...] = ()  # e.g. ({"kind": "vline", "value": 100.0, "label": "limit"},)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["annotations"] = [dict(a) for a in self.annotations]
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "PlotSpec":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown PlotSpec field(s): {unknown}")
        payload = dict(data)
        if "annotations" in payload:
            payload["annotations"] = tuple(dict(a) for a in payload["annotations"])
        return cls(**payload)


def spec_replace(spec: PlotSpec, **changes) -> PlotSpec:
    return replace(spec, **changes)


def validate_spec(spec: PlotSpec, columns: List[str]) -> None:
    """Refuse loudly (ValueError) rather than render something misleading —
    same refusal discipline as the query engine."""
    if spec.mark not in MARKS:
        raise ValueError(f"Unknown mark {spec.mark!r}; expected one of {list(MARKS)}")
    if spec.palette not in PALETTES:
        raise ValueError(f"Unknown palette {spec.palette!r}; expected one of {list(PALETTES)}")
    if spec.sort not in ("value", "label", "none"):
        raise ValueError(f"Unknown sort {spec.sort!r}; expected 'value', 'label', or 'none'")
    required = {"x": spec.x, "y": spec.y}
    for role, col in required.items():
        if not col:
            raise ValueError(f"PlotSpec.{role} is required for mark {spec.mark!r}")
        if col not in columns:
            raise ValueError(f"PlotSpec.{role}={col!r} is not a column of the data ({sorted(columns)})")
    for role, col in (("hue", spec.hue), ("facet", spec.facet), ("errorbars", spec.errorbars)):
        if col is not None and col not in columns:
            raise ValueError(f"PlotSpec.{role}={col!r} is not a column of the data ({sorted(columns)})")
    if spec.hue is not None and spec.hue in (spec.x, spec.y):
        raise ValueError(f"PlotSpec.hue={spec.hue!r} duplicates an axis role")
    if spec.facet is not None and spec.facet in (spec.x, spec.y, spec.hue):
        raise ValueError(f"PlotSpec.facet={spec.facet!r} duplicates another role")
    if spec.mark == "heatmap" and spec.hue is None:
        raise ValueError("mark 'heatmap' needs hue (the column dimension of the matrix)")
    if spec.mark == "heatmap" and spec.facet is not None:
        raise ValueError("mark 'heatmap' does not support facet")
    if spec.facet is not None and spec.hue is not None:
        raise ValueError("Use one of hue or facet, not both (combined rendering is a later stage)")


def default_axis_label(column: str) -> str:
    """Unit-honest label for engine value columns; the column name itself
    for anything else (better than inventing wording)."""
    return VALUE_AXIS_LABELS.get(column, column)


# ======================================================================
# The renderer. One @composable entry point (render_plot) makes every
# PlotSpec composable: the Figure Composer recipe is just
# (render_plot, (data, spec), {}) -- data and spec are both picklable and
# carry no live widget state. Drawing goes only through the panel
# interface (figure/canvas/ax/reset_axes/set_figure_size_inches/
# show_message), so the same call replays into ComposerCellPanel.
# ======================================================================

# Okabe-Ito colorblind-safe categorical palette (house default for hue).
OKABE_ITO = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#8a8a8a",
)


def _hue_colors(n: int, palette: str) -> List[str]:
    if palette == "single":
        return [BASE_COLOR] * n
    if palette == "viridis":
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        return [cmap(0.1 + 0.8 * (i / max(1, n - 1))) for i in range(n)]
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]


def _apply_titles(panel, ax, spec: PlotSpec, notes: List[str]) -> None:
    """Title on the axes when there is no subtitle; suptitle + small gray
    axes-title when there is one (suggested_caption prefers suptitle, so
    the caption stays the real title either way). Trim notes ride on the
    subtitle line -- loud, on the figure itself, per the no-silent-caps
    rule."""
    sub_parts = [p for p in ([spec.subtitle] + notes) if p]
    subtitle = " · ".join(sub_parts)
    if subtitle:
        panel.figure.suptitle(spec.title)
        ax.set_title(subtitle, fontsize=9, color="0.35")
    else:
        ax.set_title(spec.title)


def _draw_annotations(ax, spec: PlotSpec) -> None:
    for ann in spec.annotations:
        kind = ann.get("kind")
        value = ann.get("value")
        if value is None:
            continue
        if kind == "vline":
            ax.axvline(float(value), color="0.3", linestyle="--", linewidth=1.0)
            if ann.get("label"):
                ax.text(float(value), 0.98, str(ann["label"]), transform=ax.get_xaxis_transform(),
                        fontsize=8, color="0.3", ha="left", va="top", rotation=90)
        elif kind == "hline":
            ax.axhline(float(value), color="0.3", linestyle="--", linewidth=1.0)
            if ann.get("label"):
                ax.text(0.99, float(value), str(ann["label"]), transform=ax.get_yaxis_transform(),
                        fontsize=8, color="0.3", ha="right", va="bottom")


def _trimmed_categories(
    pdf: pd.DataFrame, spec: PlotSpec, notes: List[str], limit: int,
) -> pd.DataFrame:
    """Keep the top `limit` categories BY VALUE (trimming an alphabetical
    listing by position would drop arbitrary categories), then order them
    per spec.sort -- saying on the figure when anything was dropped.
    Input has one row per category with a _cat_value total column."""
    total = len(pdf)
    if total > limit:
        pdf = pdf.nlargest(limit, "_cat_value")
        notes.append(f"top {limit} of {total} {spec.y} shown")
    if spec.sort == "label":
        return pdf.sort_values("_cat", key=lambda s: s.astype(str))
    if spec.sort == "value":
        return pdf.sort_values("_cat_value", ascending=False)
    return pdf


def _bar_series(data: pd.DataFrame, spec: PlotSpec) -> Optional[pd.DataFrame]:
    """Category/value rows for bar/box marks: numeric and strictly
    positive, via the same filter the legacy bar renderers use."""
    pdf = positive_value_frame(data, spec.y, spec.x, value_name="_val")
    return None if pdf.empty else pdf


@composable
def render_plot(panel, data: pd.DataFrame, spec: PlotSpec) -> None:
    """Render one PlotSpec onto a PlotWidget-like panel."""
    if data is None or data.empty:
        panel.show_message("No data to plot")
        return
    try:
        validate_spec(spec, list(data.columns))
    except ValueError as exc:
        panel.show_message(f"Plot spec: {exc}")
        return

    notes: List[str] = []
    if spec.facet is not None:
        _render_faceted(panel, data, spec, notes)
    elif spec.mark in ("bar", "stackedbar"):
        _render_bars(panel, data, spec, notes)
    elif spec.mark == "heatmap":
        _render_heatmap(panel, data, spec, notes)
    elif spec.mark == "scatter":
        _render_scatter(panel, data, spec, notes)
    elif spec.mark == "box":
        _render_box(panel, data, spec, notes)
    panel.canvas.draw_idle()


def _bar_limit(spec: PlotSpec) -> int:
    return min(spec.top_n, MAX_BARS) if spec.top_n > 0 else MAX_BARS


def _render_bars(panel, data: pd.DataFrame, spec: PlotSpec, notes: List[str]) -> None:
    pdf = _bar_series(data, spec)
    if pdf is None:
        panel.show_message("No positive values to plot")
        return

    if spec.hue is None:
        grouped = pdf.groupby(spec.y, dropna=False, sort=False).agg(
            _cat_value=("_val", "sum")).reset_index().rename(columns={spec.y: "_cat"})
        # Keep per-category passenger columns (Units, errorbars) only when
        # they are unambiguous -- a category spanning several rows has no
        # single error bar or unit.
        singles = pdf.groupby(spec.y, dropna=False).size()
        unambiguous = bool((singles == 1).all())
        if unambiguous:
            extras = pdf.set_index(spec.y)
            for col in filter(None, [spec.errorbars, "Units"]):
                if col in pdf.columns:
                    grouped[col] = extras[col].reindex(grouped["_cat"]).to_numpy()
        grouped = _trimmed_categories(grouped, spec, notes, _bar_limit(spec)).iloc[::-1]
        labels = make_unique_plot_labels(
            grouped.rename(columns={"_cat": spec.y}), spec.y).astype(str)

        ax = panel.reset_axes()
        panel.set_figure_size_inches(
            panel.figure.get_size_inches()[0], bar_figure_height(len(grouped)))
        xerr = None
        if spec.errorbars and spec.errorbars in grouped.columns:
            xerr = numeric_plot_series(grouped[spec.errorbars]).to_numpy(dtype=float)
        elif spec.errorbars:
            notes.append("error bars need one row per category")
        ax.barh(labels, grouped["_cat_value"].to_numpy(dtype=float),
                color=BASE_COLOR if spec.palette == "single" else BAR_COLOR,
                xerr=xerr, ecolor="0.3", capsize=2 if xerr is not None else 0)
    else:
        pivot = pdf.pivot_table(index=spec.y, columns=spec.hue, values="_val",
                                aggfunc="sum", fill_value=0.0)
        hue_levels = list(pivot.sum(axis=0).sort_values(ascending=False).index)
        if len(hue_levels) > MAX_HUE_LEVELS:
            notes.append(f"top {MAX_HUE_LEVELS} of {len(hue_levels)} {spec.hue} series shown")
            hue_levels = hue_levels[:MAX_HUE_LEVELS]
        pivot = pivot[hue_levels]
        cats = pivot.sum(axis=1).rename("_cat_value").reset_index().rename(columns={spec.y: "_cat"})
        cats = _trimmed_categories(cats, spec, notes, _bar_limit(spec)).iloc[::-1]
        pivot = pivot.loc[cats["_cat"]]
        if spec.errorbars:
            notes.append("error bars not drawn for grouped/stacked marks")

        ax = panel.reset_axes()
        per_bar = 0.26 if spec.mark == "stackedbar" else 0.30
        panel.set_figure_size_inches(
            panel.figure.get_size_inches()[0], bar_figure_height(len(pivot), per_bar=per_bar))
        colors = _hue_colors(len(hue_levels), spec.palette)
        y_pos = np.arange(len(pivot))
        labels = [str(v) for v in pivot.index]
        if spec.mark == "stackedbar":
            left = np.zeros(len(pivot))
            for level, color in zip(hue_levels, colors):
                vals = pivot[level].to_numpy(dtype=float)
                ax.barh(y_pos, vals, left=left, color=color, label=str(level))
                left += vals
        else:
            band = 0.8 / len(hue_levels)
            for i, (level, color) in enumerate(zip(hue_levels, colors)):
                offset = (i - (len(hue_levels) - 1) / 2.0) * band
                ax.barh(y_pos + offset, pivot[level].to_numpy(dtype=float),
                        height=band * 0.9, color=color, label=str(level))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8 if len(labels) <= 45 else 6)
        ax.legend(title=spec.hue, fontsize=8, loc="lower right", framealpha=0.9)

    if spec.log_x:
        ax.set_xscale("log")
    ax.set_xlabel(spec.xlabel or default_axis_label(spec.x))
    if spec.ylabel:
        ax.set_ylabel(spec.ylabel)
    ax.grid(True, axis="x", alpha=0.25)
    _draw_annotations(ax, spec)
    _apply_titles(panel, ax, spec, notes)
    panel.figure.tight_layout()


def _render_heatmap(panel, data: pd.DataFrame, spec: PlotSpec, notes: List[str]) -> None:
    pdf = data.copy()
    pdf["_val"] = numeric_plot_series(pdf[spec.x])
    pdf = pdf.dropna(subset=[spec.y, spec.hue])
    if pdf.empty or pdf["_val"].dropna().empty:
        panel.show_message("No data to plot")
        return
    pivot = pdf.pivot_table(index=spec.y, columns=spec.hue, values="_val", aggfunc="sum")
    pivot = pivot.loc[pivot.sum(axis=1, skipna=True).sort_values(ascending=False).index]
    max_rows = max(1, MAX_HEATMAP_CELLS // max(1, pivot.shape[1]))
    if len(pivot) > max_rows:
        notes.append(f"top {max_rows} of {len(pivot)} {spec.y} rows shown")
        pivot = pivot.head(max_rows)

    values = pivot.to_numpy(dtype=float)
    if spec.log_x:
        # Same non-positive -> NaN policy as every other log view in the
        # app (matrix_science.log10_safe), not a second local copy.
        values = log10_safe(values)

    panel.figure.clear()
    n_rows, n_cols = values.shape
    height = min(max(4.5, n_rows * 0.22), 18.0)
    width = min(max(7.0, n_cols * 0.35), 18.0)
    panel.set_figure_size_inches(width, height)
    ax = panel.figure.add_subplot(111)
    panel.ax = ax
    import matplotlib
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(color="0.92")
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=90, fontsize=7)
    step = heatmap_ytick_step(n_rows)
    ax.set_yticks(range(0, n_rows, step))
    ax.set_yticklabels([str(i) for i in pivot.index[::step]], fontsize=7)
    base_label = spec.xlabel or default_axis_label(spec.x)
    cbar_label = f"log10 {base_label}" if spec.log_x else base_label
    panel.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    _draw_annotations(ax, spec)
    _apply_titles(panel, ax, spec, notes)
    panel.figure.tight_layout()


def _render_scatter(panel, data: pd.DataFrame, spec: PlotSpec, notes: List[str]) -> None:
    pdf = data.copy()
    pdf["_x"] = numeric_plot_series(pdf[spec.x])
    pdf["_y"] = numeric_plot_series(pdf[spec.y])
    pdf = pdf.dropna(subset=["_x", "_y"])
    if spec.log_x:
        pdf = pdf[pdf["_x"] > 0]
    if spec.log_y:
        pdf = pdf[pdf["_y"] > 0]
    if pdf.empty:
        panel.show_message("No plottable points")
        return

    ax = panel.reset_axes()
    if spec.hue is None:
        ax.scatter(pdf["_x"], pdf["_y"], s=28,
                   color=BASE_COLOR if spec.palette == "single" else SCATTER_COLOR, alpha=0.8)
    else:
        levels = list(pdf.groupby(spec.hue, dropna=False).size().sort_values(ascending=False).index)
        if len(levels) > MAX_HUE_LEVELS:
            notes.append(f"top {MAX_HUE_LEVELS} of {len(levels)} {spec.hue} series shown")
            levels = levels[:MAX_HUE_LEVELS]
        colors = _hue_colors(len(levels), spec.palette)
        for level, color in zip(levels, colors):
            sub = pdf[pdf[spec.hue] == level]
            ax.scatter(sub["_x"], sub["_y"], s=28, color=color, alpha=0.8, label=str(level))
        ax.legend(title=spec.hue, fontsize=8, framealpha=0.9)
    yerr = None
    if spec.errorbars and spec.errorbars in pdf.columns and spec.hue is None:
        yerr = numeric_plot_series(pdf[spec.errorbars]).to_numpy(dtype=float)
        ax.errorbar(pdf["_x"], pdf["_y"], yerr=yerr, fmt="none", ecolor="0.5",
                    elinewidth=1.0, capsize=2, zorder=0)
    if spec.log_x:
        ax.set_xscale("log")
    if spec.log_y:
        ax.set_yscale("log")
    ax.set_xlabel(spec.xlabel or default_axis_label(spec.x))
    ax.set_ylabel(spec.ylabel or default_axis_label(spec.y))
    ax.grid(True, alpha=0.25)
    _draw_annotations(ax, spec)
    _apply_titles(panel, ax, spec, notes)
    panel.figure.tight_layout()


def _render_box(panel, data: pd.DataFrame, spec: PlotSpec, notes: List[str]) -> None:
    pdf = _bar_series(data, spec)
    if pdf is None:
        panel.show_message("No positive values to plot")
        return
    grouped = pdf.groupby(spec.y, dropna=False)["_val"]
    order = grouped.sum().rename("_cat_value").reset_index().rename(columns={spec.y: "_cat"})
    order = _trimmed_categories(order, spec, notes, _bar_limit(spec)).iloc[::-1]
    cats = list(order["_cat"])
    series = [grouped.get_group(c).to_numpy(dtype=float) for c in cats]

    ax = panel.reset_axes()
    panel.set_figure_size_inches(
        panel.figure.get_size_inches()[0], bar_figure_height(len(cats), per_bar=0.30))
    ax.boxplot(series, orientation="horizontal", tick_labels=[str(c) for c in cats],
               flierprops={"markersize": 3, "alpha": 0.5})
    if spec.log_x:
        ax.set_xscale("log")
    ax.set_xlabel(spec.xlabel or default_axis_label(spec.x))
    ax.grid(True, axis="x", alpha=0.25)
    _draw_annotations(ax, spec)
    _apply_titles(panel, ax, spec, notes)
    panel.figure.tight_layout()


def _render_faceted(panel, data: pd.DataFrame, spec: PlotSpec, notes: List[str]) -> None:
    if spec.mark not in ("bar", "scatter"):
        panel.show_message(f"facet is supported for bar and scatter marks, not {spec.mark!r}")
        return
    pdf = data.dropna(subset=[spec.facet]).copy()
    if pdf.empty:
        panel.show_message("No data to plot")
        return
    value_for_rank = numeric_plot_series(pdf[spec.x]).fillna(0.0)
    totals = value_for_rank.groupby(pdf[spec.facet]).sum().sort_values(ascending=False)
    levels = list(totals.index)
    if len(levels) > MAX_FACETS:
        notes.append(f"top {MAX_FACETS} of {len(levels)} {spec.facet} facets shown")
        levels = levels[:MAX_FACETS]

    panel.figure.clear()
    n_cols = min(3, len(levels))
    n_rows = (len(levels) + n_cols - 1) // n_cols
    height = min(max(4.5, n_rows * 3.4), 16.0)
    width = min(max(7.0, n_cols * 3.6), 18.0)
    panel.set_figure_size_inches(width, height)
    axes = panel.figure.subplots(n_rows, n_cols, squeeze=False)
    per_facet = spec_replace(spec, facet=None, title="", subtitle="")
    for i, level in enumerate(levels):
        ax = axes[i // n_cols][i % n_cols]
        sub = pdf[pdf[spec.facet] == level]
        _draw_facet_cell(ax, sub, per_facet)
        ax.set_title(str(level), fontsize=9)
    for j in range(len(levels), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_axis_off()
    panel.ax = axes[0][0]
    panel.figure.suptitle(spec.title)
    footer = " · ".join(p for p in ([spec.subtitle] + notes) if p)
    if footer:
        panel.figure.text(0.5, 0.005, footer, ha="center", fontsize=8, color="0.35")
    panel.figure.tight_layout(rect=(0, 0.02 if footer else 0, 1, 0.96))


def _draw_facet_cell(ax, sub: pd.DataFrame, spec: PlotSpec) -> None:
    """One facet's worth of bar/scatter, kept deliberately lean (no
    per-facet legends or notes -- the shared suptitle/footer carry those)."""
    if spec.mark == "bar":
        vals = numeric_plot_series(sub[spec.x])
        pdf = sub.assign(_val=vals).dropna(subset=[spec.y, "_val"])
        pdf = pdf[pdf["_val"] > 0]
        if pdf.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    fontsize=8, color="0.5")
            ax.set_axis_off()
            return
        grouped = (pdf.groupby(spec.y, dropna=False)["_val"].sum()
                   .sort_values(ascending=False).head(min(_bar_limit(spec), 15)).iloc[::-1])
        ax.barh([str(v) for v in grouped.index], grouped.to_numpy(dtype=float),
                color=BASE_COLOR if spec.palette == "single" else BAR_COLOR)
        if spec.log_x:
            ax.set_xscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="x", alpha=0.25)
    else:
        x = numeric_plot_series(sub[spec.x])
        y = numeric_plot_series(sub[spec.y])
        keep = x.notna() & y.notna()
        if spec.log_x:
            keep &= x > 0
        if spec.log_y:
            keep &= y > 0
        ax.scatter(x[keep], y[keep], s=18,
                   color=BASE_COLOR if spec.palette == "single" else SCATTER_COLOR, alpha=0.8)
        if spec.log_x:
            ax.set_xscale("log")
        if spec.log_y:
            ax.set_yscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
