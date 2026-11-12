"""
qt_vitrification_rules.py -- the "Loading rules" Vitrification sub-tab:
the interface over glass_rules.

It answers one question for every tank (or waste layer): which published
LAW glass loading rule would set the sodium loading of a glass made from
its salt waste, how far below the alkali limit that is, and what it costs
in glass.

Structure (the same discipline as the Cross-correlation tab):

- Engine-backed and scope-aware. Kilograms come from the query engine, the
  shared GlobalScope is read when a run starts, and the scope a result was
  computed under is stamped on the figure, in Notes and in the export.
- The screen runs on the house worker. Everything it needs is read from
  the widgets before it is submitted, the worker function touches no
  widget, and results come back through bound methods of this widget.
- A hidden tab does not recompute: a scope or dataset change marks it
  stale and it refreshes when it is shown. It also runs once by itself the
  first time it is shown with data loaded.
- Every label describes the run that produced the standing result, never
  the live controls.

The three drawings (rule diagram, sodium by binding rule, largest extra
glass) are module-level functions over plain data, so Figure Composer can
replay them. Hover tooltips, click and double-click come from the shared
plot interaction layer of qt_widgets.PlotWidget: each drawing names its
marks through plot_helpers.mark_points / mark_categories, and the tab
listens to PlotWidget.markClicked. What the tab adds is the link between
a mark and its row of the Units table, and the ring around the selected
unit (SelectionRing), for which each drawing leaves the position of its
units on the axes (RULE_SPOTS_ATTR).
"""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy,
    QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import glass_rules as gr
import plot_helpers as ph
import qt_theme
from data_model import HanfordDataset
from qt_components import EmptyState, NotesBanner, PageHeader, StatTile
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

TAB_TITLE = "Loading rules"
HELP_SECTION = 12                      # NOTICE.md "## 12. Vitrification"
HELP_SUBSECTION = "### 12.5"           # ... and the part of it that describes this tab

PLOT_DIAGRAM = "Rule diagram"
PLOT_SHARES = "Sodium by binding rule"
PLOT_EXTRA = "Largest extra glass"
PLOT_KINDS = [PLOT_DIAGRAM, PLOT_SHARES, PLOT_EXTRA]
TOP_EXTRA_GLASS = 15

T_UNITS = "Units"
T_SUMMARY = "Summary"
T_GROUPS = "Groups"
T_NOTES = "Notes"
TABLE_TITLES = [T_UNITS, T_SUMMARY, T_GROUPS, T_NOTES]

UNIT_CHOICES: Dict[str, str] = {"Tanks": "tank", "Waste layers": "layer"}
WASTE_CHOICES: Dict[str, Optional[Tuple[str, ...]]] = {
    "Supernatant + saltcake": ("Supernatant", "Saltcake"),
    "Saltcake": ("Saltcake",),
    "Supernatant": ("Supernatant",),
    "All phases": None,
}
DEFAULT_WASTE = "Supernatant + saltcake"
TILE_NOUNS: Dict[str, str] = {"tank": "tank", "layer": "layer"}

# Colours of the four kinds of rule (fixed data colours, the same in both
# themes because chart surfaces stay white), their light tints for the
# regions of the diagram, and a darker tone for text set on a tint.
RULE_COLORS: Dict[str, str] = {
    gr.ALKALI: "#c3c2b7", gr.ALKALI_SULFUR: "#6250d6", gr.SULFUR: "#e34948", gr.HALIDE: "#eda100",
}
RULE_TINTS: Dict[str, str] = {
    gr.ALKALI: "#f6f5f2", gr.ALKALI_SULFUR: "#eceafb", gr.SULFUR: "#fce6e6", gr.HALIDE: "#fdf3d9",
}
RULE_TEXT_COLORS: Dict[str, str] = {
    gr.ALKALI: "#52514e", gr.ALKALI_SULFUR: "#4a3aa8", gr.SULFUR: "#b3261e", gr.HALIDE: "#8a5a00",
}
_SECOND_RULE_OF_CLASS = {gr.ALKALI: "#9d9c90"}       # a set with two rules of one class (WTP baseline)

_DATA = getattr(qt_theme, "DATA", {})
INK = _DATA.get("ink", "#0b0b0b")
INK_2 = _DATA.get("secondary", "#52514e")
GRID = _DATA.get("grid", "#e1e0d9")
AXIS = _DATA.get("axis", "#c3c2b7")
SURFACE = _DATA.get("surface", "#ffffff")
_CATEGORICAL = _DATA.get("categorical", ["#2a78d6", "#eb6834", "#1baf7a"])
PHASE_COLORS: Dict[str, str] = dict(_DATA.get(
    "phase", {"Supernatant": "#2a78d6", "Saltcake": "#eb6834", "Sludge": "#1baf7a"}))
SYSTEM_COLORS: Dict[str, str] = {"SST": _CATEGORICAL[0], "DST": _CATEGORICAL[1]}
SYSTEM_LABELS: Dict[str, str] = {"SST": "Single-shell tank", "DST": "Double-shell tank"}
OTHER_GROUP = "Not reported"
OTHER_COLOR = INK_2
BLEND_LABEL = "Blended feed"

# Where each unit of a drawing sits, for the selection ring:
# {unit: ("point", x, y, radius_pt)} or {unit: ("bar", x0, y0, x1, y1)}.
RULE_SPOTS_ATTR = "_ember_rule_spots"

_IDLE_TITLE = "No result yet"
_IDLE_HINT = "Load a dataset, then press Run to screen every tank against the selected loading rules."
_HEADLINE = ("Screening index, not a forecast of glass production: the rules are applied to the waste "
             "as stored, without retrieval, blending or pretreatment.")

_SUBSCRIPTS = {"Na2O": "Na₂O", "K2O": "K₂O", "SO3": "SO₃", "Cr2O3": "Cr₂O₃", "SO4": "SO₄"}


def chem(text: str) -> str:
    """Formulas with real subscripts: 'SO3 / Na2O' -> 'SO₃ / Na₂O'."""
    out = str(text)
    for plain, pretty in _SUBSCRIPTS.items():
        out = out.replace(plain, pretty)
    return out.replace("<=", "≤")


def fmt(value, digits: int = 3) -> str:
    """A number for a sentence or a tooltip: `digits` significant figures,
    thousands separators, blank for a missing value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    if math.isinf(number):
        return "no limit"
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude >= 1000:
        return f"{number:,.0f}"
    decimals = max(digits - 1 - int(math.floor(math.log10(magnitude))), 0)
    return f"{number:,.{decimals}f}"


def region_label(rule_class: str) -> str:
    """The name written on a region of the diagram ('Alkali +' and
    'sulfur rule' on two lines: that region is narrow)."""
    return f"{rule_class} rule".replace(" + ", " +\n")


def rule_color(rule_set: gr.RuleSet, rule_name: str) -> str:
    """The colour of one rule: that of its class, and a darker tone for a
    second rule of the same class so the two stay apart in a stacked bar."""
    rule_class = rule_set.rule_classes.get(rule_name, gr.ALKALI)
    same_class = rule_set.rules_of_class(rule_class)
    if len(same_class) > 1 and rule_name != same_class[0]:
        return _SECOND_RULE_OF_CLASS.get(rule_class, RULE_COLORS[rule_class])
    return RULE_COLORS[rule_class]


# ---------------------------------------------------------------------------
# Naming the marks for the shared interaction layer
# ---------------------------------------------------------------------------
def _mark_points(ax, artist, labels, **details) -> None:
    marker = getattr(ph, "mark_points", None)
    if callable(marker):
        marker(ax, artist, labels, **details)


def _declare_categories(ax, labels: Sequence[str], positions: Sequence[float], **details) -> None:
    """Category `labels[i]` sits at integer position i of the y axis.
    The shared helper sets the ticks and fits their labels to the room;
    without it the ticks are set plainly."""
    extra = details.pop("extra", None)
    declare = getattr(ph, "declare_categories", None)
    if callable(declare):
        declare(ax, labels, axis="y", positions=positions, **details)
    else:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(list(labels))
    marker = getattr(ph, "mark_categories", None)
    if callable(marker):
        marker(ax, labels, axis="y", extra=extra, **details)


class PlotPane(QStackedWidget):
    """The pane that holds the chart (or the empty state before a run).

    Its size hints are fixed. A matplotlib canvas reports the size of
    its figure as its preferred size, and the figure follows the pane:
    left alone, the pane's preferred height would be whatever height it
    was last given, and a page laid out by preferred height (inside a
    scroll area) could then only ever grow."""

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(560, 320)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(340, 280)


class SelectionRing:
    """The outline of the selected unit on whatever is drawn in a
    PlotWidget. It survives a change of drawing: the key is kept and the
    ring is drawn again where the new drawing puts that unit (or not at
    all when the new drawing does not show it)."""

    def __init__(self, plot: PlotWidget) -> None:
        self.plot = plot
        self._key: Optional[str] = None
        self._artist = None

    def key(self) -> Optional[str]:
        return self._key

    def is_drawn(self) -> bool:
        return self._artist is not None

    def spot(self, key: Optional[str]):
        return (getattr(self.plot.ax, RULE_SPOTS_ATTR, None) or {}).get(key)

    def attach(self) -> None:
        """A new drawing replaced the figure's contents (and the ring with it)."""
        self._artist = None
        self._draw()

    def select(self, key: Optional[str]) -> None:
        if key == self._key and (self._artist is not None or key is None):
            return
        self._key = key
        self._remove()
        self._draw()
        self.plot.canvas.draw_idle()

    def _remove(self) -> None:
        if self._artist is not None:
            try:
                self._artist.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
            self._artist = None

    def _draw(self) -> None:
        spot = self.spot(self._key)
        if spot is None:
            return
        ax = self.plot.ax
        accent = qt_theme.PALETTE.get("accent", "#b84a18")
        if spot[0] == "bar":
            from matplotlib.patches import Rectangle
            _kind, x0, y0, x1, y1 = spot
            artist = Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0), fill=False,
                               edgecolor=accent, linewidth=2.0, zorder=18, clip_on=False)
            ax.add_patch(artist)
        else:
            _kind, x, y, radius_pt = spot
            (artist,) = ax.plot([x], [y], marker="o", linestyle="none", markersize=2.0 * radius_pt + 6.0,
                                markerfacecolor="none", markeredgecolor=accent, markeredgewidth=2.0,
                                zorder=18, clip_on=False, scalex=False, scaley=False)
        artist.set_in_layout(False)
        self._artist = artist


# ---------------------------------------------------------------------------
# Drawings (plain data in, a figure out)
# ---------------------------------------------------------------------------
def _titles(panel, ax, title: str, subtitle: str) -> None:
    """The house convention: the title on the figure, the qualifying line
    (rule set, waste, scope) as a small grey axes title."""
    if subtitle:
        width = max(40, int(15.5 * float(panel.figure.get_size_inches()[0])))
        panel.figure.suptitle(title, fontsize=12, fontweight="semibold", color=INK)
        ax.set_title(ph.wrap_subtitle(subtitle.split(ph.SUBTITLE_SEP), width=width), fontsize=8.5,
                     fontweight="normal", color=INK_2, pad=7)
    else:
        ax.set_title(title, fontsize=12, fontweight="semibold", color=INK)


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_size(9.0)
    ax.yaxis.label.set_size(9.0)


def _finish(panel, bottom: float = 0.0) -> None:
    """Fit the margins. `bottom` keeps that fraction of the figure's
    height free under the axes (for a legend that belongs to the figure)."""
    try:
        if bottom > 0:
            panel.figure.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
        else:
            panel.figure.tight_layout()
    except Exception:       # a layout that cannot be solved must not take the drawing with it
        pass
    panel.canvas.draw_idle()


def diagram_ranges(x: Sequence[float], y: Sequence[float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Axis ranges of the rule diagram: the standard window, widened to
    hold the data where it reaches further, never by more than a factor
    of 20 downwards (a unit with almost no sulfate is drawn at the edge)."""
    def one(values, default):
        lo, hi = default
        positive = np.asarray([v for v in values if v is not None and np.isfinite(v) and v > 0], dtype=float)
        if positive.size:
            lo = max(min(lo, float(positive.min()) / 1.25), default[0] / 20.0)
            hi = max(hi, float(positive.max()) * 1.25)
        return lo, hi
    return one(x, gr.DIAGRAM_X_RANGE), one(y, gr.DIAGRAM_Y_RANGE)


def _region_spots(grid: gr.RegionGrid) -> List[Tuple[str, float, float, str, str]]:
    """Where to write each region's name: centred on the stretch of the
    bottom edge it occupies, or at the right of the top edge for a region
    that does not reach the bottom."""
    ny, _nx = grid.class_index.shape
    bottom, top = max(int(0.035 * ny), 0), min(int(0.95 * ny), ny - 1)
    spots = []
    for index, name in enumerate(gr.RULE_CLASSES):
        mask = grid.class_index == index
        if not mask.any():
            continue
        if mask[bottom].any():
            columns = np.nonzero(mask[bottom])[0]
            x = math.sqrt(grid.x[columns[0]] * grid.x[columns[-1]])
            spots.append((name, x, grid.y[bottom], "center", "bottom"))
        elif mask[top].any():
            columns = np.nonzero(mask[top])[0]
            x = grid.x[columns[-1]] / 1.06
            spots.append((name, x, grid.y[top], "right", "top"))
        else:
            rows, columns = np.nonzero(mask)
            middle = len(rows) // 2
            spots.append((name, grid.x[columns[middle]], grid.y[rows[middle]], "center", "center"))
    return spots


def _contour_spots(grid: gr.RegionGrid, levels: Sequence[float]) -> List[Tuple[float, float, float, str]]:
    """(level, x, y, rule class there) where each loading contour crosses
    one row of the grid, a quarter of the way up (above the region
    names): the place its label goes, and the region it lies in."""
    row = int(0.24 * grid.loading.shape[0])
    line = grid.loading[row]
    spots = []
    for level in levels:
        below = np.nonzero(line < level)[0]
        if not below.size or below[0] == 0:
            continue
        i = below[0]
        w0, w1 = line[i - 1], line[i]
        t = (w0 - level) / (w0 - w1) if w0 != w1 else 0.0
        x = math.exp(math.log(grid.x[i - 1]) + t * (math.log(grid.x[i]) - math.log(grid.x[i - 1])))
        nearest = i if t > 0.5 else i - 1
        spots.append((float(level), x, float(grid.y[row]), gr.RULE_CLASSES[int(grid.class_index[row, nearest])]))
    return spots


def _quiet_spot(u: np.ndarray, v: np.ndarray, anchor: Tuple[float, float]) -> Tuple[float, float]:
    """Where a one-line note can sit with the fewest markers under it.

    u, v: the markers in axes fractions; anchor: the point the note is
    about. Candidates keep clear of the region names (bottom edge, top
    right corner) and of the contour labels above the bottom edge; among
    equally quiet ones the nearest to the anchor wins."""
    candidates = [(0.26, 0.33), (0.52, 0.33), (0.74, 0.33), (0.26, 0.50), (0.74, 0.50), (0.26, 0.68),
                  (0.52, 0.68), (0.76, 0.68), (0.26, 0.88), (0.52, 0.88), (0.74, 0.82)]
    best, best_key = candidates[0], None
    for cu, cv in candidates:
        covered = int(((np.abs(u - cu) < 0.21) & (np.abs(v - cv) < 0.055)).sum())
        key = (covered, math.hypot(cu - anchor[0], cv - anchor[1]))
        if best_key is None or key < best_key:
            best, best_key = (cu, cv), key
    return best


def _off_axes_note(row: pd.Series) -> Optional[str]:
    """For a unit drawn at the edge: where it really is."""
    if row["_x"] == row["SO3_per_Na2O"] and row["_y"] == row["Halide_term_per_Na2O"]:
        return None
    return (f"drawn at the edge; SO₃ / Na₂O = {fmt(row['SO3_per_Na2O']) or '0'}, "
            f"halide term = {fmt(row['Halide_term_per_Na2O']) or '0'}")


@ph.composable
def plot_rule_diagram(
    panel, points: pd.DataFrame, blend: Optional[dict], rule_set_name: str, k2o_ratio: float,
    group_col: Optional[str], title: str, subtitle: str = "", unit_noun: str = "tank",
) -> None:
    """Every unit on the plane of the loading rules.

    x = SO3 / Na2O, y = the rule set's halide term per Na2O, both
    logarithmic. The background is tinted by the rule that binds there,
    contour lines give the Na2O loading the rules allow, one marker per
    unit has an area proportional to its sodium, and a diamond marks all
    units blended into one feed.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    rules = gr.get_rule_set(rule_set_name)
    pts = points[points["Na2O_t"] > 0].copy() if points is not None and len(points) else None
    if pts is None or pts.empty:
        panel.show_message(f"No {unit_noun} in this selection holds sodium.")
        return
    ax = panel.reset_axes()
    k_ratio = float(k2o_ratio) if k2o_ratio is not None and np.isfinite(k2o_ratio) else 0.0

    blend_x = blend.get("SO3_per_Na2O") if blend else None
    blend_y = blend.get("Halide_term_per_Na2O") if blend else None
    x_range, y_range = diagram_ranges(list(pts["SO3_per_Na2O"]) + [blend_x],
                                      list(pts["Halide_term_per_Na2O"]) + [blend_y])
    grid = gr.region_grid(rules, k2o_ratio=k_ratio, x_range=x_range, y_range=y_range)
    xx, yy = np.meshgrid(grid.x, grid.y)
    # Logarithmic before anything is drawn, so every artist is laid out
    # in the scale it will be read in.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    # Regions: where a class of rule gives the smallest limit. Drawn from
    # the margin between the rules (a smooth function) rather than from
    # the class index, so the boundaries are the true curves.
    limits = rules.limits(np.full_like(xx, k_ratio), xx, yy)
    by_class = {
        name: np.minimum.reduce([limits[rule] for rule in rules.rules_of_class(name)])
        for name in gr.RULE_CLASSES if rules.rules_of_class(name)
    }
    for name, own in by_class.items():
        others = [values for other, values in by_class.items() if other != name]
        margin = np.minimum.reduce(others) - own if others else np.ones_like(own)
        margin = np.clip(np.nan_to_num(margin, nan=-1.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        if margin.max() <= 0:
            continue
        ax.contourf(xx, yy, margin, levels=[0.0, 2e6], colors=[RULE_TINTS[name]], zorder=0)
        ax.contour(xx, yy, margin, levels=[0.0], colors=AXIS, linewidths=0.6, zorder=1)
    present = [level for level in gr.DIAGRAM_LEVELS if grid.loading.min() < level < grid.loading.max()]
    if present:
        ax.contour(xx, yy, grid.loading, levels=present, colors=INK_2, linewidths=0.75, zorder=1)
        # Each line is named where it crosses one row, upright, on a patch
        # of the region's own tint: the line passes behind its number.
        for level, x, y, region in _contour_spots(grid, present):
            ax.text(x, y, f"{level:g}", ha="center", va="center", fontsize=8, color=INK_2, zorder=2,
                    gid=f"loading:{level:g}",
                    bbox=dict(facecolor=RULE_TINTS[region], edgecolor="none", pad=1.2))

    # markers: the largest first, so a small unit is never hidden under a big one
    pts["_x"] = pts["SO3_per_Na2O"].astype(float).clip(lower=x_range[0], upper=x_range[1])
    pts["_y"] = pts["Halide_term_per_Na2O"].astype(float).clip(lower=y_range[0], upper=y_range[1])
    at_edge = int(((pts["_x"] != pts["SO3_per_Na2O"]) | (pts["_y"] != pts["Halide_term_per_Na2O"])).sum())
    largest = float(pts["Na2O_t"].max())
    pts["_area"] = 9.0 + 200.0 * pts["Na2O_t"] / largest
    pts = pts.sort_values("Na2O_t", ascending=False)
    if group_col and group_col in pts.columns:
        groups = pts[group_col].where(pts[group_col].notna(), OTHER_GROUP).astype(str)
        palette = dict(SYSTEM_COLORS if group_col == "TankSystem" else PHASE_COLORS)
        labels = dict(SYSTEM_LABELS) if group_col == "TankSystem" else {}
    else:
        groups = pd.Series(unit_noun.capitalize(), index=pts.index)
        palette, labels = {unit_noun.capitalize(): _CATEGORICAL[0]}, {}
    colors = [palette.get(g, OTHER_COLOR) for g in groups]
    scatter = ax.scatter(pts["_x"], pts["_y"], s=pts["_area"], c=colors, edgecolors=SURFACE, linewidths=0.6,
                         alpha=0.88, zorder=3, clip_on=False)
    halide = chem(rules.halide_label)
    has_tank = "WasteSiteId" in pts.columns
    _mark_points(
        ax, scatter, list(pts["Unit"].astype(str)), name=unit_noun.capitalize(),
        tank_ids=list(pts["WasteSiteId"].astype(str)) if has_tank else None,
        x_label="SO₃ / Na₂O", y_label=f"({halide}) / Na₂O",
        extra=[
            ("Na₂O loading", [f"{fmt(v)} wt%" for v in pts["Na2O_loading_wtpct"]]),
            ("Binding rule", [str(v) for v in pts["Binding_rule"]]),
            ("Na₂O", [f"{fmt(v)} t" for v in pts["Na2O_t"]]),
            ("Extra glass", [f"{fmt(v)} t" for v in pts["Extra_glass_t"]]),
            ("Position", [_off_axes_note(row) for _, row in pts.iterrows()]),
        ],
    )
    spots = {
        str(unit): ("point", float(x), float(y), math.sqrt(float(area) / math.pi))
        for unit, x, y, area in zip(pts["Unit"], pts["_x"], pts["_y"], pts["_area"])
    }

    handles = []
    order = [g for g in palette if (groups == g).any()] + sorted(set(groups) - set(palette))
    for name in order:
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                              markerfacecolor=palette.get(name, OTHER_COLOR), markeredgecolor=SURFACE,
                              label=f"{labels.get(name, name)} ({int((groups == name).sum())})"))
    if blend and blend_x and blend_y and blend_x > 0 and blend_y > 0:
        loading = blend.get("Na2O_loading_wtpct")
        diamond = ax.scatter([blend_x], [blend_y], marker="D", s=58, facecolor=INK, edgecolor=SURFACE,
                             linewidth=1.0, zorder=5)
        _mark_points(
            ax, diamond, [f"All {unit_noun}s as one blended feed"], x_label="SO₃ / Na₂O",
            y_label=f"({halide}) / Na₂O",
            extra=[
                ("Na₂O loading", [f"{fmt(loading)} wt%"]),
                ("Binding rule", [str(blend.get("Binding_rule") or "")]),
                ("Glass", [f"{fmt(blend.get('Glass_t'))} t"]),
            ],
        )
        span_x = math.log(x_range[1] / x_range[0])
        span_y = math.log(y_range[1] / y_range[0])
        u = np.log(pts["_x"].to_numpy(dtype=float) / x_range[0]) / span_x
        v = np.log(pts["_y"].to_numpy(dtype=float) / y_range[0]) / span_y
        anchor = (math.log(blend_x / x_range[0]) / span_x, math.log(blend_y / y_range[0]) / span_y)
        ax.annotate(f"All {unit_noun}s as one feed: {fmt(loading)} wt% Na₂O", (blend_x, blend_y),
                    xytext=_quiet_spot(u, v, anchor), textcoords="axes fraction", ha="center", va="center",
                    fontsize=8.5, fontweight="semibold", color=INK, zorder=6,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor=GRID, linewidth=0.6,
                              boxstyle="round,pad=0.3"),
                    arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8, shrinkA=0, shrinkB=4))
        handles.append(Line2D([0], [0], marker="D", linestyle="none", markersize=6.5, markerfacecolor=INK,
                              markeredgecolor=SURFACE, label=BLEND_LABEL))

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    plain = FuncFormatter(lambda value, _pos: f"{value:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Sulfate:  SO₃ / Na₂O in the waste (mass ratio)")
    ax.set_ylabel(f"Halide term per Na₂O (mass ratio)\n({halide}) / Na₂O")
    _style_axes(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS)

    for name, x, y, ha, va in _region_spots(grid):
        # A patch of the region's own tint under the name: a contour or a
        # boundary that crosses the label passes behind it.
        ax.text(x, y, region_label(name), ha=ha, va=va, fontsize=8.5, fontweight="semibold",
                color=RULE_TEXT_COLORS[name], zorder=2, linespacing=1.1, multialignment="center",
                bbox=dict(facecolor=RULE_TINTS[name], edgecolor="none", pad=1.5))
    if present:
        handles.append(Line2D([0], [0], color=INK_2, linewidth=0.75, label="Na₂O loading allowed (wt% of glass)"))
    # Under the axes, on the figure: a legend inside the axes would sit
    # on top of units, and one anchored to the axes would narrow them.
    panel.figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
                        frameon=False, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.3)
    full_subtitle = ph.SUBTITLE_SEP.join(part for part in (
        subtitle,
        f"Marker area: Na₂O in the {unit_noun}",
        f"Region boundaries for K₂O / Na₂O = {fmt(k_ratio, 2)}",
        f"{at_edge} beyond the axes drawn at the edge" if at_edge else "",
    ) if part)
    _titles(panel, ax, title, full_subtitle)
    setattr(ax, RULE_SPOTS_ATTR, spots)
    _finish(panel, bottom=0.075)


@ph.composable
def plot_rule_shares(panel, groups: pd.DataFrame, rule_set_name: str, title: str, subtitle: str = "") -> None:
    """Stacked horizontal bars: for each group of units, the share of its
    sodium bound by each rule."""
    rules = gr.get_rule_set(rule_set_name)
    if groups is None or groups.empty:
        panel.show_message("Nothing to show for this selection.")
        return
    ax = panel.reset_axes()
    names = [str(name) for name in groups["Group"]]
    positions = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for rule in rules.rule_names:
        values = groups[f"{rule}_pct"].fillna(0.0).to_numpy(dtype=float)
        drawn = values > 0
        if drawn.any():
            ax.barh(positions[drawn], values[drawn], left=left[drawn], height=0.62,
                    color=rule_color(rules, rule), edgecolor=SURFACE, linewidth=1.2, label=f"{rule} rule")
        dark_text = rules.rule_classes[rule] in (gr.ALKALI, gr.HALIDE)
        for y, value, start in zip(positions, values, left):
            if value >= 6.0:
                ax.text(start + value / 2.0, y, f"{value:.0f}", ha="center", va="center", fontsize=8.5,
                        fontweight="semibold", color=INK if dark_text else "white")
        left = left + values
    for y, sodium in zip(positions, groups["Na2O_kt"]):
        ax.text(101.5, y, f"{fmt(sodium)} kt Na₂O", ha="left", va="center", fontsize=8.5, color=INK_2,
                clip_on=False)
    by_position = names[::-1]
    sodium_by_position = [f"{fmt(v)} kt Na₂O" for v in groups["Na2O_kt"]][::-1]
    units_by_position = [f"{int(v):,}" for v in groups["N_units"]][::-1]
    _declare_categories(
        ax, by_position, list(range(len(names))), name="Group", value_label="Share of the sodium",
        unit="%", series_name="Binding rule",
        extra=[("Sodium in the group", sodium_by_position), ("Units in the group", units_by_position)],
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of the sodium, by the rule that sets its loading (%)")
    _style_axes(ax)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(rules.rule_names), fontsize=8.5,
              frameon=False, handlelength=1.1, columnspacing=1.4)
    _titles(panel, ax, title, subtitle)
    setattr(ax, RULE_SPOTS_ATTR, {})
    _finish(panel)


@ph.composable
def plot_extra_glass(panel, table: pd.DataFrame, rule_set_name: str, title: str, subtitle: str = "",
                     top_n: int = TOP_EXTRA_GLASS, unit_noun: str = "tank") -> None:
    """The units whose rule-limited loading costs the most glass over the
    alkali limit, coloured by the rule that binds them."""
    from matplotlib.patches import Patch

    rules = gr.get_rule_set(rule_set_name)
    data = table.dropna(subset=["Binding_rule"]) if table is not None and len(table) else None
    if data is None or data.empty:
        panel.show_message(f"No {unit_noun} in this selection holds sodium.")
        return
    data = data[data["Extra_glass_t"] > 0].sort_values("Extra_glass_t", ascending=False).head(int(top_n))
    if data.empty:
        panel.show_message(f"Every {unit_noun} in this selection loads to the alkali limit: no extra glass.")
        return
    ax = panel.reset_axes()
    n = len(data)
    positions = np.arange(n)[::-1]
    values = data["Extra_glass_t"].to_numpy(dtype=float)
    colors = [rule_color(rules, rule) for rule in data["Binding_rule"]]
    ax.barh(positions, values, height=0.68, color=colors, edgecolor=SURFACE, linewidth=0.8)
    spots = {}
    for y, unit, value, loading in zip(positions, data["Unit"], values, data["Na2O_loading_wtpct"]):
        ax.text(value, y, f" {fmt(loading)} wt%", ha="left", va="center", fontsize=8.5, color=INK_2,
                clip_on=False, zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        spots[str(unit)] = ("bar", 0.0, y - 0.34, float(value), y + 0.34)
    has_tank = "WasteSiteId" in data.columns
    _declare_categories(
        ax, [str(u) for u in data["Unit"]][::-1], list(range(n)), name=unit_noun.capitalize(),
        value_label="Extra glass over the alkali limit", unit="t",
        tank_ids=[str(t) for t in data["WasteSiteId"]][::-1] if has_tank else None,
        extra=[
            ("Na₂O loading", [f"{fmt(v)} wt%" for v in data["Na2O_loading_wtpct"]][::-1]),
            ("Binding rule", [str(v) for v in data["Binding_rule"]][::-1]),
            ("Glass", [f"{fmt(v)} t" for v in data["Glass_t"]][::-1]),
        ],
    )
    ax.set_xlim(0, float(values.max()) * 1.16)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Extra glass over the alkali limit (t); bars are labelled with the Na₂O loading")
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:,.0f}")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.spines["left"].set_visible(False)
    seen = [rule for rule in rules.rule_names if rule in set(data["Binding_rule"])]
    ax.legend(handles=[Patch(facecolor=rule_color(rules, rule), edgecolor=SURFACE, label=f"{rule} rule")
                       for rule in seen],
              loc="lower right", fontsize=8.5, frameon=False, handlelength=1.1)
    _titles(panel, ax, title, subtitle)
    setattr(ax, RULE_SPOTS_ATTR, spots)
    _finish(panel)


# ---------------------------------------------------------------------------
# Tables as shown
# ---------------------------------------------------------------------------
def units_display(result: gr.ScreenResult) -> pd.DataFrame:
    """The Units table with readable headers, the columns a reader wants
    first, sorted by extra glass. The first column is the unit's label,
    the key that links a row to its marker."""
    table = result.table
    unit_header = "Tank" if result.level == "tank" else "Waste layer"
    halide = chem(result.rule_set.halide_label)
    columns: List[Tuple[str, str]] = [
        (gr.UNIT_COLUMN, unit_header),
        ("Na2O_loading_wtpct", "Loading (wt% Na₂O)"),
        ("Binding_rule", "Binding rule"),
        ("Extra_glass_t", "Extra glass (t)"),
        ("Glass_t", "Glass (t)"),
        ("Glass_t_alkali_only", "Glass at the alkali limit (t)"),
        ("Na2O_t", "Na₂O (t)"),
        ("SO3_per_Na2O", "SO₃ / Na₂O"),
        ("Halide_term_per_Na2O", f"({halide}) / Na₂O"),
        ("K2O_per_Na2O", "K₂O / Na₂O"),
        ("Cl_per_Na2O", "Cl / Na₂O"),
        ("F_per_Na2O", "F / Na₂O"),
        ("Cr2O3_per_Na2O", "Cr₂O₃ / Na₂O"),
    ]
    columns += [(f"Limit_{rule}", f"{rule} limit (wt% Na₂O)") for rule in result.rule_set.rule_names]
    columns += [
        ("SO3_in_glass_wtpct", "SO₃ in glass (wt%)"), ("Cl_in_glass_wtpct", "Cl in glass (wt%)"),
        ("F_in_glass_wtpct", "F in glass (wt%)"), ("Cr2O3_in_glass_wtpct", "Cr₂O₃ in glass (wt%)"),
    ]
    if result.level != "tank":
        columns.append(("WasteSiteId", "Tank"))
    columns += [
        ("TankFarm", "Farm"), ("TankSystem", "System"), ("PhaseFamily", "Phase family"),
        ("N_layers", "Layers"), ("Volume_kL", "Volume (kL)"),
    ]
    present = [(raw, shown) for raw, shown in columns if raw in table.columns]
    shown = table[[raw for raw, _ in present]].copy()
    shown.columns = [label for _, label in present]
    if "Extra glass (t)" in shown.columns and len(shown):
        shown = shown.sort_values("Extra glass (t)", ascending=False, na_position="last", kind="mergesort")
    return shown.reset_index(drop=True)


def summary_display(result: gr.ScreenResult) -> pd.DataFrame:
    """The Summary table as shown: each quantity with its value and unit
    written together. The rows are different kinds of quantity (counts,
    kilotonnes, percentages, a ratio), so each is written to its own
    precision rather than to one precision for the column; the export
    keeps the numbers."""
    table = result.summary.as_table(result.unit_noun)
    values = []
    for value, unit in zip(table["Value"], table["Unit"]):
        if pd.isna(value):
            values.append("")
        elif not unit:
            values.append(f"{int(round(float(value))):,}")
        else:
            values.append(f"{fmt(value, 4)} {chem(unit)}")
    return pd.DataFrame({"Quantity": table["Quantity"].map(chem), "Value": values})


def groups_display(result: gr.ScreenResult) -> pd.DataFrame:
    groups = result.groups
    if groups.empty:
        return pd.DataFrame()
    rename = {"Group": "Group", "N_units": "Count", "Na2O_kt": "Na₂O (kt)",
              "Not_alkali_limited_pct": "Not alkali-limited (%)",
              "Glass_kt_alkali_only": "Glass at the alkali limit (kt)",
              "Glass_kt_unit_by_unit": "Glass, unit by unit (kt)", "Extra_glass_pct": "Extra glass (%)"}
    rename.update({f"{rule}_pct": f"{rule} rule (% of Na₂O)" for rule in result.rule_set.rule_names})
    return groups.rename(columns=rename)


def banner_notes(result: gr.ScreenResult) -> List[str]:
    """What the line above the result expands to: the scope of validity,
    what is particular to this result, and a count of the dataset notes
    the query engine attaches to every result. Those, the method and the
    rules are listed in full in the Notes tab: nothing is dropped, only
    kept one click away from a banner that must stay short."""
    notes = [gr.SCOPE_OF_VALIDITY] + [chem(remark) for remark in result.remarks]
    if result.warnings:
        count = len(result.warnings)
        notes.append(f"{count} dataset note{'' if count == 1 else 's'} from the query engine "
                     f"appl{'ies' if count == 1 else 'y'} to every result: see the Notes tab.")
    return notes


def notes_table(result: gr.ScreenResult, scope_text: str) -> pd.DataFrame:
    scope_row = (f"Scope: {scope_text}" if scope_text
                 else "Scope: none. Every tank and layer of the dataset.")
    rows = [{"Kind": "Scope", "Text": scope_row}]
    rows += [{"Kind": "Method", "Text": chem(note)} for note in result.notes]
    rows += [{"Kind": "Rule", "Text": f"{rule.name}: {chem(rule.equation)} (wt% of glass). {rule.source}."}
             for rule in result.rule_set.rules]
    rows.append({"Kind": "Source", "Text": result.rule_set.citation})
    rows += [{"Kind": "Data note", "Text": warning} for warning in result.warnings]
    rows += [{"Kind": "Disclosure", "Text": disclosure} for disclosure in result.disclosures]
    return pd.DataFrame(rows)


def methods_note(result: gr.ScreenResult, scope_text: str) -> str:
    """The paragraph that travels with an export."""
    rules = result.rule_set
    lines = [
        "Glass loading-rule screen",
        "",
        f"Rule set: {rules.name}. {rules.description}",
        "",
        "Rules (wt% of glass):",
    ]
    lines += [f"  {rule.name}: {rule.equation}   [{rule.source}]" for rule in rules.rules]
    lines += [
        "",
        f"Feed: the {result.waste_label} of each {result.unit_noun}, as stored. "
        f"Scope: {scope_text or 'none (the whole dataset)'}.",
        "Na2O, K2O, Cr2O3 and SO3 are computed from the reported Na, K, Cr and SO4 with gravimetric "
        "factors from the atomic masses; Cl and F are used as elements.",
        "With the waste mass ratios r_X = X / Na2O, each rule is solved for the largest Na2O loading w "
        "it allows; the loading of a unit is the smallest of these and the binding rule is the one "
        "that gives it. Glass mass = Na2O mass / (w / 100). Extra glass = glass at the rule-limited "
        "loading minus glass at the alkali limit alone.",
        "",
        gr.SCOPE_OF_VALIDITY,
        "",
        f"Source: {rules.citation}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------
@dataclass
class _Outcome:
    """What the worker hands back, tagged with the run it belongs to."""

    token: int
    params: dict = field(default_factory=dict)
    result: Optional[gr.ScreenResult] = None
    refusal: str = ""
    error: str = ""


def compute_screen(token: int, dataset: HanfordDataset, params: dict,
                   scope: Dict[str, List[str]]) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    try:
        result = gr.screen(dataset, level=params["level"], phases=params["phases"],
                           rule_set=params["rule_set"], scope=scope)
    except ValueError as exc:
        return _Outcome(token=token, params=params, refusal=str(exc))
    except Exception:
        return _Outcome(token=token, params=params, error=traceback.format_exc())
    return _Outcome(token=token, params=params, result=result)


class LoadingRulesTab(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.scope = scope if scope is not None else GlobalScope(self)
        self.dataset: Optional[HanfordDataset] = None
        self._result: Optional[gr.ScreenResult] = None
        self._params: dict = {}
        self._tables: Dict[str, pd.DataFrame] = {}
        self._tank_of_unit: Dict[str, str] = {}
        self._run_token = 0
        self._running = False
        self._stale = True                          # a run is owed the next time the tab is shown
        self._linking = False                       # a table <-> plot selection is being mirrored
        # Owned by the tab, so a refresh queued by a show can never fire
        # on a tab that has since been destroyed.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self._refresh_if_due)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 14, 20, 12)
        root.setSpacing(10)

        self.header = PageHeader(
            TAB_TITLE,
            "Which published LAW glass loading rule would set the sodium loading of each tank's salt "
            "waste, how far below the alkali limit that is, and what it costs in glass.",
        )
        self.export_btn = QPushButton("Export...")
        self.export_btn.setObjectName("Secondary")
        self.export_btn.setToolTip("Write the tables, the figure and a methods note to the output folder.")
        self.export_btn.clicked.connect(self.export_tables)
        self.header.add_action_widget(self.export_btn)
        self.help_btn = make_help_button(self, HELP_SECTION, "Vitrification")
        self.help_btn.clicked.disconnect()
        self.help_btn.clicked.connect(self.show_help)
        self.help_btn.setToolTip("How the loading rules are applied, and how to read the diagram.")
        self.header.add_action_widget(self.help_btn)
        root.addWidget(self.header)

        controls = QWidget()
        controls.setObjectName("Toolbar")
        row = QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        def field_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setObjectName("FieldLabel")
            return label

        row.addWidget(field_label("Rule set"))
        self.rule_set_combo = QComboBox()
        for index, (name, rule_set) in enumerate(gr.RULE_SETS.items()):
            self.rule_set_combo.addItem(name)
            self.rule_set_combo.setItemData(
                index, rule_set.description + "\n" + "\n".join(
                    f"{rule.name}: {chem(rule.equation)}" for rule in rule_set.rules),
                role=Qt.ToolTipRole)
        self.rule_set_combo.setToolTip("The published rules to screen against (PNNL-22631 Rev. 1).")
        self.rule_set_combo.currentTextChanged.connect(self._on_inputs_changed)
        row.addWidget(self.rule_set_combo)

        row.addSpacing(8)
        row.addWidget(field_label("Unit"))
        self.unit_combo = QComboBox()
        for label, level in UNIT_CHOICES.items():
            self.unit_combo.addItem(label, level)
        self.unit_combo.setToolTip(
            "Tanks: the selected waste of each tank, summed.\n"
            "Waste layers: one (tank, waste phase, waste type) entry at a time.")
        self.unit_combo.currentIndexChanged.connect(self._on_inputs_changed)
        row.addWidget(self.unit_combo)

        row.addSpacing(8)
        row.addWidget(field_label("Waste"))
        self.waste_combo = QComboBox()
        for label in WASTE_CHOICES:
            self.waste_combo.addItem(label)
        self.waste_combo.setCurrentText(DEFAULT_WASTE)
        self.waste_combo.setToolTip(
            "Which waste counts as the feed. Supernatant and saltcake are the salt waste that\n"
            "becomes LAW feed; sludge is included only under All phases.")
        self.waste_combo.currentTextChanged.connect(self._on_inputs_changed)
        row.addWidget(self.waste_combo)

        row.addSpacing(8)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        row.addWidget(self.run_btn)
        row.addStretch(1)

        row.addWidget(field_label("Plot"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(PLOT_KINDS)
        self.plot_combo.setToolTip(
            "Rule diagram: every unit on the plane of the rules.\n"
            "Sodium by binding rule: which rule sets the loading of how much sodium, by group.\n"
            "Largest extra glass: the units that cost the most glass over the alkali limit.")
        self.plot_combo.currentTextChanged.connect(self._on_plot_kind_changed)
        row.addWidget(self.plot_combo)
        root.addWidget(controls)

        # Notes that qualify the result: one calm line, the detail one click away.
        self.notes_banner = NotesBanner()
        self.warnings_label = self.notes_banner         # the name other pages use for this slot
        root.addWidget(self.notes_banner)

        tiles = QHBoxLayout()
        tiles.setContentsMargins(0, 0, 0, 0)
        tiles.setSpacing(12)
        self.tile_sodium = StatTile("Sodium screened")
        self.tile_not_alkali = StatTile("Not alkali-limited")
        self.tile_glass = StatTile("Glass, tank by tank")
        self.tile_blend = StatTile("Glass, one blended feed")
        self.tile_low = StatTile(f"Tanks below {gr.LOW_LOADING_WTPCT:g} wt% Na₂O")
        self.tiles = [self.tile_sodium, self.tile_not_alkali, self.tile_glass, self.tile_blend, self.tile_low]
        for tile in self.tiles:
            # A tile must never set the width of the window: on a narrow
            # window its hint is cut short (the tooltip carries it in full).
            tile.hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            tile.label_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            tile.setMinimumWidth(120)
            tiles.addWidget(tile, 1)
        root.addLayout(tiles)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        self.plot_stack = PlotPane()
        self.empty_state = EmptyState(_IDLE_TITLE, _IDLE_HINT)
        self.plot = PlotWidget(figsize=(7.0, 4.4))
        self.plot_stack.addWidget(self.empty_state)
        self.plot_stack.addWidget(self.plot)
        splitter.addWidget(self.plot_stack)

        # The shared interaction layer does the hovering and the clicking;
        # the tab links a clicked mark to its row and opens tanks through
        # the shell it was given.
        self.ring = SelectionRing(self.plot)
        clicked = getattr(self.plot, "markClicked", None)
        if clicked is not None:
            clicked.connect(self._on_mark_clicked)
        interaction = getattr(self.plot, "interaction", None)
        if interaction is not None and hasattr(interaction, "set_open_tank_handler"):
            interaction.set_open_tank_handler(self.open_tank)

        self.tables = QTabWidget()
        self._table_views: Dict[str, DataFrameTableView] = {}
        for title in TABLE_TITLES:
            view = DataFrameTableView(title=title, max_rows_display=None)
            self._table_views[title] = view
            self.tables.addTab(view, title)
        units_view = self._table_views[T_UNITS]
        units_view.selectionModel().currentRowChanged.connect(self._on_unit_row_changed)
        units_view.doubleClicked.connect(self._on_unit_double_clicked)
        units_view.setToolTip("Click a row to ring its marker on the plot; double-click to open the tank.")
        if hasattr(units_view, "set_empty_text"):
            units_view.set_empty_text("No units yet", "One row per tank or waste layer appears after a run.")
        splitter.addWidget(self.tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([640, 440])
        self.splitter = splitter

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    def help_markdown(self) -> str:
        """The part of the Vitrification section that describes this tab
        (the whole section when the subsection cannot be found)."""
        import re
        import qt_help
        section = qt_help.load_notice_section(HELP_SECTION)
        start = section.find(HELP_SUBSECTION)
        if start < 0:
            return section
        rest = section[start:]
        following = re.search(r"^#{2,3} ", rest[len(HELP_SUBSECTION):], re.MULTILINE)
        end = len(HELP_SUBSECTION) + following.start() if following else len(rest)
        return rest[:end].strip().rstrip("-").strip()

    def show_help(self) -> None:
        import qt_help
        qt_help.HelpDialog(self, markdown=self.help_markdown(), title=f"{TAB_TITLE} — Help",
                           width=720, height=620).exec()

    # ------------------------------------------------------------------
    # Dataset / scope protocol
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A result of the previous dataset must not stay on screen, and a
        # run still in flight on it must not land.
        self._run_token += 1
        self._set_running(False)
        self._clear_result()
        self._stale = True
        self._refresh_if_due()

    def on_scope_changed(self) -> None:
        """The global scope changed. A visible tab re-asks its standing
        question under the new scope; a hidden one only remembers that it
        is out of date and refreshes when it is shown."""
        self._stale = True
        self._refresh_if_due()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        # After the show has been processed: the first run of a session
        # must not hold up the paint of the tab it belongs to.
        self._refresh_timer.start()

    def is_stale(self) -> bool:
        return self._stale

    def _refresh_if_due(self) -> None:
        if not self._stale or not self.isVisible():
            return
        if self.dataset is None or not self.dataset.is_loaded():
            return
        self._start(dict(self._params) if self._result is not None else self._read_params())

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def _read_params(self) -> dict:
        waste = self.waste_combo.currentText()
        return {
            "rule_set": self.rule_set_combo.currentText(),
            "level": self.unit_combo.currentData(),
            "waste": waste,
            "phases": WASTE_CHOICES[waste],
        }

    def _on_inputs_changed(self, *_args) -> None:
        self._sync_run_label()

    def _sync_run_label(self) -> None:
        if self._running:
            return
        changed = self._result is not None and any(
            self._read_params()[key] != self._params.get(key) for key in ("rule_set", "level", "waste"))
        self.run_btn.setText("Update" if changed else "Run")

    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._status(f"{TAB_TITLE}: no dataset is loaded yet.")
            return
        self._start(self._read_params())

    def _start(self, params: dict) -> None:
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        scope = self.scope.as_scope()
        params = dict(params)
        params["scope"] = self.scope.summary()
        self._stale = False
        self._set_running(True)
        run_in_thread(
            lambda: compute_screen(token, dataset, params, scope),
            self._on_run_done, self._on_run_error,
        )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.run_btn.setEnabled(not running)
        if running:
            self.run_btn.setText("Running…")
        else:
            self._sync_run_label()

    def _status(self, message: str) -> None:
        status_bar = getattr(self.app_window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message)

    def _on_run_error(self, tb_text: str) -> None:
        self._set_running(False)
        self._status("Loading-rule screen failed.")
        QMessageBox.critical(self, f"{TAB_TITLE} failed", tb_text)

    def _on_run_done(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return          # superseded by a newer question, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._status("Loading-rule screen failed.")
            QMessageBox.critical(self, f"{TAB_TITLE} failed", outcome.error)
            return
        if outcome.refusal:
            self._status(f"Loading-rule screen not run: {outcome.refusal}")
            if self._result is None:
                self.empty_state.set_text("The screen could not be run", outcome.refusal)
            else:
                QMessageBox.warning(self, TAB_TITLE, outcome.refusal)
            return
        self._result = outcome.result
        self._params = dict(outcome.params)
        self._show_result()
        self._sync_run_label()
        summary = self._result.summary
        scope_note = f", scope: {self._params['scope']}" if self._params.get("scope") else ""
        self._status(
            f"Loading rules: {summary.n_with_sodium:,} {self._result.unit_noun}s screened against "
            f"{self._result.rule_set.name}{scope_note}."
        )

    # ------------------------------------------------------------------
    # Showing a result
    # ------------------------------------------------------------------
    def _set_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        self._tables = tables
        self._linking = True
        try:
            for title, view in self._table_views.items():
                view.set_dataframe(tables.get(title, pd.DataFrame()))
        finally:
            self._linking = False
        self._fit_first_columns()

    def _fit_first_columns(self) -> None:
        """The first column names the row (a tank, a layer, a quantity, a
        group): it is never cut short while the table has room to scroll."""
        for title, view in self._table_views.items():
            frame = view.dataframe()
            if title == T_NOTES or frame.empty:
                continue
            header = view.horizontalHeader()
            metrics = getattr(view, "table", view).fontMetrics()
            longest = max((metrics.horizontalAdvance(str(v)) for v in frame.iloc[:, 0]), default=0)
            wanted = min(longest + 34, 300)
            if header.sectionSize(0) < wanted:
                header.resizeSection(0, wanted)

    def _clear_result(self) -> None:
        self._result = None
        self._params = {}
        self._tank_of_unit = {}
        self._set_tables({})
        for tile in self.tiles:
            tile.set_value("—", "")
            tile.setToolTip("")
        self.notes_banner.set_notes([])
        self.ring.select(None)
        self.empty_state.set_text(_IDLE_TITLE, _IDLE_HINT)
        self.plot_stack.setCurrentWidget(self.empty_state)
        self._sync_run_label()

    def scope_text(self) -> str:
        return str(self._params.get("scope") or "")

    def _show_result(self) -> None:
        result = self._result
        self._tank_of_unit = (dict(zip(result.table[gr.UNIT_COLUMN].astype(str),
                                       result.table["WasteSiteId"].astype(str)))
                              if "WasteSiteId" in result.table.columns else {})
        selected = self.ring.key()
        self._set_tables({
            T_UNITS: units_display(result),
            T_SUMMARY: self._summary_display(result),
            T_GROUPS: groups_display(result),
            T_NOTES: notes_table(result, self.scope_text()),
        })
        self._fill_tiles(result)
        self.notes_banner.set_notes(banner_notes(result), headline=_HEADLINE)
        # The unit that was selected stays selected if the new result still has it.
        if selected and not self._select_unit_row(selected):
            self.ring.select(None)
        self._update_plot()

    def _summary_display(self, result: gr.ScreenResult) -> pd.DataFrame:
        return summary_display(result)

    def _fill_tiles(self, result: gr.ScreenResult) -> None:
        s = result.summary
        noun = result.unit_noun
        plural = f"{noun}s"
        short = TILE_NOUNS.get(result.level, noun)          # a tile label has one short line
        self.tile_glass.label_label.setText(f"Glass, {short} by {short}")
        self.tile_low.label_label.setText(f"{short.capitalize()}s below {s.low_loading_wtpct:g} wt% Na₂O")
        if not s.n_with_sodium:
            for tile in self.tiles:
                tile.set_value("—", "")
                tile.setToolTip("")
            self.tile_sodium.set_value("—", f"no {noun} with sodium in this selection")
            return
        values = [
            (self.tile_sodium, f"{fmt(s.na2o_kt)} kt", f"Na₂O in {s.n_with_sodium:,} {plural}",
             f"Sodium, as Na₂O, in the {result.waste_label} of the {plural} that hold any."),
            (self.tile_not_alkali, f"{s.not_alkali_limited_pct:.1f} %", "of the sodium screened",
             "Share of the sodium whose loading is set by a sulfur or halide rule, not by the alkali limit."),
            (self.tile_glass, f"{fmt(s.glass_unit_by_unit_kt, 4)} kt",
             f"+{s.extra_glass_pct:.1f} % over alkali only",
             f"Glass at the alkali limit alone: {fmt(s.glass_alkali_only_kt, 4)} kt. "
             f"The other rules add {fmt(s.extra_glass_kt)} kt (+{s.extra_glass_pct:.1f} %)."),
            (self.tile_blend, f"{fmt(s.glass_blended_kt, 4)} kt",
             f"{s.blend_loading_wtpct:.1f} wt% Na₂O, {str(s.blend_binding_rule).lower()}",
             f"All {plural} as one feed load to {s.blend_loading_wtpct:.2f} wt% Na₂O; "
             f"the {str(s.blend_binding_rule).lower()} rule binds."),
            (self.tile_low, f"{s.n_below_low_loading:,}", f"of {s.n_with_sodium:,} holding sodium",
             f"{plural.capitalize()} whose rule-limited loading is below {s.low_loading_wtpct:g} wt% Na₂O."),
        ]
        for tile, value, hint, detail in values:
            tile.set_value(value, hint)
            tile.setToolTip(f"{tile.label_label.text()}: {value} ({hint})\n{detail}")

    # ------------------------------------------------------------------
    # Units table <-> plot
    # ------------------------------------------------------------------
    def _unit_header(self) -> Optional[str]:
        shown = self._table_views[T_UNITS].dataframe()
        return str(shown.columns[0]) if len(shown.columns) else None

    def selected_unit(self) -> Optional[str]:
        """The unit on the current row of the Units table, as displayed."""
        view = self._table_views[T_UNITS]
        shown = view.dataframe()
        index = view.currentIndex()
        header = self._unit_header()
        if header is None or not index.isValid() or not (0 <= index.row() < len(shown)):
            return None
        return str(shown.iloc[index.row()][header])

    def _select_unit_row(self, unit: str) -> bool:
        """Make `unit` the current row of the Units table (clearing a
        filter that hides it). False when the table does not hold it."""
        view = self._table_views[T_UNITS]
        shown = view.dataframe()
        header = self._unit_header()
        if header is None:
            return False
        matches = np.nonzero((shown[header].astype(str) == str(unit)).to_numpy())[0]
        if not len(matches):
            return False
        row = int(matches[0])
        self._linking = True
        try:
            mask = view.model().filter_mask() if hasattr(view.model(), "filter_mask") else None
            if mask is not None and not bool(mask[row]):
                view.set_filter_text("")        # the unit that was clicked wins over the filter
            view.setCurrentIndex(view.model().index(row, 0))
            view.selectRow(row)
        finally:
            self._linking = False
        self.ring.select(str(unit))
        return True

    def _on_unit_row_changed(self, *_args) -> None:
        if self._linking:
            return
        self.ring.select(self.selected_unit())

    def _on_unit_double_clicked(self, index) -> None:
        shown = self._table_views[T_UNITS].dataframe()
        header = self._unit_header()
        if header is None or not index.isValid() or not (0 <= index.row() < len(shown)):
            return
        self.open_tank(self._tank_of_unit.get(str(shown.iloc[index.row()][header])))

    def _on_mark_clicked(self, mark: dict) -> None:
        """A mark of the chart was clicked: when it stands for a unit,
        select that unit's row (which rings the mark)."""
        unit = str((mark or {}).get("label") or "")
        if unit not in self._tank_of_unit and self.ring.spot(unit) is None:
            return
        if self._select_unit_row(unit):
            self.tables.setCurrentWidget(self._table_views[T_UNITS])

    def open_tank(self, tank_id: Optional[str]) -> bool:
        """Open a tank in the shell's tank view, when the shell offers it."""
        opener = getattr(self.app_window, "open_tank", None)
        if not tank_id or not callable(opener):
            return False
        opener(str(tank_id))
        return True

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _on_plot_kind_changed(self, _text: str) -> None:
        self._update_plot()

    def _subtitle(self) -> str:
        result = self._result
        parts = [
            result.rule_set.name,
            f"{result.waste_label.capitalize()} of each {result.unit_noun}",
            f"{result.summary.n_with_sodium:,} {result.unit_noun}s holding sodium",
        ]
        if self.scope_text():
            parts.append(f"Scope: {self.scope_text()}")
        return ph.SUBTITLE_SEP.join(parts)

    def _update_plot(self) -> None:
        result = self._result
        if result is None:
            return
        if not result.summary.n_with_sodium:
            where = f"under the scope {self.scope_text()}" if self.scope_text() else "in this dataset"
            self.empty_state.set_text(
                "Nothing to screen in this selection",
                f"No {result.unit_noun} holds sodium in the {result.waste_label} {where}. "
                "Change the waste selection or the scope.")
            self.plot_stack.setCurrentWidget(self.empty_state)
            return
        self.plot_stack.setCurrentWidget(self.plot)
        kind = self.plot_combo.currentText()
        noun = result.unit_noun
        subtitle = self._subtitle()
        if kind == PLOT_SHARES:
            plot_rule_shares(self.plot, result.groups, result.rule_set.name,
                             "Sodium by the rule that sets its loading", subtitle)
        elif kind == PLOT_EXTRA:
            plot_extra_glass(self.plot, result.table, result.rule_set.name,
                             f"Largest extra glass over the alkali limit, {noun} by {noun}", subtitle,
                             TOP_EXTRA_GLASS, noun)
        else:
            group_col = "TankSystem" if result.level == "tank" else "PhaseFamily"
            if group_col not in result.table.columns:
                group_col = None
            plot_rule_diagram(self.plot, result.screened(), result.blend.to_dict(), result.rule_set.name,
                              result.summary.k2o_per_na2o, group_col,
                              f"Loading rules: where each {noun} sits", subtitle, noun)
        self.ring.attach()
        readout = getattr(self.plot, "coords_label", None)
        if readout is not None:
            readout.setText("")          # it described a mark of the previous drawing

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_tables(self) -> None:
        if self._result is None or self.dataset is None:
            self._status(f"{TAB_TITLE}: run the screen first; there is nothing to export yet.")
            return
        result, params = self._result, self._params
        rules_table = pd.DataFrame([
            {"rule": rule.name, "class": rule.rule_class, "equation_wt_pct_of_glass": rule.equation,
             "source": rule.source} for rule in result.rule_set.rules
        ])
        parameters = pd.DataFrame([
            {"parameter": "rule_set", "value": result.rule_set.name},
            {"parameter": "unit", "value": result.level},
            {"parameter": "waste", "value": result.waste_label},
            {"parameter": "units_screened", "value": result.summary.n_units},
            {"parameter": "units_holding_sodium", "value": result.summary.n_with_sodium},
            {"parameter": "plot", "value": self.plot_combo.currentText()},
            {"parameter": "citation", "value": result.rule_set.citation},
        ])
        tables = {
            "loading_rules_units": result.table,
            "loading_rules_layers": result.layer_table if result.level == "tank" else None,
            "loading_rules_summary": result.summary.as_table(result.unit_noun),
            "loading_rules_groups": result.groups,
            "loading_rules_rule_set": rules_table,
            "loading_rules_notes": notes_table(result, self.scope_text()),
            "loading_rules_parameters": parameters,
        }
        out_dir = export_utils.export_named_tables(
            self.dataset, "loading_rules", tables,
            context={
                "rule_set": result.rule_set.name, "unit": result.level, "waste": result.waste_label,
                "global_scope": params.get("scope") or "(none)",
                "source": "PNNL-22631 Rev. 1 (ORP-58289), 2013",
            },
        )
        try:
            (out_dir / "loading_rules_methods.txt").write_text(
                methods_note(result, self.scope_text()), encoding="utf-8")
        except OSError:
            pass
        if self.plot_stack.currentWidget() is self.plot:
            try:
                self.plot.figure.savefig(out_dir / "loading_rules_plot.png", bbox_inches="tight", dpi=200)
            except Exception:
                pass
        self._status(f"Loading-rule screen exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Loading-rule outputs exported:\n{out_dir}")
