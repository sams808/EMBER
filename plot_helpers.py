"""
plot_helpers.py — shared matplotlib plotting utilities used by multiple
workspaces (Element Explorer, Tank Explorer, ...), ported from the old
app's module-level plot helper functions. Framework-agnostic aside from
taking a qt_widgets.PlotWidget (or figure_composer.ComposerCellPanel, which
mimics the same small interface) to draw into.

Besides the plot functions, this module owns what makes every chart of the
app one product:

- the chart style (colours, type, colormaps): apply_chart_style();
- what a chart tells the pointer: mark_categories / mark_points /
  mark_cells, read by qt_plot_interaction for tooltips and clicks;
- legible category axes: declare_categories() fits labels to the room
  there is (smaller type, then every k-th label, with a sentence saying
  so), and relayout_figure() repeats the fit whenever the size changes;
- containment: no chart is cut by the edge of its pane. overflow_report()
  measures it, and every layout (relayout_figure(), a draw, an export, a
  cell of a composed figure) ends with contain_region(), which corrects
  what the measurement finds;
- finish_panel(), run by @composable after every plot function, so a plot
  replayed in the Figure Composer is finished exactly like the original.
"""
from __future__ import annotations

import functools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

from elements import normalize_element_symbol
from matrix_science import log10_safe, square_matrix_lookup
from qt_table_format import humanize, humanize_text, tank_id_of

# ---------------------------------------------------------------------------
# One chart style. Every figure in the app is drawn with these colours and
# this type, whichever function draws it; chart surfaces stay white in both
# UI themes so the screen matches the export.
# ---------------------------------------------------------------------------

# Categorical series, in this fixed order (never picked by rank in a plot).
DATA_CATEGORICAL: Tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
# Magnitude: one hue, light to dark.
DATA_BLUE_RAMP: Tuple[str, ...] = (
    "#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)
DATA_GREEN_RAMP: Tuple[str, ...] = (
    "#eaf8f2", "#c5ecdc", "#93dcc0", "#5cc9a0", "#1baf7a", "#128a5f", "#0b6646", "#06442f",
)
DATA_RED_RAMP: Tuple[str, ...] = (
    "#fdeeee", "#f9d2d1", "#f3a9a4", "#ec7c79", "#e34948", "#c22f31", "#a11d21", "#6f1014",
)
DATA_GREY_RAMP: Tuple[str, ...] = (
    "#f6f6f4", "#e6e5e0", "#cfcec7", "#b0afa7", "#8a8983", "#6a6964", "#52514e", "#2e2d2b",
)
# Signed quantities: blue <-> red through a NEUTRAL grey. A coefficient of
# zero means "no association" and must read as nothing; a hue at the
# midpoint would put colour on the cells that carry no signal.
DATA_DIVERGING: Tuple[str, ...] = (
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a4", "#e34948", "#a11d21",
)
CHART_INK = "#0b0b0b"
CHART_SECONDARY = "#52514e"
CHART_FAINT = "#8a8983"
CHART_GRID = "#e1e0d9"
CHART_AXIS = "#c3c2b7"
CHART_NEUTRAL = "#a3a29b"          # marginal bars: context, not the subject
POSITIVE_COLOR = DATA_CATEGORICAL[0]
NEGATIVE_COLOR = DATA_CATEGORICAL[7]

# A single series is blue. The highlighted bar of a ranking (the element
# or tank the reader asked about) takes the second categorical colour.
BASE_COLOR = DATA_CATEGORICAL[0]
ACCENT_COLOR = DATA_CATEGORICAL[1]

CMAP_SEQUENTIAL = "ember_blues"
CMAP_CELLS = "ember_blues_cells"    # the same ramp from its second step: for matrices with empty cells
CMAP_DIVERGING = "ember_diverging"
CMAP_GREENS = "ember_greens"
CMAP_REDS = "ember_reds"
CMAP_GREYS = "ember_greys"

_CHART_FONTS = ("Segoe UI", "Segoe UI Variable", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans")


def _data_colors() -> None:
    """Adopt qt_theme.DATA when the theme defines it, so the charts and the
    rest of the interface (maps, legends) can never disagree on a colour."""
    global DATA_CATEGORICAL, DATA_BLUE_RAMP, BASE_COLOR, ACCENT_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR
    try:
        import qt_theme
        data = getattr(qt_theme, "DATA", None)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return
    categorical = data.get("categorical")
    if isinstance(categorical, (list, tuple)) and len(categorical) >= 8:
        DATA_CATEGORICAL = tuple(str(c) for c in categorical)
        BASE_COLOR, ACCENT_COLOR = DATA_CATEGORICAL[0], DATA_CATEGORICAL[1]
        POSITIVE_COLOR, NEGATIVE_COLOR = DATA_CATEGORICAL[0], DATA_CATEGORICAL[7]
    for key in ("blue_ramp", "ramp", "sequential"):
        ramp = data.get(key)
        if isinstance(ramp, (list, tuple)) and len(ramp) >= 4:
            DATA_BLUE_RAMP = tuple(str(c) for c in ramp)
            break


def _register_colormaps() -> None:
    from matplotlib.colors import LinearSegmentedColormap

    # A matrix that can hold "no value" cells starts one step into the
    # ramp, so the lowest value is visibly blue and cannot be taken for
    # the neutral "not reported" fill.
    ramps = {
        CMAP_SEQUENTIAL: DATA_BLUE_RAMP, CMAP_CELLS: DATA_BLUE_RAMP[1:], CMAP_GREENS: DATA_GREEN_RAMP,
        CMAP_REDS: DATA_RED_RAMP, CMAP_GREYS: DATA_GREY_RAMP, CMAP_DIVERGING: DATA_DIVERGING,
    }
    for name, colors in ramps.items():
        cmap = LinearSegmentedColormap.from_list(name, list(colors))
        cmap.set_bad("#f6f6f4" if name != CMAP_DIVERGING else "white")
        for registered in (cmap, cmap.reversed()):
            try:
                matplotlib.colormaps.register(registered, force=True)
            except Exception:
                pass


def _chart_font() -> str:
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return "DejaVu Sans"
    for name in _CHART_FONTS:
        if name in available:
            return name
    return "DejaVu Sans"


def chart_rc() -> Dict[str, object]:
    """The rcParams of the house chart style."""
    from cycler import cycler

    return {
        "font.family": [_CHART_FONT],
        "font.size": 9.5,
        "text.color": CHART_INK,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.titlesize": 12.0,
        "figure.titleweight": "semibold",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": CHART_AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": CHART_SECONDARY,
        "axes.labelsize": 9.5,
        "axes.labelpad": 6.0,
        "axes.titlesize": 11.0,
        "axes.titleweight": "semibold",
        "axes.titlecolor": CHART_INK,
        "axes.titlepad": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.prop_cycle": cycler(color=list(DATA_CATEGORICAL)),
        "grid.color": CHART_GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": CHART_SECONDARY,
        "ytick.color": CHART_SECONDARY,
        "xtick.labelcolor": CHART_SECONDARY,
        "ytick.labelcolor": CHART_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 8.5,
        "legend.labelcolor": CHART_SECONDARY,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.0,
        "image.cmap": CMAP_SEQUENTIAL,
        "boxplot.medianprops.color": CHART_INK,
        "boxplot.boxprops.color": BASE_COLOR,
        "boxplot.whiskerprops.color": CHART_SECONDARY,
        "boxplot.capprops.color": CHART_SECONDARY,
        "boxplot.flierprops.markeredgecolor": CHART_FAINT,
    }


def apply_chart_style() -> None:
    """Make the house style the default for everything drawn from now on.

    Called once on import and again by every function that lets seaborn
    near the rcParams. The app draws nothing through pyplot's own
    defaults, so the process-wide rcParams ARE the app's chart style."""
    matplotlib.rcParams.update(chart_rc())


_data_colors()
_register_colormaps()
_CHART_FONT = _chart_font()
apply_chart_style()


def series_colors(n: int) -> List[str]:
    """Colours for n categorical series: the fixed order, repeated when a
    plot has more series than the palette (the tooltip still names each)."""
    return [DATA_CATEGORICAL[i % len(DATA_CATEGORICAL)] for i in range(int(n))]


def ramp_colors(n: int, cmap: str = CMAP_SEQUENTIAL, low: float = 0.35, high: float = 0.95) -> List[tuple]:
    """n steps of one hue, darkest first: for bars that are RANKED (the
    first is the largest) rather than different in kind."""
    colormap = matplotlib.colormaps[cmap]
    if n <= 1:
        return [colormap(high)] * max(int(n), 0)
    return [colormap(high - (high - low) * (i / (n - 1))) for i in range(int(n))]


# ---------------------------------------------------------------------------
# What a chart tells the pointer. The interaction layer (qt_plot_interaction)
# can name any mark from the artists alone; these three calls give it what
# the artists do not hold: the full label of every category when the axis
# shows only some, the tank behind each point, the names of a matrix's rows
# and columns. They only write a dictionary on the axes -- no Qt, no cost.
# ---------------------------------------------------------------------------

MARKS_ATTR = "_ember_marks"
FIT_ATTR = "_ember_fit"
RELAYOUT_ATTR = "_ember_relayout"
MIN_SIZE_ATTR = "_ember_min_inches"


def declare_minimum_size(ax, width: float, height: float) -> None:
    """Say that the plot `ax` belongs to cannot be read in a pane smaller
    than `width` x `height` inches: a grid of charts, a matrix with
    marginal bars. In a small pane (see ENLARGE_BELOW_INCHES) the layout
    then shows one line asking for a larger window, instead of a grid
    squeezed until its labels run into each other. Above that size the
    plot is always drawn."""
    try:
        setattr(ax, MIN_SIZE_ATTR, (float(width), float(height)))
    except Exception:
        pass


def _marks(ax) -> Dict[str, object]:
    marks = getattr(ax, MARKS_ATTR, None)
    if not isinstance(marks, dict):
        marks = {}
        try:
            setattr(ax, MARKS_ATTR, marks)
        except Exception:
            pass
    return marks


def mark_categories(
    ax, labels: Sequence[object], *, axis: str = "y", name: str = "", value_label: str = "",
    unit: str = "", series_name: str = "", tank_ids: Optional[Sequence[object]] = None,
    extra: Optional[Sequence[Tuple[str, Sequence[object]]]] = None,
) -> None:
    """Category i (at position i on the category axis) is `labels[i]`."""
    marks = _marks(ax)
    marks["categories"] = [str(v) for v in labels]
    marks["category_axis"] = axis
    for key, value in (("category_name", name), ("value_label", value_label), ("unit", unit),
                       ("series_name", series_name)):
        if value:
            marks[key] = value
    if tank_ids is not None:
        marks["category_tank_ids"] = [None if v is None else str(v) for v in tank_ids]
    if extra:
        marks["category_extra"] = [(str(k), list(v)) for k, v in extra]


def mark_points(
    ax, artist, labels: Optional[Sequence[object]] = None, *, name: str = "",
    tank_ids: Optional[Sequence[object]] = None, x_label: str = "", y_label: str = "",
    series_name: str = "", extra: Optional[Sequence[Tuple[str, Sequence[object]]]] = None,
    show_position: bool = True,
) -> None:
    """Point i of the scatter `artist` is `labels[i]` (and stands for tank
    `tank_ids[i]`, when points are tanks or layers of tanks).
    show_position=False keeps x and y out of the tooltip, for points
    whose position is a layout, not a measurement (a network node)."""
    if artist is None:
        return
    entry: Dict[str, object] = {"artist": artist, "show_position": bool(show_position)}
    if labels is not None:
        entry["labels"] = [str(v) for v in labels]
    if tank_ids is not None:
        entry["tank_ids"] = [None if v is None else str(v) for v in tank_ids]
    for key, value in (("name", name), ("x_label", x_label), ("y_label", y_label), ("series_name", series_name)):
        if value:
            entry[key] = value
    if extra:
        entry["extra"] = [(str(k), list(v)) for k, v in extra]
    _marks(ax).setdefault("points", []).append(entry)


def mark_cells(
    ax, rows: Optional[Sequence[object]] = None, cols: Optional[Sequence[object]] = None, *,
    row_name: str = "", col_name: str = "", value_label: str = "", unit: Optional[str] = None,
    blank_text: str = "", triangle: str = "",
) -> None:
    """Row i / column j of the matrix drawn on `ax` are rows[i] / cols[j].
    triangle="lower" says only the cells on and below the diagonal exist
    (the other half of a symmetric matrix is left empty, not blank)."""
    cells: Dict[str, object] = {}
    if rows is not None:
        cells["rows"] = [str(v) for v in rows]
    if cols is not None:
        cells["cols"] = [str(v) for v in cols]
    for key, value in (("row_name", row_name), ("col_name", col_name), ("value_label", value_label),
                       ("blank_text", blank_text), ("triangle", triangle)):
        if value:
            cells[key] = value
    if unit is not None:
        cells["unit"] = unit                      # "" is a statement: this value has no unit
    _marks(ax)["cells"] = cells


def mark_frame_points(
    ax, frame: pd.DataFrame, x_col: str, y_col: str, *, id_col: str = "WasteSiteId", name: str = "Tank",
    extra_cols: Sequence[str] = (),
) -> None:
    """Label the scatter collections of `ax` that were drawn from `frame`
    (a seaborn call does not hand its artists back): a collection is
    matched when its points are exactly the frame's (x, y) rows, in order."""
    if frame is None or id_col not in frame.columns or not hasattr(ax, "collections"):
        return
    try:
        xy = frame[[x_col, y_col]].to_numpy(dtype=float)
    except Exception:
        return
    for coll in ax.collections:
        try:
            offsets = np.ma.filled(np.ma.asarray(coll.get_offsets(), dtype=float), np.nan)
        except Exception:
            continue
        if offsets.shape != xy.shape or not len(xy) or not np.allclose(offsets, xy, equal_nan=True):
            continue
        ids = frame[id_col].astype(str).tolist()
        mark_points(ax, coll, ids, name=name, tank_ids=ids,
                    extra=[(humanize(c), frame[c].tolist()) for c in extra_cols if c in frame.columns])


# ---------------------------------------------------------------------------
# Legible category axes. A ranking can hold more bars than the pane has
# lines of text for. The bars are all drawn (the data is not trimmed here);
# the LABELS are fitted to the room there is: a smaller font first, then
# every second or third label, with a sentence under the axis saying so.
# Long labels are cut with an ellipsis. The full text of every category
# stays on the axes for the tooltip.
# ---------------------------------------------------------------------------

LABEL_FONT_MAX = 8.5
LABEL_FONT_MIN = 6.5
LABEL_LINE_FACTOR = 1.22
LABEL_MAX_CHARS = 34


def elide_label(text: object, max_chars: int = LABEL_MAX_CHARS) -> str:
    text = str(text)
    if max_chars <= 1 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def declare_categories(
    ax, labels: Sequence[object], *, axis: str = "y", name: str = "", positions: Optional[Sequence[float]] = None,
    max_chars: int = LABEL_MAX_CHARS, rotation: Optional[float] = None, value_label: str = "",
    unit: str = "", series_name: str = "", tank_ids: Optional[Sequence[object]] = None,
    anchor: str = "", mark_word: str = "bar", note: bool = True, min_font: float = 0.0,
    record: bool = True,
) -> None:
    """Say that `ax` has a categorical axis: `labels[i]` at `positions[i]`
    (0..n-1 by default). Sets the ticks, fits the labels to the room
    available, and records the full labels for the tooltip.

    When labels have to be thinned, `anchor` says which one is always
    kept: "last" (the top bar of a ranking drawn bottom-up, the default
    on a y axis) or "first" (the top row of a matrix, the default on an
    x axis). `mark_word` is what the note calls a mark ("bar", "cell").
    note=False fits silently; min_font lowers the smallest size tried
    (short symbols stay legible smaller than names do); record=False
    leaves the tooltip's category list alone (a matrix names its rows and
    columns through mark_cells)."""
    labels = [str(v) for v in labels]
    if record:
        mark_categories(ax, labels, axis=axis, name=name, value_label=value_label, unit=unit,
                        series_name=series_name, tank_ids=tank_ids)
    if not hasattr(ax, "set_yticks" if axis == "y" else "set_xticks"):
        return                                    # a test double
    info = {
        "axis": axis, "labels": labels, "max_chars": int(max_chars), "rotation": rotation,
        "positions": list(range(len(labels))) if positions is None else [float(p) for p in positions],
        "name": name, "step": 1, "label_set": None, "mark_word": mark_word or "bar",
        "anchor": anchor or ("last" if axis == "y" else "first"), "note": bool(note),
        "min_font": float(min_font) if min_font else LABEL_FONT_MIN,
    }
    try:
        fits = [f for f in (getattr(ax, FIT_ATTR, None) or []) if f.get("axis") != axis]
        setattr(ax, FIT_ATTR, fits + [info])
    except Exception:
        return
    # Ticks now; the note about thinned labels only once the plot is
    # finished (relayout_figure). The plot function has usually not set
    # its axis label yet, and the note is a second line of that label.
    _fit_one(ax, info, with_note=False)


def category_fit(n: int, pitch_px: float, dpi: float, min_font: float = LABEL_FONT_MIN) -> Tuple[float, int]:
    """(font size in points, label every k-th category) for n categories
    whose centres are `pitch_px` apart."""
    def line_px(size: float) -> float:
        return size * dpi / 72.0 * LABEL_LINE_FACTOR

    if n <= 1 or pitch_px <= 0:
        return LABEL_FONT_MAX, 1
    size = LABEL_FONT_MAX
    while size > min_font and line_px(size) > pitch_px:
        size -= 0.5
    if line_px(size) <= pitch_px:
        return size, 1
    size = 7.5                                    # thinning: back to a comfortable size
    return size, max(2, int(math.ceil(line_px(size) / pitch_px)))


def fit_categories(ax) -> None:
    """(Re)fit the declared category axes of `ax` to its current size."""
    for info in list(getattr(ax, FIT_ATTR, None) or []):
        _fit_one(ax, info)


def _fit_one(ax, info: Dict[str, object], with_note: bool = True) -> None:
    labels, positions, axis = info["labels"], info["positions"], info["axis"]
    n = len(labels)
    if not n:
        return
    try:
        figure = ax.get_figure()
        box = ax.get_window_extent()
        extent = float(box.height if axis == "y" else box.width)
        dpi = float(figure.dpi)
        low, high = ax.get_ylim() if axis == "y" else ax.get_xlim()
    except Exception:
        extent, dpi, low, high = 0.0, 100.0, 0.0, float(n)
    spacing = float(np.min(np.diff(sorted(positions)))) if n > 1 else 1.0
    span = abs(high - low) or float(n)
    pitch = extent / span * (spacing or 1.0) if extent > 0 else 0.0
    size, step = (category_fit(n, pitch, dpi, float(info.get("min_font") or LABEL_FONT_MIN))
                  if pitch > 0 else (LABEL_FONT_MAX, 1))
    if info.get("anchor") == "last":
        chosen = [i for i in range(n) if (n - 1 - i) % step == 0]      # always the top-ranked bar
    else:
        chosen = [i for i in range(n) if i % step == 0]
    ticks = [positions[i] for i in chosen]
    # The containment pass (contain_region) may shorten the labels further,
    # or cap their size, for as long as the pane is too small for them.
    max_chars = int(info["max_chars"])
    if info.get("fit_chars"):
        max_chars = min(max_chars, int(info["fit_chars"]))
    if info.get("font_cap"):
        size = min(size, float(info["font_cap"]))
    shown = [elide_label(labels[i], max_chars) for i in chosen]
    kwargs = {"fontsize": size}
    if info.get("rotation") is not None:
        kwargs["rotation"] = info["rotation"]
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(shown, **kwargs)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(shown, **kwargs)
    info["step"] = step
    if with_note and info.get("note", True):
        _fit_note(ax, info, step)


def _ordinal(k: int) -> str:
    return {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(k, f"{k}th")


def _fit_note(ax, info: Dict[str, object], step: int) -> None:
    """Say under the value axis that not every bar is labelled. The note
    is a second line of the axis label, so the layout makes room for it;
    a label somebody else has set since (a Figure Composer override) is
    left alone."""
    getter = ax.get_xlabel if info["axis"] == "y" else ax.get_ylabel
    setter = ax.set_xlabel if info["axis"] == "y" else ax.set_ylabel
    artist = getattr(ax.xaxis if info["axis"] == "y" else ax.yaxis, "label", None)
    current = getter()
    last = info.get("label_set")
    if last is not None and current != last:
        # Wrapped for a narrow pane by the containment pass: still the
        # label written here. Anything else is somebody's override.
        if full_text(artist) != last:
            return                                # overridden: not ours to rewrite
        current = last
    base = info.get("label_base") if last is not None else current
    info["label_base"] = base
    if step <= 1:
        if last is not None and current != base:
            setter(base)
        info["label_set"] = None
        return
    noun = "label" if not info.get("name") else f"{str(info['name']).lower()} label"
    note = f"Every {_ordinal(step)} {noun} shown. Point at a {info.get('mark_word') or 'bar'} for its name."
    note = _note_for_cell(ax, artist, note)
    text = f"{base}\n{note}" if base else note
    if text != current:
        setter(text)
    info["label_set"] = text


def _note_for_cell(ax, label, note: str) -> str:
    """The note under an axis label, broken into lines no wider than the
    room its axes has: the whole pane for a single chart, one column of
    it when the axes is one of several side by side."""
    try:
        figure = ax.get_figure(root=False)
        width = float(figure.bbox.width) / _grid_share(ax)[0]
        em = float(label.get_fontsize()) * float(ax.get_figure(root=True).dpi) / 72.0
    except Exception:
        return note
    chars = int(0.94 * width / (0.50 * em))       # a character of this type is about half an em wide
    if chars >= len(note) or chars < 12:
        return note
    return _rewrap(note, chars)


def relayout_figure(figure, renderer=None) -> None:
    """Fit a figure's contents to the size it has NOW: category labels,
    any layout a plot function registered on its axes, the margins (when
    the plot asked for tight_layout), and last the containment pass, which
    checks that nothing is cut by the edge of the pane and corrects what
    is (see contain_region). Run after a render, after a resize, at a draw
    whose size no layout was made for, and around an export -- a layout
    computed for one size is wrong at another."""
    axes = list(getattr(figure, "axes", None) or [])
    if not axes:
        return
    if not CONTAINMENT or not getattr(figure, PANE_ATTR, False):
        # A cell of a composed figure while it is being drawn, a thumbnail,
        # a test double: the plot's own layout only. A cell is contained
        # when the composed figure is drawn (contain_cell).
        _own_layout(figure, axes)
        return
    if getattr(figure, "subfigs", None):
        for sub in figure.subfigs:
            contain_cell(sub, renderer, force=True)
        return
    state = _contain_state(figure, axes)
    try:
        release_containment(figure)
    except Exception:
        state["undo"] = []
    _own_layout(figure, axes)
    try:
        contain_region(figure, renderer, lambda: _own_layout(figure, axes, state))
    except Exception:
        pass                                      # cosmetic: never take a result off the screen
    state["key"] = _contain_key(figure)


def _own_layout(figure, axes, state: Optional[Dict[str, object]] = None) -> None:
    """The layout the plot itself asked for, at the figure's current size.
    With `state`, the margins start from the ones the plot function left,
    so that the result depends on the size alone and not on the sizes the
    figure had before."""
    if state is not None:
        _restore_margins(figure, state)
    for ax in axes:
        fit_categories(ax)
    moved = False
    for ax in axes:
        callback = getattr(ax, RELAYOUT_ATTR, None)
        if callable(callback):
            callback()
            moved = True
    apply_tight = getattr(figure, "apply_tight_layout", None)
    if callable(apply_tight) and getattr(figure, "_ember_tight", None) is not None:
        apply_tight()
        moved = True
    if moved:
        for ax in axes:
            fit_categories(ax)                    # the margins moved: the room changed


def figure_size_inches(figure) -> Tuple[float, float]:
    """The size a figure, or one cell of a composed figure, really has.
    (A SubFigure has no get_size_inches; the Figure Composer's adapter
    answers that call with a nominal size.)"""
    try:
        box, dpi = figure.bbox, float(figure.dpi)
        return float(box.width) / dpi, float(box.height) / dpi
    except Exception:
        width, height = figure.get_size_inches()
        return float(width), float(height)


def tighten_subfigure(subfig, renderer) -> None:
    """Fit the margins of one cell of a composed figure to its text.

    A SubFigure has no tight_layout: its axes keep matplotlib's default
    margins, so long tick labels are cut at the cell's edge and a subtitle
    of several lines climbs into the heading. This measures what the axes
    draw (tick labels, axis labels, titles, legends) and moves the four
    margins so that it fits inside the cell, below the heading. A heading
    wider than the cell is set in a smaller size. A plot that manages its
    own margins (it registered a relayout on its axes) is left to it."""
    from matplotlib.transforms import Bbox

    axes = [ax for ax in subfig.axes if ax.get_visible()]
    if not axes or any(callable(getattr(ax, RELAYOUT_ATTR, None)) for ax in axes):
        return
    box = subfig.bbox
    width, height = float(box.width), float(box.height)
    if width < 40 or height < 40:
        return
    dpi = float(subfig.dpi)
    pad = 0.07 * dpi
    top_limit = box.y1 - pad
    heading = getattr(subfig, "_suptitle", None)
    if heading is not None and heading.get_text():
        base = getattr(heading, "_ember_base_size", None)
        if base is None:
            base = heading._ember_base_size = float(heading.get_fontsize())
        size = base
        heading.set_fontsize(size)
        room = max(width - 0.62 * dpi, 0.5 * width)          # clear of the panel letter in the corner
        while size > 8.0 and heading.get_window_extent(renderer).width > room:
            size -= 0.5
            heading.set_fontsize(size)
        top_limit = heading.get_window_extent(renderer).y0 - 0.03 * dpi
    for _ in range(2):
        boxes = []
        for ax in axes:
            try:
                tight = ax.get_tightbbox(renderer)
            except Exception:
                tight = None
            if tight is not None and np.all(np.isfinite(tight.extents)):
                boxes.append(tight)
        if not boxes:
            return
        drawn = Bbox.union(boxes)
        pars = subfig.subplotpars
        left = pars.left + (box.x0 + pad - drawn.x0) / width
        right = pars.right - (drawn.x1 - (box.x1 - pad)) / width
        bottom = pars.bottom + (box.y0 + pad - drawn.y0) / height
        top = pars.top - (drawn.y1 - top_limit) / height
        left = min(max(left, 0.02), 0.60)
        right = max(min(right, 0.99), left + 0.20)
        bottom = min(max(bottom, 0.02), 0.55)
        top = max(min(top, 0.98), bottom + 0.20)
        if max(abs(left - pars.left), abs(right - pars.right), abs(bottom - pars.bottom), abs(top - pars.top)) < 0.004:
            break
        subfig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    for ax in axes:
        declutter_ticks(ax, renderer)


def declutter_ticks(ax, renderer) -> None:
    """Drop the tick labels of a log axis that would be drawn on top of
    their neighbours. A log axis shorter than a decade labels its minor
    ticks (2, 3, 4 x 10^n...), and in a narrow panel those collide into
    an unreadable run of digits. Labels are kept greedily along the axis,
    the decade labels first."""
    from matplotlib.ticker import FuncFormatter

    for axis, along_x in ((ax.xaxis, True), (ax.yaxis, False)):
        if axis.get_scale() != "log" or getattr(axis, "_ember_decluttered", False):
            continue
        try:
            minors = [t for t in axis.get_minorticklabels() if t.get_text() and t.get_visible()]
            majors = [t for t in axis.get_majorticklabels() if t.get_text() and t.get_visible()]
            if not minors:
                continue
            placed = [t.get_window_extent(renderer) for t in majors]
            keep = set()
            for label in minors:
                box = label.get_window_extent(renderer)
                gap = 3.0
                clash = any(
                    (box.x0 < other.x1 + gap and other.x0 < box.x1 + gap) if along_x
                    else (box.y0 < other.y1 + gap and other.y0 < box.y1 + gap)
                    for other in placed
                )
                if not clash:
                    placed.append(box)
                    keep.add(round(float(label.get_position()[0 if along_x else 1]), 12))
            if len(keep) == len(minors):
                continue
            inner = axis.get_minor_formatter()

            def _only_kept(value, pos=None, _inner=inner, _keep=frozenset(keep)):
                return _inner(value, pos) if round(float(value), 12) in _keep else ""

            axis.set_minor_formatter(FuncFormatter(_only_kept))
            axis._ember_decluttered = True
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Containment. A chart never extends outside its pane: no title, subtitle,
# axis label, tick label, legend, colour bar, annotation or note may be cut
# by the edge of the figure (in a composed figure: of its cell).
#
# overflow_report() is the measurable definition. contain_region() is the
# net every layout pass ends with, whichever function drew the chart: it
# measures, and only when something is outside does it correct, in this
# order: a text longer than the pane is wrapped (then set smaller, down to
# CONTAIN_FONT_FLOOR); the subplot margins move by the amount of the
# overflow; long tick labels are cut with an ellipsis (the full label stays
# with the tooltip); legends are set smaller, re-columned or moved under
# the axes; all type is set smaller. A pane too small for its chart even
# then says so in one line (ENLARGE_MESSAGE) rather than show a cut chart.
#
# Every correction is undone before the next layout pass, so the result is
# a function of the pane's size alone: laying out twice gives the same
# geometry, and a pane that grows back gets its full labels back.
# ---------------------------------------------------------------------------

CONTAINMENT = True                        # False: measure only (overflow_report), correct nothing
PANE_ATTR = "_ember_pane"                 # set on the figure of a PlotWidget
EXPORTING_ATTR = "_ember_exporting"       # set on that figure for the time of a save
CONTAIN_ATTR = "_ember_contain"
FULL_TEXT_ATTR = "_ember_full_text"
FULL_LABELS_ATTR = "_ember_full_labels"
CONTAIN_TOLERANCE_PX = 1.0
CONTAIN_PAD_IN = 0.04                     # room left between a corrected item and the edge
CONTAIN_FONT_FLOOR = 7.0
CONTAIN_TICK_FONT_FLOOR = 6.0
CONTAIN_MIN_AXES = 0.30                   # share of the pane the plotting area keeps when margins grow
# In a small pane a chart whose plotting area is left smaller than this
# (inches) is not shown as a chart any more: see ENLARGE_MESSAGE.
CONTAIN_MIN_PLOT_INCHES = (0.9, 0.5)
ENLARGE_MESSAGE = "Enlarge the window to see this chart."
ENLARGE_GID = "ember-enlarge"
PANEL_LABEL_GID = "ember-panel-label"
# The one-line message replaces a chart only in a pane this small (inches:
# 520 x 360 px at the screen's 100 dpi) in at least one direction.
ENLARGE_BELOW_INCHES = (5.2, 3.6)

_WRAPPED_KINDS = frozenset({
    "suptitle", "figure text", "title", "x label", "y label", "text", "supxlabel", "supylabel",
})


def _plain_figure(figure):
    """The matplotlib figure behind a Figure Composer adapter."""
    from matplotlib.figure import FigureBase
    if isinstance(figure, FigureBase):
        return figure
    return getattr(figure, "_figure", figure)


def _root_figure(region):
    try:
        return region.get_figure(root=True)
    except TypeError:
        return getattr(region, "figure", region)


def _layout_renderer(region, renderer=None):
    if renderer is not None:
        return renderer
    return _root_figure(region)._get_renderer()


def _own_axes(region) -> list:
    """The axes that belong to `region` itself, not to a cell inside it."""
    axes = list(region.axes)
    stack = list(getattr(region, "subfigs", None) or [])
    if not stack:
        return axes
    nested = set()
    while stack:
        sub = stack.pop()
        nested.update(id(ax) for ax in sub.axes)
        stack.extend(getattr(sub, "subfigs", None) or [])
    return [ax for ax in axes if id(ax) not in nested]


def full_text(artist) -> str:
    """The text of a title or label as the plot wrote it: the containment
    pass may have wrapped it for a narrow pane."""
    if artist is None:
        return ""
    drawn = str(artist.get_text() or "")
    kept = getattr(artist, FULL_TEXT_ATTR, None)
    # (as written, as wrapped): only while the artist still shows the
    # wrapped text; a text rewritten since stands for itself.
    if isinstance(kept, tuple) and len(kept) == 2 and kept[1] == drawn:
        return str(kept[0])
    return drawn


def full_tick_labels(axis) -> List[str]:
    """The major tick labels of `axis` in full: the containment pass may
    have cut the drawn ones with an ellipsis."""
    formatter = axis.get_major_formatter()
    inner = getattr(formatter, "ember_inner", None)
    if inner is not None:
        try:
            return [str(v) for v in inner.format_ticks(axis.get_majorticklocs())]
        except Exception:
            pass
    return [t.get_text() for t in axis.get_majorticklabels()]


def _text_box(text, renderer):
    """Window extent of a text as it is drawn (wrapped when it wraps, with
    its background patch), or None for an invisible or empty one."""
    if not text.get_visible():
        return None
    content = text.get_text()
    if not content or not content.strip():
        return None
    try:
        if text.get_wrap():
            text._renderer = renderer
            with text._cm_set(text=text._get_wrapped_text()):
                box = text.get_window_extent(renderer)
        else:
            box = text.get_window_extent(renderer)
        patch = text.get_bbox_patch()
        if patch is not None and patch.get_visible():
            from matplotlib.transforms import Bbox
            text.update_bbox_position_size(renderer)
            box = Bbox.union([box, patch.get_window_extent(renderer)])
    except Exception:
        return None
    return box


def _usable(box) -> bool:
    if box is None:
        return False
    try:
        extents = box.extents
    except Exception:
        return False
    return bool(np.all(np.isfinite(extents)) and (box.width > 0 or box.height > 0))


def _clipped_to_axes(artist) -> bool:
    """True for a mark drawn under a clip (bars, lines, points, images:
    whatever an axes method added is clipped to the frame). The clip is
    not compared with the frame, artist by artist: a chart can hold
    several hundred bars, and this runs at every layout."""
    return bool(artist.get_clip_on() and (artist.get_clip_box() is not None or artist.get_clip_path() is not None))


def _points_box(collection, dpi: float):
    """Window extent of the markers of a scatter. (matplotlib's own
    get_window_extent returns an empty box for a collection with offsets
    that is not clipped to its axes.)"""
    from matplotlib.transforms import Bbox

    try:
        offsets = np.ma.filled(np.ma.asarray(collection.get_offsets(), dtype=float), np.nan)
        if offsets.ndim != 2 or not len(offsets):
            return None
        xy = collection.get_offset_transform().transform(offsets)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if not len(xy):
            return None
        sizes = np.asarray(collection.get_sizes(), dtype=float)
        reach = (math.sqrt(float(sizes.max())) / 2.0 if sizes.size else 0.0) * dpi / 72.0
    except Exception:
        return None
    return Bbox.from_extents(xy[:, 0].min() - reach, xy[:, 1].min() - reach,
                             xy[:, 0].max() + reach, xy[:, 1].max() + reach)


def _surely_inside(text, bounds, dpi: float) -> bool:
    """True when a text anchored inside `bounds` cannot reach its edge,
    whatever its exact extent (a generous bound on its size, no layout):
    the hundreds of numbers written in the cells of a matrix are not
    measured one by one."""
    try:
        x, y = text.get_transform().transform(text.get_unitless_position())
        lines = text.get_text().split("\n")
        em = float(text.get_fontsize()) * dpi / 72.0
    except Exception:
        return False
    reach = max(max(len(line) for line in lines) * em, len(lines) * 1.7 * em) + 0.6 * em
    return bool(bounds.x0 + reach <= x <= bounds.x1 - reach and bounds.y0 + reach <= y <= bounds.y1 - reach)


def _place_axis_texts(axis, renderer, first: list, second: list) -> None:
    """Put the axis label and the offset text where a draw would, from the
    tick label boxes already measured (`first`: the bottom / left labels,
    `second`: the top / right ones). matplotlib measures every tick label
    again to do this; here it is handed the boxes."""
    try:
        groups = getattr(axis.get_figure(root=False), "_align_label_groups", None) or {}
        name = axis._get_axis_name()
        alone = name in groups and len(groups[name].get_siblings(axis.axes)) <= 1
    except Exception:
        alone = False
    try:
        if alone:
            axis._get_tick_boxes_siblings = lambda *args, **kwargs: (first, second)
            try:
                axis._update_label_position(renderer)
            finally:
                del axis._get_tick_boxes_siblings
        else:
            axis._update_label_position(renderer)
        axis._update_offset_text_position(first, second)
        axis.offsetText.set_text(axis.major.formatter.get_offset())
    except Exception:
        try:
            axis.get_tightbbox(renderer)
        except Exception:
            pass


def _place_titles(ax, renderer, titles) -> None:
    """Put the titles where a draw would. An axes with nothing above its
    frame (no tick labels or axis label on top, no offset text, no twin or
    inset axes) keeps its titles at the top of the frame: nothing to do."""
    if not any(title.get_text() for title in titles):
        return
    try:
        simple = (
            not ax.child_axes and len(ax._twinned_axes.get_siblings(ax)) <= 1
            and len(ax.get_figure(root=False)._align_label_groups["title"].get_siblings(ax)) <= 1
            and ax.xaxis.get_label_position() != "top"
            and not any(tick.label2.get_visible() for tick in (ax.xaxis.majorTicks[0], ax.xaxis.minorTicks[0]))
            and not ax.yaxis.offsetText.get_text()
            and all(abs(float(title.get_position()[1]) - 1.0) < 1e-9 for title in titles)
        )
    except Exception:
        simple = False
    if simple:
        return
    try:
        ax._update_title_position(renderer)
    except Exception:
        pass


def _axes_items(ax, renderer, out: list, bounds=None) -> None:
    """Append (kind, artist, box, True) for what `ax` draws that could
    reach outside the figure: the frame with its tick marks, tick labels,
    axis labels, titles, legend, texts, marks drawn unclipped, and the
    same for its inset axes (a colour bar set inside the axes' box)."""
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend
    from matplotlib.text import Annotation, Text
    from matplotlib.transforms import Bbox

    if not ax.get_visible():
        return
    try:
        locator = ax.get_axes_locator()
        ax.apply_aspect(locator(ax, renderer) if locator else None)
    except Exception:
        pass
    x0, y0, x1, y1 = ax.get_window_extent(renderer).extents
    dpi = float(_root_figure(ax).dpi)
    axis_map = getattr(ax, "_axis_map", None) or {"x": ax.xaxis, "y": ax.yaxis}
    if ax.axison:
        for name, axis in axis_map.items():
            if not axis.get_visible():
                continue
            try:
                ticks = axis._update_ticks()
            except Exception:
                continue
            # Each tick label is measured once; the same boxes place the
            # axis label and the offset text (what Axis.get_tightbbox does
            # in two more passes over the labels).
            sides: Tuple[list, list] = ([], [])
            kind = f"{name} tick label"
            for tick in ticks:
                for label, boxes in ((tick.label1, sides[0]), (tick.label2, sides[1])):
                    if not label.get_visible():
                        continue
                    try:
                        box = label.get_window_extent(renderer)
                    except Exception:
                        continue
                    boxes.append(box)
                    if box.width > 0 and box.height > 0 and np.all(np.isfinite(box.extents)):
                        out.append((kind, label, box, True))
            _place_axis_texts(axis, renderer, sides[0], sides[1])
            if ticks:
                reach = float(ticks[0].get_tick_padding()) * dpi / 72.0
                if ticks[0].tick1line.get_visible():
                    if name == "x":
                        y0 -= reach
                    elif name == "y":
                        x0 -= reach
                if ticks[0].tick2line.get_visible():
                    if name == "x":
                        y1 += reach
                    elif name == "y":
                        x1 += reach
            for kind, text in ((f"{name} label", axis.label), (f"{name} offset text", axis.offsetText)):
                box = _text_box(text, renderer)
                if _usable(box):
                    out.append((kind, text, box, True))
    titles = (ax.title, ax._left_title, ax._right_title)
    _place_titles(ax, renderer, titles)
    for title in titles:
        box = _text_box(title, renderer)
        if _usable(box):
            out.append(("title", title, box, True))

    skip = {id(a) for a in axis_map.values()} | {id(t) for t in titles} | {id(ax.patch)}
    skip |= {id(s) for s in ax.spines.values()}
    inked = False
    for artist in ax.get_children():
        if id(artist) in skip or not artist.get_visible():
            continue
        if isinstance(artist, Axes):
            _axes_items(artist, renderer, out, bounds)
            continue
        try:
            if isinstance(artist, Legend):
                kind, box = "legend", artist.get_window_extent(renderer)
            elif isinstance(artist, Annotation):
                kind, box = "annotation", artist.get_tightbbox(renderer)
            elif isinstance(artist, Text):
                if bounds is not None and _surely_inside(artist, bounds, dpi):
                    continue
                kind, box = "text", _text_box(artist, renderer)
                if box is not None and artist.get_clip_on() and artist.get_clip_box() is not None:
                    box = Bbox.intersection(box, artist.get_clip_box())
            else:
                if _clipped_to_axes(artist):
                    inked = True                          # drawn inside the frame, whatever its extent
                    continue
                if artist.get_alpha() == 0:
                    continue
                kind, box = "mark", artist.get_tightbbox(renderer)
                if not _usable(box) and hasattr(artist, "get_offsets"):
                    box = _points_box(artist, dpi)
        except Exception:
            continue
        if _usable(box):
            out.append((kind, artist, box, True))
    if ax.axison or inked:
        out.append(("axes", ax, Bbox.from_extents(x0, y0, x1, y1), True))


def _figure_items(region, renderer, out: list) -> None:
    """Append (kind, artist, box, False) for what the figure (or cell)
    itself carries: its heading, notes, legends and loose artists."""
    special = {
        id(getattr(region, "_suptitle", None)): "suptitle",
        id(getattr(region, "_supxlabel", None)): "supxlabel",
        id(getattr(region, "_supylabel", None)): "supylabel",
    }
    for text in list(region.texts):
        box = _text_box(text, renderer)
        if _usable(box):
            out.append((special.get(id(text), "figure text"), text, box, False))
    for legend in list(region.legends):
        if not legend.get_visible():
            continue
        try:
            box = legend.get_window_extent(renderer)
        except Exception:
            continue
        if _usable(box):
            out.append(("figure legend", legend, box, False))
    for artist in list(region.artists) + list(region.lines) + list(region.patches) + list(region.images):
        if not artist.get_visible():
            continue
        try:
            box = artist.get_tightbbox(renderer)
        except Exception:
            continue
        if _usable(box):
            out.append(("figure artist", artist, box, False))


def _layout_items(region, renderer, bounds=None) -> List[tuple]:
    out: List[tuple] = []
    _figure_items(region, renderer, out)
    for ax in _own_axes(region):
        _axes_items(ax, renderer, out, bounds)
    return out


def _sides(box, bounds, tolerance: float = CONTAIN_TOLERANCE_PX) -> Optional[Dict[str, float]]:
    """By how many pixels `box` lies outside `bounds` on each side, or
    None when it is inside (within the tolerance)."""
    over = {
        "left": float(bounds.x0 - box.x0), "right": float(box.x1 - bounds.x1),
        "bottom": float(bounds.y0 - box.y0), "top": float(box.y1 - bounds.y1),
    }
    if max(over.values()) <= tolerance:
        return None
    return {side: round(max(value, 0.0), 2) for side, value in over.items()}


def _artist_text(artist) -> str:
    getter = getattr(artist, "get_text", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return ""
    title = getattr(artist, "get_title", None)
    if callable(title):
        try:
            got = title()
            return str(got.get_text() if hasattr(got, "get_text") else got)
        except Exception:
            return ""
    return ""


def _spills(legend, box) -> bool:
    """True for a legend placed inside its axes (no anchor of its own)
    that is larger than those axes: it then covers the title above them
    or the labels beside them."""
    ax = getattr(legend, "axes", None)
    if ax is None or getattr(legend, "_bbox_to_anchor", None) is not None:
        return False
    frame = ax.bbox
    return bool(box.x0 < frame.x0 - 4 or box.x1 > frame.x1 + 4 or box.y0 < frame.y0 - 4 or box.y1 > frame.y1 + 4)


def _region_overflow(region, renderer, notes: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    """The overflow report of one region (its cells not included). With
    `notes`, notes["spill"] says whether a legend is larger than its axes."""
    bounds = region.bbox
    report: List[Dict[str, object]] = []
    for kind, artist, box, attached in _layout_items(region, renderer, bounds):
        if notes is not None and kind == "legend" and _spills(artist, box):
            notes["spill"] = True
        over = _sides(box, bounds)
        if over is None:
            continue
        entry: Dict[str, object] = {
            "kind": kind, "text": _artist_text(artist), "artist": artist, "attached": attached,
            "box": tuple(round(float(v), 2) for v in box.extents),
            "bounds": tuple(round(float(v), 2) for v in bounds.extents), "region": region,
        }
        entry.update(over)
        report.append(entry)
    return report


def overflow_report(figure, renderer=None) -> List[Dict[str, object]]:
    """Everything a figure draws outside itself.

    One entry per visible artist carrying text or ink whose window extent
    lies outside the figure rectangle by more than CONTAIN_TOLERANCE_PX:
    its `kind` ("title", "y tick label", "legend", "axes" for a frame or a
    colour bar...), its `text`, and by how many pixels it is out on each
    side (`left`, `right`, `bottom`, `top`). Looks at every axes, inset
    axes included, at legends, and at the figure's own heading and notes.
    In a composed figure each cell is held to its own rectangle.

    Not overflow: an empty or invisible artist, and marks clipped to their
    axes on purpose (data under clip_on is cut by the frame, by design).
    An empty list is the statement "nothing is cut"."""
    region = _plain_figure(figure)
    renderer = _layout_renderer(region, renderer)
    report: List[Dict[str, object]] = []
    stack = [region]
    while stack:
        current = stack.pop()
        report.extend(_region_overflow(current, renderer))
        stack.extend(getattr(current, "subfigs", None) or [])
    return report


def describe_overflow(report: Sequence[Dict[str, object]], limit: int = 8) -> str:
    """A readable account of an overflow report, worst first."""
    def worst(entry) -> float:
        return max(float(entry[s]) for s in ("left", "right", "bottom", "top"))

    lines = []
    for entry in sorted(report, key=worst, reverse=True)[:limit]:
        sides = ", ".join(f"{side} {entry[side]:.0f} px" for side in ("left", "right", "bottom", "top")
                          if float(entry[side]) > CONTAIN_TOLERANCE_PX)
        text = str(entry.get("text") or "").replace("\n", " / ")
        lines.append(f"{entry['kind']} {text[:48]!r}: {sides}")
    if len(report) > limit:
        lines.append(f"... and {len(report) - limit} more")
    return "; ".join(lines)


def showing_enlarge_message(figure, cells: bool = False) -> bool:
    """True when the pane shows ENLARGE_MESSAGE in place of its chart
    (with `cells`: or any cell of a composed figure does)."""
    region = _plain_figure(figure)
    if any(text.get_gid() == ENLARGE_GID and text.get_visible() for text in getattr(region, "texts", [])):
        return True
    return bool(cells and any(showing_enlarge_message(sub, True) for sub in getattr(region, "subfigs", None) or []))


# -- state ------------------------------------------------------------------

def _margins(region) -> Tuple[float, float, float, float]:
    pars = region.subplotpars
    return (float(pars.left), float(pars.right), float(pars.bottom), float(pars.top))


def _default_margins() -> Tuple[float, float, float, float]:
    rc = matplotlib.rcParams
    return (float(rc["figure.subplot.left"]), float(rc["figure.subplot.right"]),
            float(rc["figure.subplot.bottom"]), float(rc["figure.subplot.top"]))


def _spacing(region) -> Tuple[float, float]:
    pars = region.subplotpars
    return (float(pars.wspace), float(pars.hspace))


def _default_spacing() -> Tuple[float, float]:
    rc = matplotlib.rcParams
    return (float(rc["figure.subplot.wspace"]), float(rc["figure.subplot.hspace"]))


def _contain_state(region, axes: Sequence[object], base: Optional[tuple] = None,
                   spacing: Optional[tuple] = None) -> Dict[str, object]:
    """What the containment pass remembers about one figure or cell: how
    to undo its corrections, and the margins and the spacing between
    subplots every layout starts from (`base` and `spacing`, or the ones
    the plot function left). A new drawing (other axes) starts a new
    state."""
    ids = tuple(id(ax) for ax in axes)
    state = getattr(region, CONTAIN_ATTR, None)
    if not isinstance(state, dict) or state.get("axes") != ids:
        state = {
            "undo": [], "axes": ids, "key": None,
            "base": tuple(base) if base else _margins(region),
            "spacing": tuple(spacing) if spacing else _spacing(region),
        }
        setattr(region, CONTAIN_ATTR, state)
    return state


def _restore_margins(region, state: Dict[str, object]) -> None:
    """Back to the margins and the spacing every layout starts from, so
    that a layout depends on the size of the pane and not on the layout
    that came before it."""
    base, spacing = state.get("base"), state.get("spacing") or _spacing(region)
    if not base:
        return
    now = tuple(round(v, 6) for v in _margins(region) + _spacing(region))
    if now == tuple(round(v, 6) for v in tuple(base) + tuple(spacing)):
        return
    left, right, bottom, top = base
    try:
        region.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                               wspace=spacing[0], hspace=spacing[1])
    except Exception:
        pass


def release_containment(region) -> None:
    """Undo what the last containment pass changed (wrapped or smaller
    text, cut labels, moved legends, margins, the one-line message), so
    that the next layout starts from what the plot function drew."""
    state = getattr(region, CONTAIN_ATTR, None)
    if not isinstance(state, dict):
        return
    undo, state["undo"] = list(state.get("undo") or []), []
    for step in reversed(undo):
        try:
            step()
        except Exception:
            continue
    _restore_margins(region, state)


def _contain_key(region) -> tuple:
    """What a layout was made for: the size, the axes and the wording. A
    draw that finds another key lays the figure out again."""
    box = region.bbox
    axes = _own_axes(region)
    words = []
    for ax in axes:
        # The first line of an axis label: the lines under it are the note
        # about thinned labels, which follows the geometry, not the plot.
        words.extend((ax.title.get_text(), ax.xaxis.label.get_text().split("\n")[0],
                      ax.yaxis.label.get_text().split("\n")[0]))
    suptitle = getattr(region, "_suptitle", None)
    words.append(suptitle.get_text() if suptitle is not None else "")
    return (
        round(float(box.width), 1), round(float(box.height), 1), round(float(_root_figure(region).dpi), 3),
        tuple(id(ax) for ax in axes), hash(tuple(words)), len(region.texts), len(region.legends),
    )


def _own_subplotpars(subfig) -> None:
    """Give a cell of a composed figure margins of its own. matplotlib
    lets every SubFigure share its parent's SubplotParams object, so the
    margins fitted for one cell would be the starting point of the next."""
    parent = getattr(subfig, "_parent", None)
    if parent is None or subfig.subplotpars is not getattr(parent, "subplotpars", None):
        return
    subfig.subplotpars = type(subfig.subplotpars)()
    try:
        subfig.subplots_adjust()
    except Exception:
        pass


# -- corrections ------------------------------------------------------------

def _rewrap(text: str, chars: int, break_words: bool = False) -> str:
    import textwrap

    chars = max(int(chars), 4)
    lines: List[str] = []
    for line in str(text).split("\n"):
        if len(line) <= chars:
            lines.append(line)
        elif SUBTITLE_SEP in line and not break_words:
            lines.extend(wrap_subtitle(line.split(SUBTITLE_SEP), width=chars).split("\n"))
        else:
            # Lines of about the same length, rather than full lines and
            # one orphan word at the end.
            count = max(1, math.ceil(len(line) / chars))
            even = textwrap.wrap(line, width=min(chars, math.ceil(len(line) / count) + 4),
                                 break_long_words=break_words, break_on_hyphens=break_words)
            if not even or len(even) > count:
                even = textwrap.wrap(line, width=chars, break_long_words=break_words,
                                     break_on_hyphens=break_words) or [line]
            lines.extend(even)
    return "\n".join(lines)


def _wrap_to_fit(text, box, bounds, pad: float, renderer, undo: list, share: int = 1) -> bool:
    """Wrap a text that is longer than the pane has room for around its
    anchor; when wrapping is not enough (one long word, or more lines than
    a heading should take), set it smaller, down to CONTAIN_FONT_FLOOR.
    `share`: how many axes stand side by side along the text (the title
    of one of three columns is asked to fit a third of the pane)."""
    rotation = float(text.get_rotation()) % 360.0
    vertical = 45.0 < rotation % 180.0 < 135.0
    try:
        ax_, ay_ = text.get_transform().transform(text.get_unitless_position())
    except Exception:
        ax_, ay_ = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    if vertical:
        low, high, anchor = bounds.y0 + pad, bounds.y1 - pad, float(ay_)
    else:
        low, high, anchor = bounds.x0 + pad, bounds.x1 - pad, float(ax_)
    whole = high - low
    align = text.get_horizontalalignment()
    if vertical and rotation > 180.0:
        align = {"left": "right", "right": "left"}.get(align, align)
    if align == "left":
        room = high - anchor
    elif align == "right":
        room = anchor - low
    else:
        room = 2.0 * min(anchor - low, high - anchor)
    # An anchor near an edge is for the margins to move; the text is asked
    # to fit a fair share of the pane, not a sliver.
    room = min(whole, max(room, 0.6 * whole / max(int(share), 1)))
    if room <= 0:
        return False

    def length() -> float:
        measured = _text_box(text, renderer)
        if measured is None:
            return 0.0
        return float(measured.height if vertical else measured.width)

    if length() <= room + CONTAIN_TOLERANCE_PX:
        return False                              # short enough: where it sits is the problem
    current, size0 = text.get_text(), float(text.get_fontsize())
    original = full_text(text)                    # as the plot wrote it, if an earlier step wrapped it
    if original != current:
        text.set_text(original)
    full = length()
    paragraphs = original.split("\n")
    longest = max(len(line) for line in paragraphs)
    most_lines = len(paragraphs) + (3 if (SUBTITLE_SEP in original or len(paragraphs) > 1) else 2)

    def plan(size: float) -> Tuple[int, int]:
        """(characters per line, lines) the text takes at `size`, from the
        one measurement made: a text's length is proportional to its size."""
        chars = max(4, int(0.97 * longest * room / (full * size / size0)))
        return chars, sum(max(1, math.ceil(len(line) / chars)) for line in paragraphs)

    # The largest size at which the text is a heading and not a paragraph.
    size = size0
    while size > CONTAIN_FONT_FLOOR + 1e-9 and plan(size)[1] > most_lines:
        size = max(CONTAIN_FONT_FLOOR, size - 0.5)
    for attempt in (size, CONTAIN_FONT_FLOOR):
        text.set_fontsize(attempt)
        chars = plan(attempt)[0]
        for _ in range(5):
            text.set_text(_rewrap(original, chars))
            if length() <= room:
                break
            chars = max(4, min(chars - 1, int(chars * 0.92)))
        if length() <= room or attempt <= CONTAIN_FONT_FLOOR + 1e-9:
            break
    if length() > room:
        text.set_text(_rewrap(original, chars, break_words=True))         # one word wider than the pane
    final_text, final_size = text.get_text(), float(text.get_fontsize())
    if final_text == current and abs(final_size - size0) < 1e-9:
        return False
    before = getattr(text, FULL_TEXT_ATTR, None)
    try:
        setattr(text, FULL_TEXT_ATTR, (original, final_text) if final_text != original else None)
    except Exception:
        pass

    def _undo() -> None:
        # A text somebody has rewritten since (a Figure Composer override)
        # is theirs: only what this pass wrote is taken back.
        if text.get_text() == final_text:
            text.set_text(current)
            try:
                setattr(text, FULL_TEXT_ATTR, before)
            except Exception:
                pass
        if abs(float(text.get_fontsize()) - final_size) < 1e-9:
            text.set_fontsize(size0)

    undo.append(_undo)
    return True


def _fit_long_texts(region, renderer, pad: float, undo: list,
                    report: Optional[Sequence[Dict[str, object]]] = None) -> Tuple[bool, bool]:
    """Wrap the texts that reach outside the pane because they are longer
    than it (titles, subtitles, axis labels, notes). `report` is the
    overflow just measured, when there is one to reuse. Returns (anything
    changed, a heading or a note of the figure itself changed: the band
    it takes at the top or the bottom of the pane has grown)."""
    from matplotlib.transforms import Bbox

    bounds = region.bbox
    if report is None:
        outside = [(kind, artist, box, attached)
                   for kind, artist, box, attached in _layout_items(region, renderer, bounds)
                   if _sides(box, bounds) is not None]
    else:
        outside = [(entry["kind"], entry["artist"], Bbox.from_extents(*entry["box"]), entry["attached"])
                   for entry in report]
    if not any(kind in _WRAPPED_KINDS for kind, _artist, _box, _attached in outside):
        return False, False
    shares: Dict[int, Tuple[int, int]] = {}
    for ax in _own_axes(region):
        grid = _grid_share(ax)
        if grid != (1, 1):
            for text in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
                shares[id(text)] = grid
    changed = own = False
    for kind, artist, box, attached in outside:
        if kind not in _WRAPPED_KINDS:
            continue
        if getattr(artist, "get_gid", lambda: None)() in (ENLARGE_GID, PANEL_LABEL_GID):
            continue
        cols, rows = shares.get(id(artist), (1, 1))
        upright = 45.0 < float(artist.get_rotation()) % 180.0 < 135.0
        if _wrap_to_fit(artist, box, bounds, pad, renderer, undo, rows if upright else cols):
            changed, own = True, own or not attached
    return changed, own


def _inner_bounds(region, renderer):
    """The rectangle the axes and what hangs on them have to stay in: the
    pane, less the bands taken by its heading and by a legend or a note
    that runs along its bottom or top edge."""
    from matplotlib.transforms import Bbox

    bounds = region.bbox
    x0, y0, x1, y1 = bounds.extents
    height, width = float(bounds.height), float(bounds.width)
    gap = 0.03 * float(_root_figure(region).dpi)
    figure_level: List[tuple] = []
    _figure_items(region, renderer, figure_level)
    for kind, artist, box, _ in figure_level:
        if kind == "figure artist" or getattr(artist, "get_gid", lambda: None)() in (ENLARGE_GID, PANEL_LABEL_GID):
            continue
        if kind == "figure text" and box.width < 0.3 * width:
            continue
        centre = (box.y0 + box.y1) / 2.0
        if kind == "suptitle" or centre > bounds.y0 + 0.7 * height:
            y1 = min(y1, box.y0 - gap)
        elif centre < bounds.y0 + 0.3 * height:
            y0 = max(y0, box.y1 + gap)
    if y1 - y0 < 0.3 * height:
        return bounds
    return Bbox.from_extents(x0, y0, x1, y1)


def _fit_margins(region, renderer, pad: float) -> bool:
    """Move the subplot margins by the amount the axes and what hangs on
    them (labels, titles, legends, colour bars) reach outside the pane.
    Returns False when they cannot make enough room: the plotting area
    keeps CONTAIN_MIN_AXES of the pane."""
    axes = _own_axes(region)
    if not axes:
        return True
    whole = region.bbox
    width, height = float(whole.width), float(whole.height)
    if width <= 0 or height <= 0:
        return True
    resolved = False
    for _ in range(6):
        inner = _inner_bounds(region, renderer)
        over = [0.0, 0.0, 0.0, 0.0]               # left, right, bottom, top
        items: List[tuple] = []
        for ax in axes:
            _axes_items(ax, renderer, items, inner)
        for _kind, _artist, box, _attached in items:
            over[0] = max(over[0], inner.x0 - box.x0)
            over[1] = max(over[1], box.x1 - inner.x1)
            over[2] = max(over[2], inner.y0 - box.y0)
            over[3] = max(over[3], box.y1 - inner.y1)
        if max(over) <= CONTAIN_TOLERANCE_PX:
            resolved = True
            break
        left, right, bottom, top = _margins(region)
        move = [(value + pad) if value > CONTAIN_TOLERANCE_PX else 0.0 for value in over]
        d_left, d_right, d_bottom, d_top = move[0] / width, move[1] / width, move[2] / height, move[3] / height
        clamped = False
        spare_w = max((right - left) - min(CONTAIN_MIN_AXES, right - left), 0.0)
        if d_left + d_right > spare_w:
            scale = spare_w / (d_left + d_right)
            d_left, d_right, clamped = d_left * scale, d_right * scale, True
        spare_h = max((top - bottom) - min(CONTAIN_MIN_AXES, top - bottom), 0.0)
        if d_bottom + d_top > spare_h:
            scale = spare_h / (d_bottom + d_top)
            d_bottom, d_top, clamped = d_bottom * scale, d_top * scale, True
        if max(d_left, d_right, d_bottom, d_top) < 1e-4:
            break
        try:
            region.subplots_adjust(left=left + d_left, right=right - d_right,
                                   bottom=bottom + d_bottom, top=top - d_top)
        except Exception:
            break
        for ax in axes:
            fit_categories(ax)
        if clamped and max(d_left + d_right, d_bottom + d_top) < 2e-3:
            break
    return resolved


def _grid_share(ax) -> Tuple[int, int]:
    """(columns, rows) of the grid `ax` is one cell of."""
    try:
        spec = ax.get_subplotspec().get_topmost_subplotspec()
        grid = spec.get_gridspec()
        cols = max(1, int(round(grid.ncols / max(len(spec.colspan), 1))))
        rows = max(1, int(round(grid.nrows / max(len(spec.rowspan), 1))))
        return min(cols, 4), min(rows, 4)
    except Exception:
        return 1, 1


def _elided_formatter(inner, chars: int):
    from matplotlib.ticker import Formatter

    class _Elided(Formatter):
        """The axis' own formatter, with long labels cut by an ellipsis."""

        def __init__(self) -> None:
            self.ember_inner = inner
            self.ember_chars = int(chars)

        def __call__(self, x, pos=None):
            return elide_label(inner(x, pos), self.ember_chars)

        def format_ticks(self, values):
            return [elide_label(v, self.ember_chars) for v in inner.format_ticks(values)]

        def get_offset(self):
            return inner.get_offset()

        def set_locs(self, locs):
            inner.set_locs(locs)

    return _Elided()


def _elide_tick_labels(region, renderer, share: float, undo: list) -> bool:
    """Cut the names along a category axis so that they take at most
    `share` of the pane (of their grid cell's share of it). The tooltip
    keeps the full name."""
    bounds = region.bbox
    changed = False
    for ax in _own_axes(region):
        if not ax.get_visible() or not ax.axison:
            continue
        cols, rows = _grid_share(ax)
        fits = {info.get("axis"): info for info in (getattr(ax, FIT_ATTR, None) or [])}
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            try:
                labels = [label for tick in axis._update_ticks() for label in (tick.label1, tick.label2)
                          if label.get_visible() and label.get_text().strip()]
            except Exception:
                continue
            if not labels:
                continue
            longest = max(len(label.get_text()) for label in labels)
            info = fits.get(name)
            if info is None and (longest <= 10 or not any(c.isalpha() for label in labels for c in label.get_text())):
                continue                                  # numbers: nothing to cut
            boxes = [box for box in (_text_box(label, renderer) for label in labels) if box is not None]
            if not boxes:
                continue
            # The direction the labels take room from the plotting area in:
            # across the pane for a y axis, down it for a tilted x axis.
            turn = float(labels[0].get_rotation()) % 180.0
            if (name == "y" and not 45.0 < turn < 135.0) or (name == "x" and not 20.0 < turn < 160.0):
                extent, cap = max(b.width for b in boxes), share * float(bounds.width) / cols
            else:
                extent, cap = max(b.height for b in boxes), share * float(bounds.height) / rows
            if extent <= cap:
                continue
            chars = max(5, int(longest * cap / extent))
            if chars >= longest:
                continue
            if info is not None:
                previous = info.get("fit_chars")
                info["fit_chars"] = chars
                _fit_one(ax, info)

                def _undo(info=info, previous=previous) -> None:
                    if previous:
                        info["fit_chars"] = previous
                    else:
                        info.pop("fit_chars", None)

                undo.append(_undo)
            else:
                inner = axis.get_major_formatter()
                inner = getattr(inner, "ember_inner", None) or inner
                wrapper = _elided_formatter(inner, chars)
                before = axis.get_major_formatter()
                axis.set_major_formatter(wrapper)

                def _undo(axis=axis, wrapper=wrapper, before=before) -> None:
                    if axis.get_major_formatter() is wrapper:
                        axis.set_major_formatter(before)

                undo.append(_undo)
            changed = True
    return changed


def _legends(region) -> List[tuple]:
    """(owner, legend) for every visible legend of the region."""
    found = [(region, legend) for legend in list(region.legends) if legend.get_visible()]
    for ax in _own_axes(region):
        legend = ax.get_legend() if ax.get_visible() else None
        if legend is not None and legend.get_visible():
            found.append((ax, legend))
    return found


def _place_legend(owner, current, new) -> None:
    if hasattr(owner, "legend_"):
        owner.legend_ = new
        new._remove_method = owner._remove_legend
    else:
        owner.legends[owner.legends.index(current)] = new
        new._remove_method = owner.legends.remove
    owner.stale = True


def _refit_legend(region, owner, legend, renderer, pad: float, below: bool, undo: list) -> bool:
    """Rebuild a legend so that it fits the pane: smaller type and tighter
    spacing, fewer columns when it is too wide, more when it is too tall;
    `below` moves a legend of an axes under that axes, in as many columns
    as the pane is wide. The entries are the same handles and labels."""
    from matplotlib.legend import Legend
    from matplotlib.transforms import ScaledTranslation

    handles = list(getattr(legend, "legend_handles", None) or [])
    labels = [text.get_text() for text in legend.get_texts()]
    count = min(len(handles), len(labels))
    if not count:
        return False
    handles, labels = handles[:count], labels[:count]
    bounds = region.bbox
    is_axes = hasattr(owner, "legend_")
    if below and not is_axes:
        return False
    try:
        box = legend.get_window_extent(renderer)
        size0 = float(legend.get_texts()[0].get_fontsize())
    except Exception:
        return False
    max_w = float(bounds.width) - 2.0 * pad
    max_h = (0.45 if below else 1.0) * float(bounds.height) - 2.0 * pad
    if below:
        # Under its own axes: in a grid, not under the neighbours too.
        max_w /= _grid_share(owner)[0]
    if is_axes and not below and getattr(legend, "_bbox_to_anchor", None) is None:
        # Placed inside its axes: that is the room it has.
        max_w = min(max_w, float(owner.bbox.width) - 2.0 * pad)
        max_h = min(max_h, float(owner.bbox.height) - 2.0 * pad)
    columns0 = max(1, int(getattr(legend, "_ncols", 1) or 1))
    title = legend.get_title().get_text() if legend.get_title() is not None else ""
    kwargs: Dict[str, object] = {
        "frameon": legend.get_frame_on(), "labelspacing": 0.25, "handletextpad": 0.4, "columnspacing": 1.0,
        "handlelength": min(float(getattr(legend, "handlelength", 2.0) or 2.0), 1.2), "borderpad": 0.2,
        "borderaxespad": float(getattr(legend, "borderaxespad", 0.5) or 0.0),
    }
    if below:
        low = owner.bbox.y0
        try:
            decorations = [b for b in (owner.xaxis.get_tightbbox(renderer),) if b is not None]
            low = min([low] + [float(b.y0) for b in decorations])
        except Exception:
            pass
        root = _root_figure(region)
        drop = (float(owner.bbox.y0) - low) / float(root.dpi) + 0.06
        kwargs.update(loc="upper center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0.0,
                      bbox_transform=owner.transAxes + ScaledTranslation(0.0, -drop, root.dpi_scale_trans))
        order = list(range(min(count, 8), 0, -1))
    elif box.width > max_w:
        order = list(range(columns0, 0, -1))
    else:
        order = list(range(columns0, min(count, 6) + 1))
    best, best_excess = None, None
    for size in dict.fromkeys((min(size0, 8.0), CONTAIN_FONT_FLOOR)):
        for columns in order:
            options = dict(kwargs, fontsize=size, ncols=columns)
            if title:
                options.update(title=title, title_fontsize=size)
            try:
                candidate = Legend(owner, handles, labels, **options)
                if not below:
                    candidate._loc = legend._loc
                    anchor = getattr(legend, "_bbox_to_anchor", None)
                    if anchor is not None:
                        candidate._bbox_to_anchor = anchor
                frame, old_frame = candidate.get_frame(), legend.get_frame()
                frame.set_alpha(old_frame.get_alpha())
                frame.set_facecolor(old_frame.get_facecolor())
                frame.set_edgecolor(old_frame.get_edgecolor())
                measured = candidate.get_window_extent(renderer)
            except Exception:
                continue
            excess = max(measured.width - max_w, 0.0) + max(measured.height - max_h, 0.0)
            if best is None or excess < best_excess - 0.5:
                best, best_excess = candidate, excess
            if excess <= 0:
                break
        if best is not None and best_excess <= 0:
            break
    if best is None:
        return False
    _place_legend(owner, legend, best)

    def _undo() -> None:
        if is_axes:
            if owner.legend_ is best:
                owner.legend_ = legend
        elif best in owner.legends:
            owner.legends[owner.legends.index(best)] = legend
        owner.stale = True

    undo.append(_undo)
    return True


def _refit_legends(region, renderer, pad: float, below: bool, undo: list) -> bool:
    """Refit the legends that are part of the problem: one that reaches
    outside the pane, or one that sits outside its axes and so takes from
    the margins the room the axes need. A legend inside its axes, inside
    the pane, is left as the plot drew it."""
    bounds = region.bbox
    changed = False
    for owner, legend in _legends(region):
        is_axes = hasattr(owner, "legend_")
        try:
            box = legend.get_window_extent(renderer)
        except Exception:
            continue
        frame = owner.bbox if is_axes else None
        inside_axes = frame is not None and (frame.x0 - 2 <= box.x0 and box.x1 <= frame.x1 + 2
                                             and frame.y0 - 2 <= box.y0 and box.y1 <= frame.y1 + 2)
        if below and (not is_axes or inside_axes):
            continue
        if not below and inside_axes and _sides(box, bounds) is None:
            continue
        changed = _refit_legend(region, owner, legend, renderer, pad, below, undo) or changed
    return changed


def _shrink_type(region, scale: float, undo: list) -> bool:
    """Set every text of the region smaller by `scale`, never below the
    floor (CONTAIN_FONT_FLOOR; CONTAIN_TICK_FONT_FLOOR for tick labels)."""
    from matplotlib.axes import Axes
    from matplotlib.text import Text

    changed = False

    def smaller(text, floor: float = CONTAIN_FONT_FLOOR) -> None:
        nonlocal changed
        if text is None or not text.get_visible() or not text.get_text().strip():
            return
        if text.get_gid() in (ENLARGE_GID, PANEL_LABEL_GID):
            return
        old = float(text.get_fontsize())
        new = min(old, max(floor, old * scale))
        if new >= old - 1e-9:
            return
        text.set_fontsize(new)

        def _undo() -> None:
            if abs(float(text.get_fontsize()) - new) < 1e-9:
                text.set_fontsize(old)

        undo.append(_undo)
        changed = True

    def walk(ax) -> None:
        nonlocal changed
        if not ax.get_visible():
            return
        for title in (ax.title, ax._left_title, ax._right_title):
            smaller(title)
        fits = {info.get("axis"): info for info in (getattr(ax, FIT_ATTR, None) or [])}
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            smaller(axis.label)
            smaller(axis.offsetText, CONTAIN_TICK_FONT_FLOOR)
            info = fits.get(name)
            # Major and minor labels each keep their own size (a log axis
            # shorter than a decade labels its minor ticks, often smaller).
            for which in ("major", "minor"):
                try:
                    shown = [t for t in axis.get_ticklabels(minor=(which == "minor")) if t.get_text()]
                except Exception:
                    shown = []
                if not shown:
                    continue
                old = float(shown[0].get_fontsize())
                new = min(old, max(CONTAIN_TICK_FONT_FLOOR, old * scale))
                if new >= old - 1e-9:
                    continue
                if info is not None and which == "major":
                    previous = info.get("font_cap")
                    info["font_cap"] = new
                    _fit_one(ax, info)

                    def _undo(info=info, previous=previous) -> None:
                        if previous:
                            info["font_cap"] = previous
                        else:
                            info.pop("font_cap", None)
                else:
                    axis.set_tick_params(which=which, labelsize=new)

                    def _undo(axis=axis, which=which, old=old) -> None:
                        axis.set_tick_params(which=which, labelsize=old)

                undo.append(_undo)
                changed = True
        for child in ax.get_children():
            if isinstance(child, Axes):
                walk(child)
            elif isinstance(child, Text) and child not in (ax.title, ax._left_title, ax._right_title):
                smaller(child)
        legend = ax.get_legend()
        if legend is not None:
            for text in list(legend.get_texts()) + [legend.get_title()]:
                smaller(text)

    for text in list(region.texts):
        smaller(text)
    for legend in list(region.legends):
        for text in list(legend.get_texts()) + [legend.get_title()]:
            smaller(text)
    for ax in _own_axes(region):
        walk(ax)
    return changed


def _may_show_enlarge_message(region) -> bool:
    """A pane small in at least one direction may say so instead of
    showing a chart it has no room for. Never while a figure is being
    saved: a file holds the chart, laid out as well as its size allows."""
    root = _root_figure(region)
    if getattr(root, EXPORTING_ATTR, False):
        return False
    box, dpi = region.bbox, float(root.dpi)
    return bool(box.width / dpi <= ENLARGE_BELOW_INCHES[0] + 1e-6 or box.height / dpi <= ENLARGE_BELOW_INCHES[1] + 1e-6)


def _plot_area_too_small(region) -> bool:
    """True when the largest plotting area of the region is too small to
    be read as a chart (everything fits, but only because the axes were
    squeezed to a sliver between the title and the labels), or the pane
    is smaller than the plot declared it needs (declare_minimum_size)."""
    dpi = float(_root_figure(region).dpi)
    largest = None
    pane = (float(region.bbox.width) / dpi, float(region.bbox.height) / dpi)
    for ax in _own_axes(region):
        declared = getattr(ax, MIN_SIZE_ATTR, None)
        if declared and ax.get_visible() and (pane[0] < declared[0] or pane[1] < declared[1]):
            return True                           # smaller than the plot says it can be read in
        if not ax.get_visible() or ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar"):
            continue
        if not (ax.axison or ax.collections or ax.lines or ax.images or ax.patches):
            continue                              # a sentence, not a chart
        box = ax.bbox
        if largest is None or box.width * box.height > largest[0] * largest[1]:
            largest = (float(box.width), float(box.height))
    if largest is None:
        return False
    return bool(largest[0] / dpi < CONTAIN_MIN_PLOT_INCHES[0] or largest[1] / dpi < CONTAIN_MIN_PLOT_INCHES[1])


def _show_enlarge_message(region, undo: list) -> None:
    """Replace a chart the pane is too small for by one calm line. The
    chart is hidden, not removed: it comes back when the pane grows."""
    hidden = []
    candidates = list(_own_axes(region)) + list(region.legends) + list(region.artists)
    candidates += [t for t in region.texts if t.get_gid() != PANEL_LABEL_GID]
    for artist in candidates:
        if artist.get_visible():
            artist.set_visible(False)
            hidden.append(artist)
    note = region.text(0.5, 0.5, ENLARGE_MESSAGE, ha="center", va="center", fontsize=9.5,
                       color=CHART_SECONDARY, wrap=True, gid=ENLARGE_GID)

    def _undo() -> None:
        try:
            note.remove()
        except Exception:
            note.set_visible(False)
        for artist in hidden:
            artist.set_visible(True)

    undo.append(_undo)


def contain_region(region, renderer=None, refit=None) -> List[Dict[str, object]]:
    """Make sure nothing `region` (a pane's figure, or one cell of a
    composed figure) draws is cut by its edge. Nothing is touched when
    nothing overflows (and, in a small pane, the chart still has a
    plotting area; and no legend is larger than the axes it sits in).
    `refit` re-runs the plot's own layout (its tight_layout, its
    registered callback) after a correction changed what there is to lay
    out. Returns what still overflows: an empty list, unless the pane is
    larger than ENLARGE_BELOW_INCHES and still too small for its chart."""
    region = _plain_figure(region)
    renderer = _layout_renderer(region, renderer)
    state = _contain_state(region, _own_axes(region))
    undo: list = state["undo"]
    dpi = float(_root_figure(region).dpi)
    size = (float(region.bbox.width) / dpi, float(region.bbox.height) / dpi)
    wording = _contain_key(region)[4]
    # A pane that was too small for this chart is too small at any size
    # within that one: no need to go through every correction again while
    # a small pane is being dragged.
    small = _may_show_enlarge_message(region)
    too_small = state.get("too_small")
    if (small and too_small and too_small[2] == wording
            and size[0] <= too_small[0] + 0.005 and size[1] <= too_small[1] + 0.005):
        _show_enlarge_message(region, undo)
        return _region_overflow(region, renderer)

    def crushed() -> bool:
        return small and _plot_area_too_small(region)

    notes: Dict[str, object] = {}
    report = _region_overflow(region, renderer, notes)
    if not report and not crushed() and not notes.get("spill"):
        return report
    pad = CONTAIN_PAD_IN * dpi

    def settle(changed: bool = True, bands: bool = True) -> bool:
        """Lay out again after a correction (the plot's own layout, then
        the margins by the amount of the overflow); True when the chart
        is now whole: nothing outside the pane, a plotting area left, and
        no legend larger than the axes it sits in. `bands`: the heading
        or a legend of the figure itself changed size, so the axes may
        have to give it room even with nothing outside the pane."""
        nonlocal report
        if changed:
            if callable(refit):
                refit()
            notes.clear()
            report = _region_overflow(region, renderer, notes)
        if bands or any(entry["attached"] for entry in report):
            before = _margins(region)
            _fit_margins(region, renderer, pad)
            if _margins(region) != before:        # otherwise what was measured still stands
                notes.clear()
                report = _region_overflow(region, renderer, notes)
        return not report and not crushed() and not notes.get("spill")

    if report:
        # 1. A text longer than the pane: no margin can make room for it.
        wrapped, own = _fit_long_texts(region, renderer, pad, undo, report)
        # 2. The margins, by the amount of the overflow.
        if settle(wrapped, bands=own):
            return report
        # A title that the moved axes pushed against an edge.
        if report:
            wrapped, own = _fit_long_texts(region, renderer, pad, undo, report)
            if wrapped and settle(True, bands=own):
                return report
    # 3. Long names along a category axis, when the margins could not
    #    make room for what hangs on the axes.
    for share in (0.38, 0.26):
        if ((crushed() or any(entry["attached"] for entry in report))
                and _elide_tick_labels(region, renderer, share, undo) and settle(True, bands=False)):
            return report
    # 4. Legends: compact where they are, then under their axes. (Also
    #    for a legend larger than the axes it was placed in, which is
    #    inside the pane but on top of the title or the labels.)
    for below in (False, True):
        if _refit_legends(region, renderer, pad, below, undo) and settle():
            return report
    if not report and not crushed():
        return report                             # only a legend was the matter
    # 5. Smaller type everywhere (in one step where the pane is small
    #    enough for the one-line message to be the next thing tried).
    for scale in ((0.6,) if small else (0.8, 0.6)):
        if _shrink_type(region, scale, undo):
            _fit_long_texts(region, renderer, pad, undo)
            if settle():
                return report
    # 6. A pane too small for this chart.
    if small:
        known = state.get("too_small")
        if not known or known[2] != wording or (size[0] >= known[0] and size[1] >= known[1]):
            state["too_small"] = (size[0], size[1], wording)
        _show_enlarge_message(region, undo)
        report = _region_overflow(region, renderer)
    return report


def _cell_layout(subfig, renderer, state: Dict[str, object]) -> None:
    _restore_margins(subfig, state)
    axes = list(subfig.axes)
    for ax in axes:
        fit_categories(ax)
    for ax in axes:
        callback = getattr(ax, RELAYOUT_ATTR, None)
        if callable(callback):
            callback()
    tighten_subfigure(subfig, renderer)
    for ax in axes:
        fit_categories(ax)                        # the margins moved: the room changed


def contain_cell(subfig, renderer=None, force: bool = False) -> None:
    """Lay out one cell of a composed figure for the size it has, and keep
    everything it draws inside it. Called at every draw: a cell that was
    laid out with this wording, and overflows nowhere, is left alone
    (`force` lays it out regardless: after a resize has settled)."""
    renderer = _layout_renderer(subfig, renderer)
    _own_subplotpars(subfig)
    state = _contain_state(subfig, list(subfig.axes), base=_default_margins(), spacing=_default_spacing())
    if not force and _same_content(state, subfig) and not _region_overflow(subfig, renderer):
        return
    try:
        release_containment(subfig)
        _cell_layout(subfig, renderer, state)
        contain_region(subfig, renderer, lambda: _cell_layout(subfig, renderer, state))
    except Exception:
        pass
    state["key"] = _contain_key(subfig)


def contain_on_draw(figure, renderer) -> None:
    """What a pane's figure does before it draws: lay itself out when no
    layout was made for this content (a plot nobody finished), or when
    something overflows at the size it is drawn at (a resize still being
    dragged, an export, a pan, a zoom). Otherwise only a measurement."""
    if not CONTAINMENT:
        if getattr(figure, "_ember_tight_pending", False):
            figure.apply_tight_layout()
        for sub in getattr(figure, "subfigs", None) or []:
            tighten_subfigure(sub, renderer)
        return
    if getattr(figure, "subfigs", None):
        for sub in figure.subfigs:
            contain_cell(sub, renderer)
        return
    if not getattr(figure, "axes", None):
        return
    state = getattr(figure, CONTAIN_ATTR, None)
    if (not getattr(figure, "_ember_tight_pending", False) and _same_content(state, figure)
            and not _region_overflow(figure, renderer)):
        return
    relayout_figure(figure, renderer)


def _same_content(state, region) -> bool:
    """True when `region` was laid out with the axes and the wording it
    has now, at whatever size. At another size the layout is not redone
    for a draw unless something overflows: the widget lays the figure out
    once the resize has settled, and an export around the save."""
    if not isinstance(state, dict) or not state.get("key"):
        return False
    return state["key"][3:] == _contain_key(region)[3:]


def humanize_figure_text(figure) -> None:
    """Replace column identifiers that reached a title, an axis label or a
    legend title with their labels ("Inventory_sum" -> "Inventory")."""
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        fixed = humanize_text(suptitle.get_text())
        if fixed != suptitle.get_text():
            suptitle.set_text(fixed)
    for ax in list(getattr(figure, "axes", None) or []):
        for getter, setter in ((ax.get_title, ax.set_title), (ax.get_xlabel, ax.set_xlabel),
                               (ax.get_ylabel, ax.set_ylabel)):
            text = getter()
            if text:
                fixed = humanize_text(text)
                if fixed != text:
                    setter(fixed)
        legend = ax.get_legend()
        if legend is not None:
            title = legend.get_title()
            if title is not None and title.get_text():
                fixed = humanize(title.get_text())
                if fixed != title.get_text():
                    title.set_text(fixed)


def finish_panel(panel) -> None:
    """What every plot gets once its function has drawn it: readable text,
    labels fitted to the panel's size, and a word to the panel that its
    content changed (so it can re-index its marks and repaint)."""
    figure = getattr(panel, "figure", None)
    if figure is not None and getattr(figure, "axes", None):
        humanize_figure_text(figure)
        relayout_figure(figure)
    hook = getattr(panel, "_after_render", None)
    if callable(hook):
        hook()


def composable(fn):
    """Marks a plot function as reusable by Figure Composer. After a
    render, the panel is finished (finish_panel) and remembers this call
    -- if it has a _set_last_recipe method, which qt_widgets.PlotWidget
    does and the lightweight FakePanel test doubles don't -- so a later
    "Send to Figure Composer" can replay it into a combined figure's own
    subfigure, instead of capturing a raster snapshot. The recipe is the
    WRAPPED function, so a replay is finished the same way as the
    original: same labels, same fitted axes."""
    @functools.wraps(fn)
    def wrapper(panel, *args, **kwargs):
        fn(panel, *args, **kwargs)
        finish_panel(panel)
        set_recipe = getattr(panel, "_set_last_recipe", None)
        if set_recipe is not None:
            set_recipe(wrapper, args, kwargs)
    return wrapper

# seaborn is imported lazily (on first actual use, via _ensure_seaborn()
# below) rather than at module scope: this module is imported eagerly at
# app startup by nearly every workspace, and seaborn alone costs ~2s of
# cold-import time -- most of it ipywidgets/IPython, pulled in for Jupyter
# widget support this desktop app never touches. Most sessions never select
# a seaborn plot style, so most sessions shouldn't pay that cost at launch.
_SEABORN_UNSET = object()
sns = _SEABORN_UNSET


def _ensure_seaborn() -> None:
    global sns
    if sns is _SEABORN_UNSET:
        try:
            import seaborn as _seaborn_module
            sns = _seaborn_module
        except Exception:  # pragma: no cover - exercised via a dedicated sns=None test
            sns = None


def numeric_plot_series(series: pd.Series) -> pd.Series:
    """Numeric conversion for plotting without mangling non-numeric columns."""
    out = pd.to_numeric(series, errors="coerce")
    return out.where(np.isfinite(out), np.nan)


def positive_value_frame(
    data: pd.DataFrame, label_col: str, value_col: str, value_name: str = "_plot_value",
) -> pd.DataFrame:
    """The shared front half of every inventory bar chart: coerce the value
    column to numbers, drop rows with no label or no value, and keep only
    strictly positive values (log axes and bar lengths are meaningless
    otherwise). Shared by the legacy renderers and the plot grammar so the
    filtering policy can't drift between the two."""
    pdf = data.copy()
    pdf[value_name] = numeric_plot_series(pdf[value_col])
    pdf = pdf.dropna(subset=[label_col, value_name])
    return pdf[pdf[value_name] > 0]


def bar_figure_height(
    n_bars: int, per_bar: float = 0.26, floor: float = 4.5, ceiling: float = 16.0,
) -> float:
    """Figure height for a horizontal bar chart, scaled by bar count and
    clamped both ways. A fixed height crams every label into the same
    vertical space regardless of bar count (which is what made 40-bar
    charts illegible); an unclamped one grows past the visible panel,
    which has no scroll area. Callers pass it to
    PlotWidget.set_figure_size_inches, whose own cap enforces the
    viewport ceiling."""
    return min(max(floor, n_bars * per_bar), ceiling)


def marginal_heatmap_kwargs() -> Dict[str, object]:
    """Heatmap kwargs for a matrix that carries marginal bar charts.

    `square=True` makes matplotlib shrink the heatmap's drawn box inside
    its gridspec cell to keep cells square -- which leaves the marginal
    bars wider (or taller) than the matrix they annotate, because those
    axes keep the whole cell. Letting the matrix fill its cell instead
    makes alignment structural: the marginals share the same gridspec
    column/row and the same data limits, so they cannot drift, and no
    post-draw correction (which would have to re-run on every resize) is
    needed. The gridspec ratios already keep cells close to square.
    """
    return {"square": False}


def heatmap_ytick_step(n_rows: int, dense_limit: int = 80, target: int = 40) -> int:
    """Row-label thinning for tall heatmaps: label every row while they
    still fit, then every Nth so labels stay readable."""
    return 1 if n_rows <= dense_limit else max(1, n_rows // target)


def color_by_highlight(
    labels: Sequence[str], highlighted: Optional[str],
    base_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """Bar colors, highlighting the requested element/analyte when possible."""
    if not highlighted:
        return [BASE_COLOR for _ in labels]
    highlighted = str(highlighted)
    if base_labels is None:
        base_labels = labels
    colors: List[str] = []
    for label, base in zip(labels, base_labels):
        label_s, base_s = str(label), str(base)
        is_match = base_s == highlighted or label_s == highlighted or label_s.startswith(f"{highlighted} [")
        colors.append(ACCENT_COLOR if is_match else BASE_COLOR)
    return colors


def make_unique_plot_labels(pdf: pd.DataFrame, label_col: str, unit_col: str = "Units") -> pd.Series:
    """Build unique y-axis labels. Matplotlib categorical axes collapse
    duplicated labels (e.g. "Sr" appearing once for Ci rows and once for kg
    rows) -- append the unit whenever that would happen, and fall back to a
    disambiguating suffix if labels are still duplicated after that."""
    base = pdf[label_col].astype(str).fillna("NA")
    labels = base.copy()
    has_units = unit_col in pdf.columns
    duplicated_base = bool(base.duplicated(keep=False).any())
    multiple_units = bool(has_units and pdf[unit_col].dropna().astype(str).nunique() > 1)
    if has_units and (duplicated_base or multiple_units):
        labels = base + " [" + pdf[unit_col].astype(str).fillna("?") + "]"

    if labels.duplicated(keep=False).any():
        counts: Dict[str, int] = {}
        fixed: List[str] = []
        for idx, label in labels.items():
            label_s = str(label)
            counts[label_s] = counts.get(label_s, 0) + 1
            if counts[label_s] == 1:
                fixed.append(label_s)
                continue
            suffix = str(counts[label_s])
            for col in ("Analyte", "WasteSiteId", "WastePhase", "WasteType"):
                if col in pdf.columns and col != label_col:
                    val = pdf.at[idx, col]
                    if pd.notna(val):
                        suffix = str(val)
                        break
            fixed.append(f"{label_s} ({suffix})")
        labels = pd.Series(fixed, index=pdf.index)
    return labels


def category_barh(ax, labels: Sequence[object], values, *, name: str = "", value_label: str = "",
                  unit: str = "", tank_ids: Optional[Sequence[object]] = None, **kwargs):
    """Horizontal bars for a ranking: one bar per label, at positions
    0..n-1, with the category axis declared (fitted labels, tooltips)."""
    labels = [str(v) for v in labels]
    bars = ax.barh(np.arange(len(labels)), values, **kwargs)
    if hasattr(ax, "set_ylim") and labels:
        ax.set_ylim(-0.6, len(labels) - 0.4)
    declare_categories(ax, labels, axis="y", name=name, value_label=value_label, unit=unit, tank_ids=tank_ids)
    return bars


def bare_frame(ax) -> None:
    """A matrix needs no axis lines: the cells are the frame."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", length=0)


@composable
def plot_barh(
    panel, data: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str,
    top_n: int = 40, highlighted_label: Optional[str] = None, log_x: bool = False,
) -> None:
    if data is None or data.empty or label_col not in data or value_col not in data:
        panel.show_message("No data to plot")
        return

    pdf = positive_value_frame(data, label_col, value_col)
    if pdf.empty:
        panel.show_message("No positive values to plot")
        return

    pdf = pdf.sort_values("_plot_value", ascending=False).head(int(top_n)).copy()
    pdf["_plot_label"] = make_unique_plot_labels(pdf, label_col)
    pdf = pdf.iloc[::-1]  # reverse after sort/head so the largest value is at the top of barh

    ax = panel.reset_axes()
    width = panel.figure.get_size_inches()[0]
    panel.set_figure_size_inches(width, bar_figure_height(len(pdf)))

    colors = color_by_highlight(
        pdf["_plot_label"].astype(str).tolist(), highlighted_label,
        base_labels=pdf[label_col].astype(str).tolist(),
    )
    category_barh(
        ax, pdf["_plot_label"].astype(str).tolist(), pdf["_plot_value"].to_numpy(dtype=float), color=colors,
        name=humanize(label_col), value_label=humanize_text(xlabel),
    )
    ax.set_xlabel(humanize_text(xlabel))
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, axis="x")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_heatmap(panel, wide: pd.DataFrame, unit: str, mode: str, title: str) -> None:
    """Tank x element inventory heatmap. Rebuilds the panel's axes (rather
    than clearing in place) because figure size must scale with the number
    of tanks/elements to stay readable -- up to 177 tanks in this dataset."""
    if wide is None or wide.empty or "WasteSiteId" not in wide:
        panel.show_message("No heatmap data")
        return
    data = wide.copy()
    labels_y = data["WasteSiteId"].astype(str).tolist()
    elements = [c for c in data.columns if c != "WasteSiteId"]
    arr = data[elements].to_numpy(dtype=float)
    if mode == "log10_inventory":
        arr = log10_safe(arr)
        cbar_label = f"log10 inventory ({unit})"
    elif mode == "fraction":
        denom = np.nansum(arr, axis=1, keepdims=True)
        arr = np.divide(arr, denom, where=denom > 0)
        cbar_label = f"fraction of displayed {unit} inventory"
    else:
        cbar_label = f"inventory ({unit})"

    panel.figure.clear()
    height = min(max(5, len(labels_y) * 0.10), 18)
    width = min(max(8, len(elements) * 0.25), 18)
    panel.set_figure_size_inches(width, height)
    ax = panel.figure.add_subplot(111)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=CMAP_SEQUENTIAL)
    bare_frame(ax)
    mark_cells(ax, labels_y, elements, row_name="Tank", col_name="Element", value_label=cbar_label)
    ax.set_xticks(np.arange(len(elements)))
    ax.set_xticklabels(elements, rotation=90)
    yticks = np.arange(0, len(labels_y), heatmap_ytick_step(len(labels_y)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([labels_y[i] for i in yticks], fontsize=7)
    ax.set_xlabel("Element")
    ax.set_ylabel("Tank")
    ax.set_title(title)
    cbar = panel.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)
    panel.ax = ax
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_grouped_tank_profile(panel, data: pd.DataFrame, value_col: str, title: str, top_n: int = 25) -> None:
    """Grouped horizontal bars: one cluster per element, one series per
    tank -- for comparing composition across several selected tanks at once."""
    if data is None or data.empty or value_col not in data or "Element" not in data:
        panel.show_message("No tank profile data")
        return
    pdf = data.copy()
    pdf["_plot_value"] = numeric_plot_series(pdf[value_col])
    pdf = pdf.dropna(subset=["Element", "_plot_value"])
    pdf = pdf[pdf["_plot_value"] > 0]
    if pdf.empty:
        panel.show_message("No positive data to plot")
        return

    # One cluster per ELEMENT: the same element in two tanks is the same
    # category (that is the comparison this chart exists for). The label
    # carries the unit only when the frame mixes units -- kg and Ci of one
    # element are different quantities. Making every (element, tank) row
    # its own category, as a per-row unique label does, turned a
    # three-tank comparison into three separate rankings stacked on one
    # axis.
    elements_text = pdf["Element"].astype(str)
    if "Units" in pdf.columns and pdf["Units"].dropna().astype(str).nunique() > 1:
        pdf["_plot_element"] = elements_text + " [" + pdf["Units"].astype(str) + "]"
    else:
        pdf["_plot_element"] = elements_text

    top_labels = (
        pdf.groupby("_plot_element", as_index=True)["_plot_value"]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_n))
        .index
        .tolist()
    )
    pdf = pdf[pdf["_plot_element"].isin(top_labels)]
    pivot = pdf.pivot_table(index="_plot_element", columns="WasteSiteId", values="_plot_value", aggfunc="sum", fill_value=0.0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    ax = panel.reset_axes()
    pivot.plot(kind="barh", ax=ax, width=0.85, color=series_colors(len(pivot.columns)))
    single_unit = "Units" not in pdf or pdf["Units"].nunique() <= 1
    unit = str(pdf["Units"].iloc[0]) if ("Units" in pdf and single_unit and len(pdf)) else ""
    value_label = humanize(value_col) + (f" ({unit})" if unit and "(" not in humanize(value_col) else "")
    declare_categories(ax, [str(v) for v in pivot.index], axis="y", name="Element",
                       value_label=value_label, unit=unit, series_name="Tank")
    ax.set_xlabel(value_label)
    ax.set_ylabel("Element" if single_unit else "Element [unit]")
    ax.set_title(title)
    ax.grid(True, axis="x")
    ax.legend(title="Tank", fontsize=8)
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_correlation_scan(panel, data: pd.DataFrame, target: str, top_n: int = 30) -> None:
    if data is None or data.empty or "PartnerElement" not in data or "Correlation_r" not in data:
        panel.show_message("No correlation scan data")
        return
    pdf = data.copy().dropna(subset=["Correlation_r"])
    pdf = pdf.sort_values("AbsCorrelation", ascending=False).head(int(top_n)).copy()
    if pdf.empty:
        panel.show_message("No valid correlations")
        return
    pdf = pdf.iloc[::-1]
    ax = panel.reset_axes()
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in pdf["Correlation_r"]]
    labels = pdf["PartnerElement"].astype(str) + " (n=" + pdf["N_overlap_nonzero_tanks"].astype(str) + ")"
    category_barh(ax, labels.tolist(), pdf["Correlation_r"].astype(float).to_numpy(), color=colors,
                  name="Partner element", value_label="Correlation r")
    ax.axvline(0, color=CHART_AXIS, linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    unit = str(pdf["Units"].iloc[0]) if "Units" in pdf else ""
    metric = str(pdf["Metric"].iloc[0]) if "Metric" in pdf else ""
    ax.set_xlabel(f"Correlation coefficient r ({humanize(metric)}, {unit})")
    ax.set_ylabel("Partner element")
    ax.set_title(f"Top correlations with {target}")
    ax.grid(True, axis="x")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


def _correlation_square_from_dataframe(corr: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Numeric square correlation matrix indexed/columned by Element,
    dropping any accidental metadata columns that don't match the index."""
    if corr is None or corr.empty or "Element" not in corr:
        return pd.DataFrame(), []
    data = corr.copy().set_index("Element")
    elements = [str(e) for e in data.index.tolist()]
    keep = [e for e in elements if e in data.columns]
    data = data.loc[keep, keep]
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data, keep


def _aligned_projection_values(elements: Sequence[str], totals: Optional[Dict[str, float]]) -> np.ndarray:
    if not totals:
        return np.zeros(len(elements), dtype=float)
    vals = np.array([float(totals.get(str(e), 0.0) or 0.0) for e in elements], dtype=float)
    vals[~np.isfinite(vals)] = 0.0
    return vals


def seaborn_available() -> bool:
    _ensure_seaborn()
    return sns is not None


@composable
def plot_correlation_heatmap(
    panel, corr: pd.DataFrame, title: str = "Element correlation heatmap",
    style: str = "Matplotlib lower triangle", totals: Optional[Dict[str, float]] = None,
    unit: str = "", annotate: bool = False,
) -> None:
    """Lower-triangle-only correlation heatmap (the matrix is symmetric, so
    the upper triangle is pure redundancy). Three styles:
        Matplotlib lower triangle    no seaborn dependency
        Seaborn lower triangle       nicer labels/grid/colorbar
        Seaborn + total projections  + marginal bars of log10(total+1)
    """
    data, elements = _correlation_square_from_dataframe(corr)
    if data.empty:
        panel.show_message("No correlation matrix")
        return

    n = len(elements)
    arr = data.to_numpy(dtype=float)
    upper_mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
    style_norm = (style or "").lower()
    _ensure_seaborn()
    use_seaborn = ("seaborn" in style_norm) and (sns is not None)
    with_projection = ("projection" in style_norm or "total" in style_norm)

    if "seaborn" in style_norm and sns is None:
        title = title + " | seaborn not installed, using Matplotlib"
        use_seaborn = False
        with_projection = False

    panel.figure.clear()
    base_size = panel.cap_square_size_inches(min(max(6.5, n * 0.34), 19.0))

    if use_seaborn and with_projection:
        proj_raw = _aligned_projection_values(elements, totals)
        proj = np.log10(proj_raw + 1.0)
        panel.set_figure_size_inches(base_size + 3.2, base_size + 1.8)
        # The colorbar gets its OWN column. Left to seaborn it is carved
        # out of the heatmap's axes, which narrows the matrix while the
        # bar chart above keeps the full column -- the bars then overhang
        # the columns they label.
        gs = panel.figure.add_gridspec(
            2, 4, width_ratios=[base_size, 0.30, 1.05, 2.2],
            height_ratios=[1.4, base_size], wspace=0.05, hspace=0.05,
        )
        ax = panel.figure.add_subplot(gs[1, 0])
        ax_cbar = panel.figure.add_subplot(gs[1, 1])
        # Share both axes with the heatmap so the marginals track its data
        # limits, and let the matrix fill its gridspec cell (see
        # marginal_heatmap_kwargs) so the boxes line up structurally.
        ax_top = panel.figure.add_subplot(gs[0, 0], sharex=ax)
        ax_right = panel.figure.add_subplot(gs[1, 3], sharey=ax)

        x = np.arange(n) + 0.5   # seaborn's cell centres
        ax_top.bar(x, proj, width=0.8, color=CHART_NEUTRAL)
        mark_categories(ax_top, elements, axis="x", name="Element")
        ax_top.set_xlim(0, n)
        ax_top.set_ylabel(f"log10(total {unit} + 1)" if unit else "log10(total + 1)", fontsize=8)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_top.grid(True, axis="y")
        for spine in ("top", "right"):
            ax_top.spines[spine].set_visible(False)

        sns.heatmap(
            data, mask=upper_mask, vmin=-1, vmax=1, cmap=CMAP_DIVERGING, center=0,
            **marginal_heatmap_kwargs(),
            linewidths=0.4 if n <= 45 else 0.0, linecolor="white",
            annot=bool(annotate and n <= 25), fmt=".2f",
            cbar_ax=ax_cbar, cbar_kws={"label": "Correlation r"}, ax=ax,
        )
        # suptitle, not ax.set_title(): ax sits directly below ax_top with
        # hspace=0.05 (no room reserved between rows for a title), so an
        # axes-level title renders on top of ax_top's bars instead of above
        # them. suptitle sits above the whole figure, clear of ax_top.
        panel.figure.suptitle(title)
        ax.set_xlabel("Element")
        ax.set_ylabel("Element")
        ax.tick_params(axis="x", rotation=90, labelsize=8 if n <= 55 else 6)
        ax.tick_params(axis="y", rotation=0, labelsize=8 if n <= 55 else 6)

        y_centers = np.arange(n) + 0.5
        ax_right.barh(y_centers, proj, height=0.8, color=CHART_NEUTRAL)
        mark_categories(ax_right, elements, axis="y", name="Element")
        ax_right.set_xlabel(f"log10(total {unit} + 1)" if unit else "log10(total + 1)", fontsize=8)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        ax_right.grid(True, axis="x")
        for spine in ("top", "right"):
            ax_right.spines[spine].set_visible(False)
        declare_minimum_size(ax, 4.0, 3.0)        # a matrix, two bar charts and a colour bar
        panel.ax = ax
    elif use_seaborn:
        panel.set_figure_size_inches(base_size, base_size)
        ax = panel.figure.add_subplot(111)
        sns.heatmap(
            data, mask=upper_mask, vmin=-1, vmax=1, cmap=CMAP_DIVERGING, center=0, square=True,
            linewidths=0.4 if n <= 45 else 0.0, linecolor="white",
            annot=bool(annotate and n <= 25), fmt=".2f",
            cbar_kws={"label": "Correlation r", "shrink": 0.8}, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Element")
        ax.set_ylabel("Element")
        ax.tick_params(axis="x", rotation=90, labelsize=8 if n <= 55 else 6)
        ax.tick_params(axis="y", rotation=0, labelsize=8 if n <= 55 else 6)
        panel.ax = ax
    else:
        panel.set_figure_size_inches(base_size, base_size)
        ax = panel.figure.add_subplot(111)
        masked = np.ma.array(arr, mask=upper_mask)
        cmap = matplotlib.colormaps.get_cmap(CMAP_DIVERGING).copy()
        cmap.set_bad(color="white")
        im = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, interpolation="nearest")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(elements, rotation=90, fontsize=8 if n <= 55 else 6)
        ax.set_yticklabels(elements, fontsize=8 if n <= 55 else 6)
        ax.set_title(title)
        ax.set_xlabel("Element")
        ax.set_ylabel("Element")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        if n <= 50:
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = panel.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation r")
        if annotate and n <= 25:
            for i in range(n):
                for j in range(i + 1):
                    val = arr[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
        bare_frame(ax)
        panel.ax = ax

    mark_cells(panel.ax, elements, elements, row_name="Element", col_name="Element", value_label="Correlation r",
               triangle="lower")
    if not (use_seaborn and with_projection):
        # One matrix and its colour bar: fitted margins keep the bar and
        # its label inside the pane. (With marginal bars the gridspec is
        # the layout; fitted margins would pull the bars off the matrix.)
        panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_pair_scatter(panel, matrix: pd.DataFrame, elements: Sequence[str], unit: str, metric: str) -> None:
    clean = [normalize_element_symbol(e) for e in elements if normalize_element_symbol(e)]
    clean = [e for e in clean if e in matrix.columns]
    if matrix is None or matrix.empty or len(clean) < 2:
        panel.show_message("Need at least two selected elements with data")
        return
    a, b = clean[0], clean[1]
    pdf = matrix[["WasteSiteId"] + clean].copy()
    pdf[a] = numeric_plot_series(pdf[a])
    pdf[b] = numeric_plot_series(pdf[b])
    pdf = pdf.dropna(subset=[a, b])
    ax = panel.reset_axes()
    if len(clean) >= 3:
        c = clean[2]
        pdf[c] = numeric_plot_series(pdf[c]).fillna(0.0)
        sc = ax.scatter(pdf[a], pdf[b], c=pdf[c], s=45, alpha=0.9, cmap=CMAP_SEQUENTIAL,
                        edgecolors=CHART_AXIS, linewidths=0.5)
        cbar = panel.figure.colorbar(sc, ax=ax)
        cbar.set_label(f"{c} ({humanize(metric)}, {unit})")
        title = f"{a} vs {b}, colored by {c}"
    else:
        sc = ax.scatter(pdf[a], pdf[b], s=45, alpha=0.8, color=BASE_COLOR, edgecolors="white", linewidths=0.5)
        title = f"{a} vs {b}"
    tanks = pdf["WasteSiteId"].astype(str).tolist()
    mark_points(ax, sc, tanks, name="Tank", tank_ids=tanks)
    r = pdf[a].corr(pdf[b]) if len(pdf) >= 3 else np.nan
    ax.set_xlabel(f"{a} ({humanize(metric)}, {unit})")
    ax.set_ylabel(f"{b} ({humanize(metric)}, {unit})")
    ax.set_title(f"{title} | r = {r:.3f}" if pd.notna(r) else title)
    ax.grid(True)
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_target_vs_total(panel, data: pd.DataFrame, title: str) -> None:
    required = ["Context_TotalInventory_same_unit", "Target_Inventory_sum", "Units"]
    if data is None or data.empty or any(c not in data for c in required):
        panel.show_message("No target-vs-total data to plot")
        return
    pdf = data.copy()
    pdf["_x_total"] = numeric_plot_series(pdf["Context_TotalInventory_same_unit"])
    pdf["_y_target"] = numeric_plot_series(pdf["Target_Inventory_sum"])
    pdf = pdf.dropna(subset=["_x_total", "_y_target"])
    pdf = pdf[(pdf["_x_total"] > 0) & (pdf["_y_target"] > 0)]
    if pdf.empty:
        panel.show_message("No positive values to plot")
        return
    ax = panel.reset_axes()
    for unit in pd.unique(pdf["Units"]):
        sub = pdf[pdf["Units"] == unit]
        sc = ax.scatter(sub["_x_total"], sub["_y_target"], label=str(unit), alpha=0.8, s=35,
                        edgecolors="white", linewidths=0.5)
        if "WasteSiteId" in sub.columns:
            tanks = sub["WasteSiteId"].astype(str).tolist()
            mark_points(ax, sc, tanks, name="Tank", tank_ids=tanks, series_name="Unit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total tank inventory in same unit")
    ax.set_ylabel("Target inventory in tank")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(title="Unit")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


# ---------------------------------------------------------------------------
# kg Association Workbench (M7): coherent-colors palette + 18 plot types
# (8 functions, several parameterized by a `kind`/`mode` argument).
# ---------------------------------------------------------------------------

def _seaborn_available_or_message(panel) -> bool:
    _ensure_seaborn()
    if sns is None:
        panel.show_message("Seaborn is not installed. Run: python -m pip install seaborn")
        return False
    return True


def _use_coherent_colors(color_mode: str = "Basic") -> bool:
    """Two visual styles: Basic (conservative grayscale) and Coherent
    colors (consistent color families across correlation, co-presence,
    abundance, and positive/negative-association plots)."""
    return str(color_mode or "").strip().lower().startswith("coherent")


def _set_seaborn_theme(color_mode: str = "Basic") -> None:
    """Both colour modes draw in the house chart style. seaborn's own
    themes rewrite the process-wide rcParams (fonts, grids, spines), which
    is how two charts on one screen came to look like two products -- so
    no seaborn theme is set, and the house style is re-asserted here in
    case anything else touched it."""
    apply_chart_style()


# The two colour modes of the Association Workbench, inside one palette:
#   Basic            one hue for every quantity (blue), red only for the
#                    negative arm of a correlation
#   Coherent colors  one hue PER KIND of quantity, the same everywhere:
#                    abundance blue, co-presence green, negative
#                    association red; ranked bars shade with their rank
# A correlation is signed, so both modes use the same diverging map.
def _corr_cmap(color_mode: str = "Basic") -> str:
    return CMAP_DIVERGING


def _sequential_cmap(color_mode: str = "Basic") -> str:
    return CMAP_SEQUENTIAL


def _jaccard_cmap(color_mode: str = "Basic") -> str:
    return CMAP_GREENS if _use_coherent_colors(color_mode) else CMAP_SEQUENTIAL


def _projection_bar_color(color_mode: str = "Basic") -> str:
    return CHART_NEUTRAL


def _main_point_color(color_mode: str = "Basic") -> str:
    return BASE_COLOR


def _line_color(color_mode: str = "Basic") -> str:
    return NEGATIVE_COLOR if _use_coherent_colors(color_mode) else CHART_SECONDARY


def _pair_palette_name(mode: str, color_mode: str = "Basic") -> Optional[str]:
    """The ramp ranked bars shade along, or None for one flat colour."""
    if not _use_coherent_colors(color_mode):
        return None
    mode = str(mode or "").lower()
    if "negative" in mode:
        return CMAP_REDS
    if "jaccard" in mode:
        return CMAP_GREENS
    return CMAP_SEQUENTIAL


def _square_matrix_from_element_table(square_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return square_matrix_lookup(square_df)


@composable
def plot_seaborn_lower_triangle_matrix(
    panel, square_df: pd.DataFrame, title: str, cbar_label: str, cmap: str = CMAP_DIVERGING,
    center: Optional[float] = 0.0, annotate: bool = False, totals: Optional[pd.DataFrame] = None,
    projections: bool = False, color_mode: str = "Basic",
) -> None:
    """Lower-triangle heatmap shared by the corr heatmap, corr+projections,
    and Jaccard co-presence plot types (cbar_label picks the colormap)."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    data, elements = _square_matrix_from_element_table(square_df)
    if data.empty:
        panel.show_message("No square matrix to plot")
        return
    n = len(elements)
    mask = np.triu(np.ones_like(data.to_numpy(dtype=float), dtype=bool), k=1)
    cmap = (
        _corr_cmap(color_mode) if "correlation" in cbar_label.lower()
        else (_jaccard_cmap(color_mode) if "jaccard" in cbar_label.lower() else _sequential_cmap(color_mode))
    )
    projection_color = _projection_bar_color(color_mode)
    panel.figure.clear()
    base_size = panel.cap_square_size_inches(min(max(6.5, n * 0.34), 19.0))
    if projections:
        total_map: Dict[str, float] = {}
        if totals is not None and not totals.empty and "Element" in totals.columns and "Total_inventory_kg" in totals.columns:
            total_map = {str(r.Element): float(r.Total_inventory_kg or 0.0) for r in totals.itertuples(index=False)}
        proj = np.array([math.log10(max(total_map.get(e, 0.0), 0.0) + 1.0) for e in elements], dtype=float)
        panel.set_figure_size_inches(base_size + 3.2, base_size + 1.8)
        gs = panel.figure.add_gridspec(
            2, 4, width_ratios=[base_size, 0.30, 1.05, 2.2],
            height_ratios=[1.4, base_size], wspace=0.05, hspace=0.05)
        ax = panel.figure.add_subplot(gs[1, 0])
        ax_cbar = panel.figure.add_subplot(gs[1, 1])
        ax_top = panel.figure.add_subplot(gs[0, 0], sharex=ax)
        ax_right = panel.figure.add_subplot(gs[1, 3], sharey=ax)
        x = np.arange(n) + 0.5   # seaborn's cell centres
        ax_top.bar(x, proj, width=0.8, color=projection_color)
        mark_categories(ax_top, elements, axis="x", name="Element")
        ax_top.set_xlim(0, n)
        ax_top.set_ylabel("log10(total kg + 1)", fontsize=8)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_top.grid(True, axis="y")
        for spine in ("top", "right"):
            ax_top.spines[spine].set_visible(False)
    else:
        panel.set_figure_size_inches(base_size, base_size)
        ax = panel.figure.add_subplot(111)
        ax_cbar = None

    sns.heatmap(
        data, mask=mask, vmin=-1 if "correlation" in cbar_label.lower() else 0, vmax=1, center=center, cmap=cmap,
        square=not projections, linewidths=0.4 if n <= 45 else 0.0, linecolor="white",
        annot=bool(annotate and n <= 25), fmt=".2f",
        cbar_ax=ax_cbar,
        cbar_kws={"label": cbar_label} if ax_cbar is not None else {"label": cbar_label, "shrink": 0.75},
        ax=ax,
    )
    if projections:
        # suptitle, not ax.set_title(): ax sits directly below ax_top with
        # hspace=0.05 (no room reserved between rows for a title), so an
        # axes-level title renders on top of ax_top's bars instead of above
        # them. suptitle sits above the whole figure, clear of ax_top.
        panel.figure.suptitle(title)
    else:
        ax.set_title(title)
    ax.set_xlabel("Element")
    ax.set_ylabel("Element")
    ax.tick_params(axis="x", rotation=90, labelsize=8 if n <= 55 else 6)
    ax.tick_params(axis="y", rotation=0, labelsize=8 if n <= 55 else 6)

    if projections:
        y_centers = np.arange(n) + 0.5
        ax_right.barh(y_centers, proj, height=0.8, color=projection_color)
        mark_categories(ax_right, elements, axis="y", name="Element")
        ax_right.set_xlabel("log10(total kg + 1)", fontsize=8)
        ax_right.tick_params(axis="y", left=False, labelleft=False)
        ax_right.grid(True, axis="x")
        for spine in ("top", "right"):
            ax_right.spines[spine].set_visible(False)
        declare_minimum_size(ax, 4.0, 3.0)        # a matrix, two bar charts and a colour bar
    mark_cells(ax, elements, elements, row_name="Element", col_name="Element", value_label=cbar_label,
               triangle="lower")
    panel.ax = ax
    if not projections:
        panel.figure.tight_layout()               # the colour bar and its label, inside the pane
    panel.canvas.draw_idle()


@composable
def plot_seaborn_top_associations(panel, pair_stats: pd.DataFrame, top_n: int = 30, mode: str = "preferred", color_mode: str = "Basic") -> None:
    """mode: preferred / positive / negative / jaccard -- 4 of the 17 plot types."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if pair_stats is None or pair_stats.empty:
        panel.show_message("No pair statistics. Build the workbench data first.")
        return
    pdf = pair_stats.copy()
    if mode == "positive":
        pdf = pdf[pdf["Correlation_r"] > 0].sort_values("Correlation_r", ascending=False)
        value_col, xlabel = "Correlation_r", "Positive correlation r (kg matrix)"
    elif mode == "negative":
        pdf = pdf[pdf["Correlation_r"] < 0].sort_values("Correlation_r", ascending=True)
        value_col, xlabel = "Correlation_r", "Negative correlation r (kg matrix)"
    elif mode == "jaccard":
        pdf = pdf.sort_values("Jaccard_presence", ascending=False)
        value_col, xlabel = "Jaccard_presence", "Jaccard co-presence across tanks"
    else:
        pdf = pdf.sort_values("PreferredAssociationScore_proxy", ascending=False)
        value_col, xlabel = "PreferredAssociationScore_proxy", "Preferred association score proxy"
    pdf = pdf.head(int(top_n)).copy()
    if pdf.empty:
        panel.show_message("No associations to plot")
        return
    pdf["Pair"] = pdf["Element_A"].astype(str) + "-" + pdf["Element_B"].astype(str) + "  (n=" + pdf["N_both_present"].astype(str) + ")"
    pdf = pdf.iloc[::-1]
    ax = panel.reset_axes()
    panel.set_figure_size_inches(9, max(5, min(16, 0.32 * len(pdf) + 2)))
    palette_name = _pair_palette_name(mode, color_mode)
    if palette_name:
        colors = ramp_colors(len(pdf), palette_name)[::-1]          # the strongest pair is drawn last (top)
    else:
        colors = NEGATIVE_COLOR if mode == "negative" else BASE_COLOR
    category_barh(ax, pdf["Pair"].tolist(), pdf[value_col].astype(float).to_numpy(), color=colors,
                  name="Element pair", value_label=xlabel)
    ax.axvline(0, color=CHART_AXIS, linewidth=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Element pair")
    ax.set_title(f"Top kg element associations ({mode})")
    ax.grid(True, axis="x")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_seaborn_pair_matrix(
    panel, metric_matrix: pd.DataFrame, raw_matrix: pd.DataFrame, elements: Sequence[str], metric: str,
    kind: str = "regression", max_elements: int = 8, color_mode: str = "Basic",
) -> None:
    """kind: regression / scatter / kde -- 3 of the 17 plot types. Lower
    triangle only, diagonal = histogram+KDE, points colored by whether both
    elements are present in that tank."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    clean = [e for e in elements if e in metric_matrix.columns]
    clean = clean[:max(2, int(max_elements))]
    if len(clean) < 2:
        panel.show_message("Need at least two elements in the kg matrix")
        return
    data = metric_matrix[["WasteSiteId"] + clean].copy()
    for e in clean:
        data[e] = numeric_plot_series(data[e])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data[data[clean].notna().sum(axis=1) >= 2]
    if data.empty:
        panel.show_message("No finite values for pair matrix")
        return
    n = len(clean)
    size = panel.cap_square_size_inches(min(max(2.0 * n, 7), 18))
    panel.figure.clear()
    panel.set_figure_size_inches(size, size)
    axes = panel.figure.subplots(n, n, squeeze=False)
    raw = raw_matrix.set_index("WasteSiteId") if raw_matrix is not None and not raw_matrix.empty else pd.DataFrame()
    for i, y in enumerate(clean):
        for j, x in enumerate(clean):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            if i == j:
                vals = data[y].dropna()
                if vals.nunique() <= 1:
                    ax.text(0.5, 0.5, "constant", ha="center", va="center", transform=ax.transAxes)
                else:
                    sns.histplot(vals, kde=True, ax=ax, bins="auto", color=_main_point_color(color_mode))
                ax.set_ylabel("")
            else:
                sub = data[["WasteSiteId", x, y]].dropna()
                if raw is not None and not raw.empty and x in raw.columns and y in raw.columns:
                    both = (raw.loc[sub["WasteSiteId"], x].to_numpy(dtype=float) > 0) & (raw.loc[sub["WasteSiteId"], y].to_numpy(dtype=float) > 0)
                    sub = sub.assign(_both_present=both)
                else:
                    sub = sub.assign(_both_present=True)
                if sub.empty:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                elif kind == "kde":
                    if sub[x].nunique() > 1 and sub[y].nunique() > 1 and len(sub) >= 6:
                        try:
                            sns.kdeplot(data=sub, x=x, y=y, fill=True, levels=6, thresh=0.05, cmap=_sequential_cmap(color_mode), ax=ax)
                            sns.scatterplot(data=sub, x=x, y=y, hue="_both_present", palette={False: CHART_AXIS, True: _main_point_color(color_mode)}, legend=False, s=12, alpha=0.45, ax=ax)
                        except Exception:
                            sns.scatterplot(data=sub, x=x, y=y, hue="_both_present", palette={False: CHART_AXIS, True: _main_point_color(color_mode)}, legend=False, s=18, alpha=0.65, ax=ax)
                    else:
                        sns.scatterplot(data=sub, x=x, y=y, hue="_both_present", palette={False: CHART_AXIS, True: _main_point_color(color_mode)}, legend=False, s=18, alpha=0.65, ax=ax)
                elif kind == "scatter":
                    sns.scatterplot(data=sub, x=x, y=y, hue="_both_present", palette={False: CHART_AXIS, True: _main_point_color(color_mode)}, legend=False, s=18, alpha=0.70, ax=ax)
                else:
                    if sub[x].nunique() > 1 and sub[y].nunique() > 1 and len(sub) >= 3:
                        sns.regplot(data=sub, x=x, y=y, scatter_kws={"s": 16, "alpha": 0.55, "color": _main_point_color(color_mode)}, line_kws={"linewidth": 1.0, "color": _line_color(color_mode)}, ax=ax)
                    else:
                        sns.scatterplot(data=sub, x=x, y=y, color=_main_point_color(color_mode), s=18, alpha=0.70, ax=ax)
            if i != j and not sub.empty:
                mark_frame_points(ax, sub, x, y)
            if i < n - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(x, fontsize=8)
            if j > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(y, fontsize=8)
            ax.tick_params(labelsize=7)
    panel.figure.suptitle(f"Lower pair matrix - kg only - {humanize(metric)}", y=0.995)
    panel.ax = axes[-1, 0]
    declare_minimum_size(panel.ax, 0.7 * n + 0.9, 0.55 * n + 0.8)       # n x n charts
    panel.canvas.draw_idle()


@composable
def plot_seaborn_joint_first_two(panel, metric_matrix: pd.DataFrame, elements: Sequence[str], metric: str, kind: str = "scatter", color_mode: str = "Basic") -> None:
    """kind: regression / scatter / kde -- 3 of the 17 plot types, first two selected elements only."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    clean = [e for e in elements if e in metric_matrix.columns]
    if len(clean) < 2:
        panel.show_message("Need at least two elements")
        return
    x, y = clean[0], clean[1]
    data = metric_matrix[["WasteSiteId", x, y]].copy()
    data[x] = numeric_plot_series(data[x])
    data[y] = numeric_plot_series(data[y])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    if data.empty:
        panel.show_message("No finite values for joint plot")
        return
    panel.figure.clear()
    panel.set_figure_size_inches(8, 8)
    gs = panel.figure.add_gridspec(2, 2, width_ratios=[5, 1.2], height_ratios=[1.2, 5], wspace=0.05, hspace=0.05)
    ax_top = panel.figure.add_subplot(gs[0, 0])
    ax = panel.figure.add_subplot(gs[1, 0])
    ax_right = panel.figure.add_subplot(gs[1, 1], sharey=ax)
    if kind == "kde" and data[x].nunique() > 1 and data[y].nunique() > 1 and len(data) >= 6:
        try:
            sns.kdeplot(data=data, x=x, y=y, fill=True, levels=8, thresh=0.05, cmap=_sequential_cmap(color_mode), ax=ax)
            sns.scatterplot(data=data, x=x, y=y, color=_main_point_color(color_mode), s=18, alpha=0.45, ax=ax)
        except Exception:
            sns.scatterplot(data=data, x=x, y=y, color=_main_point_color(color_mode), s=30, alpha=0.7, ax=ax)
    elif kind == "regression" and data[x].nunique() > 1 and data[y].nunique() > 1 and len(data) >= 3:
        sns.regplot(data=data, x=x, y=y, scatter_kws={"s": 30, "alpha": 0.65, "color": _main_point_color(color_mode)}, line_kws={"linewidth": 1.2, "color": _line_color(color_mode)}, ax=ax)
    else:
        sns.scatterplot(data=data, x=x, y=y, color=_main_point_color(color_mode), s=30, alpha=0.7, ax=ax)
    sns.histplot(data[x].dropna(), kde=True, color=_main_point_color(color_mode), ax=ax_top)
    sns.histplot(y=data[y].dropna(), kde=True, color=_main_point_color(color_mode), ax=ax_right)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    r = data[x].corr(data[y]) if len(data) >= 3 else np.nan
    ax.set_xlabel(f"{x} ({humanize(metric)}, kg)")
    ax.set_ylabel(f"{y} ({humanize(metric)}, kg)")
    mark_frame_points(ax, data, x, y)
    ax.set_title(f"{x} vs {y} - kg only" + (f" | r={r:.3f}" if pd.notna(r) else ""))
    panel.ax = ax
    declare_minimum_size(ax, 3.2, 2.6)            # a scatter and its two marginal histograms
    panel.canvas.draw_idle()


@composable
def plot_seaborn_tank_similarity(panel, tank_similarity: pd.DataFrame, raw_matrix: pd.DataFrame, top_tanks: int = 40, annotate: bool = False, color_mode: str = "Basic") -> None:
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if tank_similarity is None or tank_similarity.empty or "WasteSiteId" not in tank_similarity.columns:
        panel.show_message("No tank similarity matrix")
        return
    data = tank_similarity.copy().set_index("WasteSiteId")
    tanks = [str(t) for t in data.index.tolist()]
    keep = [t for t in tanks if t in data.columns]
    data = data.loc[keep, keep]
    if raw_matrix is not None and not raw_matrix.empty and "WasteSiteId" in raw_matrix.columns:
        raw = raw_matrix.set_index("WasteSiteId")
        totals = raw.sum(axis=1).sort_values(ascending=False).head(int(top_tanks)).index.astype(str).tolist()
        keep = [t for t in totals if t in data.index]
        if len(keep) >= 2:
            data = data.loc[keep, keep]
    else:
        data = data.iloc[:int(top_tanks), :int(top_tanks)]
    n = len(data)
    if n < 2:
        panel.show_message("Need at least two tanks for tank similarity")
        return
    panel.figure.clear()
    base_size = panel.cap_square_size_inches(min(max(7, n * 0.23), 18))
    panel.set_figure_size_inches(base_size, base_size)
    ax = panel.figure.add_subplot(111)
    mask = np.triu(np.ones_like(data.to_numpy(dtype=float), dtype=bool), k=1)
    sns.heatmap(
        data, mask=mask, vmin=-1, vmax=1, center=0, cmap=_corr_cmap(color_mode), square=True,
        linewidths=0.2 if n <= 50 else 0, annot=bool(annotate and n <= 20), fmt=".2f",
        cbar_kws={"label": "Tank-to-tank correlation across selected kg elements"}, ax=ax,
    )
    ax.set_title(f"Tank similarity lower triangle - kg element vectors - top {n} tanks")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    tank_names = [str(t) for t in data.index]
    mark_cells(ax, tank_names, tank_names, row_name="Tank", col_name="Tank", value_label="Correlation r",
               triangle="lower")
    panel.ax = ax
    panel.figure.tight_layout()                   # upright tank names under the matrix need the room
    panel.canvas.draw_idle()


@composable
def plot_seaborn_tank_element_map(panel, raw_matrix: pd.DataFrame, elements: Sequence[str], top_tanks: int = 60, metric: str = "log10_plus1", color_mode: str = "Basic") -> None:
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if raw_matrix is None or raw_matrix.empty or "WasteSiteId" not in raw_matrix.columns:
        panel.show_message("No raw kg matrix")
        return
    clean = [e for e in elements if e in raw_matrix.columns]
    if not clean:
        panel.show_message("No selected elements in raw matrix")
        return
    raw = raw_matrix[["WasteSiteId"] + clean].copy().set_index("WasteSiteId")
    raw = raw.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    top_ids = raw.sum(axis=1).sort_values(ascending=False).head(int(top_tanks)).index.tolist()
    raw = raw.loc[top_ids]
    if metric == "fraction":
        denom = raw.sum(axis=1).replace(0.0, np.nan)
        data = raw.div(denom, axis=0)
        label = "Fraction of selected kg inventory"
    else:
        data = np.log10(raw + 1.0)
        label = "log10(inventory in kg + 1)"
    panel.figure.clear()
    width = min(max(7, len(clean) * 0.5), 18)
    height = min(max(6, len(raw) * 0.16), 18)
    panel.set_figure_size_inches(width, height)
    ax = panel.figure.add_subplot(111)
    sns.heatmap(data, cmap=_sequential_cmap(color_mode), cbar_kws={"label": label}, linewidths=0.0, ax=ax)
    ax.set_title(f"Tank x selected element map - kg only - top {len(raw)} tanks")
    ax.set_xlabel("Element")
    ax.set_ylabel("Tank")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    mark_cells(ax, [str(t) for t in data.index], [str(c) for c in data.columns],
               row_name="Tank", col_name="Element", value_label=label)
    panel.ax = ax
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_tank_element_bars(
    panel, raw_matrix: pd.DataFrame, elements: Sequence[str], *, top_n: int = 30,
    mode: str = "Stacked", sort_by: str = "Total (desc)", log_scale: bool = False,
    color_mode: str = "Basic",
) -> None:
    """Per-tank kg inventory for N selected elements as horizontal bars --
    Stacked (each tank's bar split into one segment per element, total
    length = total inventory -- the classic publication summary-figure
    shape) or Grouped (one bar per element per tank, side by side, for
    direct magnitude comparison instead of a cumulative total). kg only,
    same reasoning as the rest of this workbench: mixing kg and Ci on one
    bar length would be physically meaningless."""
    clean = [e for e in elements if e in raw_matrix.columns] if raw_matrix is not None else []
    if raw_matrix is None or raw_matrix.empty or not clean:
        panel.show_message("No selected elements in raw kg matrix")
        return
    df = raw_matrix[["WasteSiteId"] + clean].copy()
    for e in clean:
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0.0)
    df["_total"] = df[clean].sum(axis=1)
    df = df[df["_total"] > 0]
    if df.empty:
        panel.show_message("No positive inventory for the selected elements")
        return

    sort_key = str(sort_by or "Total (desc)").lower()
    if sort_key.startswith("total (desc"):
        df = df.sort_values("_total", ascending=False)
    elif sort_key.startswith("total (asc"):
        df = df.sort_values("_total", ascending=True)
    else:
        df = df.sort_values("WasteSiteId", ascending=True)
    df = df.head(int(top_n))
    df = df.iloc[::-1]  # barh draws bottom-to-top; reverse for natural reading order top-to-bottom

    ax = panel.reset_axes()
    width = panel.figure.get_size_inches()[0]
    panel.set_figure_size_inches(width, bar_figure_height(len(df), per_bar=0.28))

    labels = make_unique_plot_labels(df, "WasteSiteId")
    y = np.arange(len(df))
    n = len(clean)
    # Elements differ in kind: categorical colours, in the fixed order.
    colors = series_colors(n)

    is_stacked = str(mode or "Stacked").strip().lower().startswith("stack")
    if is_stacked:
        left = np.zeros(len(df))
        for e, color in zip(clean, colors):
            vals = df[e].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, height=0.75, color=color, label=e)
            left += vals
        if log_scale:
            # Not blocked, but a log-scaled x-axis makes stacked segment
            # lengths visually non-additive -- Grouped is the honest choice
            # for a log comparison; this is left available for whoever
            # wants it anyway (e.g. one dominant element vs the rest).
            ax.set_xscale("log")
    else:
        bar_h = 0.8 / max(n, 1)
        for i, (e, color) in enumerate(zip(clean, colors)):
            offset = (i - (n - 1) / 2) * bar_h
            ax.barh(y + offset, df[e].to_numpy(dtype=float), height=bar_h * 0.9, color=color, label=e)
        if log_scale:
            ax.set_xscale("log")

    declare_categories(ax, labels.astype(str).tolist(), axis="y", name="Tank", value_label="Inventory (kg)",
                       unit="kg", series_name="Element", tank_ids=df["WasteSiteId"].astype(str).tolist())
    ax.set_xlabel(f"Inventory ({', '.join(clean)}, kg)" if n <= 4 else f"Inventory ({n} elements, kg)")
    ax.set_ylabel("Tank")
    ax.set_title(f"Tank inventory by element - kg only ({'stacked' if is_stacked else 'grouped'})")
    ax.grid(True, axis="x")
    ax.legend(title="Element", fontsize=8, loc="lower right")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_seaborn_presence_patterns(panel, presence_matrix: pd.DataFrame, elements: Sequence[str], top_n: int = 30, color_mode: str = "Basic") -> None:
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if presence_matrix is None or presence_matrix.empty or "WasteSiteId" not in presence_matrix.columns:
        panel.show_message("No presence matrix")
        return
    clean = [e for e in elements if e in presence_matrix.columns]
    if len(clean) < 2:
        panel.show_message("Need at least two elements")
        return
    df = presence_matrix[["WasteSiteId"] + clean].copy()
    for e in clean:
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0).astype(int)
    df["N_selected_elements_present"] = df[clean].sum(axis=1)
    df["Combination"] = df[clean].apply(lambda row: "+".join([e for e, v in row.items() if int(v) > 0]) or "none", axis=1)
    counts = df.groupby(["Combination", "N_selected_elements_present"], as_index=False).size().rename(columns={"size": "N_tanks"})
    counts = counts[counts["Combination"] != "none"].sort_values("N_tanks", ascending=False).head(int(top_n))
    if counts.empty:
        panel.show_message("No non-empty presence patterns")
        return
    counts = counts.iloc[::-1]
    ax = panel.reset_axes()
    panel.set_figure_size_inches(10, max(5, min(16, 0.32 * len(counts) + 2)))
    levels = sorted(int(v) for v in counts["N_selected_elements_present"].unique())
    if _use_coherent_colors(color_mode):
        level_colors = dict(zip(levels, ramp_colors(len(levels), _jaccard_cmap(color_mode))[::-1]))
    else:
        level_colors = {level: BASE_COLOR for level in levels}
    category_barh(
        ax, counts["Combination"].astype(str).tolist(), counts["N_tanks"].astype(float).to_numpy(),
        color=[level_colors[int(v)] for v in counts["N_selected_elements_present"]],
        name="Combination", value_label="Number of tanks",
    )
    ax.set_xlabel("Number of tanks")
    ax.set_ylabel("Element combination present in tank")
    ax.set_title("Most common selected-element co-presence patterns - kg only")
    ax.grid(True, axis="x")
    if _use_coherent_colors(color_mode) and len(levels) > 1:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=level_colors[level], label=str(level)) for level in levels],
                  title="Elements present", fontsize=8)
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_seaborn_stats_dashboard(panel, element_stats: pd.DataFrame, pair_stats: pd.DataFrame, top_n: int = 20, color_mode: str = "Basic") -> None:
    """4-panel dashboard: total-kg bars, abundance-vs-spread scatter,
    r-vs-Jaccard scatter, top-12 preferred-association bars."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if element_stats is None or element_stats.empty:
        panel.show_message("No element statistics")
        return
    elems = element_stats.sort_values("Total_inventory_kg", ascending=False).head(int(top_n)).copy()
    pairs = pair_stats.sort_values("PreferredAssociationScore_proxy", ascending=False).head(int(top_n)).copy() if pair_stats is not None and not pair_stats.empty else pd.DataFrame()
    panel.figure.clear()
    panel.set_figure_size_inches(14, 10)
    # No spacing of its own on the gridspec: the gaps between the four
    # charts are set on the figure (see the end of this function), where a
    # fitted layout can adjust them to what each chart writes.
    gs = panel.figure.add_gridspec(2, 2)
    ax1 = panel.figure.add_subplot(gs[0, 0])
    ax2 = panel.figure.add_subplot(gs[0, 1])
    ax3 = panel.figure.add_subplot(gs[1, 0])
    ax4 = panel.figure.add_subplot(gs[1, 1])
    ranked = elems.iloc[::-1]
    category_barh(
        ax1, ranked["Element"].astype(str).tolist(), ranked["Total_inventory_kg"].astype(float).to_numpy(),
        color=(ramp_colors(len(ranked))[::-1] if _use_coherent_colors(color_mode) else BASE_COLOR),
        name="Element", value_label="Total inventory (kg)", unit="kg",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Total inventory (kg, log scale)")
    ax1.set_ylabel("Element")
    ax1.set_title("Selected elements: total kg")
    sns.scatterplot(data=elems, x="PresenceFraction_pct", y="Total_inventory_kg", size="Max_kg_in_one_tank", color=_main_point_color(color_mode), sizes=(30, 300), legend=False, ax=ax2)
    mark_frame_points(ax2, elems, "PresenceFraction_pct", "Total_inventory_kg", id_col="Element", name="Element")
    ax2.set_yscale("log")
    ax2.set_xlabel("Presence fraction across tanks (%)")
    ax2.set_ylabel("Total inventory (kg)")
    ax2.set_title("Abundance vs spread")
    if not pairs.empty:
        p = pairs.copy()
        p["Pair"] = p["Element_A"].astype(str) + "-" + p["Element_B"].astype(str)
        sns.scatterplot(data=p, x="Correlation_r", y="Jaccard_presence", size="N_both_present", hue="PreferredAssociationScore_proxy", palette=_sequential_cmap(color_mode), sizes=(30, 250), ax=ax3)
        mark_frame_points(ax3, p, "Correlation_r", "Jaccard_presence", id_col="Pair", name="Pair")
        legend3 = ax3.get_legend()
        if legend3 is not None:
            # seaborn's legend lists two variables (colour and size) one
            # under the other in full-size type: taller than a quarter of
            # the dashboard. The same entries, small, in two columns.
            ax3.legend(
                list(legend3.legend_handles), [humanize(text.get_text()) for text in legend3.get_texts()],
                ncol=2, fontsize=7, labelspacing=0.2, handletextpad=0.3, columnspacing=0.8, borderpad=0.2,
                markerscale=0.6, loc="best", frameon=False,
            )
        ax3.set_xlim(-1.05, 1.05)
        ax3.set_xlabel("Correlation r")
        ax3.set_ylabel("Jaccard co-presence")
        ax3.set_title("Pair associations")
        p2 = p.sort_values("PreferredAssociationScore_proxy", ascending=False).head(12).iloc[::-1]
        category_barh(
            ax4, p2["Pair"].astype(str).tolist(), p2["PreferredAssociationScore_proxy"].astype(float).to_numpy(),
            color=(ramp_colors(len(p2))[::-1] if _use_coherent_colors(color_mode) else BASE_COLOR),
            name="Pair", value_label="Association score (proxy)",
        )
        ax4.set_xlabel("Association score (proxy)")
        ax4.set_ylabel("Pair")
        ax4.set_title("Top preferred associations")
    else:
        ax3.text(0.5, 0.5, "No pair stats", ha="center", va="center", transform=ax3.transAxes)
        ax4.text(0.5, 0.5, "No pair stats", ha="center", va="center", transform=ax4.transAxes)
    panel.ax = ax1
    # Four charts with their own titles and axis labels. The gap between
    # two rows holds tick labels, an axis label (three lines when it says,
    # over two lines, that labels were thinned) and a title; the gap
    # between two columns, tick labels and an axis label. Those are sizes
    # of text, not shares of the pane, so the relative spacing is
    # recomputed for the size the figure has (a resized pane, an export,
    # a Figure Composer cell). In a pane the fitted layout requested
    # below then replaces this estimate by the gaps it measures.
    figure = panel.figure
    row_gap, column_gap = 1.0, 0.8                # inches

    def _spacing() -> None:
        width, height = figure_size_inches(figure)
        row = max((0.78 * height - row_gap) / 2.0, 0.4)
        column = max((0.82 * width - column_gap) / 2.0, 0.4)
        figure.subplots_adjust(hspace=row_gap / row, wspace=column_gap / column)

    _spacing()
    setattr(ax1, RELAYOUT_ATTR, _spacing)
    declare_minimum_size(ax1, 5.6, 3.8)           # four charts, each with a title and two axis labels
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Correlations "Structure" sub-tab (M8): PCA scatter, dendrogram, partial-
# vs-raw correlation comparison, element-association network graph.
# ---------------------------------------------------------------------------

@composable
def plot_pca_scatter(panel, tank_summary: pd.DataFrame, color_by: Optional[str], pca_variance: pd.DataFrame, color_mode: str = "Basic") -> None:
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if tank_summary is None or tank_summary.empty or "PC1" not in tank_summary.columns or "PC2" not in tank_summary.columns:
        panel.show_message("No PCA scores. Build the structure data first.")
        return
    pdf = tank_summary.copy()
    var_map: Dict[str, float] = {}
    if pca_variance is not None and not pca_variance.empty:
        var_map = {str(r.PC): float(r.ExplainedVarianceRatio) for r in pca_variance.itertuples(index=False)}
    xlabel = f"PC1 ({var_map.get('PC1', 0.0) * 100:.1f}%)" if var_map else "PC1"
    ylabel = f"PC2 ({var_map.get('PC2', 0.0) * 100:.1f}%)" if var_map else "PC2"

    panel.figure.clear()
    ax = panel.figure.add_subplot(111)
    hue_col = color_by if color_by and color_by in pdf.columns and pdf[color_by].notna().any() else None
    has_ids = "WasteSiteId" in pdf.columns

    def _tag(collection, frame) -> None:
        if has_ids:
            tanks = frame["WasteSiteId"].astype(str).tolist()
            extra = [("Cluster", frame["Cluster"].tolist())] if "Cluster" in frame.columns else None
            mark_points(ax, collection, tanks, name="Tank", tank_ids=tanks,
                        series_name=humanize(hue_col) if hue_col else "", extra=extra)

    if hue_col:
        pdf[hue_col] = pdf[hue_col].astype(str).fillna("Unknown")
        levels = [str(v) for v in pd.unique(pdf[hue_col])]
        try:
            levels = sorted(levels, key=lambda v: (0, float(v)) if v.replace(".", "", 1).isdigit() else (1, v))
        except Exception:
            levels = sorted(levels)
        # Colour follows the fixed categorical order; past eight levels the
        # marker shape changes with each pass through the palette, so two
        # levels never share both.
        shapes = ("o", "s", "^", "D", "v", "P")
        colors = series_colors(len(levels))
        for i, level in enumerate(levels):
            group = pdf[pdf[hue_col] == level]
            sc = ax.scatter(group["PC1"], group["PC2"], s=52, alpha=0.9, color=colors[i],
                            marker=shapes[(i // len(DATA_CATEGORICAL)) % len(shapes)],
                            edgecolors="white", linewidths=0.6, label=level)
            _tag(sc, group)
        ax.legend(title=humanize(hue_col), bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=8,
                  ncol=1 if len(levels) <= 18 else 2, borderaxespad=0.0)
    else:
        sc = ax.scatter(pdf["PC1"], pdf["PC2"], s=52, alpha=0.9, color=_main_point_color(color_mode),
                        edgecolors="white", linewidths=0.6)
        _tag(sc, pdf)
    ax.axhline(0, color=CHART_GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=CHART_GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Tank PCA (kg, standardized)" + (f" - colored by {humanize(color_by)}" if hue_col else ""))
    ax.grid(True)
    panel.ax = ax
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_dendrogram(panel, cluster_linkage, cluster_labels: Sequence[str], color_mode: str = "Basic") -> None:
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if cluster_linkage is None or len(cluster_linkage) == 0 or not cluster_labels:
        panel.show_message("No clustering result. Build the structure data first.")
        return
    from scipy.cluster.hierarchy import dendrogram

    n = len(cluster_labels)
    panel.figure.clear()
    base_size = min(max(6.5, n * 0.28), 19.0)
    panel.set_figure_size_inches(base_size, 6.0)
    ax = panel.figure.add_subplot(111)
    tree = dendrogram(
        cluster_linkage, labels=list(cluster_labels), ax=ax, leaf_rotation=90,
        leaf_font_size=8 if n <= 55 else 6, color_threshold=0, above_threshold_color=_line_color(color_mode),
    )
    # The leaves are tanks: give each one something to point at (scipy
    # draws only lines). Leaf i sits at x = 5 + 10 i on the baseline.
    leaves = [str(v) for v in (tree.get("ivl") or [])]
    if leaves:
        feet = ax.scatter(5.0 + 10.0 * np.arange(len(leaves)), np.zeros(len(leaves)), s=30,
                          color=BASE_COLOR, alpha=0.0, zorder=3, clip_on=False)
        mark_points(ax, feet, leaves, name="Tank", tank_ids=[tank_id_of(v) for v in leaves],
                    show_position=False)
    ax.set_title("Tank hierarchical clustering (kg, standardized)")
    ax.set_ylabel("Distance")
    ax.grid(True, axis="y")
    panel.ax = ax
    panel.figure.tight_layout()                   # upright tank names under the tree need the room
    panel.canvas.draw_idle()


@composable
def plot_partial_correlation_comparison(panel, partial_df: pd.DataFrame, raw_df: pd.DataFrame, annotate: bool = False, color_mode: str = "Basic") -> None:
    """Raw and partial (controlling for tank size) correlation heatmaps
    side by side, so the size of the "everything correlates because both
    scale with tank size" effect is directly visible."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    raw_data, raw_elements = _square_matrix_from_element_table(raw_df)
    partial_data, _partial_elements = _square_matrix_from_element_table(partial_df)
    if raw_data.empty or partial_data.empty:
        panel.show_message("No partial-correlation matrix. Build the structure data first.")
        return
    n = len(raw_elements)
    cmap = _corr_cmap(color_mode)
    panel.figure.clear()
    base_size = min(max(5.5, n * 0.32), 14.0)
    # Two side-by-side square heatmaps sharing base_size: cap it against
    # the width formula's own 2x+1.2 factor, not the raw available width,
    # or the pair would overflow even though the figure fits the ceiling.
    avail_w = panel.available_content_width_inches()
    avail_h = panel.available_content_height_inches()
    if avail_w is not None:
        base_size = min(base_size, (avail_w - 1.2) / 2)
    if avail_h is not None:
        base_size = min(base_size, avail_h - 1.0)
    panel.set_figure_size_inches(base_size * 2 + 1.2, base_size + 1.0)
    # Two equal panels. The colour bar they share hangs off the right edge
    # of the second one, in that panel's own coordinates, so it is exactly
    # as tall as the matrix and takes no room from either panel.
    ax1 = panel.figure.add_subplot(1, 2, 1)
    ax2 = panel.figure.add_subplot(1, 2, 2)
    cax = ax2.inset_axes([1.05, 0.0, 0.05, 1.0])
    mask1 = np.triu(np.ones_like(raw_data.to_numpy(dtype=float), dtype=bool), k=1)
    mask2 = np.triu(np.ones_like(partial_data.to_numpy(dtype=float), dtype=bool), k=1)
    annot = bool(annotate and n <= 25)
    # The two panels share one colour scale, so they share one colour bar
    # (at the right of the pair): a bar between them collides with the
    # second panel's axis label, and a bar taken out of the second panel
    # alone makes it smaller than the first.
    sns.heatmap(raw_data, mask=mask1, vmin=-1, vmax=1, center=0, cmap=cmap, square=True,
                linewidths=0.4 if n <= 45 else 0.0, linecolor="white", annot=annot, fmt=".2f",
                cbar=False, ax=ax1)
    sns.heatmap(partial_data, mask=mask2, vmin=-1, vmax=1, center=0, cmap=cmap, square=True,
                linewidths=0.4 if n <= 45 else 0.0, linecolor="white", annot=annot, fmt=".2f",
                cbar_ax=cax, cbar_kws={"label": "Correlation r (both panels)"}, ax=ax2)
    ax1.set_title("Raw correlation")
    ax2.set_title("Partial correlation\n(controlling for tank size)")
    mark_cells(ax1, raw_elements, raw_elements, row_name="Element", col_name="Element", value_label="Correlation r",
               triangle="lower")
    mark_cells(ax2, _partial_elements, _partial_elements, row_name="Element", col_name="Element",
               value_label="Partial correlation r", triangle="lower")
    for ax in (ax1, ax2):
        ax.set_xlabel("Element")
        ax.set_ylabel("Element" if ax is ax1 else "")
        ax.tick_params(axis="x", rotation=90, labelsize=8 if n <= 55 else 6)
        ax.tick_params(axis="y", rotation=0, labelsize=8 if n <= 55 else 6)
    panel.ax = ax1
    declare_minimum_size(ax1, 4.4, 2.6)           # two matrices side by side and a colour bar
    # The colour bar sits outside the second panel's box: only a fitted
    # layout leaves it room at the right edge of the figure.
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_element_network(panel, network_nodes: pd.DataFrame, network_edges: pd.DataFrame, color_mode: str = "Basic") -> None:
    """Element-association network: node position/size come straight from
    structure_science.element_network (a shared spring layout also used by
    html_export's Plotly version, so the two views always agree). Every
    selected element is drawn even with zero edges -- an isolated node at a
    strict threshold is itself informative."""
    if not _seaborn_available_or_message(panel):
        return
    _set_seaborn_theme(color_mode)
    if network_nodes is None or network_nodes.empty:
        panel.show_message("No network data. Build the structure data first.")
        return
    from matplotlib.lines import Line2D

    panel.figure.clear()
    ax = panel.figure.add_subplot(111)
    pos = {str(r.Element): (float(r.x), float(r.y)) for r in network_nodes.itertuples(index=False)}
    sizes = {str(r.Element): float(r.LogTotalInventory) for r in network_nodes.itertuples(index=False)}
    max_size = max(sizes.values()) if sizes else 0.0
    node_sizes = [200.0 + 1400.0 * (sizes[e] / max_size if max_size > 0 else 0.0) for e in pos]

    positive_color = POSITIVE_COLOR
    negative_color = NEGATIVE_COLOR
    if network_edges is not None and not network_edges.empty:
        for row in network_edges.itertuples(index=False):
            x1, y1 = pos[row.Element_A]
            x2, y2 = pos[row.Element_B]
            width = 0.6 + 3.0 * float(row.AbsCorrelation)
            color = positive_color if row.Sign == "positive" else negative_color
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.55, zorder=1)
    xs = [pos[e][0] for e in pos]
    ys = [pos[e][1] for e in pos]
    # Not clipped to the frame: a disc at the edge of the layout is drawn
    # whole, and the layout keeps it inside the pane like any other ink.
    nodes = ax.scatter(xs, ys, s=node_sizes, color=DATA_BLUE_RAMP[2], edgecolor=BASE_COLOR, linewidth=1.0, zorder=2,
                       clip_on=False)
    node_extra = [("log10 total inventory", [sizes[e] for e in pos])]
    if "N_edges" in network_nodes.columns:
        links = {str(r.Element): int(r.N_edges) for r in network_nodes.itertuples(index=False)}
        node_extra.append(("Links", [links.get(e, 0) for e in pos]))
    mark_points(ax, nodes, list(pos), name="Element", extra=node_extra, show_position=False)
    for e, (x, y) in pos.items():
        ax.annotate(e, (x, y), ha="center", va="center", fontsize=9, fontweight="semibold",
                    color=CHART_INK, zorder=3)

    legend_handles = []
    if network_edges is not None and not network_edges.empty:
        if (network_edges["Sign"] == "positive").any():
            legend_handles.append(Line2D([0], [0], color=positive_color, lw=2, label="positive r"))
        if (network_edges["Sign"] == "negative").any():
            legend_handles.append(Line2D([0], [0], color=negative_color, lw=2, label="negative r"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
    ax.set_title("Element association network (kg)")
    ax.axis("off")
    # A node is a disc of fixed size around its centre: the outermost
    # ones need room inside the frame, or the frame cuts them.
    ax.margins(0.14)
    panel.ax = ax
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Vitrification Screening / Candidate Search / Blend Partners (M10), ported
# from the old app's plot_vitrification_burden / plot_candidate_scores /
# plot_blend_scores.
# ---------------------------------------------------------------------------

# Term names/labels come from vitrification_science.SCORE_CONTRIB_TERMS --
# the single registry tank_category_summary's contrib_* columns are defined
# against -- so a new weighted term added there shows up in this plot by
# construction (review finding: a local copy would let the stacked bars
# silently stop summing to the net-score tick). Colors cycle if the
# registry ever outgrows this palette.
_SCORE_COLORS = list(DATA_CATEGORICAL)


def _score_contrib_columns() -> Dict[str, Tuple[str, str]]:
    from vitrification_science import SCORE_CONTRIB_TERMS

    return {
        col: (label, _SCORE_COLORS[i % len(_SCORE_COLORS)])
        for i, (col, label) in enumerate(SCORE_CONTRIB_TERMS.items())
    }


@composable
def plot_vitrification_score_breakdown(panel, data: pd.DataFrame, title: str = "Score breakdown", top_n: int = 20) -> None:
    """Stacked horizontal bars showing exactly how each tank's screening
    score is built from its five weighted terms -- the direct fix for "the
    score is a black box": instead of one unlabeled number, each tank's
    bar shows the actual +/- contribution of glass-former, modifier,
    problem, volatile, and redox content (tank_category_summary's own
    contrib_* columns, so this can never drift out of sync with the real
    score), with a tick mark at the net (clipped) score."""
    contrib_columns = _score_contrib_columns()
    required = list(contrib_columns) + ["WasteSiteId", "Vitrification_screening_score_proxy"]
    if data is None or data.empty or any(c not in data for c in required):
        panel.show_message("No score breakdown data")
        return
    # Trim to the plotted slice BEFORE copying/coercing -- the summary frame
    # carries ~30 columns for all 177 tanks and only top_n rows x these
    # columns are drawn (review finding: whole-frame copy per repaint).
    pdf = (
        data.loc[:, required]
        .nlargest(int(top_n), "Vitrification_screening_score_proxy")
        .copy()
    )
    for c in contrib_columns:
        pdf[c] = numeric_plot_series(pdf[c]).fillna(0.0)
    pdf = pdf.iloc[::-1]  # barh draws bottom-to-top; reverse for highest score at the top

    ax = panel.reset_axes()
    width = panel.figure.get_size_inches()[0]
    panel.set_figure_size_inches(width, bar_figure_height(len(pdf), per_bar=0.32))

    labels = make_unique_plot_labels(pdf, "WasteSiteId")
    y = np.arange(len(pdf))
    left_pos = np.zeros(len(pdf))
    left_neg = np.zeros(len(pdf))
    for col, (label, color) in contrib_columns.items():
        vals = pdf[col].to_numpy(dtype=float)
        lefts = np.where(vals >= 0, left_pos, left_neg)
        ax.barh(y, vals, left=lefts, height=0.7, color=color, label=label)
        left_pos = np.where(vals >= 0, left_pos + vals, left_pos)
        left_neg = np.where(vals < 0, left_neg + vals, left_neg)

    net = ax.scatter(
        pdf["Vitrification_screening_score_proxy"].to_numpy(dtype=float), y,
        color=CHART_INK, marker="|", s=220, linewidths=1.6, zorder=5, label="net score",
    )
    tanks = pdf["WasteSiteId"].astype(str).tolist()
    mark_points(ax, net, tanks, name="Tank", tank_ids=tanks, x_label="Net screening score", y_label="Rank position")
    declare_categories(ax, labels.astype(str).tolist(), axis="y", name="Tank", tank_ids=tanks,
                       value_label="Contribution to screening score", series_name="Term")
    ax.axvline(0, color=CHART_AXIS, linewidth=0.8)
    ax.set_xlabel("Contribution to screening score")
    ax.set_title(title)
    ax.grid(True, axis="x")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_vitrification_burden(panel, data: pd.DataFrame, title: str = "Vitrification screening map") -> None:
    required = ["Total_kg_inventory", "Total_Ci_inventory", "frac_problem_elements_proxy", "frac_glass_former_or_intermediate", "WasteSiteId"]
    if data is None or data.empty or any(c not in data for c in required):
        panel.show_message("No vitrification screening data")
        return
    pdf = data.copy()
    for c in required[:-1]:
        pdf[c] = numeric_plot_series(pdf[c]).fillna(0.0)
    pdf = pdf[(pdf["Total_kg_inventory"] > 0) | (pdf["Total_Ci_inventory"] > 0)]
    if pdf.empty:
        panel.show_message("No positive tank inventory")
        return
    ax = panel.reset_axes()
    size = 40 + 500 * pdf["frac_glass_former_or_intermediate"].clip(lower=0, upper=1)
    sc = ax.scatter(
        pdf["Total_kg_inventory"].clip(lower=1e-30), pdf["Total_Ci_inventory"].clip(lower=1e-30),
        c=pdf["frac_problem_elements_proxy"].clip(lower=0), s=size, alpha=0.85, cmap=CMAP_SEQUENTIAL,
        edgecolors=CHART_SECONDARY, linewidths=0.4,
    )
    tanks = pdf["WasteSiteId"].astype(str).tolist()
    mark_points(ax, sc, tanks, name="Tank", tank_ids=tanks, extra=[
        ("Glass formers and intermediates (fraction)", pdf["frac_glass_former_or_intermediate"].tolist()),
    ])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total chemical inventory (kg)")
    ax.set_ylabel("Total radiological inventory (Ci)")
    ax.set_title(title)
    ax.grid(True)
    cbar = panel.figure.colorbar(sc, ax=ax)
    cbar.set_label("Problem-element proxy fraction (kg/kg)")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_candidate_scores(panel, data: pd.DataFrame, score_col: str = "User_search_score_proxy", top_n: int = 30) -> None:
    if data is None or data.empty or score_col not in data or "WasteSiteId" not in data:
        panel.show_message("No candidate ranking data")
        return
    pdf = data.copy()
    pdf["_score"] = numeric_plot_series(pdf[score_col])
    pdf = pdf.dropna(subset=["_score"]).sort_values("_score", ascending=False).head(int(top_n)).iloc[::-1]
    if pdf.empty:
        panel.show_message("No candidate scores to plot")
        return
    ax = panel.reset_axes()
    tanks = pdf["WasteSiteId"].astype(str).tolist()
    category_barh(ax, tanks, pdf["_score"].astype(float).to_numpy(), color=BASE_COLOR,
                  name="Tank", value_label=humanize(score_col), tank_ids=tanks)
    ax.set_xlabel(f"{humanize(score_col)}, dimensionless")
    ax.set_ylabel("Tank")
    ax.set_title("Candidate tank ranking")
    ax.grid(True, axis="x")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


@composable
def plot_blend_scores(panel, data: pd.DataFrame, top_n: int = 30) -> None:
    if data is None or data.empty or "Blend_complement_score_proxy" not in data or "PartnerTank" not in data:
        panel.show_message("No blend partner data")
        return
    pdf = data.copy()
    pdf["_score"] = numeric_plot_series(pdf["Blend_complement_score_proxy"])
    pdf = pdf.dropna(subset=["_score"]).sort_values("_score", ascending=False).head(int(top_n)).iloc[::-1]
    if pdf.empty:
        panel.show_message("No blend scores to plot")
        return
    ax = panel.reset_axes()
    partners = pdf["PartnerTank"].astype(str).tolist()
    category_barh(ax, partners, pdf["_score"].astype(float).to_numpy(), color=BASE_COLOR,
                  name="Partner tank", value_label="Blend complement score (proxy)", tank_ids=partners)
    ax.set_xlabel("Blend complement score (proxy), dimensionless")
    ax.set_ylabel("Partner tank")
    base = str(pdf["BaseTank"].iloc[0]) if "BaseTank" in pdf and len(pdf) else "base"
    ax.set_title(f"Potential blend partners for {base}")
    ax.grid(True, axis="x")
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Cross-correlation tab (association_query): the coefficient matrix and the
# scatter behind any one coefficient.
# ---------------------------------------------------------------------------

# Diverging blue <-> red with a NEUTRAL GRAY midpoint and equal steps per
# arm. A coefficient of zero means "no association" and must read as
# nothing: a hue at the midpoint (or a rainbow) would put color on the
# cells that carry no signal.
ASSOCIATION_DIVERGING: Tuple[str, ...] = DATA_DIVERGING
# Color follows the phase family, never its rank in a given plot, and the
# marker shape repeats the distinction so it is not carried by hue alone.
PHASE_FAMILY_STYLE: Dict[str, Tuple[str, str]] = {
    "Supernatant": (DATA_CATEGORICAL[0], "o"),
    "Saltcake": (DATA_CATEGORICAL[1], "s"),
    "Sludge": (DATA_CATEGORICAL[2], "^"),
}
_OTHER_FAMILY_STYLE: Tuple[str, str] = (CHART_FAINT, "D")
# Above this many components a written coefficient is too small to read.
ASSOCIATION_MAX_ANNOTATED = 30


def association_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ember_association", list(ASSOCIATION_DIVERGING))
    cmap.set_bad("white")
    return cmap


SUBTITLE_SEP = " · "


def wrap_subtitle(parts: Sequence[str], width: int = 96, sep: str = SUBTITLE_SEP) -> str:
    """Join subtitle parts, breaking BETWEEN parts so a long disclosure
    line wraps instead of running off both edges of the figure."""
    lines: List[str] = []
    current = ""
    for part in (str(p) for p in parts if p):
        candidate = f"{current}{sep}{part}" if current else part
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    # One statement longer than the line (a narrow Figure Composer cell):
    # break it at a space rather than let it run off both edges.
    out: List[str] = []
    for line in lines:
        if len(line) > width and " " in line:
            import textwrap
            out.extend(textwrap.wrap(line, width=max(int(width), 12), break_long_words=False) or [line])
        else:
            out.append(line)
    return "\n".join(out)


def _title_with_subtitle(panel, ax, title: str, *subtitle_lines: str) -> int:
    """Same convention as the plot grammar: suptitle + small gray axes
    title when there is a subtitle, a plain axes title otherwise.

    Each subtitle line is a ' · '-separated run of statements, wrapped
    here to the width of the figure it is drawn on: a subtitle that
    runs off the edge loses exactly the disclosures it exists to carry,
    and the same text has to fit a full panel and a Figure Composer
    cell alike. Returns the number of subtitle lines drawn."""
    width = max(30, int(12.0 * figure_size_inches(panel.figure)[0]))
    wrapped = [wrap_subtitle(line.split(SUBTITLE_SEP), width=width) for line in subtitle_lines if line]
    if wrapped:
        text = "\n".join(wrapped)
        panel.figure.suptitle(title)
        ax.set_title(text, fontsize=9, color=CHART_SECONDARY, fontweight="normal")
        return text.count("\n") + 1
    ax.set_title(title)
    return 0


def association_square(corr_table: pd.DataFrame, label_col: str = "Component") -> Tuple[pd.DataFrame, List[str]]:
    """The numeric square behind a displayed correlation table (one label
    column + one column per component), in the table's own row order."""
    if corr_table is None or corr_table.empty:
        return pd.DataFrame(), []
    data = corr_table.set_index(label_col) if label_col in corr_table.columns else corr_table.copy()
    data.index = [str(i) for i in data.index]
    data.columns = [str(c) for c in data.columns]
    names = [name for name in data.index if name in data.columns]
    if not names:
        return pd.DataFrame(), []
    data = data.loc[names, names].apply(pd.to_numeric, errors="coerce")
    return data, names


def format_coefficient(value: float) -> str:
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


@composable
def plot_association_heatmap(
    panel, corr_table: pd.DataFrame, title: str = "Cross-correlation", subtitle: str = "",
    cbar_label: str = "Correlation r", label_col: str = "Component",
) -> None:
    """Lower triangle of a correlation table with every coefficient
    written in its cell. The diagonal is left out (a component against
    itself is 1 by construction), so the grid is (n-1) x (n-1): rows are
    components 2..n, columns components 1..n-1.

    A pair the engine left blank (too few shared observations, or a
    constant component) is drawn HATCHED with no number -- distinguishable
    from both a coefficient of zero (gray, "0.00") and the unused upper
    triangle (empty)."""
    data, names = association_square(corr_table, label_col)
    n = len(names)
    if n < 2:
        panel.show_message("No correlation matrix")
        return
    from matplotlib.patches import Rectangle

    arr = data.to_numpy(dtype=float)
    sub = arr[1:, :-1].copy()                       # (row i+1, column j)
    rows, cols = np.indices(sub.shape)
    in_triangle = cols <= rows
    blank = in_triangle & ~np.isfinite(sub)
    masked = np.ma.array(sub, mask=~in_triangle | ~np.isfinite(sub))

    panel.figure.clear()
    size = panel.cap_square_size_inches(min(max(5.5, 0.7 * (n - 1) + 2.5), 14.0))
    panel.set_figure_size_inches(size + 0.9, size)
    ax = panel.figure.add_subplot(111)
    cmap = association_cmap()
    im = ax.imshow(masked, vmin=-1.0, vmax=1.0, cmap=cmap, interpolation="nearest")

    annotate = n <= ASSOCIATION_MAX_ANNOTATED
    font = 9 if n <= 10 else (8 if n <= 16 else 6.5)
    for r in range(n - 1):
        for c in range(r + 1):
            value = sub[r, c]
            if not np.isfinite(value):
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1.0, 1.0, facecolor="none", edgecolor=CHART_AXIS,
                    hatch="////", linewidth=0.0, gid=f"blank:{names[r + 1]}|{names[c]}",
                ))
                continue
            if not annotate:
                continue
            red, green, blue, _alpha = cmap((value + 1.0) / 2.0)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                c, r, format_coefficient(float(value)), ha="center", va="center", fontsize=font,
                color="white" if luminance < 0.42 else CHART_INK, gid=f"cell:{names[r + 1]}|{names[c]}",
            )

    ax.set_xticks(np.arange(n - 1))
    ax.set_yticks(np.arange(n - 1))
    # Short labels of a small matrix read best upright; a long or crowded
    # axis is tilted, anchored at the end of each label.
    tilt = n > 10 or max((len(str(name)) for name in names), default=0) > 6
    ax.set_xticklabels(names[:-1], rotation=45 if tilt else 0, ha="right" if tilt else "center",
                       rotation_mode="anchor", fontsize=9 if n <= 16 else 7)
    ax.set_yticklabels(names[1:], fontsize=9 if n <= 16 else 7)
    ax.set_xticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    footnotes = []
    if blank.any():
        footnotes.append(f"hatched: {int(blank.sum())} pair(s) left blank")
    if not annotate:
        footnotes.append(f"coefficients not written ({n} components): see the Correlation matrix table")
    if footnotes:
        ax.set_xlabel(SUBTITLE_SEP.join(footnotes), fontsize=8, color=CHART_SECONDARY)
    mark_cells(ax, names[1:], names[:-1], row_name=label_col, col_name=label_col, value_label=cbar_label,
               blank_text="left blank (too few shared observations, or a constant component)", triangle="lower")

    # The colorbar rides on the matrix's own box. Carved out of the axes
    # (colorbar(ax=...)) it pins the square matrix to the right edge of
    # whatever width the panel happens to have, leaving the row labels
    # stranded beside an empty half-figure.
    cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
    cbar = panel.figure.colorbar(im, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_label(cbar_label)
    subtitle_lines = _title_with_subtitle(panel, ax, title, subtitle)

    # Margins are set in inches rather than left to tight_layout(), which
    # measures neither an inset colorbar nor the box an equal-aspect
    # image actually occupies: on a large matrix it clipped the colorbar
    # labels and ran the subtitle into the title. They are fractions of
    # the figure's size, so they are recomputed whenever that size
    # changes (a resized panel, an export): see relayout_figure.
    figure = panel.figure
    label_in = 0.085 * max(len(name) for name in names) + 0.35
    has_footnotes = bool(footnotes)

    def _set_margins() -> None:
        fig_w, fig_h = figure_size_inches(figure)
        left_in = label_in
        right_in = (0.8 + 0.07 * max(fig_w - left_in, 0.0)) / 1.07      # colorbar + its labels
        top_in = 0.5 + 0.17 * subtitle_lines if subtitle_lines else 0.45
        bottom_in = (0.75 * label_in if n > 5 else 0.3) + 0.25 + (0.25 if has_footnotes else 0.0)
        figure.subplots_adjust(
            left=min(0.30, left_in / fig_w), right=1.0 - min(0.30, right_in / fig_w),
            top=1.0 - min(0.35, top_in / fig_h), bottom=min(0.30, bottom_in / fig_h),
        )

    _set_margins()
    setattr(ax, RELAYOUT_ATTR, _set_margins)
    panel.ax = ax
    panel.canvas.draw_idle()


def association_scatter_counts(values: pd.DataFrame, a: str, b: str) -> Dict[str, int]:
    """How the rows offered to a log-log pair scatter divide: drawn, left
    off because a value is zero or negative, and not reporting both."""
    if values is None or values.empty or a not in values or b not in values:
        return {"offered": 0, "both": 0, "drawn": 0, "left_off": 0, "missing": 0}
    xa, xb = numeric_plot_series(values[a]), numeric_plot_series(values[b])
    both = xa.notna() & xb.notna()
    drawn = both & (xa > 0) & (xb > 0)
    return {
        "offered": int(len(values)), "both": int(both.sum()), "drawn": int(drawn.sum()),
        "left_off": int(both.sum() - drawn.sum()), "missing": int(len(values) - both.sum()),
    }


def _mark_row_points(ax, collection, frame: pd.DataFrame, series_name: str = "") -> None:
    """Name the points of a scatter whose rows are tanks, or layers of
    tanks (tank + waste phase)."""
    if "WasteSiteId" not in frame.columns:
        return
    tanks = frame["WasteSiteId"].astype(str).tolist()
    labels = tanks
    if "WastePhase" in frame.columns:
        labels = [f"{t}, {ph}" for t, ph in zip(tanks, frame["WastePhase"].astype(str))]
    extra = [(humanize(c), frame[c].astype(str).tolist()) for c in ("WasteType", "TankFarm") if c in frame.columns]
    mark_points(ax, collection, labels, name="Tank", tank_ids=tanks, series_name=series_name, extra=extra)


@composable
def plot_association_scatter(
    panel, values: pd.DataFrame, a: str, b: str, value_label: str = "", title: str = "",
    subtitle: str = "", color_col: Optional[str] = None, unit_word: str = "row",
) -> None:
    """Log-log scatter of two components over the rows a coefficient was
    computed from, colored by `color_col` (the phase family) when given.

    A row with a zero or negative value has no position on a log axis.
    It is not dropped quietly: the subtitle states how many were left
    off, and how many rows do not report both components at all."""
    counts = association_scatter_counts(values, a, b)
    if not counts["offered"]:
        panel.show_message("No values for this pair")
        return
    plural = "" if counts["offered"] == 1 else "s"
    count_parts = [f"{counts['drawn']} of {counts['offered']} {unit_word}{plural} drawn"]
    if counts["left_off"]:
        count_parts.append(f"{counts['left_off']} left off (zero or negative: no position on a log axis)")
    if counts["missing"]:
        count_parts.append(f"{counts['missing']} do not report both")
    count_line = SUBTITLE_SEP.join(count_parts)
    if not counts["drawn"]:
        panel.show_message(f"No {unit_word} reports a positive value for both {a} and {b}.\n{count_line}")
        return

    pdf = values.copy()
    pdf[a], pdf[b] = numeric_plot_series(pdf[a]), numeric_plot_series(pdf[b])
    pdf = pdf[(pdf[a] > 0) & (pdf[b] > 0)]
    ax = panel.reset_axes()
    panel.set_figure_size_inches(7.6, 5.8)
    if color_col and color_col in pdf.columns:
        present = [str(v) for v in pd.unique(pdf[color_col].astype(str))]
        ordered = [f for f in PHASE_FAMILY_STYLE if f in present] + sorted(
            f for f in present if f not in PHASE_FAMILY_STYLE)
        for family in ordered:
            group = pdf[pdf[color_col].astype(str) == family]
            color, marker = PHASE_FAMILY_STYLE.get(family, _OTHER_FAMILY_STYLE)
            sc = ax.scatter(group[a], group[b], s=34, alpha=0.8, color=color, marker=marker,
                            edgecolors="white", linewidths=0.5, label=f"{family} ({len(group)})")
            _mark_row_points(ax, sc, group, series_name="Phase family")
        ax.legend(title="Phase family", fontsize=8, title_fontsize=8, frameon=False)
    else:
        sc = ax.scatter(pdf[a], pdf[b], s=34, alpha=0.8, color=BASE_COLOR,
                        edgecolors="white", linewidths=0.5)
        _mark_row_points(ax, sc, pdf)
    ax.set_xscale("log")
    ax.set_yscale("log")
    suffix = f" ({value_label})" if value_label else ""
    ax.set_xlabel(f"{a}{suffix}")
    ax.set_ylabel(f"{b}{suffix}")
    ax.grid(True, which="major")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _title_with_subtitle(panel, ax, title or f"{a} vs {b}", subtitle, count_line)
    panel.figure.tight_layout()
    panel.canvas.draw_idle()
