"""
figure_composer.py — combines several captured plot "recipes" (see
composer_store.ComposerItem) into one multi-panel figure with
publication-style panel labels (A, B, C, ...).

Each panel is redrawn from its own recipe -- the exact plot_helpers
function and arguments used to draw it the first time -- into its own
matplotlib SubFigure (one cell of the combined figure's grid), rather than
composited as a raster image of what it looked like at capture time. This
keeps every panel a real, live, vector Axes in the final figure: sharp at
any export size, and still carrying its own real colorbars/legends/
gridspec-based marginal plots exactly as the source plot built them --
because it's the SAME plotting code, not a picture of its output.

A SubFigure fully isolates one panel's figure.clear()/add_subplot() calls
from its siblings (confirmed empirically: clearing and rebuilding one
subfigure leaves every other subfigure in the parent figure untouched),
which is what makes reusing plot_helpers functions completely unmodified
safe here -- most of them call panel.figure.clear() as their first step to
reset for a fresh render, which would wipe out every other panel if
panel.figure were the shared top-level Figure instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, Optional, Sequence, Tuple

from compose_science import DEFAULT_LABEL_STYLE, compute_grid_shape, panel_label

PANEL_WIDTH_IN = 4.0
PANEL_HEIGHT_IN = 3.2


class _NullCanvas:
    """draw_idle() is meaningless off-screen -- every plot_helpers function
    calls panel.canvas.draw_idle() as its last step, so this just needs to
    accept and ignore the call."""

    def draw_idle(self) -> None:
        pass


class _FigureCompat:
    """SubFigure implements most of Figure's interface (add_subplot,
    add_gridspec, subplots, colorbar, suptitle, clear, set_facecolor -- all
    confirmed to work identically) but not quite all of it: get_size_inches
    and tight_layout raise AttributeError on a SubFigure, and several
    plot_helpers functions call panel.figure.get_size_inches()/
    tight_layout() directly. Rather than touch those functions (which need
    to stay identical to how they draw outside the composer), wrap
    whichever figure-like object a panel is given so those two calls
    degrade gracefully; everything else forwards straight through
    unchanged via __getattr__."""

    def __init__(self, figure) -> None:
        self._figure = figure

    def __getattr__(self, name):
        return getattr(self._figure, name)

    def get_size_inches(self):
        if hasattr(self._figure, "get_size_inches"):
            return self._figure.get_size_inches()
        return (PANEL_WIDTH_IN, PANEL_HEIGHT_IN)

    def tight_layout(self, *args, **kwargs) -> None:
        if hasattr(self._figure, "tight_layout"):
            self._figure.tight_layout(*args, **kwargs)
        # else: a SubFigure has no equivalent -- its child axes still get
        # matplotlib's normal default padding, just not auto-tightened.


class ComposerCellPanel:
    """Minimal PlotWidget-shaped adapter so any existing plot_helpers
    function can draw into one cell of a shared composed Figure -- a
    SubFigure -- instead of owning a whole standalone Figure. This is what
    lets Figure Composer reuse the exact same plotting code that draws
    each panel in its own workspace, rather than maintaining a second
    rendering path. Also used, wrapping a bare standalone Figure instead of
    a SubFigure, to render gallery thumbnails off-screen.

    A SubFigure's size is dictated by the parent figure's gridspec cell,
    not independently settable (no set_size_inches), so size-related calls
    are no-ops there; a standalone Figure (thumbnails) does support it, and
    resizing it IS worth doing so content-driven sizing (e.g. a heatmap
    wanting to be square, or taller for more bars) still shapes the
    thumbnail sensibly. Either way, the *relative* proportions plot_helpers
    functions build internally (gridspec width/height_ratios for e.g. a
    heatmap's marginal bars) work unchanged, since those are ratios, not
    absolute inches."""

    def __init__(self, figure) -> None:
        self.figure = _FigureCompat(figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas = _NullCanvas()
        # Set by _render_item when the recipe raised -- the axes then hold
        # an error placeholder, not the plot, and post-render text overrides
        # must NOT be applied to it (the full pipeline skips them too).
        self.render_failed = False
        # Whether this panel's TITLE lives on the subfigure suptitle
        # rather than the axes (grammar-rendered panels put a disclosure
        # line on the axes title instead). Set by _render_item from what
        # the recipe actually drew; read by set_panel_title.
        self.title_on_suptitle = False
        # Recorded, not acted on, when wrapping a SubFigure (see
        # set_figure_size_inches below) -- compose_figure() reads this back
        # from a throwaway pre-measurement pass so a content-heavy panel
        # (e.g. plot_barh asking for extra height because it has 40 bars)
        # can get a taller row instead of being crushed into the uniform
        # default cell size, which a SubFigure can't be individually
        # resized out of after the fact.
        self.requested_size: Optional[Tuple[float, float]] = None

    def reset_axes(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        return self.ax

    def show_message(self, message: str) -> None:
        self.reset_axes()
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                      transform=self.ax.transAxes, fontsize=10, color="0.4")

    def set_figure_size_inches(self, width: float, height: float, *, cap_to_visible: bool = True) -> None:
        self.requested_size = (width, height)
        if hasattr(self.figure, "set_size_inches"):
            self.figure.set_size_inches(width, height)

    def available_content_width_inches(self, reserve_px: int = 280):
        return None

    def available_content_height_inches(self):
        return None

    def cap_square_size_inches(self, ideal: float) -> float:
        return ideal


def _render_item(cell: ComposerCellPanel, item) -> None:
    """Replays one item's recipe into `cell`, then layers its title/axis-
    label overrides on top -- applied AFTER the recipe renders, directly on
    whichever axes the recipe itself left as the panel's "main" one (every
    plot_helpers function ends by setting panel.ax to it, even when it
    built several sub-axes internally e.g. marginal bars). This works
    generically for any recipe without each plot_helpers function needing
    to know Figure Composer exists."""
    try:
        item.render_fn(cell, *item.render_args, **item.effective_kwargs())
    except Exception as exc:
        cell.render_failed = True
        cell.show_message(f"Could not re-render this panel:\n{exc}")
        return
    ax = cell.ax
    if ax is None:
        return
    # Which slot holds THIS panel's title, decided by what the recipe just
    # drew. A grammar-rendered panel puts its title on the subfigure's
    # suptitle and its disclosure line on the axes title, so a title
    # override aimed blindly at the axes would erase the disclosures and
    # leave the heading standing (review finding).
    cell.title_on_suptitle = _has_suptitle(cell)
    if item.title_override is not None:
        set_panel_title(cell, item.title_override)
    if item.xlabel_override is not None:
        ax.set_xlabel(item.xlabel_override)
    if item.ylabel_override is not None:
        ax.set_ylabel(item.ylabel_override)


def _has_suptitle(cell) -> bool:
    figure = getattr(cell, "figure", None)
    suptitle = getattr(figure, "_suptitle", None) if figure is not None else None
    return bool(suptitle is not None and suptitle.get_text())


def set_panel_title(cell, value: str) -> None:
    """Write a panel's title into whichever slot that panel uses."""
    if getattr(cell, "title_on_suptitle", False) and getattr(cell, "figure", None) is not None:
        cell.figure.suptitle(value)
    elif cell.ax is not None:
        cell.ax.set_title(value)


# Measurement results, ONE entry per uid, each remembering the kwarg-
# overrides token it was measured under. uid is stable across replace()-
# based edits and render_args/render_kwargs are frozen at capture time
# (deep-copied, never mutated afterward), so the ONLY thing that can change
# what a recipe requests size-wise is its kwarg_overrides -- text overrides
# are applied post-render and never reach the recipe. This turns
# compose_figure's measurement pass from N low-DPI re-renders per rebuild
# into cache hits for every unedited panel (add one panel: 2N+1 renders ->
# N+1). A new token OVERWRITES the old entry rather than accumulating
# beside it -- otherwise a session stepping top_n 30->40->50->... would
# grow this module-level dict monotonically per live panel (review finding).
_measure_cache: Dict[int, Tuple[str, Optional[Tuple[float, float]]]] = {}


def _overrides_token(item) -> str:
    """Stable, order-independent cache token for kwarg_overrides. repr()
    rather than hashing the values directly because override values may be
    unhashable (lists)."""
    return repr(sorted(item.kwarg_overrides.items(), key=lambda kv: kv[0]))


def _prune_measure_cache(items: Sequence) -> None:
    live_uids = {item.uid for item in items}
    for uid in [u for u in _measure_cache if u not in live_uids]:
        del _measure_cache[uid]


def measure_preferred_size(item, *, use_cache: bool = True) -> Optional[Tuple[float, float]]:
    """Cheaply discovers whether this item's recipe asks for a content-
    driven figure size (e.g. plot_barh growing taller for more bars) by
    replaying it once into a tiny, low-DPI, throwaway Figure and reading
    back what it requested via ComposerCellPanel.requested_size. A
    SubFigure can't be resized after the fact (see _FigureCompat), so
    compose_figure needs to know this BEFORE building the real grid, to
    size that item's row/column generously enough in the first place.
    Returns None if the recipe never calls set_figure_size_inches (happy
    with the uniform default) or fails to render at all. Starts from the
    uniform default size (not some arbitrary tiny placeholder) because a
    function like plot_barh reads back "whatever the current figure width
    already is" to preserve it while only changing height -- measured from
    a 1x1in figure, that read-back would itself just be the meaningless
    number 1.0, not a sensible width."""
    token = _overrides_token(item)
    if use_cache:
        cached = _measure_cache.get(item.uid)
        if cached is not None and cached[0] == token:
            return cached[1]

    from matplotlib.figure import Figure

    fig = Figure(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN), dpi=40)
    cell = ComposerCellPanel(fig)
    try:
        item.render_fn(cell, *item.render_args, **item.effective_kwargs())
    except Exception:
        _measure_cache[item.uid] = (token, None)
        return None
    _measure_cache[item.uid] = (token, cell.requested_size)
    return cell.requested_size


@dataclass
class ComposedPanels:
    """Live handles into the most recent compose_figure() output, kept so
    per-panel edits can touch ONE panel instead of rebuilding the grid:

    - text overrides mutate the panel's main axes in place (pixel-identical
      to a full recompose by construction, because _render_item also
      applies overrides post-render with no layout pass -- see
      apply_text_override),
    - parameter edits clear and replay one SubFigure (rerender_panel),
    - only structural changes (add/remove/move, or an edit that changes a
      panel's measured preferred size) force a full compose_figure().
    """

    subfigs: Dict[int, Any] = field(default_factory=dict)
    cells: Dict[int, "ComposerCellPanel"] = field(default_factory=dict)
    measured: Dict[int, Optional[Tuple[float, float]]] = field(default_factory=dict)
    uids: Dict[int, int] = field(default_factory=dict)
    label_style: str = DEFAULT_LABEL_STYLE
    label_fontsize: int = 14

    def matches(self, items: Sequence) -> bool:
        """True when this layout still corresponds to `items` (same panels
        in the same slots) -- the precondition for any single-panel path."""
        return self.uids == {i: item.uid for i, item in enumerate(items)}


def _draw_panel_label(subfig, index: int, label_style: str, label_fontsize: int) -> None:
    label = panel_label(index, label_style)
    if label:
        # The gid names the letter for the layout pass (plot_helpers.
        # PANEL_LABEL_GID): it is never wrapped, resized or hidden.
        subfig.text(
            0.0, 0.99, label, transform=subfig.transSubfigure, fontsize=label_fontsize,
            fontweight="bold", va="top", ha="left", color="black", gid="ember-panel-label",
        )


def rerender_panel(panel, composed: ComposedPanels, index: int, item) -> None:
    """Replays one item's recipe into its existing SubFigure slot, leaving
    every sibling panel's artists untouched (SubFigure isolation). The slot
    keeps its current grid size -- callers must fall back to a full
    compose_figure() when the item's measured preferred size changed."""
    subfig = composed.subfigs.get(index)
    if subfig is None:
        return
    subfig.clear()
    subfig.set_facecolor("white")
    cell = ComposerCellPanel(subfig)
    _render_item(cell, item)
    _draw_panel_label(subfig, index, composed.label_style, composed.label_fontsize)
    composed.cells[index] = cell
    composed.uids[index] = item.uid
    panel.canvas.draw_idle()


def apply_text_override(panel, composed: ComposedPanels, index: int, item, fieldname: str) -> bool:
    """Applies one title/xlabel/ylabel override to the already-rendered
    panel IN PLACE -- zero recipe replays -- and returns True. This is
    pixel-identical to what a full recompose would produce, because the
    full pipeline (_render_item) also applies these overrides post-render
    on the same axes with no subsequent layout pass; there is nothing a
    re-render would compute differently for a text-only change.

    Returns False when the in-place path can't produce the correct result
    and the caller must re-render this one panel instead: an override
    cleared back to None means "restore whatever text the recipe itself
    draws", which only the recipe knows."""
    cell = composed.cells.get(index)
    if cell is None or cell.ax is None or cell.render_failed:
        # A failed panel's axes hold the error placeholder; the full
        # pipeline never applies overrides to it, so neither can we --
        # returning False routes this through the (equally refusing, but
        # honest) re-render path instead of drawing text over the error box.
        return False
    value = getattr(item, f"{fieldname}_override", None)
    if value is None:
        return False
    ax = cell.ax
    if fieldname == "title":
        set_panel_title(cell, value)
    elif fieldname == "xlabel":
        ax.set_xlabel(value)
    elif fieldname == "ylabel":
        ax.set_ylabel(value)
    else:
        return False
    panel.canvas.draw_idle()
    return True


def compose_figure(
    panel, items: Sequence, *, cols: int = 0, label_style: str = DEFAULT_LABEL_STYLE,
    label_fontsize: int = 14, spacing: float = 0.05,
) -> Optional[ComposedPanels]:
    """Draws `items` (composer_store.ComposerItem) into `panel` (a
    qt_widgets.PlotWidget) as a labeled grid of live subfigures, and
    returns per-panel handles (ComposedPanels) so later cosmetic/param
    edits can touch one panel instead of rebuilding this grid. cols=0
    auto-picks a near-square layout; label_style is one of
    compose_science.LABEL_STYLES. Returns None when there was nothing to
    compose."""
    n = len(items)
    if n == 0:
        # show_message() reuses panel.ax, so this must run BEFORE
        # figure.clear() below -- clearing first would orphan panel.ax
        # (removed from the figure's axes list, but the Python reference
        # still exists), leaving show_message() drawing into an axes that's
        # no longer actually part of the figure.
        show_empty = getattr(panel, "show_empty", None)
        if callable(show_empty):
            show_empty("No panels yet",
                       'Use "Send to Figure Composer" on the toolbar of any chart to add it here.')
        else:
            panel.show_message('No panels yet: use "Send to Figure Composer" on any chart to add one.')
        return None
    panel.figure.clear()
    _prune_measure_cache(items)

    rows, ncols = compute_grid_shape(n, cols)
    composed = ComposedPanels(label_style=label_style, label_fontsize=label_fontsize)

    # Each row/column starts at the uniform default and grows to fit
    # whichever item in it asked for more -- a wide heatmap or a bar chart
    # with many rows gets the room it needs instead of being crushed into
    # a fixed-size cell (which would otherwise leave e.g. its tick labels
    # illegibly overlapping).
    col_widths = [PANEL_WIDTH_IN] * ncols
    row_heights = [PANEL_HEIGHT_IN] * rows
    for i, item in enumerate(items):
        preferred = measure_preferred_size(item)
        composed.measured[i] = preferred
        if preferred is None:
            continue
        r, c = divmod(i, ncols)
        w, h = preferred
        col_widths[c] = max(col_widths[c], w)
        row_heights[r] = max(row_heights[r], h)

    panel.set_figure_size_inches(sum(col_widths), sum(row_heights))
    panel.figure.set_facecolor("white")

    subfigs = panel.figure.subfigures(
        rows, ncols, wspace=spacing, hspace=spacing + 0.08, squeeze=False,
        width_ratios=col_widths, height_ratios=row_heights,
    )
    first_ax = None
    for i, item in enumerate(items):
        r, c = divmod(i, ncols)
        subfig = subfigs[r, c]
        subfig.set_facecolor("white")
        cell = ComposerCellPanel(subfig)
        _render_item(cell, item)
        if first_ax is None:
            first_ax = cell.ax
        _draw_panel_label(subfig, i, label_style, label_fontsize)
        composed.subfigs[i] = subfig
        composed.cells[i] = cell
        composed.uids[i] = item.uid
    panel.ax = first_ax
    panel.canvas.draw_idle()
    return composed


def render_thumbnail_png(item, *, width_in: float = 1.9, height_in: float = 1.4, dpi: int = 90) -> bytes:
    """Small PNG for the Figure Composer gallery's list icon -- re-renders
    the item's recipe into a standalone throwaway Figure (not a Qt canvas)
    at thumbnail size, reusing the same ComposerCellPanel adapter (and
    override-application logic) the real composed figure uses."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    fig.set_facecolor("white")
    cell = ComposerCellPanel(fig)
    _render_item(cell, item)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    return buf.getvalue()
