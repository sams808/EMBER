"""
qt_widgets.py -- the shared Qt widgets every workspace is built from: a
matplotlib chart panel (PlotWidget), a DataFrame-backed table with a filter
box (DataFrameTableModel / DataFrameTableView), a collapsible section and a
timestamped status log.

Every page draws its charts through PlotWidget and shows its tables through
DataFrameTableView, so what these two do, every chart and table in the app
does: the chart fills its pane, answers to the pointer (tooltip, outline,
readout, click, double click, context menu: see qt_plot_interaction) and
exports at a size that does not depend on the window; the table writes
numbers and headers for a reader (qt_table_format), filters, sorts
numerically, and opens a tank on double click.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import (
    QAbstractTableModel, QEvent, QModelIndex, QObject, QSize, QSortFilterProxyModel, Qt, QTimer, Signal,
)
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu,
    QMessageBox, QSizePolicy, QTableView, QToolButton, QVBoxLayout, QWidget,
)

from qt_plot_interaction import PlotInteraction
from qt_table_format import format_cell, format_full, humanize, number_style, tank_id_of

# Kept under its old name: other modules and tests import it from here.
_format_cell = format_cell

EMPTY_TITLE = "Nothing to show yet"
EMPTY_HINT = "The chart appears here after a run."
CHART_MUTED = "#52514e"
CHART_FAINT = "#8a8983"


def _ui_palette() -> Dict[str, str]:
    """The palette the window is using right now.

    qt_theme owns the answer; until it exposes it, the active stylesheet is
    compared with the dark palette's page colour."""
    try:
        import qt_theme
    except Exception:
        return {}
    current = getattr(qt_theme, "current_palette", None)
    if callable(current):
        try:
            return dict(current())
        except Exception:
            pass
    light = dict(getattr(qt_theme, "PALETTE", {}))
    dark = dict(getattr(qt_theme, "DARK_PALETTE", {}))
    app = QApplication.instance()
    sheet = app.styleSheet() if app is not None else ""
    if dark and dark.get("bg") and dark["bg"] in sheet and light.get("bg") not in sheet:
        return dark
    return light


def _open_tank_in_window(widget: QWidget, tank_id: str) -> bool:
    """Ask the main window to open a tank. True when it has a way to."""
    window = widget.window() if widget is not None else None
    handler = getattr(window, "open_tank", None)
    if callable(handler) and tank_id:
        handler(str(tank_id))
        return True
    return False


class _PanelFigure(Figure):
    """A Figure that remembers how it was laid out and exports at a size
    that does not depend on the window.

    `tight_layout()` computes margins for the size the figure has at that
    moment. On screen the figure follows its pane, so the request is
    RECORDED here and carried out by apply_tight_layout(): once when the
    plot is finished (after its labels have been fitted, so the margins
    are measured on the final text), again after every resize, and around
    an export. A figure nobody finished is laid out at its next draw.
    Every layout ends with a measurement of what the figure draws outside
    itself, and a correction when there is anything (plot_helpers.
    contain_region): a chart is never cut by the edge of its pane.
    `savefig()` goes through a hook the owning PlotWidget installs: for
    the duration of the save the figure takes its export size, then
    returns to the pane's."""

    _ember_tight: Optional[dict] = None
    _ember_tight_pending: bool = False
    _ember_save_hook: Optional[Callable] = None
    # The figure of a pane: its layout ends with the containment pass
    # (plot_helpers.contain_region), so nothing it draws is cut by its edge.
    _ember_pane: bool = True
    _ember_contain: Optional[dict] = None
    _ember_exporting: bool = False

    def clear(self, keep_observers: bool = False) -> None:
        super().clear(keep_observers=keep_observers)
        self._ember_tight = None
        self._ember_tight_pending = False
        self._ember_contain = None
        try:
            import plot_helpers
            if plot_helpers.CONTAINMENT:
                # matplotlib keeps the subplot margins across a clear: the
                # next chart would start from the ones fitted to this one.
                left, right, bottom, top = plot_helpers._default_margins()
                wspace, hspace = plot_helpers._default_spacing()
                self.subplotpars.update(left=left, right=right, bottom=bottom, top=top,
                                        wspace=wspace, hspace=hspace)
        except Exception:
            pass

    def tight_layout(self, **kwargs) -> None:
        self._ember_tight = dict(kwargs)
        self._ember_tight_pending = True

    def apply_tight_layout(self) -> None:
        """Carry out the recorded tight_layout() for the current size."""
        self._ember_tight_pending = False
        if self._ember_tight is None:
            return
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Figure.tight_layout(self, **self._ember_tight)
            except Exception:
                pass

    def draw(self, renderer):
        # Laid out for this size and content, and contained, before every
        # draw: a figure nobody finished, a resize still being dragged, an
        # export at another size or resolution. A composed figure has one
        # SubFigure per cell, which has no layout engine of its own: each
        # cell's margins are fitted to its text there.
        try:
            import plot_helpers
            plot_helpers.contain_on_draw(self, renderer)
        except Exception:
            pass
        if self._ember_tight_pending:
            self.apply_tight_layout()
        return super().draw(renderer)

    def savefig(self, fname, **kwargs):
        hook = self._ember_save_hook
        if hook is None:
            return super().savefig(fname, **kwargs)
        with hook():
            return super().savefig(fname, **kwargs)


class _PanelCanvas(FigureCanvasQTAgg):
    """A canvas whose preferred size does not follow its figure.

    matplotlib's canvas reports the figure's current size as its size
    hint. Inside a scrolling page that hint becomes the page's preferred
    height, so a chart that once grew could never shrink again, and a
    tall figure pushed its own axis labels below the fold."""

    PREFERRED = QSize(560, 400)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(self.PREFERRED)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(*PlotWidget.MIN_CANVAS_SIZE)

    def _draw_idle(self):
        """matplotlib queues a redraw with a zero-delay timer. If the
        widget is destroyed before the timer fires (a page closed, the
        window torn down), the queued call lands on a dead C++ object and
        raises from inside the event loop. Such a draw has nothing left to
        draw on: drop it."""
        try:
            super()._draw_idle()
        except RuntimeError:
            try:
                from shiboken6 import isValid
                alive = isValid(self)
            except Exception:
                alive = False
            if alive:
                raise


class PlotWidget(QWidget):
    """A chart: toolbar, canvas, and a one-line readout.

    The canvas fills whatever room the layout gives it and the figure is
    laid out again (debounced) when that room changes. The size a plot
    function asks for through set_figure_size_inches() is the chart's
    natural size: it is what an export uses, and what the figure has while
    the widget has never been shown.

    Signals carry a dictionary describing the mark under the pointer:
    label, series, value, unit, row / column for a heat-map cell, and
    `tank_id` when the mark stands for a tank. A double click on such a
    mark asks the main window to open that tank.
    """

    markHovered = Signal(dict)
    markClicked = Signal(dict)
    markActivated = Signal(dict)

    MIN_CANVAS_SIZE = (384, 220)     # wide enough for the whole toolbar in its short wording
    READOUT_HEIGHT = 24
    COMPOSER_TEXT = "Send to Figure Composer"
    COMPOSER_TEXT_SHORT = "To composer"
    COMPACT_BELOW = 500          # px: narrower than this, the toolbar uses the short wording

    def __init__(self, parent: Optional[QWidget] = None, figsize=(6.0, 4.5), dpi: int = 100,
                 debounce_ms: int = 120):
        super().__init__(parent)
        self._default_figsize = (float(figsize[0]), float(figsize[1]))
        self._natural_size = self._default_figsize
        self._last_recipe: Optional[tuple] = None
        self._ever_shown = False
        self._pristine = True
        self._exporting = False

        self.figure = _PanelFigure(figsize=figsize, dpi=dpi)
        self.figure.set_facecolor("white")
        self.figure._ember_save_hook = self._export_size
        self.ax = self.figure.add_subplot(111)
        self._first_ax = self.ax
        self.canvas = _PanelCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(*self.MIN_CANVAS_SIZE)
        self.toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)
        self._tidy_toolbar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.coords_label = QLabel("")
        self.coords_label.setObjectName("SectionNote")
        self.coords_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.coords_label.setFixedHeight(self.READOUT_HEIGHT)
        self.coords_label.setContentsMargins(8, 0, 8, 0)
        # A long readout must never widen the window: the layout decides.
        self.coords_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addWidget(self.coords_label)

        self.interaction = PlotInteraction(self.canvas, self.figure, toolbar=self.toolbar, parent=self)
        self.interaction.set_open_tank_handler(
            self._open_tank, available=lambda: callable(getattr(self.window(), "open_tank", None)))
        self.interaction.readout_changed.connect(self.coords_label.setText)
        self.interaction.mark_hovered.connect(self.markHovered)
        self.interaction.mark_clicked.connect(self.markClicked)
        self.interaction.mark_activated.connect(self.markActivated)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._flush_redraw)
        self._pending: Optional[tuple] = None

        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.setInterval(140)
        self._relayout_timer.timeout.connect(self.relayout)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resized)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    _TOOLBAR_ICONS = {"home": "home", "back": "back", "forward": "forward", "pan": "move", "zoom": "zoom_to_rect"}

    def _tidy_toolbar(self) -> None:
        """Navigation on the left, then a spacer, then the two things a
        reader does with a finished chart: send it on, or save it."""
        tb = self.toolbar
        tb.setObjectName("PlotToolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        actions = getattr(tb, "_actions", {})
        keep = {actions.get(name) for name in self._TOOLBAR_ICONS}
        for action in list(tb.actions()):
            if action not in keep:
                tb.removeAction(action)
        if actions.get("pan") is not None:
            tb.insertSeparator(actions["pan"])
        spacer = QWidget(tb)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        tb.addWidget(spacer)
        self.composer_action = tb.addAction(self.COMPOSER_TEXT, self._send_to_composer)
        self.composer_action.setToolTip(
            "Add this chart to the Figure Composer, where charts are combined into one figure.")
        self.export_action = tb.addAction("Export…", self._export_dialog)
        self.export_action.setToolTip("Save this chart as PNG, SVG or PDF (300 dpi).")
        self._tint_toolbar_icons()

    def _tint_toolbar_icons(self) -> None:
        """Draw the navigation icons in the theme's text colour. matplotlib
        picks black or white from the widget palette, which a stylesheet
        does not change: the icons then vanish on a light toolbar."""
        actions = getattr(self.toolbar, "_actions", {})
        color = QColor(_ui_palette().get("ink", "#20242a"))
        try:
            from matplotlib import cbook
        except Exception:
            return
        for name, image in self._TOOLBAR_ICONS.items():
            action = actions.get(name)
            if action is None:
                continue
            try:
                path = cbook._get_data_path("images", f"{image}_large.png")
                pixmap = QPixmap(str(path))
            except Exception:
                continue
            if pixmap.isNull():
                continue
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
            painter.fillRect(pixmap.rect(), color)
            painter.end()
            action.setIcon(QIcon(pixmap))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        # In a narrow pane the full wording pushes "Export…" off the
        # toolbar; the short one keeps both actions in reach.
        text = self.COMPOSER_TEXT_SHORT if self.width() < self.COMPACT_BELOW else self.COMPOSER_TEXT
        if getattr(self, "composer_action", None) is not None and self.composer_action.text() != text:
            self.composer_action.setText(text)

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().changeEvent(event)
        if event.type() in (QEvent.StyleChange, QEvent.PaletteChange) and hasattr(self, "toolbar"):
            self._tint_toolbar_icons()

    # ------------------------------------------------------------------
    # Redraw / layout
    # ------------------------------------------------------------------
    def request_redraw(self, fn: Callable, *args, **kwargs) -> None:
        """Coalesce rapid-fire redraw requests into ONE redraw shortly after
        the last call. `fn` should do the actual plotting/artist updates;
        canvas.draw_idle() is called after."""
        self._pending = (fn, args, kwargs)
        self._debounce.start()

    def _flush_redraw(self) -> None:
        if self._pending is None:
            return
        fn, args, kwargs = self._pending
        self._pending = None
        fn(*args, **kwargs)
        self.canvas.draw_idle()

    def _on_canvas_resized(self, _event=None) -> None:
        if not self._exporting:
            self._relayout_timer.start()

    def relayout(self, draw: bool = True) -> None:
        """Fit the figure's contents to the size it has now: category
        labels are re-fitted (font size, thinning) and the margins are
        recomputed. Called after a render, after a resize, and around an
        export."""
        try:
            import plot_helpers
            plot_helpers.relayout_figure(self.figure)
        except Exception:
            # Layout is cosmetic: a failure here must never take a result
            # off the screen.
            pass
        if draw:
            self.canvas.draw_idle()

    def _after_render(self) -> None:
        """Called by the plot layers when a plot function has finished."""
        self._pristine = False
        self.interaction.index.invalidate()
        self.canvas.draw_idle()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self._ever_shown = True
        if self._pristine and self._untouched():
            self.show_empty()

    def _untouched(self) -> bool:
        """True while nobody has drawn on, or written to, the first axes."""
        ax = self._first_ax
        if list(self.figure.axes) != [ax] or self.ax is not ax or not ax.axison:
            return False
        return not (ax.has_data() or ax.texts or ax.patches or ax.images or ax.collections
                    or ax.get_title() or ax.get_xlabel() or ax.get_ylabel() or self.figure.get_suptitle())

    # ------------------------------------------------------------------
    # Size
    # ------------------------------------------------------------------
    def _pixel_ratio(self) -> float:
        return float(getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0)

    def _logical_dpi(self) -> float:
        """Figure dpi per logical (Qt) pixel: matplotlib multiplies the
        figure's dpi by the screen's pixel ratio."""
        return float(self.figure.dpi) / self._pixel_ratio()

    def set_figure_size_inches(self, width: float, height: float, *, cap_to_visible: bool = True) -> None:
        """State the chart's natural size.

        On screen the figure always has the size of its pane, so once the
        widget has been shown this records the request (it is what an
        export uses) and changes nothing visible. Before that, with no
        pane to fill, the figure and the canvas take the requested size.

        cap_to_visible only concerns a widget that is visible but was
        never laid out (it clamps the figure to the window, as it always
        did); the recorded natural size is never clamped."""
        width, height = max(float(width), 1.0), max(float(height), 1.0)
        # The natural size is the request itself, not what the window can
        # show: an export must not depend on how tall the window was.
        self._natural_size = (width, height)
        if self._ever_shown:
            return
        if cap_to_visible:
            max_w = self.available_content_width_inches()
            max_h = self.available_content_height_inches()
            if max_w is not None:
                width = min(width, max_w)
            if max_h is not None:
                height = min(height, max_h)
        self.figure.set_size_inches(width, height)
        w_px, h_px = self.canvas.get_width_height()
        self.canvas.resize(w_px, h_px)

    def natural_size_inches(self) -> tuple:
        return self._natural_size

    def export_size_inches(self) -> tuple:
        """The size an export is made at: the height the chart asked for,
        and the width it asked for or was laid out at, whichever is larger
        (a chart's text is wrapped to the width it was drawn at)."""
        width, height = self._natural_size
        if self._ever_shown:
            width = max(width, float(self.figure.get_size_inches()[0]))
        return (width, height)

    def available_content_height_inches(self) -> Optional[float]:
        """Vertical room for the canvas within this widget's current
        on-screen size, or None while the widget is not shown (its height
        is then a placeholder, not real room)."""
        if not self.isVisible():
            return None
        chrome_px = self.toolbar.height() + self.READOUT_HEIGHT
        return max(self.height() - chrome_px, 0) / self._logical_dpi()

    def available_content_width_inches(self, reserve_px: int = 280) -> Optional[float]:
        """Horizontal room for the canvas without widening the window: the
        window's width less a reserve for the sibling panel. None while the
        widget is not shown."""
        if not self.isVisible():
            return None
        top = self.window()
        if top is None or not top.isVisible():
            return None
        return max(top.width() - reserve_px, 0) / self._logical_dpi()

    def cap_square_size_inches(self, ideal: float) -> float:
        """The largest size, at most `ideal`, that fits both the available
        width and height: for a caller about to ask for a square figure."""
        caps = [ideal]
        avail_w = self.available_content_width_inches()
        if avail_w is not None:
            caps.append(avail_w)
        avail_h = self.available_content_height_inches()
        if avail_h is not None:
            caps.append(avail_h)
        return min(caps)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------
    def reset_axes(self):
        """Full reset before drawing a new plot into this panel: clears the
        WHOLE figure (a previous render may have left a colorbar or
        marginal axes that ax.clear() cannot reach) and restores the
        default natural size. Returns the fresh axes."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._pristine = False
        self.set_figure_size_inches(*self._default_figsize)
        return self.ax

    def clear(self, title: str = "") -> None:
        """Back to the empty state (with `title` as its sentence, if given)."""
        self.show_empty(title or EMPTY_TITLE, "" if title else EMPTY_HINT)

    def show_empty(self, title: str = EMPTY_TITLE, hint: str = EMPTY_HINT) -> None:
        """Nothing drawn yet: a quiet sentence instead of bare axes."""
        self._last_recipe = None
        ax = self.reset_axes()
        ax.axis("off")
        ax.text(0.5, 0.53 if hint else 0.5, title, ha="center", va="bottom" if hint else "center",
                transform=ax.transAxes, fontsize=11.5, fontweight="semibold", color=CHART_MUTED,
                wrap=True, gid="ember-empty-title")
        if hint:
            ax.text(0.5, 0.49, hint, ha="center", va="top", transform=ax.transAxes, fontsize=9.5,
                    color=CHART_FAINT, wrap=True, gid="ember-empty-hint")
        self.interaction.index.invalidate()
        self.canvas.draw_idle()

    def show_message(self, message: str) -> None:
        """Blank the plot and show a centered message (e.g. "no data")
        instead of a stale or empty axes."""
        self._last_recipe = None
        ax = self.reset_axes()
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                transform=ax.transAxes, fontsize=11, color=CHART_MUTED)
        self.interaction.index.invalidate()
        self.canvas.draw_idle()

    def suggested_caption(self) -> str:
        """The figure's suptitle if it has one, otherwise the main axes'
        own title."""
        # As the plot wrote it: on a narrow pane the layout may have wrapped
        # the drawn title over several lines.
        try:
            from plot_helpers import full_text
        except Exception:
            def full_text(artist):
                return artist.get_text() if artist is not None else ""
        suptitle = full_text(getattr(self.figure, "_suptitle", None))
        if suptitle:
            return suptitle
        if self.ax is not None:
            title = getattr(self.ax, "title", None)
            return (full_text(title) if hasattr(title, "get_text") else self.ax.get_title()) or ""
        return ""

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    @contextmanager
    def _export_size(self):
        """For the duration of a save, give the figure its export size."""
        if self._exporting or not self._ever_shown:
            yield
            return
        live = tuple(float(v) for v in self.figure.get_size_inches())
        target = self.export_size_inches()
        if abs(live[0] - target[0]) < 0.02 and abs(live[1] - target[1]) < 0.02 and not self._shows_enlarge_message():
            yield
            return
        self._exporting = True
        # A file holds the chart, never the "enlarge the window" line a
        # small pane shows in its place.
        self.figure._ember_exporting = True
        try:
            self.figure.set_size_inches(*target, forward=False)
            self.relayout(draw=False)
            yield
        finally:
            self.figure._ember_exporting = False
            self.figure.set_size_inches(*live, forward=False)
            self.relayout(draw=False)
            self._exporting = False
            self.canvas.draw_idle()

    def _shows_enlarge_message(self) -> bool:
        try:
            import plot_helpers
            return bool(plot_helpers.showing_enlarge_message(self.figure, cells=True))
        except Exception:
            return False

    def export_at_size_cm(self, path: str, width_cm: float, height_cm: float, dpi: int = 300) -> None:
        """Export the current figure at an exact physical size."""
        w_in, h_in = width_cm / 2.54, height_cm / 2.54
        old_size = tuple(float(v) for v in self.figure.get_size_inches())
        was_exporting = self._exporting
        self._exporting = True
        self.figure._ember_exporting = True
        try:
            self.figure.set_size_inches(w_in, h_in, forward=False)
            self.relayout(draw=False)
            self.figure.savefig(path, dpi=dpi)
        finally:
            self.figure._ember_exporting = False
            self.figure.set_size_inches(*old_size, forward=False)
            self.relayout(draw=False)
            self._exporting = was_exporting
            self.canvas.draw_idle()

    def export_figure(self, path: str, dpi: int = 300) -> None:
        """Save the chart at its export size; the format follows the suffix."""
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")

    def _export_dialog(self) -> None:
        if self._last_recipe is None and self._untouched():
            QMessageBox.information(self, "Export chart", "There is no chart to export yet.")
            return
        stem = "".join(c if (c.isalnum() or c in " -_") else "_" for c in self.suggested_caption()).strip()
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export chart", f"{stem[:80] or 'chart'}.png", "PNG image (*.png);;SVG vector (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".svg" if "svg" in chosen.lower() else (".pdf" if "pdf" in chosen.lower() else ".png")
        try:
            self.export_figure(path, dpi=300)
        except Exception as exc:
            QMessageBox.warning(self, "Export chart", f"The chart could not be saved.\n{exc}")

    # ------------------------------------------------------------------
    # Figure Composer
    # ------------------------------------------------------------------
    def _set_last_recipe(self, render_fn, args: tuple, kwargs: dict) -> None:
        """Remembers how this panel's current content was drawn (the plot
        function plus the exact arguments) so "Send to Figure Composer"
        can replay the same call later instead of capturing a picture.
        Set by plot_helpers.composable-wrapped functions after a render."""
        self._last_recipe = (render_fn, args, kwargs)

    def _send_to_composer(self) -> None:
        from composer_store import store
        if self._last_recipe is None:
            QMessageBox.information(
                self, "Figure Composer", "Nothing to add yet. Run a plot on this panel first.",
            )
            return
        render_fn, args, kwargs = self._last_recipe
        caption = self.suggested_caption()
        store.add_recipe(render_fn, args, kwargs, caption, source=caption)

    # ------------------------------------------------------------------
    # Tanks
    # ------------------------------------------------------------------
    def _open_tank(self, tank_id: str) -> None:
        _open_tank_in_window(self, tank_id)


class CollapsibleSection(QWidget):
    """A clickable '#SectionTitle'-style header that expands/collapses a
    content area below it -- the building block every workspace uses to
    hide its long tail of rarely-needed controls (min-overlap thresholds,
    plot-count caps, ...) behind one click instead of showing everything
    at equal visual weight all the time. Collapsed by default; callers add
    their own rows to content_layout()."""

    toggled = Signal(bool)

    def __init__(self, title: str = "Advanced options", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("CollapsibleHeader")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.clicked.connect(self._on_toggled)
        outer.addWidget(self.toggle_button)

        self.content = QWidget(self)
        self.content.setObjectName("CollapsibleContent")
        self.content.setVisible(False)
        self._content_layout = QVBoxLayout(self.content)
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(6)
        outer.addWidget(self.content)

    def content_layout(self) -> QVBoxLayout:
        """Add rows (e.g. QHBoxLayout instances via addLayout) to this."""
        return self._content_layout

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(expanded)
        self._on_toggled(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self.content.setVisible(checked)
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggled.emit(checked)


class StatusLogger(QObject):
    """Timestamped, in-memory app log. HanfordDataset takes `.log` as its
    logger callback; the Debug/Export workspace shows `full_text()` and can
    connect to `logged` for a live-updating view."""

    logged = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._lines: List[str] = []

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self._lines.append(line)
        print(line)
        self.logged.emit(line)

    def full_text(self) -> str:
        return "\n".join(self._lines)

    def clear(self) -> None:
        self._lines.clear()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

SORT_ROLE = Qt.UserRole          # the raw value of a cell (what sorting compares)
NAME_ROLE = Qt.UserRole + 1      # a header's original column name

# Columns whose value names a tank, in the order they are tried.
TANK_COLUMNS = ("WasteSiteId", "PartnerTank", "Tank", "Name", "BaseTank")


def _is_number_column(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype))


class DataFrameTableModel(QAbstractTableModel):
    """Read-only Qt table model backed by a pandas DataFrame.

    The DataFrame stays exactly what the caller gave (same column names,
    full precision): `dataframe()` is what code and exports read. Only
    what is DISPLAYED is written for a reader -- formatted numbers,
    right-aligned, under a readable header whose tooltip is the original
    column name. Sorting re-orders the DataFrame itself, by value."""

    def __init__(self, df: Optional[pd.DataFrame] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self._numeric: List[bool] = []
        self._styles: List[Optional[dict]] = []
        self._filter_text = ""
        self._mask: Optional[np.ndarray] = None
        self._haystack: Optional[Dict[int, np.ndarray]] = None
        self._profile()

    def _profile(self) -> None:
        self._numeric = [_is_number_column(self._df.iloc[:, i]) for i in range(len(self._df.columns))]
        # One way of writing numbers per column (see number_style).
        self._styles = [
            number_style(self._df.iloc[:, i].to_numpy())
            if self._numeric[i] and pd.api.types.is_float_dtype(self._df.iloc[:, i].dtype) else None
            for i in range(len(self._df.columns))
        ]
        self._haystack = None
        self._refresh_mask()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def set_dataframe(self, df: pd.DataFrame, filter_text: Optional[str] = None) -> None:
        """Replace the data. `filter_text`, when given, replaces the filter
        in the same pass (the rows are read once, not twice)."""
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        if filter_text is not None:
            self._filter_text = filter_text.strip()
        self._profile()
        self.endResetModel()

    def column_name(self, section: int) -> str:
        return str(self._df.columns[section])

    def column_style(self, section: int) -> Optional[dict]:
        return self._styles[section] if 0 <= section < len(self._styles) else None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.TextAlignmentRole:
            numeric = self._numeric[index.column()] if index.column() < len(self._numeric) else False
            return int((Qt.AlignRight if numeric else Qt.AlignLeft) | Qt.AlignVCenter)
        if role not in (Qt.DisplayRole, Qt.ToolTipRole, SORT_ROLE):
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return format_cell(value, self.column_style(index.column()))
        if role == Qt.ToolTipRole:
            return format_full(value)
        if isinstance(value, np.generic):
            return value.item()
        return value if isinstance(value, (str, int, float, bool)) else format_full(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if orientation == Qt.Horizontal:
            if not (0 <= section < len(self._df.columns)):
                return None
            name = str(self._df.columns[section])
            if role == Qt.DisplayRole:
                return humanize(name)
            if role == NAME_ROLE:
                return name
            if role == Qt.ToolTipRole:
                label = humanize(name)
                return name if label == name else f"{label}\nColumn: {name}"
            if role == Qt.TextAlignmentRole:
                numeric = self._numeric[section] if section < len(self._numeric) else False
                return int((Qt.AlignRight if numeric else Qt.AlignLeft) | Qt.AlignVCenter)
            return None
        if role != Qt.DisplayRole or not (0 <= section < len(self._df.index)):
            return None
        return str(self._df.index[section])

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if self._df.empty or column < 0 or column >= len(self._df.columns):
            return
        ascending = order == Qt.AscendingOrder
        self.layoutAboutToBeChanged.emit()
        key = self._df.iloc[:, column].reset_index(drop=True)
        try:
            ranked = key.sort_values(ascending=ascending, kind="mergesort").index.to_numpy()
        except TypeError:
            # A column of mixed types (numbers and text) has no natural
            # order: compare as text rather than refuse to sort.
            ranked = key.map(format_full).sort_values(ascending=ascending, kind="mergesort").index.to_numpy()
        self._df = self._df.iloc[ranked].reset_index(drop=True)
        # The rows moved, their content did not: carry the filter's state
        # along instead of reading every cell again.
        if self._haystack:
            self._haystack = {position: text[ranked] for position, text in self._haystack.items()}
        if self._mask is not None:
            self._mask = self._mask[ranked]
        self.layoutChanged.emit()

    # -- filtering --------------------------------------------------------
    def filter_text(self) -> str:
        return self._filter_text

    def set_filter_text(self, text: str) -> None:
        """Keep the rows in which every word of `text` occurs, in any
        column, regardless of case. The view's proxy reads filter_mask()."""
        self._filter_text = (text or "").strip()
        self._refresh_mask()

    def filter_mask(self) -> Optional[np.ndarray]:
        """One boolean per row, or None when no filter is set."""
        return self._mask

    def _column_text(self, position: int) -> np.ndarray:
        """One column as an array of lower-case text, cached until the
        data changes. Built per column and on demand: turning every number
        of a large table into text is the expensive part, and a word with
        no digit in it never needs the numeric columns at all."""
        if self._haystack is None:
            self._haystack = {}
        cached = self._haystack.get(position)
        if cached is None:
            column = self._df.iloc[:, position]
            if self._numeric[position]:
                try:
                    text = column.to_numpy().astype("U32")       # the value as data, in full
                    text[text == "nan"] = ""
                except (TypeError, ValueError):
                    text = np.asarray([format_full(v) for v in column], dtype=str)
            else:
                # Text, booleans, dates, lists: as the table displays them.
                text = np.asarray([format_cell(v) for v in column], dtype=str)
            cached = self._haystack[position] = np.char.lower(text) if len(text) else text
        return cached

    def _refresh_mask(self) -> None:
        words = [w for w in self._filter_text.lower().split() if w]
        if not words or self._df.empty:
            self._mask = None
            return
        mask = np.ones(len(self._df), dtype=bool)
        for word in words:
            # "1,234" as displayed is 1234 in the data.
            needles = {word, word.replace(",", "")} - {""}
            numeric_needle = any(ch.isdigit() for ch in word) or word in ("true", "false", "nan", "-", ".")
            hit = np.zeros(len(self._df), dtype=bool)
            for position in range(len(self._df.columns)):
                if self._numeric[position] and not numeric_needle:
                    continue
                text = self._column_text(position)
                if not len(text):
                    continue
                for needle in needles:
                    hit |= np.char.find(text, needle) >= 0
                if hit.all():
                    break
            mask &= hit
        self._mask = mask


class _FilterProxy(QSortFilterProxyModel):
    """Shows the rows the model's filter keeps. Sorting is the model's job
    (it re-orders the DataFrame, so `dataframe()` reads as displayed); the
    proxy only passes the request on."""

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        mask = self.sourceModel().filter_mask()
        if mask is None or source_row >= len(mask):
            return True
        return bool(mask[source_row])

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        self.sourceModel().sort(column, order)

    def refresh(self) -> None:
        self.invalidate()


class DataFrameTableView(QWidget):
    """A DataFrame as a table a person can read and search.

    A filter box and a row count above a QTableView. Headers are readable
    labels (the column name is in the header's tooltip), numbers are
    formatted and right-aligned, sorting is by value, an unnamed pandas
    index is not shown, and an empty table says so. Right click copies a
    cell or the selection at full precision, or exports CSV. A double
    click on a row that names a tank opens that tank.

    The widget stands where a QTableView used to: `model()` is the
    DataFrameTableModel, and indexes given to or returned by this class
    (`currentIndex`, `setCurrentIndex`, the `doubleClicked` / `clicked`
    signals) are MODEL indexes, whose row is a row of `dataframe()`, with
    or without a filter. Anything else is forwarded to the inner view
    (`table`).
    """

    doubleClicked = Signal(QModelIndex)
    clicked = Signal(QModelIndex)
    activated = Signal(QModelIndex)
    tankActivated = Signal(str)

    MAX_COLUMN_WIDTH = 280
    MIN_COLUMN_WIDTH = 56
    SAMPLE_ROWS = 40

    def __init__(self, parent: Optional[QWidget] = None, *, title: str = "",
                 max_rows_display: Optional[int] = None):
        super().__init__(parent)
        self.title = title
        self.max_rows_display = max_rows_display
        self._total_rows = 0
        self._empty_title = "Nothing to show yet"
        self._empty_hint = "Rows appear here after a run."

        self._model = DataFrameTableModel(parent=self)
        self._proxy = _FilterProxy(self)
        self._proxy.setSourceModel(self._model)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.filter_bar = QWidget(self)
        self.filter_bar.setObjectName("TableFilterBar")
        bar = QHBoxLayout(self.filter_bar)
        bar.setContentsMargins(8, 6, 8, 6)
        bar.setSpacing(8)
        self.filter_edit = QLineEdit(self.filter_bar)
        self.filter_edit.setObjectName("SearchBox")
        self.filter_edit.setPlaceholderText("Filter rows")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip("Show only the rows that contain this text, in any column (Ctrl+F).")
        self.filter_edit.setMaximumWidth(280)
        self.count_label = QLabel("", self.filter_bar)
        self.count_label.setObjectName("SectionNote")
        self.count_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar.addWidget(self.filter_edit, 1)
        bar.addWidget(self.count_label, 1)
        outer.addWidget(self.filter_bar)

        self.table = QTableView(self)
        self.table.setModel(self._proxy)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setMinimumSectionSize(self.MIN_COLUMN_WIDTH)
        self.table.setSortingEnabled(True)
        header.setSortIndicator(-1, Qt.AscendingOrder)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.doubleClicked.connect(self._on_double_clicked)
        self.table.clicked.connect(lambda index: self.clicked.emit(self._to_source(index)))
        self.table.activated.connect(lambda index: self.activated.emit(self._to_source(index)))
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        outer.addWidget(self.table, 1)

        from qt_components import EmptyState
        self.empty_state = EmptyState(self._empty_title, self._empty_hint, parent=self.table.viewport())
        self.empty_state.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(160)
        self._filter_timer.timeout.connect(self._apply_filter)
        self.filter_edit.textChanged.connect(lambda _text: self._filter_timer.start())
        self.filter_edit.returnPressed.connect(self._apply_filter)
        self.filter_edit.installEventFilter(self)
        self._refresh_chrome()

    # -- the QTableView this widget stands in for ---------------------------
    def __getattr__(self, name: str):
        # Only reached for names this class does not define: pass them to
        # the inner view, so page code written against QTableView works.
        table = self.__dict__.get("table")
        if table is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(table, name)

    def model(self) -> DataFrameTableModel:
        return self._model

    def proxy_model(self) -> QSortFilterProxyModel:
        return self._proxy

    def selectionModel(self):  # noqa: N802 (Qt naming)
        return self.table.selectionModel()

    def horizontalHeader(self) -> QHeaderView:  # noqa: N802
        return self.table.horizontalHeader()

    def verticalHeader(self) -> QHeaderView:  # noqa: N802
        return self.table.verticalHeader()

    def viewport(self) -> QWidget:
        return self.table.viewport()

    def sortByColumn(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:  # noqa: N802
        self.table.sortByColumn(column, order)

    def _to_source(self, index: QModelIndex) -> QModelIndex:
        if index is None or not index.isValid():
            return QModelIndex()
        return self._proxy.mapToSource(index) if index.model() is self._proxy else index

    def _to_proxy(self, index: QModelIndex) -> QModelIndex:
        if index is None or not index.isValid():
            return QModelIndex()
        return self._proxy.mapFromSource(index) if index.model() is self._model else index

    def currentIndex(self) -> QModelIndex:  # noqa: N802
        return self._to_source(self.table.currentIndex())

    def setCurrentIndex(self, index: QModelIndex) -> None:  # noqa: N802
        self.table.setCurrentIndex(self._to_proxy(index))

    def selectedIndexes(self) -> List[QModelIndex]:  # noqa: N802
        return [self._to_source(i) for i in self.table.selectedIndexes()]

    def selectRow(self, row: int) -> None:  # noqa: N802
        """Select a row of `dataframe()` (hidden by the filter: no-op)."""
        index = self._proxy.mapFromSource(self._model.index(row, 0))
        if index.isValid():
            self.table.selectRow(index.row())

    # -- data ---------------------------------------------------------------
    def set_dataframe(self, df: pd.DataFrame) -> None:
        df = df if df is not None else pd.DataFrame()
        self._total_rows = len(df)
        if self.max_rows_display is not None and len(df) > self.max_rows_display:
            df = df.head(self.max_rows_display)
        self._filter_timer.stop()
        self._model.set_dataframe(df, filter_text=self.filter_edit.text())
        # A new result arrives in the page's own order: no column is "the
        # sorted one" until the reader clicks a header.
        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.table.verticalHeader().setVisible(df.index.name is not None)
        self._autosize_columns()
        self._refresh_chrome()

    def dataframe(self) -> pd.DataFrame:
        """Every row held by the table (filtered or not), as sorted."""
        return self._model.dataframe()

    def visible_dataframe(self) -> pd.DataFrame:
        """The rows the filter keeps, in display order."""
        mask = self._model.filter_mask()
        df = self._model.dataframe()
        return df if mask is None else df[mask]

    def set_empty_text(self, title: str, hint: str = "") -> None:
        """What the table says while it has no rows."""
        self._empty_title, self._empty_hint = title, hint
        self._refresh_chrome()

    def set_filter_text(self, text: str) -> None:
        self.filter_edit.setText(text)
        self._apply_filter()

    def filter_text(self) -> str:
        return self._model.filter_text()

    def _apply_filter(self) -> None:
        self._filter_timer.stop()
        text = self.filter_edit.text()
        if text.strip() == self._model.filter_text() and self._model.filter_mask() is not None:
            return
        self._model.set_filter_text(text)
        self._proxy.refresh()
        self._refresh_chrome()

    def _refresh_chrome(self) -> None:
        """The row count beside the filter, and the empty state."""
        held = self._model.rowCount()
        shown = self._proxy.rowCount()
        filtered = self._model.filter_mask() is not None
        if filtered:
            text = f"{shown:,} of {held:,} rows"
        else:
            text = f"{held:,} row{'' if held == 1 else 's'}"
        if self._total_rows > held:
            text += f"  (first {held:,} of {self._total_rows:,})"
            self.count_label.setToolTip(
                f"This table holds {self._total_rows:,} rows and displays the first {held:,}. "
                "The page's export writes what the page states.")
        else:
            self.count_label.setToolTip("")
        self.count_label.setText(text if held or filtered else "")
        self.filter_edit.setEnabled(bool(held) or bool(self.filter_edit.text()))
        if shown:
            self.empty_state.hide()
            return
        if held and filtered:
            self.empty_state.set_text("No rows match the filter", "Clear the filter to see every row.")
        else:
            self.empty_state.set_text(self._empty_title, self._empty_hint)
        self.empty_state.setGeometry(self.table.viewport().rect())
        self.empty_state.show()
        self.empty_state.raise_()

    def _autosize_columns(self) -> None:
        """Fit each column to its header and to a SAMPLE of its cells, up
        to a cap. Measuring every cell (what resizeColumnsToContents does)
        costs seconds on a table of tens of thousands of rows."""
        df = self._model.dataframe()
        n_cols = len(df.columns)
        if not n_cols:
            return
        header = self.table.horizontalHeader()
        # The stylesheet's font is only known once the widget is polished.
        self.table.ensurePolished()
        header.ensurePolished()
        cell_metrics = QFontMetrics(self.table.font())
        header_font = header.font()
        header_font.setBold(True)
        header_metrics = QFontMetrics(header_font)
        n_rows = len(df)
        sample = min(n_rows, self.SAMPLE_ROWS if n_cols <= 60 else 12)
        if n_rows > sample:
            rows = np.unique(np.linspace(0, n_rows - 1, sample).astype(int))
        else:
            rows = np.arange(n_rows)
        for col in range(n_cols):
            label = str(self._model.headerData(col, Qt.Horizontal, Qt.DisplayRole) or "")
            # Headers are upper-cased and letter-spaced by the theme: allow for it.
            width = int(header_metrics.horizontalAdvance(label.upper()) * 1.12) + 34
            style = self._model.column_style(col)
            for row in rows:
                text = format_cell(df.iat[int(row), col], style)
                if text:
                    width = max(width, cell_metrics.horizontalAdvance(text[:60]) + 30)
            header.resizeSection(col, max(self.MIN_COLUMN_WIDTH, min(width, self.MAX_COLUMN_WIDTH)))

    # -- events ---------------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        kind = event.type()
        table = self.__dict__.get("table")
        if table is None:
            return False
        if obj is table.viewport() and kind == QEvent.Resize:
            empty = self.__dict__.get("empty_state")
            if empty is not None:
                empty.setGeometry(table.viewport().rect())
        elif kind in (QEvent.ShortcutOverride, QEvent.KeyPress) and obj in (table, self.filter_edit):
            if event.matches(QKeySequence.Find):
                # Claim Ctrl+F while the table has the focus, whatever the
                # window binds it to.
                event.accept()
                if kind == QEvent.KeyPress:
                    self.focus_filter()
                return True
            if obj is self.filter_edit and kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                if self.filter_edit.text():
                    self.filter_edit.clear()
                    self._apply_filter()
                table.setFocus()
                return True
        return False

    def focus_filter(self) -> None:
        self.filter_edit.setFocus(Qt.ShortcutFocusReason)
        self.filter_edit.selectAll()

    def _on_double_clicked(self, proxy_index: QModelIndex) -> None:
        index = self._to_source(proxy_index)
        has_listener = self._has_external_listener()
        self.doubleClicked.emit(index)
        tank = self.tank_id_at(index)
        if tank:
            self.tankActivated.emit(tank)
            if not has_listener:
                # A page that handles the double click itself keeps its
                # own behaviour; otherwise the row opens its tank.
                _open_tank_in_window(self, tank)

    def _has_external_listener(self) -> bool:
        try:
            from PySide6.QtCore import SIGNAL
            return self.receivers(SIGNAL("doubleClicked(QModelIndex)")) > 0
        except Exception:
            return False

    def tank_id_at(self, index: QModelIndex) -> Optional[str]:
        """The tank a row names: from the clicked cell when it names one,
        otherwise from the row's tank column."""
        df = self._model.dataframe()
        if not index.isValid() or not (0 <= index.row() < len(df)):
            return None
        columns = [str(c) for c in df.columns]
        clicked = columns[index.column()] if 0 <= index.column() < len(columns) else None
        order = ([clicked] if clicked in TANK_COLUMNS else []) + [c for c in TANK_COLUMNS if c in columns]
        for name in order:
            value = df.iat[index.row(), columns.index(name)]
            tank = tank_id_of(value)
            if tank:
                return tank
            if name == "WasteSiteId" and value is not None and str(value).strip():
                return str(value)
        if clicked is not None:
            return tank_id_of(df.iat[index.row(), index.column()])
        return None

    # -- context menu -----------------------------------------------------------
    def build_menu(self, index: QModelIndex) -> QMenu:
        """The right-click menu for a model index (invalid: the table)."""
        menu = QMenu(self)
        df = self._model.dataframe()
        if index.isValid() and 0 <= index.row() < len(df):
            value = df.iat[index.row(), index.column()]
            menu.addAction("Copy cell", lambda v=value: QApplication.clipboard().setText(format_full(v)))
        menu.addAction("Copy selection", self._copy_selection)
        menu.addSeparator()
        if self._model.filter_mask() is not None:
            menu.addAction("Export filtered rows as CSV…", lambda: self._export_csv(filtered=True))
            menu.addAction("Export all rows as CSV…", self._export_csv)
        else:
            menu.addAction("Export CSV…", self._export_csv)
        tank = self.tank_id_at(index) if index.isValid() else None
        if tank and callable(getattr(self.window(), "open_tank", None)):
            menu.addSeparator()
            menu.addAction(f"Open tank {tank}", lambda t=tank: _open_tank_in_window(self, t))
        return menu

    def _show_context_menu(self, pos) -> None:
        index = self._to_source(self.table.indexAt(pos))
        self.build_menu(index).exec(self.table.viewport().mapToGlobal(pos))

    def _copy_selection(self) -> None:
        """The selected cells (or the whole table), tab-separated, with the
        original column names and full precision."""
        df = self.dataframe()
        if df.empty:
            return
        indexes = self.selectedIndexes()
        if not indexes:
            text = self.visible_dataframe().to_csv(sep="\t", index=False)
        else:
            rows = sorted({i.row() for i in indexes})
            cols = sorted({i.column() for i in indexes})
            text = df.iloc[rows, cols].to_csv(sep="\t", index=False)
        QApplication.clipboard().setText(text)

    def _export_csv(self, filtered: bool = False) -> None:
        df = self.visible_dataframe() if filtered else self.dataframe()
        if df.empty:
            QMessageBox.information(self, "Export CSV", "Nothing to export yet.")
            return
        name = f"{self.title or 'table'}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", name, "CSV (*.csv)")
        if path:
            df.to_csv(path, index=False)
