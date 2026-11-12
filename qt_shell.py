"""
qt_shell.py -- the Ember application shell.

The frame every workspace lives in:

    menu bar          File, Data, View, Help
    application header (AppHeader)
                      mark, product name, what is loaded, global search,
                      Help, theme toggle
    sidebar           one row per workspace, grouped, with icons
    scope strip       the global filter (qt_scope), at a fixed height
    page area         a scroll area around the stack of workspaces
    status bar        message, busy indicator, dataset summary, version

Rules the shell keeps
- Nav row -> page is resolved by NAME (_pages_by_nav), never by index.
- The page area never dictates the size of the window: the stack sits in
  a scroll area and reports only its current page's minimum size, capped
  in width. A page that asks for more scrolls; the window does not grow.
- Work is done for the page on screen. A scope change or a new dataset
  reaches the visible workspace at once; the hidden ones are marked stale
  and brought up to date when they are shown (see _bring_up_to_date).

API for workspaces (reach it through getattr(app_window, name, None), so a
page also works with a bare QMainWindow in tests):
    open_tank(tank_id)        show one tank in the Tank Explorer
    open_page(nav_name)       switch workspace
    open_element(subject)     search one element / analyte in Element Explorer
    ask_in_query(kind, subject, basis)
    set_busy(text or None)    busy indicator and text in the status bar
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QScrollArea, QSizePolicy,
    QStackedWidget, QStyle, QStyledItemDelegate, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from data_model import HanfordDataset
from qt_correlations import CorrelationsPage
from qt_debug import DebugPage
from qt_explorer import ExplorerPage
from qt_figure_composer import FigureComposerPage
from qt_heatmap import HeatmapPage
from qt_help import (
    ABOUT_HTML, APP_DISPLAY_NAME, APP_NAME, APP_SUBTITLE, APP_VERSION, GettingStartedDialog,
    HelpDialog, asset_path,
)
from qt_icons import WORKSPACE_ICONS, ember_mark_pixmap, icon, paint_icon
from qt_overview import OverviewPage
from qt_query import QueryPage
from plot_grammar import MARK_BY_LABEL
from qt_scope import SCOPE_STRIP_HEIGHT, GlobalScope, GlobalScopeBar
from qt_search import (
    KIND_ANALYTE, KIND_ELEMENT, KIND_TANK, KIND_WORKSPACE, GlobalSearch, SearchHit, SearchIndex,
)
from qt_tank_attributes import TankAttributesPage
from qt_tank_explorer import TankExplorerPage
from qt_theme import current_palette
from qt_vitrification import VitrificationPage
from query_templates import TEMPLATES
from qt_widgets import StatusLogger
from qt_worker import run_in_thread

NAV_OVERVIEW = "Overview"
NAV_QUERY = "Query"
NAV_EXPLORER = "Element Explorer"
NAV_TANK_ATTRS = "Tank Attributes"
NAV_TANK_EXPLORER = "Tank Explorer"
NAV_HEATMAPS = "Heatmaps"
NAV_CORRELATIONS = "Correlations"
NAV_VITRIFICATION = "Vitrification"
NAV_FIGURE_COMPOSER = "Figure Composer"

NAV_ITEMS = [
    NAV_OVERVIEW, NAV_QUERY, NAV_EXPLORER, NAV_TANK_ATTRS, NAV_TANK_EXPLORER,
    NAV_HEATMAPS, NAV_CORRELATIONS, NAV_VITRIFICATION, NAV_FIGURE_COMPOSER,
]

# One-line, plain-language "what is this for", shown on hover.
NAV_DESCRIPTIONS: Dict[str, str] = {
    NAV_OVERVIEW: "Start here. The site at a glance: headline figures, the tank-farm map coloured by any component, a ranking of tanks, and the dataset audit tables.",
    NAV_QUERY: "Ask any combined question: an element, analyte or basket, in kg, Ci, grams, wt% or g/L, grouped any way, with the unit bridging stated.",
    NAV_EXPLORER: "Search one element or analyte: which tanks hold it, what else tends to be there, typical composition.",
    NAV_TANK_ATTRS: "Browse tank engineering data (type, capacity, integrity, status) joined from the attributes file.",
    NAV_TANK_EXPLORER: "One tank as a profile (layers, headline figures, full composition, raw rows), or several chosen tanks side by side.",
    NAV_HEATMAPS: "Tank by element inventory as a color matrix, stoichiometrically apportioned: abundance patterns across the whole dataset at once.",
    NAV_CORRELATIONS: "Which components tend to occur together, statistically: cross-correlation on a concentration basis (scoped), quick scan, kg-only association workbench, PCA and clustering.",
    NAV_VITRIFICATION: "Screen tanks for glass formulation: screening score, oxide chemistry, candidate search, blend partners, and published glass loading rules.",
    NAV_FIGURE_COMPOSER: "Combine plots captured from any workspace into one labeled, publication-ready multi-panel figure.",
}

# Captions painted above the first row of each group. The rows themselves
# stay one per NAV_ITEMS entry, in NAV_ITEMS order: the caption is drawn by
# the item delegate, so no row index ever shifts.
NAV_GROUPS = [
    ("Explore", [NAV_OVERVIEW, NAV_QUERY, NAV_EXPLORER]),
    ("Tanks", [NAV_TANK_ATTRS, NAV_TANK_EXPLORER]),
    ("Analyze", [NAV_HEATMAPS, NAV_CORRELATIONS, NAV_VITRIFICATION]),
    ("Report", [NAV_FIGURE_COMPOSER]),
]
NAV_GROUP_OF_FIRST: Dict[str, str] = {names[0]: caption for caption, names in NAV_GROUPS}

SIDEBAR_WIDTH = 216
SIDEBAR_COLLAPSED_WIDTH = 56
HEADER_HEIGHT = 56
DEFAULT_SIZE = (1440, 900)
MINIMUM_SIZE = (1100, 700)
#: The widest minimum a page may impose on the page area: the width of the
#: page area at the default window size, less the room a vertical scroll
#: bar takes when the page is also too tall (otherwise that bar alone would
#: bring a horizontal one with it). Beyond it the page is laid out at this
#: width and its longest item gives way, rather than the window or a
#: horizontal scroll bar growing without bound.
PAGE_MAX_MIN_WIDTH = DEFAULT_SIZE[0] - SIDEBAR_WIDTH - 12

_EXPORT_METHODS = (
    "export_current", "export_tables", "export_figure", "_export", "_export_bundle",
    "_export_tables", "_export_inputs", "_export_debug_bundle",
)
_DATE_RE = re.compile(r"^(\d{4})[/-](\d{2})[/-](\d{2})")


# ----------------------------------------------------------------------
# Small widgets of the frame
# ----------------------------------------------------------------------
class _NavDelegate(QStyledItemDelegate):
    """Paints a nav row: icon, name, the selection mark, and (above the
    first row of a group) the group caption."""

    ROW_HEIGHT = 36
    CAPTION_HEIGHT = 28
    FIRST_CAPTION_HEIGHT = 24

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.collapsed = False

    def _caption_height(self, index) -> int:
        name = index.data(Qt.DisplayRole)
        if name not in NAV_GROUP_OF_FIRST:
            return 0
        if self.collapsed:
            return 0 if index.row() == 0 else 13
        return self.FIRST_CAPTION_HEIGHT if index.row() == 0 else self.CAPTION_HEIGHT

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(40, self.ROW_HEIGHT + self._caption_height(index))

    def paint(self, painter: QPainter, option, index) -> None:
        palette = current_palette()
        name = str(index.data(Qt.DisplayRole) or "")
        caption_h = self._caption_height(index)
        full = option.rect
        row = QRect(full.left(), full.top() + caption_h, full.width(), full.height() - caption_h)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if caption_h:
            if self.collapsed:
                painter.setPen(QColor(palette["border"]))
                y = full.top() + caption_h // 2
                painter.drawLine(full.left() + 14, y, full.right() - 14, y)
            else:
                font = QFont(option.font)
                font.setPixelSize(11)
                font.setWeight(QFont.DemiBold)
                font.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
                painter.setFont(font)
                painter.setPen(QColor(palette["muted"]))
                painter.drawText(QRect(full.left() + 20, full.top(), full.width() - 28, caption_h - 4),
                                 Qt.AlignLeft | Qt.AlignBottom, NAV_GROUP_OF_FIRST[name].upper())

        pill = QRectF(row.left() + 8, row.top() + 2, row.width() - 16, row.height() - 4)
        if selected or hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(palette["selection_bg"] if selected else palette["hover"]))
            painter.drawRoundedRect(pill, 4, 4)
        if selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(palette["accent"]))
            painter.drawRoundedRect(QRectF(row.left(), row.top() + 6, 3, row.height() - 12), 1.5, 1.5)

        icon_color = palette["accent"] if selected else palette["muted"]
        icon_size = 18.0
        if self.collapsed:
            icon_rect = QRectF(row.center().x() - icon_size / 2 + 0.5, row.center().y() - icon_size / 2 + 0.5,
                               icon_size, icon_size)
        else:
            icon_rect = QRectF(row.left() + 20, row.center().y() - icon_size / 2 + 0.5, icon_size, icon_size)
        paint_icon(painter, WORKSPACE_ICONS.get(name, "panel"), icon_rect, icon_color)

        if not self.collapsed:
            font = QFont(option.font)
            font.setPixelSize(13)
            font.setWeight(QFont.DemiBold if selected else QFont.Normal)
            painter.setFont(font)
            painter.setPen(QColor(palette["ink"]))
            text_rect = QRect(row.left() + 48, row.top(), row.width() - 56, row.height())
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft,
                             QFontMetrics(font).elidedText(name, Qt.ElideRight, text_rect.width()))
        if option.state & QStyle.State_HasFocus and option.state & QStyle.State_KeyboardFocusChange:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QColor(palette["focus"]))
            painter.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        painter.restore()

    def helpEvent(self, event, view, option, index) -> bool:  # noqa: N802 (Qt naming)
        # With the sidebar collapsed the name is not on screen: lead the tip with it.
        if self.collapsed and event.type() == QEvent.ToolTip and index.isValid():
            name = str(index.data(Qt.DisplayRole) or "")
            QToolTip.showText(event.globalPos(), f"{name}\n{index.data(Qt.ToolTipRole) or ''}".strip(), view)
            return True
        return super().helpEvent(event, view, option, index)


class _ElidedLabel(QLabel):
    """A one-line label that shortens its text with an ellipsis instead of
    asking for room: what it shows can never widen the window."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, allowance: int = 24) -> None:
        super().__init__(parent)
        self._full = ""
        self._allowance = allowance      # padding + border the stylesheet adds
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = str(text)
        self.updateGeometry()
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        room = max(0, self.width() - self._allowance)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, room))

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + self._allowance + 2, hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(56, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._elide()


class _BusyBar(QWidget):
    """A short indeterminate progress mark for the status bar: a segment
    sliding along a track while something is running."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(84, 14)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._phase = (self._phase + 0.035) % 1.0
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track = QRectF(0, self.height() / 2 - 2, self.width(), 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(palette["bg_alt"]))
        painter.drawRoundedRect(track, 2, 2)
        segment = track.width() * 0.35
        x = -segment + self._phase * (track.width() + segment)
        painter.setClipRect(track)
        painter.setBrush(QColor(palette["focus"]))
        painter.drawRoundedRect(QRectF(x, track.top(), segment, track.height()), 2, 2)
        painter.end()


class _PageStack(QStackedWidget):
    """The stack of workspaces. Unlike a plain QStackedWidget it reports
    the minimum size of the page on screen only (not the largest of all
    nine), with the width capped: the scroll area around it then gives a
    tall page a scroll bar and never passes a page's demands on to the
    window."""

    def _current_minimum(self) -> QSize:
        page = self.currentWidget()
        if page is None:
            return QSize(0, 0)
        hint = page.minimumSizeHint()
        fixed = page.minimumSize()
        return QSize(max(hint.width(), fixed.width(), 0), max(hint.height(), fixed.height(), 0))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        size = self._current_minimum()
        return QSize(min(size.width(), PAGE_MAX_MIN_WIDTH), size.height())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return self.minimumSizeHint().expandedTo(QSize(640, 420))


# ----------------------------------------------------------------------
# The main window
# ----------------------------------------------------------------------
class EmberMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        icon_path = asset_path("ember_logo.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self._apply_default_size()

        self.status_logger = StatusLogger()
        self.dataset = HanfordDataset(logger=self.status_logger.log)
        self.global_scope = GlobalScope(self)
        self.global_scope.changed.connect(self._on_scope_changed)

        #: Bring hidden workspaces up to date when they are shown rather
        #: than at once (see _lazy_refresh). Set False for the old eager
        #: fan-out.
        self.lazy_refresh = True
        self._dataset_stale: set = set()
        self._scope_stale: set = set()
        self._busy_text: Optional[str] = None
        self._loading = False
        self._scroll_positions: dict = {}

        self._build_actions()
        self._build_menus()
        self._build_status_bar()
        self._build_body()
        self._refresh_theme_dependents()

        self.statusBar().showMessage("Ready.")
        self._restore_settings()
        QTimer.singleShot(0, self._auto_load_on_startup)
        QTimer.singleShot(0, self._maybe_show_getting_started)

    # ------------------------------------------------------------------
    # Window size
    # ------------------------------------------------------------------
    def _available_size(self) -> Optional[QSize]:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return None
        return screen.availableGeometry().size()

    def _apply_default_size(self) -> None:
        width, height = DEFAULT_SIZE
        min_w, min_h = MINIMUM_SIZE
        available = self._available_size()
        if available is not None and available.width() > 0:
            # Never ask for more than the screen has.
            width = min(width, max(640, available.width() - 40))
            height = min(height, max(480, available.height() - 40))
            min_w = min(min_w, width)
            min_h = min(min_h, height)
        self.setMinimumSize(min_w, min_h)
        self.resize(width, height)

    def _fit_to_screen(self) -> None:
        """A geometry saved by an earlier session (or on a larger screen)
        must not come back bigger than the screen in front of the user."""
        available = self._available_size()
        if available is None or available.width() <= 0:
            return
        width = min(self.width(), available.width())
        height = min(self.height(), available.height())
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_actions(self) -> None:
        """The data actions, built before the menus and the header that
        show them. The attribute names are part of the shell's contract
        (load code and tests read them)."""
        self.browse_action = QAction("Open data file…", self)
        self.browse_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.browse_action.setToolTip("Open a CSV or Parquet export of the tank inventory.")
        self.browse_action.triggered.connect(lambda _checked=False: self._browse_and_load())

        self.load_local_action = QAction("Open default dataset", self)
        self.load_local_action.setToolTip(
            "Load Hanford.csv (and Tank_attributes.csv) from the application folder.")
        self.load_local_action.triggered.connect(lambda _checked=False: self._load_local())

        self.reload_action = QAction("Reload", self)
        self.reload_action.setShortcut(QKeySequence("F5"))
        self.reload_action.setToolTip("Read the current data file again.")
        self.reload_action.triggered.connect(lambda _checked=False: self._reload())

        # Checkable actions: `.isChecked()` is what the load code reads.
        self.use_cache_check = QAction("Use the cached copy when loading", self)
        self.use_cache_check.setCheckable(True)
        self.use_cache_check.setChecked(True)
        self.use_cache_check.setToolTip(
            "Read the pre-cleaned copy kept beside the data file (faster). "
            "Clear this to read and clean the source file on every load.")
        self.refresh_cache_check = QAction("Rebuild the cached copy on next load", self)
        self.refresh_cache_check.setCheckable(True)
        self.refresh_cache_check.setToolTip(
            "Clean the source file again and replace the cached copy the next time data is loaded.")

    def _build_menus(self) -> None:
        bar = self.menuBar()
        # Every menu is held on self: a menu whose only Python reference is a
        # local can be collected while Qt still owns the C++ object.
        file_menu = self.file_menu = bar.addMenu("&File")
        file_menu.setToolTipsVisible(True)
        file_menu.addAction(self.browse_action)
        file_menu.addAction(self.load_local_action)
        file_menu.addAction(self.reload_action)
        file_menu.addSeparator()
        file_menu.addAction("Open project…", self.open_project, "Ctrl+O")
        file_menu.addAction("Save project…", self.save_project, "Ctrl+S")
        file_menu.addSeparator()
        self.export_action = file_menu.addAction("Export current view…", self.export_current_page, "Ctrl+E")
        self.export_action.setToolTip("Export the tables or the figure of the workspace on screen.")
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, "Ctrl+Q")

        data_menu = self.data_menu = bar.addMenu("&Data")
        data_menu.setToolTipsVisible(True)
        data_menu.addAction(self.use_cache_check)
        data_menu.addAction(self.refresh_cache_check)
        data_menu.addSeparator()
        data_menu.addAction("Open output folder", self._open_output_folder)

        view_menu = self.view_menu = bar.addMenu("&View")
        view_menu.setToolTipsVisible(True)
        self.workspace_actions: List[QAction] = []
        for number, name in enumerate(NAV_ITEMS, start=1):
            action = view_menu.addAction(name)
            if number <= 9:
                action.setShortcut(QKeySequence(f"Ctrl+{number}"))
            action.setToolTip(NAV_DESCRIPTIONS.get(name, ""))
            action.triggered.connect(lambda _checked=False, n=name: self.open_page(n))
            self.workspace_actions.append(action)
        view_menu.addSeparator()
        # Held on self: a submenu whose only Python reference is a local
        # can be collected while Qt still owns the C++ object, which shows
        # up as an access violation during widget teardown.
        self.guided_menu = view_menu.addMenu("Guided questions")
        self.guided_menu.setToolTipsVisible(True)
        self.guided_actions = []
        for template in TEMPLATES:
            action = self.guided_menu.addAction(template.question)
            action.setToolTip(template.explanation)
            action.triggered.connect(
                lambda checked=False, t=template: self.run_guided_question(t))
            self.guided_actions.append(action)
        view_menu.addSeparator()
        self.search_action = view_menu.addAction("Search…", self.focus_search, "Ctrl+K")
        self.collapse_action = view_menu.addAction("Collapse sidebar")
        self.collapse_action.setCheckable(True)
        self.collapse_action.setShortcut(QKeySequence("Ctrl+B"))
        self.collapse_action.toggled.connect(self._set_sidebar_collapsed)
        self.dark_mode_action = view_menu.addAction("Dark mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self._on_dark_mode_toggled)

        self.help_menu = bar.addMenu("&Help")
        self.help_menu.addAction("Getting Started…", self.show_getting_started)
        self.help_menu.addAction("User guide (Notice)", self.show_notice, "F1")
        self.help_menu.addAction(f"About {APP_DISPLAY_NAME}", self.show_about)
        self.help_menu.addSeparator()
        self.help_menu.addAction("Debug / Export…", self.show_debug)
        # The menus' own title actions are held too. PySide ties a menu to
        # the QAction wrapper that .menu() was called on; a caller walking
        # menuBar().actions() and asking each for its menu would otherwise
        # invalidate that menu, and every action in it, when its loop
        # variable goes away (seen as "QAction already deleted" on the next
        # theme toggle).
        self._menu_title_actions = [
            menu.menuAction()
            for menu in (self.file_menu, self.data_menu, self.view_menu, self.guided_menu, self.help_menu)
        ]

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.busy_bar = _BusyBar()
        self.busy_label = QLabel("")
        self.busy_bar.setVisible(False)
        self.busy_label.setVisible(False)
        self.status_dataset_label = QLabel("No dataset loaded")
        self.status_version_label = QLabel(f"{APP_DISPLAY_NAME} {APP_VERSION}")
        bar.addPermanentWidget(self.busy_label)
        bar.addPermanentWidget(self.busy_bar)
        bar.addPermanentWidget(self.status_dataset_label)
        bar.addPermanentWidget(self.status_version_label)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("AppHeader")
        header.setFixedHeight(HEADER_HEIGHT)
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 0, 12, 0)
        row.setSpacing(12)

        self.header_mark = QLabel()
        self.header_mark.setFixedSize(30, 30)
        self.header_mark.setPixmap(ember_mark_pixmap(30))
        row.addWidget(self.header_mark, 0, Qt.AlignVCenter)

        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        names.addStretch(1)
        self.app_title_label = QLabel(APP_DISPLAY_NAME)
        self.app_title_label.setObjectName("AppTitle")
        names.addWidget(self.app_title_label)
        self.app_subtitle_label = QLabel(APP_SUBTITLE)
        self.app_subtitle_label.setObjectName("AppSubtitle")
        names.addWidget(self.app_subtitle_label)
        names.addStretch(1)
        row.addLayout(names)
        row.addSpacing(16)

        self.dataset_chip = _ElidedLabel("No dataset loaded")
        self.dataset_chip.setObjectName("DatasetChip")
        self.dataset_chip.setToolTip("No dataset loaded. Use File → Open data file.")
        row.addWidget(self.dataset_chip, 0, Qt.AlignVCenter)
        row.addStretch(1)

        self.search_index = SearchIndex({name: NAV_DESCRIPTIONS.get(name, "") for name in NAV_ITEMS})
        self.search_box = GlobalSearch(self.search_index, header)
        self.search_box.setMinimumWidth(180)
        self.search_box.setMaximumWidth(320)
        self.search_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_box.activated.connect(self.open_hit)
        row.addWidget(self.search_box, 2, Qt.AlignVCenter)

        self.help_button = QToolButton()
        self.help_button.setObjectName("HeaderButton")
        self.help_button.setText("Help")
        self.help_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.help_button.setPopupMode(QToolButton.InstantPopup)
        self.help_button.setMenu(self.help_menu)
        self.help_button.setToolTip("Getting started, the user guide and the About box.")
        row.addWidget(self.help_button, 0, Qt.AlignVCenter)

        self.theme_button = QToolButton()
        self.theme_button.setObjectName("HeaderButton")
        self.theme_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.theme_button.clicked.connect(lambda: self.dark_mode_action.toggle())
        row.addWidget(self.theme_button, 0, Qt.AlignVCenter)
        return header

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setAttribute(Qt.WA_StyledBackground, True)
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 4, 1, 0)
        layout.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setFrameShape(QFrame.NoFrame)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setMouseTracking(True)
        self.nav.viewport().setAttribute(Qt.WA_Hover, True)
        self._nav_delegate = _NavDelegate(self.nav)
        self.nav.setItemDelegate(self._nav_delegate)
        for name in NAV_ITEMS:
            item = QListWidgetItem(name, self.nav)
            item.setToolTip(NAV_DESCRIPTIONS.get(name, ""))
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self.nav, 1)

        self.sidebar_toggle = QToolButton()
        self.sidebar_toggle.setObjectName("SidebarToggle")
        self.sidebar_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sidebar_toggle.setToolTip("Collapse the sidebar to icons (Ctrl+B).")
        self.sidebar_toggle.clicked.connect(lambda: self.collapse_action.toggle())
        layout.addWidget(self.sidebar_toggle)
        self.sidebar = sidebar
        return sidebar

    def _build_body(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        outer = QHBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        root.addLayout(outer, 1)
        outer.addWidget(self._build_sidebar())

        # Building all 9 pages (each with its own matplotlib figures/tables)
        # is the single most expensive part of startup -- a couple of
        # seconds with nothing pumping the Windows message queue is enough
        # to trip the OS's "Not Responding" ghost-window detector on
        # whatever window is currently visible (the splash screen, during
        # qt_main's startup sequence). Pumping events between each page
        # keeps that queue draining without changing what gets built.
        # The scope bar and its state must exist BEFORE the page loop below:
        # pages read app_window.global_scope in their constructors, and the
        # pump() calls deliver queued events that must never hit half-built
        # scope wiring.
        self.scope_bar = GlobalScopeBar(self.global_scope)

        pump = QApplication.processEvents
        self.stack = _PageStack()
        self.overview_page = OverviewPage(self, scope=self.global_scope)
        self.stack.addWidget(self.overview_page)
        pump()
        self.query_page = QueryPage(self, scope=self.global_scope)
        self.stack.addWidget(self.query_page)
        pump()
        self.explorer_page = ExplorerPage(self, scope=self.global_scope)
        self.stack.addWidget(self.explorer_page)
        pump()
        self.tank_attrs_page = TankAttributesPage(self)
        self.stack.addWidget(self.tank_attrs_page)
        pump()
        self.tank_explorer_page = TankExplorerPage(self)
        self.stack.addWidget(self.tank_explorer_page)
        pump()
        self.heatmaps_page = HeatmapPage(self, scope=self.global_scope)
        self.stack.addWidget(self.heatmaps_page)
        pump()
        self.correlations_page = CorrelationsPage(self, scope=self.global_scope)
        self.stack.addWidget(self.correlations_page)
        pump()
        self.vitrification_page = VitrificationPage(self)
        self.stack.addWidget(self.vitrification_page)
        pump()
        self.figure_composer_page = FigureComposerPage(self)
        self.stack.addWidget(self.figure_composer_page)
        pump()

        # The scope strip keeps one height on every workspace. Above a
        # workspace that honors the scope it holds the selectors; above one
        # that does not, a line saying so. Either way the page below starts
        # at the same place, so nothing jumps when switching workspace.
        self.scope_strip = QFrame()
        self.scope_strip.setObjectName("ScopeStrip")
        self.scope_strip.setFixedHeight(SCOPE_STRIP_HEIGHT)
        strip_row = QHBoxLayout(self.scope_strip)
        strip_row.setContentsMargins(0, 0, 0, 0)
        strip_row.setSpacing(0)
        strip_row.addWidget(self.scope_bar, 1)
        self.scope_idle = QWidget()
        idle_row = QHBoxLayout(self.scope_idle)
        idle_row.setContentsMargins(20, 0, 20, 0)
        idle_row.setSpacing(16)
        idle_title = QLabel("Scope")
        idle_title.setObjectName("ScopeTitle")
        idle_row.addWidget(idle_title)
        self.scope_idle_label = QLabel("Not used by this workspace: it shows the whole dataset.")
        self.scope_idle_label.setObjectName("ScopeIdle")
        idle_row.addWidget(self.scope_idle_label)
        idle_row.addStretch(1)
        strip_row.addWidget(self.scope_idle, 1)
        self.scope_idle.setVisible(False)

        # The page area. The stack lives in a scroll area so that no page
        # (a long label, a tall figure) can set the window's minimum size.
        self.page_scroll = QScrollArea()
        self.page_scroll.setObjectName("PageScroll")
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setWidget(self.stack)
        self.stack.currentChanged.connect(lambda _index: self.stack.updateGeometry())

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self.scope_strip)
        right.addWidget(self.page_scroll, 1)
        outer.addLayout(right, 1)

        # Nav row -> page by NAME: the rail order and the stack's
        # construction order happen to match today, but must never be
        # assumed to -- always resolve through this dict.
        self._pages_by_nav: Dict[str, QWidget] = {
            NAV_OVERVIEW: self.overview_page,
            NAV_QUERY: self.query_page,
            NAV_EXPLORER: self.explorer_page,
            NAV_TANK_ATTRS: self.tank_attrs_page,
            NAV_TANK_EXPLORER: self.tank_explorer_page,
            NAV_HEATMAPS: self.heatmaps_page,
            NAV_CORRELATIONS: self.correlations_page,
            NAV_VITRIFICATION: self.vitrification_page,
            NAV_FIGURE_COMPOSER: self.figure_composer_page,
        }
        # Debug/Export is a troubleshooting utility, not a workspace -- it
        # lives in the Help menu (see show_debug()) instead of taking a nav
        # rail slot, and is built lazily on first open rather than eagerly
        # here. Broadcast membership is DERIVED in _dataset_aware_pages()
        # rather than kept as a second hand-maintained list -- a list would
        # have to be remembered at every future page addition, and a missed
        # append silently starves that page of dataset updates.
        self.debug_page: Optional[DebugPage] = None
        self._debug_dialog: Optional[QDialog] = None
        # A workspace whose sub-tabs do not all honor the scope says so
        # when the visible one changes (see _scope_bar_applies).
        for page in self._pages_by_nav.values():
            relevance_changed = getattr(page, "scope_relevance_changed", None)
            if relevance_changed is not None:
                relevance_changed.connect(self._refresh_scope_bar)

        self.nav.setCurrentRow(0)

    def _dataset_aware_pages(self) -> list:
        pages = list(self._pages_by_nav.values())
        if self.debug_page is not None:
            pages.append(self.debug_page)
        return pages

    def _scoped_nav_items(self) -> set:
        """Workspaces that honor the global scope bar -- DERIVED from the
        same protocol the shell dispatches on (on_scope_changed), never a
        second hand-maintained list. The bar is shown only above these: a
        "global" filter hovering over a workspace that ignores it would be
        a lie about what's on screen, and a workspace that honors it
        without showing it filters data by an invisible control. Legacy
        workspaces join automatically as they migrate onto the engine."""
        return {
            name for name, page in self._pages_by_nav.items()
            if hasattr(page, "on_scope_changed")
        }

    def _scope_bar_applies(self, name: str) -> bool:
        """Whether the scope bar belongs above workspace `name` right
        now. A scoped workspace may narrow that further through an
        optional scope_applies() -- Correlations hosts one scoped
        sub-tab beside three legacy ones that ignore the scope, and
        the bar must not hover over those."""
        if name not in self._scoped_nav_items():
            return False
        applies = getattr(self._pages_by_nav[name], "scope_applies", None)
        return bool(applies()) if callable(applies) else True

    def _show_scope_bar(self, applies: bool) -> None:
        """Selectors or the "not used here" line; the strip itself stays."""
        self.scope_bar.setVisible(applies)
        self.scope_idle.setVisible(not applies)

    def _refresh_scope_bar(self) -> None:
        row = self.nav.currentRow()
        if 0 <= row < len(NAV_ITEMS):
            self._show_scope_bar(self._scope_bar_applies(NAV_ITEMS[row]))

    # ------------------------------------------------------------------
    # Theme-dependent bits the stylesheet cannot reach (painted icons)
    # ------------------------------------------------------------------
    def _refresh_theme_dependents(self) -> None:
        palette = current_palette()
        dark = self.dark_mode_action.isChecked()
        chrome_ink = palette["chrome_ink"]
        self.help_button.setIcon(icon("help", chrome_ink, 16))
        self.help_button.setIconSize(QSize(16, 16))
        self.theme_button.setIcon(icon("sun" if dark else "moon", chrome_ink, 18))
        self.theme_button.setIconSize(QSize(18, 18))
        self.theme_button.setToolTip("Switch to the light theme." if dark else "Switch to the dark theme.")
        self.search_box.refresh_icon()
        self._refresh_sidebar_toggle()
        self.nav.viewport().update()

    def _refresh_sidebar_toggle(self) -> None:
        palette = current_palette()
        collapsed = self.collapse_action.isChecked()
        self.sidebar_toggle.setIcon(icon("chevron_right" if collapsed else "chevron_left", palette["muted"], 16))
        self.sidebar_toggle.setIconSize(QSize(16, 16))
        if collapsed:
            self.sidebar_toggle.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.sidebar_toggle.setToolTip("Expand the sidebar (Ctrl+B).")
        else:
            self.sidebar_toggle.setText("Collapse")
            self.sidebar_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.sidebar_toggle.setToolTip("Collapse the sidebar to icons (Ctrl+B).")

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        self._nav_delegate.collapsed = bool(collapsed)
        self.sidebar.setFixedWidth(SIDEBAR_COLLAPSED_WIDTH if collapsed else SIDEBAR_WIDTH)
        self._refresh_sidebar_toggle()
        # Row heights depend on the mode (captions become separators).
        self.nav.doItemsLayout()
        self.nav.viewport().update()

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        # Restyles the Qt chrome only -- matplotlib plot areas stay white
        # so on-screen plots always match PNG/SVG/PDF export.
        from qt_theme import apply_theme
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, dark=enabled)
        self._refresh_theme_dependents()

    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------
    def set_busy(self, text: Optional[str]) -> None:
        """Show (text) or clear (None) the busy indicator of the status
        bar. For work that leaves the UI thread free; the caller clears it
        when the work is done."""
        self._busy_text = str(text) if text else None
        busy = self._busy_text is not None
        self.busy_label.setText(self._busy_text or "")
        self.busy_label.setVisible(busy)
        self.busy_bar.setVisible(busy)

    def is_busy(self) -> bool:
        return self._busy_text is not None

    def _set_busy(self, busy: bool) -> None:
        """Loading state: the load actions are unavailable while a load runs."""
        self._loading = bool(busy)
        for action in (self.load_local_action, self.browse_action, self.reload_action):
            action.setEnabled(not busy)
        self.set_busy("Loading data…" if busy else None)
        if busy:
            self.dataset_chip.set_full_text("Loading data…")
        else:
            self._refresh_dataset_summary()

    # ------------------------------------------------------------------
    # What is loaded (header chip, status bar)
    # ------------------------------------------------------------------
    def dataset_summary(self) -> Dict[str, object]:
        """Facts about the loaded data for the header: source file, record
        and tank counts, and the published / decay date range when the
        data carries those columns. Empty dict when nothing is loaded."""
        if not self.dataset.is_loaded():
            return {}
        df = self.dataset.require_df()
        summary: Dict[str, object] = {
            "source": self.dataset.path.name if self.dataset.path else "Unsaved data",
            "path": str(self.dataset.path) if self.dataset.path else "",
            "records": int(df.height),
            "columns": int(df.width),
        }
        if "WasteSiteId" in df.columns:
            summary["tanks"] = int(df.get_column("WasteSiteId").n_unique())
        for column, key in (("PublishedDate", "published"), ("DecayDate", "decay")):
            if column not in df.columns:
                continue
            try:
                values = df.get_column(column).drop_nulls().unique().to_list()
            except Exception:
                continue
            dates = sorted({
                "-".join(match.groups()) for match in (_DATE_RE.match(str(v).strip()) for v in values) if match
            })
            if dates:
                summary[key] = dates
        report = self.dataset.report
        if report is not None:
            summary["cache_used"] = bool(report.cache_used)
            if report.attributes_path is not None:
                summary["attributes"] = Path(report.attributes_path).name
        return summary

    def _refresh_dataset_summary(self) -> None:
        info = self.dataset_summary()
        if not info:
            self.dataset_chip.set_full_text("No dataset loaded")
            self.dataset_chip.setToolTip("No dataset loaded. Use File → Open data file.")
            self.status_dataset_label.setText("No dataset loaded")
            return
        parts = [str(info["source"]), f"{info['records']:,} records"]
        if "tanks" in info:
            parts.append(f"{info['tanks']:,} tanks")
        published = info.get("published")
        if published:
            first, last = published[0], published[-1]
            years = first[:4] if first[:4] == last[:4] else f"{first[:4]}–{last[:4]}"
            parts.append(f"published {years}")
        decay_years = sorted({date[:4] for date in info.get("decay") or []})
        if 0 < len(decay_years) <= 3:
            parts.append("decay date " + " / ".join(decay_years))
        self.dataset_chip.set_full_text("  ·  ".join(parts))

        tip = [str(info.get("path") or info["source"]),
               f"{info['records']:,} records, {info['columns']} columns"
               + (f", {info['tanks']:,} tanks" if "tanks" in info else "")]
        if published:
            tip.append(f"Published: {published[0]} to {published[-1]}" if len(published) > 1
                       else f"Published: {published[0]}")
        decay = info.get("decay")
        if decay:
            tip.append("Decay date" + ("s" if len(decay) > 1 else "") + ": " + ", ".join(decay))
        if "attributes" in info:
            tip.append(f"Tank attributes: {info['attributes']}")
        if info.get("cache_used"):
            tip.append("Read from the cached copy of the data file.")
        self.dataset_chip.setToolTip("\n".join(tip))
        status = f"{info['records']:,} records"
        if "tanks" in info:
            status += f"  ·  {info['tanks']:,} tanks"
        self.status_dataset_label.setText(status)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_local(self) -> None:
        use_cache, refresh_cache = self.use_cache_check.isChecked(), self.refresh_cache_check.isChecked()
        self._start_load(lambda: self.dataset.load_local_default(use_cache=use_cache, refresh_cache=refresh_cache))

    def _browse_and_load(self) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings(APP_NAME, APP_NAME)
        last_dir = settings.value("last_browse_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", last_dir, "Data files (*.csv *.tsv *.txt *.parquet *.pq);;All files (*.*)"
        )
        if not path:
            return
        settings.setValue("last_browse_dir", str(Path(path).resolve().parent))
        use_cache, refresh_cache = self.use_cache_check.isChecked(), self.refresh_cache_check.isChecked()
        self._start_load(lambda: self.dataset.load(path, use_cache=use_cache, refresh_cache=refresh_cache))

    def _reload(self) -> None:
        if self.dataset.path is None:
            QMessageBox.information(self, "Reload", "Nothing loaded yet.")
            return
        path = self.dataset.path
        use_cache, refresh_cache = self.use_cache_check.isChecked(), self.refresh_cache_check.isChecked()
        self._start_load(lambda: self.dataset.load(path, use_cache=use_cache, refresh_cache=refresh_cache))

    def _auto_load_on_startup(self) -> None:
        self._set_busy(True)
        self.statusBar().showMessage("Looking for Hanford.csv next to the application…")
        run_in_thread(
            lambda: self.dataset.load_local_default(use_cache=True, refresh_cache=False),
            on_done=self._on_load_done,
            on_error=self._on_startup_load_error,
        )

    def _start_load(self, fn) -> None:
        self._set_busy(True)
        self.statusBar().showMessage("Loading…")
        run_in_thread(fn, on_done=self._on_load_done, on_error=self._on_load_error)

    def _on_load_done(self, report) -> None:
        self._set_busy(False)
        if self.refresh_cache_check.isChecked():
            # "Rebuild on next load" has now happened once: it is not a mode.
            self.refresh_cache_check.setChecked(False)
        info = self.dataset_summary()
        tanks = f", {info['tanks']:,} tanks" if "tanks" in info else ""
        self.statusBar().showMessage(
            f"Loaded {report.source_path.name}: {report.rows:,} records{tanks}, {report.columns} columns."
        )
        self.search_index.set_dataset(self.dataset)
        # Reconcile the scope bar FIRST -- pages refreshed below must see
        # the already-reconciled scope, not a stale filter pointing at
        # values the new dataset doesn't have. The coalesce block spans
        # the dispatch so any scope change from that reconciliation is
        # announced exactly once, AFTER every page has the new dataset.
        lazy = self._lazy_refresh()
        current = self.stack.currentWidget()
        with self.global_scope.coalesce():
            self.scope_bar.on_dataset_changed(self.dataset)
            for page in self._dataset_aware_pages():
                if not hasattr(page, "on_dataset_changed"):
                    continue
                if lazy and page is not current and page is not self.debug_page:
                    # Not on screen: it takes the new dataset when shown.
                    self._dataset_stale.add(page)
                    self._scope_stale.discard(page)
                    continue
                self._dataset_stale.discard(page)
                self._scope_stale.discard(page)
                page.on_dataset_changed(self.dataset)

    # ------------------------------------------------------------------
    # Keeping hidden workspaces cheap
    # ------------------------------------------------------------------
    def _lazy_refresh(self) -> bool:
        """Whether hidden workspaces may wait for their refresh.

        Deferring is only sound when "later, when it is shown" will
        happen under the shell's eyes, i.e. when the window is on screen.
        A window that is not shown (a script, the test-suite) is driven
        by code that reads a hidden page right after changing the scope
        or the dataset, so there every page is refreshed at once, as
        before."""
        return bool(self.lazy_refresh) and self.isVisible()

    def _bring_up_to_date(self, page) -> None:
        """Deliver to `page` what it missed while hidden: the dataset if
        one was loaded meanwhile (that also resets its results), otherwise
        the scope change."""
        if page in self._dataset_stale:
            self._dataset_stale.discard(page)
            self._scope_stale.discard(page)
            self._call_with_wait_cursor(lambda: page.on_dataset_changed(self.dataset))
        elif page in self._scope_stale:
            self._scope_stale.discard(page)
            self._call_with_wait_cursor(page.on_scope_changed)

    def ensure_page_current(self, name_or_page) -> None:
        """Public form of _bring_up_to_date, for code that is about to
        read a workspace it is not showing."""
        page = self._pages_by_nav.get(name_or_page, name_or_page) if isinstance(name_or_page, str) else name_or_page
        if page is not None:
            self._bring_up_to_date(page)

    @staticmethod
    def _call_with_wait_cursor(fn) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setOverrideCursor(Qt.WaitCursor)
        try:
            fn()
        finally:
            if app is not None:
                app.restoreOverrideCursor()

    def _on_scope_changed(self) -> None:
        lazy = self._lazy_refresh()
        current = self.stack.currentWidget()
        for page in self._dataset_aware_pages():
            if not hasattr(page, "on_scope_changed"):
                continue
            if page in self._dataset_stale:
                continue      # it will be rebuilt on the new dataset anyway
            if lazy and page is not current:
                self._scope_stale.add(page)
                continue
            self._scope_stale.discard(page)
            page.on_scope_changed()

    def _on_load_error(self, tb_text: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Load failed.")
        QMessageBox.critical(self, "Load failed", tb_text)

    def _on_startup_load_error(self, tb_text: str) -> None:
        # Expected on a clean first run (no Hanford.csv next to the app
        # yet) -- a status message is enough, no need to interrupt with a
        # popup on startup.
        self._set_busy(False)
        self.statusBar().showMessage("No dataset loaded. Use File → Open data file.")

    def _open_output_folder(self) -> None:
        folder = self.dataset.output_root
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder.resolve()))

    def export_current_page(self) -> bool:
        """File -> Export current view: the export of the workspace (or of
        its sub-tab) on screen. Returns False when it has none."""
        page = self.stack.currentWidget()
        candidates = []
        tabs = getattr(page, "tabs", None)
        current_tab = getattr(tabs, "currentWidget", None)
        if callable(current_tab):
            candidates.append(current_tab())
        candidates.append(page)
        for target in candidates:
            for method_name in _EXPORT_METHODS:
                method = getattr(target, method_name, None)
                if callable(method):
                    method()
                    return True
        self.statusBar().showMessage("This workspace has no export of its own.")
        return False

    # ------------------------------------------------------------------
    # Nav / settings / misc
    # ------------------------------------------------------------------
    def _on_nav_changed(self, row: int) -> None:
        if not (0 <= row < len(NAV_ITEMS)):
            return
        page = self._pages_by_nav[NAV_ITEMS[row]]
        previous = self.stack.currentWidget()
        if previous is not None and previous is not page:
            # One scroll area serves every workspace: remember where this
            # one was scrolled to, so coming back finds it as it was left.
            self._scroll_positions[previous] = (
                self.page_scroll.horizontalScrollBar().value(), self.page_scroll.verticalScrollBar().value())
        self.stack.setCurrentWidget(page)
        # The selectors appear only above workspaces that honor the scope.
        self._show_scope_bar(self._scope_bar_applies(NAV_ITEMS[row]))
        self._bring_up_to_date(page)
        if previous is not page:
            x, y = self._scroll_positions.get(page, (0, 0))
            self.page_scroll.horizontalScrollBar().setValue(x)
            self.page_scroll.verticalScrollBar().setValue(y)
            if x or y:
                # The scroll range follows the new page only after the
                # layout has run; set the position again once it has.
                QTimer.singleShot(0, lambda p=page, px=x, py=y: self._restore_scroll(p, px, py))

    def _restore_scroll(self, page, x: int, y: int) -> None:
        if self.stack.currentWidget() is page:
            self.page_scroll.horizontalScrollBar().setValue(x)
            self.page_scroll.verticalScrollBar().setValue(y)

    # ------------------------------------------------------------------
    # Projects (.ember): saved baskets, questions and scope
    # ------------------------------------------------------------------
    def current_project(self):
        """The session's saveable state. Kept small and declarative on
        purpose -- a project is data, not a pickled widget tree."""
        import provenance as prov
        from ember_project import EmberProject, SavedQuestion

        project = EmberProject(name=self.dataset.path.stem if self.dataset.path else "")
        if self.dataset.is_loaded():
            project.dataset_fingerprint = prov.dataset_fingerprint(self.dataset)
        project.scope = self.global_scope.as_scope()

        page = self.query_page
        self._bring_up_to_date(page)   # never save a question left over from another dataset or scope
        if page._result is not None:
            kind = page.kind_combo.currentText().lower()
            subject = (page.basket_combo.currentText() if kind == "basket"
                       else page.subject_edit.text().strip())
            project.add_question(SavedQuestion(
                name=f"{subject} ({page._result.basis})", subject_kind=kind,
                subject=subject, basis=page._result.basis,
                group_by=list(page._result_group_by),
                scope=self.global_scope.as_scope(),
                mark=MARK_BY_LABEL.get(page.mark_combo.currentText(), "bar"),
            ))
        return project

    def save_project(self) -> None:
        from ember_project import PROJECT_SUFFIX
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", str(self.dataset.output_root),
            f"Ember project (*{PROJECT_SUFFIX})")
        if not path:
            return
        saved = self.current_project().save(path)
        self.statusBar().showMessage(f"Project saved: {saved}")

    def open_project(self) -> None:
        from ember_project import PROJECT_SUFFIX, EmberProject, dataset_mismatch
        import provenance as prov

        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", str(self.dataset.output_root),
            f"Ember project (*{PROJECT_SUFFIX})")
        if not path:
            return
        try:
            project = EmberProject.load(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Open project", str(exc))
            return

        notes = list(project.problems())
        if self.dataset.is_loaded():
            mismatch = dataset_mismatch(project, prov.dataset_fingerprint(self.dataset))
            if mismatch:
                notes.insert(0, mismatch)

        with self.global_scope.coalesce():
            self.global_scope.clear()
            for column, values in (project.scope or {}).items():
                self.global_scope.set_filter(column, list(values))
        self.scope_bar.on_dataset_changed(self.dataset)

        if notes:
            # A project that no longer fits the data must SAY so, not
            # quietly load two thirds of itself.
            QMessageBox.information(
                self, "Project opened with notes", chr(10).join(notes))
        self.statusBar().showMessage(
            f"Project opened: {Path(path).name} "
            f"({len(project.questions)} saved question(s))")

    def run_guided_question(self, template) -> bool:
        """Show the Query workspace and answer one shipped starter
        question there. Returns False without answering if the workspace
        can't be shown -- a completed query on a hidden page helps no one."""
        if not self._go_to_nav(NAV_QUERY):
            return False
        self.query_page.apply_template(template)
        return True

    def show_getting_started(self) -> None:
        dialog = GettingStartedDialog(self)
        dialog.workspace_chosen.connect(self._go_to_workspace)
        dialog.exec()

    def _maybe_show_getting_started(self) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings(APP_NAME, APP_NAME)
        if settings.value("getting_started_shown", False, type=bool):
            return
        settings.setValue("getting_started_shown", True)
        self.show_getting_started()

    def _go_to_workspace(self, name: str) -> None:
        if name in NAV_ITEMS:
            self.nav.setCurrentRow(NAV_ITEMS.index(name))

    def ask_in_query(self, kind: str, subject: str, basis: Optional[str] = None) -> bool:
        """Public cross-link: show the Query workspace and answer `subject`
        there. Returns False (having done nothing) if the workspace can't
        be shown -- a caller must never leave the user on one page while a
        hidden one reports a completed query. Workspaces call THIS rather
        than reaching into query_page through a private nav method."""
        if not self._go_to_nav(NAV_QUERY):
            return False
        self.query_page.prefill_and_run(kind, subject, basis)
        return True

    def _go_to_nav(self, name: str) -> bool:
        if name not in NAV_ITEMS:
            return False
        self.nav.setCurrentRow(NAV_ITEMS.index(name))
        return self.stack.currentWidget() is self._pages_by_nav[name]

    # ------------------------------------------------------------------
    # Public navigation API (workspaces, search)
    # ------------------------------------------------------------------
    def open_page(self, nav_name: str) -> bool:
        """Switch to the workspace called `nav_name` (one of NAV_ITEMS).
        Returns False for an unknown name."""
        return self._go_to_nav(str(nav_name))

    def open_tank(self, tank_id: str) -> bool:
        """Show one tank in the Tank Explorer: switch to it, select the
        tank and run. Anything that stands for a tank (a point, a bar, a
        map disc, a table row) calls this on double-click.

        Uses the page's own show_tank(tank_id) when it has one; otherwise
        selects the tank in its list and calls its run method. Returns
        False if the tank cannot be shown (unknown id, nothing loaded)."""
        tank_id = str(tank_id or "").strip()
        if not tank_id:
            return False
        if not self.dataset.is_loaded():
            self.statusBar().showMessage("Load a dataset to open a tank.")
            return False
        if not self._go_to_nav(NAV_TANK_EXPLORER):
            return False
        page = self.tank_explorer_page
        show = getattr(page, "show_tank", None)
        if callable(show):
            return show(tank_id) is not False
        return self._show_tank_by_selection(page, tank_id)

    def _show_tank_by_selection(self, page, tank_id: str) -> bool:
        tank_list = getattr(page, "tank_list", None)
        if tank_list is None:
            return False
        items = tank_list.findItems(tank_id, Qt.MatchExactly)
        if not items:
            # The list may be narrowed to one farm: widen it and look again.
            farm_combo = getattr(page, "farm_combo", None)
            if farm_combo is not None and farm_combo.currentIndex() != 0:
                farm_combo.setCurrentIndex(0)
                items = tank_list.findItems(tank_id, Qt.MatchExactly)
        if not items:
            self.statusBar().showMessage(f"Tank {tank_id} is not in the loaded dataset.")
            return False
        tank_list.clearSelection()
        items[0].setSelected(True)
        tank_list.setCurrentItem(items[0])
        tank_list.scrollToItem(items[0])
        run = getattr(page, "run", None)
        if callable(run):
            run()
        return True

    def open_element(self, subject: str, kind: str = KIND_ELEMENT) -> bool:
        """Search one element or analyte in the Element Explorer: switch
        to it, fill the field and run."""
        subject = str(subject or "").strip()
        if not subject or not self._go_to_nav(NAV_EXPLORER):
            return False
        page = self.explorer_page
        show = getattr(page, "show_subject", None)
        if callable(show):
            return show(subject, kind) is not False
        edit = getattr(page, "query_edit", None)
        if edit is None:
            return False
        mode_combo = getattr(page, "mode_combo", None)
        if mode_combo is not None:
            wanted = "element" if kind == KIND_ELEMENT else "analyte_exact"
            index = mode_combo.findText(wanted)
            mode_combo.setCurrentIndex(index if index >= 0 else max(0, mode_combo.findText("auto")))
        edit.setText(subject)
        run = getattr(page, "run_search", None)
        if callable(run):
            run()
        return True

    def open_hit(self, hit: SearchHit) -> bool:
        """Open one result of the global search."""
        if hit is None:
            return False
        if hit.kind == KIND_TANK:
            return self.open_tank(hit.key)
        if hit.kind in (KIND_ELEMENT, KIND_ANALYTE):
            return self.open_element(hit.key, hit.kind)
        if hit.kind == KIND_WORKSPACE:
            return self.open_page(hit.key)
        return False

    def focus_search(self) -> None:
        self.search_box.setFocus(Qt.ShortcutFocusReason)
        self.search_box.selectAll()

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    def show_about(self) -> None:
        HelpDialog(
            self, html=ABOUT_HTML, title=f"About {APP_NAME}",
            subtitle=f"{APP_SUBTITLE}  ·  version {APP_VERSION}",
        ).exec()

    def show_notice(self) -> None:
        from qt_help import load_notice_markdown
        HelpDialog(
            self, markdown=load_notice_markdown(), title=f"{APP_NAME} — Detailed Notice", width=920, height=760,
        ).exec()

    def show_debug(self) -> None:
        """Debug/Export is a troubleshooting utility, not a workspace --
        built lazily on first open and kept around afterward (not
        recreated) so the live log view's history survives across opens.
        Shown NON-modally: the whole point of a live log viewer is watching
        events stream in while you operate the app to reproduce them --
        exec() would block the very actions the user wants to observe."""
        if self._debug_dialog is None:
            self.debug_page = DebugPage(self)
            if self.dataset.is_loaded():
                self.debug_page.on_dataset_changed(self.dataset)
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{APP_NAME} — Debug / Export")
            dialog.resize(820, 560)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.debug_page)
            self._debug_dialog = dialog
        self._debug_dialog.show()
        self._debug_dialog.raise_()
        self._debug_dialog.activateWindow()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _restore_settings(self) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings(APP_NAME, APP_NAME)
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
            self._fit_to_screen()
        dark = settings.value("dark_mode", False, type=bool)
        if dark:
            self.dark_mode_action.setChecked(True)
        if settings.value("sidebar_collapsed", False, type=bool):
            self.collapse_action.setChecked(True)
        nav_row = settings.value("nav_row", 0, type=int)
        if 0 <= nav_row < self.nav.count():
            self.nav.setCurrentRow(nav_row)
            # setCurrentRow doesn't emit currentRowChanged when the row is
            # unchanged from its default -- call the handler explicitly so
            # the restored page still gets selected in the stack.
            self._on_nav_changed(self.nav.currentRow())

    def closeEvent(self, event) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings(APP_NAME, APP_NAME)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("nav_row", self.nav.currentRow())
        settings.setValue("dark_mode", self.dark_mode_action.isChecked())
        settings.setValue("sidebar_collapsed", self.collapse_action.isChecked())
        super().closeEvent(event)
