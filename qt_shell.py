"""
qt_shell.py — the Ember application shell: a left nav rail of workspaces
over a shared HanfordDataset, mirroring the sibling Dataapp project's shell
architecture (nav-by-name lookup, QSettings geometry/nav-row restore).

Single-domain app (unlike Dataapp's multi-technique Modules system) --
every nav row is always visible, no show/hide groups.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFileDialog, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QStackedWidget, QToolBar, QVBoxLayout, QWidget,
)

from data_model import HanfordDataset
from qt_correlations import CorrelationsPage
from qt_debug import DebugPage
from qt_explorer import ExplorerPage
from qt_figure_composer import FigureComposerPage
from qt_heatmap import HeatmapPage
from qt_help import ABOUT_HTML, APP_NAME, APP_VERSION, GettingStartedDialog, HelpDialog, asset_path
from qt_overview import OverviewPage
from qt_query import QueryPage
from qt_scope import GlobalScope, GlobalScopeBar
from qt_tank_attributes import TankAttributesPage
from qt_tank_explorer import TankExplorerPage
from qt_vitrification import VitrificationPage
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

# One-line, plain-language "what is this for" shown on hover -- the nav
# rail used to be plain text with no orientation for a first-time user;
# this is the cheapest, always-visible form of self-explanation.
NAV_DESCRIPTIONS: Dict[str, str] = {
    NAV_OVERVIEW: "Start here: dataset audit -- units, top elements/analytes, waste-phase/type/farm breakdowns, missing data.",
    NAV_QUERY: "Ask any combined question: element/analyte/basket x kg/Ci/grams/wt%/g-per-L x any grouping -- with honest unit bridging.",
    NAV_EXPLORER: "Search one element or analyte: which tanks have it, what else tends to be there, typical composition.",
    NAV_TANK_ATTRS: "Browse tank engineering metadata -- type, capacity, integrity, status -- joined from the attributes file.",
    NAV_TANK_EXPLORER: "Compare full composition across several chosen tanks side by side.",
    NAV_HEATMAPS: "Tank x element inventory as a color matrix, stoichiometrically apportioned -- abundance patterns across the whole dataset at once.",
    NAV_CORRELATIONS: "Which elements tend to occur together, statistically -- quick scan, kg-only association workbench, PCA/clustering.",
    NAV_VITRIFICATION: "Screen tanks for glass-formulation suitability -- oxide chemistry, candidate search, blend partners.",
    NAV_FIGURE_COMPOSER: "Combine plots captured from any workspace into one labeled, publication-ready multi-panel figure.",
}


class EmberMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        icon_path = asset_path("ember_logo.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 820)

        self.status_logger = StatusLogger()
        self.dataset = HanfordDataset(logger=self.status_logger.log)
        self.global_scope = GlobalScope(self)
        self.global_scope.changed.connect(self._on_scope_changed)

        self._build_toolbar()
        self._build_body()
        self._build_menus()

        self.statusBar().showMessage("Ready.")
        self._restore_settings()
        QTimer.singleShot(0, self._auto_load_on_startup)
        QTimer.singleShot(0, self._maybe_show_getting_started)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Data", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.load_local_action = toolbar.addAction("Load local CSVs", self._load_local)
        self.browse_action = toolbar.addAction("Load CSV/Parquet…", self._browse_and_load)
        self.reload_action = toolbar.addAction("Reload", self._reload)
        toolbar.addSeparator()

        self.use_cache_check = QCheckBox("Use parquet cache")
        self.use_cache_check.setChecked(True)
        toolbar.addWidget(self.use_cache_check)
        self.refresh_cache_check = QCheckBox("Refresh cache")
        toolbar.addWidget(self.refresh_cache_check)
        toolbar.addSeparator()

        toolbar.addAction("Open output folder", self._open_output_folder)

    def _build_body(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        from PySide6.QtWidgets import QHBoxLayout
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(190)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 12, 0, 0)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        for name in NAV_ITEMS:
            item = QListWidgetItem(name, self.nav)
            item.setToolTip(NAV_DESCRIPTIONS.get(name, ""))
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        sidebar_layout.addWidget(self.nav)
        outer.addWidget(sidebar)

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
        self.stack = QStackedWidget()
        self.overview_page = OverviewPage(self, scope=self.global_scope)
        self.stack.addWidget(self.overview_page)
        pump()
        self.query_page = QueryPage(self, scope=self.global_scope)
        self.stack.addWidget(self.query_page)
        pump()
        self.explorer_page = ExplorerPage(self)
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
        self.correlations_page = CorrelationsPage(self)
        self.stack.addWidget(self.correlations_page)
        pump()
        self.vitrification_page = VitrificationPage(self)
        self.stack.addWidget(self.vitrification_page)
        pump()
        self.figure_composer_page = FigureComposerPage(self)
        self.stack.addWidget(self.figure_composer_page)
        pump()

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self.scope_bar)
        right.addWidget(self.stack, 1)
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

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Load local CSVs", self._load_local)
        file_menu.addAction("Load CSV/Parquet…", self._browse_and_load)
        file_menu.addAction("Reload", self._reload, "F5")
        file_menu.addSeparator()
        file_menu.addAction("Open output folder", self._open_output_folder)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")

        view_menu = self.menuBar().addMenu("&View")
        self.dark_mode_action = view_menu.addAction("Dark mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self._on_dark_mode_toggled)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Getting Started…", self.show_getting_started)
        help_menu.addAction("About", self.show_about)
        help_menu.addAction("Notice", self.show_notice)
        help_menu.addSeparator()
        help_menu.addAction("Debug / Export…", self.show_debug)

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
            self, "Load CSV/Parquet", last_dir, "Data files (*.csv *.tsv *.txt *.parquet *.pq);;All files (*.*)"
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
        self.statusBar().showMessage("Looking for Hanford.csv next to the app…")
        run_in_thread(
            lambda: self.dataset.load_local_default(use_cache=True, refresh_cache=False),
            on_done=self._on_load_done,
            on_error=self._on_startup_load_error,
        )

    def _start_load(self, fn) -> None:
        self._set_busy(True)
        self.statusBar().showMessage("Loading…")
        run_in_thread(fn, on_done=self._on_load_done, on_error=self._on_load_error)

    def _set_busy(self, busy: bool) -> None:
        for action in (self.load_local_action, self.browse_action, self.reload_action):
            action.setEnabled(not busy)

    def _on_load_done(self, report) -> None:
        self._set_busy(False)
        self.statusBar().showMessage(
            f"Loaded {report.rows:,} rows x {report.columns} columns from {report.source_path.name}"
        )
        # Reconcile the scope bar FIRST -- pages refreshed below must see
        # the already-reconciled scope, not a stale filter pointing at
        # values the new dataset doesn't have. The coalesce block spans
        # the dispatch so any scope change from that reconciliation is
        # announced exactly once, AFTER every page has the new dataset.
        with self.global_scope.coalesce():
            self.scope_bar.on_dataset_changed(self.dataset)
            for page in self._dataset_aware_pages():
                if hasattr(page, "on_dataset_changed"):
                    page.on_dataset_changed(self.dataset)

    def _on_scope_changed(self) -> None:
        for page in self._dataset_aware_pages():
            if hasattr(page, "on_scope_changed"):
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
        self.statusBar().showMessage("No dataset loaded — use Load local CSVs or Load CSV/Parquet.")

    def _open_output_folder(self) -> None:
        folder = self.dataset.output_root
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder.resolve()))

    # ------------------------------------------------------------------
    # Nav / settings / misc
    # ------------------------------------------------------------------
    def _on_nav_changed(self, row: int) -> None:
        if not (0 <= row < len(NAV_ITEMS)):
            return
        page = self._pages_by_nav[NAV_ITEMS[row]]
        self.stack.setCurrentWidget(page)
        # The scope bar appears only above workspaces that honor it.
        self.scope_bar.setVisible(NAV_ITEMS[row] in self._scoped_nav_items())

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        # Restyles the Qt chrome only -- matplotlib plot areas stay white
        # so on-screen plots always match PNG/SVG/PDF export.
        from PySide6.QtWidgets import QApplication
        from qt_theme import apply_theme
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, dark=enabled)

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

    def show_about(self) -> None:
        HelpDialog(self, html=ABOUT_HTML, title=f"About {APP_NAME}").exec()

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
            dialog.resize(760, 560)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.debug_page)
            self._debug_dialog = dialog
        self._debug_dialog.show()
        self._debug_dialog.raise_()
        self._debug_dialog.activateWindow()

    def _restore_settings(self) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings(APP_NAME, APP_NAME)
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        dark = settings.value("dark_mode", False, type=bool)
        if dark:
            self.dark_mode_action.setChecked(True)
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
        super().closeEvent(event)
