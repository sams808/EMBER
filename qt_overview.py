"""
qt_overview.py -- the Site overview: the landing page of the application.

Top to bottom: a header stating the scope in words, a row of headline
figures, the interactive tank-farm map with a ranked list of the mapped
quantity beside it (hover and selection are linked both ways), and the
dataset audit tables with their chart lower on the page.

Every number comes from the query engine or overview_science under the
global scope (site_summary.py holds the calls). The work runs through
qt_worker so the page paints at once and a scope change never freezes the
window; a result that arrives after a newer request is dropped. While the
page is hidden behind another workspace it does not recompute: it marks
itself stale and refreshes when shown.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QEvent, Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox, QCompleter, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

import matrix_query as mq
import overview_science as ov
import plot_helpers as ph
import site_summary as ss
import tank_science as tsci
from data_model import HanfordDataset
from plot_grammar import PlotSpec, render_plot
from qt_components import EmptyState, NotesBanner, PageHeader, StatTile, wrapped_label
from qt_farm_map import SCALES, FarmMapWidget, RankedTankList
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import DataFrameTableView, PlotWidget
from qt_worker import run_in_thread
from query_engine import BASES

PLOT_TYPES = ["Top elements by inventory", "Farm rollup by inventory"]

# The Overview unit picker offers "All" alongside the dataset's own units.
# For element totals that maps onto the engine's bridged mass basis rather
# than ranking kilograms against curies on one axis, which is not a
# comparison of anything.
_ELEMENT_BASIS = {"All": "g"}

DEFAULT_COMPONENT = "Na"
DEFAULT_BASIS = "kg"
SCALE_LABELS = {"quantile": "Quantile classes", "log": "Logarithmic", "linear": "Linear"}

# Wording of the engine's notes as the banner shows them: the sentences
# are the engine's, minus the names of internal tables and columns.
_PLAIN_WORDING = (
    ("reference_data.SPECIES_FORMULAS", "the species formula table"),
    ("reference_data.NUCLIDE_HALF_LIFE_YEARS", "the nuclide half-life table"),
    ("(Inventory_RSD = 0 means unassigned, not exact)",
     "(a relative standard deviation of 0 means unassigned, not exact)"),
    (" (harmonizer: Stage 4)", ""),
)


def _element_basis(unit: str) -> Optional[str]:
    """The engine basis for a Unit-combo entry, or None when the dataset
    reports in a unit the engine has no mass/activity semantics for. The
    combo is filled from the data, so a CSV using "MT" or "gal" is
    entirely possible -- passing it through as a basis raised straight out
    of refresh() and, since Overview is refreshed first, aborted the
    shell's whole dataset broadcast (review finding)."""
    basis = _ELEMENT_BASIS.get(unit, unit)
    return basis if basis in BASES else None


def plain_note(note: str) -> str:
    for old, new in _PLAIN_WORDING:
        note = note.replace(old, new)
    return note


# ---------------------------------------------------------------------------
# worker jobs: pure computation, no widgets
# ---------------------------------------------------------------------------
def _dashboard_job(token: int, dataset: HanfordDataset, scope: Dict[str, List[str]],
                   component: str, basis: str, with_stats: bool) -> dict:
    out: dict = {"token": token, "dataset": dataset, "scope": scope, "with_stats": with_stats,
                 "component": component, "basis": basis, "error": ""}
    if with_stats:
        try:
            out["stats"] = ss.site_stats(dataset, scope)
            out["tanks_all"] = ss.tank_table(dataset, None)
            out["tanks_scope"] = ss.tank_table(dataset, scope) if scope else out["tanks_all"]
        except ValueError as exc:
            # A scope the dataset cannot honour (an attribute filter with
            # no attributes file): say so, calmly, instead of numbers.
            out["error"] = str(exc)
            return out
    try:
        out["values"] = ss.tank_values(dataset, component, basis, scope)
    except ValueError as exc:
        out["values"] = None
        out["error"] = str(exc)
    return out


def _tables_job(token: int, dataset: HanfordDataset, scope: Dict[str, List[str]],
                unit: str, top_n: int, farm_unit: Optional[str]) -> dict:
    out: dict = {"token": token, "dataset": dataset, "unit": unit, "top_n": top_n,
                 "farm_unit": farm_unit}
    out["overview"] = ov.overview(dataset)
    out["units"] = ov.units_audit(dataset)
    # Element totals come from the query engine, so this page agrees
    # with Heatmaps about what the waste contains: each element gets
    # its stoichiometric share of every species reporting it, and
    # "All" means kg and Ci bridged onto one mass axis rather than
    # ranked against each other.
    basis = _element_basis(unit)
    if basis is None:
        out["element_totals"] = mq.ElementTotals(warnings=[
            f"Element totals need a mass or activity basis; this dataset reports in "
            f"{unit!r}, which the engine has no chemistry for. The audit tables below "
            f"are unaffected."
        ])
    else:
        try:
            out["element_totals"] = mq.element_totals(dataset, basis=basis, top_n=top_n, scope=scope)
        except ValueError as exc:
            out["element_totals"] = mq.ElementTotals(warnings=[str(exc)])
    out["top_analytes"] = ov.top_analytes(dataset, unit=unit, top_n=top_n)
    out["phases"] = ov.phase_audit(dataset)
    out["types"] = ov.type_audit(dataset)
    rollup = tsci.farm_inventory_rollup(dataset, unit=farm_unit) if farm_unit else pd.DataFrame()
    out["farm_rollup"] = rollup
    out["farms"] = rollup if not rollup.empty else ov.farm_audit(dataset)
    out["missing"] = ov.missing_audit(dataset)
    out["raw"] = dataset.raw_preview(250)
    return out


class OverviewPage(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.scope = scope if scope is not None else GlobalScope(self)
        self.dataset: Optional[HanfordDataset] = None
        self._element_totals = mq.ElementTotals()
        self._farm_rollup_df = pd.DataFrame()
        self._farm_rollup_unit_used: Optional[str] = None
        self._tank_values: Optional[ss.TankValues] = None
        self._stats: Optional[ss.SiteStats] = None
        self._dashboard_error = ""
        self._dashboard_token = 0
        self._tables_token = 0
        self._pending = 0
        self._stale = False
        self._linking = False
        self._requested_key: Optional[tuple] = None
        self._tanks_scope: Optional[pd.DataFrame] = None
        self._announce = False
        # Deferred work goes through timers owned by the page, so nothing
        # can fire on a page that has been deleted.
        self._fit_timer = self._deferred(self._fit_main_block)
        self._plot_timer = self._deferred(self._update_plot)
        self._refresh_timer = self._deferred(lambda: self.refresh(announce=False))
        self._build_ui()

    def _deferred(self, slot) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(0)
        timer.timeout.connect(slot)
        return timer

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll)
        content = QWidget()
        self.scroll.setWidget(content)
        self.scroll.viewport().installEventFilter(self)
        root = QVBoxLayout(content)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)
        self._root = root

        self.header = PageHeader("Site overview", "No dataset loaded.")
        self.export_btn = QPushButton("Export...")
        self.export_btn.setObjectName("Secondary")      # the quiet header action, as on every page
        self.export_btn.setToolTip(
            "Write the audit tables, the element totals, the tank table and the mapped values "
            "to a dated folder of CSV files.")
        self.export_btn.clicked.connect(self._export_debug_bundle)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 5, "Overview"))
        root.addWidget(self.header)

        # The engine's caveats: one calm line, the detail one click away.
        self.notes_banner = NotesBanner()
        self.notes_banner.toggle.toggled.connect(lambda _checked: self._fit_main_block_later())
        self.warnings_label = self.notes_banner
        root.addWidget(self.notes_banner)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_tanks = StatTile("Tanks")
        self.tile_volume = StatTile("Waste volume")
        self.tile_records = StatTile("Records")
        self.tile_analytes = StatTile("Analytes")
        self.tile_sodium = StatTile("Sodium")
        self.tile_activity = StatTile("Cesium, strontium")
        self.stat_tiles = [self.tile_tanks, self.tile_volume, self.tile_records,
                           self.tile_analytes, self.tile_sodium, self.tile_activity]
        for tile in self.stat_tiles:
            tiles.addWidget(tile, 1)
        self._tiles_row = tiles
        root.addLayout(tiles)

        self.main_block = QWidget()
        main = QHBoxLayout(self.main_block)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(16)
        main.addWidget(self._build_map_card(), 2)
        main.addWidget(self._build_rank_card(), 1)
        self.main_block.setFixedHeight(480)
        root.addWidget(self.main_block, 1)

        root.addWidget(self._build_tables_section())

        self._set_tiles_blank()
        self._show_map_message("No dataset loaded",
                               "Load the tank inventory to see the tank farms, the headline "
                               "figures and the dataset tables.")

    def _build_map_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 8)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        title = QLabel("Tank farms")
        title.setObjectName("SectionTitle")
        row.addWidget(title)
        row.addSpacing(12)

        label = QLabel("Colour by")
        label.setObjectName("FieldLabel")
        row.addWidget(label)
        self.component_combo = QComboBox()
        self.component_combo.setEditable(True)
        self.component_combo.setInsertPolicy(QComboBox.NoInsert)
        self.component_combo.setMinimumWidth(150)
        self.component_combo.setToolTip(
            "What the map shows: the waste volume, an element (Na, Al), an analyte (SO4, NO3) "
            "or a radionuclide (137Cs). Type to search, Enter to apply.")
        self.component_combo.setAccessibleName("Component shown on the map")
        self.component_combo.addItems([ss.WASTE_VOLUME, DEFAULT_COMPONENT])
        self.component_combo.setCurrentText(DEFAULT_COMPONENT)
        self.component_combo.activated.connect(lambda _i: self._on_colour_changed())
        self.component_combo.lineEdit().editingFinished.connect(self._on_colour_changed)
        row.addWidget(self.component_combo)

        label = QLabel("Basis")
        label.setObjectName("FieldLabel")
        row.addWidget(label)
        self.basis_combo = QComboBox()
        for key, basis in ss.MAP_BASES.items():
            self.basis_combo.addItem(basis.label, key)
        self.basis_combo.setToolTip(
            "Inventory in kg (activities converted to mass) or Ci, or a concentration: "
            "mol per mol Na, wt% of waste mass, g per litre of waste.")
        self.basis_combo.setAccessibleName("Basis of the mapped value")
        self.basis_combo.currentIndexChanged.connect(lambda _i: self._on_colour_changed())
        row.addWidget(self.basis_combo)

        label = QLabel("Scale")
        label.setObjectName("FieldLabel")
        row.addWidget(label)
        self.scale_combo = QComboBox()
        for key in SCALES:
            self.scale_combo.addItem(SCALE_LABELS[key], key)
        self.scale_combo.setToolTip(
            "How values become colour classes. Quantile: equal numbers of tanks per class, "
            "from the tanks holding at least 100 kL of waste.")
        self.scale_combo.setAccessibleName("Colour scale")
        self.scale_combo.currentIndexChanged.connect(lambda _i: self._apply_values())
        row.addWidget(self.scale_combo)
        row.addStretch(1)
        self.busy_label = QLabel("")
        self.busy_label.setObjectName("SectionNote")
        row.addWidget(self.busy_label)
        layout.addLayout(row)

        self.map_stack = QStackedWidget()
        self.map_empty = EmptyState("No dataset loaded")
        self.farm_map = FarmMapWidget()
        self.map_stack.addWidget(self.map_empty)
        self.map_stack.addWidget(self.farm_map)
        layout.addWidget(self.map_stack, 1)

        self.farm_map.selectionChanged.connect(self._on_map_selection)
        self.farm_map.tankHovered.connect(self._on_map_hover)
        self.farm_map.tankActivated.connect(self._open_tank)
        return card

    def _build_rank_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(6)
        title = QLabel("Ranked tanks")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.rank_subtitle = wrapped_label("", "SectionNote")
        layout.addWidget(self.rank_subtitle)

        self.rank_stack = QStackedWidget()
        self.rank_empty = EmptyState("Nothing to rank yet")
        self.rank_list = RankedTankList()
        self.rank_stack.addWidget(self.rank_empty)
        self.rank_stack.addWidget(self.rank_list)
        layout.addWidget(self.rank_stack, 1)

        self.rank_note = wrapped_label("", "SectionNote")
        self.rank_note.setVisible(False)
        layout.addWidget(self.rank_note)

        self.selection_label = wrapped_label(
            "Click a tank to select it, double-click to open it.", "SectionNote")
        layout.addWidget(self.selection_label)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)
        self.open_btn = QPushButton("Open in Tank explorer")
        self.open_btn.setObjectName("Secondary")
        self.open_btn.setToolTip("Open the selected tank as a profile, or compare the selected tanks.")
        self.open_btn.clicked.connect(self._open_selection)
        self.open_btn.setVisible(False)
        footer.addWidget(self.open_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("Secondary")
        self.clear_btn.clicked.connect(lambda: self._set_selection([], source=None))
        self.clear_btn.setVisible(False)
        footer.addWidget(self.clear_btn)
        layout.addLayout(footer)

        self.rank_list.selectionChanged_.connect(self._on_list_selection)
        self.rank_list.tankHovered.connect(self._on_list_hover)
        self.rank_list.tankActivated.connect(self._open_tank)
        return card

    def _build_tables_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        title = QLabel("Dataset tables")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        layout.addWidget(wrapped_label(
            "Audit of the loaded dataset: units, element and analyte totals, waste phases and "
            "types, farms, missing values. Element totals follow the scope; the audit tables "
            "describe the whole dataset.", "SectionNote"))

        controls = QHBoxLayout()
        controls.setSpacing(8)
        label = QLabel("Basis")
        label.setObjectName("FieldLabel")
        controls.addWidget(label)
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("All")
        self.unit_combo.setToolTip(
            "Unit of the element and analyte totals. All: kilograms and curies brought onto "
            "one mass axis.")
        controls.addWidget(self.unit_combo)
        label = QLabel("Rows")
        label.setObjectName("FieldLabel")
        controls.addWidget(label)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 300)
        self.top_n_spin.setValue(30)
        self.top_n_spin.setToolTip("How many elements and analytes the top tables and the chart keep.")
        controls.addWidget(self.top_n_spin)
        label = QLabel("Chart")
        label.setObjectName("FieldLabel")
        controls.addWidget(label)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.currentTextChanged.connect(lambda _: self._update_plot())
        controls.addWidget(self.plot_type_combo)
        self.refresh_btn = QPushButton("Update")
        self.refresh_btn.setToolTip("Recompute the page with these settings.")
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        controls.addWidget(self.refresh_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        self.tables = QTabWidget()
        self._table_views: dict[str, DataFrameTableView] = {}
        for title_text, max_rows in [
            ("Overview", None), ("Units", None), ("Top elements", None),
            ("Top analytes", None), ("Waste phases", None), ("Waste types", None),
            ("Tank farms", None), ("Missing", None), ("Raw preview", 250),
            ("Notes", 200),
        ]:
            view = DataFrameTableView(title=title_text, max_rows_display=max_rows)
            self._table_views[title_text] = view
            self.tables.addTab(view, title_text)
        splitter.addWidget(self.tables)
        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setFixedHeight(540)
        layout.addWidget(splitter, 1)
        self._tables_splitter = splitter
        return section

    # ------------------------------------------------------------------
    # layout: the map block fills the first screen, the tables follow
    # ------------------------------------------------------------------
    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt naming)
        if watched is self.scroll.viewport() and event.type() == QEvent.Resize:
            self._fit_main_block_later()
        return super().eventFilter(watched, event)

    def _fit_main_block_later(self) -> None:
        self._fit_timer.start()

    def _fit_main_block(self) -> None:
        viewport = self.scroll.viewport().height()
        if viewport <= 0:
            return
        # The header, the notes line and the tiles above the block: their
        # heights, not the block's current position, which lags the layout.
        above = self._root.contentsMargins().top()
        for widget in (self.header, self.notes_banner, self.tile_tanks):
            if not widget.isHidden():
                above += max(widget.height(), widget.sizeHint().height()) + self._root.spacing()
        target = max(440, viewport - above - 20)
        # A fixed height, not a minimum: inside a scroll area a block is
        # otherwise given its preferred height, which ignores the window.
        if abs(self.main_block.height() - target) > 2 or self.main_block.minimumHeight() != target:
            self.main_block.setFixedHeight(target)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._fit_main_block_later()
        if self._stale:
            self._stale = False
            self._refresh_timer.start()

    def _defer_while_hidden(self) -> bool:
        """True when the page sits hidden behind another workspace of a
        window that is on screen: computing now would be work nobody sees."""
        window = self.window()
        return window is not self and window.isVisible() and not self.isVisible()

    # ------------------------------------------------------------------
    # dataset and scope
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        current_unit = self.unit_combo.currentText()
        self.unit_combo.blockSignals(True)
        self.unit_combo.clear()
        self.unit_combo.addItem("All")
        units = dataset.available_units() if dataset.is_loaded() else []
        self.unit_combo.addItems(units)
        restore_index = self.unit_combo.findText(current_unit)
        self.unit_combo.setCurrentIndex(restore_index if restore_index >= 0 else 0)
        self.unit_combo.blockSignals(False)
        self._tank_values = None
        self._requested_key = None
        if not dataset.is_loaded():
            self._set_tiles_blank()
            self.header.set_subtitle("No dataset loaded.")
            self._show_map_message("No dataset loaded",
                                   "Load the tank inventory to see the tank farms, the headline "
                                   "figures and the dataset tables.")
            return
        self._fill_component_choices(dataset)
        if self._defer_while_hidden():
            self._stale = True
            return
        self.refresh(announce=False)

    def _fill_component_choices(self, dataset: HanfordDataset) -> None:
        current = self.component_combo.currentText().strip() or DEFAULT_COMPONENT
        choices = ss.component_choices(dataset)
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self.component_combo.addItems(choices)
        if current not in choices:
            current = DEFAULT_COMPONENT if DEFAULT_COMPONENT in choices else ss.WASTE_VOLUME
        self.component_combo.setCurrentText(current)
        completer = QCompleter(choices, self.component_combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.component_combo.setCompleter(completer)
        self.component_combo.blockSignals(False)

    def on_scope_changed(self) -> None:
        """Element totals, tiles and the map are scoped, so this page must
        follow the scope bar or it contradicts Query and Heatmaps about
        the same element (review finding: Na read 46.2 Mkg here against
        6.2 Mkg there)."""
        if self.dataset is None or not self.dataset.is_loaded():
            return
        if self._defer_while_hidden():
            self._stale = True
            return
        self.refresh(announce=False)

    # ------------------------------------------------------------------
    # refresh: two jobs, the dashboard first
    # ------------------------------------------------------------------
    def refresh(self, announce: bool = True) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Site overview", "Load a dataset first.")
            return
        self._stale = False
        self._start_dashboard(with_stats=True)
        self._start_tables(announce)

    def _colour_request(self) -> tuple:
        component = self.component_combo.currentText().strip() or DEFAULT_COMPONENT
        basis = self.basis_combo.currentData() or DEFAULT_BASIS
        return component, basis

    def _set_busy(self, delta: int) -> None:
        self._pending = max(0, self._pending + delta)
        self.busy_label.setText("Updating..." if self._pending else "")

    def _start_dashboard(self, with_stats: bool) -> None:
        component, basis = self._colour_request()
        scope = self.scope.as_scope()
        self._requested_key = (component, basis, repr(sorted(scope.items())))
        self._dashboard_token += 1
        if self._tank_values is None and self.map_stack.currentWidget() is not self.farm_map:
            self._show_map_message("Loading the site overview",
                                   "Reading tank volumes and inventories.")
        self._set_busy(+1)
        run_in_thread(_dashboard_job, self._on_dashboard_done, self._on_job_error,
                      self._dashboard_token, self.dataset, scope, component, basis, with_stats)

    def _start_tables(self, announce: bool) -> None:
        unit = self.unit_combo.currentText() or "All"
        top_n = self.top_n_spin.value()
        self._tables_token += 1
        self._announce = bool(announce)
        self._set_busy(+1)
        run_in_thread(_tables_job, self._on_tables_done, self._on_job_error,
                      self._tables_token, self.dataset, self.scope.as_scope(), unit, top_n,
                      self._farm_rollup_unit())

    @Slot(str)
    def _on_job_error(self, traceback_text: str) -> None:
        self._set_busy(-1)
        last = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "unknown error"
        self._dashboard_error = f"The overview could not be computed: {last}"
        self._update_notes()
        self.app_window.statusBar().showMessage("Site overview: the last update failed.")

    @Slot(object)
    def _on_dashboard_done(self, result: dict) -> None:
        self._set_busy(-1)
        if result["token"] != self._dashboard_token or result["dataset"] is not self.dataset:
            return  # a newer request is on its way
        self._dashboard_error = result.get("error", "")
        if result["with_stats"]:
            if "stats" not in result:
                # The scope cannot be applied to this dataset: no figures.
                self._stats, self._tank_values = None, None
                self._set_tiles_blank()
                self._show_map_message("The overview cannot be shown", self._dashboard_error)
                self._update_notes()
                return
            self._stats = result["stats"]
            self._tanks_scope = result["tanks_scope"]
            self._apply_stats(result["stats"], result["scope"])
            self.farm_map.set_tanks(result["tanks_all"])
            self.farm_map.set_scope_tanks(
                [str(t) for t in result["tanks_scope"]["WasteSiteId"]] if result["scope"] else None)
        if self._stats is None:
            return
        self._tank_values = result.get("values")
        self.map_stack.setCurrentWidget(self.farm_map)
        self._apply_values()
        self._update_notes()

    @Slot(object)
    def _on_tables_done(self, result: dict) -> None:
        self._set_busy(-1)
        if result["token"] != self._tables_token or result["dataset"] is not self.dataset:
            return
        element_totals = result["element_totals"]
        self._element_totals = element_totals
        self._table_views["Overview"].set_dataframe(result["overview"])
        self._table_views["Units"].set_dataframe(result["units"])
        self._table_views["Top elements"].set_dataframe(element_totals.table)
        self._table_views["Top analytes"].set_dataframe(result["top_analytes"])
        self._table_views["Waste phases"].set_dataframe(result["phases"])
        self._table_views["Waste types"].set_dataframe(result["types"])
        self._table_views["Tank farms"].set_dataframe(result["farms"])
        self._table_views["Missing"].set_dataframe(result["missing"])
        self._table_views["Raw preview"].set_dataframe(result["raw"])

        self._farm_rollup_df = result["farm_rollup"]
        # The unit the cached rollup was actually computed in -- the plot
        # must label with THIS, never the live combo, which the user may
        # have changed since the last Refresh (a "(kg)" title over Ci bars
        # is a silently wrong chart).
        self._farm_rollup_unit_used = result["farm_unit"]

        # The engine's warnings and disclosures belong ON the page: this
        # is the first screen anyone sees, and its headline table carries
        # exclusions (trims, unattributable mass, isotopic restatements).
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in element_totals.warnings]
            + [{"kind": "disclosure", "text": d} for d in element_totals.disclosures]
        )
        self._table_views["Notes"].set_dataframe(notes)
        self._update_notes()
        # The chart is drawn on the next turn of the event loop: filling the
        # tables and drawing the figure are then two short steps, not one long one.
        self._plot_timer.start()
        if self._announce:
            self.app_window.statusBar().showMessage(
                f"Site overview updated ({result['unit']}, top {result['top_n']}).")

    # ------------------------------------------------------------------
    # applying results
    # ------------------------------------------------------------------
    def _set_tiles_blank(self) -> None:
        for tile in self.stat_tiles:
            tile.set_value(ss.BLANK, "")

    def _apply_stats(self, stats: ss.SiteStats, scope: Dict[str, Sequence[str]]) -> None:
        self.header.set_subtitle(ss.scope_sentence(scope, stats.n_tanks, stats.n_tanks_total))
        # A tile is narrow: short second lines, the full statement in the tooltip.
        if stats.n_single_shell is not None:
            split = f"{stats.n_single_shell:,} SST · {stats.n_double_shell:,} DST"
            self.tile_tanks.setToolTip(
                f"{stats.n_single_shell:,} single-shell tanks (SST) and "
                f"{stats.n_double_shell:,} double-shell tanks (DST)"
                + (f", of {stats.n_tanks_total:,} in the dataset." if stats.n_tanks != stats.n_tanks_total else "."))
        else:
            split = "tank system not stated"
            self.tile_tanks.setToolTip("The dataset carries no tank attributes: single or double shell is not stated.")
        if stats.n_tanks != stats.n_tanks_total:
            split = f"of {stats.n_tanks_total:,} · " + split
        self.tile_tanks.set_value(f"{stats.n_tanks:,}", split)

        if stats.volume_kL is None:
            self.tile_volume.set_value(ss.BLANK, "no volumes in this dataset")
        else:
            volume = stats.volume_kL
            text = (f"{ss.format_number(volume / 1000.0, 4)} ML" if volume >= 10_000
                    else f"{ss.format_number(volume, 4)} kL")
            self.tile_volume.set_value(text, ss.format_gallons(volume))

        records_hint = f"{stats.n_layers:,} waste layers" if stats.n_layers else ""
        if stats.n_records != stats.n_records_total:
            records_hint = f"of {stats.n_records_total:,}" + (f" · {stats.n_layers:,} layers" if stats.n_layers else "")
        self.tile_records.set_value(f"{stats.n_records:,}", records_hint)
        self.tile_records.setToolTip(
            f"{stats.n_records:,} inventory records in {stats.n_layers:,} waste layers "
            f"(a layer is one waste phase and waste type in one tank)."
            if stats.n_layers else f"{stats.n_records:,} inventory records.")
        self.tile_analytes.set_value(f"{stats.n_analytes:,}", f"{stats.n_elements:,} elements")

        if stats.sodium_kg is None:
            self.tile_sodium.set_value(ss.BLANK, "not reported in scope")
        else:
            hint = (f"{ss.format_number(stats.sodium_wt_pct)} wt% of waste"
                    if stats.sodium_wt_pct is not None else "as Na")
            self.tile_sodium.set_value(ss.format_mass(stats.sodium_kg), hint)

        total, split, sentence = ss.format_activity_split(
            [("137Cs", stats.cs137_ci), ("90Sr", stats.sr90_ci)])
        self.tile_activity.set_value(total, split or "not reported in scope")
        self.tile_activity.setToolTip(
            f"Activity of {sentence}, as reported (the decay dates of the records differ)."
            if sentence else "Neither 137Cs nor 90Sr is reported in scope.")

    def _show_map_message(self, title: str, hint: str = "") -> None:
        self.map_empty.set_text(title, hint)
        self.map_stack.setCurrentWidget(self.map_empty)
        self.rank_empty.set_text("Nothing to rank yet", "")
        self.rank_stack.setCurrentWidget(self.rank_empty)
        self.rank_subtitle.setText("")
        self.rank_note.setVisible(False)

    def _apply_values(self) -> None:
        """Push the mapped quantity to the map and the ranked list."""
        values = self._tank_values
        scale = self.scale_combo.currentData() or "quantile"
        is_volume = self.component_combo.currentText().strip().lower() == ss.WASTE_VOLUME.lower()
        self.basis_combo.setEnabled(not is_volume)
        if values is None:
            self.farm_map.set_values({}, label="", unit="", scale=scale)
            self.rank_empty.set_text("Nothing to rank", self._dashboard_error)
            self.rank_stack.setCurrentWidget(self.rank_empty)
            self.rank_subtitle.setText("")
            self.rank_note.setVisible(False)
            return
        self.farm_map.set_values(values.values, label=values.label, unit=values.unit,
                                 scale=scale, additive=values.additive)
        if not values.values:
            hint = " ".join(values.disclosures[:1] + values.warnings[:1])
            self.rank_empty.set_text(f"No tank in scope reports {values.label} on this basis", hint)
            self.rank_stack.setCurrentWidget(self.rank_empty)
        else:
            self.rank_list.set_table(values.table, values.label, values.unit)
            self.rank_stack.setCurrentWidget(self.rank_list)
        n_scope = len(self._tanks_scope) if self._tanks_scope is not None else len(values.values)
        without = max(0, n_scope - len(values.values))
        n_values = len(values.values)
        subtitle = (f"{values.label}, {values.unit} · {n_values:,} "
                    f"tank{'s' if n_values != 1 else ''} with a record")
        if without:
            subtitle += f" · {without:,} without"
        self.rank_subtitle.setText(subtitle)
        # On the activity basis nothing is converted, so the engine's remark
        # about the mass bridge of a nuclide does not qualify these numbers.
        remarks = [plain_note(d) for d in values.disclosures
                   if not (values.basis == "Ci" and "bridged" in d)]
        self.rank_note.setText(" ".join(remarks))
        self.rank_note.setVisible(bool(remarks))
        self._set_selection(self.farm_map.selection(), source="map")

    def _update_notes(self) -> None:
        notes: List[str] = []
        if self._dashboard_error:
            notes.append(self._dashboard_error)
        for note in list(self._element_totals.warnings) + (
                list(self._tank_values.warnings) if self._tank_values is not None else []):
            note = plain_note(note)
            if note not in notes:
                notes.append(note)
        if not notes:
            self.notes_banner.set_notes([])
        elif len(notes) == 1:
            self.notes_banner.set_notes(notes)
        else:
            self.notes_banner.set_notes(
                notes, headline=f"{len(notes)} notes qualify the figures on this page: what was "
                                f"set aside, converted or left out, and why.")
        self._fit_main_block_later()

    # ------------------------------------------------------------------
    # colour-by controls
    # ------------------------------------------------------------------
    def _on_colour_changed(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            return
        component, basis = self._colour_request()
        key = (component, basis, repr(sorted(self.scope.as_scope().items())))
        if key == self._requested_key:
            return
        self._start_dashboard(with_stats=False)

    def set_colour_by(self, component: str, basis: Optional[str] = None,
                      scale: Optional[str] = None) -> None:
        """Public: map a component (used by links from other workspaces)."""
        self.component_combo.blockSignals(True)
        self.basis_combo.blockSignals(True)
        self.component_combo.setCurrentText(component)
        if basis is not None:
            index = self.basis_combo.findData(basis)
            if index >= 0:
                self.basis_combo.setCurrentIndex(index)
        self.component_combo.blockSignals(False)
        self.basis_combo.blockSignals(False)
        if scale is not None:
            index = self.scale_combo.findData(scale)
            if index >= 0:
                self.scale_combo.setCurrentIndex(index)
        self._on_colour_changed()

    # ------------------------------------------------------------------
    # linked hover and selection
    # ------------------------------------------------------------------
    def _on_map_selection(self, tanks: list) -> None:
        self._set_selection(tanks, source="map")

    def _on_list_selection(self, tanks: list) -> None:
        self._set_selection(tanks, source="list")

    def _set_selection(self, tanks: Sequence[str], source: Optional[str]) -> None:
        if self._linking:
            return
        self._linking = True
        try:
            tanks = list(tanks)
            if source != "map":
                self.farm_map.set_selection(tanks)
            if source != "list":
                self.rank_list.set_selection(tanks, scroll=source == "map")
        finally:
            self._linking = False
        self._update_selection_summary(tanks)

    def selected_tanks(self) -> List[str]:
        return self.farm_map.selection()

    def _update_selection_summary(self, tanks: Sequence[str]) -> None:
        has = bool(tanks)
        self.open_btn.setVisible(has)
        self.clear_btn.setVisible(has)
        if not has:
            self.selection_label.setText("Click a tank to select it, double-click to open it.")
            return
        self.open_btn.setText("Open in Tank explorer" if len(tanks) == 1 else "Compare in Tank explorer")
        text = tanks[0] if len(tanks) == 1 else f"{len(tanks)} tanks selected"
        values = self._tank_values
        if values is not None:
            numbers = [values.values[t] for t in tanks if t in values.values]
            if numbers:
                if len(numbers) == 1:
                    text += f" · {ss.format_number(numbers[0])} {values.unit}"
                elif values.additive:
                    text += f" · {ss.format_number(sum(numbers))} {values.unit} in total"
                else:
                    ordered = sorted(numbers)
                    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (
                        (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0)
                    text += f" · median {ss.format_number(median)} {values.unit}"
        self.selection_label.setText(text)

    def _on_map_hover(self, tank: str) -> None:
        self.rank_list.highlight(tank)

    def _on_list_hover(self, tank: str) -> None:
        self.farm_map.highlight([tank] if tank else [])

    def _open_tank(self, tank: str) -> None:
        opener = getattr(self.app_window, "open_tank", None)
        if callable(opener):
            opener(tank)
        else:
            self.app_window.statusBar().showMessage(f"{tank}: the Tank explorer is not available here.")

    def _open_selection(self) -> None:
        tanks = self.farm_map.selection()
        if not tanks:
            return
        many = getattr(self.app_window, "open_tanks", None)
        if len(tanks) > 1 and callable(many):
            many(tanks)
            return
        self._open_tank(tanks[0])
        explorer = getattr(self.app_window, "tank_explorer_page", None)
        if len(tanks) > 1 and explorer is not None and hasattr(explorer, "show_tanks"):
            explorer.show_tanks(tanks)

    # ------------------------------------------------------------------
    # dataset tables: chart and export
    # ------------------------------------------------------------------
    def _farm_rollup_unit(self) -> Optional[str]:
        """Farm rollup needs one concrete unit (kg and Ci can't be summed
        together) -- honor an explicit kg/Ci choice, otherwise default to
        kg (falling back to whatever's actually available) the same way
        Quick Scan/Heatmaps default their own Unit picker."""
        if self.dataset is None or not self.dataset.is_loaded():
            return None
        units = self.dataset.available_units()
        if not units:
            return None
        current = self.unit_combo.currentText()
        if current in units:
            return current
        return "kg" if "kg" in units else units[0]

    def _update_plot(self) -> None:
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "Farm rollup by inventory":
            unit = self._farm_rollup_unit_used
            if self._farm_rollup_df.empty or not unit:
                self.plot.show_message("No farm rollup data for this dataset.")
                return
            ph.plot_barh(
                self.plot, self._farm_rollup_df, "TankFarm", f"Total_inventory_{unit}",
                f"Total inventory by tank farm ({unit})", f"Total inventory ({unit})",
                top_n=50,
            )
        else:
            totals = self._element_totals
            if totals.table.empty:
                self.plot.show_message("No element totals for this dataset.")
                return
            subtitle = ["Apportioned (stoichiometric)"]
            if self.scope.summary():
                subtitle.append(f"Scope: {self.scope.summary()}")
            # Trims and exclusions belong ON the figure: the exported PNG
            # is what ends up in a report, and it carried no trace of a
            # top-N cut (review finding).
            subtitle.extend(w for w in totals.warnings if "omitted" in w)
            render_plot(self.plot, totals.table, PlotSpec(
                mark="bar", x=totals.value_column, y="Element",
                title="Top elements by inventory",
                subtitle=" · ".join(subtitle),
                log_x=True, top_n=len(totals.table),
            ))

    def _export_debug_bundle(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Export", "Load a dataset first.")
            return
        # Hand the bundle the SAME element totals the page is showing, or
        # the archive and the screen disagree about what the waste holds.
        # A file named "_all" must not hold the on-screen top N, and the
        # bundle must carry the warnings that qualify its own numbers
        # (review finding).
        basis = _element_basis(self.unit_combo.currentText() or "All")
        untrimmed = (
            mq.element_totals(self.dataset, basis=basis, top_n=10_000,
                              scope=self.scope.as_scope())
            if basis else self._element_totals
        )
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in untrimmed.warnings]
            + [{"kind": "disclosure", "text": d} for d in untrimmed.disclosures]
        )
        extra = {
            "top_elements_all": untrimmed.table if not untrimmed.table.empty else None,
            "notes": notes if not notes.empty else None,
        }
        values = self._tank_values
        if values is not None and not values.table.empty:
            mapped = values.table.copy()
            mapped.insert(1, "Component", values.label)
            mapped.insert(2, "Unit", values.unit)
            extra["map_values"] = mapped
        out_dir = ov.export_global_debug_bundle(
            self.dataset, extra_info=self._app_extra_info(), extra_tables=extra,
        )
        QMessageBox.information(self, "Export complete", f"Tables written to:\n{out_dir}")

    def _app_extra_info(self) -> dict:
        try:
            from qt_help import APP_NAME, APP_VERSION
            return {"app_name": APP_NAME, "app_version": APP_VERSION}
        except Exception:
            return {}
