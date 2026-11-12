"""
qt_correlations_workbench.py — the "Association Workbench (kg)" Correlations
sub-tab: kg-only element association screening for vitrification/blending
(Jaccard co-presence, preferred-association scoring, 18 plot types, coherent
colors, skip-elements). Ported from the old app's SeabornCorrelationTab.

What the tab owns:
- One question (a list of elements, or the top elements by mass) answered
  by correlation_science.kg_correlation_workbench on the house worker: the
  Kendall method or a long list takes seconds. Everything the worker needs
  is read from the widgets before it is submitted and the answer comes back
  to bound methods of this widget, tagged with the run it belongs to.
- Every label on the chart and every line of the exported settings comes
  from the parameters that PRODUCED the standing tables, never from the
  live controls. Display choices (chart type, colors, annotation, how many
  pairs or tanks) redraw at once and are read live.
- The notes above the result say, calmly, which typed elements carry no kg
  inventory and in which unit the dataset reports them instead.

The last section of this module holds the small helpers the three
element-list tabs of the workspace share (readable combo choices, the
comma-list completer, the reporting notes, mark-to-row selection).
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QCompleter, QDoubleSpinBox, QHBoxLayout, QLineEdit, QMenu, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import correlation_science as csci
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from elements import normalize_element_symbol
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
)
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

METRICS = ["log10_plus1", "log10_inventory", "fraction", "inventory", "presence"]
METHODS = ["pearson", "spearman", "kendall"]
COLOR_MODES = ["Basic", "Coherent colors"]
BAR_MODES = ["Stacked", "Grouped"]
BAR_SORT_MODES = ["Total (desc)", "Total (asc)", "Tank ID (A-Z)"]
PLOT_TYPES = [
    "Corr heatmap lower triangle", "Corr heatmap + total kg projections", "Jaccard co-presence heatmap",
    "Top preferred associations", "Top positive correlations", "Top negative correlations", "Top Jaccard co-presence",
    "Pair matrix regression", "Pair matrix scatter", "Pair matrix KDE",
    "Joint regression first two", "Joint scatter first two", "Joint KDE first two",
    "Tank similarity heatmap", "Tank x element heatmap", "Tank inventory bars",
    "Presence pattern bars", "Stats dashboard",
]
_TABLE_KEY_BY_TITLE = {
    "Element stats": "element_stats", "Pair stats": "pair_stats", "Metric matrix": "metric_matrix",
    "Raw kg matrix": "raw_matrix", "Tank similarity": "tank_similarity", "Skipped": "excluded_elements",
}
_TABLE_HINTS = {
    "Element stats": "One row per element: total kg, tanks holding it, share of the selection.",
    "Pair stats": "One row per pair of elements: correlation, co-presence and association score.",
    "Metric matrix": "One row per tank, one column per element, on the chosen metric.",
    "Raw kg matrix": "One row per tank, one column per element, in kg.",
    "Tank similarity": "Tank-to-tank correlation of the element profiles.",
    "Skipped": "Elements left out through the skip list.",
}

# Major constituents that carry a mass inventory in every tank farm. Cesium,
# technetium and iodine are reported as activity (Ci) only, so they belong
# on the Cross-correlation tab, which can place a nuclide beside a mass.
DEFAULT_ELEMENTS = "Na, Al, Fe, Si, P, S, Cr, F, Cl, U, Sr"
PAGE_TITLE = "Association workbench"
PAGE_SUBTITLE = (
    "Which elements are found together, tank by tank: correlation, co-presence and an association "
    "score, on mass inventory (kg) only."
)
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "List the elements to compare, then press Run. The Plot selector redraws the result at once."
SELECTION_USER = "User list"
SELECTION_TOP = "Top kg elements"


@dataclass
class _Outcome:
    """What the worker hands back, tagged with the run it belongs to."""

    token: int
    params: dict = field(default_factory=dict)
    results: Optional[dict] = None
    notes: List[str] = field(default_factory=list)
    refusal: str = ""
    error: str = ""


def science_arguments(params: dict) -> dict:
    """The keyword arguments of kg_correlation_workbench, from one run's parameters."""
    return {
        "elements_text": params["elements_text"], "selection_mode": params["selection_mode"],
        "top_n_elements": params["top_n_elements"], "value_mode": params["value_mode"],
        "method": params["method"], "min_inventory": params["min_inventory"],
        "include_zeros": params["include_zeros"], "skip_elements_text": params["skip_elements_text"],
    }


def compute_workbench(token: int, dataset: HanfordDataset, params: dict) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    try:
        results = csci.kg_correlation_workbench(dataset, **science_arguments(params))
        notes = list_notes(dataset, params, "kg")
    except ValueError as exc:
        return _Outcome(token=token, params=params, refusal=str(exc))
    except Exception:
        return _Outcome(token=token, params=params, error=traceback.format_exc())
    return _Outcome(token=token, params=params, results=results, notes=notes)


class WorkbenchTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.results: dict = {}
        # The settings that produced `results` (empty while there are none).
        self._params: dict = {}
        self._run_token = 0
        self._running = False
        self._busy_shown = False
        self._after_build: Optional[Callable[[], None]] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader(PAGE_TITLE, PAGE_SUBTITLE)
        self.export_btn = secondary_button(
            "Export...", "Write the tables and the chart, or one image of every chart type, to the output folder.")
        self.export_menu = QMenu(self.export_btn)
        self.export_menu.setToolTipsVisible(True)
        action = self.export_menu.addAction("Tables and current chart")
        action.setToolTip("Every table of the result, the settings that produced it, and the chart on screen.")
        action.triggered.connect(lambda _checked=False: self.export_tables())
        action = self.export_menu.addAction("Every chart type")
        action.setToolTip("The tables plus one image per chart type, with a manifest.")
        action.triggered.connect(lambda _checked=False: self.export_plot_suite())
        self.export_btn.clicked.connect(self._show_export_menu)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 10, "Correlations"))
        root.addWidget(self.header)

        # Control row: which elements, then Run; the chart selector at the
        # right. Everything else (selection mode, skip list, thresholds,
        # metric, method, per-chart tuning) lives in Advanced options.
        controls, row = control_row()
        row.addWidget(field_label("Elements"))
        self.elements_edit = QLineEdit(DEFAULT_ELEMENTS)
        self.elements_edit.setMinimumWidth(220)
        self.elements_edit.setToolTip(
            "Element symbols separated by commas. Only elements with a mass inventory (kg) take part.\n"
            "Not used when Selection (Advanced options) is set to the top elements by mass.")
        self.elements_edit.textChanged.connect(self._sync_run_label)
        self.elements_edit.returnPressed.connect(self.build_data)
        row.addWidget(self.elements_edit, 2)

        row.addSpacing(8)
        self.build_btn = QPushButton("Run")
        self.build_btn.setObjectName("Primary")
        self.build_btn.clicked.connect(self.build_data)
        self.run_btn = self.build_btn
        row.addWidget(self.build_btn)
        row.addStretch(1)

        row.addWidget(field_label("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.setCurrentText("Corr heatmap + total kg projections")
        self.plot_type_combo.setToolTip("How the result is drawn. Changing it redraws the standing result; nothing is recomputed.")
        self.plot_type_combo.currentTextChanged.connect(self._redraw)
        row.addWidget(self.plot_type_combo)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        adv_row1 = QHBoxLayout()
        adv_row1.setSpacing(8)
        adv_row1.addWidget(field_label("Selection"))
        self.selection_combo = QComboBox()
        self.selection_combo.addItems([SELECTION_USER, SELECTION_TOP])
        self.selection_combo.setToolTip(
            "User list: the elements typed above.\n"
            "Top kg elements: the elements with the largest mass inventory, however many Top elements says.")
        self.selection_combo.currentTextChanged.connect(self._on_selection_changed)
        adv_row1.addWidget(self.selection_combo)
        adv_row1.addWidget(field_label("Top elements"))
        self.top_kg_spin = QSpinBox()
        self.top_kg_spin.setRange(2, 500)
        self.top_kg_spin.setValue(25)
        self.top_kg_spin.setToolTip("How many elements the Top kg elements selection keeps, counted after the skip list.")
        self.top_kg_spin.valueChanged.connect(self._sync_run_label)
        adv_row1.addWidget(self.top_kg_spin)
        adv_row1.addWidget(field_label("Skip"))
        self.skip_edit = QLineEdit()
        self.skip_edit.setMaximumWidth(160)
        self.skip_edit.setPlaceholderText("e.g. Na, N")
        self.skip_edit.setToolTip("Elements left out even when the selection would include them.")
        self.skip_edit.textChanged.connect(self._sync_run_label)
        adv_row1.addWidget(self.skip_edit)
        adv_row1.addWidget(field_label("Minimum kg per record"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        self.min_inv_spin.setToolTip("Inventory records at or below this many kg are ignored.")
        self.min_inv_spin.valueChanged.connect(self._sync_run_label)
        adv_row1.addWidget(self.min_inv_spin)
        adv_row1.addStretch(1)
        adv.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        adv_row2.setSpacing(8)
        adv_row2.addWidget(field_label("Metric"))
        self.metric_combo = QComboBox()
        fill_choices(self.metric_combo, METRICS, METRIC_LABELS)
        self.metric_combo.setToolTip(METRIC_TOOLTIP)
        self.metric_combo.currentIndexChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.metric_combo)
        adv_row2.addWidget(field_label("Method"))
        self.method_combo = QComboBox()
        fill_choices(self.method_combo, METHODS, METHOD_LABELS)
        self.method_combo.setToolTip(METHOD_TOOLTIP)
        self.method_combo.currentIndexChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.method_combo)
        self.include_zeros_check = QCheckBox("Include tanks with none")
        self.include_zeros_check.setChecked(True)
        self.include_zeros_check.setToolTip(
            "On: a tank that holds none of an element counts as a zero.\n"
            "Off: a pair is correlated only over the tanks that hold both elements.")
        self.include_zeros_check.toggled.connect(self._sync_run_label)
        adv_row2.addWidget(self.include_zeros_check)
        adv_row2.addStretch(1)
        adv.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.setSpacing(8)
        adv_row3.addWidget(field_label("Colors"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(COLOR_MODES)
        self.color_mode_combo.setCurrentText("Coherent colors")
        self.color_mode_combo.setToolTip("Coherent colors: one color scheme across every chart type of this tab.")
        self.color_mode_combo.currentTextChanged.connect(self._redraw)
        adv_row3.addWidget(self.color_mode_combo)
        self.annotate_check = QCheckBox("Annotate heat maps")
        self.annotate_check.setToolTip("Write the value in every cell of a heat map.")
        self.annotate_check.toggled.connect(self._redraw)
        adv_row3.addWidget(self.annotate_check)
        self.show_projections_check = QCheckBox("Show total kg beside heat maps")
        self.show_projections_check.setChecked(True)
        self.show_projections_check.setToolTip("Bars of each element's total kg along the two edges of the heat map.")
        self.show_projections_check.toggled.connect(self._redraw)
        adv_row3.addWidget(self.show_projections_check)
        adv_row3.addWidget(field_label("Pair-matrix elements"))
        self.max_pair_spin = QSpinBox()
        self.max_pair_spin.setRange(2, 20)
        self.max_pair_spin.setValue(8)
        self.max_pair_spin.setToolTip(
            "The pair-matrix charts draw at most this many elements: a longer list gives panels too small to read.")
        self.max_pair_spin.valueChanged.connect(self._redraw_soon)
        adv_row3.addWidget(self.max_pair_spin)
        adv_row3.addWidget(field_label("Top pairs"))
        self.top_pairs_spin = QSpinBox()
        self.top_pairs_spin.setRange(1, 500)
        self.top_pairs_spin.setValue(30)
        self.top_pairs_spin.setToolTip("How many pairs the ranked-pair charts, the presence patterns and the dashboard show.")
        self.top_pairs_spin.valueChanged.connect(self._redraw_soon)
        adv_row3.addWidget(self.top_pairs_spin)
        adv_row3.addWidget(field_label("Top tanks"))
        self.top_tanks_spin = QSpinBox()
        self.top_tanks_spin.setRange(2, 500)
        self.top_tanks_spin.setValue(50)
        self.top_tanks_spin.setToolTip("How many tanks the tank charts show, largest selected inventory first.")
        self.top_tanks_spin.valueChanged.connect(self._redraw_soon)
        adv_row3.addWidget(self.top_tanks_spin)
        adv_row3.addStretch(1)
        adv.addLayout(adv_row3)

        adv_row4 = QHBoxLayout()
        adv_row4.setSpacing(8)
        adv_row4.addWidget(field_label("Tank inventory bars"))
        self.bar_mode_combo = QComboBox()
        self.bar_mode_combo.addItems(BAR_MODES)
        self.bar_mode_combo.setToolTip(
            "Stacked: one bar per tank, one segment per element; the length is the total.\n"
            "Grouped: one bar per element and tank, side by side, to compare magnitudes.")
        self.bar_mode_combo.currentTextChanged.connect(self._redraw)
        adv_row4.addWidget(self.bar_mode_combo)
        adv_row4.addWidget(field_label("Sort tanks"))
        self.bar_sort_combo = QComboBox()
        self.bar_sort_combo.addItems(BAR_SORT_MODES)
        self.bar_sort_combo.setToolTip("Order of the tanks on the Tank inventory bars chart.")
        self.bar_sort_combo.currentTextChanged.connect(self._redraw)
        adv_row4.addWidget(self.bar_sort_combo)
        self.bar_log_check = QCheckBox("Log scale")
        self.bar_log_check.setToolTip(
            "Applies to Grouped bars. On a log axis the segments of a stacked bar no longer add up.")
        self.bar_log_check.toggled.connect(self._redraw)
        adv_row4.addWidget(self.bar_log_check)
        adv_row4.addStretch(1)
        adv.addLayout(adv_row4)

        root.addWidget(self.advanced_section)

        self.notes_banner = NotesBanner()
        self.warnings_label = self.notes_banner         # the name every page uses for this slot
        root.addWidget(self.notes_banner)

        splitter = ResultSplitter()
        root.addWidget(splitter, 1)

        self.plot = PlotWidget()
        self.plot.markClicked.connect(self._on_mark_clicked)
        splitter.addWidget(self.plot)

        self.tables = QTabWidget()
        self._table_views: Dict[str, DataFrameTableView] = {}
        for title_ in _TABLE_KEY_BY_TITLE:
            view = DataFrameTableView(title=title_, max_rows_display=500)
            view.set_empty_text(EMPTY_TITLE, _TABLE_HINTS[title_])
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        splitter.addWidget(self.tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

        self._on_selection_changed()
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)

    def _show_export_menu(self) -> None:
        """The two exports of this tab, under the one Export button."""
        self.export_menu.popup(self.export_btn.mapToGlobal(QPoint(0, self.export_btn.height() + 2)))

    def _on_selection_changed(self, *_args) -> None:
        by_list = self.selection_combo.currentText() == SELECTION_USER
        self.elements_edit.setEnabled(by_list)
        self.top_kg_spin.setEnabled(not by_list)
        self._sync_run_label()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset  # kg-only by design; nothing else to sync from the dataset
        # Tables of the previous dataset must not stay on screen (nor be
        # exported into the new dataset's folder), and a run still in
        # flight on it must not land.
        self._run_token += 1
        self._after_build = None
        self._set_running(False)
        self._clear_result()
        symbols = dataset_values(dataset, "Element")
        set_list_completions(self.elements_edit, symbols)
        set_list_completions(self.skip_edit, symbols)

    def _clear_result(self) -> None:
        had_result = bool(self.results)
        self.results = {}
        self._params = {}
        if had_result:
            for view in self._table_views.values():
                view.set_dataframe(pd.DataFrame())
            self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self.notes_banner.set_notes([])
        self._sync_run_label()

    def _say(self, message: str) -> None:
        say(self.app_window, message)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def _read_params(self) -> dict:
        return {
            "elements_text": self.elements_edit.text(),
            "selection_mode": self.selection_combo.currentText(),
            "top_n_elements": self.top_kg_spin.value(),
            "value_mode": choice(self.metric_combo),
            "method": choice(self.method_combo),
            "min_inventory": self.min_inv_spin.value(),
            "include_zeros": self.include_zeros_check.isChecked(),
            "skip_elements_text": self.skip_edit.text(),
        }

    def _sync_run_label(self, *_args) -> None:
        """"Run" builds the tables; "Update" rebuilds them after a control moved under a standing result."""
        if not hasattr(self, "notes_banner") or self._running:
            return
        changed = bool(self.results) and self._read_params() != self._params
        self.build_btn.setText("Update" if changed else "Run")
        self.build_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else "Build the association tables for these elements and draw the selected chart.")

    def build_data(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._after_build = None
            self._say("Association workbench: no dataset is loaded yet.")
            return
        self._start(self._read_params())

    run = build_data

    def _start(self, params: dict) -> None:
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        params = dict(params)
        self._set_running(True)
        run_in_thread(
            lambda: compute_workbench(token, dataset, params),
            self._on_run_done, self._on_run_error,
        )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.build_btn.setEnabled(not running)
        if running:
            self.build_btn.setText("Running…")
        else:
            self._sync_run_label()
        set_busy = getattr(self.app_window, "set_busy", None)
        if callable(set_busy) and (running or self._busy_shown):
            set_busy("Building the association tables…" if running else None)
            self._busy_shown = running

    def _on_run_error(self, tb_text: str) -> None:
        # compute_workbench catches everything, so this is reached only if
        # the worker itself fails.
        self._after_build = None
        self._set_running(False)
        self._say("Association workbench failed.")
        QMessageBox.critical(self, f"{PAGE_TITLE} failed", tb_text)

    def _on_run_done(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return          # superseded by a newer run, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._after_build = None
            self._say("Association workbench failed.")
            QMessageBox.critical(self, f"{PAGE_TITLE} failed", outcome.error)
            return
        if outcome.refusal:
            # The standing result (if any) stays, with the labels of the
            # settings that produced it.
            self._after_build = None
            self._say(f"Association workbench not run: {outcome.refusal}")
            QMessageBox.warning(self, PAGE_TITLE, outcome.refusal)
            return
        self.results = outcome.results
        self._params = dict(outcome.params)
        for title_, key in _TABLE_KEY_BY_TITLE.items():
            self._table_views[title_].set_dataframe(self.results.get(key, pd.DataFrame()))
        self.notes_banner.set_result_notes(outcome.notes)
        self._draw()
        self._sync_run_label()
        n_elements = len(self._current_elements())
        self._say(f"Association workbench: tables built for {n_elements} element(s), kg.")
        follow_up, self._after_build = self._after_build, None
        if follow_up is not None:
            follow_up()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _current_elements(self) -> list:
        raw_matrix = self.results.get("raw_matrix") if self.results else None
        if raw_matrix is not None and not raw_matrix.empty:
            return [c for c in raw_matrix.columns if c != "WasteSiteId"]
        return csci.parse_element_list(self.elements_edit.text())

    def _ensure_results(self, then: Optional[Callable[[], None]] = None) -> bool:
        """True when tables are standing. Otherwise a build is started and
        `then` runs once it has landed (never after a refusal)."""
        if self.results:
            return True
        self._after_build = then
        self.build_data()
        return False

    def plot_selected(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Association workbench: no dataset is loaded yet.")
            return
        if not self._ensure_results():
            return              # the build draws the selected chart when it lands
        self._draw()

    def _redraw(self, *_args) -> None:
        """A display choice changed: redraw the standing result, compute nothing."""
        if self.results:
            self._draw()

    def _redraw_soon(self, *_args) -> None:
        """The same for a spin box, which fires once per step: one redraw after the last step."""
        if self.results:
            self.plot.request_redraw(self._redraw)

    def _draw(self) -> None:
        ptype = self.plot_type_combo.currentText()
        elements = self._current_elements()
        corr = self.results.get("corr_matrix", pd.DataFrame())
        jacc = self.results.get("jaccard_matrix", pd.DataFrame())
        element_stats = self.results.get("element_stats", pd.DataFrame())
        pair_stats = self.results.get("pair_stats", pd.DataFrame())
        metric_matrix = self.results.get("metric_matrix", pd.DataFrame())
        raw_matrix = self.results.get("raw_matrix", pd.DataFrame())
        tank_similarity = self.results.get("tank_similarity", pd.DataFrame())
        presence_matrix = self.results.get("presence_matrix", pd.DataFrame())
        # The metric of the RUN: a combo moved since must not relabel
        # tables it never touched.
        metric = self._params.get("value_mode") or choice(self.metric_combo)
        show_proj = self.show_projections_check.isChecked()
        color_mode = self.color_mode_combo.currentText()
        annotate = self.annotate_check.isChecked()
        top_pairs = self.top_pairs_spin.value()
        top_tanks = self.top_tanks_spin.value()
        max_pair = self.max_pair_spin.value()

        if ptype == "Corr heatmap lower triangle":
            ph.plot_seaborn_lower_triangle_matrix(self.plot, corr, f"Element correlation heatmap - kg only - {metric}", "Correlation r", center=0, annotate=annotate, totals=element_stats, projections=False, color_mode=color_mode)
        elif ptype == "Corr heatmap + total kg projections":
            title = "Element correlation heatmap" + (" + total kg projections" if show_proj else " - projections hidden") + f" - {metric}"
            ph.plot_seaborn_lower_triangle_matrix(self.plot, corr, title, "Correlation r", center=0, annotate=annotate, totals=element_stats, projections=show_proj, color_mode=color_mode)
        elif ptype == "Jaccard co-presence heatmap":
            title = "Jaccard co-presence heatmap - kg only" + (" + total kg projections" if show_proj else " - projections hidden")
            ph.plot_seaborn_lower_triangle_matrix(self.plot, jacc, title, "Jaccard co-presence", center=None, annotate=annotate, totals=element_stats, projections=show_proj, color_mode=color_mode)
        elif ptype == "Top preferred associations":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="preferred", color_mode=color_mode)
        elif ptype == "Top positive correlations":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="positive", color_mode=color_mode)
        elif ptype == "Top negative correlations":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="negative", color_mode=color_mode)
        elif ptype == "Top Jaccard co-presence":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="jaccard", color_mode=color_mode)
        elif ptype == "Pair matrix regression":
            ph.plot_seaborn_pair_matrix(self.plot, metric_matrix, raw_matrix, elements, metric, kind="regression", max_elements=max_pair, color_mode=color_mode)
        elif ptype == "Pair matrix scatter":
            ph.plot_seaborn_pair_matrix(self.plot, metric_matrix, raw_matrix, elements, metric, kind="scatter", max_elements=max_pair, color_mode=color_mode)
        elif ptype == "Pair matrix KDE":
            ph.plot_seaborn_pair_matrix(self.plot, metric_matrix, raw_matrix, elements, metric, kind="kde", max_elements=max_pair, color_mode=color_mode)
        elif ptype == "Joint regression first two":
            ph.plot_seaborn_joint_first_two(self.plot, metric_matrix, elements, metric, kind="regression", color_mode=color_mode)
        elif ptype == "Joint scatter first two":
            ph.plot_seaborn_joint_first_two(self.plot, metric_matrix, elements, metric, kind="scatter", color_mode=color_mode)
        elif ptype == "Joint KDE first two":
            ph.plot_seaborn_joint_first_two(self.plot, metric_matrix, elements, metric, kind="kde", color_mode=color_mode)
        elif ptype == "Tank similarity heatmap":
            ph.plot_seaborn_tank_similarity(self.plot, tank_similarity, raw_matrix, top_tanks=top_tanks, annotate=annotate, color_mode=color_mode)
        elif ptype == "Tank x element heatmap":
            ph.plot_seaborn_tank_element_map(self.plot, raw_matrix, elements, top_tanks=top_tanks, metric=metric, color_mode=color_mode)
        elif ptype == "Tank inventory bars":
            ph.plot_tank_element_bars(
                self.plot, raw_matrix, elements, top_n=top_tanks, mode=self.bar_mode_combo.currentText(),
                sort_by=self.bar_sort_combo.currentText(), log_scale=self.bar_log_check.isChecked(), color_mode=color_mode,
            )
        elif ptype == "Presence pattern bars":
            ph.plot_seaborn_presence_patterns(self.plot, presence_matrix, elements, top_n=top_pairs, color_mode=color_mode)
        elif ptype == "Stats dashboard":
            ph.plot_seaborn_stats_dashboard(self.plot, element_stats, pair_stats, top_n=top_pairs, color_mode=color_mode)
        else:
            self.plot.show_message(f"Unknown plot type: {ptype}")

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a cell, a bar or a point selects the row it was drawn from."""
        select_row_for_mark(
            self.tables,
            [self._table_views[t] for t in ("Element stats", "Pair stats", "Raw kg matrix", "Tank similarity")],
            mark)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _settings_table(self) -> pd.DataFrame:
        """The settings of the RUN, plus the display choices as they stand."""
        params = self._params
        return pd.DataFrame([{
            "unit": "kg", "selection_mode": params["selection_mode"],
            "elements_text": params["elements_text"], "skip_elements_text": params["skip_elements_text"],
            "metric": params["value_mode"], "method": params["method"],
            "top_elements": params["top_n_elements"], "min_inventory_kg_row": params["min_inventory"],
            "include_zeros": params["include_zeros"],
            "show_total_projections": self.show_projections_check.isChecked(),
            "color_mode": self.color_mode_combo.currentText(),
        }])

    def export_tables(self) -> None:
        if not self.results or self.dataset is None:
            self._say("Association workbench: run first; there is nothing to export yet.")
            return
        tables = dict(self.results)
        tables["settings"] = self._settings_table()
        out_dir = export_utils.export_named_tables(self.dataset, "seaborn_correlations", tables)
        try:
            self.plot.figure.savefig(out_dir / "current_seaborn_plot.png", bbox_inches="tight", dpi=220)
        except Exception:
            pass
        self._say(f"Association workbench exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Association workbench tables exported:\n{out_dir}")

    def export_plot_suite(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Association workbench: no dataset is loaded yet.")
            return
        if not self._ensure_results(then=self.export_plot_suite):
            return
        tables = dict(self.results)
        out_dir = export_utils.export_named_tables(self.dataset, "seaborn_plot_suite", tables)
        previous = self.plot_type_combo.currentText()
        manifest_rows = []
        # The selector is stepped silently: each chart is drawn once, here.
        self.plot_type_combo.blockSignals(True)
        try:
            for ptype in PLOT_TYPES:
                self.plot_type_combo.setCurrentText(ptype)
                try:
                    self._draw()
                    filename = f"{export_utils.safe_name(ptype)}.png"
                    self.plot.figure.savefig(out_dir / filename, bbox_inches="tight", dpi=220)
                    manifest_rows.append({"plot_type": ptype, "status": "ok", "file": filename})
                except Exception as exc:
                    manifest_rows.append({"plot_type": ptype, "status": "failed", "error": repr(exc), "file": ""})
            self.plot_type_combo.setCurrentText(previous)
        finally:
            self.plot_type_combo.blockSignals(False)
        self._draw()
        pd.DataFrame(manifest_rows).to_csv(out_dir / "plot_suite_manifest.csv", index=False)
        self._say(f"Plot suite exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Plot suite exported:\n{out_dir}")


# ----------------------------------------------------------------------
# Shared by the element-list tabs of the Correlations workspace
# ----------------------------------------------------------------------
METRIC_LABELS = {
    "log10_plus1": "Log10 of inventory + 1",
    "log10_inventory": "Log10 of inventory",
    "fraction": "Fraction of the selected inventory",
    "inventory": "Inventory",
    "presence": "Presence (0 or 1)",
}
METRIC_TOOLTIP = (
    "What is correlated, tank by tank.\n"
    "Log10 of inventory + 1: tanks holding none of an element stay in, as zero.\n"
    "Log10 of inventory: only the tanks that hold the element.\n"
    "Fraction of the selected inventory: the element's share of the tank's total over the selected elements.\n"
    "Inventory: the amount as reported.\n"
    "Presence: 1 where the tank holds the element, 0 where it does not."
)
METHOD_LABELS = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}
METHOD_TOOLTIP = (
    "Pearson: linear correlation of the metric.\n"
    "Spearman: rank correlation, insensitive to the scale of the metric.\n"
    "Kendall: rank correlation, more conservative with ties; the slowest of the three."
)
_TOKEN_SPLIT = re.compile(r"[,;\s]+")


class ResultSplitter(QSplitter):
    """Chart on the left, tables on the right.

    Its preferred height is its minimum height. The page sits in a scroll
    area that makes a page at least as tall as it would like to be, and a
    chart would like 400 px: with that preference an opened Advanced
    section or notes list pushed the page past the window and brought a
    page scroll bar, although the chart could simply have given up the
    room. The splitter still takes all the height the page has to spare.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self.setChildrenCollapsible(False)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(super().sizeHint().width(), self.minimumSizeHint().height())


def fill_choices(combo: QComboBox, identifiers: Sequence[str], labels: Dict[str, str]) -> None:
    """Readable items; the identifier the science call takes rides as item data."""
    for identifier in identifiers:
        combo.addItem(labels.get(identifier, identifier), identifier)


def choice(combo: QComboBox) -> str:
    """The identifier behind the current item (its text for an item added without one)."""
    data = combo.currentData()
    return str(data) if data is not None else combo.currentText()


def set_choice(combo: QComboBox, identifier: str) -> None:
    index = combo.findData(identifier)
    if index >= 0:
        combo.setCurrentIndex(index)


def say(app_window, message: str) -> None:
    """One sentence to the status bar of the window that hosts the page."""
    status_bar = getattr(app_window, "statusBar", None)
    if callable(status_bar):
        status_bar().showMessage(message)


class _ListCompleter(QCompleter):
    """Completes the entry being typed in a comma-separated field, keeping the entries before it."""

    def splitPath(self, path: str) -> List[str]:  # noqa: N802 (Qt naming)
        return [re.split(r"[,;]", path)[-1].strip()]

    def pathFromIndex(self, index) -> str:  # noqa: N802 (Qt naming)
        completion = super().pathFromIndex(index)
        widget = self.widget()
        text = widget.text() if isinstance(widget, QLineEdit) else ""
        cut = max(text.rfind(","), text.rfind(";"))
        if cut < 0:
            return completion
        return f"{text[:cut + 1]} {completion}"


def set_list_completions(edit: QLineEdit, items: Iterable[str]) -> None:
    """Offer `items` for the last entry of a comma-separated field (case-insensitive, from its start).

    Replaces the previous list: call it again when the dataset changes.
    """
    completer = _ListCompleter(sorted({str(item) for item in items if str(item)}, key=str.lower), edit)
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    completer.setCompletionMode(QCompleter.PopupCompletion)
    edit.setCompleter(completer)


def completion_items(edit: QLineEdit) -> List[str]:
    """What a field's completer offers (empty without one)."""
    completer = edit.completer()
    if completer is None:
        return []
    model = completer.model()
    return [str(model.data(model.index(i, 0))) for i in range(model.rowCount())]


def _and(names: Sequence[str]) -> str:
    names = [str(n) for n in names]
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def unrecognized_entries(text: str) -> List[str]:
    """Entries of an element list that are not element symbols (the science drops them silently)."""
    out: List[str] = []
    for token in _TOKEN_SPLIT.split(str(text or "").strip()):
        if token and not normalize_element_symbol(token) and token not in out:
            out.append(token)
    return out


def reporting_notes(dataset: HanfordDataset, elements: Sequence[str], unit: str,
                    consequence: str = "") -> List[str]:
    """One calm sentence per group of elements that carry no inventory in
    `unit`, naming the unit the dataset reports them in instead. The
    `consequence` follows the sentence when another unit does report them
    ("{units}" in it stands for that unit)."""
    try:
        units = list(dataset.available_units())
    except Exception:
        return []
    reported = {u: {e for e, total in csci.element_totals_by_unit(dataset, u).items() if total > 0}
                for u in dict.fromkeys(units + [unit])}
    missing = [e for e in dict.fromkeys(elements) if e not in reported.get(unit, set())]
    groups: Dict[tuple, List[str]] = {}
    for element in missing:
        elsewhere = tuple(u for u in reported if u != unit and element in reported[u])
        groups.setdefault(elsewhere, []).append(element)
    notes = []
    for elsewhere, names in groups.items():
        verb = "has" if len(names) == 1 else "have"
        subject = "it is" if len(names) == 1 else "they are"
        if elsewhere:
            sentence = (f"{_and(names)} {verb} no inventory in {unit} in this dataset: "
                        f"{subject} reported in {_and(elsewhere)} only.")
            sentence = f"{sentence} {consequence.replace('{units}', _and(elsewhere))}".strip()
        else:
            sentence = f"{_and(names)} {verb} no inventory in this dataset, in any unit."
        notes.append(sentence)
    return notes


def list_notes(dataset: HanfordDataset, params: dict, unit: str) -> List[str]:
    """The notes of an element-list run: entries that are not symbols, and
    elements the dataset does not report in `unit`. A selection made by the
    data (top elements by mass) needs neither."""
    if "top" in str(params.get("selection_mode", "")).lower():
        return []
    text = params.get("elements_text", "")
    notes = []
    unknown = unrecognized_entries(text)
    if unknown:
        verb = "is not an element symbol and was" if len(unknown) == 1 else "are not element symbols and were"
        notes.append(f"{_and(unknown)} {verb} left out.")
    skipped = set(csci.parse_element_list(params.get("skip_elements_text", "")))
    elements = [e for e in csci.parse_element_list(text) if e not in skipped]
    notes += reporting_notes(
        dataset, elements, unit,
        consequence="Such a column is zero in every tank and adds nothing to the result; the "
                    "Cross-correlation tab can place a nuclide beside a mass.")
    return notes


def mark_tokens(mark: dict) -> set:
    """The names a chart mark carries: its label, its row and column, its series, its tank."""
    tokens = set()
    for key in ("label", "row", "column", "series", "tank_id"):
        value = mark.get(key)
        if value in (None, ""):
            continue
        text = str(value)
        tokens.add(text)
        name = text.split(" (")[0].strip()       # "Al (n=176)" names Al
        tokens.add(name)
        tokens.update(part.strip() for part in text.split(", "))    # "241-A-101, Supernatant": tank and phase
        pair = name.split("-")                   # "Na-Cl" names a pair of elements
        if len(pair) == 2 and all(part.isalpha() and len(part) <= 2 for part in pair):
            tokens.update(pair)
    return tokens


def select_row_for_mark(tabs: QTabWidget, views: Sequence[DataFrameTableView], mark: dict) -> bool:
    """Select, in the first of `views` that matches best, the row a chart
    mark was drawn from: the row whose text cells name most of what the
    mark names. False when no table names any of it."""
    tokens = mark_tokens(mark)
    if not tokens:
        return False
    best_hits, best_view, best_row = 0, None, -1
    for view in views:
        frame = view.dataframe()
        if frame is None or frame.empty:
            continue
        text_columns = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        if not text_columns:
            continue
        hits = frame[text_columns].astype(str).isin(tokens).sum(axis=1).to_numpy()
        if hits.size and int(hits.max()) > best_hits:
            best_hits, best_view, best_row = int(hits.max()), view, int(hits.argmax())
    if best_view is None:
        return False
    tabs.setCurrentWidget(best_view)
    best_view.selectRow(best_row)
    index = best_view.proxy_model().mapFromSource(best_view.model().index(best_row, 0))
    if index.isValid():
        best_view.table.scrollTo(index)
    return True
