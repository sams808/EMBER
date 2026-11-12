"""
qt_correlations_structure.py — the "Structure" Correlations sub-tab: PCA +
hierarchical clustering of tanks, a partial-correlation matrix controlling
for tank size, and an element-association network graph, with interactive
Plotly HTML export. Entirely NEW capability -- no old-app equivalent to
port; reuses kg_correlation_workbench's element-selection controls (same
shape as the Association Workbench tab) since structure_science.
structure_workbench builds directly on top of it.

What the tab owns:
- One question answered by structure_science.structure_workbench on the
  house worker (the first run of a session, a long list or the Kendall
  method takes seconds). Everything the worker needs is read from the
  widgets before it is submitted; the answer comes back to bound methods
  of this widget, tagged with the run it belongs to.
- The exported settings describe the run that produced the tables, not
  the live controls. Display choices (chart type, what colors the PCA,
  colors, annotation) redraw at once and are read live.
- A click on a tank of the PCA scatter or of the dendrogram selects that
  tank in the Tank summary table; a double click opens the tank.
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QMenu, QMessageBox, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import html_export as he
import plot_helpers as ph
import structure_science as ssci
from data_model import HanfordDataset
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
)
from qt_correlations_workbench import (
    DEFAULT_ELEMENTS, METHOD_LABELS, METHOD_TOOLTIP, METRIC_LABELS, METRIC_TOOLTIP, SELECTION_TOP,
    SELECTION_USER, ResultSplitter, _and, choice, fill_choices, list_notes, say, select_row_for_mark,
    set_list_completions,
)
from qt_help import make_help_button
from qt_table_format import humanize
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

METRICS = ["log10_plus1", "log10_inventory", "fraction", "inventory", "presence"]
METHODS = ["pearson", "spearman", "kendall"]
COLOR_MODES = ["Basic", "Coherent colors"]
STRUCTURE_PLOT_TYPES = ["PCA scatter", "Dendrogram", "Partial vs raw correlation", "Element network"]
# What the PCA scatter can be colored by: nothing, a tank attribute, or
# the cluster the tank was assigned to. The column name rides as item data.
COLOR_BY_FIELDS = list(ssci.CATEGORY_FIELDS) + ["Cluster"]
_TABLE_KEY_BY_TITLE = {
    "Tank summary": "tank_summary", "PCA loadings": "pca_loadings", "PCA variance": "pca_variance",
    "Raw corr matrix": "raw_corr_matrix", "Partial corr matrix": "partial_corr_matrix",
    "Network nodes": "network_nodes", "Network edges": "network_edges",
    "Element stats": "element_stats", "Skipped": "excluded_elements",
}
_TABLE_HINTS = {
    "Tank summary": "One row per tank: PCA scores, cluster and tank attributes.",
    "PCA loadings": "How much each element weighs in the two principal components.",
    "PCA variance": "The share of the variance each principal component explains.",
    "Raw corr matrix": "Element-to-element correlation, as computed.",
    "Partial corr matrix": "The same correlation with the tank's total inventory held constant.",
    "Network nodes": "One row per element of the network, with its position and number of links.",
    "Network edges": "One row per pair of elements that passes both network thresholds.",
    "Element stats": "One row per element: total kg, tanks holding it, share of the selection.",
    "Skipped": "Elements left out through the skip list.",
}
_CLUSTER_METHOD_LABELS = {
    "ward": "Ward (compact clusters)", "average": "Average linkage",
    "complete": "Complete linkage", "single": "Single linkage",
}

PAGE_TITLE = "Structure"
PAGE_SUBTITLE = (
    "How the tanks group by composition: principal components, clusters, correlations with tank "
    "size held constant and an element network, on mass inventory (kg) only."
)
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "List the elements that describe a tank, then press Run. The Plot selector redraws the result at once."


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
    """The keyword arguments of structure_workbench, from one run's parameters."""
    return {
        "elements_text": params["elements_text"], "selection_mode": params["selection_mode"],
        "top_n_elements": params["top_n_elements"], "value_mode": params["value_mode"],
        "method": params["method"], "min_inventory": params["min_inventory"],
        "include_zeros": params["include_zeros"], "skip_elements_text": params["skip_elements_text"],
        "n_clusters": params["n_clusters"], "cluster_method": params["cluster_method"],
        "min_abs_r": params["min_abs_r"], "min_jaccard": params["min_jaccard"],
        "use_partial_for_network": params["use_partial_for_network"],
    }


def structure_notes(dataset: HanfordDataset, params: dict, results: dict) -> List[str]:
    """What qualifies a structure result: entries that are not symbols,
    elements without a kg inventory, and elements the PCA could not use."""
    notes = list_notes(dataset, params, "kg")
    dropped = [str(e) for e in (results.get("pca_dropped_elements") or [])]
    if dropped:
        verbs = "has the same value in every tank and was" if len(dropped) == 1 else \
            "have the same value in every tank and were"
        notes.append(f"{_and(dropped)} {verbs} left out of the principal components.")
    summary = results.get("tank_summary")
    raw = results.get("raw_matrix")
    if isinstance(raw, pd.DataFrame) and not raw.empty and isinstance(summary, pd.DataFrame) and summary.empty:
        notes.append("Too few tanks or elements for principal components and clusters: those tables "
                     "and charts stay empty.")
    return notes


def compute_structure(token: int, dataset: HanfordDataset, params: dict) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    try:
        results = ssci.structure_workbench(dataset, **science_arguments(params))
        notes = structure_notes(dataset, params, results)
    except ValueError as exc:
        return _Outcome(token=token, params=params, refusal=str(exc))
    except Exception:
        return _Outcome(token=token, params=params, error=traceback.format_exc())
    return _Outcome(token=token, params=params, results=results, notes=notes)


class StructureTab(QWidget):
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
            "Export...", "Write the tables and the chart, or the interactive HTML views, to the output folder.")
        self.export_menu = QMenu(self.export_btn)
        self.export_menu.setToolTipsVisible(True)
        action = self.export_menu.addAction("Tables and current chart")
        action.setToolTip("Every table of the result, the settings that produced it, and the chart on screen.")
        action.triggered.connect(lambda _checked=False: self.export_tables())
        action = self.export_menu.addAction("Interactive HTML views")
        action.setToolTip("The correlation heat map, the PCA scatter and the network as pages a browser opens.")
        action.triggered.connect(lambda _checked=False: self.export_html_views())
        self.export_btn.clicked.connect(self._show_export_menu)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 10, "Correlations"))
        root.addWidget(self.header)

        # Control row: which elements, then Run; the chart selector at the
        # right. Everything else (selection mode, thresholds, clustering
        # and network tuning) lives in Advanced options.
        controls, row = control_row()
        row.addWidget(field_label("Elements"))
        self.elements_edit = QLineEdit(DEFAULT_ELEMENTS)
        self.elements_edit.setMinimumWidth(220)
        self.elements_edit.setToolTip(
            "Element symbols separated by commas: the composition each tank is described by.\n"
            "Only elements with a mass inventory (kg) take part.\n"
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
        self.plot_type_combo.addItems(STRUCTURE_PLOT_TYPES)
        self.plot_type_combo.setToolTip(
            "PCA scatter: every tank on its two principal components.\n"
            "Dendrogram: the hierarchy the clusters are cut from.\n"
            "Partial vs raw correlation: element correlations with and without tank size held constant.\n"
            "Element network: the pairs that pass the network thresholds.\n"
            "Changing it redraws the standing result; nothing is recomputed.")
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        row.addWidget(self.plot_type_combo)
        row.addWidget(field_label("Color by"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItem("None", None)
        for column in COLOR_BY_FIELDS:
            self.color_by_combo.addItem(humanize(column), column)
        self.color_by_combo.setToolTip("What colors the tanks of the PCA scatter: a tank attribute, or the cluster.")
        self.color_by_combo.currentIndexChanged.connect(self._redraw)
        row.addWidget(self.color_by_combo)
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
        adv_row2.addWidget(field_label("Clusters"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(1, 20)
        self.n_clusters_spin.setValue(4)
        self.n_clusters_spin.setToolTip("How many clusters the hierarchy of tanks is cut into.")
        self.n_clusters_spin.valueChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.n_clusters_spin)
        adv_row2.addWidget(field_label("Linkage"))
        self.cluster_method_combo = QComboBox()
        fill_choices(self.cluster_method_combo, ssci.CLUSTER_METHODS, _CLUSTER_METHOD_LABELS)
        self.cluster_method_combo.setToolTip(
            "How the distance between two groups of tanks is measured while the hierarchy is built.")
        self.cluster_method_combo.currentIndexChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.cluster_method_combo)
        adv_row2.addStretch(1)
        adv.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.setSpacing(8)
        adv_row3.addWidget(field_label("Network: minimum |r|"))
        self.min_abs_r_spin = QDoubleSpinBox()
        self.min_abs_r_spin.setRange(0.0, 1.0)
        self.min_abs_r_spin.setSingleStep(0.05)
        self.min_abs_r_spin.setDecimals(2)
        self.min_abs_r_spin.setToolTip("A pair of elements is linked only when its correlation is at least this strong.")
        self.min_abs_r_spin.valueChanged.connect(self._sync_run_label)
        adv_row3.addWidget(self.min_abs_r_spin)
        adv_row3.addWidget(field_label("Minimum co-presence"))
        self.min_jaccard_spin = QDoubleSpinBox()
        self.min_jaccard_spin.setRange(0.0, 1.0)
        self.min_jaccard_spin.setSingleStep(0.05)
        self.min_jaccard_spin.setDecimals(2)
        self.min_jaccard_spin.setToolTip(
            "A pair is linked only when the two elements share at least this fraction of the tanks\n"
            "that hold either of them (Jaccard index).")
        self.min_jaccard_spin.valueChanged.connect(self._sync_run_label)
        adv_row3.addWidget(self.min_jaccard_spin)
        self.use_partial_check = QCheckBox("Links use partial correlation")
        self.use_partial_check.setToolTip(
            "Weigh the links by the correlation with tank size held constant, instead of the raw correlation.")
        self.use_partial_check.toggled.connect(self._sync_run_label)
        adv_row3.addWidget(self.use_partial_check)
        adv_row3.addWidget(field_label("Colors"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(COLOR_MODES)
        self.color_mode_combo.setCurrentText("Coherent colors")
        self.color_mode_combo.setToolTip("Coherent colors: one color scheme across every chart type of this tab.")
        self.color_mode_combo.currentTextChanged.connect(self._redraw)
        adv_row3.addWidget(self.color_mode_combo)
        self.annotate_check = QCheckBox("Annotate heat maps")
        self.annotate_check.setToolTip("Write the value in every cell of the correlation heat maps.")
        self.annotate_check.toggled.connect(self._redraw)
        adv_row3.addWidget(self.annotate_check)
        adv_row3.addStretch(1)
        adv.addLayout(adv_row3)

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

    def _on_plot_type_changed(self, *_args) -> None:
        # Only the PCA scatter is colored by a tank attribute.
        self.color_by_combo.setEnabled(self.plot_type_combo.currentText() == "PCA scatter")
        self._redraw()

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

    def _current_color_by(self) -> Optional[str]:
        data = self.color_by_combo.currentData()
        return str(data) if data else None

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
            "n_clusters": self.n_clusters_spin.value(),
            "cluster_method": choice(self.cluster_method_combo),
            "min_abs_r": self.min_abs_r_spin.value(),
            "min_jaccard": self.min_jaccard_spin.value(),
            "use_partial_for_network": self.use_partial_check.isChecked(),
        }

    def _sync_run_label(self, *_args) -> None:
        """"Run" builds the tables; "Update" rebuilds them after a control moved under a standing result."""
        if not hasattr(self, "notes_banner") or self._running:
            return
        changed = bool(self.results) and self._read_params() != self._params
        self.build_btn.setText("Update" if changed else "Run")
        self.build_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else "Build the structure tables for these elements and draw the selected chart.")

    def build_data(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._after_build = None
            self._say("Structure: no dataset is loaded yet.")
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
            lambda: compute_structure(token, dataset, params),
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
            set_busy("Computing the tank structure…" if running else None)
            self._busy_shown = running

    def _on_run_error(self, tb_text: str) -> None:
        # compute_structure catches everything, so this is reached only if
        # the worker itself fails.
        self._after_build = None
        self._set_running(False)
        self._say("Structure failed.")
        QMessageBox.critical(self, f"{PAGE_TITLE} failed", tb_text)

    def _on_run_done(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return          # superseded by a newer run, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._after_build = None
            self._say("Structure failed.")
            QMessageBox.critical(self, f"{PAGE_TITLE} failed", outcome.error)
            return
        if outcome.refusal:
            # The standing result (if any) stays, with the labels of the
            # settings that produced it.
            self._after_build = None
            self._say(f"Structure not run: {outcome.refusal}")
            QMessageBox.warning(self, PAGE_TITLE, outcome.refusal)
            return
        self.results = outcome.results
        self._params = dict(outcome.params)
        for title_, key in _TABLE_KEY_BY_TITLE.items():
            value = self.results.get(key, pd.DataFrame())
            self._table_views[title_].set_dataframe(value if isinstance(value, pd.DataFrame) else pd.DataFrame())
        self.notes_banner.set_result_notes(outcome.notes)
        self._draw()
        self._sync_run_label()
        n_elements = len([c for c in self.results.get("raw_matrix", pd.DataFrame()).columns if c != "WasteSiteId"])
        self._say(f"Structure: tables built for {n_elements} element(s), kg.")
        follow_up, self._after_build = self._after_build, None
        if follow_up is not None:
            follow_up()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
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
            self._say("Structure: no dataset is loaded yet.")
            return
        if not self._ensure_results():
            return              # the build draws the selected chart when it lands
        self._draw()

    def _redraw(self, *_args) -> None:
        """A display choice changed: redraw the standing result, compute nothing."""
        if self.results:
            self._draw()

    def _draw(self) -> None:
        ptype = self.plot_type_combo.currentText()
        color_mode = self.color_mode_combo.currentText()
        if ptype == "PCA scatter":
            ph.plot_pca_scatter(
                self.plot, self.results.get("tank_summary", pd.DataFrame()), self._current_color_by(),
                self.results.get("pca_variance", pd.DataFrame()), color_mode=color_mode,
            )
        elif ptype == "Dendrogram":
            ph.plot_dendrogram(
                self.plot, self.results.get("cluster_linkage"), self.results.get("cluster_labels", []),
                color_mode=color_mode,
            )
        elif ptype == "Partial vs raw correlation":
            ph.plot_partial_correlation_comparison(
                self.plot, self.results.get("partial_corr_matrix", pd.DataFrame()),
                self.results.get("raw_corr_matrix", pd.DataFrame()), annotate=self.annotate_check.isChecked(),
                color_mode=color_mode,
            )
        elif ptype == "Element network":
            ph.plot_element_network(
                self.plot, self.results.get("network_nodes", pd.DataFrame()),
                self.results.get("network_edges", pd.DataFrame()), color_mode=color_mode,
            )
        else:
            self.plot.show_message(f"Unknown plot type: {ptype}")

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a tank selects it in the Tank summary table; a click
        on an element or a pair selects its row in the element tables."""
        select_row_for_mark(
            self.tables,
            [self._table_views[t] for t in ("Tank summary", "Element stats", "Network edges")],
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
            "include_zeros": params["include_zeros"], "n_clusters": params["n_clusters"],
            "cluster_method": params["cluster_method"], "min_abs_r": params["min_abs_r"],
            "min_jaccard": params["min_jaccard"], "network_uses_partial_r": params["use_partial_for_network"],
            "color_pca_by": self._current_color_by() or "None", "color_mode": self.color_mode_combo.currentText(),
        }])

    def export_tables(self) -> None:
        if not self.results or self.dataset is None:
            self._say("Structure: run first; there is nothing to export yet.")
            return
        tables = {k: v for k, v in self.results.items() if isinstance(v, pd.DataFrame)}
        linkage_arr = self.results.get("cluster_linkage")
        if linkage_arr is not None and len(linkage_arr) > 0:
            tables["cluster_linkage"] = pd.DataFrame(linkage_arr, columns=["Cluster_A", "Cluster_B", "Distance", "N_samples"])
        tables["settings"] = self._settings_table()
        out_dir = export_utils.export_named_tables(self.dataset, "structure", tables)
        try:
            self.plot.figure.savefig(out_dir / "current_plot.png", bbox_inches="tight", dpi=220)
        except Exception:
            pass
        self._say(f"Structure exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Structure tables exported:\n{out_dir}")

    def export_html_views(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Structure: no dataset is loaded yet.")
            return
        if not self._ensure_results(then=self.export_html_views):
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.dataset.output_root / f"structure_html_{stamp}"
        try:
            he.export_correlation_heatmap_html(
                out_dir / "correlation_heatmap.html", self.results.get("raw_corr_matrix", pd.DataFrame()),
                title="Element correlation heatmap (kg)",
            )
            he.export_pca_scatter_html(
                out_dir / "pca_scatter.html", self.results.get("tank_summary", pd.DataFrame()),
                color_by=self._current_color_by(), pca_variance=self.results.get("pca_variance", pd.DataFrame()),
            )
            he.export_network_html(
                out_dir / "network_graph.html", self.results.get("network_nodes", pd.DataFrame()),
                self.results.get("network_edges", pd.DataFrame()),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "HTML export failed", str(exc))
            return
        os.startfile(str(out_dir.resolve()))
        self._say(f"Interactive HTML exported: {out_dir}")
