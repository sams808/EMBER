"""
qt_correlations_association.py — the "Cross-correlation" Correlations
sub-tab: the UI over association_query (which components occur TOGETHER
across the tank farm).

What separates it from the three legacy sub-tabs beside it:

- It is engine-backed. Members go through the query engine, so a curie-only
  nuclide (129I, 79Se) can sit beside sulfate, the default basis is a
  concentration rather than an inventory, and every caveat the engine
  raises is shown (notes banner + Notes tab).
- It honors the global scope bar, the same way Query and Heatmaps do: the
  shared GlobalScope is read when a run starts, a scope change re-asks the
  standing question, and the scope a result was computed under is stamped
  on the figure, in Notes and in the export.
- The computation (several engine queries plus a pairwise replicate
  search, a few seconds on the full dataset) runs on the house worker.
  Everything the worker needs is read from the widgets BEFORE it is
  submitted, the worker function touches no widget, and results come back
  through bound methods of this widget -- a signal connected to a plain
  function or lambda would run on the worker thread, where building a
  dialog hangs.

Every label on the figure, in Notes and in the export is taken from the
parameters that PRODUCED the standing result, never from the live
controls: a combo changed after a run must not relabel numbers it never
touched.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

import association_query as aq
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
)
from qt_correlations_workbench import ResultSplitter, say, select_row_for_mark, set_list_completions
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread
from query_engine import BASKET_PRESETS

TAB_TITLE = "Cross-correlation"
DEFAULT_PRESET = "Salt formers (as reported)"
DEFAULT_COMPONENTS = "F, Cl, 129I, SO4, PO4, Cr"
ALL_FAMILIES = "All layers"
PLOT_MATRIX = "Correlation matrix"
PLOT_SCATTER = "Pair scatter"
PLOT_KINDS = [PLOT_MATRIX, PLOT_SCATTER]

T_PAIRS = "Pairs"
T_MATRIX = "Correlation matrix"
T_COUNTS = "N per pair"
T_QUALITY = "What each component rests on"
T_COOCCURRENCE = "Co-occurrence"
T_VALUES = "Values"
T_NOTES = "Notes"
TABLE_TITLES = [T_PAIRS, T_MATRIX, T_COUNTS, T_QUALITY, T_COOCCURRENCE, T_VALUES, T_NOTES]
# Rows looked at when sizing a table's columns (see _build_ui).
_RESIZE_ROWS = 40
_EXPORT_NAMES = {
    T_PAIRS: "association_pairs", T_MATRIX: "association_correlation_matrix",
    T_COUNTS: "association_n_per_pair", T_QUALITY: "association_component_quality",
    T_COOCCURRENCE: "association_co_occurrence", T_VALUES: "association_values",
    T_NOTES: "association_notes",
}
COMPONENT_COLUMN = "Component"
PAGE_SUBTITLE = (
    "Which components occur together across the waste: elements, species and activity-only "
    "nuclides side by side, on a concentration basis, within the global scope."
)
EMPTY_TITLE = "No result yet"
_IDLE_HINT = "Choose components and press Run."
_TABLE_HINTS = {
    T_PAIRS: "One row per pair of components, strongest correlation first.",
    T_MATRIX: "The coefficient of every pair, as a matrix.",
    T_COUNTS: "How many observations report both components of each pair.",
    T_QUALITY: "Which analytes each component is, and how well it is reported.",
    T_COOCCURRENCE: "Where several components are high at the same time.",
    T_VALUES: "One row per waste layer or tank: the values that were correlated.",
    T_NOTES: "The scope, warnings and disclosures of the result.",
}
# The settings a run is made of: a change to any of them makes "Run" read "Update".
_RUN_KEYS = ("components", "basis", "level", "family", "method", "distinct", "min_n", "quantile")

METHOD_LABELS = {
    "spearman": "Spearman",
    "pearson (log10)": "Pearson on log10 values",
    "kendall": "Kendall",
}
COEFFICIENT_LABELS = {
    "spearman": "Spearman r",
    "pearson (log10)": "Pearson r (log10 values)",
    "kendall": "Kendall tau",
}

_DISTINCT_TOOLTIP = (
    "Many layers are one waste-type template composition multiplied by a volume.\n"
    "Counting every copy inflates N and manufactures significance, so layers whose\n"
    f"components all agree within {100 * aq.REPLICATE_REL_TOL:.1f}% are counted as one observation."
)
_DISTINCT_KG_TOOLTIP = (
    "Not available on the kg basis: an inventory scales with the volume it was\n"
    "written into, so two copies of one composition never look alike. Every row is counted."
)
_FAMILY_TOOLTIP = "Restrict the correlation to one phase family (Supernatant / Saltcake / Sludge)."
_FAMILY_TANK_TOOLTIP = (
    "Not available at the tank level: a tank sums layers of several phases,\n"
    "so it has no single phase family."
)


# Engine warnings that change how THIS matrix reads. Everything else the
# engine says is true of the dataset whatever the question (unresolvable
# trace organics, mixed decay baselines, unassigned uncertainties): it
# belongs in Notes, not in a four-line banner above every result.
_RESULT_SPECIFIC_MARKS = ("Not one quantity", "left blank", "no single molar mass")


def banner_text(own_warnings: List[str], engine_warnings: List[str]) -> str:
    """The line above the result: what bears on reading this matrix, in
    full, then a count of the dataset-wide notes kept in the Notes tab
    (and in the tooltip). Nothing is dropped -- only moved one click away."""
    specific = [w for w in engine_warnings if any(mark in w for mark in _RESULT_SPECIFIC_MARKS)]
    shown = list(own_warnings) + specific
    n_other = len(engine_warnings) - len(specific)
    parts = ["  •  ".join(shown)] if shown else []
    if n_other:
        parts.append(f"{n_other} dataset note(s) apply to every result — see the Notes tab.")
    return "  •  ".join(parts)


def parse_components(text: str) -> List[str]:
    """Comma-separated component list (semicolons and line breaks are
    accepted too). Labels are kept verbatim: the engine decides what an
    entry means and refuses what it cannot answer."""
    out: List[str] = []
    for chunk in str(text or "").replace(";", ",").replace("\n", ",").split(","):
        name = chunk.strip()
        if name and name not in out:
            out.append(name)
    return out


def counted_rows(matrix: aq.AssociationMatrix, family: Optional[str], distinct_only: bool) -> pd.DataFrame:
    """The rows association_query.correlate() computes on: the selected
    phase family, then one row per distinct composition. Reproduced here
    (with the engine's own replicate_groups) because AssociationResult
    reports how many rows it used, not which -- and the pair scatter has
    to draw the observations behind the coefficient, not all of them."""
    values = matrix.values
    if family is not None and "PhaseFamily" in values.columns:
        values = values[values["PhaseFamily"] == family]
    if distinct_only and matrix.basis != "kg" and len(values):
        groups = aq.replicate_groups(values, matrix.members)
        values = values[~groups.duplicated()]
    return values


@dataclass
class _Outcome:
    """What the worker hands back. A refusal and an unexpected failure
    travel as data, tagged with the run they belong to, so the UI thread
    can drop an answer that a newer question has superseded."""

    token: int
    params: Optional[dict] = None
    matrix: Optional[aq.AssociationMatrix] = None
    result: Optional[aq.AssociationResult] = None
    quality: Optional[pd.DataFrame] = None
    co_occurrence: Optional[pd.DataFrame] = None
    used: Optional[pd.DataFrame] = None
    refusal: str = ""
    error: str = ""


def compute_association(token: int, dataset: HanfordDataset, params: dict,
                        scope: Dict[str, List[str]]) -> _Outcome:
    """Pure computation -- runs on the worker thread, touches no widget."""
    try:
        matrix = aq.association_matrix(
            dataset, params["components"], basis=params["basis"], level=params["level"], scope=scope,
        )
        result = aq.correlate(
            matrix, method=params["method"], distinct_only=params["distinct"],
            family=params["family"], min_n=params["min_n"],
        )
        quality = aq.member_quality(dataset, matrix.members, scope=scope)
        co = aq.co_occurrence(matrix, quantile=params["quantile"])
        used = counted_rows(matrix, params["family"], params["distinct"])
    except ValueError as exc:
        # The engine's refusals are readable sentences, not failures.
        return _Outcome(token=token, params=params, refusal=str(exc))
    except Exception:
        return _Outcome(token=token, params=params, error=traceback.format_exc())
    return _Outcome(token=token, params=params, matrix=matrix, result=result,
                    quality=quality, co_occurrence=co, used=used)


class AssociationTab(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        # The shell's shared global scope, handed down by CorrelationsPage;
        # standalone construction (tests, tooling) gets a private, empty one.
        self.scope = scope if scope is not None else GlobalScope(self)
        self.dataset: Optional[HanfordDataset] = None
        self._matrix: Optional[aq.AssociationMatrix] = None
        self._result: Optional[aq.AssociationResult] = None
        self._used = pd.DataFrame()
        self._tables: Dict[str, pd.DataFrame] = {}
        # Every setting that produced the standing result, scope included.
        self._params: dict = {}
        self._run_token = 0
        self._running = False
        self._busy_shown = False
        self._filling = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader(TAB_TITLE, PAGE_SUBTITLE)
        self.export_btn = secondary_button(
            "Export...", "Write every table of the result, its notes, the settings and the chart to the output folder.")
        self.export_btn.clicked.connect(self.export_tables)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 10, "Correlations"))
        root.addWidget(self.header)

        # Control row: what to correlate, on which basis, at which level --
        # then Run; the chart selector at the right. Everything else lives
        # in Advanced options.
        controls, row = control_row()
        row.addWidget(field_label("Components"))
        self.components_edit = QLineEdit(DEFAULT_COMPONENTS)
        self.components_edit.setMinimumWidth(170)
        self.components_edit.setToolTip(
            "Comma-separated elements, species or nuclides, e.g. F, Cl, SO4, 129I.\n"
            "Iodine and selenium are reported site-wide only as 129I and 79Se: name the nuclide."
        )
        self.components_edit.textChanged.connect(self._sync_run_label)
        self.components_edit.returnPressed.connect(self.run)
        row.addWidget(self.components_edit, 2)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem(DEFAULT_PRESET, DEFAULT_COMPONENTS)
        self.preset_combo.setItemData(
            0, "Halides, sulfate, phosphate, chromium, the two volatile nuclides and uranium.",
            role=Qt.ToolTipRole)
        for name, basket in BASKET_PRESETS.items():
            self.preset_combo.addItem(name, ", ".join(basket.members))
            self.preset_combo.setItemData(self.preset_combo.count() - 1, basket.description,
                                          role=Qt.ToolTipRole)
        self.preset_combo.setToolTip("Fill the component list from a shipped preset.")
        self.preset_combo.activated.connect(self._on_preset_activated)
        row.addWidget(self.preset_combo)

        row.addSpacing(8)
        row.addWidget(field_label("Basis"))
        self.basis_combo = QComboBox()
        for index, basis in enumerate(aq.BASES):
            self.basis_combo.addItem(basis)
            self.basis_combo.setItemData(index, aq.BASIS_LABELS[basis], role=Qt.ToolTipRole)
        self.basis_combo.setCurrentText(aq.MOL_PER_MOL_NA)
        self.basis_combo.setToolTip(
            "mol/mol Na, wt% and g/L are concentrations: they compare compositions.\n"
            "kg is an inventory: every component then scales with how much waste a\n"
            "tank holds, so every pair correlates and the matrix describes tank size."
        )
        self.basis_combo.currentTextChanged.connect(self._sync_enabled)
        row.addWidget(self.basis_combo)

        row.addWidget(field_label("Level"))
        self.level_combo = QComboBox()
        for key in aq.LEVEL_KEYS:
            self.level_combo.addItem(aq.LEVEL_LABELS[key], key)
        self.level_combo.setCurrentIndex(self.level_combo.findData("layer"))
        self.level_combo.setToolTip(
            "Waste layer: one (tank, phase, waste type) entry — the level a composition is stated at.\n"
            "Tank: its layers summed, which averages unrelated chemistries (saltcake over sludge)."
        )
        self.level_combo.currentIndexChanged.connect(self._sync_enabled)
        row.addWidget(self.level_combo)

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
            "Correlation matrix: every pair at once.\n"
            "Pair scatter: the pair selected in the Pairs table (the strongest pair if none is selected)."
        )
        self.plot_combo.currentTextChanged.connect(self._on_plot_kind_changed)
        row.addWidget(self.plot_combo)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()
        adv_row = QHBoxLayout()
        adv_row.setSpacing(8)
        adv_row.addWidget(field_label("Method"))
        self.method_combo = QComboBox()
        for method in aq.METHODS:
            self.method_combo.addItem(METHOD_LABELS[method], method)
        self.method_combo.setToolTip(
            "Spearman: rank correlation, keeps reported zeros as ties.\n"
            "Pearson on log10 values: linear correlation of log10 values — needs both values positive.\n"
            "Kendall: rank correlation, more conservative with many ties."
        )
        self.method_combo.currentIndexChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.method_combo)
        adv_row.addWidget(field_label("Phase family"))
        self.family_combo = QComboBox()
        self.family_combo.addItems([ALL_FAMILIES] + list(aq.PHASE_FAMILIES))
        self.family_combo.currentTextChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.family_combo)
        self.distinct_check = QCheckBox("Count each distinct composition once")
        self.distinct_check.setChecked(True)
        self.distinct_check.toggled.connect(self._sync_run_label)
        adv_row.addWidget(self.distinct_check)
        adv_row.addWidget(field_label("Minimum shared observations"))
        self.min_n_spin = QSpinBox()
        self.min_n_spin.setRange(3, 100000)
        self.min_n_spin.setValue(8)
        self.min_n_spin.setToolTip("A pair with fewer observations reporting BOTH components is left blank.")
        self.min_n_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.min_n_spin)
        adv_row.addWidget(field_label("Co-occurrence quantile"))
        self.quantile_spin = QDoubleSpinBox()
        self.quantile_spin.setRange(0.05, 0.99)
        self.quantile_spin.setSingleStep(0.05)
        self.quantile_spin.setDecimals(2)
        self.quantile_spin.setValue(0.75)
        self.quantile_spin.setToolTip(
            "Co-occurrence table: a component counts as high in a layer (or tank) when it\n"
            "sits above this quantile of its own values across the matrix."
        )
        self.quantile_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.quantile_spin)
        adv_row.addStretch(1)
        adv.addLayout(adv_row)
        root.addWidget(self.advanced_section)

        # Engine warnings are data-integrity statements -- always on the
        # page with the result, never tucked into a tab: one calm line
        # (banner_text), the full list one click away and in Notes.
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
        for title_ in TABLE_TITLES:
            # No display cap: a cap cuts a table short without saying so, and
            # Values / Co-occurrence carry one row per layer by design.
            view = DataFrameTableView(title=title_, max_rows_display=None)
            # Column widths from the first rows only. Measuring every cell
            # of the per-layer tables (Qt's default looks at 1000 rows)
            # costs seconds on the UI thread, which is exactly what the
            # worker exists to avoid.
            view.horizontalHeader().setResizeContentsPrecision(_RESIZE_ROWS)
            view.set_empty_text(EMPTY_TITLE, _TABLE_HINTS[title_])
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        self.tables.setTabToolTip(
            TABLE_TITLES.index(T_QUALITY),
            "Read this before the matrix: which analytes a component is, how many tanks\n"
            "report it, and how much of it carries no uncertainty or a default one.",
        )
        # The model object is created once and only ever reset, so its
        # selection model (and this connection) lives as long as the view.
        self._table_views[T_PAIRS].selectionModel().currentRowChanged.connect(self._on_pair_row_changed)
        splitter.addWidget(self.tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

        self._sync_enabled()
        self.plot.show_empty(EMPTY_TITLE, _IDLE_HINT)

    def _sync_enabled(self, *_args) -> None:
        on_kg = self.basis_combo.currentText() == "kg"
        self.distinct_check.setEnabled(not on_kg)
        self.distinct_check.setToolTip(_DISTINCT_KG_TOOLTIP if on_kg else _DISTINCT_TOOLTIP)
        at_tank = self.level_combo.currentData() != "layer"
        self.family_combo.setEnabled(not at_tank)
        self.family_combo.setToolTip(_FAMILY_TANK_TOOLTIP if at_tank else _FAMILY_TOOLTIP)
        self._sync_run_label()

    def _sync_run_label(self, *_args) -> None:
        """"Run" asks the question; "Update" re-asks it after a control moved under a standing result."""
        if not hasattr(self, "notes_banner") or self._running:
            return
        live = self._read_params()
        changed = self._result is not None and any(live[key] != self._params.get(key) for key in _RUN_KEYS)
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else "Correlate these components on the chosen basis and level.")

    def _say(self, message: str) -> None:
        say(self.app_window, message)

    def _on_preset_activated(self, index: int) -> None:
        members = self.preset_combo.itemData(index)
        if members:
            self.components_edit.setText(str(members))

    # ------------------------------------------------------------------
    # Dataset / scope protocol
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A result computed on the PREVIOUS dataset must not stay on screen
        # (nor export into the new dataset's folder), and a run still in
        # flight on it must not land: bumping the token orphans it.
        self._run_token += 1
        self._set_running(False)
        if self._result is not None:
            self._clear_result()
        set_list_completions(
            self.components_edit,
            dataset_values(dataset, "Element") + dataset_values(dataset, "Analyte"))

    def on_scope_changed(self) -> None:
        """Global scope bar changed: re-ask the STANDING question -- the
        settings that produced the result on screen, not whatever the
        controls have been moved to since -- under the new scope. A
        scope that matches nothing still yields a (empty) result, so a
        later scope change re-asks again; without a standing result
        there is nothing to re-ask."""
        if self._result is not None and self.dataset is not None and self.dataset.is_loaded():
            self._start(dict(self._params))

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def _read_params(self) -> dict:
        level = self.level_combo.currentData()
        family = self.family_combo.currentText()
        return {
            "components": parse_components(self.components_edit.text()),
            "basis": self.basis_combo.currentText(),
            "level": level,
            "family": family if (level == "layer" and family != ALL_FAMILIES) else None,
            "method": self.method_combo.currentData(),
            "distinct": self.distinct_check.isChecked(),
            "min_n": self.min_n_spin.value(),
            "quantile": self.quantile_spin.value(),
        }

    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Cross-correlation: no dataset is loaded yet.")
            return
        self._start(self._read_params())

    def _start(self, params: dict) -> None:
        """Submit one question to the worker. Everything it needs is read
        here, on the UI thread; the callbacks are bound methods of this
        widget so they are delivered back on the UI thread."""
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        scope = self.scope.as_scope()
        params = dict(params)
        params["scope"] = self.scope.summary()
        self._set_running(True)
        run_in_thread(
            lambda: compute_association(token, dataset, params, scope),
            self._on_run_done, self._on_run_error,
        )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.run_btn.setEnabled(not running)
        if running:
            self.run_btn.setText("Running…")
        else:
            self._sync_run_label()
        set_busy = getattr(self.app_window, "set_busy", None)
        if callable(set_busy) and (running or self._busy_shown):
            set_busy("Computing the cross-correlation…" if running else None)
            self._busy_shown = running

    def _on_run_error(self, tb_text: str) -> None:
        # compute_association catches everything, so this is reached only
        # if the worker itself fails.
        self._set_running(False)
        self.app_window.statusBar().showMessage("Cross-correlation failed.")
        QMessageBox.critical(self, f"{TAB_TITLE} failed", tb_text)

    def _on_run_done(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return  # superseded by a newer question (or a dataset change)
        self._set_running(False)
        if outcome.error:
            self.app_window.statusBar().showMessage("Cross-correlation failed.")
            QMessageBox.critical(self, f"{TAB_TITLE} failed", outcome.error)
            return
        if outcome.refusal:
            # The standing result (if any) stays, still labeled with the
            # settings and scope that produced it.
            self.app_window.statusBar().showMessage(f"Cross-correlation not run: {outcome.refusal}")
            QMessageBox.warning(self, TAB_TITLE, outcome.refusal)
            return

        self._matrix, self._result = outcome.matrix, outcome.result
        self._used = outcome.used
        self._params = dict(outcome.params)
        own_warnings = self._blank_component_warnings(outcome.matrix)
        counted = outcome.result.n_distinct if self._counts_distinct() else outcome.result.n_rows
        if len(outcome.used) != counted:
            # counted_rows() mirrors the engine's selection; if the two ever
            # part ways the scatter must not pass for the coefficient's data.
            own_warnings.append(
                f"Pair scatter shows {len(outcome.used)} observations where the correlation "
                f"counted {counted}: the points are not exactly the observations behind r."
            )
        warnings = list(outcome.matrix.warnings) + own_warnings
        self._tables = {
            T_PAIRS: outcome.result.pairs,
            T_MATRIX: outcome.result.correlation.rename_axis(COMPONENT_COLUMN).reset_index(),
            T_COUNTS: outcome.result.n_used.rename_axis(COMPONENT_COLUMN).reset_index(),
            T_QUALITY: outcome.quality,
            T_COOCCURRENCE: outcome.co_occurrence,
            T_VALUES: outcome.matrix.values,
            T_NOTES: self._notes_table(warnings, outcome.matrix.disclosures, outcome.result.notes),
        }
        self._filling = True
        try:
            for title_, frame in self._tables.items():
                self._table_views[title_].set_dataframe(frame)
        finally:
            self._filling = False

        # One calm line (what bears on reading this matrix, in full; the
        # dataset-wide notes as a count); every note behind "Show all",
        # in the tooltip and in the Notes tab.
        banner = banner_text(own_warnings, list(outcome.matrix.warnings))
        notes = own_warnings + list(outcome.matrix.warnings)
        self.notes_banner.set_notes(notes, headline=None if notes == [banner] else (banner or None))

        self._update_plot()
        self._sync_run_label()
        scope_note = f", scope: {self._params['scope']}" if self._params.get("scope") else ""
        self.app_window.statusBar().showMessage(
            f"Cross-correlation completed: {len(outcome.matrix.members)} components, "
            f"{self._params['basis']}, {self._n_statement()}{scope_note}."
        )

    def _clear_result(self) -> None:
        self._matrix = None
        self._result = None
        self._used = pd.DataFrame()
        self._tables = {}
        self._params = {}
        self._filling = True
        try:
            for view in self._table_views.values():
                view.set_dataframe(pd.DataFrame())
        finally:
            self._filling = False
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, _IDLE_HINT)
        self._sync_run_label()

    # ------------------------------------------------------------------
    # Words that describe the standing result
    # ------------------------------------------------------------------
    def _unit_word(self) -> str:
        return aq.LEVEL_LABELS[self._params.get("level", "layer")].lower()   # "waste layer" / "tank"

    def _pool_word(self, count: int) -> str:
        """'layers' / 'Saltcake layers' / 'tanks', singular when count is 1."""
        noun = "layer" if self._params.get("level", "layer") == "layer" else "tank"
        if count != 1:
            noun += "s"
        family = self._params.get("family")
        return f"{family} {noun}" if family else noun

    def _counts_distinct(self) -> bool:
        return bool(self._params.get("distinct")) and self._params.get("basis") != "kg"

    def _n_statement(self) -> str:
        if self._result is None:
            return ""
        n_rows, n_distinct = self._result.n_rows, self._result.n_distinct
        if self._counts_distinct():
            word = "composition" if n_distinct == 1 else "compositions"
            return f"n = {n_distinct} distinct {word} of {n_rows} {self._pool_word(n_rows)}"
        return f"n = {n_rows} {self._pool_word(n_rows)}, every one counted"

    def _subtitle_parts(self) -> List[str]:
        params = self._params
        parts = [f"Level: {self._unit_word()}"]
        if params.get("level") == "layer":
            parts.append(f"Phase family: {params.get('family') or ALL_FAMILIES.lower()}")
        parts.append(self._n_statement())
        if params.get("scope"):
            parts.append(f"Scope: {params['scope']}")
        return parts

    def _blank_component_warnings(self, matrix: aq.AssociationMatrix) -> List[str]:
        """A component with no value anywhere produces an empty row of the
        matrix. Say so in words: a misspelt label and a component that is
        simply not reported on this basis look identical otherwise."""
        if matrix.values.empty:
            return [f"No {aq.LEVEL_LABELS[matrix.level].lower()} matches these settings — nothing to correlate."]
        blank = [m for m in matrix.members if not matrix.values[m].notna().any()]
        if not blank:
            return []
        return [
            f"{', '.join(blank)}: no value in any {aq.LEVEL_LABELS[matrix.level].lower()} on the "
            f"{matrix.basis} basis — check the label, or it is not reported in this scope."
        ]

    def _notes_table(self, warnings: List[str], disclosures: List[str], notes: List[str]) -> pd.DataFrame:
        params = self._params
        scope_text = (f"Scope: {params['scope']}" if params.get("scope")
                      else "Scope: none — every tank and layer of the dataset.")
        extra = [
            "Co-occurrence and Values cover every "
            f"{self._unit_word()} in scope; the phase family and the distinct-composition "
            "count apply to the correlation only.",
        ]
        return pd.DataFrame(
            [{"kind": "scope", "text": scope_text}]
            + [{"kind": "warning", "text": w} for w in warnings]
            + [{"kind": "disclosure", "text": d} for d in disclosures]
            + [{"kind": "note", "text": n} for n in list(notes) + extra]
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _on_plot_kind_changed(self, _text: str) -> None:
        self._update_plot()

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a cell of the matrix selects that pair in the Pairs
        table (the Pair scatter then draws it); a click on a point of the
        scatter selects its layer or tank in Values."""
        if self._result is None:
            return
        title = T_VALUES if self.plot_combo.currentText() == PLOT_SCATTER else T_PAIRS
        select_row_for_mark(self.tables, [self._table_views[title]], mark)

    def _on_pair_row_changed(self, *_args) -> None:
        if self._filling:
            return
        if self.plot_combo.currentText() == PLOT_SCATTER:
            self._update_plot()

    def selected_pair(self) -> Optional[Tuple[str, str]]:
        """The pair on the current row of the Pairs table -- read from the
        table as displayed, which the user may have re-sorted -- or the
        strongest pair when no row is current."""
        if self._result is None or self._result.pairs.empty:
            return None
        view = self._table_views[T_PAIRS]
        shown = view.dataframe()
        index = view.currentIndex()
        if index.isValid() and 0 <= index.row() < len(shown):
            row = shown.iloc[index.row()]
        else:
            row = self._result.pairs.iloc[0]
        return str(row["A"]), str(row["B"])

    def _update_plot(self) -> None:
        if self._result is None or self._matrix is None:
            return
        if self._matrix.values.empty:
            # Never leave a previous figure standing beside empty tables.
            self.plot.show_message("No data for these settings.")
            return
        if self.plot_combo.currentText() == PLOT_SCATTER:
            self._draw_scatter()
        else:
            self._draw_matrix()

    def _draw_matrix(self) -> None:
        params = self._params
        ph.plot_association_heatmap(
            self.plot, self._tables[T_MATRIX],
            title=f"Cross-correlation — {METHOD_LABELS[params['method']]}, {aq.BASIS_LABELS[params['basis']]}",
            subtitle=ph.SUBTITLE_SEP.join(self._subtitle_parts()),
            cbar_label=COEFFICIENT_LABELS[params["method"]],
            label_col=COMPONENT_COLUMN,
        )

    def _draw_scatter(self) -> None:
        pair = self.selected_pair()
        if pair is None:
            self.plot.show_message("No pair to plot.")
            return
        a, b = pair
        params = self._params
        pairs = self._result.pairs
        match = pairs[((pairs["A"] == a) & (pairs["B"] == b)) | ((pairs["A"] == b) & (pairs["B"] == a))]
        title = f"{a} vs {b}"
        if not match.empty:
            row = match.iloc[0]
            if pd.notna(row["r"]):
                title += f" — {COEFFICIENT_LABELS[params['method']]} = {row['r']:.2f}, N = {int(row['N'])}"
            else:
                title += f" — left blank (N = {int(row['N'])})"
        ph.plot_association_scatter(
            self.plot, self._used, a, b,
            value_label=aq.BASIS_LABELS[params["basis"]], title=title,
            subtitle=ph.SUBTITLE_SEP.join(self._subtitle_parts()),
            color_col="PhaseFamily" if params["level"] == "layer" else None,
            unit_word="distinct composition" if self._counts_distinct() else self._pool_word(1),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_tables(self) -> None:
        if self._result is None or self.dataset is None:
            self._say("Cross-correlation: run first; there is nothing to export yet.")
            return
        params = self._params
        # Everything here describes the RUN, not the current widget state.
        # The plot kind is the exception: it is a display choice wired to a
        # redraw, so the live value is what the saved PNG shows.
        parameters = pd.DataFrame([
            {"parameter": "components", "value": ", ".join(self._matrix.members)},
            {"parameter": "basis", "value": params["basis"]},
            {"parameter": "level", "value": params["level"]},
            {"parameter": "phase_family", "value": params["family"] or ALL_FAMILIES},
            {"parameter": "method", "value": params["method"]},
            {"parameter": "count_each_distinct_composition_once", "value": self._counts_distinct()},
            {"parameter": "min_shared_observations", "value": params["min_n"]},
            {"parameter": "co_occurrence_quantile", "value": params["quantile"]},
            {"parameter": "rows_offered", "value": self._result.n_rows},
            {"parameter": "distinct_compositions", "value": self._result.n_distinct},
            {"parameter": "plot", "value": self.plot_combo.currentText()},
        ])
        tables = {_EXPORT_NAMES[title_]: frame for title_, frame in self._tables.items()}
        tables["association_parameters"] = parameters
        out_dir = export_utils.export_named_tables(
            self.dataset, "cross_correlation", tables,
            context={
                "basis": params["basis"], "level": params["level"],
                "phase_family": params["family"] or ALL_FAMILIES,
                "method": params["method"],
                "observations": self._n_statement(),
                "global_scope": params.get("scope") or "(none)",
            },
        )
        try:
            self.plot.figure.savefig(out_dir / "cross_correlation_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Cross-correlation exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Cross-correlation outputs exported:\n{out_dir}")

