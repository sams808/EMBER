"""
qt_explorer.py — the Element Explorer workspace: search by element symbol,
analyte label, text fragment or regular expression; see which tanks hold
the subject, what is reported alongside it, and the typical composition of
those tanks.

What the page owns:
- The search controls. The match mode is shown in words and kept as the
  identifier explorer_query and element_science expect (MODES), so the
  science calls are the ones they always were.
- Two kinds of answer, side by side and labelled as such. The Target
  tables come from the query engine (explorer_query): an element receives
  its stoichiometric share of every species that reports it, under the
  global scope. The co-occurrence and composition tables come from
  element_science, which counts reported rows as they are, over the whole
  dataset; when a scope is active the page says that these are not scoped.
- The run goes through the house worker (about a second on the shipped
  dataset): everything is read from the widgets before it is submitted,
  the worker function touches no widget, and a result that arrives after
  a newer question or a dataset change is discarded.
- Ten tables, chosen by name from a list beside the chart (they do not fit
  a row of tabs). Choosing a plot brings the table it is drawn from
  forward; a click on a chart mark selects its row there; a double click
  on anything that stands for a tank opens it.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

import element_science as esci
import explorer_query as eqy
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
    set_completions,
)
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

PAGE_TITLE = "Element explorer"
PAGE_SUBTITLE = (
    "Where one element or analyte is found: the tanks that hold it, what is reported alongside it "
    "and the typical composition of those tanks."
)
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "Enter an element symbol or an analyte label, then press Run."

# Match modes: the identifiers the science modules take, and the words the
# combo shows for them.
MODES = ["auto", "element", "analyte_exact", "analyte_contains", "regex"]
MODE_LABELS: Dict[str, str] = {
    "auto": "Automatic",
    "element": "Element",
    "analyte_exact": "Analyte, exact label",
    "analyte_contains": "Analyte containing the text",
    "regex": "Regular expression",
}
MODE_TOOLTIPS: Dict[str, str] = {
    "auto": "An element when the text is an element symbol, otherwise an exact analyte label.",
    "element": "One element, with its share of every species that reports it.",
    "analyte_exact": "One analyte, by the exact label the dataset uses (for example 137Cs or SO4).",
    "analyte_contains": "Every analyte whose label contains the text, asked as one group.",
    "regex": "Every analyte whose label matches the regular expression, asked as one group.",
}
# What the search field offers while typing, per match mode.
_COMPLETION_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "auto": ("Element", "Analyte"),
    "element": ("Element",),
    "analyte_exact": ("Analyte",),
    "analyte_contains": ("Analyte",),
    "regex": (),
}

# The Basis picker offers "All" plus whatever the data reports. For the
# engine-backed Target views "All" maps onto the bridged mass basis rather
# than ranking kilograms against curies, which compares nothing.
_ENGINE_BASIS = {"All": "g"}

PLOT_TYPES = [
    "Target inventory by tank",
    "Target fraction by tank",
    "Target by waste phase",
    "Target by waste type",
    "Target vs total inventory",
    "Co-elements inventory",
    "Co-elements presence (%)",
    "Mean absolute composition",
    "Mean fraction composition",
]
# The table each chart is drawn from, and the column that names its marks.
PLOT_SOURCES: Dict[str, Tuple[str, str]] = {
    "Target inventory by tank": ("Target by tank", "WasteSiteId"),
    "Target fraction by tank": ("Target by tank", "WasteSiteId"),
    "Target by waste phase": ("Target by phase", "WastePhase"),
    "Target by waste type": ("Target by type", "WasteType"),
    "Target vs total inventory": ("Target by tank", "WasteSiteId"),
    "Co-elements inventory": ("Co-elements", "Element"),
    "Co-elements presence (%)": ("Co-elements", "Element"),
    "Mean absolute composition": ("Composition (absolute)", "Element"),
    "Mean fraction composition": ("Composition (fraction)", "Element"),
}
_ENGINE_TABLES = ("Target by tank", "Target by phase", "Target by type")

# (table name, rows displayed at most, what the table holds)
TABLES: List[Tuple[str, Optional[int], str]] = [
    ("Target by tank", None, "The subject in each tank, with its share of the tank's attributed total."),
    ("Target by phase", None, "The subject summed by waste phase."),
    ("Target by type", None, "The subject summed by waste type."),
    ("Co-elements", None,
     "Every element reported in the tanks that hold the subject: how often and how much."),
    ("Co-analytes", None,
     "Every analyte reported in the tanks that hold the subject: how often and how much."),
    ("Composition (absolute)", None,
     "Each element's inventory across the tanks that hold the subject: mean, spread and maximum."),
    ("Composition (fraction)", None,
     "Each element's fraction of the tank total across the tanks that hold the subject."),
    ("Composition (by tank)", None,
     "Inventory and fraction of every element, one row per tank and element."),
    ("Raw target rows", 500, "The dataset records behind the answer, as reported (the first 500)."),
    ("Notes", 200, "What qualifies the result, and how the subject was read."),
]

ROW_COUNT_DISCLOSURE = (
    "Co-elements, Co-analytes and the Composition tabs count reported rows "
    "as they are (each analyte's full mass under its first parsed element); "
    "rows whose analyte parses to no element at all are excluded from those "
    "tabs and their tank totals. The Target tabs above are apportioned."
)
SCOPE_CAVEAT = (
    "The scope applies to the Target tables and the raw rows. Co-elements, Co-analytes and the "
    "Composition tables are computed on the whole dataset."
)
NOT_SCOPED_SUBTITLE = "Whole dataset: the scope does not apply to this view"

# Wording of the engine's notes as the banner shows them: the sentences
# are the engine's, minus the names of internal tables and columns. The
# Notes tab and the export keep the engine's exact text.
_PLAIN_WORDING = (
    (" — add their formulas to reference_data.SPECIES_FORMULAS to include them",
     "; their formulas are not in the species formula table"),
    ("reference_data.SPECIES_FORMULAS", "the species formula table"),
    ("reference_data.NUCLIDE_HALF_LIFE_YEARS", "the nuclide half-life table"),
    ("(Inventory_RSD = 0 means unassigned, not exact)",
     "(a relative standard deviation of 0 means unassigned, not exact)"),
    (" (harmonizer: Stage 4)", ""),
    ("; use basis 'g' to bridge them", "; the g basis of the Query workspace converts them to mass"),
)
_LABEL_WITH_UNIT_RE = re.compile(r"^(.*) \[(.+)\]$")


def plain_note(note: str) -> str:
    """One engine note as a banner sentence: no internal identifiers, a capital first letter."""
    text = str(note)
    for old, new in _PLAIN_WORDING:
        text = text.replace(old, new)
    return text[:1].upper() + text[1:]


def _engine_basis(unit: str) -> str:
    from query_engine import BASES
    basis = _ENGINE_BASIS.get(unit, unit)
    return basis if basis in BASES else "g"


def _unit_suffix(df: Optional[pd.DataFrame]) -> str:
    """" (kg)" from the Units column of a results table when it holds one
    unit, a statement that each bar carries its own when it holds several,
    and "" for a table with no Units column (a dimensionless quantity)."""
    if df is None or df.empty or "Units" not in df.columns:
        return ""
    units = sorted({str(u) for u in df["Units"].dropna().unique()})
    if len(units) == 1:
        return f" ({units[0]})"
    return " (unit as labelled on each bar)" if units else ""


@ph.composable
def plot_ranked_bars(
    panel, data: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str,
    subtitle: str = "", top_n: int = 40, highlighted_label: Optional[str] = None, log_x: bool = False,
) -> None:
    """plot_helpers.plot_barh, with the qualifying line (the scope a result
    was computed under) as the small grey axes title under the figure
    title, the convention of the plot grammar."""
    ph.plot_barh(panel, data, label_col, value_col, title, xlabel, top_n=top_n,
                 highlighted_label=highlighted_label, log_x=log_x)
    marks = getattr(panel.ax, ph.MARKS_ATTR, None)
    if not subtitle or not isinstance(marks, dict) or not marks.get("categories"):
        return                                    # nothing was drawn: the panel shows a sentence
    width = max(30, int(12.0 * float(panel.figure.get_size_inches()[0])))
    panel.figure.suptitle(title)
    panel.ax.set_title(ph.wrap_subtitle(subtitle.split(ph.SUBTITLE_SEP), width=width), fontsize=9,
                       color=ph.CHART_SECONDARY, fontweight="normal")
    try:
        panel.figure.tight_layout()
    except Exception:       # a layout that cannot be solved must not take the drawing with it
        pass


@dataclass
class _Outcome:
    """What the worker hands back, tagged with the run it belongs to."""

    token: int
    params: dict = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    value_column: str = ""
    refusal: str = ""
    error: str = ""


def compute_search(token: int, dataset: HanfordDataset, params: dict,
                   scope: Dict[str, List[str]]) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    resolved = params["resolved"]
    query, mode, unit = params["query"], params["mode"], params["unit"]
    min_target, min_context = params["min_target"], params["min_context"]
    try:
        # The QUANTITATIVE views come from the query engine, so this page
        # agrees with Query, Heatmaps and Overview: an element receives its
        # stoichiometric share of every species reporting it instead of the
        # full mass of every analyte whose formula mentions it (searching
        # "S" used to book all of every sulfate row, and searching "O"
        # booked the same sulfate again).
        engine = eqy.target_tables(dataset, resolved, basis=params["basis"], scope=scope)
        co_el = esci.co_elements(dataset, query, mode, unit, min_target, min_context)
        co_an = esci.co_analytes(dataset, query, mode, unit, min_target, min_context)
        abs_stats, frac_stats, tank_stats = esci.composition_stats(
            dataset, query, mode, unit, min_target, min_context)
        raw = eqy.raw_target_rows(dataset, resolved, scope=scope)
    except ValueError as exc:
        return _Outcome(token=token, params=params, refusal=str(exc))
    except Exception:
        return _Outcome(token=token, params=params, error=traceback.format_exc())

    disclosures = list(engine.disclosures) + [ROW_COUNT_DISCLOSURE]
    if params.get("scope"):
        disclosures.append(SCOPE_CAVEAT)
    notes = pd.DataFrame(
        [{"kind": "warning", "text": w} for w in engine.warnings]
        + [{"kind": "disclosure", "text": d} for d in disclosures]
    )
    tables = {name: engine.tables.get(name, pd.DataFrame()) for name in _ENGINE_TABLES}
    tables.update({
        "Co-elements": co_el, "Co-analytes": co_an,
        "Composition (absolute)": abs_stats, "Composition (fraction)": frac_stats,
        "Composition (by tank)": tank_stats,
        "Raw target rows": raw, "Notes": notes,
    })
    return _Outcome(token=token, params=params, tables=tables, warnings=list(engine.warnings),
                    value_column=engine.value_column)


class ExplorerPage(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.scope = scope if scope is not None else GlobalScope(self)
        self._resolved = None
        self.dataset: Optional[HanfordDataset] = None
        self._last_query: Optional[str] = None
        self._results: dict = {}
        # Every setting that produced the standing result. Labels, the
        # subtitle and the export read THIS, never the live widgets.
        self._params: dict = {}
        self._result_inputs: Optional[tuple] = None
        self._engine_value_column = ""
        self._symbol: Optional[str] = None
        self._run_token = 0
        self._running = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader(PAGE_TITLE, PAGE_SUBTITLE)
        self.ask_query_btn = secondary_button(
            "Ask in Query",
            "Open the same subject in the Query workspace: any basis (g, kg, Ci, wt%, g/L),\n"
            "any grouping, and the per-source disclosures.")
        self.ask_query_btn.clicked.connect(self._ask_in_query)
        self.header.add_action_widget(self.ask_query_btn)
        self.export_btn = secondary_button(
            "Export...", "Write every table of the result, the notes and the chart to a folder.")
        self.export_btn.clicked.connect(self._export_bundle)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 6, "Element Explorer"))
        root.addWidget(self.header)

        controls, row = control_row()
        row.addWidget(field_label("Search"))
        self.query_edit = QLineEdit("Se")
        self.query_edit.setMinimumWidth(110)
        self.query_edit.setMaximumWidth(180)
        self.query_edit.setPlaceholderText("Element or analyte")
        self.query_edit.setToolTip(
            "What to look for: an element symbol (Se), an analyte label (137Cs, SO4),\n"
            "a fragment of a label or a regular expression, as Match says.")
        self.query_edit.textChanged.connect(self._sync_run_label)
        self.query_edit.returnPressed.connect(self.run_search)
        row.addWidget(self.query_edit)

        row.addSpacing(8)
        row.addWidget(field_label("Match"))
        self.mode_combo = QComboBox()
        for index, key in enumerate(MODES):
            self.mode_combo.addItem(MODE_LABELS[key], key)
            self.mode_combo.setItemData(index, MODE_TOOLTIPS[key], Qt.ToolTipRole)
        self.mode_combo.setToolTip(
            "How the text is read.\n" + "\n".join(f"{MODE_LABELS[k]}: {MODE_TOOLTIPS[k]}" for k in MODES))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(self.mode_combo)

        row.addSpacing(8)
        row.addWidget(field_label("Basis"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("All")
        self.unit_combo.setToolTip(
            "All: the Target tables in grams, activity converted to mass through ICRP-107\n"
            "specific activities; the co-occurrence tables keep kg and Ci apart.\n"
            "kg or Ci: rows reported in that unit only.")
        self.unit_combo.currentTextChanged.connect(self._sync_run_label)
        row.addWidget(self.unit_combo)

        row.addSpacing(8)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run_search)
        row.addWidget(self.run_btn)
        row.addStretch(1)

        row.addWidget(field_label("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.setToolTip(
            "Which table of the result is drawn. Target charts are apportioned by the query engine;\n"
            "co-element and composition charts count reported rows as they are.")
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        row.addWidget(self.plot_type_combo)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv_controls, adv_row = control_row()
        adv_row.addWidget(field_label("Minimum target inventory"))
        self.min_target_spin = QDoubleSpinBox()
        self.min_target_spin.setRange(0.0, 1e12)
        self.min_target_spin.setDecimals(6)
        self.min_target_spin.setToolTip(
            "Co-occurrence and composition tables: ignore reported rows of the subject at or below\n"
            "this inventory, in the unit of the row. The Target tables are not filtered.")
        self.min_target_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.min_target_spin)
        adv_row.addSpacing(8)
        adv_row.addWidget(field_label("Minimum context inventory"))
        self.min_context_spin = QDoubleSpinBox()
        self.min_context_spin.setRange(0.0, 1e12)
        self.min_context_spin.setDecimals(6)
        self.min_context_spin.setToolTip(
            "Co-occurrence and composition tables: ignore rows of the other elements and analytes\n"
            "at or below this inventory, in the unit of the row.")
        self.min_context_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.min_context_spin)
        adv_row.addSpacing(8)
        adv_row.addWidget(field_label("Bars shown"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 300)
        self.top_n_spin.setValue(40)
        self.top_n_spin.setToolTip("How many bars a ranking draws, largest first. The tables keep every row.")
        self.top_n_spin.valueChanged.connect(lambda _value: self._update_plot())
        adv_row.addWidget(self.top_n_spin)
        adv_row.addStretch(1)
        self.advanced_section.content_layout().addWidget(adv_controls)
        root.addWidget(self.advanced_section)

        # Engine warnings are data-integrity statements (mixed decay
        # cohorts, exclusions, unassigned uncertainty): always on the page
        # with the result, as one line, the full list one click away.
        self.notes_banner = NotesBanner()
        self.warnings_label = self.notes_banner         # the name every page uses for this slot
        root.addWidget(self.notes_banner)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        self.plot = PlotWidget()
        self.plot.markClicked.connect(self._on_mark_clicked)
        splitter.addWidget(self.plot)

        # Ten tables do not fit a row of tabs beside the chart: they are
        # chosen by name from a list, and the chart brings its own forward.
        self.tables_pane = QWidget()
        pane = QVBoxLayout(self.tables_pane)
        pane.setContentsMargins(0, 0, 0, 0)
        pane.setSpacing(8)
        table_controls, table_row = control_row()
        table_row.addWidget(field_label("Table"))
        self.table_combo = QComboBox()
        self.table_combo.setToolTip("Which table of the result is shown. Choosing a plot brings its table forward.")
        table_row.addWidget(self.table_combo, 1)
        pane.addWidget(table_controls)
        self.tables = QTabWidget()
        self.tables.tabBar().hide()
        self.tables.setDocumentMode(True)
        self._table_views: dict[str, DataFrameTableView] = {}
        for title, max_rows, holds in TABLES:
            view = DataFrameTableView(title=title, max_rows_display=max_rows)
            view.set_empty_text(EMPTY_TITLE, holds)
            self._table_views[title] = view
            index = self.tables.addTab(view, title)
            self.table_combo.addItem(title)
            self.table_combo.setItemData(index, holds, Qt.ToolTipRole)
        self.table_combo.currentIndexChanged.connect(self.tables.setCurrentIndex)
        self.tables.currentChanged.connect(self.table_combo.setCurrentIndex)
        pane.addWidget(self.tables, 1)
        splitter.addWidget(self.tables_pane)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

        self._refresh_completions()
        self._sync_run_label()
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def mode_key(self) -> str:
        """The match mode as the science modules name it."""
        return str(self.mode_combo.currentData() or MODES[0])

    def set_mode(self, key: str) -> None:
        index = self.mode_combo.findData(key)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

    def _on_mode_changed(self, *_args) -> None:
        self._refresh_completions()
        self._sync_run_label()

    def _refresh_completions(self) -> None:
        """Offer what the loaded dataset holds for the current match mode."""
        items: List[str] = []
        for column in _COMPLETION_COLUMNS.get(self.mode_key(), ()):
            items.extend(dataset_values(self.dataset, column))
        set_completions(self.query_edit, items)

    def _read_inputs(self) -> tuple:
        return (
            self.query_edit.text().strip(), self.mode_key(), self.unit_combo.currentText() or "All",
            self.min_target_spin.value(), self.min_context_spin.value(),
        )

    def _sync_run_label(self, *_args) -> None:
        """"Run" asks a question; "Update" re-asks after a control changed under a standing result."""
        if not hasattr(self, "run_btn") or self._running:
            return
        changed = bool(self._results) and self._result_inputs != self._read_inputs()
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else "Search the inventory for the subject the controls describe.")

    def _say(self, message: str) -> None:
        self.app_window.statusBar().showMessage(message)

    # ------------------------------------------------------------------
    # Cross-links
    # ------------------------------------------------------------------
    def show_subject(self, subject: str, kind: str = "element") -> bool:
        """Entry point of the shell (global search, links from other
        pages): fill the field for one element or analyte, and run."""
        self.set_mode("element" if str(kind).lower() == "element" else "analyte_exact")
        self.query_edit.setText(str(subject))
        self.run_search()
        return True

    def _ask_in_query(self) -> None:
        """Cross-link to the Query workspace: same subject, engine
        semantics (apportioned masses, Ci to g bridge, disclosures)."""
        query = self.query_edit.text().strip()
        if not query:
            self._say("Element explorer: enter an element or an analyte first.")
            self.query_edit.setFocus()
            return
        symbol = None
        if self.mode_key() in ("auto", "element"):
            from elements import normalize_element_symbol
            symbol = normalize_element_symbol(query)
        kind, subject = ("element", symbol) if symbol else ("analyte", query)
        ask = getattr(self.app_window, "ask_in_query", None)
        if ask is None:  # standalone construction in tests/tooling
            return
        if not ask(kind, subject):
            self._say("Element explorer: the Query workspace is not available.")

    # ------------------------------------------------------------------
    # Dataset / scope protocol
    # ------------------------------------------------------------------
    def on_scope_changed(self) -> None:
        if self._results:
            self.run_search()

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A result of the previous dataset must not stay on screen, and a
        # run still in flight on it must not land.
        self._run_token += 1
        self._set_running(False)
        current_unit = self.unit_combo.currentText()
        self.unit_combo.blockSignals(True)
        self.unit_combo.clear()
        self.unit_combo.addItem("All")
        units = dataset.available_units() if dataset.is_loaded() else []
        self.unit_combo.addItems(units)
        restore_index = self.unit_combo.findText(current_unit)
        self.unit_combo.setCurrentIndex(restore_index if restore_index >= 0 else 0)
        self.unit_combo.blockSignals(False)
        self._results = {}
        self._resolved = None
        self._params = {}
        self._result_inputs = None
        self._engine_value_column = ""
        self._symbol = None
        self.notes_banner.set_notes([])
        for view in self._table_views.values():
            view.set_dataframe(pd.DataFrame())
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self._refresh_completions()
        self._sync_run_label()

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run_search(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Element explorer: no dataset is loaded yet.")
            return
        query = self.query_edit.text().strip()
        if not query:
            self._say("Element explorer: enter an element or an analyte first.")
            self.query_edit.setFocus()
            return
        mode = self.mode_key()
        resolved = eqy.resolve_subject(self.dataset, query, mode)
        if not resolved.ok:
            self._refuse(resolved.error)
            return
        unit = self.unit_combo.currentText() or "All"
        params = {
            "query": query, "mode": mode, "unit": unit, "basis": _engine_basis(unit),
            "min_target": self.min_target_spin.value(), "min_context": self.min_context_spin.value(),
            "resolved": resolved, "scope": self.scope.summary(), "inputs": self._read_inputs(),
        }
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        scope = self.scope.as_scope()
        self._set_running(True)
        run_in_thread(
            lambda: compute_search(token, dataset, params, scope),
            self._on_run_done, self._on_run_error,
        )

    def _refuse(self, message: str) -> None:
        """The question cannot be answered as asked: say why, and leave a
        standing answer under its own labels."""
        self._say(f"Element explorer: {message}")
        if self._results:
            QMessageBox.information(self, PAGE_TITLE, message)
        else:
            self.plot.show_empty("Nothing to show", message)

    def _set_running(self, running: bool) -> None:
        was_running = self._running
        self._running = running
        self.run_btn.setEnabled(not running)
        if running:
            self.run_btn.setText("Running…")
        else:
            self._sync_run_label()
        if running or was_running:
            set_busy = getattr(self.app_window, "set_busy", None)
            if callable(set_busy):
                set_busy("Searching the inventory…" if running else None)

    def _on_run_error(self, tb_text: str) -> None:
        self._set_running(False)
        self._say("Element explorer: the search failed.")
        QMessageBox.critical(self, "Search failed", tb_text)

    def _on_run_done(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return          # superseded by a newer question, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._say("Element explorer: the search failed.")
            QMessageBox.critical(self, "Search failed", outcome.error)
            return
        if outcome.refusal:
            self._say(f"Element explorer: {outcome.refusal}")
            QMessageBox.warning(self, "Search failed", outcome.refusal)
            return

        params = outcome.params
        resolved = params["resolved"]
        tables = outcome.tables
        self._resolved = resolved
        self._params = dict(params)
        self._result_inputs = params["inputs"]
        self._results = {name: frame for name, frame in tables.items()
                         if name not in ("Raw target rows", "Notes")}
        for name, view in self._table_views.items():
            view.set_dataframe(tables.get(name, pd.DataFrame()))
        banner = [plain_note(w) for w in outcome.warnings]
        if params.get("scope"):
            banner.append(SCOPE_CAVEAT)
        self.notes_banner.set_result_notes(banner)

        self._engine_value_column = outcome.value_column
        self._last_query = params["query"]
        self._symbol = resolved.label
        self._update_plot()
        self._sync_run_label()

        target_by_tank = tables["Target by tank"]
        n_tanks = target_by_tank["WasteSiteId"].nunique() if not target_by_tank.empty else 0
        scope_note = f", scope: {params['scope']}" if params.get("scope") else ""
        self._say(
            f"Element explorer: {resolved.label} read as {resolved.kind}, basis {params['basis']}, "
            f"{n_tanks:,} tank(s){scope_note}."
        )

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------
    def _on_plot_type_changed(self, plot_type: str) -> None:
        self._update_plot()
        # The chart and the table it is drawn from, side by side.
        source = PLOT_SOURCES.get(plot_type)
        if self._results and source is not None:
            self.tables.setCurrentWidget(self._table_views[source[0]])

    def _update_plot(self) -> None:
        if not self._results:
            return
        plot_type = self.plot_type_combo.currentText()
        symbol = self._symbol or "Target"
        target_by_tank = self._results.get("Target by tank")
        target_by_phase = self._results.get("Target by phase")
        target_by_type = self._results.get("Target by type")
        co_el = self._results.get("Co-elements")
        abs_stats = self._results.get("Composition (absolute)")
        frac_stats = self._results.get("Composition (fraction)")
        top_n = self.top_n_spin.value()

        # Labels come from the settings that PRODUCED this result. The unit
        # sits in the axis label: that is where the tooltips read it from.
        basis = self._params.get("basis", "")
        scope_text = self._params.get("scope", "")
        scoped = f"Scope: {scope_text}" if scope_text else ""
        not_scoped = NOT_SCOPED_SUBTITLE if scope_text else ""
        value_col = self._engine_value_column or "Total_g"
        target_label = f"{symbol} inventory ({basis})" if basis else f"{symbol} inventory"

        if plot_type == "Target inventory by tank":
            plot_ranked_bars(self.plot, target_by_tank, "WasteSiteId", value_col,
                             f"{symbol} inventory by tank", target_label, subtitle=scoped,
                             top_n=top_n, highlighted_label=symbol, log_x=True)
        elif plot_type in ("Target by waste phase", "Target by waste type"):
            # One shared body (a mapping, not twin branches) so the next
            # grouped dimension is a table entry, not a pasted block.
            df, label_col, title = {
                "Target by waste phase": (target_by_phase, "WastePhase", f"{symbol} inventory by waste phase"),
                "Target by waste type": (target_by_type, "WasteType", f"{symbol} inventory by waste type"),
            }[plot_type]
            plot_ranked_bars(self.plot, df, label_col, value_col, title, target_label,
                             subtitle=scoped, top_n=top_n, log_x=True)
        elif plot_type == "Target fraction by tank":
            plot_ranked_bars(self.plot, target_by_tank, "WasteSiteId", "Fraction_of_tank",
                             f"{symbol} share of each tank's attributed total",
                             f"{symbol} as a fraction of the tank's attributed total",
                             subtitle=scoped, top_n=top_n, highlighted_label=symbol)
        elif plot_type == "Target vs total inventory":
            from plot_grammar import PlotSpec, render_plot
            unit = f" ({basis})" if basis else ""
            render_plot(self.plot, target_by_tank, PlotSpec(
                mark="scatter", x="Context_total", y=value_col,
                title=f"{symbol} against the tank's attributed total", subtitle=scoped,
                xlabel=f"Tank total, all elements{unit}", ylabel=f"{symbol} inventory{unit}",
                log_x=True, log_y=True))
        elif plot_type == "Co-elements inventory":
            plot_ranked_bars(self.plot, co_el, "Element", "Total_inventory_in_target_tanks",
                             f"Elements in the tanks that hold {symbol}",
                             f"Total inventory in the tanks that hold {symbol}" + _unit_suffix(co_el),
                             subtitle=not_scoped, top_n=top_n, highlighted_label=symbol, log_x=True)
        elif plot_type == "Co-elements presence (%)":
            plot_ranked_bars(self.plot, co_el, "Element", "PresenceFraction_pct",
                             f"How often each element is reported with {symbol}",
                             f"Tanks that hold {symbol} and report the element (%)",
                             subtitle=not_scoped, top_n=top_n, highlighted_label=symbol)
        elif plot_type == "Mean absolute composition":
            plot_ranked_bars(self.plot, abs_stats, "Element", "Mean_inventory_all_target_tanks_zero_filled",
                             f"Mean composition of the tanks that hold {symbol}",
                             "Mean inventory per tank, absent counted as zero" + _unit_suffix(abs_stats),
                             subtitle=not_scoped, top_n=top_n, highlighted_label=symbol, log_x=True)
        elif plot_type == "Mean fraction composition":
            plot_ranked_bars(self.plot, frac_stats, "Element", "Mean_fraction_all_target_tanks_zero_filled",
                             f"Mean fractional composition of the tanks that hold {symbol}",
                             "Mean fraction of the tank total, absent counted as zero",
                             subtitle=not_scoped, top_n=top_n, highlighted_label=symbol)

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a bar or a point selects the row it was drawn from."""
        source = PLOT_SOURCES.get(self.plot_type_combo.currentText())
        if source is None or not self._results:
            return
        table_name, column = source
        view = self._table_views[table_name]
        frame = view.dataframe()
        label = str(mark.get("label") or "")
        if not label or frame is None or frame.empty or column not in frame.columns:
            return
        values = frame[column].astype(str)
        hits = values == label
        if not hits.any():
            # A bar of a mixed-unit ranking is named "Sr [Ci]".
            match = _LABEL_WITH_UNIT_RE.match(label)
            if match and "Units" in frame.columns:
                hits = (values == match.group(1)) & (frame["Units"].astype(str) == match.group(2))
        if not hits.any():
            return
        row = int(hits.to_numpy().argmax())
        self.tables.setCurrentWidget(view)
        view.selectRow(row)
        index = view.proxy_model().mapFromSource(view.model().index(row, 0))
        if index.isValid():
            view.table.scrollTo(index)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_bundle(self) -> None:
        if not self._results or self.dataset is None:
            self._say("Element explorer: run a search first; there is nothing to export yet.")
            return
        tables = dict(self._results)
        tables["Raw target rows"] = self._table_views["Raw target rows"].dataframe()
        tables["Notes"] = self._table_views["Notes"].dataframe()
        context = {
            "search": self._params.get("query", ""),
            "match": MODE_LABELS.get(self._params.get("mode", ""), ""),
            "read_as": f"{self._resolved.label} ({self._resolved.kind})" if self._resolved else "",
            "basis": self._params.get("unit", ""),
            "target_tables_basis": self._params.get("basis", ""),
            "global_scope": self._params.get("scope") or "(none)",
            "scope_applies_to": "Target tables and raw rows; co-occurrence and composition tables "
                                "are computed on the whole dataset",
        }
        out_dir = export_utils.export_named_tables(
            self.dataset, f"search_{self._last_query}", tables, context=context)
        try:
            self.plot.figure.savefig(out_dir / "plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Element explorer exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Search results exported to:\n{out_dir}")
