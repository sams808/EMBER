"""
qt_vitrification_screening.py — the Vitrification "Screening", "Candidate
Search", and "Blend Partners" sub-tabs: three transparent composition
scores over vitrification_science, each with its chart on the left and its
table on the right.

What the three tabs share (one base class, _ScoreTab):
- The score weights are editable, labelled spin boxes in the Advanced
  options (with "Reset to defaults"); the documented defaults are
  vitrification_science's *_WEIGHT_DEFAULTS. Edited weights persist across
  restarts through QSettings (settings_prefix keeps the three tabs'
  weights independent), and a tab whose weights differ from the defaults
  opens with the section expanded and says so beside its result.
- A run leaves the UI thread (qt_worker), carries a token so a late answer
  to an old question is discarded, and the primary button reads "Update"
  when the controls differ from the result on screen.
- What the scores cannot count (inventory reported under a label that
  names no single element) is stated above the result, one sentence per
  note.
- A click on a chart mark selects the tank's row; a double click on a tank
  opens it, except in the Screening summary, where it finds blend partners
  for that tank (the tank is opened from the header button or the row's
  context menu).
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import polars as pl
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QCompleter, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import plot_helpers as ph
import vitrification_science as vsci
from correlation_science import parse_element_list
from data_model import HanfordDataset
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
    set_completions, wrapped_label,
)
from qt_help import make_help_button
from qt_table_format import format_number, humanize_text
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

BASIS_OPTIONS = ["elemental", "oxide"]
BASIS_LABELS = {"elemental": "Elemental", "oxide": "Oxide"}
SCREENING_PLOT_TYPES = ["Score breakdown (top tanks)", "Screening map (kg vs Ci)"]

EMPTY_TITLE = "No result yet"
HELP_SECTION = 12

SCREENING_SUBTITLE = (
    "A transparent composition score for every tank: a screening metric only, not an official "
    "waste classification or glass formulation model."
)
CANDIDATE_SUBTITLE = (
    "Rank the tanks richest in chosen target elements and poorest in penalty elements, by mass and "
    "by activity: a screening metric, not a glass formulation model."
)
BLEND_SUBTITLE = (
    "Rank the tanks whose composition would best complement one base tank in a blend: a screening "
    "metric, not a blending plan."
)


def _group_elements(name: str) -> str:
    return ", ".join(vsci.VITRIFICATION_GROUPS[name])


_GLASS_FORMERS = _group_elements("glass_former_or_intermediate")
_PROBLEM_ELEMENTS = ", ".join(vsci.VITRIFICATION_PROBLEM_ELEMENTS)

BASIS_TOOLTIP = (
    "How the glass-former share of a tank is measured.\n"
    f"Elemental: the mass fraction of {_GLASS_FORMERS} among the elements.\n"
    "Oxide: the weight fraction of the former oxides (SiO2, B2O3, P2O5, Al2O3) once the\n"
    "inventory is converted to oxides. Every other term is the same on both."
)

# What each weight multiplies: key -> (label shown beside the spin box, the term in words).
SCREENING_TERMS: Dict[str, Tuple[str, str]] = {
    "glass_former_weight": (
        "Glass formers",
        f"the fraction of glass formers and intermediates ({_GLASS_FORMERS})"),
    "modifier_weight": (
        "Modifiers",
        f"the mass fraction of alkali and alkaline-earth modifiers ({_group_elements('alkali_alkaline_modifier')})"),
    "problem_weight": (
        "Problem elements",
        f"the mass fraction of problem elements ({_PROBLEM_ELEMENTS})"),
    "volatile_weight": (
        "Volatiles",
        f"the mass fraction of volatile, halide and sulfate formers ({_group_elements('volatile_halide_sulfate')})"),
    "redox_weight": (
        "Redox-sensitive",
        f"the mass fraction of redox-sensitive elements ({_group_elements('transition_redox_sensitive')})"),
}
CANDIDATE_TERMS: Dict[str, Tuple[str, str]] = {
    "target_weight": (
        "Target elements",
        "the share of the target elements in the tank (their mass fraction plus their activity fraction)"),
    "glass_former_weight": (
        "Glass formers",
        f"the fraction of glass formers and intermediates ({_GLASS_FORMERS})"),
    "problem_weight": (
        "Problem elements",
        f"the mass fraction of problem elements ({_PROBLEM_ELEMENTS})"),
    "penalty_weight": (
        "Penalty elements",
        "the share of the penalty elements in the tank (their mass fraction plus their activity fraction)"),
}
BLEND_TERMS: Dict[str, Tuple[str, str]] = {
    "glass_former_gain_weight": (
        "Glass-former gain",
        "how much higher the partner's glass-former fraction is than the base tank's (a lower one counts as zero)"),
    "problem_reduction_weight": (
        "Problem-element reduction",
        "how much lower the partner's problem-element fraction is than the base tank's (a higher one counts as zero)"),
    "dissimilarity_weight": (
        "Dissimilarity",
        "one minus the cosine similarity of the two tanks' element mass-fraction profiles"),
    "glass_former_weight": (
        "Partner glass formers",
        "the partner's own glass-former fraction"),
}
CANDIDATE_FORMULA = (
    "Score = 100 × the sum of the four terms below, each multiplied by its weight. The share of the "
    "target and of the penalty elements adds their mass fraction (kg) and their activity fraction (Ci)."
)
BLEND_FORMULA = (
    "Score = 100 × the sum of the four terms below, each multiplied by its weight. Fractions are mass "
    "fractions of the tank's elemental inventory (kg)."
)

# The columns a reader looks for first; the rest follow in the order the science module writes them.
SUMMARY_LEAD = (
    "WasteSiteId", "TankFarm", "Vitrification_screening_score_proxy", "contrib_glass_former",
    "contrib_modifier", "contrib_problem", "contrib_volatile", "contrib_redox",
)
CANDIDATE_LEAD = (
    "WasteSiteId", "TankFarm", "User_search_score_proxy", "UserTargetFraction_sum",
    "UserPenaltyFraction_sum", "Vitrification_screening_score_proxy",
)
BLEND_LEAD = (
    "PartnerTank", "TankFarm", "Blend_complement_score_proxy", "GlassFormerGain_vs_base",
    "ProblemReduction_vs_base", "CosineSimilarity_to_base_fraction_profile",
)


# ----------------------------------------------------------------------
# Small shared pieces
# ----------------------------------------------------------------------
def weight_tooltip(term: str, default: float) -> str:
    """Which term of the score a weight multiplies, and which way it pushes."""
    if default > 0:
        direction = f"The default, {default:g}, is positive: a larger value raises the score."
    elif default < 0:
        direction = f"The default, {default:g}, is negative: a larger value lowers the score."
    else:
        direction = "The default is 0: the term is not counted."
    return (f"Multiplies {term}.\n{direction}\n"
            "A weight of the opposite sign reverses the effect; 0 removes the term.")


def _build_weight_spins(layout: QGridLayout, defaults: Dict[str, float], settings_prefix: str,
                        terms: Optional[Dict[str, Tuple[str, str]]] = None,
                        columns: int = 3) -> Dict[str, QDoubleSpinBox]:
    """One labelled spin box per weight, `columns` to a row, seeded from
    QSettings (the documented default when nothing was saved) and saved
    back on every change."""
    from PySide6.QtCore import QSettings
    from qt_help import APP_NAME
    settings = QSettings(APP_NAME, APP_NAME)
    terms = terms or {}
    spins: Dict[str, QDoubleSpinBox] = {}
    for position, (key, default) in enumerate(defaults.items()):
        fallback = key.replace("_weight", "").replace("_", " ").capitalize()
        label_text, term = terms.get(key, (fallback, f"the {fallback.lower()} term"))
        tooltip = weight_tooltip(term, default)
        label = field_label(label_text)
        label.setToolTip(tooltip)
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setDecimals(3)
        spin.setMinimumWidth(92)
        spin.setToolTip(tooltip)
        settings_key = f"{settings_prefix}_{key}"
        spin.setValue(settings.value(settings_key, default, type=float))
        spin.valueChanged.connect(lambda value, k=settings_key: QSettings(APP_NAME, APP_NAME).setValue(k, value))
        spins[key] = spin
        row, column = divmod(position, columns)
        layout.addWidget(label, row, 3 * column, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(spin, row, 3 * column + 1)
    for column in range(columns):
        layout.setColumnMinimumWidth(3 * column + 2, 20)        # the gutter between two pairs
    layout.setColumnStretch(3 * columns, 1)
    return spins


def _reset_weight_spins(spins: Dict[str, QDoubleSpinBox], defaults: Dict[str, float]) -> None:
    for key, spin in spins.items():
        spin.setValue(defaults[key])  # also persists the default via the valueChanged connection above


def _current_weights(spins: Dict[str, QDoubleSpinBox]) -> Dict[str, float]:
    return {key: spin.value() for key, spin in spins.items()}


_SCORE_TERMS = (
    ("glass_former_weight", "glass_former"),
    ("modifier_weight", "modifier"),
    ("problem_weight", "problem"),
    ("volatile_weight", "volatile"),
    ("redox_weight", "redox"),
)


def _format_score_formula(weights: Dict[str, float]) -> str:
    """The score written with each term's own sign.

    The readout used to hardcode a minus in front of abs() for the
    problem, volatile and redox terms, while the scoring code applies the
    raw signed weight. The defaults are negative so it read correctly at
    the defaults -- but the spin boxes accept -1000..1000, and +45 and
    -45 produced the byte-identical string. The one human-readable
    statement of the formula could not distinguish a penalty from a
    bonus, and _disclose_custom_weights puts it on screen precisely when
    a weight has been changed (review finding).
    """
    parts: List[str] = []
    for key, name in _SCORE_TERMS:
        value = float(weights.get(key, 0.0))
        sign = "-" if value < 0 else "+"
        parts.append(f"{sign} {abs(value):.3g}*{name}")
    body = " ".join(parts)
    if body.startswith("+ "):
        body = body[2:]
    return f"score = {body} (clipped to [-100, 100])"


def _disclose_custom_weights(section, spins: Dict[str, QDoubleSpinBox], defaults: Dict[str, float]) -> None:
    """QSettings-persisted weight edits survive restarts -- if any differ
    from the documented defaults, the collapsed Advanced section would
    silently apply a non-default score. Auto-expand it so the weights (and
    the formula readout) are on screen before anyone builds a summary."""
    if any(spins[key].value() != default for key, default in defaults.items()):
        section.set_expanded(True)


def make_basis_combo() -> QComboBox:
    """The basis selector: readable labels, the science module's own value as item data."""
    combo = QComboBox()
    for value in BASIS_OPTIONS:
        combo.addItem(BASIS_LABELS[value], value)
    combo.setToolTip(BASIS_TOOLTIP)
    return combo


def basis_value(combo: QComboBox) -> str:
    """The value the science call receives ("elemental" or "oxide")."""
    return str(combo.currentData() or combo.currentText().lower())


def lead_with(frame: pd.DataFrame, first: Sequence[str]) -> pd.DataFrame:
    """The same table with the columns named in `first` moved to the front."""
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    lead = [column for column in first if column in frame.columns]
    return frame[lead + [column for column in frame.columns if column not in lead]]


def cap_column_width(view: DataFrameTableView, column: str, width: int) -> None:
    """Keep one long-titled column from pushing its own numbers out of sight
    (the header shortens; its tooltip carries the full title)."""
    names = [str(c) for c in view.dataframe().columns]
    if column in names:
        header = view.horizontalHeader()
        section = names.index(column)
        if header.sectionSize(section) > width:
            header.resizeSection(section, width)


def fit_wrapped_rows(view: DataFrameTableView, cell_padding: int = 24, row_padding: int = 14) -> None:
    """Make every row of a word-wrapped table tall enough for its longest
    cell at the current column widths (a few sentences per cell: the
    reference notes are meant to be read in place, not elided)."""
    table = view.table
    model = table.model()
    header = table.horizontalHeader()
    metrics = table.fontMetrics()
    for row in range(model.rowCount()):
        height = table.verticalHeader().defaultSectionSize()
        for column in range(model.columnCount()):
            text = str(model.data(model.index(row, column), Qt.DisplayRole) or "")
            width = max(header.sectionSize(column) - cell_padding, 40)
            wrapped = metrics.boundingRect(0, 0, width, 100000, int(Qt.TextWordWrap | Qt.AlignLeft), text)
            height = max(height, wrapped.height() + row_padding)
        if table.rowHeight(row) != height:
            table.setRowHeight(row, height)


def unattributed_notes(dataset: HanfordDataset) -> List[str]:
    """One sentence per unit for the inventory the scores cannot count.

    Every total, fraction and score here is built over rows with a parsed
    element; whatever is reported under a label that names no single
    element is outside them. Same filter as
    tank_science.unattributed_mass_note (a positive inventory without an
    element), written for a reader: totals with thousands separators and
    their unit."""
    try:
        df = dataset.require_df().filter(pl.col("WasteSiteId").is_in(list(dataset.available_tanks())))
        dropped = df.filter((pl.col("Inventory") > 0) & pl.col("Element").is_null())
    except Exception:
        return []
    if dropped.is_empty():
        return []
    notes: List[str] = []
    units = sorted({str(unit) for unit in dropped.get_column("Units").to_list()})
    for unit in units:
        sub = dropped.filter(pl.col("Units").cast(pl.Utf8) == unit)
        total = format_number(float(sub.get_column("Inventory").sum()))
        labels = sorted({str(label) for label in sub.get_column("Analyte").to_list()})
        if len(labels) == 1:
            notes.append(
                f"Not counted in any total, fraction or score: {total} {unit} reported as {labels[0]}, "
                "a label that names no single element.")
            continue
        shown = ", ".join(labels[:3])
        rest = len(labels) - 3
        listed = f"{shown} and {rest:,} more" if rest > 0 else ", ".join(labels)
        notes.append(
            f"Not counted in any total, fraction or score: {total} {unit} reported under {len(labels):,} "
            f"labels that name no single element ({listed}).")
    return notes


class ListCompleter(QCompleter):
    """Completion for a comma-separated list: it completes the entry being typed, not the whole field."""

    _SPLIT = re.compile(r"[,;\s]+")

    def __init__(self, items: Sequence[str], edit: QLineEdit) -> None:
        super().__init__(sorted({str(item) for item in items if str(item)}, key=str.lower), edit)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterMode(Qt.MatchStartsWith)
        self.setCompletionMode(QCompleter.PopupCompletion)

    def splitPath(self, path: str) -> List[str]:  # noqa: N802 (Qt naming)
        return [self._SPLIT.split(path)[-1]]

    def pathFromIndex(self, index) -> str:  # noqa: N802 (Qt naming)
        completion = super().pathFromIndex(index)
        widget = self.widget()
        text = widget.text() if isinstance(widget, QLineEdit) else ""
        typed = self._SPLIT.split(text)[-1]
        return text[: len(text) - len(typed)] + completion


def set_list_completions(edit: QLineEdit, items: Sequence[str]) -> None:
    """Offer `items` for each entry of a comma-separated field."""
    edit.setCompleter(ListCompleter(items, edit))


@dataclass
class _Outcome:
    """What a run hands back to the UI thread."""
    token: int
    inputs: tuple
    frame: Optional[pd.DataFrame] = None
    notes: Tuple[str, ...] = ()
    error: str = ""


def _compute(token: int, inputs: tuple, dataset: HanfordDataset, job: Callable[[], pd.DataFrame]) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    try:
        return _Outcome(token=token, inputs=inputs, frame=job(), notes=tuple(unattributed_notes(dataset)))
    except Exception:
        return _Outcome(token=token, inputs=inputs, error=traceback.format_exc())


# ----------------------------------------------------------------------
# What the three tabs have in common
# ----------------------------------------------------------------------
class _ScoreTab(QWidget):
    """Run / Update, the worker round trip, the notes, the status line."""

    NAME = "Vitrification"                  # how the tab names itself in the status bar
    BUSY_TEXT = "Scoring tanks…"
    IDLE_HINT = "Press Run."
    RUN_TOOLTIP = "Score the tanks with the settings on this tab."
    WEIGHT_DEFAULTS: Dict[str, float] = {}

    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._run_token = 0
        self._running = False
        self._result_inputs: Optional[tuple] = None
        self._build_ui()
        self._reset_result()

    # -- pieces the three layouts share ---------------------------------
    def _root_layout(self) -> QVBoxLayout:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        return root

    def _add_header(self, root: QVBoxLayout, title: str, subtitle: str,
                    first_actions: Sequence[QWidget] = ()) -> None:
        self.header = PageHeader(title, subtitle)
        for widget in first_actions:                    # the tab's own quiet buttons come first
            self.header.add_action_widget(widget)
        self.export_btn = secondary_button(
            "Export...", "Write the tables and the chart of this tab to a dated folder.")
        self.export_btn.clicked.connect(self.export_tables)
        self.header.add_action_widget(self.export_btn)
        self.help_btn = make_help_button(self, HELP_SECTION, "Vitrification")
        self.header.add_action_widget(self.help_btn)
        root.addWidget(self.header)

    def _make_run_button(self) -> QPushButton:
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        return self.run_btn

    def _add_weights(self, layout: QVBoxLayout, defaults: Dict[str, float], prefix: str,
                     terms: Dict[str, Tuple[str, str]], columns: int) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.weight_spins = _build_weight_spins(grid, defaults, prefix, terms, columns)
        self.reset_btn = secondary_button(
            "Reset to defaults", "Put every score weight of this tab back to its documented default.")
        self.reset_btn.clicked.connect(lambda: _reset_weight_spins(self.weight_spins, defaults))
        last_row = (len(defaults) - 1) // columns
        grid.addWidget(self.reset_btn, last_row, 3 * columns + 1, Qt.AlignRight | Qt.AlignVCenter)
        layout.addLayout(grid)
        for spin in self.weight_spins.values():
            spin.valueChanged.connect(self._sync_run_label)

    def _add_notes(self, root: QVBoxLayout) -> None:
        # What qualifies a result: one calm line, the detail one click away.
        self.notes_banner = NotesBanner()
        self.warnings_label = self.notes_banner         # the name every page uses for this slot
        root.addWidget(self.notes_banner)

    def _add_splitter(self, root: QVBoxLayout, right: QWidget) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        self.plot = PlotWidget()
        self.plot.markClicked.connect(self._on_mark_clicked)
        splitter.addWidget(self.plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

    # -- what a tab provides ----------------------------------------------
    def _build_ui(self) -> None:
        raise NotImplementedError

    def _read_inputs(self) -> tuple:
        raise NotImplementedError

    def _show_result(self, outcome: _Outcome) -> None:
        raise NotImplementedError

    def _reset_result(self) -> None:
        raise NotImplementedError

    def _result_view(self) -> DataFrameTableView:
        raise NotImplementedError

    def export_tables(self) -> None:
        raise NotImplementedError

    # -- status ---------------------------------------------------------
    def _say(self, message: str) -> None:
        status_bar = getattr(self.app_window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message)

    def _set_busy(self, text: Optional[str]) -> None:
        busy = getattr(self.app_window, "set_busy", None)
        if callable(busy):
            busy(text)

    def _loaded(self) -> bool:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say(f"{self.NAME}: no dataset is loaded yet.")
            return False
        return True

    def _weights(self) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted(_current_weights(self.weight_spins).items()))

    def _custom_weight_note(self, weights: Dict[str, float]) -> List[str]:
        if all(weights.get(key) == default for key, default in self.WEIGHT_DEFAULTS.items()):
            return []
        return ["The score weights differ from the documented defaults (see Advanced options)."]

    # -- Run / Update -----------------------------------------------------
    def _sync_run_label(self, *_args) -> None:
        """"Run" asks; "Update" re-asks after a control changed under a standing result."""
        if self._running or not hasattr(self, "run_btn"):
            return
        changed = self._result_inputs is not None and self._result_inputs != self._read_inputs()
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else self.RUN_TOOLTIP)

    def _set_running(self, running: bool) -> None:
        was_running = self._running
        self._running = running
        self.run_btn.setEnabled(not running)
        if running:
            self.run_btn.setText("Running…")
            self._set_busy(self.BUSY_TEXT)
        else:
            if was_running:
                self._set_busy(None)
            self._sync_run_label()

    def _start(self, job: Callable[[], pd.DataFrame], inputs: tuple) -> None:
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        self._set_running(True)
        run_in_thread(lambda: _compute(token, inputs, dataset, job), self._on_run_done, self._on_run_error)

    def _on_run_error(self, tb_text: str) -> None:
        self._set_running(False)
        self._say(f"{self.NAME}: the run failed.")
        QMessageBox.critical(self, f"{self.NAME} failed", tb_text)

    def _on_run_done(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return          # superseded by a newer question, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._say(f"{self.NAME}: the run failed.")
            QMessageBox.critical(self, f"{self.NAME} failed", outcome.error)
            return
        self._result_inputs = outcome.inputs
        self._show_result(outcome)
        self._sync_run_label()

    # -- dataset ------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A result of the previous dataset must not stay on screen (or be
        # exported into the new dataset's folder), and a run still in
        # flight on it must not land.
        self._run_token += 1
        self._set_running(False)
        self._reset_result()
        self._refresh_choices()
        self._sync_run_label()

    def _refresh_choices(self) -> None:
        """Completions and lists that come from the loaded dataset."""

    # -- chart -> table -----------------------------------------------------
    def _tank_column(self) -> str:
        return "WasteSiteId"

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a bar or a point selects the row of the tank it stands for."""
        view = self._result_view()
        frame = view.dataframe()
        column = self._tank_column()
        if frame is None or frame.empty or column not in frame.columns:
            return
        wanted = {str(mark.get(key)) for key in ("tank_id", "label", "row") if mark.get(key) not in (None, "")}
        if not wanted:
            return
        hits = frame[column].astype(str).isin(wanted).to_numpy()
        if not hits.any():
            return
        row = int(hits.argmax())
        self._reveal(view)
        view.selectRow(row)
        index = view.proxy_model().mapFromSource(view.model().index(row, 0))
        if index.isValid():
            view.table.scrollTo(index)

    def _reveal(self, view: DataFrameTableView) -> None:
        """Bring `view` to the front when it shares a tab bar with others."""


# ----------------------------------------------------------------------
# Screening
# ----------------------------------------------------------------------
class ScreeningTab(_ScoreTab):
    # Emits a WasteSiteId when the user double-clicks a row in the
    # Screening summary table -- lets VitrificationPage jump straight to
    # Blend Partners with that tank pre-filled, instead of making the user
    # re-find and re-type the same tank ID in a different tab.
    jump_to_blend_partners = Signal(str)

    NAME = "Screening"
    BUSY_TEXT = "Scoring every tank…"
    IDLE_HINT = "Press Run to score every tank of the dataset."
    RUN_TOOLTIP = "Score every tank of the dataset with the weights of this tab."
    WEIGHT_DEFAULTS = vsci.SCREENING_WEIGHT_DEFAULTS

    def _build_ui(self) -> None:
        root = self._root_layout()
        self.open_tank_btn = secondary_button(
            "Open tank", "Open the tank selected in the Screening summary in the Tank explorer.")
        self.open_tank_btn.clicked.connect(self._open_selected_tank)
        self._add_header(root, "Screening", SCREENING_SUBTITLE, first_actions=[self.open_tank_btn])
        self.note_label = self.header.subtitle_label       # where the screening caveat is stated

        controls, row = control_row()
        row.addWidget(field_label("Basis"))
        self.basis_combo = make_basis_combo()
        self.basis_combo.currentIndexChanged.connect(self._sync_run_label)
        row.addWidget(self.basis_combo)
        row.addSpacing(8)
        row.addWidget(self._make_run_button())
        self.run_btn.clicked.connect(self.run_summary)
        row.addStretch(1)
        row.addWidget(field_label("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(SCREENING_PLOT_TYPES)
        self.plot_type_combo.setToolTip(
            "Score breakdown: which weighted terms drove the score of the best-ranked tanks.\n"
            "Screening map: every tank by its chemical (kg) and radiological (Ci) inventory."
        )
        self.plot_type_combo.currentTextChanged.connect(lambda _: self._update_plot())
        row.addWidget(self.plot_type_combo)
        root.addWidget(controls)

        # Score weights are tunable defaults, not something a first-time
        # user needs to touch -- collapsed by default, with a live formula
        # readout so the current weighting is never hidden, just the knobs.
        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()
        adv.addWidget(wrapped_label(
            "The score is the sum of five mass fractions of the tank's elemental inventory, each "
            "multiplied by its weight, limited to the range -100 to 100."))
        self._add_weights(adv, vsci.SCREENING_WEIGHT_DEFAULTS, "screening", SCREENING_TERMS, columns=3)
        self.formula_label = wrapped_label("")
        adv.addWidget(self.formula_label)
        for spin in self.weight_spins.values():
            spin.valueChanged.connect(self._update_formula_label)
        self._update_formula_label()
        root.addWidget(self.advanced_section)
        _disclose_custom_weights(self.advanced_section, self.weight_spins, vsci.SCREENING_WEIGHT_DEFAULTS)

        self._add_notes(root)

        self.tables = QTabWidget()
        self._table_views: Dict[str, DataFrameTableView] = {}
        for title_ in ["Screening summary", "Groups", "Waste notes"]:
            view = DataFrameTableView(title=title_, max_rows_display=500)
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        summary = self._table_views["Screening summary"]
        summary.doubleClicked.connect(self._on_summary_row_double_clicked)
        summary.clicked.connect(lambda _index: self._sync_open_tank())
        summary.setToolTip(
            "Double-click a tank to find blend partners for it.\n"
            "Open tank (above) or a right click on the row opens the tank in the Tank explorer.")
        self.tables.setTabToolTip(0, "One row per tank, best score first.")
        self.tables.setTabToolTip(1, "The element groups the score is built from.")
        self.tables.setTabToolTip(2, "What the waste classes mean, and what this screen does not say about them.")
        self._add_splitter(root, self.tables)

        # The two reference tables, as a reader would write them: group names
        # in words (the export keeps the identifiers), notes in full.
        groups = vsci.vitrification_group_definitions()
        groups["Group"] = groups["Group"].map(humanize_text)
        self._table_views["Groups"].set_dataframe(groups)
        notes_view = self._table_views["Waste notes"]
        notes_view.set_dataframe(vsci.waste_class_notes())
        notes_view.table.setWordWrap(True)
        notes_view.table.setTextElideMode(Qt.ElideNone)
        notes_view.horizontalHeader().resizeSection(0, 150)
        notes_view.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        fit = lambda *_args: fit_wrapped_rows(notes_view)       # noqa: E731
        notes_view.horizontalHeader().sectionResized.connect(fit)
        notes_view.proxy_model().layoutChanged.connect(fit)
        fit()

    def _update_formula_label(self) -> None:
        self.formula_label.setText(_format_score_formula(_current_weights(self.weight_spins)))

    # ------------------------------------------------------------------
    def _result_view(self) -> DataFrameTableView:
        return self._table_views["Screening summary"]

    def _reveal(self, view: DataFrameTableView) -> None:
        self.tables.setCurrentWidget(view)

    def _read_inputs(self) -> tuple:
        return (basis_value(self.basis_combo), self._weights())

    def _reset_result(self) -> None:
        self.summary_df = pd.DataFrame()
        self._result_inputs = None
        view = self._table_views["Screening summary"]
        view.set_dataframe(pd.DataFrame())
        view.set_empty_text(EMPTY_TITLE, "One row per tank appears after a run, best score first.")
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, self.IDLE_HINT)
        self._sync_open_tank()

    def run_summary(self) -> None:
        if not self._loaded():
            return
        dataset = self.dataset
        basis = basis_value(self.basis_combo)
        weights = _current_weights(self.weight_spins)
        self._start(lambda: vsci.tank_category_summary(dataset, weights=weights, basis=basis),
                    self._read_inputs())

    def _show_result(self, outcome: _Outcome) -> None:
        self.summary_df = outcome.frame if outcome.frame is not None else pd.DataFrame()
        view = self._table_views["Screening summary"]
        view.set_dataframe(lead_with(self.summary_df, SUMMARY_LEAD))
        cap_column_width(view, "Vitrification_screening_score_proxy", 200)
        self._update_plot()
        # Every denominator on this page is built over rows with a parsed
        # element; whatever parses to none is named beside the result
        # rather than silently missing from the totals and every share.
        basis, weights = outcome.inputs
        self.notes_banner.set_result_notes(self._custom_weight_note(dict(weights)) + list(outcome.notes))
        self._sync_open_tank()
        self._say(f"Screening: {len(self.summary_df):,} tank(s) scored on the "
                  f"{BASIS_LABELS.get(basis, basis).lower()} basis.")

    def _update_plot(self) -> None:
        if self.summary_df.empty:
            if self._result_inputs is None:
                self.plot.show_empty(EMPTY_TITLE, self.IDLE_HINT)
            else:
                self.plot.show_message("No tank holds an inventory with a recognised element.")
            return
        if self.plot_type_combo.currentText() == "Screening map (kg vs Ci)":
            ph.plot_vitrification_burden(self.plot, self.summary_df)
        else:
            ph.plot_vitrification_score_breakdown(self.plot, self.summary_df)

    def _on_summary_row_double_clicked(self, index) -> None:
        df = self._table_views["Screening summary"].dataframe()
        if df.empty or not (0 <= index.row() < len(df)) or "WasteSiteId" not in df.columns:
            return
        tank_id = str(df.iloc[index.row()]["WasteSiteId"])
        self.jump_to_blend_partners.emit(tank_id)

    # -- the tank of the selected row --------------------------------------
    def selected_tank(self) -> Optional[str]:
        view = self._table_views["Screening summary"]
        index = view.currentIndex()
        return view.tank_id_at(index) if index.isValid() else None

    def _sync_open_tank(self) -> None:
        self.open_tank_btn.setEnabled(not self.summary_df.empty)

    def _open_selected_tank(self) -> None:
        tank = self.selected_tank()
        if not tank:
            self._say("Screening: select a tank in the Screening summary first.")
            return
        handler = getattr(self.app_window, "open_tank", None)
        if callable(handler):
            handler(tank)
        else:
            self._say(f"{tank}: the Tank explorer is not available here.")

    def _on_mark_clicked(self, mark: dict) -> None:
        super()._on_mark_clicked(mark)
        self._sync_open_tank()

    def export_tables(self) -> None:
        if self.summary_df.empty or self.dataset is None:
            self._say("Screening: run first; there is nothing to export yet.")
            return
        tables = {
            "tank_screening_summary": self.summary_df,
            "group_definitions": vsci.vitrification_group_definitions(),
            "waste_class_notes": vsci.waste_class_notes(),
        }
        out_dir = export_utils.export_named_tables(self.dataset, "vitrification_screening", tables)
        try:
            self.plot.figure.savefig(out_dir / "screening_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Screening exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Screening outputs exported:\n{out_dir}")


# ----------------------------------------------------------------------
# Candidate search
# ----------------------------------------------------------------------
class CandidateSearchTab(_ScoreTab):
    NAME = "Candidate search"
    BUSY_TEXT = "Ranking candidate tanks…"
    IDLE_HINT = "Name the target elements, then press Run."
    RUN_TOOLTIP = "Rank the tanks for the target and penalty elements of this tab."
    WEIGHT_DEFAULTS = vsci.CANDIDATE_WEIGHT_DEFAULTS

    def _build_ui(self) -> None:
        root = self._root_layout()
        self._add_header(root, "Candidate search", CANDIDATE_SUBTITLE)

        controls, row = control_row()
        row.addWidget(field_label("Target elements"))
        self.target_edit = QLineEdit("Cs, Sr, Tc, I")
        self.target_edit.setMinimumWidth(180)
        self.target_edit.setMaximumWidth(320)
        self.target_edit.setPlaceholderText("Symbols, e.g. Cs, Sr")
        self.target_edit.setToolTip(
            "Elements a good candidate tank is rich in, separated by commas.\n"
            "Both the mass (kg) and the activity (Ci) reported for them count.")
        self.target_edit.textChanged.connect(self._sync_run_label)
        self.target_edit.returnPressed.connect(self.run_candidates)
        row.addWidget(self.target_edit, 1)
        row.addSpacing(8)
        row.addWidget(field_label("Basis"))
        self.basis_combo = make_basis_combo()
        self.basis_combo.currentIndexChanged.connect(self._sync_run_label)
        row.addWidget(self.basis_combo)
        row.addSpacing(8)
        row.addWidget(self._make_run_button())
        self.run_btn.clicked.connect(self.run_candidates)
        row.addStretch(2)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        filters = QHBoxLayout()
        filters.setSpacing(8)
        filters.addWidget(field_label("Penalty elements"))
        self.penalty_edit = QLineEdit("S, Cl, F, I, Cr, Mo, Tc")
        self.penalty_edit.setMinimumWidth(180)
        self.penalty_edit.setToolTip("Elements that count against a tank in proportion to their share of it.")
        self.penalty_edit.textChanged.connect(self._sync_run_label)
        filters.addWidget(self.penalty_edit, 2)
        filters.addSpacing(12)
        filters.addWidget(field_label("Required elements"))
        self.required_edit = QLineEdit("")
        self.required_edit.setMinimumWidth(110)
        self.required_edit.setPlaceholderText("None")
        self.required_edit.setToolTip(
            "Elements a tank must report (as mass or as activity) to be ranked at all. Empty: no requirement.")
        self.required_edit.textChanged.connect(self._sync_run_label)
        filters.addWidget(self.required_edit, 1)
        filters.addSpacing(12)
        filters.addWidget(field_label("Minimum inventory (kg)"))
        self.min_kg_spin = QDoubleSpinBox()
        self.min_kg_spin.setRange(0.0, 1e12)
        self.min_kg_spin.setDecimals(3)
        self.min_kg_spin.setGroupSeparatorShown(True)
        self.min_kg_spin.setMinimumWidth(110)
        self.min_kg_spin.setToolTip("Leave out the tanks whose total chemical inventory is below this mass.")
        self.min_kg_spin.valueChanged.connect(self._sync_run_label)
        filters.addWidget(self.min_kg_spin)
        filters.addSpacing(12)
        filters.addWidget(field_label("Tanks shown"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 500)
        self.top_n_spin.setValue(40)
        self.top_n_spin.setToolTip("How many of the best-ranked tanks the table and the chart keep.")
        self.top_n_spin.valueChanged.connect(self._sync_run_label)
        filters.addWidget(self.top_n_spin)
        adv.addLayout(filters)

        adv.addWidget(wrapped_label(CANDIDATE_FORMULA))
        self._add_weights(adv, vsci.CANDIDATE_WEIGHT_DEFAULTS, "candidate", CANDIDATE_TERMS, columns=3)

        root.addWidget(self.advanced_section)
        _disclose_custom_weights(self.advanced_section, self.weight_spins, vsci.CANDIDATE_WEIGHT_DEFAULTS)

        self._add_notes(root)

        self.table = DataFrameTableView(title="Candidate ranking", max_rows_display=500)
        self.table.setToolTip("Double-click a tank to open it in the Tank explorer.")
        self._add_splitter(root, self.table)

    # ------------------------------------------------------------------
    def _result_view(self) -> DataFrameTableView:
        return self.table

    def _refresh_choices(self) -> None:
        elements = dataset_values(self.dataset, "Element")
        for edit in (self.target_edit, self.penalty_edit, self.required_edit):
            set_list_completions(edit, elements)

    def _read_inputs(self) -> tuple:
        return (
            tuple(parse_element_list(self.target_edit.text())),
            tuple(parse_element_list(self.penalty_edit.text())),
            tuple(parse_element_list(self.required_edit.text())),
            basis_value(self.basis_combo), float(self.min_kg_spin.value()), int(self.top_n_spin.value()),
            self._weights(),
        )

    def _reset_result(self) -> None:
        self.candidates_df = pd.DataFrame()
        self._result_inputs = None
        self.table.set_dataframe(pd.DataFrame())
        self.table.set_empty_text(EMPTY_TITLE, "One row per ranked tank appears after a run, best score first.")
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, self.IDLE_HINT)

    def run_candidates(self) -> None:
        if not self._loaded():
            return
        dataset = self.dataset
        inputs = self._read_inputs()
        target, penalty, required, basis, min_kg, top_n, _weights = inputs
        weights = _current_weights(self.weight_spins)
        self._start(
            lambda: vsci.vitrification_candidate_search(
                dataset, list(target), list(penalty), list(required),
                min_total_kg=min_kg, top_n=top_n, weights=weights, basis=basis),
            inputs)

    def _show_result(self, outcome: _Outcome) -> None:
        self.candidates_df = outcome.frame if outcome.frame is not None else pd.DataFrame()
        target, _penalty, required, _basis, _min_kg, top_n, weights = outcome.inputs
        self.table.set_dataframe(lead_with(self.candidates_df, CANDIDATE_LEAD))
        cap_column_width(self.table, "User_search_score_proxy", 200)
        if self.candidates_df.empty:
            self.table.set_empty_text(
                "No tank qualifies", "Relax the required elements or the minimum inventory in Advanced options.")
            self.plot.show_message("No tank meets the required elements and the minimum inventory.")
        else:
            ph.plot_candidate_scores(self.plot, self.candidates_df, top_n=top_n)
        notes = self._custom_weight_note(dict(weights))
        if not target:
            notes.append("No target element is named: the ranking rests on the other three terms alone.")
        self.notes_banner.set_result_notes(notes + list(outcome.notes))
        subject = ", ".join(target) if target else "no target element"
        self._say(f"Candidate search: {len(self.candidates_df):,} tank(s) ranked for {subject}.")

    def export_tables(self) -> None:
        if self.candidates_df.empty or self.dataset is None:
            self._say("Candidate search: run first; there is nothing to export yet.")
            return
        out_dir = export_utils.export_named_tables(self.dataset, "vitrification_candidates", {"candidate_ranking": self.candidates_df})
        try:
            self.plot.figure.savefig(out_dir / "candidate_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Candidate ranking exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Candidate ranking exported:\n{out_dir}")


# ----------------------------------------------------------------------
# Blend partners
# ----------------------------------------------------------------------
class BlendPartnersTab(_ScoreTab):
    NAME = "Blend partners"
    BUSY_TEXT = "Ranking blend partners…"
    IDLE_HINT = "Choose a base tank, then press Run."
    RUN_TOOLTIP = "Rank every other tank as a blend partner of the base tank."
    WEIGHT_DEFAULTS = vsci.BLEND_WEIGHT_DEFAULTS

    def _build_ui(self) -> None:
        root = self._root_layout()
        self._add_header(root, "Blend partners", BLEND_SUBTITLE)

        controls, row = control_row()
        row.addWidget(field_label("Base tank"))
        self.base_tank_combo = QComboBox()
        self.base_tank_combo.setEditable(True)
        self.base_tank_combo.setInsertPolicy(QComboBox.NoInsert)
        self.base_tank_combo.setMinimumWidth(150)
        self.base_tank_combo.setToolTip(
            "The tank to find partners for. Pick it from the list or type its name:\n"
            "tank names of the dataset are offered as they are typed.")
        self.base_tank_combo.currentTextChanged.connect(self._sync_run_label)
        self.base_tank_combo.lineEdit().returnPressed.connect(self.run_blend)
        row.addWidget(self.base_tank_combo)
        row.addSpacing(8)
        row.addWidget(field_label("Basis"))
        self.basis_combo = make_basis_combo()
        self.basis_combo.currentIndexChanged.connect(self._sync_run_label)
        row.addWidget(self.basis_combo)
        row.addSpacing(8)
        row.addWidget(self._make_run_button())
        self.run_btn.clicked.connect(self.run_blend)
        row.addStretch(1)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        shown = QHBoxLayout()
        shown.setSpacing(8)
        shown.addWidget(field_label("Tanks shown"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 500)
        self.top_n_spin.setValue(40)
        self.top_n_spin.setToolTip("How many of the best-ranked partners the table and the chart keep.")
        self.top_n_spin.valueChanged.connect(self._sync_run_label)
        shown.addWidget(self.top_n_spin)
        shown.addSpacing(12)
        shown.addWidget(wrapped_label(BLEND_FORMULA), 1)
        adv.addLayout(shown)

        self._add_weights(adv, vsci.BLEND_WEIGHT_DEFAULTS, "blend", BLEND_TERMS, columns=3)

        root.addWidget(self.advanced_section)
        _disclose_custom_weights(self.advanced_section, self.weight_spins, vsci.BLEND_WEIGHT_DEFAULTS)

        self._add_notes(root)

        self.table = DataFrameTableView(title="Blend partners", max_rows_display=500)
        self.table.setToolTip("Double-click a partner tank to open it in the Tank explorer.")
        self._add_splitter(root, self.table)

    # ------------------------------------------------------------------
    def _result_view(self) -> DataFrameTableView:
        return self.table

    def _tank_column(self) -> str:
        return "PartnerTank"

    def _read_inputs(self) -> tuple:
        return (self.base_tank_combo.currentText().strip(), basis_value(self.basis_combo),
                int(self.top_n_spin.value()), self._weights())

    def _reset_result(self) -> None:
        self.blend_df = pd.DataFrame()
        self._result_inputs = None
        self.table.set_dataframe(pd.DataFrame())
        self.table.set_empty_text(EMPTY_TITLE, "One row per partner tank appears after a run, best score first.")
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, self.IDLE_HINT)

    def _refresh_choices(self) -> None:
        dataset = self.dataset
        if dataset is None or not dataset.is_loaded():
            return
        tanks = dataset.available_tanks()
        current = self.base_tank_combo.currentText()
        self.base_tank_combo.blockSignals(True)
        self.base_tank_combo.clear()
        self.base_tank_combo.addItems(tanks)
        if current and self.base_tank_combo.findText(current) >= 0:
            self.base_tank_combo.setCurrentText(current)
        elif tanks:
            self.base_tank_combo.setCurrentIndex(0)
        self.base_tank_combo.blockSignals(False)
        set_completions(self.base_tank_combo.lineEdit(), tanks)

    def set_base_tank_and_run(self, tank_id: str) -> None:
        """Public cross-tab entry point (VitrificationPage's jump from a
        double-clicked Screening row lands here) -- keeps the combo and
        run_blend() as private implementation details this tab is free to
        refactor without breaking its sibling."""
        self.base_tank_combo.setCurrentText(tank_id)
        self.run_blend()

    def run_blend(self) -> None:
        if not self._loaded():
            return
        inputs = self._read_inputs()
        base, basis, top_n, _weights = inputs
        if not base:
            self._say("Blend partners: choose a base tank first.")
            self.base_tank_combo.setFocus()
            return
        if base not in set(self.dataset.available_tanks()):
            self._say(f"Blend partners: the dataset has no tank named {base}.")
            self.base_tank_combo.setFocus()
            return
        dataset = self.dataset
        weights = _current_weights(self.weight_spins)
        self._start(
            lambda: vsci.blend_partner_search(dataset, base, top_n=top_n, weights=weights, basis=basis),
            inputs)

    def _show_result(self, outcome: _Outcome) -> None:
        self.blend_df = outcome.frame if outcome.frame is not None else pd.DataFrame()
        base, _basis, top_n, weights = outcome.inputs
        self.table.set_dataframe(lead_with(self.blend_df, BLEND_LEAD))
        cap_column_width(self.table, "Blend_complement_score_proxy", 250)
        if self.blend_df.empty:
            self.table.set_empty_text("No partner found", f"{base} reports no chemical inventory to compare.")
            self.plot.show_message(f"{base} reports no chemical inventory (kg) to compare with other tanks.")
        else:
            ph.plot_blend_scores(self.plot, self.blend_df, top_n=top_n)
        self.notes_banner.set_result_notes(self._custom_weight_note(dict(weights)) + list(outcome.notes))
        self._say(f"Blend partners: {len(self.blend_df):,} partner(s) ranked for {base}.")

    def export_tables(self) -> None:
        if self.blend_df.empty or self.dataset is None:
            self._say("Blend partners: run first; there is nothing to export yet.")
            return
        out_dir = export_utils.export_named_tables(self.dataset, "vitrification_blend", {"blend_partners": self.blend_df})
        try:
            self.plot.figure.savefig(out_dir / "blend_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Blend partner ranking exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Blend partner search exported:\n{out_dir}")
