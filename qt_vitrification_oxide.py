"""
qt_vitrification_oxide.py — Vitrification "Oxide Chemistry" sub-tab: the
element -> oxide stoichiometric conversion, NBO/T, optical basicity, and
envelope-comparison workspace.

Layout, as on its sibling tabs: a header, one control row (farm, tank,
Run, and the sample on screen), the Advanced options (oxide mapping,
NBO/T roles, envelope, GlassNet), the notes that qualify the result, then
the tank list, the chart of the sample and its tables side by side.

A tank is selected as soon as a dataset is loaded, so the first press of
Run gives a result. The conversion leaves the UI thread (qt_worker) and
carries a token, so a late answer to an old question is discarded.

GlassNet integration reuses glass_science verbatim (it needs glasspy and
PyTorch: available when run from Python, shown disabled with the reason
in the packaged application).
"""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import glass_science as gscience
import oxide_science as oxsci
import tank_science as tsci
from data_model import HanfordDataset
from plot_grammar import PlotSpec, render_plot
from qt_components import (
    NotesBanner, PageHeader, control_row, field_label, secondary_button, set_completions,
    wrapped_label,
)
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

PAGE_SUBTITLE = (
    "The oxide-basis composition of one or more tanks, with optical basicity, NBO/T and an optional "
    "envelope check: a screening tool, not an official glass model."
)
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "Select one or more tanks, then press Run."
HELP_SECTION = 12
PLOT_TOP_N = 15
# The order the composition table is read in: what it is, how much of it,
# then the rest. Only the header is rearranged; the table handed to the
# view (and to the export) is the science module's own.
COLUMN_ORDER = ("Component", "Wt_pct", "Mol_pct", "Kind", "Source_elements", "Mass_g", "Mol")

NBO_T_NOTE = (
    "NBO/T is a simplified approximation: it does not model the charge balance of Al and B by "
    "alkali. Verify it against a preferred reference before publication use."
)
GLASSNET_SOURCE = (
    "Property predictions: GlassNet (Cassar, Ceram. Int. 49 (2023) 36013), trained on SciGlass."
)
BASICITY_SOURCE = "Λ: Rodriguez & McCloy, PNNL-20184 (2011) Table B.1."
# Everything the tab says about its own basis and sources, in one piece
# (the notes above a result carry the same statements, one per line).
SOURCES_NOTE = (
    oxsci.oxide_basis_note() + " "
    "Screening tool, not an official glass model. " + BASICITY_SOURCE + " "
    + GLASSNET_SOURCE + " " + NBO_T_NOTE
)

GLASSNET_BUTTON = "Predict properties (GlassNet)"
GLASSNET_TOOLTIP = (
    "Predict glass properties of the sample on screen with GlassNet.\n"
    "The first prediction loads the model, which takes a while."
)
GLASSNET_UNAVAILABLE = (
    "GlassNet predictions need the optional glasspy package (with PyTorch), which is not part of\n"
    "the packaged application. They are available when the application runs from Python with\n"
    "glasspy installed."
)


@dataclass
class _Outcome:
    """What a conversion hands back to the UI thread."""
    token: int
    tanks: Tuple[str, ...]
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    error: str = ""


def _convert(token: int, dataset: HanfordDataset, tanks: Tuple[str, ...],
             oxide_map: Dict[str, Optional[str]]) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    try:
        return _Outcome(token=token, tanks=tanks,
                        tables=oxsci.tank_oxide_composition(dataset, list(tanks), oxide_map))
    except Exception:
        return _Outcome(token=token, tanks=tanks, error=traceback.format_exc())


def _small_table(headers: List[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.verticalHeader().setVisible(False)
    table.setMinimumHeight(100)
    table.setMaximumHeight(150)
    return table


class OxideChemistryTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.oxide_tables: Dict[str, pd.DataFrame] = {}
        self._result_tanks: Optional[Tuple[str, ...]] = None
        self._run_token = 0
        self._running = False
        self._filling = False                       # a table is being filled by the tab, not edited
        self._excluded_from_basicity: List[str] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader("Oxide chemistry", PAGE_SUBTITLE)
        self.export_btn = secondary_button(
            "Export...", "Write every converted composition and the assumptions used to a dated folder.")
        self.export_btn.clicked.connect(self.export_tables)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, HELP_SECTION, "Vitrification"))
        root.addWidget(self.header)

        # Primary path: pick a farm, pick tanks, run. Oxide-mapping
        # overrides, NBO/T roles, the envelope check, and GlassNet are all
        # tuning/extra steps on top of that -- collapsed by default.
        controls, row = control_row()
        row.addWidget(field_label("Farm"))
        self.farm_combo = QComboBox()
        self.farm_combo.addItem("All")
        self.farm_combo.setToolTip("Limit the tank list to one tank farm.")
        self.farm_combo.currentTextChanged.connect(lambda _: self._refresh_tank_list())
        row.addWidget(self.farm_combo)
        row.addSpacing(8)
        row.addWidget(field_label("Tank"))
        self.tank_edit = QLineEdit()
        self.tank_edit.setPlaceholderText("e.g. 241-AN-102")
        self.tank_edit.setMinimumWidth(150)
        self.tank_edit.setMaximumWidth(200)
        self.tank_edit.setToolTip(
            "Type a tank name and press Enter to select that tank and run.\n"
            "Tank names of the dataset are offered as they are typed.")
        self.tank_edit.returnPressed.connect(self._run_typed_tank)
        row.addWidget(self.tank_edit)
        row.addSpacing(8)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.build_composition)
        row.addWidget(self.run_btn)
        row.addStretch(1)
        row.addWidget(field_label("Sample"))
        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumWidth(190)
        self.sample_combo.setToolTip(
            "Which converted composition is on screen: one of the selected tanks, or their blend.")
        self.sample_combo.setEnabled(False)
        self.sample_combo.currentTextChanged.connect(self._on_sample_changed)
        row.addWidget(self.sample_combo)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        self._build_advanced(self.advanced_section.content_layout())
        root.addWidget(self.advanced_section)

        self.notes_banner = NotesBanner()
        self.warnings_label = self.notes_banner         # the name every page uses for this slot
        root.addWidget(self.notes_banner)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        tanks_pane = QWidget()
        tanks_layout = QVBoxLayout(tanks_pane)
        tanks_layout.setContentsMargins(0, 0, 8, 0)
        tanks_layout.setSpacing(6)
        tanks_layout.addWidget(field_label("Tanks"))
        self.tank_list = QListWidget()
        self.tank_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tank_list.setMinimumWidth(130)
        self.tank_list.setToolTip(
            "The tanks to convert. Select several with Ctrl or Shift:\n"
            "two or more also give the blend obtained by combining their contents.")
        self.tank_list.itemSelectionChanged.connect(self._sync_run_label)
        tanks_layout.addWidget(self.tank_list, 1)
        tanks_layout.addWidget(wrapped_label("Two or more tanks also give their blend."))
        splitter.addWidget(tanks_pane)

        self.plot = PlotWidget()
        self.plot.markClicked.connect(self._on_mark_clicked)
        splitter.addWidget(self.plot)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.tables = QTabWidget()
        self._table_views: Dict[str, DataFrameTableView] = {}
        for title_ in ["Oxide composition", "Envelope check", "GlassNet"]:
            view = DataFrameTableView(title=title_, max_rows_display=500)
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        self._table_views["Oxide composition"].set_empty_text(
            EMPTY_TITLE, "One row per oxide or element of the sample appears after a run.")
        self._table_views["Envelope check"].set_empty_text(
            "No envelope set", "Add oxide limits under Advanced options to compare the sample with them.")
        if gscience.glassnet_available():
            self._table_views["GlassNet"].set_empty_text(
                "No prediction yet", f"Open Advanced options and choose {GLASSNET_BUTTON}.")
        else:
            self._table_views["GlassNet"].set_empty_text(
                "Not available in this installation",
                "Property predictions need the optional glasspy package.")
        rl.addWidget(self.tables, 1)
        self.summary_label = wrapped_label("")
        self.summary_label.setVisible(False)
        rl.addWidget(self.summary_label)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([170, 560, 420])
        self.splitter = splitter

        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self._sync_run_label()

    def _build_advanced(self, adv: QVBoxLayout) -> None:
        columns = QHBoxLayout()
        columns.setSpacing(16)

        mapping = QVBoxLayout()
        mapping.setSpacing(4)
        mapping_caption = field_label("Element to assumed oxide")
        mapping.addWidget(mapping_caption)
        self.oxide_map_table = _small_table(["Element", "Assumed oxide"])
        self.oxide_map_table.setToolTip(
            "The oxide each element of the selected tanks is converted to.\n"
            "Multivalent elements offer alternatives; any cell can be edited.\n"
            "An empty cell keeps the element as it is. Filled by the first run.")
        mapping.addWidget(self.oxide_map_table)
        columns.addLayout(mapping, 1)

        roles = QVBoxLayout()
        roles.setSpacing(4)
        roles_caption = field_label("Oxide role (for NBO/T)")
        roles.addWidget(roles_caption)
        self.role_table = _small_table(["Oxide", "Role"])
        self.role_table.setToolTip(
            "Whether each oxide counts as a network former, a modifier or neither\n"
            "in the NBO/T estimate. Filled by the first run.")
        roles.addWidget(self.role_table)
        columns.addLayout(roles, 1)

        envelope = QVBoxLayout()
        envelope.setSpacing(4)
        env_head = QHBoxLayout()
        env_head.setSpacing(6)
        env_head.addWidget(field_label("Envelope (wt% limits)"))
        env_head.addStretch(1)
        for text, tooltip, handler in (
            ("Add row", "Add an oxide with its lowest and highest allowed wt%.", self._add_envelope_row),
            ("Remove row", "Remove the selected envelope row.", self._remove_envelope_row),
            ("Load...", "Read an envelope from a JSON file.", self.load_envelope),
            ("Save...", "Write this envelope to a JSON file.", self.save_envelope),
        ):
            button = secondary_button(text, tooltip)
            button.clicked.connect(handler)
            env_head.addWidget(button)
        envelope.addLayout(env_head)
        for caption in (mapping_caption, roles_caption):       # the three captions share one line
            caption.setMinimumHeight(button.sizeHint().height())
        self.envelope_table = _small_table(["Oxide", "Min wt%", "Max wt%"])
        self.envelope_table.setToolTip(
            "Composition limits to compare the sample with. Empty by default: no envelope is assumed.")
        self.envelope_table.itemChanged.connect(self._on_envelope_edited)
        envelope.addWidget(self.envelope_table)
        columns.addLayout(envelope, 2)
        adv.addLayout(columns)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.recompute_btn = secondary_button(
            "Apply assumptions",
            "Convert the selected tanks again with the element to oxide table above,\n"
            "then recompute optical basicity, NBO/T and the envelope check.")
        self.recompute_btn.clicked.connect(self.apply_assumptions)
        actions.addWidget(self.recompute_btn)
        self.gn_btn = secondary_button(GLASSNET_BUTTON, GLASSNET_TOOLTIP)
        self.gn_btn.clicked.connect(self.run_glassnet)
        actions.addWidget(self.gn_btn)
        if not gscience.glassnet_available():
            self.gn_btn.setEnabled(False)
            self.gn_btn.setToolTip(GLASSNET_UNAVAILABLE)
        actions.addStretch(1)
        adv.addLayout(actions)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _say(self, message: str) -> None:
        status_bar = getattr(self.app_window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message)

    def _set_busy(self, text: Optional[str]) -> None:
        busy = getattr(self.app_window, "set_busy", None)
        if callable(busy):
            busy(text)

    def _sync_run_label(self, *_args) -> None:
        """"Run" converts; "Update" converts again after the tank selection changed."""
        if self._running or not hasattr(self, "run_btn"):
            return
        changed = self._result_tanks is not None and self._result_tanks != tuple(self.selected_tanks())
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The tank selection differs from the result on screen: run again to bring it up to date."
            if changed else "Convert the selected tanks to an oxide composition.")

    def _set_running(self, running: bool) -> None:
        was_running = self._running
        self._running = running
        self.run_btn.setEnabled(not running)
        if running:
            self.run_btn.setText("Running…")
            self._set_busy("Converting to oxides…")
        else:
            if was_running:
                self._set_busy(None)
            self._sync_run_label()

    # ------------------------------------------------------------------
    # Dataset, farm and tank selection
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A composition of the previous dataset must not stay on screen (or
        # be exported into the new dataset's folder), and a conversion
        # still in flight on it must not land.
        self._run_token += 1
        self._set_running(False)
        self._clear_result()
        if not dataset.is_loaded():
            return
        current_farm = self.farm_combo.currentText()
        self.farm_combo.blockSignals(True)
        self.farm_combo.clear()
        self.farm_combo.addItems(tsci.available_farms_with_all(dataset))
        restore = self.farm_combo.findText(current_farm)
        self.farm_combo.setCurrentIndex(restore if restore >= 0 else 0)
        self.farm_combo.blockSignals(False)
        self._refresh_tank_list()
        set_completions(self.tank_edit, dataset.available_tanks())
        completer = self.tank_edit.completer()
        if completer is not None:
            completer.activated.connect(self._select_typed_tank)

    def _refresh_tank_list(self) -> None:
        """The tanks of the chosen farm. What was selected stays selected
        when it is still listed; otherwise the first tank is, so that Run
        always has something to convert."""
        if self.dataset is None or not self.dataset.is_loaded():
            return
        keep = set(self.selected_tanks())
        farm = self.farm_combo.currentText()
        tanks = tsci.tanks_in_farm(self.dataset, farm)
        self.tank_list.blockSignals(True)
        self.tank_list.clear()
        self.tank_list.addItems(tanks)
        kept = False
        for i in range(self.tank_list.count()):
            item = self.tank_list.item(i)
            if item.text() in keep:
                item.setSelected(True)
                kept = True
        if not kept and self.tank_list.count():
            self.tank_list.item(0).setSelected(True)
            self.tank_list.setCurrentRow(0)
        self.tank_list.blockSignals(False)
        self._sync_run_label()

    def selected_tanks(self) -> List[str]:
        return [item.text() for item in self.tank_list.selectedItems()]

    def select_tank(self, tank_id: str) -> bool:
        """Select one tank by name (any case), showing every farm if the
        current one does not list it. False when the dataset has no such tank."""
        wanted = str(tank_id or "").strip().lower()
        if not wanted or self.dataset is None or not self.dataset.is_loaded():
            return False
        names = {str(t).lower(): str(t) for t in self.dataset.available_tanks()}
        name = names.get(wanted)
        if name is None:
            return False
        if not self.tank_list.findItems(name, Qt.MatchExactly):
            self.farm_combo.setCurrentIndex(0)          # "All"
        items = self.tank_list.findItems(name, Qt.MatchExactly)
        if not items:
            return False
        self.tank_list.clearSelection()
        items[0].setSelected(True)
        self.tank_list.setCurrentItem(items[0])
        self.tank_list.scrollToItem(items[0])
        return True

    def _select_typed_tank(self, text=None) -> bool:
        """Select the tank named in the field (or by the completion just chosen)."""
        text = (text if isinstance(text, str) else self.tank_edit.text()).strip()
        if not text:
            return bool(self.selected_tanks())
        if not self.select_tank(text):
            self._say(f"Oxide chemistry: the dataset has no tank named {text}.")
            return False
        return True

    def _run_typed_tank(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Oxide chemistry: no dataset is loaded yet.")
            return
        if self._select_typed_tank():
            self.build_composition()

    # ------------------------------------------------------------------
    # Element -> oxide table
    # ------------------------------------------------------------------
    def _elements_in_tanks(self, tank_ids: List[str]) -> List[str]:
        df = self.dataset.require_df().filter(
            (pl.col("Units") == "kg") & pl.col("Element").is_not_null() & pl.col("WasteSiteId").is_in(tank_ids)
        )
        return [str(e) for e in df.get_column("Element").drop_nulls().unique().to_list()]

    def _populate_oxide_map_table(self, elements: List[str]) -> None:
        """Rebuild the rows for `elements`, KEEPING whatever the user has
        already chosen for an element that survives the rebuild.

        This table is the tab's one editable assumption, and it used to be
        wiped and reseeded from the defaults immediately before
        _rebuild_and_show read it, so no edit could ever reach a
        computation (review finding)."""
        keep = self._current_oxide_map()
        self.oxide_map_table.setRowCount(0)
        for element in sorted(elements):
            row = self.oxide_map_table.rowCount()
            self.oxide_map_table.insertRow(row)
            self.oxide_map_table.setItem(row, 0, QTableWidgetItem(element))
            alternatives = oxsci.ALTERNATIVE_OXIDES.get(element)
            chosen = keep.get(element) if element in keep else oxsci.DEFAULT_OXIDE_MAP.get(element)
            if alternatives:
                combo = QComboBox()
                combo.addItems(alternatives)
                if chosen in alternatives:
                    combo.setCurrentText(chosen)
                self.oxide_map_table.setCellWidget(row, 1, combo)
            else:
                self.oxide_map_table.setItem(row, 1, QTableWidgetItem(chosen or ""))

    def _current_oxide_map(self) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        for row in range(self.oxide_map_table.rowCount()):
            element_item = self.oxide_map_table.item(row, 0)
            if element_item is None or not element_item.text().strip():
                continue
            element = element_item.text().strip()
            widget = self.oxide_map_table.cellWidget(row, 1)
            if isinstance(widget, QComboBox):
                text = widget.currentText().strip()
            else:
                item = self.oxide_map_table.item(row, 1)
                text = item.text().strip() if item is not None else ""
            out[element] = text or None
        return out

    # ------------------------------------------------------------------
    # Oxide role table (NBO/T)
    # ------------------------------------------------------------------
    def _populate_role_table(self, oxides: List[str]) -> None:
        self.role_table.setRowCount(0)
        for oxide in sorted(oxides):
            row = self.role_table.rowCount()
            self.role_table.insertRow(row)
            self.role_table.setItem(row, 0, QTableWidgetItem(oxide))
            combo = QComboBox()
            combo.addItems(["former", "modifier", "other"])
            combo.setCurrentText(oxsci.default_oxide_role(oxide))
            # A changed role changes NBO/T at once: no second button to press.
            combo.currentTextChanged.connect(lambda _text: self.recompute_summary())
            self.role_table.setCellWidget(row, 1, combo)

    def _current_role_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for row in range(self.role_table.rowCount()):
            oxide_item = self.role_table.item(row, 0)
            widget = self.role_table.cellWidget(row, 1)
            if oxide_item is None or not isinstance(widget, QComboBox):
                continue
            out[oxide_item.text().strip()] = widget.currentText()
        return out

    # ------------------------------------------------------------------
    # Envelope table
    # ------------------------------------------------------------------
    def _add_envelope_row(self) -> None:
        self._filling = True
        try:
            row = self.envelope_table.rowCount()
            self.envelope_table.insertRow(row)
            self.envelope_table.setItem(row, 0, QTableWidgetItem(""))
            self.envelope_table.setItem(row, 1, QTableWidgetItem("0"))
            self.envelope_table.setItem(row, 2, QTableWidgetItem("100"))
        finally:
            self._filling = False

    def _remove_envelope_row(self) -> None:
        row = self.envelope_table.currentRow()
        if row >= 0:
            self.envelope_table.removeRow(row)
            self.recompute_summary()

    def _populate_envelope_table(self, envelope: Dict[str, Tuple[float, float]]) -> None:
        self._filling = True
        try:
            self.envelope_table.setRowCount(0)
            for oxide, (lo, hi) in envelope.items():
                row = self.envelope_table.rowCount()
                self.envelope_table.insertRow(row)
                self.envelope_table.setItem(row, 0, QTableWidgetItem(oxide))
                self.envelope_table.setItem(row, 1, QTableWidgetItem(str(lo)))
                self.envelope_table.setItem(row, 2, QTableWidgetItem(str(hi)))
        finally:
            self._filling = False

    def _on_envelope_edited(self, _item) -> None:
        """An edited limit is compared with the sample at once."""
        if not self._filling:
            self.recompute_summary()

    def _current_envelope(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for row in range(self.envelope_table.rowCount()):
            oxide_item = self.envelope_table.item(row, 0)
            if oxide_item is None or not oxide_item.text().strip():
                continue
            min_item = self.envelope_table.item(row, 1)
            max_item = self.envelope_table.item(row, 2)
            try:
                lo = float(min_item.text()) if min_item is not None and min_item.text().strip() else 0.0
                hi = float(max_item.text()) if max_item is not None and max_item.text().strip() else 100.0
            except ValueError:
                continue
            out[oxide_item.text().strip()] = (lo, hi)
        return out

    def load_envelope(self) -> None:
        from PySide6.QtCore import QSettings
        from qt_help import APP_NAME
        settings = QSettings(APP_NAME, APP_NAME)
        last_dir = settings.value("oxide_envelope_last_path", "", type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Load envelope JSON", last_dir, "JSON (*.json)")
        if not path:
            return
        try:
            envelope = oxsci.load_envelope_json(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load envelope", str(exc))
            return
        self._populate_envelope_table(envelope)
        settings.setValue("oxide_envelope_last_path", path)
        self.recompute_summary()
        self._say(f"Envelope loaded: {path}")

    def save_envelope(self) -> None:
        envelope = self._current_envelope()
        if not envelope:
            self._say("Oxide chemistry: add at least one envelope row before saving.")
            return
        from PySide6.QtCore import QSettings
        from qt_help import APP_NAME
        settings = QSettings(APP_NAME, APP_NAME)
        last_dir = settings.value("oxide_envelope_last_path", "", type=str)
        path, _ = QFileDialog.getSaveFileName(self, "Save envelope JSON", last_dir, "JSON (*.json)")
        if not path:
            return
        oxsci.save_envelope_json(path, envelope)
        settings.setValue("oxide_envelope_last_path", path)
        self._say(f"Envelope saved: {path}")

    # ------------------------------------------------------------------
    # Build / recompute
    # ------------------------------------------------------------------
    def build_composition(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Oxide chemistry: no dataset is loaded yet.")
            return
        tanks = self.selected_tanks()
        if not tanks:
            self._say("Oxide chemistry: select one or more tanks first.")
            return
        elements = self._elements_in_tanks(tanks)
        if not elements:
            self._say("Oxide chemistry: the selected tanks report no chemical inventory (kg) to convert.")
            if not self.oxide_tables:
                self.plot.show_message("The selected tanks report no chemical inventory (kg) to convert.")
            return
        self._populate_oxide_map_table(elements)
        self._rebuild_and_show()

    def _rebuild_and_show(self) -> None:
        tanks = tuple(self.selected_tanks())
        if not tanks:
            self._say("Oxide chemistry: select one or more tanks first.")
            return
        oxide_map = self._current_oxide_map()
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        self._set_running(True)
        run_in_thread(lambda: _convert(token, dataset, tanks, oxide_map),
                      self._on_converted, self._on_convert_error)

    def _on_convert_error(self, tb_text: str) -> None:
        self._set_running(False)
        self._say("Oxide chemistry: the conversion failed.")
        QMessageBox.critical(self, "Oxide chemistry failed", tb_text)

    def _on_converted(self, outcome: _Outcome) -> None:
        if outcome.token != self._run_token:
            return          # superseded by a newer run, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._say("Oxide chemistry: the conversion failed.")
            QMessageBox.critical(self, "Oxide chemistry failed", outcome.error)
            return
        if not outcome.tables:
            self._say("Oxide chemistry: the selected tanks report no chemical inventory (kg) to convert.")
            if not self.oxide_tables:
                self.plot.show_message("The selected tanks report no chemical inventory (kg) to convert.")
            return
        self.oxide_tables = outcome.tables
        self._result_tanks = outcome.tanks
        all_oxides = sorted({
            str(c) for table in self.oxide_tables.values() for c in table.loc[table["Kind"] == "oxide", "Component"]
        })
        self._populate_role_table(all_oxides)

        current = self.sample_combo.currentText()
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.addItems(list(self.oxide_tables.keys()))
        restore = self.sample_combo.findText(current)
        self.sample_combo.setCurrentIndex(restore if restore >= 0 else 0)
        self.sample_combo.blockSignals(False)
        self.sample_combo.setEnabled(True)
        self._show_current_sample()
        self._sync_run_label()
        count = len(self.oxide_tables)
        self._say(f"Oxide chemistry: {count} composition{'' if count == 1 else 's'} built for "
                  f"{len(outcome.tanks)} tank{'' if len(outcome.tanks) == 1 else 's'}.")

    def _clear_result(self) -> None:
        self.oxide_tables = {}
        self._result_tanks = None
        self._excluded_from_basicity = []
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.blockSignals(False)
        self.sample_combo.setEnabled(False)
        for view in self._table_views.values():
            view.set_dataframe(pd.DataFrame())
        self.summary_label.setText("")
        self.summary_label.setVisible(False)
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self._sync_run_label()

    def _on_sample_changed(self, _text: str) -> None:
        self._show_current_sample()

    def _show_current_sample(self) -> None:
        table = self.oxide_tables.get(self.sample_combo.currentText())
        if table is None:
            return
        view = self._table_views["Oxide composition"]
        view.set_dataframe(table)
        self._order_columns(view, COLUMN_ORDER)
        self._update_plot()
        self.recompute_summary()

    @staticmethod
    def _order_columns(view: DataFrameTableView, order) -> None:
        """Show the columns named in `order` first, by moving header sections."""
        names = [str(c) for c in view.dataframe().columns]
        header = view.horizontalHeader()
        position = 0
        for name in order:
            if name in names:
                header.moveSection(header.visualIndex(names.index(name)), position)
                position += 1

    def _update_plot(self) -> None:
        sample = self.sample_combo.currentText()
        table = self.oxide_tables.get(sample)
        if table is None or table.empty:
            self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
            return
        render_plot(self.plot, table, PlotSpec(
            mark="bar", x="Wt_pct", y="Component", top_n=PLOT_TOP_N,
            title=f"Oxide-basis composition of {sample}",
            xlabel="Share of the converted inventory (wt%)", ylabel="Oxide or element",
        ))

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a bar selects the row of that oxide or element."""
        view = self._table_views["Oxide composition"]
        frame = view.dataframe()
        if frame is None or frame.empty or "Component" not in frame.columns:
            return
        wanted = {str(mark.get(key)) for key in ("label", "row") if mark.get(key) not in (None, "")}
        hits = frame["Component"].astype(str).isin(wanted).to_numpy()
        if not wanted or not hits.any():
            return
        row = int(hits.argmax())
        self.tables.setCurrentWidget(view)
        view.selectRow(row)
        index = view.proxy_model().mapFromSource(view.model().index(row, 0))
        if index.isValid():
            view.table.scrollTo(index)

    def apply_assumptions(self) -> None:
        """Re-convert with the user's current Element -> oxide choices,
        then recompute the derived quantities.

        Reconversion has to happen here: the only other caller of
        _rebuild_and_show is build_composition, which reseeds the map
        first, so before this the edited table could not reach any
        computation at all (review finding)."""
        if not self.oxide_tables:
            self.build_composition()
            return
        self._rebuild_and_show()

    def recompute_summary(self) -> None:
        table = self.oxide_tables.get(self.sample_combo.currentText())
        if table is None or table.empty:
            return
        role_map = self._current_role_map()
        envelope = self._current_envelope()
        summary = oxsci.composition_summary(table, role_map=role_map, envelope=envelope)
        self._table_views["Envelope check"].set_dataframe(summary["envelope_table"])

        basicity_text = "n/a" if math.isnan(summary["optical_basicity"]) else f"{summary['optical_basicity']:.3f}"
        nbo_text = "n/a (no former-role oxide present)" if math.isnan(summary["NBO_T"]) else f"{summary['NBO_T']:.3f}"
        parts = [f"Optical basicity Λ = {basicity_text}", f"NBO/T = {nbo_text}"]
        self._excluded_from_basicity = list(summary["excluded_from_basicity"])
        if self._excluded_from_basicity:
            parts.append("Excluded from Λ (no PNNL-20184 value): " + ", ".join(self._excluded_from_basicity))
        self.summary_label.setText("   ·   ".join(parts))
        self.summary_label.setVisible(True)
        self.notes_banner.set_result_notes(self.result_notes())

    def result_notes(self) -> List[str]:
        """What qualifies the composition on screen, one statement per note."""
        notes = [oxsci.oxide_basis_note(), NBO_T_NOTE, gscience.OPTICAL_BASICITY_SOURCE]
        if self._excluded_from_basicity:
            notes.append(
                "No optical basicity value is tabulated for " + ", ".join(self._excluded_from_basicity)
                + ": they are left out of Λ.")
        notes.append(GLASSNET_SOURCE)
        return notes

    # ------------------------------------------------------------------
    # GlassNet + export
    # ------------------------------------------------------------------
    def run_glassnet(self) -> None:
        table = self.oxide_tables.get(self.sample_combo.currentText())
        if table is None or table.empty:
            self._say("GlassNet: run the oxide conversion first.")
            return
        row = oxsci.glassnet_input_row(table)
        if row.empty:
            self._say("GlassNet: the sample has no oxide-basis component to predict from.")
            return
        self.gn_btn.setEnabled(False)
        self.gn_btn.setText("Predicting…")
        self._set_busy("Predicting glass properties…")
        run_in_thread(lambda: gscience.glassnet_predict(row), self._on_glassnet_done, self._on_glassnet_error)

    def _on_glassnet_error(self, traceback_text: str) -> None:
        self.gn_btn.setEnabled(True)
        self.gn_btn.setText(GLASSNET_BUTTON)
        self._set_busy(None)
        self._say("GlassNet: the prediction failed.")
        QMessageBox.critical(self, "GlassNet", traceback_text)

    def _on_glassnet_done(self, pred) -> None:
        self.gn_btn.setEnabled(True)
        self.gn_btn.setText(GLASSNET_BUTTON)
        self._set_busy(None)
        self._table_views["GlassNet"].set_dataframe(pred)
        self.tables.setCurrentWidget(self._table_views["GlassNet"])
        self._say("GlassNet: properties predicted for the sample on screen.")

    def export_tables(self) -> None:
        if not self.oxide_tables or self.dataset is None:
            self._say("Oxide chemistry: run first; there is nothing to export yet.")
            return
        tables = dict(self.oxide_tables)
        tables["oxide_map_used"] = pd.DataFrame(
            [{"Element": k, "Assumed_oxide": v or ""} for k, v in self._current_oxide_map().items()]
        )
        tables["role_map_used"] = pd.DataFrame(
            [{"Oxide": k, "Role": v} for k, v in self._current_role_map().items()]
        )
        envelope_view = self._table_views["Envelope check"].dataframe()
        if envelope_view is not None and not envelope_view.empty:
            tables["envelope_check"] = envelope_view
        out_dir = export_utils.export_named_tables(self.dataset, "oxide_chemistry", tables)
        try:
            self.plot.figure.savefig(out_dir / "oxide_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Oxide chemistry exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Oxide chemistry tables exported:\n{out_dir}")
