"""
qt_correlations.py — Correlations workspace: a nav-shell QTabWidget hosting
four sub-tabs (mirrors Dataapp's precedent of one nav row + many internal
tabs for a complex domain): "Cross-correlation" (engine-backed, the only
one that honors the global scope bar -- see qt_correlations_association),
"Quick Scan" (ported from the old app's CorrelationTab), "Association
Workbench (kg)", and "Structure" (PCA/clustering/partial correlation/
network graph).

The first tab is built with the page. The other three are built the first
time they are opened (or first asked for through their attribute), so a
session that never leaves Cross-correlation never pays for them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

import correlation_science as csci
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
    set_completions,
)
from qt_correlations_association import TAB_TITLE as ASSOCIATION_TAB_TITLE, AssociationTab
from qt_correlations_structure import StructureTab
from qt_correlations_workbench import (
    METHOD_LABELS, METHOD_TOOLTIP, METRIC_LABELS, METRIC_TOOLTIP, ResultSplitter, WorkbenchTab, _and, choice,
    fill_choices, reporting_notes, say, select_row_for_mark, set_list_completions,
    unrecognized_entries,
)
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

METRICS = ["log10_plus1", "log10_inventory", "fraction", "inventory", "presence"]
METHODS = ["pearson", "spearman"]
HEATMAP_STYLES = ["Matplotlib lower triangle", "Seaborn lower triangle", "Seaborn + total projections"]

QUICK_SCAN_TITLE = "Quick Scan"
WORKBENCH_TITLE = "Association Workbench (kg)"
STRUCTURE_TITLE = "Structure"

# The three questions the Quick scan answers, in the order of the selector.
ANALYSIS_SCAN = "scan"
ANALYSIS_COMPARE = "compare"
ANALYSIS_HEATMAP = "heatmap"
ANALYSIS_LABELS = {
    ANALYSIS_SCAN: "One element against all others",
    ANALYSIS_COMPARE: "Two or three elements",
    ANALYSIS_HEATMAP: "Every pair (heat map)",
}
# Major constituents with a mass inventory in every farm, so the page
# answers on the basis it opens with (kg). Cesium is reported as activity:
# it is scanned on the Ci basis.
DEFAULT_TARGET = "Na"
DEFAULT_COMPARE = "Na, Al, Fe"
PAGE_TITLE = "Quick scan"
PAGE_SUBTITLE = (
    "Element-to-element correlation of tank inventories, on one reported unit at a time: one "
    "element against all others, two or three elements side by side, or every pair at once."
)
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "Choose what to correlate and press Run."
T_SCAN = "Target scan"
T_PAIRS = "Pairs"
T_JOINT = "Joint"
T_CORRELATION = "Correlation matrix"
T_MATRIX = "Matrix"
_TABLES = [
    (T_SCAN, 500, "One row per partner element, strongest correlation first."),
    (T_PAIRS, 200, "One row per pair of the compared elements."),
    (T_JOINT, 50, "How many tanks hold all of the compared elements."),
    (T_CORRELATION, 250, "The element-by-element correlation behind the heat map."),
    (T_MATRIX, 250, "One row per tank, one column per element: the values that were correlated."),
]


class QuickScanTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._scan_df = pd.DataFrame()
        self._selected_df = pd.DataFrame()
        self._joint_df = pd.DataFrame()
        self._matrix_df = pd.DataFrame()
        self._heatmap_corr_df = pd.DataFrame()
        self._heatmap_projection_df = pd.DataFrame()
        # What the chart on screen was drawn from: the analysis and the
        # settings of that run, never the live controls.
        self._result: Optional[dict] = None
        self._result_inputs: Optional[tuple] = None
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
            "Export...", "Write the tables of every analysis run so far and the chart to the output folder.")
        self.export_btn.clicked.connect(self._export_tables)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 10, "Correlations"))
        root.addWidget(self.header)

        # Control row: which question, about which element(s), on which
        # unit -- then Run. Every tuning knob lives in Advanced options.
        controls, row = control_row()
        row.addWidget(field_label("Correlate"))
        self.analysis_combo = QComboBox()
        for key, label in ANALYSIS_LABELS.items():
            self.analysis_combo.addItem(label, key)
        self.analysis_combo.setToolTip(
            "One element against all others: ranks every other element by its correlation with the target.\n"
            "Two or three elements: the pairwise correlations, and a scatter of the first two.\n"
            "Every pair: the correlation matrix of the most abundant elements, as a heat map.")
        self.analysis_combo.currentIndexChanged.connect(self._on_analysis_changed)
        row.addWidget(self.analysis_combo)

        self.target_label = field_label("Target")
        row.addWidget(self.target_label)
        self.target_edit = QLineEdit(DEFAULT_TARGET)
        self.target_edit.setMaximumWidth(96)
        self.target_edit.setPlaceholderText("Symbol, e.g. Na")
        self.target_edit.setToolTip("Element symbol to correlate with every other element, for example Na or Cr.")
        self.target_edit.textChanged.connect(self._sync_run_label)
        self.target_edit.returnPressed.connect(self.run)
        row.addWidget(self.target_edit)

        self.elements_label = field_label("Elements")
        row.addWidget(self.elements_label)
        self.elements_edit = QLineEdit(DEFAULT_COMPARE)
        self.elements_edit.setMinimumWidth(140)
        self.elements_edit.setMaximumWidth(220)
        self.elements_edit.setPlaceholderText("e.g. Na, Al, Fe")
        self.elements_edit.setToolTip(
            "Two or three element symbols separated by commas. The first two are the axes of the scatter;\n"
            "a third one colors the points.")
        self.elements_edit.textChanged.connect(self._sync_run_label)
        self.elements_edit.returnPressed.connect(self.run)
        row.addWidget(self.elements_edit)

        row.addSpacing(8)
        row.addWidget(field_label("Basis"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kg", "Ci"])
        self.unit_combo.setToolTip(
            "The reported unit the inventories are read in: kg for mass, Ci for activity.\n"
            "An element reported only as activity (cesium, technetium, iodine) has no kg inventory.")
        self.unit_combo.currentTextChanged.connect(self._sync_run_label)
        row.addWidget(self.unit_combo)

        row.addSpacing(8)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        row.addWidget(self.run_btn)
        row.addStretch(1)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        adv_row1 = QHBoxLayout()
        adv_row1.setSpacing(8)
        adv_row1.addWidget(field_label("Metric"))
        self.metric_combo = QComboBox()
        fill_choices(self.metric_combo, METRICS, METRIC_LABELS)
        self.metric_combo.setToolTip(METRIC_TOOLTIP)
        self.metric_combo.currentIndexChanged.connect(self._sync_run_label)
        adv_row1.addWidget(self.metric_combo)
        adv_row1.addWidget(field_label("Method"))
        self.method_combo = QComboBox()
        fill_choices(self.method_combo, METHODS, METHOD_LABELS)
        self.method_combo.setToolTip(METHOD_TOOLTIP.rsplit("\n", 1)[0])
        self.method_combo.currentIndexChanged.connect(self._sync_run_label)
        adv_row1.addWidget(self.method_combo)
        self.include_zeros_check = QCheckBox("Include tanks with none")
        self.include_zeros_check.setChecked(True)
        self.include_zeros_check.setToolTip(
            "On: a tank that holds none of an element counts as a zero.\n"
            "Off: a pair is correlated only over the tanks that hold both elements.")
        self.include_zeros_check.toggled.connect(self._sync_run_label)
        adv_row1.addWidget(self.include_zeros_check)
        self.control_for_size_check = QCheckBox("Control for tank size (partial r)")
        self.control_for_size_check.setToolTip(
            "Two elements correlate when both simply scale with how much waste a tank holds.\n"
            "This removes that effect: the correlation is computed with each tank's total inventory held constant.")
        self.control_for_size_check.toggled.connect(self._sync_run_label)
        adv_row1.addWidget(self.control_for_size_check)
        adv_row1.addStretch(1)
        adv.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        adv_row2.setSpacing(8)
        adv_row2.addWidget(field_label("Minimum shared tanks"))
        self.min_overlap_spin = QSpinBox()
        self.min_overlap_spin.setRange(0, 1000)
        self.min_overlap_spin.setValue(5)
        self.min_overlap_spin.setToolTip("A partner is ranked only when at least this many tanks hold both elements.")
        self.min_overlap_spin.valueChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.min_overlap_spin)
        adv_row2.addWidget(field_label("Minimum inventory per record"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        self.min_inv_spin.setToolTip("Inventory records at or below this value, in the chosen unit, are ignored.")
        self.min_inv_spin.valueChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.min_inv_spin)
        adv_row2.addWidget(field_label("Elements scanned"))
        self.scan_top_spin = QSpinBox()
        self.scan_top_spin.setRange(2, 500)
        self.scan_top_spin.setValue(90)
        self.scan_top_spin.setToolTip(
            "How many of the most abundant elements take part in the scan and in the heat map.")
        self.scan_top_spin.valueChanged.connect(self._sync_run_label)
        adv_row2.addWidget(self.scan_top_spin)
        adv_row2.addWidget(field_label("Partners drawn"))
        self.plot_top_n_spin = QSpinBox()
        self.plot_top_n_spin.setRange(5, 200)
        self.plot_top_n_spin.setValue(30)
        self.plot_top_n_spin.setToolTip("How many of the strongest partners the scan chart shows; the table keeps them all.")
        self.plot_top_n_spin.valueChanged.connect(self._redraw_soon)
        adv_row2.addWidget(self.plot_top_n_spin)
        adv_row2.addStretch(1)
        adv.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.setSpacing(8)
        adv_row3.addWidget(field_label("Heat map style"))
        self.heatmap_style_combo = QComboBox()
        self.heatmap_style_combo.addItems(HEATMAP_STYLES)
        self.heatmap_style_combo.setCurrentText("Seaborn + total projections")
        self.heatmap_style_combo.setToolTip(
            "The heat map shows the lower triangle only. With total projections, bars along two edges give\n"
            "log10(total inventory + 1) of each element in the chosen unit.")
        self.heatmap_style_combo.currentTextChanged.connect(self._redraw)
        adv_row3.addWidget(self.heatmap_style_combo)
        self.annotate_check = QCheckBox("Annotate r values")
        self.annotate_check.setToolTip("Write the coefficient in every cell of the heat map.")
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
        for title, max_rows, hint in _TABLES:
            view = DataFrameTableView(title=title, max_rows_display=max_rows)
            view.set_empty_text(EMPTY_TITLE, hint)
            self._table_views[title] = view
            self.tables.addTab(view, title)
        splitter.addWidget(self.tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

        self._on_analysis_changed()
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)

    def _analysis(self) -> str:
        return str(self.analysis_combo.currentData())

    def _set_analysis(self, key: str) -> None:
        index = self.analysis_combo.findData(key)
        if index >= 0 and index != self.analysis_combo.currentIndex():
            self.analysis_combo.setCurrentIndex(index)

    def _on_analysis_changed(self, *_args) -> None:
        analysis = self._analysis()
        for widget in (self.target_label, self.target_edit):
            widget.setVisible(analysis == ANALYSIS_SCAN)
        for widget in (self.elements_label, self.elements_edit):
            widget.setVisible(analysis == ANALYSIS_COMPARE)
        self._sync_run_label()

    def _say(self, message: str) -> None:
        say(self.app_window, message)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        units = dataset.available_units() if dataset.is_loaded() else []
        current = self.unit_combo.currentText()
        self.unit_combo.blockSignals(True)
        self.unit_combo.clear()
        self.unit_combo.addItems(units or ["kg", "Ci"])
        if "kg" in units:
            self.unit_combo.setCurrentText("kg")
        elif current:
            idx = self.unit_combo.findText(current)
            if idx >= 0:
                self.unit_combo.setCurrentIndex(idx)
        self.unit_combo.blockSignals(False)
        # Tables of the previous dataset must not stay on screen, nor be
        # exported into the new dataset's folder.
        had_result = self._result is not None
        self._scan_df = pd.DataFrame()
        self._selected_df = pd.DataFrame()
        self._joint_df = pd.DataFrame()
        self._matrix_df = pd.DataFrame()
        self._heatmap_corr_df = pd.DataFrame()
        self._heatmap_projection_df = pd.DataFrame()
        self._result = None
        self._result_inputs = None
        if had_result:
            for view in self._table_views.values():
                view.set_dataframe(pd.DataFrame())
            self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self.notes_banner.set_notes([])
        symbols = dataset_values(dataset, "Element")
        set_completions(self.target_edit, symbols)
        set_list_completions(self.elements_edit, symbols)
        self._sync_run_label()

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def _read_inputs(self) -> tuple:
        analysis = self._analysis()
        subject = {ANALYSIS_SCAN: self.target_edit.text().strip(),
                   ANALYSIS_COMPARE: self.elements_edit.text().strip()}.get(analysis, "")
        return (
            analysis, subject, self.unit_combo.currentText(), choice(self.metric_combo), choice(self.method_combo),
            self.include_zeros_check.isChecked(), self.control_for_size_check.isChecked(),
            self.min_overlap_spin.value(), self.min_inv_spin.value(), self.scan_top_spin.value(),
        )

    def _sync_run_label(self, *_args) -> None:
        """"Run" answers the question; "Update" re-answers it after a control moved under a standing result."""
        if not hasattr(self, "notes_banner"):
            return
        changed = self._result is not None and self._result_inputs != self._read_inputs()
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else "Compute the correlation the controls describe.")

    def run(self) -> None:
        """The primary button: run the analysis the selector names."""
        {ANALYSIS_SCAN: self.run_target_scan, ANALYSIS_COMPARE: self.run_selected,
         ANALYSIS_HEATMAP: self.run_heatmap}[self._analysis()]()

    def _ready(self) -> bool:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Quick scan: no dataset is loaded yet.")
            return False
        return True

    def _busy(self, busy: bool) -> None:
        """The scan and the heat map are computed and drawn on the UI
        thread (well under a second of computation; the drawing cannot
        leave it): the pointer says so while it lasts."""
        self.run_btn.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _finish(self, analysis: str, notes: List[str], table: str, **drawn) -> None:
        self._set_analysis(analysis)
        self._result = {"analysis": analysis, **drawn}
        self._result_inputs = self._read_inputs()
        self.notes_banner.set_result_notes(notes)
        self.tables.setCurrentWidget(self._table_views[table])
        self._draw()
        self._sync_run_label()

    def run_target_scan(self) -> None:
        if not self._ready():
            return
        target = self.target_edit.text().strip()
        if not target:
            self._say("Quick scan: enter a target element first.")
            self.target_edit.setFocus()
            return
        unit = self.unit_combo.currentText()
        self._busy(True)
        try:
            scan_df, matrix_df = csci.element_correlation_scan(
                self.dataset, target, unit=unit,
                value_mode=choice(self.metric_combo), method=choice(self.method_combo),
                top_n_elements=self.scan_top_spin.value(), min_overlap=self.min_overlap_spin.value(),
                min_inventory=self.min_inv_spin.value(), include_zeros=self.include_zeros_check.isChecked(),
                control_for_total_inventory=self.control_for_size_check.isChecked(),
            )
            symbol = (csci.parse_element_list(target) or [target])[0]
            notes = reporting_notes(self.dataset, [symbol], unit,
                                    consequence="Set Basis to {units} to scan it.")
        except ValueError as exc:
            self._say(f"Quick scan not run: {exc}")
            QMessageBox.warning(self, "Quick scan", str(exc))
            return
        finally:
            self._busy(False)
        self._scan_df, self._matrix_df = scan_df, matrix_df
        self._table_views[T_SCAN].set_dataframe(self._scan_df)
        self._table_views[T_MATRIX].set_dataframe(self._matrix_df)
        self._finish(ANALYSIS_SCAN, notes, T_SCAN, target=symbol, unit=unit)
        if self._scan_df.empty:
            self._say(f"Quick scan: no partner to rank for {symbol} in {unit}.")
        else:
            self._say(f"Quick scan: {len(self._scan_df):,} partner element(s) ranked for {symbol}, {unit}.")

    def run_selected(self) -> None:
        if not self._ready():
            return
        text = self.elements_edit.text()
        elements = csci.parse_element_list(text)
        unit = self.unit_combo.currentText()
        metric = choice(self.metric_combo)
        self._busy(True)
        try:
            selected_df, joint_df, matrix_df = csci.selected_element_correlations(
                self.dataset, elements, unit=unit,
                value_mode=metric, method=choice(self.method_combo),
                min_inventory=self.min_inv_spin.value(), include_zeros=self.include_zeros_check.isChecked(),
                control_for_total_inventory=self.control_for_size_check.isChecked(),
            )
            notes = []
            unknown = unrecognized_entries(text)
            if unknown:
                verb = "is not an element symbol and was" if len(unknown) == 1 else "are not element symbols and were"
                notes.append(f"{_and(unknown)} {verb} left out.")
            notes += reporting_notes(self.dataset, elements, unit,
                                     consequence="Set Basis to {units} to compare on that unit.")
        except ValueError as exc:
            self._say(f"Quick scan not run: {exc}")
            QMessageBox.warning(self, "Quick scan", str(exc))
            return
        finally:
            self._busy(False)
        self._selected_df, self._joint_df, self._matrix_df = selected_df, joint_df, matrix_df
        self._table_views[T_PAIRS].set_dataframe(self._selected_df)
        self._table_views[T_JOINT].set_dataframe(self._joint_df)
        self._table_views[T_MATRIX].set_dataframe(self._matrix_df)
        self._finish(ANALYSIS_COMPARE, notes, T_PAIRS, elements=list(elements), unit=unit, metric=metric)
        self._say(f"Quick scan: {', '.join(elements)} compared, {unit}.")

    def run_heatmap(self) -> None:
        if not self._ready():
            return
        unit = self.unit_combo.currentText()
        metric = choice(self.metric_combo)
        self._busy(True)
        try:
            self._heatmap_corr_df, self._matrix_df = csci.full_correlation_matrix(
                self.dataset, unit=unit, top_n_elements=self.scan_top_spin.value(),
                value_mode=metric, method=choice(self.method_combo),
                min_inventory=self.min_inv_spin.value(),
            )
            totals = csci.element_totals_by_unit(self.dataset, unit)
            heatmap_elements = self._heatmap_corr_df["Element"].astype(str).tolist() if "Element" in self._heatmap_corr_df else []
            self._heatmap_projection_df = pd.DataFrame({
                "Element": heatmap_elements,
                f"Total_inventory_{unit}": [float(totals.get(e, 0.0) or 0.0) for e in heatmap_elements],
            })
            if not self._heatmap_projection_df.empty:
                import numpy as np
                self._heatmap_projection_df[f"log10_total_inventory_{unit}_plus1"] = np.log10(
                    self._heatmap_projection_df[f"Total_inventory_{unit}"] + 1.0
                )
            self._table_views[T_MATRIX].set_dataframe(self._matrix_df)
            self._table_views[T_CORRELATION].set_dataframe(self._heatmap_corr_df)
            self._finish(ANALYSIS_HEATMAP, [], T_CORRELATION, unit=unit, metric=metric, totals=totals)
        finally:
            self._busy(False)
        self._say(f"Quick scan: correlation of every pair among {len(self._heatmap_corr_df):,} elements, {unit}.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _redraw(self, *_args) -> None:
        """A display choice changed: redraw the standing result, compute nothing."""
        if self._result is not None:
            self._draw()

    def _redraw_soon(self, *_args) -> None:
        """The same for a spin box, which fires once per step: one redraw after the last step."""
        if self._result is not None:
            self.plot.request_redraw(self._redraw)

    def _draw(self) -> None:
        result = self._result
        if result is None:
            return
        analysis = result["analysis"]
        if analysis == ANALYSIS_SCAN:
            if self._scan_df.empty:
                notes = self.notes_banner.notes()
                self.plot.show_empty(
                    f"Nothing to rank for {result['target']} in {result['unit']}",
                    notes[0] if notes else "No other element shares enough tanks with it on these settings.")
                return
            ph.plot_correlation_scan(self.plot, self._scan_df, result["target"], top_n=self.plot_top_n_spin.value())
        elif analysis == ANALYSIS_COMPARE:
            ph.plot_pair_scatter(self.plot, self._matrix_df, result["elements"], result["unit"], result["metric"])
        else:
            ph.plot_correlation_heatmap(
                self.plot, self._heatmap_corr_df,
                title=f"Element correlation heatmap ({result['metric']}, {result['unit']})",
                style=self.heatmap_style_combo.currentText(), totals=result["totals"], unit=result["unit"],
                annotate=self.annotate_check.isChecked(),
            )

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a bar, a point or a cell selects the row it was drawn from."""
        if self._result is None:
            return
        analysis = self._result["analysis"]
        if analysis == ANALYSIS_SCAN:
            views = [self._table_views[T_SCAN]]
        elif analysis == ANALYSIS_COMPARE:
            views = [self._table_views[T_MATRIX]]
        else:
            views = [self._table_views[T_CORRELATION]]
            if mark.get("row") not in (None, ""):
                mark = {"label": mark["row"]}       # a cell belongs to the row of its row element
        select_row_for_mark(self.tables, views, mark)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_tables(self) -> None:
        tables = {
            "target_correlation_scan": self._scan_df, "selected_pair_correlations": self._selected_df,
            "joint_summary": self._joint_df, "matrix_used": self._matrix_df,
            "correlation_heatmap_matrix": self._heatmap_corr_df,
            "correlation_heatmap_total_projections": self._heatmap_projection_df,
        }
        if all(df.empty for df in tables.values()) or self.dataset is None:
            self._say("Quick scan: run first; there is nothing to export yet.")
            return
        out_dir = export_utils.export_named_tables(self.dataset, "correlations", tables)
        try:
            self.plot.figure.savefig(out_dir / "correlation_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        self._say(f"Quick scan exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Correlation outputs exported:\n{out_dir}")


class CorrelationsPage(QWidget):
    # The visible sub-tab changed, and with it whether the global scope
    # applies to what is on screen (see scope_applies).
    scope_relevance_changed = Signal()

    # Sub-tabs built on first use: attribute -> (tab title, constructor).
    _DEFERRED: Dict[str, tuple] = {
        "quick_scan_tab": (QUICK_SCAN_TITLE, QuickScanTab),
        "workbench_tab": (WORKBENCH_TITLE, WorkbenchTab),
        "structure_tab": (STRUCTURE_TITLE, StructureTab),
    }

    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self._dataset: Optional[HanfordDataset] = None
        self._built: Dict[str, QWidget] = {}
        self._placeholders: Dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # First: the engine-backed tab is the one to start from. It is
        # also the only sub-tab that receives the shared global scope;
        # the three legacy tabs below never see it.
        self.association_tab = AssociationTab(app_window, scope=scope)
        self.tabs.addTab(self.association_tab, ASSOCIATION_TAB_TITLE)
        # The legacy tabs stand as empty pages until they are first opened
        # (or first asked for through quick_scan_tab / workbench_tab /
        # structure_tab): building all four with the page made this the
        # slowest workspace to construct.
        for attribute, (title, _factory) in self._DEFERRED.items():
            placeholder = QWidget()
            self._placeholders[attribute] = placeholder
            self.tabs.addTab(placeholder, title)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # -- deferred sub-tabs ------------------------------------------------
    @property
    def quick_scan_tab(self) -> QuickScanTab:
        return self._sub_tab("quick_scan_tab")

    @property
    def workbench_tab(self) -> WorkbenchTab:
        return self._sub_tab("workbench_tab")

    @property
    def structure_tab(self) -> StructureTab:
        return self._sub_tab("structure_tab")

    def is_built(self, attribute: str) -> bool:
        """Whether a deferred sub-tab ("quick_scan_tab", ...) exists yet."""
        return attribute in self._built

    def _sub_tab(self, attribute: str) -> QWidget:
        tab = self._built.get(attribute)
        if tab is not None:
            return tab
        title, factory = self._DEFERRED[attribute]
        tab = factory(self.app_window)
        self._built[attribute] = tab
        if self._dataset is not None:
            tab.on_dataset_changed(self._dataset)
        placeholder = self._placeholders.pop(attribute)
        index = self.tabs.indexOf(placeholder)
        current = self.tabs.currentIndex()
        # The swap is silent: the tab order and the current tab are the
        # same before and after, so nobody needs to hear about it.
        self.tabs.blockSignals(True)
        self.tabs.setUpdatesEnabled(False)
        try:
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, tab, title)
            self.tabs.setCurrentIndex(current)
        finally:
            self.tabs.setUpdatesEnabled(True)
            self.tabs.blockSignals(False)
        placeholder.deleteLater()
        return tab

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        for attribute, placeholder in list(self._placeholders.items()):
            if placeholder is widget:
                self._sub_tab(attribute)
                break
        self.scope_relevance_changed.emit()

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        # A tab that is not built yet receives the dataset when it is.
        self._dataset = dataset
        self.association_tab.on_dataset_changed(dataset)
        for tab in self._built.values():
            tab.on_dataset_changed(dataset)

    def on_scope_changed(self) -> None:
        """Forward a global scope change to the sub-tabs that honor it
        -- derived from the same protocol the shell dispatches on, so a
        legacy tab joins automatically when it migrates onto the
        engine. Today that is the Cross-correlation tab alone: Quick
        Scan, the Association Workbench and Structure compute on the
        whole dataset and are not told about the scope at all."""
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if hasattr(tab, "on_scope_changed"):
                tab.on_scope_changed()

    def scope_applies(self) -> bool:
        """Whether the sub-tab on screen honors the global scope. The
        shell shows the scope bar only then: a filter displayed above
        a tab that ignores it would misdescribe what is on screen."""
        return hasattr(self.tabs.currentWidget(), "on_scope_changed")
