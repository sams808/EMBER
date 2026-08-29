"""
qt_correlations.py — Correlations workspace: a nav-shell QTabWidget hosting
three sub-tabs (mirrors Dataapp's precedent of one nav row + many internal
tabs for a complex domain): "Quick Scan" (ported from the old app's
CorrelationTab), "Association Workbench (kg)", and "Structure" (PCA/
clustering/partial correlation/network graph -- entirely new capability).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

import correlation_science as csci
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_correlations_structure import StructureTab
from qt_correlations_workbench import WorkbenchTab
from qt_help import show_workspace_help
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

METRICS = ["log10_plus1", "log10_inventory", "fraction", "inventory", "presence"]
METHODS = ["pearson", "spearman"]
HEATMAP_STYLES = ["Matplotlib lower triangle", "Seaborn lower triangle", "Seaborn + total projections"]


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
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        header.addWidget(QLabel("Unit"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kg", "Ci"])
        self.unit_combo.setToolTip("Inventory unit used by every scan, comparison, and heatmap on this tab.")
        header.addWidget(self.unit_combo)
        header.addStretch(1)
        help_btn = QPushButton("?")
        help_btn.setObjectName("HelpButton")
        help_btn.setToolTip("What is this workspace for?")
        help_btn.clicked.connect(lambda: show_workspace_help(self, 10, "Correlations"))
        header.addWidget(help_btn)
        root.addLayout(header)

        # Primary row: the three actions this tab exists to run, and nothing
        # else -- every tuning knob lives in "Advanced options" below instead
        # of competing for attention up here.
        primary = QHBoxLayout()
        primary.addWidget(QLabel("Target element"))
        self.target_edit = QLineEdit("Cs")
        self.target_edit.setMaximumWidth(80)
        self.target_edit.setToolTip("Element to scan against every other element for correlation.")
        primary.addWidget(self.target_edit)
        self.scan_btn = QPushButton("Scan target correlations")
        self.scan_btn.setObjectName("Primary")
        self.scan_btn.clicked.connect(self.run_target_scan)
        primary.addWidget(self.scan_btn)

        primary.addSpacing(20)
        primary.addWidget(QLabel("Compare elements"))
        self.elements_edit = QLineEdit("Cs, Sr, Tc")
        self.elements_edit.setMaximumWidth(160)
        self.elements_edit.setToolTip("Comma-separated list of 2-3 elements to correlate pairwise.")
        primary.addWidget(self.elements_edit)
        self.selected_btn = QPushButton("Run dual/triple")
        self.selected_btn.clicked.connect(self.run_selected)
        primary.addWidget(self.selected_btn)

        primary.addSpacing(20)
        self.heatmap_btn = QPushButton("Correlation heatmap")
        self.heatmap_btn.clicked.connect(self.run_heatmap)
        primary.addWidget(self.heatmap_btn)

        primary.addStretch(1)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_tables)
        primary.addWidget(export_btn)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        adv_row1 = QHBoxLayout()
        adv_row1.addWidget(QLabel("Metric"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(METRICS)
        adv_row1.addWidget(self.metric_combo)
        adv_row1.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        adv_row1.addWidget(self.method_combo)
        self.include_zeros_check = QCheckBox("include zeros")
        self.include_zeros_check.setChecked(True)
        adv_row1.addWidget(self.include_zeros_check)
        self.control_for_size_check = QCheckBox("control for tank size (partial r)")
        self.control_for_size_check.setToolTip(
            "Strip out the \"everything correlates because both elements scale with tank size\" "
            "effect by computing partial correlation controlling for each tank's total inventory."
        )
        adv_row1.addWidget(self.control_for_size_check)
        adv_row1.addStretch(1)
        adv.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        adv_row2.addWidget(QLabel("min overlap"))
        self.min_overlap_spin = QSpinBox()
        self.min_overlap_spin.setRange(0, 1000)
        self.min_overlap_spin.setValue(5)
        self.min_overlap_spin.setToolTip("Minimum number of tanks both elements must both report data in.")
        adv_row2.addWidget(self.min_overlap_spin)
        adv_row2.addWidget(QLabel("min inventory"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        self.min_inv_spin.setToolTip("Drop tanks below this total inventory before correlating.")
        adv_row2.addWidget(self.min_inv_spin)
        adv_row2.addWidget(QLabel("scan top elements"))
        self.scan_top_spin = QSpinBox()
        self.scan_top_spin.setRange(2, 500)
        self.scan_top_spin.setValue(90)
        self.scan_top_spin.setToolTip("How many of the most abundant elements to include in the scan/heatmap.")
        adv_row2.addWidget(self.scan_top_spin)
        adv_row2.addWidget(QLabel("plot top N"))
        self.plot_top_n_spin = QSpinBox()
        self.plot_top_n_spin.setRange(5, 200)
        self.plot_top_n_spin.setValue(30)
        self.plot_top_n_spin.setToolTip("How many top partners to show on the target-scan plot.")
        adv_row2.addWidget(self.plot_top_n_spin)
        adv_row2.addStretch(1)
        adv.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.addWidget(QLabel("Heatmap style"))
        self.heatmap_style_combo = QComboBox()
        self.heatmap_style_combo.addItems(HEATMAP_STYLES)
        self.heatmap_style_combo.setCurrentText("Seaborn + total projections")
        adv_row3.addWidget(self.heatmap_style_combo)
        self.annotate_check = QCheckBox("annotate r values")
        adv_row3.addWidget(self.annotate_check)
        note = QLabel("Lower triangle only; projection bars are log10(total inventory + 1) in the selected unit.")
        note.setObjectName("SectionNote")
        adv_row3.addWidget(note)
        adv_row3.addStretch(1)
        adv.addLayout(adv_row3)

        root.addWidget(self.advanced_section)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self._table_views: dict[str, DataFrameTableView] = {}
        for title, max_rows in [
            ("Target scan", 500), ("Pairs", 200), ("Joint", 50), ("Matrix", 250),
        ]:
            view = DataFrameTableView(title=title, max_rows_display=max_rows)
            self._table_views[title] = view
            self.tables.addTab(view, title)
        splitter.addWidget(self.tables)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

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

    def run_target_scan(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Correlations", "Load a dataset first.")
            return
        target = self.target_edit.text().strip()
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning…")
        QApplication.processEvents()
        try:
            self._scan_df, self._matrix_df = csci.element_correlation_scan(
                self.dataset, target, unit=self.unit_combo.currentText(),
                value_mode=self.metric_combo.currentText(), method=self.method_combo.currentText(),
                top_n_elements=self.scan_top_spin.value(), min_overlap=self.min_overlap_spin.value(),
                min_inventory=self.min_inv_spin.value(), include_zeros=self.include_zeros_check.isChecked(),
                control_for_total_inventory=self.control_for_size_check.isChecked(),
            )
        except ValueError as exc:
            self.scan_btn.setEnabled(True)
            self.scan_btn.setText("Scan target correlations")
            QMessageBox.warning(self, "Correlation scan failed", str(exc))
            return
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan target correlations")
        self._table_views["Target scan"].set_dataframe(self._scan_df)
        self._table_views["Matrix"].set_dataframe(self._matrix_df)
        ph.plot_correlation_scan(self.plot, self._scan_df, target, top_n=self.plot_top_n_spin.value())
        self.app_window.statusBar().showMessage(f"Correlation scan completed for {target}: {len(self._scan_df):,} partners.")

    def run_selected(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Correlations", "Load a dataset first.")
            return
        elements = csci.parse_element_list(self.elements_edit.text())
        self.selected_btn.setEnabled(False)
        self.selected_btn.setText("Running…")
        QApplication.processEvents()
        try:
            self._selected_df, self._joint_df, self._matrix_df = csci.selected_element_correlations(
                self.dataset, elements, unit=self.unit_combo.currentText(),
                value_mode=self.metric_combo.currentText(), method=self.method_combo.currentText(),
                min_inventory=self.min_inv_spin.value(), include_zeros=self.include_zeros_check.isChecked(),
                control_for_total_inventory=self.control_for_size_check.isChecked(),
            )
        except ValueError as exc:
            self.selected_btn.setEnabled(True)
            self.selected_btn.setText("Run dual/triple")
            QMessageBox.warning(self, "Dual/triple correlation failed", str(exc))
            return
        self.selected_btn.setEnabled(True)
        self.selected_btn.setText("Run dual/triple")
        self._table_views["Pairs"].set_dataframe(self._selected_df)
        self._table_views["Joint"].set_dataframe(self._joint_df)
        self._table_views["Matrix"].set_dataframe(self._matrix_df)
        ph.plot_pair_scatter(self.plot, self._matrix_df, elements, self.unit_combo.currentText(), self.metric_combo.currentText())
        self.app_window.statusBar().showMessage(f"Dual/triple correlation completed for: {', '.join(elements)}.")

    def run_heatmap(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Correlations", "Load a dataset first.")
            return
        unit = self.unit_combo.currentText()
        self.heatmap_btn.setEnabled(False)
        self.heatmap_btn.setText("Building heatmap…")
        QApplication.processEvents()
        self._heatmap_corr_df, self._matrix_df = csci.full_correlation_matrix(
            self.dataset, unit=unit, top_n_elements=self.scan_top_spin.value(),
            value_mode=self.metric_combo.currentText(), method=self.method_combo.currentText(),
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
        self._table_views["Matrix"].set_dataframe(self._matrix_df)
        self._table_views["Target scan"].set_dataframe(self._heatmap_corr_df)
        ph.plot_correlation_heatmap(
            self.plot, self._heatmap_corr_df, title=f"Element correlation heatmap ({self.metric_combo.currentText()}, {unit})",
            style=self.heatmap_style_combo.currentText(), totals=totals, unit=unit,
            annotate=self.annotate_check.isChecked(),
        )
        self.heatmap_btn.setEnabled(True)
        self.heatmap_btn.setText("Correlation heatmap")
        self.app_window.statusBar().showMessage(
            f"Correlation heatmap built: {len(self._heatmap_corr_df):,} elements, style={self.heatmap_style_combo.currentText()}."
        )

    def _export_tables(self) -> None:
        tables = {
            "target_correlation_scan": self._scan_df, "selected_pair_correlations": self._selected_df,
            "joint_summary": self._joint_df, "matrix_used": self._matrix_df,
            "correlation_heatmap_matrix": self._heatmap_corr_df,
            "correlation_heatmap_total_projections": self._heatmap_projection_df,
        }
        if all(df.empty for df in tables.values()) or self.dataset is None:
            QMessageBox.information(self, "Export", "Run a correlation analysis first.")
            return
        out_dir = export_utils.export_named_tables(self.dataset, "correlations", tables)
        try:
            self.plot.figure.savefig(out_dir / "correlation_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Correlation outputs exported:\n{out_dir}")


class CorrelationsPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.quick_scan_tab = QuickScanTab(app_window)
        self.tabs.addTab(self.quick_scan_tab, "Quick Scan")
        self.workbench_tab = WorkbenchTab(app_window)
        self.tabs.addTab(self.workbench_tab, "Association Workbench (kg)")
        self.structure_tab = StructureTab(app_window)
        self.tabs.addTab(self.structure_tab, "Structure")

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.quick_scan_tab.on_dataset_changed(dataset)
        self.workbench_tab.on_dataset_changed(dataset)
        self.structure_tab.on_dataset_changed(dataset)
