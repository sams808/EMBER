"""
qt_heatmap.py — Heatmaps workspace: tank x element inventory matrices.
Ported from the old app's HeatmapTab.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import matrix_science as msci
import plot_helpers as ph
from data_model import HanfordDataset
from qt_widgets import DataFrameTableView, PlotWidget

MODES = ["log10_inventory", "inventory", "fraction"]


class HeatmapPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._long_df: pd.DataFrame = pd.DataFrame()
        self._wide_df: pd.DataFrame = pd.DataFrame()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Unit"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kg", "Ci"])
        controls.addWidget(self.unit_combo)
        controls.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        controls.addWidget(self.mode_combo)
        controls.addWidget(QLabel("Top elements"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 120)
        self.top_n_spin.setValue(35)
        controls.addWidget(self.top_n_spin)
        controls.addWidget(QLabel("min inventory"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        controls.addWidget(self.min_inv_spin)
        controls.addWidget(QLabel("max tanks (0=all)"))
        self.max_tanks_spin = QSpinBox()
        self.max_tanks_spin.setRange(0, 1000)
        controls.addWidget(self.max_tanks_spin)
        build_btn = QPushButton("Build heatmap")
        build_btn.setObjectName("Primary")
        build_btn.clicked.connect(self.run)
        controls.addWidget(build_btn)
        controls.addStretch(1)
        export_btn = QPushButton("Export inputs")
        export_btn.clicked.connect(self._export_inputs)
        controls.addWidget(export_btn)
        root.addLayout(controls)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self.long_view = DataFrameTableView(title="Heatmap long input")
        self.wide_view = DataFrameTableView(title="Heatmap wide matrix", max_rows_display=250)
        self.tables.addTab(self.long_view, "Long")
        self.tables.addTab(self.wide_view, "Wide")
        splitter.addWidget(self.tables)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        current_unit = self.unit_combo.currentText()
        units = dataset.available_units() if dataset.is_loaded() else []
        self.unit_combo.blockSignals(True)
        self.unit_combo.clear()
        self.unit_combo.addItems(units or ["kg", "Ci"])
        if "kg" in units:
            self.unit_combo.setCurrentText("kg")
        elif current_unit:
            restore = self.unit_combo.findText(current_unit)
            if restore >= 0:
                self.unit_combo.setCurrentIndex(restore)
        self.unit_combo.blockSignals(False)
        self._long_df, self._wide_df = pd.DataFrame(), pd.DataFrame()
        self.long_view.set_dataframe(self._long_df)
        self.wide_view.set_dataframe(self._wide_df)
        self.plot.show_message("Build a heatmap to see results.")

    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Heatmaps", "Load a dataset first.")
            return
        unit = self.unit_combo.currentText()
        top_n = self.top_n_spin.value()
        min_inv = self.min_inv_spin.value()
        mode = self.mode_combo.currentText()
        max_tanks = self.max_tanks_spin.value()

        tank_subset = None
        if max_tanks > 0:
            tank_subset = msci.top_tanks_by_inventory(self.dataset, unit, max_tanks)

        self._long_df, self._wide_df = msci.matrix_long_wide(
            self.dataset, unit=unit, top_n_elements=top_n, min_inventory=min_inv,
            value_mode=mode, tank_subset=tank_subset,
        )
        self.long_view.set_dataframe(self._long_df)
        self.wide_view.set_dataframe(self._wide_df)
        ph.plot_heatmap(self.plot, self._wide_df, unit, mode, f"Tank x element heatmap ({unit}, {mode})")
        n_tanks = len(self._wide_df) if not self._wide_df.empty else 0
        self.app_window.statusBar().showMessage(
            f"Heatmap built: unit={unit}, top_elements={top_n}, tanks={n_tanks}."
        )

    def _export_inputs(self) -> None:
        if self._long_df.empty and self._wide_df.empty:
            QMessageBox.information(self, "Export", "Build a heatmap first.")
            return
        params = pd.DataFrame([
            {"parameter": "unit", "value": self.unit_combo.currentText()},
            {"parameter": "mode", "value": self.mode_combo.currentText()},
            {"parameter": "top_n_elements", "value": self.top_n_spin.value()},
            {"parameter": "min_inventory", "value": self.min_inv_spin.value()},
            {"parameter": "max_tanks", "value": self.max_tanks_spin.value()},
        ])
        tables = {"heatmap_long": self._long_df, "heatmap_wide": self._wide_df, "heatmap_parameters": params}
        out_dir = export_utils.export_named_tables(self.dataset, f"heatmap_{self.unit_combo.currentText()}", tables)
        try:
            self.plot.figure.savefig(out_dir / "heatmap.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Heatmap inputs exported:\n{out_dir}")
