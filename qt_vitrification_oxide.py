"""
qt_vitrification_oxide.py — Vitrification "Oxide Chemistry" sub-tab: the
element -> oxide stoichiometric conversion, NBO/T, optical basicity, and
envelope-comparison workspace. Layout pattern borrowed from Dataapp's
qt_glass.py (left-controls/right-results QSplitter); GlassNet integration
reuses glass_science verbatim, gated the same way Dataapp gates it (needs
glasspy/PyTorch -- available when run from Python, disabled with a tooltip
in the packaged exe).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QListWidget, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import glass_science as gscience
import oxide_science as oxsci
import tank_science as tsci
from data_model import HanfordDataset
from qt_widgets import DataFrameTableView

SOURCES_NOTE = (
    "Screening tool, not an official glass model. Λ: Rodriguez & McCloy, PNNL-20184 (2011) Table B.1. "
    "ML predictions: GlassNet -- Cassar, Ceram. Int. 49 (2023) 36013, trained on SciGlass. "
    "NBO/T is a simplified approximation (does not model Al/B charge-balance by alkali) -- "
    "verify against your preferred reference for publication use."
)


class OxideChemistryTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.oxide_tables: Dict[str, pd.DataFrame] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left.setObjectName("Card")
        left.setMaximumWidth(420)
        ll = QVBoxLayout(left)
        title = QLabel("Oxide Chemistry")
        title.setObjectName("SectionTitle")
        ll.addWidget(title)

        ll.addWidget(QLabel("Farm"))
        self.farm_combo = QComboBox()
        self.farm_combo.addItem("All")
        self.farm_combo.currentTextChanged.connect(lambda _: self._refresh_tank_list())
        ll.addWidget(self.farm_combo)
        ll.addWidget(QLabel("Tanks (2+ also builds a physically-combined blend)"))
        self.tank_list = QListWidget()
        self.tank_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tank_list.setMaximumHeight(140)
        ll.addWidget(self.tank_list)
        build_btn = QPushButton("Build oxide composition")
        build_btn.setObjectName("Primary")
        build_btn.clicked.connect(self.build_composition)
        ll.addWidget(build_btn)

        ll.addWidget(QLabel("Element -> assumed oxide (multivalent elements offer alternatives; any cell is editable)"))
        self.oxide_map_table = QTableWidget(0, 2)
        self.oxide_map_table.setHorizontalHeaderLabels(["Element", "Assumed oxide"])
        self.oxide_map_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.oxide_map_table.setMaximumHeight(170)
        ll.addWidget(self.oxide_map_table)

        ll.addWidget(QLabel("Oxide role (for NBO/T)"))
        self.role_table = QTableWidget(0, 2)
        self.role_table.setHorizontalHeaderLabels(["Oxide", "Role"])
        self.role_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.role_table.setMaximumHeight(140)
        ll.addWidget(self.role_table)

        ll.addWidget(QLabel("Envelope (user-supplied; empty by default)"))
        env_btn_row = QHBoxLayout()
        add_row_btn = QPushButton("Add row")
        add_row_btn.clicked.connect(self._add_envelope_row)
        env_btn_row.addWidget(add_row_btn)
        remove_row_btn = QPushButton("Remove row")
        remove_row_btn.clicked.connect(self._remove_envelope_row)
        env_btn_row.addWidget(remove_row_btn)
        load_env_btn = QPushButton("Load JSON…")
        load_env_btn.clicked.connect(self.load_envelope)
        env_btn_row.addWidget(load_env_btn)
        save_env_btn = QPushButton("Save JSON…")
        save_env_btn.clicked.connect(self.save_envelope)
        env_btn_row.addWidget(save_env_btn)
        ll.addLayout(env_btn_row)
        self.envelope_table = QTableWidget(0, 3)
        self.envelope_table.setHorizontalHeaderLabels(["Oxide", "Min wt%", "Max wt%"])
        self.envelope_table.setMaximumHeight(140)
        ll.addWidget(self.envelope_table)

        recompute_btn = QPushButton("Recompute basicity / NBO-T / envelope")
        recompute_btn.clicked.connect(self.recompute_summary)
        ll.addWidget(recompute_btn)

        self.gn_btn = QPushButton("GlassNet predict (selected sample)")
        self.gn_btn.clicked.connect(self.run_glassnet)
        ll.addWidget(self.gn_btn)
        if not gscience.glassnet_available():
            self.gn_btn.setEnabled(False)
            self.gn_btn.setToolTip(
                "pip install glasspy (needs PyTorch) to enable GlassNet predictions. "
                "Not available in the packaged exe."
            )

        export_btn = QPushButton("Export all tables")
        export_btn.clicked.connect(self.export_tables)
        ll.addWidget(export_btn)

        src = QLabel(SOURCES_NOTE)
        src.setWordWrap(True)
        src.setObjectName("SectionNote")
        ll.addWidget(src)
        ll.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Sample"))
        self.sample_combo = QComboBox()
        self.sample_combo.currentTextChanged.connect(self._on_sample_changed)
        rl.addWidget(self.sample_combo)
        self.tables = QTabWidget()
        self._table_views: Dict[str, DataFrameTableView] = {}
        for title_ in ["Oxide composition", "Envelope check", "GlassNet"]:
            view = DataFrameTableView(title=title_, max_rows_display=500)
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        rl.addWidget(self.tables, 1)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("SectionNote")
        rl.addWidget(self.summary_label)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
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

    def _refresh_tank_list(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            return
        farm = self.farm_combo.currentText()
        tanks = tsci.tanks_in_farm(self.dataset, farm)
        self.tank_list.clear()
        self.tank_list.addItems(tanks)

    def selected_tanks(self) -> List[str]:
        return [item.text() for item in self.tank_list.selectedItems()]

    # ------------------------------------------------------------------
    # Element -> oxide table
    # ------------------------------------------------------------------
    def _elements_in_tanks(self, tank_ids: List[str]) -> List[str]:
        df = self.dataset.require_df().filter(
            (pl.col("Units") == "kg") & pl.col("Element").is_not_null() & pl.col("WasteSiteId").is_in(tank_ids)
        )
        return [str(e) for e in df.get_column("Element").drop_nulls().unique().to_list()]

    def _populate_oxide_map_table(self, elements: List[str]) -> None:
        self.oxide_map_table.setRowCount(0)
        for element in sorted(elements):
            row = self.oxide_map_table.rowCount()
            self.oxide_map_table.insertRow(row)
            self.oxide_map_table.setItem(row, 0, QTableWidgetItem(element))
            alternatives = oxsci.ALTERNATIVE_OXIDES.get(element)
            if alternatives:
                combo = QComboBox()
                combo.addItems(alternatives)
                self.oxide_map_table.setCellWidget(row, 1, combo)
            else:
                default = oxsci.DEFAULT_OXIDE_MAP.get(element)
                self.oxide_map_table.setItem(row, 1, QTableWidgetItem(default or ""))

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
        row = self.envelope_table.rowCount()
        self.envelope_table.insertRow(row)
        self.envelope_table.setItem(row, 0, QTableWidgetItem(""))
        self.envelope_table.setItem(row, 1, QTableWidgetItem("0"))
        self.envelope_table.setItem(row, 2, QTableWidgetItem("100"))

    def _remove_envelope_row(self) -> None:
        row = self.envelope_table.currentRow()
        if row >= 0:
            self.envelope_table.removeRow(row)

    def _populate_envelope_table(self, envelope: Dict[str, Tuple[float, float]]) -> None:
        self.envelope_table.setRowCount(0)
        for oxide, (lo, hi) in envelope.items():
            row = self.envelope_table.rowCount()
            self.envelope_table.insertRow(row)
            self.envelope_table.setItem(row, 0, QTableWidgetItem(oxide))
            self.envelope_table.setItem(row, 1, QTableWidgetItem(str(lo)))
            self.envelope_table.setItem(row, 2, QTableWidgetItem(str(hi)))

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

    def save_envelope(self) -> None:
        envelope = self._current_envelope()
        if not envelope:
            QMessageBox.information(self, "Save envelope", "Add at least one envelope row first.")
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
        QMessageBox.information(self, "Save envelope", f"Saved {path}")

    # ------------------------------------------------------------------
    # Build / recompute
    # ------------------------------------------------------------------
    def build_composition(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Oxide Chemistry", "Load a dataset first.")
            return
        tanks = self.selected_tanks()
        if not tanks:
            QMessageBox.warning(self, "No tanks", "Select one or more tanks first.")
            return
        elements = self._elements_in_tanks(tanks)
        if not elements:
            QMessageBox.warning(self, "Oxide Chemistry", "No kg-unit elemental composition found for the selected tank(s).")
            return
        self._populate_oxide_map_table(elements)
        self._rebuild_and_show()

    def _rebuild_and_show(self) -> None:
        tanks = self.selected_tanks()
        oxide_map = self._current_oxide_map()
        self.oxide_tables = oxsci.tank_oxide_composition(self.dataset, tanks, oxide_map)
        if not self.oxide_tables:
            QMessageBox.warning(self, "Oxide Chemistry", "No kg-unit elemental composition found for the selected tank(s).")
            return
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
        self._show_current_sample()
        self.app_window.statusBar().showMessage(f"Oxide composition built for {len(self.oxide_tables)} sample(s).")

    def _on_sample_changed(self, _text: str) -> None:
        self._show_current_sample()

    def _show_current_sample(self) -> None:
        table = self.oxide_tables.get(self.sample_combo.currentText())
        if table is None:
            return
        self._table_views["Oxide composition"].set_dataframe(table)
        self.recompute_summary()

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
        if summary["excluded_from_basicity"]:
            parts.append("Excluded from Λ (no PNNL-20184 value): " + ", ".join(summary["excluded_from_basicity"]))
        self.summary_label.setText(" | ".join(parts) + "\n" + gscience.OPTICAL_BASICITY_SOURCE)

    # ------------------------------------------------------------------
    # GlassNet + export
    # ------------------------------------------------------------------
    def run_glassnet(self) -> None:
        table = self.oxide_tables.get(self.sample_combo.currentText())
        if table is None or table.empty:
            QMessageBox.information(self, "GlassNet", "Build the oxide composition first.")
            return
        row = oxsci.glassnet_input_row(table)
        if row.empty:
            QMessageBox.information(self, "GlassNet", "No oxide-basis components to predict from (only non-oxide elements present).")
            return
        self.gn_btn.setEnabled(False)
        self.gn_btn.setText("Predicting… (first run loads the model)")
        from qt_worker import run_in_thread
        run_in_thread(lambda: gscience.glassnet_predict(row), self._on_glassnet_done, self._on_glassnet_error)

    def _on_glassnet_error(self, traceback_text: str) -> None:
        self.gn_btn.setEnabled(True)
        self.gn_btn.setText("GlassNet predict (selected sample)")
        QMessageBox.critical(self, "GlassNet", traceback_text)

    def _on_glassnet_done(self, pred) -> None:
        self.gn_btn.setEnabled(True)
        self.gn_btn.setText("GlassNet predict (selected sample)")
        self._table_views["GlassNet"].set_dataframe(pred)
        self.tables.setCurrentWidget(self._table_views["GlassNet"])

    def export_tables(self) -> None:
        if not self.oxide_tables or self.dataset is None:
            QMessageBox.information(self, "Export", "Build the oxide composition first.")
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
        QMessageBox.information(self, "Export complete", f"Oxide chemistry tables exported:\n{out_dir}")
