"""
qt_tank_attributes.py -- Tank attributes: the engineering and status
metadata joined from the attributes file (type, capacity, integrity,
ventilation), independent of composition. One row per tank, an audit
grouped by a chosen attribute, and a chart of that audit. A double-click
on a tank row opens the tank in the Tank explorer.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import plot_helpers as ph
import tank_science as tsci
from data_model import HanfordDataset
from qt_components import EmptyState, PageHeader
from qt_help import make_help_button
from qt_widgets import DataFrameTableView, PlotWidget

CATEGORY_CHOICES = ["TankType", "TankSystem", "TankIntegrity", "TankStatus", "Ventilation", "TankFarm"]
CATEGORY_LABELS = {
    "TankType": "Tank type",
    "TankSystem": "Tank system (single or double shell)",
    "TankIntegrity": "Tank integrity",
    "TankStatus": "Tank status",
    "Ventilation": "Ventilation",
    "TankFarm": "Tank farm",
}
PLOT_TYPES = ["Count by category", "Capacity by category", "DIL by category", "Integrity count"]
_PREVIEW_COLUMNS = [
    "WasteSiteId", "Analyte", "Inventory", "Units", "TankType",
    "TankSystem", "TankIntegrity", "Ventilation", "DIL_Gal",
]


class TankAttributesPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._audit_df: Optional[pd.DataFrame] = None
        self._category = "TankType"
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 12)
        root.setSpacing(16)

        self.header = PageHeader(
            "Tank attributes",
            "Type, capacity, integrity and status of every tank, from the attributes file. "
            "Double-click a tank to open it.")
        self.export_btn = QPushButton("Export...")
        self.export_btn.setObjectName("Secondary")      # the quiet header action, as on every page
        self.export_btn.setToolTip("Write the four tables to a dated folder of CSV files.")
        self.export_btn.clicked.connect(self._export_tables)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 7, "Tank Attributes"))
        root.addWidget(self.header)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        label = QLabel("Group by")
        label.setObjectName("FieldLabel")
        controls.addWidget(label)
        self.category_combo = QComboBox()
        for column in CATEGORY_CHOICES:
            self.category_combo.addItem(CATEGORY_LABELS[column], column)
        self.category_combo.setToolTip("The attribute the audit table and the chart are grouped by.")
        controls.addWidget(self.category_combo)
        label = QLabel("Chart")
        label.setObjectName("FieldLabel")
        controls.addWidget(label)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.setToolTip(
            "Tank count, total capacity, drainable interstitial liquid (DIL), or the number of "
            "leakers and assumed leakers, per group.")
        controls.addWidget(self.plot_type_combo)
        self.refresh_btn = QPushButton("Update")
        self.refresh_btn.setToolTip("Recompute the tables and the chart.")
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        controls.addWidget(self.refresh_btn)
        controls.addStretch(1)
        root.addLayout(controls)

        self.stack = QStackedWidget()
        self.empty_state = EmptyState(
            "No dataset loaded",
            "Load the tank inventory and its attributes file to browse the tanks.")
        self.stack.addWidget(self.empty_state)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        self.tables = QTabWidget()
        self._table_views: dict[str, DataFrameTableView] = {}
        for title, max_rows in [
            ("Attributes", None), ("Category audit", None),
            ("Numeric summary", None), ("Merged preview", 500),
        ]:
            view = DataFrameTableView(title=title, max_rows_display=max_rows)
            self._table_views[title] = view
            self.tables.addTab(view, title)
        splitter.addWidget(self.tables)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.stack.addWidget(splitter)
        self._splitter = splitter
        root.addWidget(self.stack, 1)

        self.category_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self.plot_type_combo.currentTextChanged.connect(lambda _: self._update_plot())

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        if dataset.is_loaded():
            self.stack.setCurrentWidget(self._splitter)
            self.refresh(announce=False)
        else:
            for view in self._table_views.values():
                view.set_dataframe(view.dataframe().iloc[0:0])
            self.plot.show_message("Load a dataset first.")
            self.stack.setCurrentWidget(self.empty_state)

    def refresh(self, announce: bool = True) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Tank attributes", "Load a dataset first.")
            return
        category = self.category_combo.currentData() or "TankType"
        attrs = tsci.tank_attributes_table(self.dataset)
        audit = tsci.tank_attribute_audit(self.dataset, category)
        numeric = tsci.tank_attribute_numeric_summary(self.dataset)
        df = self.dataset.require_df()
        preview_cols = [c for c in _PREVIEW_COLUMNS if c in df.columns]
        merged_preview = df.select(preview_cols).head(500).to_pandas() if preview_cols else pd.DataFrame()

        self._table_views["Attributes"].set_dataframe(attrs)
        self._table_views["Category audit"].set_dataframe(audit)
        self._table_views["Numeric summary"].set_dataframe(numeric)
        self._table_views["Merged preview"].set_dataframe(merged_preview)
        self._audit_df = audit
        self._category = category
        self.stack.setCurrentWidget(self._splitter)
        self._update_plot()
        if announce:
            self.app_window.statusBar().showMessage(
                f"Tank attributes updated ({CATEGORY_LABELS.get(category, category).lower()}).")

    def _update_plot(self) -> None:
        if self._audit_df is None or self._audit_df.empty:
            self.plot.show_message("No tank attributes in this dataset")
            return
        category = self._category
        name = CATEGORY_LABELS.get(category, category).split(" (")[0].lower()
        ptype = self.plot_type_combo.currentText()
        if ptype == "Capacity by category" and "Total_capacity_kgal" in self._audit_df.columns:
            ph.plot_barh(self.plot, self._audit_df, category, "Total_capacity_kgal",
                         f"Total capacity by {name}", "Total capacity (kgal)", top_n=50)
        elif ptype == "DIL by category" and "Total_DIL_gal" in self._audit_df.columns:
            ph.plot_barh(self.plot, self._audit_df, category, "Total_DIL_gal",
                         f"Drainable interstitial liquid by {name}", "Drainable interstitial liquid (gal)",
                         top_n=50)
        elif ptype == "Integrity count" and "N_leaker_or_assumed" in self._audit_df.columns:
            ph.plot_barh(self.plot, self._audit_df, category, "N_leaker_or_assumed",
                         f"Leakers and assumed leakers by {name}", "Number of tanks", top_n=50)
        else:
            ph.plot_barh(self.plot, self._audit_df, category, "N_tanks",
                         f"Tanks by {name}", "Number of tanks", top_n=50)

    def _export_tables(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Export", "Load a dataset first.")
            return
        tables = {name: view.dataframe() for name, view in self._table_views.items()}
        out_dir = export_utils.export_named_tables(self.dataset, "tank_attributes", tables)
        QMessageBox.information(self, "Export complete", f"Tank attributes tables exported:\n{out_dir}")
