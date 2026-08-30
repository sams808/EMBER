"""
qt_explorer.py — the Element / Analyte Explorer workspace: search by
element symbol, analyte name, or regex; see which tanks contain it, what
else tends to appear alongside it, and its typical composition. Ported
from the old app's SearchTab onto the new DataFrameTableView/PlotWidget
infrastructure.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import element_science as esci
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

MODES = ["auto", "element", "analyte_exact", "analyte_contains", "regex"]


def _unit_suffix(df: Optional[pd.DataFrame]) -> str:
    """" (kg)" / " (kg, Ci)" from whichever units are actually present in a
    results table's Units column -- "" for a dimensionless quantity (a
    fraction/percentage table, or one with no Units column at all)."""
    if df is None or df.empty or "Units" not in df.columns:
        return ""
    units = sorted({str(u) for u in df["Units"].dropna().unique()})
    return f" ({', '.join(units)})" if units else ""
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


class ExplorerPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._last_query: Optional[str] = None
        self._results: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("Element Explorer")
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(make_help_button(self, 6, "Element Explorer"))
        export_btn = QPushButton("Export result bundle")
        export_btn.clicked.connect(self._export_bundle)
        header.addWidget(export_btn)
        root.addLayout(header)

        primary = QHBoxLayout()
        primary.addWidget(QLabel("Query"))
        self.query_edit = QLineEdit("Se")
        self.query_edit.setMaximumWidth(160)
        self.query_edit.setToolTip("Element symbol, analyte name, or regex, depending on Mode.")
        primary.addWidget(self.query_edit)
        primary.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.setToolTip("\"auto\" guesses whether the query is an element symbol or an analyte name.")
        primary.addWidget(self.mode_combo)
        primary.addWidget(QLabel("Unit"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("All")
        primary.addWidget(self.unit_combo)
        run_btn = QPushButton("Run")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run_search)
        primary.addWidget(run_btn)
        ask_query_btn = QPushButton("Ask in Query →")
        ask_query_btn.setToolTip(
            "Answer this subject with the query engine instead: stoichiometrically\n"
            "apportioned element masses, Ci→g bridging, wt%/g-per-L bases, and\n"
            "per-source disclosures. (This workspace sums reported rows as-is.)"
        )
        ask_query_btn.clicked.connect(self._ask_in_query)
        primary.addWidget(ask_query_btn)
        primary.addStretch(1)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("min target inventory"))
        self.min_target_spin = QDoubleSpinBox()
        self.min_target_spin.setRange(0.0, 1e12)
        self.min_target_spin.setDecimals(6)
        self.min_target_spin.setToolTip("Drop tanks below this inventory of the query element/analyte itself.")
        adv_row.addWidget(self.min_target_spin)
        adv_row.addWidget(QLabel("min context inventory"))
        self.min_context_spin = QDoubleSpinBox()
        self.min_context_spin.setRange(0.0, 1e12)
        self.min_context_spin.setDecimals(6)
        self.min_context_spin.setToolTip("Drop co-elements/co-analytes below this inventory when computing what tends to appear alongside the query.")
        adv_row.addWidget(self.min_context_spin)
        adv_row.addWidget(QLabel("Top N"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 300)
        self.top_n_spin.setValue(40)
        self.top_n_spin.setToolTip("How many bars to show on the plot.")
        adv_row.addWidget(self.top_n_spin)
        adv_row.addStretch(1)
        adv.addLayout(adv_row)
        root.addWidget(self.advanced_section)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self._table_views: dict[str, DataFrameTableView] = {}
        for title, max_rows in [
            ("Target by tank", None), ("Target by phase", None), ("Target by type", None),
            ("Co-elements", None), ("Co-analytes", None),
            ("Composition (absolute)", None), ("Composition (fraction)", None),
            ("Composition (by tank)", None), ("Raw target rows", 500),
        ]:
            view = DataFrameTableView(title=title, max_rows_display=max_rows)
            self._table_views[title] = view
            self.tables.addTab(view, title)
        splitter.addWidget(self.tables)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.currentTextChanged.connect(self._update_plot)
        plot_row.addWidget(self.plot_type_combo)
        plot_row.addStretch(1)
        right_layout.addLayout(plot_row)
        self.plot = PlotWidget()
        right_layout.addWidget(self.plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    def _ask_in_query(self) -> None:
        """Cross-link to the Query workspace: same subject, engine
        semantics (apportioned masses, Ci→g bridge, disclosures)."""
        query = self.query_edit.text().strip()
        if not query:
            QMessageBox.information(self, "Element Explorer", "Enter a query first.")
            return
        symbol = None
        if self.mode_combo.currentText() in ("auto", "element"):
            from elements import normalize_element_symbol
            symbol = normalize_element_symbol(query)
        kind, subject = ("element", symbol) if symbol else ("analyte", query)
        ask = getattr(self.app_window, "ask_in_query", None)
        if ask is None:  # standalone construction in tests/tooling
            return
        if not ask(kind, subject):
            QMessageBox.information(self, "Element Explorer", "The Query workspace isn't available.")

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
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
        for view in self._table_views.values():
            view.set_dataframe(view.dataframe().iloc[0:0])
        self.plot.show_message("Run a search to see results.")

    def run_search(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Element Explorer", "Load a dataset first.")
            return
        query = self.query_edit.text().strip()
        if not query:
            QMessageBox.information(self, "Element Explorer", "Enter a query first.")
            return
        mode = self.mode_combo.currentText()
        unit = self.unit_combo.currentText() or "All"
        min_target = self.min_target_spin.value()
        min_context = self.min_context_spin.value()

        try:
            target_by_tank = esci.target_by_tank_unit(self.dataset, query, mode, unit, min_target)
            target_by_phase = esci.target_by_phase(self.dataset, query, mode, unit, min_target)
            target_by_type = esci.target_by_type(self.dataset, query, mode, unit, min_target)
            co_el = esci.co_elements(self.dataset, query, mode, unit, min_target, min_context)
            co_an = esci.co_analytes(self.dataset, query, mode, unit, min_target, min_context)
            abs_stats, frac_stats, tank_stats = esci.composition_stats(self.dataset, query, mode, unit, min_target, min_context)
            raw_rows, resolved_mode, symbol = esci.target_rows(self.dataset, query, mode, unit, min_target)
        except ValueError as exc:
            QMessageBox.warning(self, "Search failed", str(exc))
            return

        self._results = {
            "Target by tank": target_by_tank, "Target by phase": target_by_phase,
            "Target by type": target_by_type, "Co-elements": co_el, "Co-analytes": co_an,
            "Composition (absolute)": abs_stats, "Composition (fraction)": frac_stats,
            "Composition (by tank)": tank_stats,
        }
        self._table_views["Target by tank"].set_dataframe(target_by_tank)
        self._table_views["Target by phase"].set_dataframe(target_by_phase)
        self._table_views["Target by type"].set_dataframe(target_by_type)
        self._table_views["Co-elements"].set_dataframe(co_el)
        self._table_views["Co-analytes"].set_dataframe(co_an)
        self._table_views["Composition (absolute)"].set_dataframe(abs_stats)
        self._table_views["Composition (fraction)"].set_dataframe(frac_stats)
        self._table_views["Composition (by tank)"].set_dataframe(tank_stats)
        self._table_views["Raw target rows"].set_dataframe(raw_rows.to_pandas())

        self._last_query = query
        self._symbol = symbol
        self._update_plot()

        n_tanks = target_by_tank["WasteSiteId"].nunique() if not target_by_tank.empty else 0
        self.app_window.statusBar().showMessage(
            f"'{query}' ({resolved_mode}{' -> ' + symbol if symbol else ''}): {n_tanks} tank(s), "
            f"{raw_rows.height} matching row(s)."
        )

    def _update_plot(self) -> None:
        if not self._results:
            return
        plot_type = self.plot_type_combo.currentText()
        symbol = getattr(self, "_symbol", None)
        target_by_tank = self._results.get("Target by tank")
        target_by_phase = self._results.get("Target by phase")
        target_by_type = self._results.get("Target by type")
        co_el = self._results.get("Co-elements")
        abs_stats = self._results.get("Composition (absolute)")
        frac_stats = self._results.get("Composition (fraction)")

        if plot_type == "Target inventory by tank":
            xlabel = (f"Target inventory: {symbol}" if symbol else "Target inventory") + _unit_suffix(target_by_tank)
            ph.plot_barh(self.plot, target_by_tank, "WasteSiteId", "Target_Inventory_sum",
                         "Target inventory by tank", xlabel, top_n=self.top_n_spin.value(),
                         highlighted_label=symbol, log_x=True)
        elif plot_type in ("Target by waste phase", "Target by waste type"):
            # These tables have existed since day one, one tab over -- this
            # branch is what finally lets them be *drawn*. One shared body
            # (mapping, not twin copy-paste branches) so the next grouped
            # dimension is a table entry, not a pasted block. Mixed-unit
            # results (unit filter "All") get per-bar unit suffixes from
            # make_unique_plot_labels, so kg and Ci bars stay tellable
            # apart rather than silently sharing an axis.
            df, label_col, title = {
                "Target by waste phase": (target_by_phase, "WastePhase", "Target inventory by waste phase"),
                "Target by waste type": (target_by_type, "WasteType", "Target inventory by waste type"),
            }[plot_type]
            xlabel = (f"Target inventory: {symbol}" if symbol else "Target inventory") + _unit_suffix(df)
            ph.plot_barh(self.plot, df, label_col, "Target_Inventory_sum",
                         title, xlabel, top_n=self.top_n_spin.value(), log_x=True)
        elif plot_type == "Target fraction by tank":
            xlabel = f"Fraction of tank total: {symbol}" if symbol else "Fraction"
            ph.plot_barh(self.plot, target_by_tank, "WasteSiteId", "TargetFractionOfTankUnitInventory",
                         "Target fraction of tank total", xlabel, top_n=self.top_n_spin.value(),
                         highlighted_label=symbol)
        elif plot_type == "Target vs total inventory":
            ph.plot_target_vs_total(self.plot, target_by_tank, "Target vs total tank inventory")
        elif plot_type == "Co-elements inventory":
            xlabel = "Total inventory" + _unit_suffix(co_el) + (f", tanks containing {symbol}" if symbol else "")
            ph.plot_barh(self.plot, co_el, "Element", "Total_inventory_in_target_tanks",
                         "Co-elements inventory", xlabel, top_n=self.top_n_spin.value(),
                         highlighted_label=symbol, log_x=True)
        elif plot_type == "Co-elements presence (%)":
            ph.plot_barh(self.plot, co_el, "Element", "PresenceFraction_pct",
                         "Co-elements presence", "% of target tanks", top_n=self.top_n_spin.value(),
                         highlighted_label=symbol)
        elif plot_type == "Mean absolute composition":
            xlabel = "Mean inventory (zero-filled)" + _unit_suffix(abs_stats)
            ph.plot_barh(self.plot, abs_stats, "Element", "Mean_inventory_all_target_tanks_zero_filled",
                         "Mean absolute composition", xlabel, top_n=self.top_n_spin.value(),
                         highlighted_label=symbol, log_x=True)
        elif plot_type == "Mean fraction composition":
            ph.plot_barh(self.plot, frac_stats, "Element", "Mean_fraction_all_target_tanks_zero_filled",
                         "Mean fraction composition", "Mean fraction (zero-filled)", top_n=self.top_n_spin.value(),
                         highlighted_label=symbol)

    def _export_bundle(self) -> None:
        if not self._results or self.dataset is None:
            QMessageBox.information(self, "Export", "Run a search first.")
            return
        tables = dict(self._results)
        tables["Raw target rows"] = self._table_views["Raw target rows"].dataframe()
        out_dir = export_utils.export_named_tables(self.dataset, f"search_{self._last_query}", tables)
        try:
            self.plot.figure.savefig(out_dir / "plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Search results exported to:\n{out_dir}")
