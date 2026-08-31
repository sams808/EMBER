"""
qt_overview.py — the Overview workspace: first sanity check of the loaded
dataset (audits + a top-N inventory chart), ported from the old app's
OverviewTab onto the new DataFrameTableView/PlotWidget infrastructure.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import matrix_query as mq
import overview_science as ov
import plot_helpers as ph
import tank_science as tsci
from data_model import HanfordDataset
from plot_grammar import PlotSpec, render_plot
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import DataFrameTableView, PlotWidget
from query_engine import BASES

PLOT_TYPES = ["Top elements by inventory", "Farm rollup by inventory"]

# The Overview unit picker offers "All" alongside the dataset's own units.
# For element totals that maps onto the engine's bridged mass basis rather
# than ranking kilograms against curies on one axis, which is not a
# comparison of anything.
_ELEMENT_BASIS = {"All": "g"}


def _element_basis(unit: str) -> Optional[str]:
    """The engine basis for a Unit-combo entry, or None when the dataset
    reports in a unit the engine has no mass/activity semantics for. The
    combo is filled from the data, so a CSV using "MT" or "gal" is
    entirely possible -- passing it through as a basis raised straight out
    of refresh() and, since Overview is refreshed first, aborted the
    shell's whole dataset broadcast (review finding)."""
    basis = _ELEMENT_BASIS.get(unit, unit)
    return basis if basis in BASES else None


class OverviewPage(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.scope = scope if scope is not None else GlobalScope(self)
        self.dataset: Optional[HanfordDataset] = None
        self._element_totals = mq.ElementTotals()
        self._farm_rollup_df = pd.DataFrame()
        self._farm_rollup_unit_used: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("Overview")
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(make_help_button(self, 5, "Overview"))
        root.addLayout(header)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Unit"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("All")
        controls.addWidget(self.unit_combo)
        controls.addWidget(QLabel("Top N"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 300)
        self.top_n_spin.setValue(40)
        controls.addWidget(self.top_n_spin)
        controls.addWidget(QLabel("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.currentTextChanged.connect(lambda _: self._update_plot())
        controls.addWidget(self.plot_type_combo)
        refresh_btn = QPushButton("Refresh overview")
        refresh_btn.setObjectName("Primary")
        refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(refresh_btn)
        controls.addStretch(1)
        export_btn = QPushButton("Export global debug bundle")
        export_btn.clicked.connect(self._export_debug_bundle)
        controls.addWidget(export_btn)
        root.addLayout(controls)

        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setObjectName("WarningNote")
        self.warnings_label.setVisible(False)
        root.addWidget(self.warnings_label)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self._table_views: dict[str, DataFrameTableView] = {}
        for title, max_rows in [
            ("Overview", None), ("Units", None), ("Top elements", None),
            ("Top analytes", None), ("Waste phases", None), ("Waste types", None),
            ("Tank farms", None), ("Missing", None), ("Raw preview", 250),
            ("Notes", 200),
        ]:
            view = DataFrameTableView(title=title, max_rows_display=max_rows)
            self._table_views[title] = view
            self.tables.addTab(view, title)
        splitter.addWidget(self.tables)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
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
        if dataset.is_loaded():
            self.refresh(announce=False)

    def refresh(self, announce: bool = True) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Overview", "Load a dataset first.")
            return
        unit = self.unit_combo.currentText() or "All"
        top_n = self.top_n_spin.value()

        self._table_views["Overview"].set_dataframe(ov.overview(self.dataset))
        self._table_views["Units"].set_dataframe(ov.units_audit(self.dataset))
        # Element totals come from the query engine, so this page agrees
        # with Heatmaps about what the waste contains: each element gets
        # its stoichiometric share of every species reporting it, and
        # "All" means kg and Ci bridged onto one mass axis rather than
        # ranked against each other.
        basis = _element_basis(unit)
        if basis is None:
            element_totals = mq.ElementTotals(warnings=[
                f"Element totals need a mass or activity basis; this dataset reports in "
                f"{unit!r}, which the engine has no chemistry for. The audit tables below "
                f"are unaffected."
            ])
        else:
            element_totals = mq.element_totals(self.dataset, basis=basis, top_n=top_n,
                                               scope=self.scope.as_scope())
        self._element_totals = element_totals
        self._table_views["Top elements"].set_dataframe(element_totals.table)
        self._table_views["Top analytes"].set_dataframe(ov.top_analytes(self.dataset, unit=unit, top_n=top_n))
        self._table_views["Waste phases"].set_dataframe(ov.phase_audit(self.dataset))
        self._table_views["Waste types"].set_dataframe(ov.type_audit(self.dataset))

        farm_unit = self._farm_rollup_unit()
        farm_rollup = tsci.farm_inventory_rollup(self.dataset, unit=farm_unit) if farm_unit else pd.DataFrame()
        self._table_views["Tank farms"].set_dataframe(farm_rollup if not farm_rollup.empty else ov.farm_audit(self.dataset))
        self._table_views["Missing"].set_dataframe(ov.missing_audit(self.dataset))
        self._table_views["Raw preview"].set_dataframe(self.dataset.raw_preview(250))

        self._farm_rollup_df = farm_rollup
        # The unit the cached rollup was actually computed in -- the plot
        # must label with THIS, never the live combo, which the user may
        # have changed since the last Refresh (a "(kg)" title over Ci bars
        # is a silently wrong chart).
        self._farm_rollup_unit_used = farm_unit

        # The engine's warnings and disclosures belong ON the page: this
        # is the first screen anyone sees, and its headline table now
        # carries exclusions (trims, unattributable mass, isotopic
        # restatements) that were being computed and discarded.
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in element_totals.warnings]
            + [{"kind": "disclosure", "text": d} for d in element_totals.disclosures]
        )
        self._table_views["Notes"].set_dataframe(notes)
        if element_totals.warnings:
            self.warnings_label.setText("⚠ " + "  •  ".join(element_totals.warnings))
            self.warnings_label.setVisible(True)
        else:
            self.warnings_label.setVisible(False)

        self._update_plot()
        if announce:
            self.app_window.statusBar().showMessage(f"Overview refreshed ({unit}, top {top_n}).")

    def on_scope_changed(self) -> None:
        """Element totals are scoped, so this page must follow the scope
        bar or it contradicts Query and Heatmaps about the same element
        (review finding: Na read 46.2 Mkg here against 6.2 Mkg there)."""
        if self.dataset is not None and self.dataset.is_loaded():
            self.refresh(announce=False)

    def _farm_rollup_unit(self) -> Optional[str]:
        """Farm rollup needs one concrete unit (kg and Ci can't be summed
        together) -- honor an explicit kg/Ci choice, otherwise default to
        kg (falling back to whatever's actually available) the same way
        Quick Scan/Heatmaps default their own Unit picker."""
        if self.dataset is None or not self.dataset.is_loaded():
            return None
        units = self.dataset.available_units()
        if not units:
            return None
        current = self.unit_combo.currentText()
        if current in units:
            return current
        return "kg" if "kg" in units else units[0]

    def _update_plot(self) -> None:
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "Farm rollup by inventory":
            unit = self._farm_rollup_unit_used
            if self._farm_rollup_df.empty or not unit:
                self.plot.show_message("No farm rollup data for this dataset.")
                return
            ph.plot_barh(
                self.plot, self._farm_rollup_df, "TankFarm", f"Total_inventory_{unit}",
                f"Total inventory by tank farm ({unit})", f"Total inventory ({unit})",
                top_n=50,
            )
        else:
            totals = self._element_totals
            if totals.table.empty:
                self.plot.show_message("No element totals for this dataset.")
                return
            subtitle = ["Apportioned (stoichiometric)"]
            if self.scope.summary():
                subtitle.append(f"Scope: {self.scope.summary()}")
            # Trims and exclusions belong ON the figure: the exported PNG
            # is what ends up in a report, and it carried no trace of a
            # top-N cut (review finding).
            subtitle.extend(w for w in totals.warnings if "omitted" in w)
            render_plot(self.plot, totals.table, PlotSpec(
                mark="bar", x=totals.value_column, y="Element",
                title="Top elements by inventory",
                subtitle=" · ".join(subtitle),
                log_x=True, top_n=len(totals.table),
            ))

    def _export_debug_bundle(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Export", "Load a dataset first.")
            return
        # Hand the bundle the SAME element totals the page is showing, or
        # the archive and the screen disagree about what the waste holds.
        # A file named "_all" must not hold the on-screen top N, and the
        # bundle must carry the warnings that qualify its own numbers
        # (review finding).
        basis = _element_basis(self.unit_combo.currentText() or "All")
        untrimmed = (
            mq.element_totals(self.dataset, basis=basis, top_n=10_000,
                              scope=self.scope.as_scope())
            if basis else self._element_totals
        )
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in untrimmed.warnings]
            + [{"kind": "disclosure", "text": d} for d in untrimmed.disclosures]
        )
        out_dir = ov.export_global_debug_bundle(
            self.dataset, extra_info=self._app_extra_info(),
            extra_tables={
                "top_elements_all": untrimmed.table if not untrimmed.table.empty else None,
                "notes": notes if not notes.empty else None,
            },
        )
        QMessageBox.information(self, "Export complete", f"Debug bundle written to:\n{out_dir}")

    def _app_extra_info(self) -> dict:
        try:
            from qt_help import APP_NAME, APP_VERSION
            return {"app_name": APP_NAME, "app_version": APP_VERSION}
        except Exception:
            return {}
