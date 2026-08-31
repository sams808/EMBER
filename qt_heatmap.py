"""
qt_heatmap.py — Heatmaps workspace: tank x element inventory matrices.

Stage 2 migration: the matrix now comes from the query engine
(matrix_query.engine_matrix), so an element receives its stoichiometric
share of every species reporting it instead of the whole mass of every
analyte that happens to name it first. That changes real numbers -- on
the shipped dataset nitrogen falls to 0.24x and oxygen rises 15.6x, from
rank 6 to rank 1 -- so the older attribution stays selectable, and the
Attribution row on the figure says which one produced it.

Rendering goes through the L2 plot grammar rather than the frozen legacy
heatmap renderer, which is what gives this workspace the global scope
bar (it honors on_scope_changed) and the guardrail/disclosure behavior
for free.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import matrix_query as mq
import matrix_science as msci
from data_model import HanfordDataset
from plot_grammar import PlotSpec, render_plot
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

# Display mode -> (log_x, normalize). The engine supplies absolute values;
# these are presentation transforms, applied by the grammar.
MODES = {
    "log10_inventory": (True, "none"),
    "inventory": (False, "none"),
    "fraction": (False, "row"),
}


class HeatmapPage(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.scope = scope if scope is not None else GlobalScope(self)
        self.dataset: Optional[HanfordDataset] = None
        self._result = mq.MatrixResult()
        # Every setting that produced the standing result. Labels, the
        # subtitle and the export all read THIS, never the live widgets:
        # only the Mode combo triggers a redraw, so a Basis or Attribution
        # changed afterwards would otherwise relabel data it never touched
        # (review finding -- a figure titled "Ci" over kg data).
        self._params: dict = {}
        self._has_built = False
        self._build_ui()

    @property
    def _result_scope(self) -> str:
        return self._params.get("scope", "")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("Heatmaps")
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        note = QLabel("Tank × element matrix — every element gets its stoichiometric share.")
        note.setObjectName("SectionNote")
        header.addWidget(note)
        header.addStretch(1)
        header.addWidget(make_help_button(self, 9, "Heatmaps"))
        export_btn = QPushButton("Export inputs")
        export_btn.clicked.connect(self._export_inputs)
        header.addWidget(export_btn)
        root.addLayout(header)

        primary = QHBoxLayout()
        primary.addWidget(QLabel("Basis"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kg", "Ci"])
        self.unit_combo.setToolTip("kg keeps mass rows only; Ci keeps activity rows only.")
        primary.addWidget(self.unit_combo)
        primary.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODES))
        self.mode_combo.setToolTip(
            "How cell values are scaled: raw inventory, log10 inventory,\n"
            "or each tank's composition as a fraction of its own row total."
        )
        self.mode_combo.currentTextChanged.connect(lambda _t: self._update_plot())
        primary.addWidget(self.mode_combo)
        primary.addWidget(QLabel("Attribution"))
        self.attribution_combo = QComboBox()
        self.attribution_combo.addItems(mq.ATTRIBUTIONS)
        self.attribution_combo.setToolTip(
            "Apportioned: an element receives its mass fraction of every species\n"
            "reporting it (S gets 32.06/96.06 of SO4).\n"
            "As reported: each analyte's full mass goes to its first parsed element\n"
            "— the older behavior, kept for reproducing earlier figures."
        )
        primary.addWidget(self.attribution_combo)
        build_btn = QPushButton("Build heatmap")
        build_btn.setObjectName("Primary")
        build_btn.clicked.connect(self.run)
        primary.addWidget(build_btn)
        primary.addStretch(1)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("Top elements"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 120)
        self.top_n_spin.setValue(35)
        self.top_n_spin.setToolTip("How many of the most abundant elements to include as heatmap columns.")
        adv_row.addWidget(self.top_n_spin)
        adv_row.addWidget(QLabel("min inventory"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        self.min_inv_spin.setToolTip(
            "Row-level floor, applied before summing (as-reported attribution only)."
        )
        adv_row.addWidget(self.min_inv_spin)
        adv_row.addWidget(QLabel("max tanks (0=all)"))
        self.max_tanks_spin = QSpinBox()
        self.max_tanks_spin.setRange(0, 1000)
        self.max_tanks_spin.setToolTip("Limit to the N highest-inventory tanks; 0 uses every tank.")
        adv_row.addWidget(self.max_tanks_spin)
        adv_row.addStretch(1)
        adv.addLayout(adv_row)
        root.addWidget(self.advanced_section)

        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setObjectName("WarningNote")
        self.warnings_label.setVisible(False)
        root.addWidget(self.warnings_label)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self.long_view = DataFrameTableView(title="Heatmap long input")
        self.wide_view = DataFrameTableView(title="Heatmap wide matrix", max_rows_display=250)
        self.notes_view = DataFrameTableView(title="Notes", max_rows_display=200)
        self.tables.addTab(self.long_view, "Long")
        self.tables.addTab(self.wide_view, "Wide")
        self.tables.addTab(self.notes_view, "Notes")
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
        self._result = mq.MatrixResult()
        self._params = {}
        self._has_built = False
        for view in (self.long_view, self.wide_view, self.notes_view):
            view.set_dataframe(pd.DataFrame())
        self.warnings_label.setVisible(False)
        self.plot.show_message("Build a heatmap to see results.")

    def on_scope_changed(self) -> None:
        """Rebuild under the new scope once the user has asked for a
        matrix at all. Guarding on "the last build returned rows" instead
        would strand the workspace: a scope that emptied the matrix would
        stop every later scope change from rebuilding it (review
        finding)."""
        if self._has_built:
            self.run()

    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Heatmaps", "Load a dataset first.")
            return
        unit = self.unit_combo.currentText()
        top_n = self.top_n_spin.value()
        min_inv = self.min_inv_spin.value()
        max_tanks = self.max_tanks_spin.value()
        attribution = self.attribution_combo.currentText()
        scope = self.scope.as_scope()

        tank_subset = None
        if max_tanks > 0:
            tank_subset = msci.top_tanks_by_inventory(
                mq._scoped_copy(self.dataset, scope), unit, max_tanks)

        try:
            if attribution == mq.APPORTIONED:
                result = mq.engine_matrix(
                    self.dataset, basis=unit, top_n_elements=top_n,
                    scope=scope, tank_subset=tank_subset,
                )
                if min_inv > 0:
                    result.warnings.append(
                        "min inventory is a row-level filter of the as-reported "
                        "attribution and is not applied to apportioned values."
                    )
            else:
                result = mq.legacy_matrix(
                    self.dataset, unit=unit, top_n_elements=top_n,
                    min_inventory=min_inv, scope=scope, tank_subset=tank_subset,
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Heatmaps", str(exc))
            return

        self._result = result
        self._has_built = True
        self._params = {
            "basis": unit, "attribution": attribution, "top_n_elements": top_n,
            "min_inventory": min_inv, "max_tanks": max_tanks,
            "scope": self.scope.summary(),
        }
        self.long_view.set_dataframe(result.long_df)
        self.wide_view.set_dataframe(result.wide_df)
        self.notes_view.set_dataframe(pd.DataFrame(
            [{"kind": "warning", "text": w} for w in result.warnings]
            + [{"kind": "disclosure", "text": d} for d in result.disclosures]
        ))
        if result.warnings:
            self.warnings_label.setText("⚠ " + "  •  ".join(result.warnings))
            self.warnings_label.setVisible(True)
        else:
            self.warnings_label.setVisible(False)

        self._update_plot()
        scope_note = f", scope: {self._result_scope}" if self._result_scope else ""
        self.app_window.statusBar().showMessage(
            f"Heatmap built: {unit}, {len(result.elements)} elements, "
            f"{len(result.wide_df)} tanks{scope_note}."
        )

    def _update_plot(self) -> None:
        result = self._result
        if result.long_df.empty:
            # Never leave the previous build's figure standing beside
            # empty tables -- an unscoped chart under a scope that
            # matched nothing (review finding).
            self.plot.show_message(
                "No data for these settings." if self._has_built
                else "Build a heatmap to see results."
            )
            return
        log_x, normalize = MODES[self.mode_combo.currentText()]
        # Labels come from the settings that PRODUCED this result, never
        # from the live widgets.
        attribution = self._params.get("attribution", "")
        subtitle_parts = [p for p in [attribution] if p]
        if self._result_scope:
            subtitle_parts.append(f"Scope: {self._result_scope}")
        spec = PlotSpec(
            mark="heatmap", x=result.value_column, y="WasteSiteId", hue="Element",
            title=f"Tank × element — {self._params.get('basis', '')}",
            subtitle=" · ".join(subtitle_parts),
            log_x=log_x, normalize=normalize,
        )
        render_plot(self.plot, result.long_df, spec)

    def _export_inputs(self) -> None:
        if self._result.wide_df.empty and self._result.long_df.empty:
            QMessageBox.information(self, "Export", "Build a heatmap first.")
            return
        # Every value here describes the BUILD, not the current widget
        # state -- an export that labels tables with settings that did not
        # produce them is a self-consistent lie on disk (review finding).
        # Mode is the exception: it is a display transform, wired to a
        # redraw, so the live value is the one the saved PNG shows.
        params = pd.DataFrame(
            [{"parameter": "mode", "value": self.mode_combo.currentText()}]
            + [{"parameter": key, "value": value} for key, value in self._params.items()
               if key != "scope"]
        )
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in self._result.warnings]
            + [{"kind": "disclosure", "text": d} for d in self._result.disclosures]
        )
        tables = {
            "heatmap_long": self._result.long_df,
            "heatmap_wide": self._result.wide_df,
            "heatmap_parameters": params,
            "heatmap_notes": notes,
        }
        out_dir = export_utils.export_named_tables(
            self.dataset, f"heatmap_{self._params.get('basis', '')}", tables,
            context={
                "attribution": self._params.get("attribution", ""),
                "basis": self._params.get("basis", ""),
                "global_scope": self._result_scope or "(none)",
            },
        )
        try:
            self.plot.figure.savefig(out_dir / "heatmap.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Heatmap inputs exported:\n{out_dir}")
