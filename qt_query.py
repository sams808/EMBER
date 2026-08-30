"""
qt_query.py — the Query workspace: the first UI over the L1 query engine
(query_engine.py). Ask one flexible question -- any element, analyte, or
basket x any basis (kg / Ci / bridged grams / wt% / g/L) x any grouping
dimension -- instead of hunting for a workspace whose fixed plot list
happens to include it.

House pattern: primary row with the essentials, Advanced options collapsed
(scope filters), "?" help scoped to this workspace's NOTICE.md section.
Engine warnings render prominently (accent-colored) because they are data-
integrity statements, not chrome; per-member disclosures land in the Notes
tab AND in every export.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from query_engine import BASES, BASKET_PRESETS, DIMENSIONS, run_query

_SUBJECT_KINDS = ["Basket", "Element", "Analyte"]
_NO_DIM = "—"


class QueryPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._result = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("Query")
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        note = QLabel("Any element, analyte, or basket × any basis × any grouping — one engine, honest units.")
        note.setObjectName("SectionNote")
        header.addWidget(note)
        header.addStretch(1)
        header.addWidget(make_help_button(self, 19, "Query"))
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export)
        header.addWidget(export_btn)
        root.addLayout(header)

        primary = QHBoxLayout()
        primary.addWidget(QLabel("Subject"))
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(_SUBJECT_KINDS)
        self.kind_combo.currentTextChanged.connect(self._on_kind_changed)
        primary.addWidget(self.kind_combo)

        self.basket_combo = QComboBox()
        self.basket_combo.addItems(list(BASKET_PRESETS))
        self.basket_combo.setToolTip("Shipped starter baskets — saved groups of elements/analytes that query as one species.")
        primary.addWidget(self.basket_combo)

        self.subject_edit = QLineEdit("S")
        self.subject_edit.setMaximumWidth(120)
        self.subject_edit.setToolTip("Element symbol (apportioned from species) or exact analyte label.")
        self.subject_edit.setVisible(False)
        primary.addWidget(self.subject_edit)

        primary.addWidget(QLabel("Basis"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(BASES)
        self.basis_combo.setToolTip(
            "g bridges Ci rows to grams via ICRP-107 specific activities;\n"
            "kg / Ci keep one unit only; wt% and g/L divide by layer mass/volume."
        )
        primary.addWidget(self.basis_combo)

        primary.addWidget(QLabel("Group by"))
        self.group1_combo = QComboBox()
        self.group2_combo = QComboBox()
        dim_labels = [_NO_DIM] + [f"{label}" for label in DIMENSIONS.values()]
        for combo in (self.group1_combo, self.group2_combo):
            combo.addItems(dim_labels)
        self.group1_combo.setCurrentText(DIMENSIONS["WasteSiteId"])
        primary.addWidget(self.group1_combo)
        primary.addWidget(self.group2_combo)

        run_btn = QPushButton("Run query")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run)
        primary.addWidget(run_btn)
        primary.addStretch(1)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options (scope filters)")
        adv = self.advanced_section.content_layout()
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("Farm"))
        self.farm_combo = QComboBox()
        self.farm_combo.addItem("All")
        adv_row.addWidget(self.farm_combo)
        adv_row.addWidget(QLabel("Waste phase"))
        self.phase_combo = QComboBox()
        self.phase_combo.addItem("All")
        adv_row.addWidget(self.phase_combo)
        adv_row.addWidget(QLabel("SST/DST"))
        self.system_combo = QComboBox()
        self.system_combo.addItems(["All", "SST", "DST"])
        adv_row.addWidget(self.system_combo)
        adv_row.addStretch(1)
        adv.addLayout(adv_row)
        root.addWidget(self.advanced_section)

        # Engine warnings are data-integrity statements (mixed decay
        # cohorts, exclusions, unassigned uncertainty) -- always visible,
        # never tucked into a tab.
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setObjectName("WarningNote")
        self.warnings_label.setVisible(False)
        root.addWidget(self.warnings_label)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self.result_view = DataFrameTableView(title="Query result", max_rows_display=1000)
        self.member_view = DataFrameTableView(title="Per member", max_rows_display=1000)
        self.notes_view = DataFrameTableView(title="Notes", max_rows_display=200)
        self.tables.addTab(self.result_view, "Result")
        self.tables.addTab(self.member_view, "Per member")
        self.tables.addTab(self.notes_view, "Notes")
        splitter.addWidget(self.tables)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

    # ------------------------------------------------------------------
    def _on_kind_changed(self, kind: str) -> None:
        self.basket_combo.setVisible(kind == "Basket")
        self.subject_edit.setVisible(kind != "Basket")

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # Review finding: a result computed on the PREVIOUS dataset stayed
        # on screen (and would export into the new dataset's output folder)
        # -- clear everything result-shaped before repopulating the combos.
        self._result = None
        self._subject_label = ""
        for view in (self.result_view, self.member_view, self.notes_view):
            view.set_dataframe(pd.DataFrame())
        self.warnings_label.setVisible(False)
        self.plot.show_message("Run a query to see results here.")
        if not dataset.is_loaded():
            return
        df = dataset.require_df()
        for combo, column in ((self.farm_combo, "TankFarm"), (self.phase_combo, "WastePhase")):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            values = sorted({str(v) for v in df.get_column(column).drop_nulls().unique().to_list()})
            combo.addItems(values)
            restore = combo.findText(current)
            combo.setCurrentIndex(restore if restore >= 0 else 0)
            combo.blockSignals(False)

    def _dims_by_label(self) -> Dict[str, str]:
        return {label: col for col, label in DIMENSIONS.items()}

    def _selected_group_by(self) -> list:
        by_label = self._dims_by_label()
        out = []
        for combo in (self.group1_combo, self.group2_combo):
            label = combo.currentText()
            if label != _NO_DIM and label in by_label and by_label[label] not in out:
                out.append(by_label[label])
        return out

    def _selected_scope(self) -> Dict[str, list]:
        scope: Dict[str, list] = {}
        if self.farm_combo.currentText() != "All":
            scope["TankFarm"] = [self.farm_combo.currentText()]
        if self.phase_combo.currentText() != "All":
            scope["WastePhase"] = [self.phase_combo.currentText()]
        if self.system_combo.currentText() != "All":
            scope["TankSystem"] = [self.system_combo.currentText()]
        return scope

    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Query", "Load a dataset first.")
            return
        kind = self.kind_combo.currentText()
        if kind == "Basket":
            subject = ("basket", BASKET_PRESETS[self.basket_combo.currentText()])
            subject_label = self.basket_combo.currentText()
        elif kind == "Element":
            subject = ("element", self.subject_edit.text().strip())
            subject_label = self.subject_edit.text().strip()
        else:
            subject = ("analyte", self.subject_edit.text().strip())
            subject_label = self.subject_edit.text().strip()
        if not subject_label:
            QMessageBox.information(self, "Query", "Enter a subject first.")
            return

        try:
            result = run_query(
                self.dataset, subject=subject, basis=self.basis_combo.currentText(),
                group_by=self._selected_group_by(), scope=self._selected_scope(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Query", str(exc))
            return

        self._result = result
        self._subject_label = subject_label
        self.result_view.set_dataframe(result.table)
        self.member_view.set_dataframe(result.member_table)
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in result.warnings]
            + [{"kind": "disclosure", "text": d} for d in result.disclosures]
        )
        self.notes_view.set_dataframe(notes)

        if result.warnings:
            self.warnings_label.setText("⚠ " + "  •  ".join(result.warnings))
            self.warnings_label.setVisible(True)
        else:
            self.warnings_label.setVisible(False)

        self._update_plot()
        self.app_window.statusBar().showMessage(
            f"Query completed: {subject_label}, {self.basis_combo.currentText()}, "
            f"{len(result.table):,} group(s)."
        )

    def _update_plot(self) -> None:
        result = self._result
        if result is None:
            return
        group_by = self._selected_group_by()
        log_x = result.basis in ("g", "kg", "Ci")
        xlabel = {"g": "Total mass (g)", "kg": "Total mass (kg)", "Ci": "Total activity (Ci)",
                  "wt%": "wt% of layer mass", "g/L": "g per liter"}[result.basis]
        title = f"{self._subject_label} — {xlabel}"
        if not group_by:
            # No grouping: the informative picture is the per-member split.
            df = result.member_table
            label_col = "Member"
        elif len(group_by) == 1:
            df = result.table
            label_col = group_by[0]
        else:
            df = result.table.copy()
            df["_group"] = df[group_by[0]].astype(str) + " · " + df[group_by[1]].astype(str)
            label_col = "_group"
        ph.plot_barh(self.plot, df, label_col, result.value_column, title, xlabel,
                     top_n=30, log_x=log_x)

    def _export(self) -> None:
        if self._result is None or self.dataset is None:
            QMessageBox.information(self, "Export", "Run a query first.")
            return
        notes = pd.DataFrame(
            [{"kind": "warning", "text": w} for w in self._result.warnings]
            + [{"kind": "disclosure", "text": d} for d in self._result.disclosures]
        )
        tables = {
            "query_result": self._result.table,
            "query_per_member": self._result.member_table,
            "query_notes": notes,
        }
        out_dir = export_utils.export_named_tables(self.dataset, f"query_{self._subject_label}", tables)
        try:
            self.plot.figure.savefig(out_dir / "query_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Query outputs exported:\n{out_dir}")
