"""
qt_query.py — the Query workspace: the first UI over the L1 query engine
(query_engine.py) and, since Stage 2, the first grammar-driven perspective
(plot_grammar.py). Ask one flexible question -- any element, analyte, or
basket x any basis (kg / Ci / bridged grams / wt% / g/L) x any grouping
dimension -- and choose how to see it (bars, stacked bars, heatmap, box).

Stage-2 changes:
- Guided questions: one-click templates (query_templates.py) that fill the
  controls AND run, so a first-time user gets a correct, disclosed answer
  before learning any knobs.
- Scope comes from the app-wide global scope bar (qt_scope.py), not local
  combos -- one filter mechanism, stamped into the plot subtitle and the
  export manifest. A scope change re-runs the current question.
- Plotting goes through the L2 grammar: a second grouping dimension
  becomes hue (grouped/stacked bars) or the heatmap column axis, instead
  of joined text labels.
"""
from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
from data_model import HanfordDataset
from plot_grammar import MARK_BY_LABEL, MARK_LABELS, PlotSpec, render_plot
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import DataFrameTableView, PlotWidget
from query_engine import BASES, BASKET_PRESETS, DIMENSIONS, run_query
from query_plots import plan_plot
from query_templates import TEMPLATES

_SUBJECT_KINDS = ["Basket", "Element", "Analyte"]
_NO_DIM = "—"
_TEMPLATE_PLACEHOLDER = "Guided questions…"
# Marks this workspace offers, and the reverse lookups -- all derived from
# the grammar's own registry so a mark can never exist there but be
# untranslatable (or silently mislabeled) here. Scatter needs two value
# columns, which a single-measure query result doesn't have.
_MARKS_OFFERED = [m for m in MARK_LABELS if m != "scatter"]
_PLOT_MARKS = [MARK_LABELS[m] for m in _MARKS_OFFERED]
_COL_BY_DIM_LABEL = {label: col for col, label in DIMENSIONS.items()}


class QueryPage(QWidget):
    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        # The shell passes the shared global scope; standalone construction
        # (tests, tooling) gets a private, initially-empty one.
        self.scope = scope if scope is not None else GlobalScope(self)
        self.dataset: Optional[HanfordDataset] = None
        self._result = None
        self._subject_label = ""
        # The grouping and scope the STANDING result was computed under.
        # The live combos can differ (the user changed one without
        # re-running), and labelling a result with anything other than
        # what produced it is how a chart comes to claim a scope or a
        # grouping that never touched the numbers (review finding).
        self._result_group_by: List[str] = []
        self._result_scope = ""
        self._last_spec: Optional[PlotSpec] = None
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
        # Guided questions FIRST: the easiest entry point sits where a new
        # user reads first. Templates fill the controls and run, so the
        # knobs that produced the answer stay visible, not hidden.
        self.template_combo = QComboBox()
        self.template_combo.addItem(_TEMPLATE_PLACEHOLDER)
        for template in TEMPLATES:
            self.template_combo.addItem(template.question)
        for i, template in enumerate(TEMPLATES, start=1):
            self.template_combo.setItemData(i, template.explanation, 3)  # Qt.ToolTipRole
        self.template_combo.setToolTip("One-click starter questions — picking one fills the controls below and runs.")
        self.template_combo.activated.connect(self._on_template_activated)
        primary.addWidget(self.template_combo)

        primary.addSpacing(10)
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

        primary.addWidget(QLabel("Plot"))
        self.mark_combo = QComboBox()
        self.mark_combo.addItems(_PLOT_MARKS)
        self.mark_combo.setToolTip(
            "How the result is drawn. With two grouping dimensions, the second one\n"
            "becomes the bar series (grouped/stacked) or the heatmap columns;\n"
            "Box shows the distribution of dimension-1 values within each dimension-2 level."
        )
        self.mark_combo.currentTextChanged.connect(lambda _t: self._update_plot())
        primary.addWidget(self.mark_combo)

        run_btn = QPushButton("Run query")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run)
        primary.addWidget(run_btn)
        primary.addStretch(1)
        root.addLayout(primary)

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

    def _on_template_activated(self, index: int) -> None:
        if index <= 0:
            return
        template = TEMPLATES[index - 1]
        self.apply_template(template)

    def apply_template(self, template) -> None:
        """Fill every control from a guided template, then run."""
        self.kind_combo.setCurrentText(template.subject_kind.capitalize())
        if template.subject_kind == "basket":
            self.basket_combo.setCurrentText(template.subject)
        else:
            self.subject_edit.setText(template.subject)
        self.basis_combo.setCurrentText(template.basis)
        dims = list(template.group_by)
        self.group1_combo.setCurrentText(DIMENSIONS[dims[0]] if dims else _NO_DIM)
        self.group2_combo.setCurrentText(DIMENSIONS[dims[1]] if len(dims) > 1 else _NO_DIM)
        self.mark_combo.setCurrentText(MARK_LABELS[template.mark])
        self.run()

    def prefill_and_run(self, kind: str, subject: str, basis: Optional[str] = None) -> None:
        """Cross-link entry point (e.g. Element Explorer's "Ask in Query"):
        set the subject, keep the current grouping, and answer."""
        self.kind_combo.setCurrentText(kind.capitalize())
        if kind.lower() == "basket":
            self.basket_combo.setCurrentText(subject)
        else:
            self.subject_edit.setText(subject)
        if basis is not None:
            self.basis_combo.setCurrentText(basis)
        self.run()

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A result computed on the PREVIOUS dataset must not stay on screen
        # (and must never export into the new dataset's output folder).
        self._result = None
        self._subject_label = ""
        self._result_group_by = []
        self._result_scope = ""
        self._last_spec = None
        for view in (self.result_view, self.member_view, self.notes_view):
            view.set_dataframe(pd.DataFrame())
        self.warnings_label.setVisible(False)
        self.plot.show_message("Run a query to see results here.")

    def on_scope_changed(self) -> None:
        """Global scope bar changed: re-ask the current question under the
        new scope. Without a standing result there is nothing to re-ask."""
        if self._result is not None:
            self.run()

    def _selected_group_by(self) -> List[str]:
        out: List[str] = []
        for combo in (self.group1_combo, self.group2_combo):
            col = _COL_BY_DIM_LABEL.get(combo.currentText())
            if col is not None and col not in out:
                out.append(col)
        return out

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

        group_by = self._selected_group_by()
        try:
            result = run_query(
                self.dataset, subject=subject, basis=self.basis_combo.currentText(),
                group_by=group_by, scope=self.scope.as_scope(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Query", str(exc))
            return

        self._result = result
        self._subject_label = subject_label
        self._result_group_by = group_by
        self._result_scope = self.scope.summary()
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
        scope_note = f", scope: {self.scope.summary()}" if self.scope.is_active() else ""
        self.app_window.statusBar().showMessage(
            f"Query completed: {subject_label}, {self.basis_combo.currentText()}, "
            f"{len(result.table):,} group(s){scope_note}."
        )

    # ------------------------------------------------------------------
    def _scope_subtitle(self) -> str:
        # The scope the STANDING result was computed under, never the live
        # one: a scope change whose re-run early-returns (blank subject,
        # engine refusal) must not relabel the old numbers.
        return f"Scope: {self._result_scope}" if self._result_scope else ""

    def _update_plot(self) -> None:
        if self._result is None:
            return
        plan = plan_plot(
            self._result, self._result_group_by,
            MARK_BY_LABEL[self.mark_combo.currentText()],
            subject_label=self._subject_label, subtitle=self._scope_subtitle(),
        )
        if not plan.is_drawable:
            self._last_spec = None
            self.plot.show_message(plan.refusal)
            return
        self._last_spec = plan.spec
        render_plot(self.plot, plan.data, plan.spec)

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
        context = {
            "subject": self._subject_label,
            "basis": self._result.basis,
            "global_scope": self._result_scope or "(none)",
            "plot_spec": json.dumps(self._last_spec.to_dict()) if self._last_spec else "(none)",
        }
        import provenance as prov
        from qt_help import APP_NAME, APP_VERSION
        record = prov.build_provenance(
            self.dataset, app_name=APP_NAME, app_version=APP_VERSION,
            question={"subject": self._subject_label, "basis": self._result.basis,
                      "group_by": list(self._result_group_by)},
            plot_spec=self._last_spec.to_dict() if self._last_spec else {},
            scope=self._result_scope,
            warnings=self._result.warnings, disclosures=self._result.disclosures,
        )
        out_dir = export_utils.export_named_tables(
            self.dataset, f"query_{self._subject_label}", tables, context=context,
            provenance_record=record,
        )
        try:
            self.plot.figure.savefig(out_dir / "query_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Query outputs exported:\n{out_dir}")
