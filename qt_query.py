"""
qt_query.py — the Query workspace: one flexible question over the L1 query
engine (query_engine.py), drawn by the L2 plot grammar (plot_grammar.py).
Any element, analyte, or basket x any basis (kg / Ci / bridged grams /
wt% / g/L) x any grouping dimension, seen as bars, stacked bars, a heat
map or a box plot.

What the page owns:
- Guided questions: one-click templates (query_templates.py) that fill the
  controls AND run, so a first-time user gets a correct, disclosed answer
  before learning any control. They are offered on the page header and in
  View -> Guided questions.
- Scope comes from the app-wide scope bar (qt_scope.py), not local combos:
  one filter mechanism, stamped into the plot subtitle and the export
  context. A scope change re-asks the standing question.
- The notes that qualify a result (mixed decay cohorts, exclusions,
  unassigned uncertainty) sit above the result as one line, with the full
  list one click away and in the Notes tab.
- A click on a chart mark selects its row in the table; a double click on
  anything that stands for a tank opens that tank.
"""
from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QLineEdit, QMenu, QMessageBox, QPushButton, QSplitter, QTabWidget, QVBoxLayout,
    QWidget,
)

import export_utils
from data_model import HanfordDataset
from plot_grammar import MARK_BY_LABEL, MARK_LABELS, PlotSpec, render_plot
from qt_components import (
    NotesBanner, PageHeader, control_row, dataset_values, field_label, secondary_button,
    set_completions,
)
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import DataFrameTableView, PlotWidget
from query_engine import BASES, BASKET_PRESETS, DIMENSIONS, run_query
from query_plots import SOURCE_MEMBER_TABLE, plan_plot
from query_templates import TEMPLATES

_SUBJECT_KINDS = ["Basket", "Element", "Analyte"]
_NO_DIM = "—"
# Marks this workspace offers, and the reverse lookups -- all derived from
# the grammar's own registry so a mark can never exist there but be
# untranslatable (or silently mislabeled) here. Scatter needs two value
# columns, which a single-measure query result doesn't have.
_MARKS_OFFERED = [m for m in MARK_LABELS if m != "scatter"]
_PLOT_MARKS = [MARK_LABELS[m] for m in _MARKS_OFFERED]
_COL_BY_DIM_LABEL = {label: col for col, label in DIMENSIONS.items()}

PAGE_SUBTITLE = (
    "Ask one question of the inventory: any element, analyte or basket, on any basis, grouped by "
    "any dimension."
)
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "Choose a subject, a basis and a grouping, then press Run. Guided questions fill the controls and run."
_SUBJECT_TOOLTIPS = {
    "Element": "Element symbol, for example S or Cr. The element receives its share of every species that reports it.",
    "Analyte": "Exact analyte label as the dataset writes it, for example SO4 or 137Cs.",
}
_SUBJECT_PLACEHOLDERS = {"Element": "Symbol, e.g. S", "Analyte": "Label, e.g. 137Cs"}
_COMPLETION_COLUMN = {"Element": "Element", "Analyte": "Analyte"}


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
        self._result_inputs: Optional[tuple] = None
        self._last_spec: Optional[PlotSpec] = None
        self._plot_source = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader("Query", PAGE_SUBTITLE)
        self.guided_btn = secondary_button(
            "Guided questions",
            "Ready-made questions. Picking one fills every control on this page and runs it.")
        self.guided_btn_menu = QMenu(self.guided_btn)
        self.guided_btn_menu.setToolTipsVisible(True)
        for template in TEMPLATES:
            action = self.guided_btn_menu.addAction(template.question)
            action.setToolTip(template.explanation)
            action.triggered.connect(lambda _checked=False, t=template: self.apply_template(t))
        self.guided_btn.setMenu(self.guided_btn_menu)
        self.header.add_action_widget(self.guided_btn)
        self.export_btn = secondary_button(
            "Export...", "Write the result tables, the notes, the chart and a methods paragraph to a folder.")
        self.export_btn.clicked.connect(self._export)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 19, "Query"))
        root.addWidget(self.header)

        controls, row = control_row()
        row.addWidget(field_label("Subject"))
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(_SUBJECT_KINDS)
        self.kind_combo.setToolTip(
            "Basket: a saved group of elements and analytes, asked as one.\n"
            "Element: one element, apportioned from the species that contain it.\n"
            "Analyte: one reported label, exactly as the dataset writes it.")
        self.kind_combo.currentTextChanged.connect(self._on_kind_changed)
        row.addWidget(self.kind_combo)

        self.basket_combo = QComboBox()
        self.basket_combo.addItems(list(BASKET_PRESETS))
        for index, basket in enumerate(BASKET_PRESETS.values()):
            tip = ", ".join(str(member) for member in basket.members)
            if basket.description:
                tip = f"{tip}\n{basket.description}"
            self.basket_combo.setItemData(index, tip, Qt.ToolTipRole)
        self.basket_combo.setToolTip("Shipped baskets: saved groups of elements and analytes that are asked as one.")
        self.basket_combo.currentTextChanged.connect(self._sync_run_label)
        row.addWidget(self.basket_combo)

        self.subject_edit = QLineEdit("S")
        self.subject_edit.setMinimumWidth(96)
        self.subject_edit.setMaximumWidth(150)
        self.subject_edit.setVisible(False)
        self.subject_edit.textChanged.connect(self._sync_run_label)
        self.subject_edit.returnPressed.connect(self.run)
        row.addWidget(self.subject_edit)

        row.addSpacing(8)
        row.addWidget(field_label("Basis"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(BASES)
        self.basis_combo.setToolTip(
            "g: mass, with activity rows converted to grams through ICRP-107 specific activities.\n"
            "kg / Ci: one reported unit only.\n"
            "wt% and g/L: divided by the mass or the volume of each waste layer.")
        self.basis_combo.currentTextChanged.connect(self._sync_run_label)
        row.addWidget(self.basis_combo)

        row.addSpacing(8)
        row.addWidget(field_label("Group by"))
        self.group1_combo = QComboBox()
        self.group2_combo = QComboBox()
        dim_labels = [_NO_DIM] + [f"{label}" for label in DIMENSIONS.values()]
        for combo, tip in (
            (self.group1_combo, "First grouping dimension: one row and one bar per value."),
            (self.group2_combo, "Optional second dimension: the bar series, the heat-map columns or the box categories."),
        ):
            combo.addItems(dim_labels)
            combo.setToolTip(tip)
            combo.currentTextChanged.connect(self._sync_run_label)
        self.group1_combo.setCurrentText(DIMENSIONS["WasteSiteId"])
        row.addWidget(self.group1_combo)
        row.addWidget(self.group2_combo)

        row.addSpacing(8)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        row.addWidget(self.run_btn)
        row.addStretch(1)

        row.addWidget(field_label("Plot"))
        self.mark_combo = QComboBox()
        self.mark_combo.addItems(_PLOT_MARKS)
        self.mark_combo.setToolTip(
            "How the result is drawn. With two grouping dimensions, the second one\n"
            "becomes the bar series (grouped or stacked) or the heat-map columns;\n"
            "Box shows the distribution of dimension-1 values within each dimension-2 level.")
        self.mark_combo.currentTextChanged.connect(lambda _t: self._update_plot())
        row.addWidget(self.mark_combo)
        root.addWidget(controls)

        # Engine warnings are data-integrity statements (mixed decay
        # cohorts, exclusions, unassigned uncertainty): always on the page
        # with the result, as one line, the full list one click away.
        self.notes_banner = NotesBanner()
        self.warnings_label = self.notes_banner         # the name every page uses for this slot
        root.addWidget(self.notes_banner)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        self.plot = PlotWidget()
        self.plot.markClicked.connect(self._on_mark_clicked)
        splitter.addWidget(self.plot)

        self.tables = QTabWidget()
        self.result_view = DataFrameTableView(title="Query result", max_rows_display=1000)
        self.member_view = DataFrameTableView(title="Per member", max_rows_display=1000)
        self.notes_view = DataFrameTableView(title="Notes", max_rows_display=200)
        self.result_view.set_empty_text("No result yet", "One row per group appears after a run.")
        self.member_view.set_empty_text("No result yet", "One row per basket member appears after a run.")
        self.notes_view.set_empty_text("No notes", "What qualifies a result is listed here after a run.")
        self.tables.addTab(self.result_view, "Result")
        self.tables.addTab(self.member_view, "Per member")
        self.tables.addTab(self.notes_view, "Notes")
        splitter.addWidget(self.tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

        self._on_kind_changed(self.kind_combo.currentText())
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)

    # ------------------------------------------------------------------
    def _on_kind_changed(self, kind: str) -> None:
        self.basket_combo.setVisible(kind == "Basket")
        self.subject_edit.setVisible(kind != "Basket")
        self.subject_edit.setToolTip(_SUBJECT_TOOLTIPS.get(kind, ""))
        self.subject_edit.setPlaceholderText(_SUBJECT_PLACEHOLDERS.get(kind, ""))
        self._refresh_completions()
        self._sync_run_label()

    def _refresh_completions(self) -> None:
        """Offer what the loaded dataset actually holds for the current kind of subject."""
        column = _COMPLETION_COLUMN.get(self.kind_combo.currentText())
        if column is not None:
            set_completions(self.subject_edit, dataset_values(self.dataset, column))

    def _read_inputs(self) -> tuple:
        kind = self.kind_combo.currentText()
        subject = self.basket_combo.currentText() if kind == "Basket" else self.subject_edit.text().strip()
        return (kind, subject, self.basis_combo.currentText(), tuple(self._selected_group_by()))

    def _sync_run_label(self, *_args) -> None:
        """"Run" asks a question; "Update" re-asks after a control changed under a standing result."""
        if not hasattr(self, "run_btn"):
            return
        changed = self._result is not None and self._result_inputs != self._read_inputs()
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The controls differ from the result on screen: run again to bring it up to date." if changed
            else "Answer the question the controls describe.")

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
        self._result_inputs = None
        self._last_spec = None
        self._plot_source = ""
        for view in (self.result_view, self.member_view, self.notes_view):
            view.set_dataframe(pd.DataFrame())
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self._refresh_completions()
        self._sync_run_label()

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

    def _say(self, message: str) -> None:
        self.app_window.statusBar().showMessage(message)

    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Query: no dataset is loaded yet.")
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
            self._say("Query: enter a subject first.")
            self.subject_edit.setFocus()
            return

        group_by = self._selected_group_by()
        try:
            result = run_query(
                self.dataset, subject=subject, basis=self.basis_combo.currentText(),
                group_by=group_by, scope=self.scope.as_scope(),
            )
        except ValueError as exc:
            # The question cannot be answered as asked: say why, and leave
            # the previous answer standing under its own labels.
            QMessageBox.information(self, "Query", str(exc))
            return

        self._result = result
        self._subject_label = subject_label
        self._result_group_by = group_by
        self._result_scope = self.scope.summary()
        self._result_inputs = self._read_inputs()
        self.result_view.set_dataframe(result.table)
        self.member_view.set_dataframe(result.member_table)
        self.notes_view.set_dataframe(self._notes_frame(result))
        self.notes_banner.set_result_notes(result.warnings)

        self._update_plot()
        self._sync_run_label()
        scope_note = f", scope: {self.scope.summary()}" if self.scope.is_active() else ""
        self._say(
            f"Query completed: {subject_label}, {self.basis_combo.currentText()}, "
            f"{len(result.table):,} group(s){scope_note}."
        )

    @staticmethod
    def _notes_frame(result) -> pd.DataFrame:
        return pd.DataFrame(
            [{"kind": "warning", "text": w} for w in result.warnings]
            + [{"kind": "disclosure", "text": d} for d in result.disclosures]
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
            self._plot_source = ""
            self.plot.show_message(plan.refusal)
            return
        self._last_spec = plan.spec
        self._plot_source = plan.source
        render_plot(self.plot, plan.data, plan.spec)

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a bar, a cell or a box selects the row it was drawn from."""
        view = self.member_view if self._plot_source == SOURCE_MEMBER_TABLE else self.result_view
        frame = view.dataframe()
        if frame is None or frame.empty:
            return
        wanted = {str(mark.get(key)) for key in ("label", "row", "column", "series")
                  if mark.get(key) not in (None, "")}
        if not wanted:
            return
        text_columns = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        if not text_columns:
            return
        cells = frame[text_columns].astype(str)
        hits = cells.isin(wanted).sum(axis=1).to_numpy()
        if not hits.size or hits.max() == 0:
            return
        row = int(hits.argmax())
        self.tables.setCurrentWidget(view)
        view.selectRow(row)
        index = view.proxy_model().mapFromSource(view.model().index(row, 0))
        if index.isValid():
            view.table.scrollTo(index)

    def _export(self) -> None:
        if self._result is None or self.dataset is None:
            self._say("Query: run a question first; there is nothing to export yet.")
            return
        tables = {
            "query_result": self._result.table,
            "query_per_member": self._result.member_table,
            "query_notes": self._notes_frame(self._result),
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
        self._say(f"Query exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Query outputs exported:\n{out_dir}")
