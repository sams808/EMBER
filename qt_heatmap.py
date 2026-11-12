"""
qt_heatmap.py — Heatmaps workspace: tank x element inventory matrices.

The matrix comes from the query engine (matrix_query.engine_matrix), so an
element receives its stoichiometric share of every species reporting it
instead of the whole mass of every analyte that happens to name it first.
That changes real numbers -- on the shipped dataset nitrogen falls to
0.24x and oxygen rises 15.6x, from rank 6 to rank 1 -- so the older
attribution stays selectable, and the subtitle of the figure says which
one produced it.

Rendering goes through the L2 plot grammar, which is what gives this
workspace the global scope (it honors on_scope_changed), the guardrail
and disclosure behavior, and the hover tooltips of the cells.

What the page owns:
- The build runs on the house worker (about a second on the shipped
  dataset): the settings are read from the widgets before it is
  submitted, the worker function touches no widget, and a result that
  arrives after a newer request or a dataset change is discarded.
- The scale (log10, inventory, fraction of the tank's total) is a display
  transform: changing it redraws the standing matrix without a new build.
- A click on a cell selects the matching row of the Long table; a double
  click on a cell or on a table row opens the tank.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

import export_utils
import matrix_query as mq
import matrix_science as msci
from data_model import HanfordDataset
from plot_grammar import PlotSpec, render_plot
from qt_components import NotesBanner, PageHeader, control_row, field_label, secondary_button
from qt_explorer import plain_note
from qt_help import make_help_button
from qt_scope import GlobalScope
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget
from qt_worker import run_in_thread

PAGE_TITLE = "Heatmaps"
PAGE_SUBTITLE = "Every tank against the most abundant elements, as one matrix of inventories."
EMPTY_TITLE = "No result yet"
EMPTY_HINT = "Choose a basis and an attribution, then press Run to build the tank by element matrix."

# Display mode -> (log_x, normalize). The engine supplies absolute values;
# these are presentation transforms, applied by the grammar. The keys are
# what an export records; MODE_LABELS is what the combo shows.
MODES = {
    "log10_inventory": (True, "none"),
    "inventory": (False, "none"),
    "fraction": (False, "row"),
}
MODE_LABELS: Dict[str, str] = {
    "log10_inventory": "Log10 of inventory",
    "inventory": "Inventory",
    "fraction": "Fraction of the tank's total",
}
MODE_TOOLTIPS: Dict[str, str] = {
    "log10_inventory": "Colour follows the logarithm of the inventory: small and large tanks both stay readable.",
    "inventory": "Colour follows the inventory itself: the largest cells dominate.",
    "fraction": "Each tank's row divided by its own total: composition, whatever the size of the tank.",
}

# Appended to the result of an apportioned build asked with a minimum
# inventory, which only the as-reported attribution can apply.
MIN_INVENTORY_NOTE = (
    "min inventory is a row-level filter of the as-reported "
    "attribution and is not applied to apportioned values."
)


@dataclass
class _Outcome:
    """What the worker hands back, tagged with the run it belongs to."""

    token: int
    params: dict = field(default_factory=dict)
    result: Optional[mq.MatrixResult] = None
    refusal: str = ""
    error: str = ""


def compute_matrix(token: int, dataset: HanfordDataset, params: dict,
                   scope: Dict[str, List[str]]) -> _Outcome:
    """Pure computation: runs on the worker thread, touches no widget."""
    unit, top_n = params["basis"], params["top_n_elements"]
    min_inv, max_tanks = params["min_inventory"], params["max_tanks"]
    try:
        tank_subset = None
        if max_tanks > 0:
            tank_subset = msci.top_tanks_by_inventory(
                mq._scoped_copy(dataset, scope), unit, max_tanks)
        if params["attribution"] == mq.APPORTIONED:
            result = mq.engine_matrix(
                dataset, basis=unit, top_n_elements=top_n,
                scope=scope, tank_subset=tank_subset,
            )
            if min_inv > 0:
                result.warnings.append(MIN_INVENTORY_NOTE)
        else:
            result = mq.legacy_matrix(
                dataset, unit=unit, top_n_elements=top_n,
                min_inventory=min_inv, scope=scope, tank_subset=tank_subset,
            )
    except ValueError as exc:
        return _Outcome(token=token, params=params, refusal=str(exc))
    except Exception:
        return _Outcome(token=token, params=params, error=traceback.format_exc())
    return _Outcome(token=token, params=params, result=result)


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
        # only the Scale combo triggers a redraw, so a Basis or Attribution
        # changed afterwards would otherwise relabel data it never touched
        # (review finding -- a figure titled "Ci" over kg data).
        self._params: dict = {}
        self._result_inputs: Optional[tuple] = None
        self._has_built = False
        self._run_token = 0
        self._running = False
        self._build_ui()

    @property
    def _result_scope(self) -> str:
        return self._params.get("scope", "")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader(PAGE_TITLE, PAGE_SUBTITLE)
        self.export_btn = secondary_button(
            "Export...", "Write the long and wide matrices, the settings, the notes and the chart to a folder.")
        self.export_btn.clicked.connect(self._export_inputs)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 9, "Heatmaps"))
        root.addWidget(self.header)

        controls, row = control_row()
        row.addWidget(field_label("Basis"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kg", "Ci"])
        self.unit_combo.setToolTip("kg keeps the rows reported as mass; Ci keeps the rows reported as activity.")
        self.unit_combo.currentTextChanged.connect(self._sync_run_label)
        row.addWidget(self.unit_combo)

        row.addSpacing(8)
        row.addWidget(field_label("Attribution"))
        self.attribution_combo = QComboBox()
        self.attribution_combo.addItems(mq.ATTRIBUTIONS)
        self.attribution_combo.setToolTip(
            "Apportioned: an element receives its mass fraction of every species\n"
            "reporting it (S gets 32.06/96.06 of SO4).\n"
            "As reported: each analyte's full mass goes to its first parsed element,\n"
            "the older behavior, kept for reproducing earlier figures."
        )
        self.attribution_combo.currentTextChanged.connect(self._sync_run_label)
        row.addWidget(self.attribution_combo)

        row.addSpacing(8)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        row.addWidget(self.run_btn)
        row.addStretch(1)

        row.addWidget(field_label("Scale"))
        self.mode_combo = QComboBox()
        for index, key in enumerate(MODES):
            self.mode_combo.addItem(MODE_LABELS[key], key)
            self.mode_combo.setItemData(index, MODE_TOOLTIPS[key], Qt.ToolTipRole)
        self.mode_combo.setToolTip(
            "How the cells are coloured. Changing it redraws the matrix on screen; nothing is recomputed.\n"
            + "\n".join(f"{MODE_LABELS[k]}: {MODE_TOOLTIPS[k]}" for k in MODES))
        self.mode_combo.currentIndexChanged.connect(lambda _i: self._update_plot())
        row.addWidget(self.mode_combo)
        root.addWidget(controls)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv_controls, adv_row = control_row()
        adv_row.addWidget(field_label("Elements shown"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 120)
        self.top_n_spin.setValue(35)
        self.top_n_spin.setToolTip("How many of the most abundant elements become columns of the matrix.")
        self.top_n_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.top_n_spin)
        adv_row.addSpacing(8)
        adv_row.addWidget(field_label("Minimum inventory"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        self.min_inv_spin.setToolTip(
            "As-reported attribution only: reported rows at or below this inventory, in the unit of\n"
            "the basis, are dropped before summing. Apportioned values are not filtered."
        )
        self.min_inv_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.min_inv_spin)
        adv_row.addSpacing(8)
        adv_row.addWidget(field_label("Largest tanks only"))
        self.max_tanks_spin = QSpinBox()
        self.max_tanks_spin.setRange(0, 1000)
        self.max_tanks_spin.setSpecialValueText("All tanks")
        self.max_tanks_spin.setToolTip(
            "Keep only this many tanks, the largest by reported inventory on the basis. 0 keeps every tank.")
        self.max_tanks_spin.valueChanged.connect(self._sync_run_label)
        adv_row.addWidget(self.max_tanks_spin)
        adv_row.addStretch(1)
        self.advanced_section.content_layout().addWidget(adv_controls)
        root.addWidget(self.advanced_section)

        # Engine warnings are data-integrity statements (exclusions, trims,
        # unassigned uncertainty): always on the page with the result, as
        # one line, the full list one click away.
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
        self.long_view = DataFrameTableView(title="Heatmap long input")
        self.wide_view = DataFrameTableView(title="Heatmap wide matrix", max_rows_display=250)
        self.notes_view = DataFrameTableView(title="Notes", max_rows_display=200)
        self.long_view.set_empty_text(EMPTY_TITLE, "One row per tank and element appears after a run.")
        self.wide_view.set_empty_text(EMPTY_TITLE, "One row per tank and one column per element appear after a run.")
        self.notes_view.set_empty_text("No notes", "What qualifies a matrix is listed here after a run.")
        self.tables.addTab(self.long_view, "Long")
        self.tables.addTab(self.wide_view, "Wide")
        self.tables.addTab(self.notes_view, "Notes")
        self.tables.setTabToolTip(0, "One row per tank and element: what a cell of the matrix is drawn from.")
        self.tables.setTabToolTip(1, "The matrix itself: tanks in rows, elements in columns.")
        self.tables.setTabToolTip(2, "What was set aside, converted or left out, and how each element was sourced.")
        splitter.addWidget(self.tables)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 440])
        self.splitter = splitter

        self._sync_run_label()
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def mode_key(self) -> str:
        """The display mode as MODES and the export name it."""
        key = self.mode_combo.currentData()
        return key if key in MODES else next(iter(MODES))

    def set_mode(self, key: str) -> None:
        index = self.mode_combo.findData(key)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

    def _read_inputs(self) -> tuple:
        return (
            self.unit_combo.currentText(), self.attribution_combo.currentText(),
            self.top_n_spin.value(), self.min_inv_spin.value(), self.max_tanks_spin.value(),
        )

    def _sync_run_label(self, *_args) -> None:
        """"Run" builds a matrix; "Update" rebuilds after a control changed under a standing one."""
        if not hasattr(self, "run_btn") or self._running:
            return
        changed = self._has_built and self._result_inputs != self._read_inputs()
        self.run_btn.setText("Update" if changed else "Run")
        self.run_btn.setToolTip(
            "The controls differ from the matrix on screen: run again to bring it up to date." if changed
            else "Build the tank by element matrix the controls describe.")

    def _say(self, message: str) -> None:
        self.app_window.statusBar().showMessage(message)

    # ------------------------------------------------------------------
    # Dataset / scope protocol
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        # A matrix of the previous dataset must not stay on screen, and a
        # build still in flight on it must not land.
        self._run_token += 1
        self._set_running(False)
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
        self._result_inputs = None
        self._has_built = False
        for view in (self.long_view, self.wide_view, self.notes_view):
            view.set_dataframe(pd.DataFrame())
        self.notes_banner.set_notes([])
        self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
        self._sync_run_label()

    def on_scope_changed(self) -> None:
        """Rebuild under the new scope once the user has asked for a
        matrix at all. Guarding on "the last build returned rows" instead
        would strand the workspace: a scope that emptied the matrix would
        stop every later scope change from rebuilding it (review
        finding)."""
        if self._has_built:
            self.run()

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            self._say("Heatmaps: no dataset is loaded yet.")
            return
        params = {
            "basis": self.unit_combo.currentText(),
            "attribution": self.attribution_combo.currentText(),
            "top_n_elements": self.top_n_spin.value(),
            "min_inventory": self.min_inv_spin.value(),
            "max_tanks": self.max_tanks_spin.value(),
            "scope": self.scope.summary(),
        }
        inputs = self._read_inputs()
        self._run_token += 1
        token = self._run_token
        dataset = self.dataset
        scope = self.scope.as_scope()
        self._set_running(True)
        run_in_thread(
            lambda: (compute_matrix(token, dataset, params, scope), inputs),
            self._on_run_done, self._on_run_error,
        )

    def _set_running(self, running: bool) -> None:
        was_running = self._running
        self._running = running
        self.run_btn.setEnabled(not running)
        if running:
            self.run_btn.setText("Running…")
        else:
            self._sync_run_label()
        if running or was_running:
            set_busy = getattr(self.app_window, "set_busy", None)
            if callable(set_busy):
                set_busy("Building the matrix…" if running else None)

    def _on_run_error(self, tb_text: str) -> None:
        self._set_running(False)
        self._say("Heatmaps: the build failed.")
        QMessageBox.critical(self, "Heatmaps failed", tb_text)

    def _on_run_done(self, delivered: tuple) -> None:
        outcome, inputs = delivered
        if outcome.token != self._run_token:
            return          # superseded by a newer request, or by a dataset change
        self._set_running(False)
        if outcome.error:
            self._say("Heatmaps: the build failed.")
            QMessageBox.critical(self, "Heatmaps failed", outcome.error)
            return
        if outcome.refusal:
            # The matrix cannot be built as asked: say why, and leave the
            # previous one standing under its own labels.
            self._say(f"Heatmaps: {outcome.refusal}")
            QMessageBox.warning(self, PAGE_TITLE, outcome.refusal)
            return

        result = outcome.result
        self._result = result
        self._has_built = True
        self._params = dict(outcome.params)
        self._result_inputs = inputs
        self.long_view.set_dataframe(result.long_df)
        self.wide_view.set_dataframe(result.wide_df)
        self.notes_view.set_dataframe(pd.DataFrame(
            [{"kind": "warning", "text": w} for w in result.warnings]
            + [{"kind": "disclosure", "text": d} for d in result.disclosures]
        ))
        self.notes_banner.set_result_notes([plain_note(w) for w in result.warnings], "this matrix")

        self._update_plot()
        self._sync_run_label()
        scope_note = f", scope: {self._result_scope}" if self._result_scope else ""
        self._say(
            f"Heatmap built: {self._params['basis']}, {len(result.elements)} elements, "
            f"{len(result.wide_df)} tanks{scope_note}."
        )

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        result = self._result
        if result.long_df.empty:
            # Never leave the previous build's figure standing beside
            # empty tables -- an unscoped chart under a scope that
            # matched nothing (review finding).
            if self._has_built:
                self.plot.show_empty("Nothing to show", "No inventory matches these settings and the current scope.")
            else:
                self.plot.show_empty(EMPTY_TITLE, EMPTY_HINT)
            return
        log_x, normalize = MODES[self.mode_key()]
        # Labels come from the settings that PRODUCED this result, never
        # from the live widgets.
        attribution = self._params.get("attribution", "")
        subtitle_parts = [p for p in [attribution] if p]
        if self._result_scope:
            subtitle_parts.append(f"Scope: {self._result_scope}")
        # Trims belong on the figure, not only in the banner: the exported
        # PNG is what ends up in a report (review finding).
        subtitle_parts.extend(w for w in result.warnings if "omitted" in w)
        spec = PlotSpec(
            mark="heatmap", x=result.value_column, y="WasteSiteId", hue="Element",
            title=f"Inventory by tank and element ({self._params.get('basis', '')})",
            subtitle=" · ".join(subtitle_parts),
            log_x=log_x, normalize=normalize,
        )
        render_plot(self.plot, result.long_df, spec)

    def _on_mark_clicked(self, mark: dict) -> None:
        """A click on a cell selects the row of the Long table it was drawn from."""
        tank, element = mark.get("row"), mark.get("column")
        frame = self.long_view.dataframe()
        if tank in (None, "") or element in (None, "") or frame is None or frame.empty:
            return
        if "WasteSiteId" not in frame.columns or "Element" not in frame.columns:
            return
        hits = (frame["WasteSiteId"].astype(str) == str(tank)) & (frame["Element"].astype(str) == str(element))
        if not hits.any():
            self._say(f"Heatmaps: {tank} has no {element} row on this basis; the cell is blank.")
            return
        row = int(hits.to_numpy().argmax())
        self.tables.setCurrentWidget(self.long_view)
        self.long_view.selectRow(row)
        index = self.long_view.proxy_model().mapFromSource(self.long_view.model().index(row, 0))
        if index.isValid():
            self.long_view.table.scrollTo(index)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_inputs(self) -> None:
        if self._result.wide_df.empty and self._result.long_df.empty:
            self._say("Heatmaps: run first; there is nothing to export yet.")
            return
        # Every value here describes the BUILD, not the current widget
        # state -- an export that labels tables with settings that did not
        # produce them is a self-consistent lie on disk (review finding).
        # Mode is the exception: it is a display transform, wired to a
        # redraw, so the live value is the one the saved PNG shows.
        params = pd.DataFrame(
            [{"parameter": "mode", "value": self.mode_key()}]
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
        self._say(f"Heatmaps exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Heatmap inputs exported:\n{out_dir}")
