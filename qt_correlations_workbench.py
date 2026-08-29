"""
qt_correlations_workbench.py — the "Association Workbench (kg)" Correlations
sub-tab: kg-only element association screening for vitrification/blending
(Jaccard co-presence, preferred-association scoring, 18 plot types, coherent
colors, skip-elements). Ported from the old app's SeabornCorrelationTab.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

import correlation_science as csci
import export_utils
import plot_helpers as ph
from data_model import HanfordDataset
from qt_help import show_workspace_help
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

METRICS = ["log10_plus1", "log10_inventory", "fraction", "inventory", "presence"]
METHODS = ["pearson", "spearman", "kendall"]
COLOR_MODES = ["Basic", "Coherent colors"]
BAR_MODES = ["Stacked", "Grouped"]
BAR_SORT_MODES = ["Total (desc)", "Total (asc)", "Tank ID (A-Z)"]
PLOT_TYPES = [
    "Corr heatmap lower triangle", "Corr heatmap + total kg projections", "Jaccard co-presence heatmap",
    "Top preferred associations", "Top positive correlations", "Top negative correlations", "Top Jaccard co-presence",
    "Pair matrix regression", "Pair matrix scatter", "Pair matrix KDE",
    "Joint regression first two", "Joint scatter first two", "Joint KDE first two",
    "Tank similarity heatmap", "Tank x element heatmap", "Tank inventory bars",
    "Presence pattern bars", "Stats dashboard",
]
_TABLE_KEY_BY_TITLE = {
    "Element stats": "element_stats", "Pair stats": "pair_stats", "Metric matrix": "metric_matrix",
    "Raw kg matrix": "raw_matrix", "Tank similarity": "tank_similarity", "Skipped": "excluded_elements",
}


class WorkbenchTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.results: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        top = QHBoxLayout()
        title = QLabel("Kg-only Association Workbench")
        title.setObjectName("SectionTitle")
        top.addWidget(title)
        note = QLabel("Element-to-element associations based on tank-level kg inventory only. Ci is deliberately excluded.")
        note.setObjectName("SectionNote")
        top.addWidget(note)
        top.addStretch(1)
        help_btn = QPushButton("?")
        help_btn.setObjectName("HelpButton")
        help_btn.setToolTip("What is this workspace for?")
        help_btn.clicked.connect(lambda: show_workspace_help(self, 10, "Correlations"))
        top.addWidget(help_btn)
        export_tables_btn = QPushButton("Export all tables")
        export_tables_btn.clicked.connect(self.export_tables)
        top.addWidget(export_tables_btn)
        export_suite_btn = QPushButton("Export full plot suite")
        export_suite_btn.clicked.connect(self.export_plot_suite)
        top.addWidget(export_suite_btn)
        root.addLayout(top)

        # Primary row: pick elements, build the data, pick a plot, draw it.
        # Everything else (selection mode, skip list, thresholds, metric/
        # method, colors, per-plot-type tuning) lives in Advanced options.
        primary = QHBoxLayout()
        primary.addWidget(QLabel("Elements"))
        self.elements_edit = QLineEdit("Cs, Sr, Tc, I, Se, U, Cr, Fe, Al, Na, P, B, Si")
        self.elements_edit.setMinimumWidth(260)
        self.elements_edit.setToolTip(
            "Comma-separated element list to build associations for. Ignored if Selection "
            "(Advanced options) is set to \"Top kg elements\"."
        )
        primary.addWidget(self.elements_edit)
        self.build_btn = QPushButton("Build kg correlation data")
        self.build_btn.setObjectName("Primary")
        self.build_btn.clicked.connect(self.build_data)
        primary.addWidget(self.build_btn)
        primary.addSpacing(20)
        primary.addWidget(QLabel("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(PLOT_TYPES)
        self.plot_type_combo.setCurrentText("Corr heatmap + total kg projections")
        primary.addWidget(self.plot_type_combo)
        plot_btn = QPushButton("Plot selected")
        plot_btn.clicked.connect(self.plot_selected)
        primary.addWidget(plot_btn)
        primary.addStretch(1)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        adv_row1 = QHBoxLayout()
        adv_row1.addWidget(QLabel("Selection"))
        self.selection_combo = QComboBox()
        self.selection_combo.addItems(["User list", "Top kg elements"])
        self.selection_combo.setToolTip("\"Top kg elements\" ignores the Elements field above and auto-picks the highest-inventory elements instead.")
        adv_row1.addWidget(self.selection_combo)
        adv_row1.addWidget(QLabel("Skip"))
        self.skip_edit = QLineEdit()
        self.skip_edit.setMaximumWidth(140)
        self.skip_edit.setToolTip("Elements to exclude even if otherwise selected.")
        adv_row1.addWidget(self.skip_edit)
        adv_row1.addWidget(QLabel("top kg"))
        self.top_kg_spin = QSpinBox()
        self.top_kg_spin.setRange(2, 500)
        self.top_kg_spin.setValue(25)
        adv_row1.addWidget(self.top_kg_spin)
        adv_row1.addWidget(QLabel("min kg row"))
        self.min_inv_spin = QDoubleSpinBox()
        self.min_inv_spin.setRange(0.0, 1e12)
        self.min_inv_spin.setDecimals(6)
        adv_row1.addWidget(self.min_inv_spin)
        adv_row1.addStretch(1)
        adv.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        adv_row2.addWidget(QLabel("Metric"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(METRICS)
        adv_row2.addWidget(self.metric_combo)
        adv_row2.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        adv_row2.addWidget(self.method_combo)
        self.include_zeros_check = QCheckBox("include zero tanks")
        self.include_zeros_check.setChecked(True)
        adv_row2.addWidget(self.include_zeros_check)
        self.annotate_check = QCheckBox("annotate heatmaps")
        adv_row2.addWidget(self.annotate_check)
        self.show_projections_check = QCheckBox("show total projections")
        self.show_projections_check.setChecked(True)
        adv_row2.addWidget(self.show_projections_check)
        adv_row2.addWidget(QLabel("Colors"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(COLOR_MODES)
        self.color_mode_combo.setCurrentText("Coherent colors")
        adv_row2.addWidget(self.color_mode_combo)
        adv_row2.addStretch(1)
        adv.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.addWidget(QLabel("max pair-matrix elements"))
        self.max_pair_spin = QSpinBox()
        self.max_pair_spin.setRange(2, 20)
        self.max_pair_spin.setValue(8)
        adv_row3.addWidget(self.max_pair_spin)
        adv_row3.addWidget(QLabel("top pairs"))
        self.top_pairs_spin = QSpinBox()
        self.top_pairs_spin.setRange(1, 500)
        self.top_pairs_spin.setValue(30)
        adv_row3.addWidget(self.top_pairs_spin)
        adv_row3.addWidget(QLabel("top tanks"))
        self.top_tanks_spin = QSpinBox()
        self.top_tanks_spin.setRange(2, 500)
        self.top_tanks_spin.setValue(50)
        adv_row3.addWidget(self.top_tanks_spin)
        hint = QLabel("For long element lists, pair matrices are automatically limited to avoid unusable figures.")
        hint.setObjectName("SectionNote")
        adv_row3.addWidget(hint)
        adv_row3.addStretch(1)
        adv.addLayout(adv_row3)

        adv_row4 = QHBoxLayout()
        adv_row4.addWidget(QLabel("Tank inventory bars:"))
        adv_row4.addWidget(QLabel("Mode"))
        self.bar_mode_combo = QComboBox()
        self.bar_mode_combo.addItems(BAR_MODES)
        self.bar_mode_combo.setToolTip(
            "Stacked: each tank's bar splits into one segment per selected element (total length = total "
            "inventory) -- the classic publication summary shape.\n"
            "Grouped: one bar per element per tank, side by side -- for comparing magnitudes directly "
            "instead of a cumulative total."
        )
        adv_row4.addWidget(self.bar_mode_combo)
        adv_row4.addWidget(QLabel("Sort tanks"))
        self.bar_sort_combo = QComboBox()
        self.bar_sort_combo.addItems(BAR_SORT_MODES)
        adv_row4.addWidget(self.bar_sort_combo)
        self.bar_log_check = QCheckBox("log scale")
        self.bar_log_check.setToolTip("Applies to Grouped mode; a log-scaled axis makes Stacked segment lengths non-additive.")
        adv_row4.addWidget(self.bar_log_check)
        bar_hint = QLabel("Uses \"top tanks\" above and the selected elements/color mode.")
        bar_hint.setObjectName("SectionNote")
        adv_row4.addWidget(bar_hint)
        adv_row4.addStretch(1)
        adv.addLayout(adv_row4)

        root.addWidget(self.advanced_section)

        splitter = QSplitter()
        root.addWidget(splitter, 1)

        self.tables = QTabWidget()
        self._table_views: dict[str, DataFrameTableView] = {}
        for title_ in ["Element stats", "Pair stats", "Metric matrix", "Raw kg matrix", "Tank similarity", "Skipped"]:
            view = DataFrameTableView(title=title_, max_rows_display=500)
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        splitter.addWidget(self.tables)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset  # kg-only by design; nothing else to sync from the dataset

    def _current_elements(self) -> list:
        raw_matrix = self.results.get("raw_matrix") if self.results else None
        if raw_matrix is not None and not raw_matrix.empty:
            return [c for c in raw_matrix.columns if c != "WasteSiteId"]
        return csci.parse_element_list(self.elements_edit.text())

    def build_data(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Association Workbench", "Load a dataset first.")
            return
        self.build_btn.setEnabled(False)
        self.build_btn.setText("Building…")
        QApplication.processEvents()
        try:
            self.results = csci.kg_correlation_workbench(
                self.dataset, elements_text=self.elements_edit.text(),
                selection_mode=self.selection_combo.currentText(), top_n_elements=self.top_kg_spin.value(),
                value_mode=self.metric_combo.currentText(), method=self.method_combo.currentText(),
                min_inventory=self.min_inv_spin.value(), include_zeros=self.include_zeros_check.isChecked(),
                skip_elements_text=self.skip_edit.text(),
            )
        except ValueError as exc:
            self.build_btn.setEnabled(True)
            self.build_btn.setText("Build kg correlation data")
            QMessageBox.warning(self, "Build failed", str(exc))
            return
        self.build_btn.setEnabled(True)
        self.build_btn.setText("Build kg correlation data")
        for title_, key in _TABLE_KEY_BY_TITLE.items():
            self._table_views[title_].set_dataframe(self.results.get(key, pd.DataFrame()))
        self.plot_selected()
        n_elements = len(self._current_elements())
        self.app_window.statusBar().showMessage(f"Kg correlations built for {n_elements} element(s).")

    def _ensure_results(self) -> bool:
        if not self.results:
            self.build_data()
        return bool(self.results)

    def plot_selected(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Association Workbench", "Load a dataset first.")
            return
        if not self._ensure_results():
            return
        ptype = self.plot_type_combo.currentText()
        elements = self._current_elements()
        corr = self.results.get("corr_matrix", pd.DataFrame())
        jacc = self.results.get("jaccard_matrix", pd.DataFrame())
        element_stats = self.results.get("element_stats", pd.DataFrame())
        pair_stats = self.results.get("pair_stats", pd.DataFrame())
        metric_matrix = self.results.get("metric_matrix", pd.DataFrame())
        raw_matrix = self.results.get("raw_matrix", pd.DataFrame())
        tank_similarity = self.results.get("tank_similarity", pd.DataFrame())
        presence_matrix = self.results.get("presence_matrix", pd.DataFrame())
        metric = self.metric_combo.currentText()
        show_proj = self.show_projections_check.isChecked()
        color_mode = self.color_mode_combo.currentText()
        annotate = self.annotate_check.isChecked()
        top_pairs = self.top_pairs_spin.value()
        top_tanks = self.top_tanks_spin.value()
        max_pair = self.max_pair_spin.value()

        if ptype == "Corr heatmap lower triangle":
            ph.plot_seaborn_lower_triangle_matrix(self.plot, corr, f"Element correlation heatmap - kg only - {metric}", "Correlation r", center=0, annotate=annotate, totals=element_stats, projections=False, color_mode=color_mode)
        elif ptype == "Corr heatmap + total kg projections":
            title = "Element correlation heatmap" + (" + total kg projections" if show_proj else " - projections hidden") + f" - {metric}"
            ph.plot_seaborn_lower_triangle_matrix(self.plot, corr, title, "Correlation r", center=0, annotate=annotate, totals=element_stats, projections=show_proj, color_mode=color_mode)
        elif ptype == "Jaccard co-presence heatmap":
            title = "Jaccard co-presence heatmap - kg only" + (" + total kg projections" if show_proj else " - projections hidden")
            ph.plot_seaborn_lower_triangle_matrix(self.plot, jacc, title, "Jaccard co-presence", center=None, annotate=annotate, totals=element_stats, projections=show_proj, color_mode=color_mode)
        elif ptype == "Top preferred associations":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="preferred", color_mode=color_mode)
        elif ptype == "Top positive correlations":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="positive", color_mode=color_mode)
        elif ptype == "Top negative correlations":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="negative", color_mode=color_mode)
        elif ptype == "Top Jaccard co-presence":
            ph.plot_seaborn_top_associations(self.plot, pair_stats, top_n=top_pairs, mode="jaccard", color_mode=color_mode)
        elif ptype == "Pair matrix regression":
            ph.plot_seaborn_pair_matrix(self.plot, metric_matrix, raw_matrix, elements, metric, kind="regression", max_elements=max_pair, color_mode=color_mode)
        elif ptype == "Pair matrix scatter":
            ph.plot_seaborn_pair_matrix(self.plot, metric_matrix, raw_matrix, elements, metric, kind="scatter", max_elements=max_pair, color_mode=color_mode)
        elif ptype == "Pair matrix KDE":
            ph.plot_seaborn_pair_matrix(self.plot, metric_matrix, raw_matrix, elements, metric, kind="kde", max_elements=max_pair, color_mode=color_mode)
        elif ptype == "Joint regression first two":
            ph.plot_seaborn_joint_first_two(self.plot, metric_matrix, elements, metric, kind="regression", color_mode=color_mode)
        elif ptype == "Joint scatter first two":
            ph.plot_seaborn_joint_first_two(self.plot, metric_matrix, elements, metric, kind="scatter", color_mode=color_mode)
        elif ptype == "Joint KDE first two":
            ph.plot_seaborn_joint_first_two(self.plot, metric_matrix, elements, metric, kind="kde", color_mode=color_mode)
        elif ptype == "Tank similarity heatmap":
            ph.plot_seaborn_tank_similarity(self.plot, tank_similarity, raw_matrix, top_tanks=top_tanks, annotate=annotate, color_mode=color_mode)
        elif ptype == "Tank x element heatmap":
            ph.plot_seaborn_tank_element_map(self.plot, raw_matrix, elements, top_tanks=top_tanks, metric=metric, color_mode=color_mode)
        elif ptype == "Tank inventory bars":
            ph.plot_tank_element_bars(
                self.plot, raw_matrix, elements, top_n=top_tanks, mode=self.bar_mode_combo.currentText(),
                sort_by=self.bar_sort_combo.currentText(), log_scale=self.bar_log_check.isChecked(), color_mode=color_mode,
            )
        elif ptype == "Presence pattern bars":
            ph.plot_seaborn_presence_patterns(self.plot, presence_matrix, elements, top_n=top_pairs, color_mode=color_mode)
        elif ptype == "Stats dashboard":
            ph.plot_seaborn_stats_dashboard(self.plot, element_stats, pair_stats, top_n=top_pairs, color_mode=color_mode)
        else:
            self.plot.show_message(f"Unknown plot type: {ptype}")

    def export_tables(self) -> None:
        if not self.results or self.dataset is None:
            QMessageBox.information(self, "Export", "Build the kg correlation data first.")
            return
        tables = dict(self.results)
        settings = pd.DataFrame([{
            "unit": "kg", "selection_mode": self.selection_combo.currentText(),
            "elements_text": self.elements_edit.text(), "skip_elements_text": self.skip_edit.text(),
            "metric": self.metric_combo.currentText(), "method": self.method_combo.currentText(),
            "top_elements": self.top_kg_spin.value(), "min_inventory_kg_row": self.min_inv_spin.value(),
            "include_zeros": self.include_zeros_check.isChecked(),
            "show_total_projections": self.show_projections_check.isChecked(),
            "color_mode": self.color_mode_combo.currentText(),
        }])
        tables["settings"] = settings
        out_dir = export_utils.export_named_tables(self.dataset, "seaborn_correlations", tables)
        try:
            self.plot.figure.savefig(out_dir / "current_seaborn_plot.png", bbox_inches="tight", dpi=220)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Association workbench tables exported:\n{out_dir}")

    def export_plot_suite(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Association Workbench", "Load a dataset first.")
            return
        if not self._ensure_results():
            return
        tables = dict(self.results)
        out_dir = export_utils.export_named_tables(self.dataset, "seaborn_plot_suite", tables)
        previous = self.plot_type_combo.currentText()
        manifest_rows = []
        for ptype in PLOT_TYPES:
            self.plot_type_combo.setCurrentText(ptype)
            try:
                self.plot_selected()
                filename = f"{export_utils.safe_name(ptype)}.png"
                self.plot.figure.savefig(out_dir / filename, bbox_inches="tight", dpi=220)
                manifest_rows.append({"plot_type": ptype, "status": "ok", "file": filename})
            except Exception as exc:
                manifest_rows.append({"plot_type": ptype, "status": "failed", "error": repr(exc), "file": ""})
        self.plot_type_combo.setCurrentText(previous)
        self.plot_selected()
        pd.DataFrame(manifest_rows).to_csv(out_dir / "plot_suite_manifest.csv", index=False)
        self.app_window.statusBar().showMessage(f"Plot suite exported: {out_dir}")
        QMessageBox.information(self, "Export complete", f"Plot suite exported:\n{out_dir}")
