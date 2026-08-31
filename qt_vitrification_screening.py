"""
qt_vitrification_screening.py — the Vitrification "Screening", "Candidate
Search", and "Blend Partners" sub-tabs. Ported from the old app's single
VitrificationTab, split into three tabs to match the nav shell's other
multi-tab domains (Correlations). The old app's hardcoded score weights
are now QDoubleSpinBox controls (with "Reset to defaults") instead of
buried constants -- old constants are vitrification_science's *_WEIGHT_DEFAULTS,
used to seed every spin box. Edited weights persist across restarts via
QSettings (settings_prefix keeps the three tabs' weights independent).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import plot_helpers as ph
import vitrification_science as vsci
from correlation_science import parse_element_list
from data_model import HanfordDataset
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

BASIS_OPTIONS = ["elemental", "oxide"]
SCREENING_PLOT_TYPES = ["Score breakdown (top tanks)", "Screening map (kg vs Ci)"]


def _build_weight_spins(layout: QHBoxLayout, defaults: Dict[str, float], settings_prefix: str) -> Dict[str, QDoubleSpinBox]:
    from PySide6.QtCore import QSettings
    from qt_help import APP_NAME
    settings = QSettings(APP_NAME, APP_NAME)
    spins: Dict[str, QDoubleSpinBox] = {}
    for key, default in defaults.items():
        label = key.replace("_weight", "").replace("_", " ")
        layout.addWidget(QLabel(f"{label}"))
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setDecimals(3)
        settings_key = f"{settings_prefix}_{key}"
        spin.setValue(settings.value(settings_key, default, type=float))
        spin.valueChanged.connect(lambda value, k=settings_key: QSettings(APP_NAME, APP_NAME).setValue(k, value))
        spins[key] = spin
        layout.addWidget(spin)
    return spins


def _reset_weight_spins(spins: Dict[str, QDoubleSpinBox], defaults: Dict[str, float]) -> None:
    for key, spin in spins.items():
        spin.setValue(defaults[key])  # also persists the default via the valueChanged connection above


def _current_weights(spins: Dict[str, QDoubleSpinBox]) -> Dict[str, float]:
    return {key: spin.value() for key, spin in spins.items()}


_SCORE_TERMS = (
    ("glass_former_weight", "glass_former"),
    ("modifier_weight", "modifier"),
    ("problem_weight", "problem"),
    ("volatile_weight", "volatile"),
    ("redox_weight", "redox"),
)


def _format_score_formula(weights: Dict[str, float]) -> str:
    """The score written with each term's own sign.

    The readout used to hardcode a minus in front of abs() for the
    problem, volatile and redox terms, while the scoring code applies the
    raw signed weight. The defaults are negative so it read correctly at
    the defaults -- but the spin boxes accept -1000..1000, and +45 and
    -45 produced the byte-identical string. The one human-readable
    statement of the formula could not distinguish a penalty from a
    bonus, and _disclose_custom_weights puts it on screen precisely when
    a weight has been changed (review finding).
    """
    parts: List[str] = []
    for key, name in _SCORE_TERMS:
        value = float(weights.get(key, 0.0))
        sign = "-" if value < 0 else "+"
        parts.append(f"{sign} {abs(value):.3g}*{name}")
    body = " ".join(parts)
    if body.startswith("+ "):
        body = body[2:]
    return f"score = {body} (clipped to [-100, 100])"


def _disclose_custom_weights(section, spins: Dict[str, QDoubleSpinBox], defaults: Dict[str, float]) -> None:
    """QSettings-persisted weight edits survive restarts -- if any differ
    from the documented defaults, the collapsed Advanced section would
    silently apply a non-default score. Auto-expand it so the weights (and
    the formula readout) are on screen before anyone builds a summary."""
    if any(spins[key].value() != default for key, default in defaults.items()):
        section.set_expanded(True)


class ScreeningTab(QWidget):
    # Emits a WasteSiteId when the user double-clicks a row in the
    # Screening summary table -- lets VitrificationPage jump straight to
    # Blend Partners with that tank pre-filled, instead of making the user
    # re-find and re-type the same tank ID in a different tab.
    jump_to_blend_partners = Signal(str)

    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.summary_df = pd.DataFrame()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        top = QHBoxLayout()
        title = QLabel("Vitrification Screening")
        title.setObjectName("SectionTitle")
        top.addWidget(title)
        note = QLabel("Screening metric only; not an official waste classification or glass formulation model.")
        note.setObjectName("SectionNote")
        top.addWidget(note)
        top.addStretch(1)
        top.addWidget(make_help_button(self, 12, "Vitrification"))
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_tables)
        top.addWidget(export_btn)
        root.addLayout(top)

        primary = QHBoxLayout()
        primary.addWidget(QLabel("Basis"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(BASIS_OPTIONS)
        self.basis_combo.setToolTip("Score tanks using elemental inventory or oxide-converted inventory.")
        primary.addWidget(self.basis_combo)
        build_btn = QPushButton("Build screening summary")
        build_btn.setObjectName("Primary")
        build_btn.clicked.connect(self.run_summary)
        primary.addWidget(build_btn)
        primary.addWidget(QLabel("Plot"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(SCREENING_PLOT_TYPES)
        self.plot_type_combo.setToolTip(
            "Score breakdown: which weighted terms drove each tank's score.\n"
            "Screening map: every tank plotted by chemical vs radiological inventory."
        )
        self.plot_type_combo.currentTextChanged.connect(lambda _: self._update_plot())
        primary.addWidget(self.plot_type_combo)
        primary.addStretch(1)
        root.addLayout(primary)

        # Score weights are tunable defaults, not something a first-time
        # user needs to touch -- collapsed by default, with a live formula
        # readout so the current weighting is never hidden, just the knobs.
        self.advanced_section = CollapsibleSection("Advanced options (score weights)")
        adv = self.advanced_section.content_layout()

        weights_row = QHBoxLayout()
        self.weight_spins = _build_weight_spins(weights_row, vsci.SCREENING_WEIGHT_DEFAULTS, "screening")
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(lambda: _reset_weight_spins(self.weight_spins, vsci.SCREENING_WEIGHT_DEFAULTS))
        weights_row.addWidget(reset_btn)
        weights_row.addStretch(1)
        adv.addLayout(weights_row)

        self.formula_label = QLabel("")
        self.formula_label.setObjectName("SectionNote")
        self.formula_label.setWordWrap(True)
        adv.addWidget(self.formula_label)
        for spin in self.weight_spins.values():
            spin.valueChanged.connect(self._update_formula_label)
        self._update_formula_label()

        root.addWidget(self.advanced_section)
        _disclose_custom_weights(self.advanced_section, self.weight_spins, vsci.SCREENING_WEIGHT_DEFAULTS)

        hint = QLabel("Double-click a tank in Screening summary to find blend partners for it.")
        hint.setObjectName("SectionNote")
        root.addWidget(hint)

        splitter = QSplitter()
        root.addWidget(splitter, 1)
        self.tables = QTabWidget()
        self._table_views: Dict[str, DataFrameTableView] = {}
        for title_ in ["Screening summary", "Groups", "Waste notes"]:
            view = DataFrameTableView(title=title_, max_rows_display=500)
            self._table_views[title_] = view
            self.tables.addTab(view, title_)
        self._table_views["Screening summary"].doubleClicked.connect(self._on_summary_row_double_clicked)
        splitter.addWidget(self.tables)
        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        self._table_views["Groups"].set_dataframe(vsci.vitrification_group_definitions())
        self._table_views["Waste notes"].set_dataframe(vsci.waste_class_notes())

    def _update_formula_label(self) -> None:
        self.formula_label.setText(_format_score_formula(_current_weights(self.weight_spins)))

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset

    def run_summary(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Screening", "Load a dataset first.")
            return
        self.summary_df = vsci.tank_category_summary(
            self.dataset, weights=_current_weights(self.weight_spins), basis=self.basis_combo.currentText(),
        )
        self._table_views["Screening summary"].set_dataframe(self.summary_df)
        self._update_plot()
        self.app_window.statusBar().showMessage(f"Vitrification summary built for {len(self.summary_df):,} tank(s).")

    def _update_plot(self) -> None:
        if self.plot_type_combo.currentText() == "Screening map (kg vs Ci)":
            ph.plot_vitrification_burden(self.plot, self.summary_df)
        else:
            ph.plot_vitrification_score_breakdown(self.plot, self.summary_df)

    def _on_summary_row_double_clicked(self, index) -> None:
        df = self._table_views["Screening summary"].dataframe()
        if df.empty or not (0 <= index.row() < len(df)) or "WasteSiteId" not in df.columns:
            return
        tank_id = str(df.iloc[index.row()]["WasteSiteId"])
        self.jump_to_blend_partners.emit(tank_id)

    def export_tables(self) -> None:
        if self.summary_df.empty or self.dataset is None:
            QMessageBox.information(self, "Export", "Build the screening summary first.")
            return
        tables = {
            "tank_screening_summary": self.summary_df,
            "group_definitions": vsci.vitrification_group_definitions(),
            "waste_class_notes": vsci.waste_class_notes(),
        }
        out_dir = export_utils.export_named_tables(self.dataset, "vitrification_screening", tables)
        try:
            self.plot.figure.savefig(out_dir / "screening_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Screening outputs exported:\n{out_dir}")


class CandidateSearchTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.candidates_df = pd.DataFrame()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        top = QHBoxLayout()
        title = QLabel("Candidate Search")
        title.setObjectName("SectionTitle")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(make_help_button(self, 12, "Vitrification"))
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_tables)
        top.addWidget(export_btn)
        root.addLayout(top)

        primary = QHBoxLayout()
        primary.addWidget(QLabel("Target elements"))
        self.target_edit = QLineEdit("Cs, Sr, Tc, I")
        self.target_edit.setToolTip("Elements a good candidate tank should be rich in.")
        primary.addWidget(self.target_edit)
        run_btn = QPushButton("Rank vitrification candidates")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run_candidates)
        primary.addWidget(run_btn)
        primary.addStretch(1)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        adv_row1 = QHBoxLayout()
        adv_row1.addWidget(QLabel("Penalty elements"))
        self.penalty_edit = QLineEdit("S, Cl, F, I, Cr, Mo, Tc")
        self.penalty_edit.setToolTip("Elements that count against a tank's score if present.")
        adv_row1.addWidget(self.penalty_edit)
        adv_row1.addWidget(QLabel("Required"))
        self.required_edit = QLineEdit("")
        self.required_edit.setMaximumWidth(120)
        self.required_edit.setToolTip("Elements a tank must contain to be included at all.")
        adv_row1.addWidget(self.required_edit)
        adv_row1.addWidget(QLabel("Basis"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(BASIS_OPTIONS)
        adv_row1.addWidget(self.basis_combo)
        adv_row1.addWidget(QLabel("min kg"))
        self.min_kg_spin = QDoubleSpinBox()
        self.min_kg_spin.setRange(0.0, 1e12)
        self.min_kg_spin.setDecimals(3)
        adv_row1.addWidget(self.min_kg_spin)
        adv_row1.addWidget(QLabel("top N"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 500)
        self.top_n_spin.setValue(40)
        adv_row1.addWidget(self.top_n_spin)
        adv_row1.addStretch(1)
        adv.addLayout(adv_row1)

        weights_row = QHBoxLayout()
        self.weight_spins = _build_weight_spins(weights_row, vsci.CANDIDATE_WEIGHT_DEFAULTS, "candidate")
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(lambda: _reset_weight_spins(self.weight_spins, vsci.CANDIDATE_WEIGHT_DEFAULTS))
        weights_row.addWidget(reset_btn)
        weights_row.addStretch(1)
        adv.addLayout(weights_row)

        root.addWidget(self.advanced_section)
        _disclose_custom_weights(self.advanced_section, self.weight_spins, vsci.CANDIDATE_WEIGHT_DEFAULTS)

        splitter = QSplitter()
        root.addWidget(splitter, 1)
        self.table = DataFrameTableView(title="Candidate ranking", max_rows_display=500)
        splitter.addWidget(self.table)
        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset

    def run_candidates(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Candidate Search", "Load a dataset first.")
            return
        self.candidates_df = vsci.vitrification_candidate_search(
            self.dataset, parse_element_list(self.target_edit.text()),
            parse_element_list(self.penalty_edit.text()), parse_element_list(self.required_edit.text()),
            min_total_kg=self.min_kg_spin.value(), top_n=self.top_n_spin.value(),
            weights=_current_weights(self.weight_spins), basis=self.basis_combo.currentText(),
        )
        self.table.set_dataframe(self.candidates_df)
        ph.plot_candidate_scores(self.plot, self.candidates_df, top_n=self.top_n_spin.value())
        self.app_window.statusBar().showMessage(f"Candidate ranking built: {len(self.candidates_df):,} tank(s).")

    def export_tables(self) -> None:
        if self.candidates_df.empty or self.dataset is None:
            QMessageBox.information(self, "Export", "Run a candidate search first.")
            return
        out_dir = export_utils.export_named_tables(self.dataset, "vitrification_candidates", {"candidate_ranking": self.candidates_df})
        try:
            self.plot.figure.savefig(out_dir / "candidate_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Candidate ranking exported:\n{out_dir}")


class BlendPartnersTab(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self.blend_df = pd.DataFrame()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        top = QHBoxLayout()
        title = QLabel("Blend Partners")
        title.setObjectName("SectionTitle")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(make_help_button(self, 12, "Vitrification"))
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_tables)
        top.addWidget(export_btn)
        root.addLayout(top)

        primary = QHBoxLayout()
        primary.addWidget(QLabel("Base tank"))
        self.base_tank_combo = QComboBox()
        self.base_tank_combo.setEditable(True)
        self.base_tank_combo.setToolTip("Find tanks that best complement this tank's composition.")
        primary.addWidget(self.base_tank_combo)
        run_btn = QPushButton("Find blend partners")
        run_btn.setObjectName("Primary")
        run_btn.clicked.connect(self.run_blend)
        primary.addWidget(run_btn)
        primary.addStretch(1)
        root.addLayout(primary)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()

        adv_row1 = QHBoxLayout()
        adv_row1.addWidget(QLabel("Basis"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(BASIS_OPTIONS)
        adv_row1.addWidget(self.basis_combo)
        adv_row1.addWidget(QLabel("top N"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 500)
        self.top_n_spin.setValue(40)
        adv_row1.addWidget(self.top_n_spin)
        adv_row1.addStretch(1)
        adv.addLayout(adv_row1)

        weights_row = QHBoxLayout()
        self.weight_spins = _build_weight_spins(weights_row, vsci.BLEND_WEIGHT_DEFAULTS, "blend")
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(lambda: _reset_weight_spins(self.weight_spins, vsci.BLEND_WEIGHT_DEFAULTS))
        weights_row.addWidget(reset_btn)
        weights_row.addStretch(1)
        adv.addLayout(weights_row)

        root.addWidget(self.advanced_section)
        _disclose_custom_weights(self.advanced_section, self.weight_spins, vsci.BLEND_WEIGHT_DEFAULTS)

        splitter = QSplitter()
        root.addWidget(splitter, 1)
        self.table = DataFrameTableView(title="Blend partners", max_rows_display=500)
        splitter.addWidget(self.table)
        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        if not dataset.is_loaded():
            return
        tanks = dataset.available_tanks()
        current = self.base_tank_combo.currentText()
        self.base_tank_combo.blockSignals(True)
        self.base_tank_combo.clear()
        self.base_tank_combo.addItems(tanks)
        if current and self.base_tank_combo.findText(current) >= 0:
            self.base_tank_combo.setCurrentText(current)
        elif tanks:
            self.base_tank_combo.setCurrentIndex(0)
        self.base_tank_combo.blockSignals(False)

    def set_base_tank_and_run(self, tank_id: str) -> None:
        """Public cross-tab entry point (VitrificationPage's jump from a
        double-clicked Screening row lands here) -- keeps the combo and
        run_blend() as private implementation details this tab is free to
        refactor without breaking its sibling."""
        self.base_tank_combo.setCurrentText(tank_id)
        self.run_blend()

    def run_blend(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Blend Partners", "Load a dataset first.")
            return
        base = self.base_tank_combo.currentText().strip()
        if not base:
            QMessageBox.warning(self, "Blend Partners", "Select a base tank first.")
            return
        self.blend_df = vsci.blend_partner_search(
            self.dataset, base, top_n=self.top_n_spin.value(),
            weights=_current_weights(self.weight_spins), basis=self.basis_combo.currentText(),
        )
        self.table.set_dataframe(self.blend_df)
        ph.plot_blend_scores(self.plot, self.blend_df, top_n=self.top_n_spin.value())
        self.app_window.statusBar().showMessage(f"Blend partner search completed for {base}: {len(self.blend_df):,} partner(s).")

    def export_tables(self) -> None:
        if self.blend_df.empty or self.dataset is None:
            QMessageBox.information(self, "Export", "Run a blend partner search first.")
            return
        out_dir = export_utils.export_named_tables(self.dataset, "vitrification_blend", {"blend_partners": self.blend_df})
        try:
            self.plot.figure.savefig(out_dir / "blend_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Blend partner search exported:\n{out_dir}")
