"""
qt_tank_explorer.py -- the Tank explorer: one tank as a profile, or
several tanks side by side.

A tank is a place that can be opened from anywhere in the application
(show_tank). Its profile reads top to bottom: name, farm, type, capacity
and status; the headline figures (waste volume, layers, sodium, 137Cs and
90Sr); the waste layers it holds; its composition as horizontal bars on a
logarithmic axis; and a small farm map locating it. Several tanks
selected together get the same page as a comparison.

Headline figures come from the query engine (site_summary); the
composition table is tank_science.tank_profile, as before.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import export_utils
import plot_helpers as ph
import site_summary as ss
import tank_science as tsci
from data_model import HanfordDataset
from qt_components import EmptyState, NotesBanner, PageHeader, Pill, StatTile
from qt_farm_map import CATEGORICAL, SERIES_BLUE, FarmMapWidget
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, DataFrameTableView, PlotWidget

PLOT_VALUES = ["Inventory_sum", "Fraction_of_tank_unit_inventory"]
PLOT_VALUE_LABELS = {
    "Inventory_sum": "Inventory",
    "Fraction_of_tank_unit_inventory": "Share of the tank's inventory",
}
MAX_SERIES = 6          # more tanks than this are drawn as a range, not as bars
_TABLE_TOP_N = 300      # the composition table lists every element; the chart trims


def _series_colours(n: int) -> List[str]:
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(n)]


@ph.composable
def plot_tank_composition(panel, data: pd.DataFrame, value_col: str, title: str,
                          top_n: int = 20, unit: str = "") -> None:
    """Composition of one or several tanks: one row per element, largest
    at the top, on a logarithmic axis so the minor elements stay visible.

    One tank: a single series of bars. Up to six tanks: grouped bars, one
    colour per tank. More: the range across the tanks with the median
    marked, since that many bar colours cannot be told apart.
    A tank that does not report an element has no bar there (not a zero).
    """
    if data is None or data.empty or value_col not in data or "Element" not in data:
        panel.show_message("No composition data for this selection")
        return
    pdf = data.copy()
    pdf["_value"] = ph.numeric_plot_series(pdf[value_col])
    pdf = pdf.dropna(subset=["Element", "_value"])
    pdf = pdf[pdf["_value"] > 0]
    if pdf.empty:
        panel.show_message("No positive values to plot")
        return
    several_units = "Units" in pdf.columns and pdf["Units"].nunique() > 1
    pdf["_label"] = (pdf["Element"].astype(str) + " [" + pdf["Units"].astype(str) + "]"
                     if several_units else pdf["Element"].astype(str))
    pivot = pdf.pivot_table(index="_label", columns="WasteSiteId", values="_value", aggfunc="sum")
    order = pivot.max(axis=1).sort_values(ascending=False).head(int(top_n)).index
    pivot = pivot.loc[order[::-1]]          # barh draws upward: largest last
    tanks = [str(t) for t in pivot.columns]
    labels = [str(v) for v in pivot.index]
    n, k = len(labels), len(tanks)

    value_label = PLOT_VALUE_LABELS.get(value_col, value_col)
    if value_col == "Inventory_sum" and unit:
        value_label = f"{value_label} ({unit})"
    elif unit:
        value_label = f"{value_label} ({unit} basis)"

    ax = panel.reset_axes()
    width = panel.figure.get_size_inches()[0]
    per_row = 0.18 if k == 1 else 0.24 + 0.07 * (min(k, MAX_SERIES) - 2)
    panel.set_figure_size_inches(width, min(max(3.2, n * per_row + 0.9), 14.0))
    positions = np.arange(n)
    if k == 1:
        ax.barh(positions, pivot[pivot.columns[0]].to_numpy(dtype=float), height=0.72, color=SERIES_BLUE)
    elif k <= MAX_SERIES:
        bar = 0.8 / k
        for i, (column, colour) in enumerate(zip(pivot.columns, _series_colours(k))):
            offset = (k - 1) / 2.0 * bar - i * bar      # first tank at the top of each group
            ax.barh(positions + offset, pivot[column].to_numpy(dtype=float), height=bar * 0.9,
                    color=colour, label=str(column))
        # Above the bars, where a tall chart shows it without scrolling.
        ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                  ncol=min(k, 3), borderaxespad=0.2, handlelength=1.2, columnspacing=1.4)
    else:
        low = pivot.min(axis=1).to_numpy(dtype=float)
        high = pivot.max(axis=1).to_numpy(dtype=float)
        median = pivot.median(axis=1).to_numpy(dtype=float)
        ax.hlines(positions, low, high, color=SERIES_BLUE, linewidth=3, alpha=0.45,
                  label=f"range across {k} tanks")
        ax.plot(median, positions, "o", color=SERIES_BLUE, markersize=5, label="median")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xscale("log")
    ax.set_ylim(-0.7, n - 0.3)
    declare = getattr(ph, "declare_categories", None)
    if callable(declare):
        declare(ax, labels, axis="y", name="Element", value_label=value_label)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    ax.set_xlabel(f"{value_label} — log scale")
    legend_rows = 0 if k == 1 or k > MAX_SERIES else (k + 2) // 3
    ax.set_title(title, pad=8 + 17 * legend_rows)
    ax.grid(True, axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    panel.figure.tight_layout()
    panel.canvas.draw_idle()


class SmallTable(QTableWidget):
    """A few rows that must be read at a glance (the layers of a tank, the
    tanks of a comparison): every column visible without scrolling,
    numbers formatted and right-aligned, no filter box. The large result
    tables of the page stay DataFrameTableView."""

    tankActivated = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._df = pd.DataFrame()
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(28)
        self.horizontalHeader().setHighlightSections(False)
        self.itemDoubleClicked.connect(self._on_double_click)

    def set_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        self._df = df if df is not None else pd.DataFrame()
        frame = self._df
        self.clear()
        self.setRowCount(len(frame))
        self.setColumnCount(len(frame.columns))
        self.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        header = self.horizontalHeader()
        for column, name in enumerate(frame.columns):
            numeric = pd.api.types.is_numeric_dtype(frame[name])
            header.setSectionResizeMode(
                column, QHeaderView.ResizeToContents if numeric or name == "Tank" else QHeaderView.Stretch)
            item = self.horizontalHeaderItem(column)
            if item is not None:
                item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignRight if numeric else Qt.AlignLeft))
            for row in range(len(frame)):
                value = frame.iat[row, column]
                if numeric:
                    text = ss.format_number(value, 4)
                else:
                    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
                cell = QTableWidgetItem(text)
                cell.setToolTip(text)
                cell.setTextAlignment(Qt.AlignVCenter | (Qt.AlignRight if numeric else Qt.AlignLeft))
                self.setItem(row, column, cell)

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def set_row_colours(self, colours: Sequence[str]) -> None:
        """A colour swatch in front of each row (the series colours of the
        chart and the map, so the three read as one)."""
        for row, colour in enumerate(colours):
            item = self.item(row, 0)
            if item is None:
                continue
            swatch = QPixmap(10, 10)
            swatch.fill(QColor(colour))
            item.setIcon(QIcon(swatch))

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        if "Tank" in self._df.columns and 0 <= item.row() < len(self._df):
            self.tankActivated.emit(str(self._df["Tank"].iloc[item.row()]))


class TankExplorerPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self.dataset: Optional[HanfordDataset] = None
        self._profile_df: Optional[pd.DataFrame] = None
        self._raw_df: Optional[pd.DataFrame] = None
        self._headlines: Optional[pd.DataFrame] = None
        self._layers_df: Optional[pd.DataFrame] = None
        self._result_tanks: List[str] = []
        self._unattributed_note: Optional[str] = None
        self._chart_rows = 0
        self._fit_timer = QTimer(self)        # owned by the page: dies with it
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(0)
        self._fit_timer.timeout.connect(self._fit_columns)
        self._build_ui()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 12)
        root.setSpacing(16)

        self.header = PageHeader(
            "Tank explorer",
            "Open one tank as a profile, or select several tanks to compare their composition.")
        self.export_btn = QPushButton("Export...")
        self.export_btn.setObjectName("Secondary")      # the quiet header action, as on every page
        self.export_btn.setToolTip("Write the composition, the raw rows and the chart to a dated folder.")
        self.export_btn.clicked.connect(self._export_tables)
        self.header.add_action_widget(self.export_btn)
        self.header.add_action_widget(make_help_button(self, 8, "Tank Explorer"))
        root.addWidget(self.header)

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_selection_panel(), 0)
        self.result_stack = QStackedWidget()
        self.empty_state = EmptyState(
            "Select a tank",
            "Choose one tank for its profile, or several to compare them, then Run. "
            "A double-click on a tank opens it directly.")
        self.result_stack.addWidget(self.empty_state)
        self.result_stack.addWidget(self._build_result())
        body.addWidget(self.result_stack, 1)
        root.addLayout(body, 1)

    def _build_selection_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        panel.setFixedWidth(248)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        label = QLabel("Farm")
        label.setObjectName("FieldLabel")
        layout.addWidget(label)
        self.farm_combo = QComboBox()
        self.farm_combo.addItem("All")
        self.farm_combo.setToolTip("Limit the list below to one tank farm.")
        self.farm_combo.currentTextChanged.connect(lambda _: self._refresh_tank_list())
        layout.addWidget(self.farm_combo)

        label = QLabel("Tanks")
        label.setObjectName("FieldLabel")
        layout.addWidget(label)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter tanks, e.g. AW or 106")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setAccessibleName("Filter the tank list")
        self.filter_edit.textChanged.connect(lambda _: self._apply_filter())
        layout.addWidget(self.filter_edit)
        self.tank_list = QListWidget()
        self.tank_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tank_list.setAccessibleName("Tanks")
        self.tank_list.setToolTip("Click to select, Ctrl or Shift to select several, double-click to open.")
        self.tank_list.itemSelectionChanged.connect(self._update_selection_label)
        self.tank_list.itemDoubleClicked.connect(lambda item: self.show_tank(item.text()))
        layout.addWidget(self.tank_list, 1)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.selection_label = QLabel("No tank selected")
        self.selection_label.setObjectName("SectionNote")
        row.addWidget(self.selection_label, 1)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("Secondary")
        self.clear_btn.clicked.connect(self.tank_list.clearSelection)
        self.clear_btn.setVisible(False)
        row.addWidget(self.clear_btn)
        layout.addLayout(row)

        label = QLabel("Basis")
        label.setObjectName("FieldLabel")
        layout.addWidget(label)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kg", "Ci"])
        self.unit_combo.setToolTip("Composition by mass (kg) or by activity (Ci).")
        layout.addWidget(self.unit_combo)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)

        self.advanced_section = CollapsibleSection("Advanced options")
        adv = self.advanced_section.content_layout()
        label = QLabel("Elements on the chart")
        label.setObjectName("FieldLabel")
        adv.addWidget(label)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 300)
        self.top_n_spin.setValue(20)
        self.top_n_spin.setToolTip("How many elements the chart shows, largest first. "
                                   "The composition table always lists every element.")
        adv.addWidget(self.top_n_spin)
        label = QLabel("Chart value")
        label.setObjectName("FieldLabel")
        adv.addWidget(label)
        self.value_combo = QComboBox()
        for column in PLOT_VALUES:
            self.value_combo.addItem(PLOT_VALUE_LABELS[column], column)
        self.value_combo.setToolTip("Inventory, or each element's share of the tank's own inventory.")
        adv.addWidget(self.value_combo)
        layout.addWidget(self.advanced_section)
        return panel

    def _build_result(self) -> QWidget:
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.result_scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 8)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.profile_title = QLabel("")
        self.profile_title.setObjectName("PageTitle")
        self.profile_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title_row.addWidget(self.profile_title)
        self.pill_row = QHBoxLayout()
        self.pill_row.setSpacing(6)
        title_row.addLayout(self.pill_row)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.notes_banner = NotesBanner()
        layout.addWidget(self.notes_banner)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.tile_volume = StatTile("Waste volume")
        self.tile_layers = StatTile("Waste layers")
        self.tile_sodium = StatTile("Sodium")
        self.tile_activity = StatTile("Cesium, strontium")
        self.stat_tiles = [self.tile_volume, self.tile_layers, self.tile_sodium, self.tile_activity]
        for tile in self.stat_tiles:
            tiles.addWidget(tile, 1)
        layout.addLayout(tiles)

        self.columns_block = QWidget()
        columns = QHBoxLayout(self.columns_block)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(12)

        side = QVBoxLayout()
        side.setSpacing(12)
        layers_card = QFrame()
        layers_card.setObjectName("Card")
        layers_layout = QVBoxLayout(layers_card)
        layers_layout.setContentsMargins(12, 10, 12, 10)
        layers_layout.setSpacing(6)
        self.side_title = QLabel("Waste layers")
        self.side_title.setObjectName("SectionTitle")
        layers_layout.addWidget(self.side_title)
        self.layers_view = SmallTable()
        self.layers_view.setAccessibleName("Waste layers of the tank, or the tanks compared")
        self.layers_view.setToolTip("In a comparison, double-click a tank to open its profile.")
        self.layers_view.tankActivated.connect(self.show_tank)
        self.layers_view.setFixedHeight(120)
        layers_layout.addWidget(self.layers_view, 1)
        layers_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        side.addWidget(layers_card, 0)

        map_card = QFrame()
        map_card.setObjectName("Card")
        map_layout = QVBoxLayout(map_card)
        map_layout.setContentsMargins(12, 10, 12, 8)
        map_layout.setSpacing(4)
        self.map_title = QLabel("Location")
        self.map_title.setObjectName("SectionTitle")
        map_layout.addWidget(self.map_title)
        self.mini_map = FarmMapWidget(compact=True)
        self.mini_map.setMinimumHeight(190)
        self.mini_map.setToolTip("Double-click a tank to open it.")
        self.mini_map.tankActivated.connect(self.show_tank)
        map_layout.addWidget(self.mini_map, 1)
        map_card.setMaximumHeight(300)
        side.addWidget(map_card, 10)
        side.addStretch(1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side.setContentsMargins(0, 0, 0, 0)
        side_widget.setMinimumWidth(380)
        side_widget.setMaximumWidth(560)
        columns.addWidget(side_widget, 6)

        self.tables = QTabWidget()
        self.plot = PlotWidget()
        self.profile_view = DataFrameTableView(title="Tank composition")
        self.raw_view = DataFrameTableView(title="Raw rows (first selected tank)", max_rows_display=5000)
        self.tables.addTab(self.plot, "Composition chart")
        self.tables.addTab(self.profile_view, "Composition table")
        self.tables.addTab(self.raw_view, "Raw rows")
        columns.addWidget(self.tables, 5)
        self.columns_block.setFixedHeight(460)
        layout.addWidget(self.columns_block)
        layout.addStretch(1)
        self._result_layout = layout
        self._tiles_row = tiles
        self.result_scroll.viewport().installEventFilter(self)
        return self.result_scroll

    # ------------------------------------------------------------------
    # layout: the profile fills the pane it is given, no more
    # ------------------------------------------------------------------
    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt naming)
        if watched is self.result_scroll.viewport() and event.type() == QEvent.Resize:
            self._fit_timer.start()
        return super().eventFilter(watched, event)

    def _fit_columns(self) -> None:
        """Give the layers, map and chart the height left under the header
        and the tiles. A fixed height: inside a scroll area a block is
        otherwise given its preferred height, which ignores the window."""
        viewport = self.result_scroll.viewport().height()
        if viewport <= 0:
            return
        spacing = self._result_layout.spacing()
        above = max(self.profile_title.height(), self.profile_title.sizeHint().height()) + spacing
        if not self.notes_banner.isHidden():
            above += max(self.notes_banner.height(), self.notes_banner.sizeHint().height()) + spacing
        above += max(self.tile_volume.height(), self.tile_volume.sizeHint().height()) + spacing
        chart_rows = 0.0
        if self._chart_rows:
            chart_rows = self._chart_rows * 15.0 + 150.0     # every label legible
        target = int(max(400, viewport - above - 10, min(chart_rows, 1400.0)))
        if self.columns_block.height() != target:
            self.columns_block.setFixedHeight(target)

    def _fit_layers_table(self, n_rows: int) -> None:
        self.layers_view.setFixedHeight(int(min(210, 40 + 28 * max(1, n_rows))))

    # ------------------------------------------------------------------
    # dataset, list, selection
    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.dataset = dataset
        self._profile_df = self._raw_df = self._headlines = self._layers_df = None
        self._result_tanks = []
        self.result_stack.setCurrentWidget(self.empty_state)
        if not dataset.is_loaded():
            self.empty_state.set_text("No dataset loaded", "Load the tank inventory to open a tank.")
            self.plot.show_message("Load a dataset first.")
            return
        self.empty_state.set_text(
            "Select a tank",
            "Choose one tank for its profile, or several to compare them, then Run. "
            "A double-click on a tank opens it directly.")
        units = dataset.available_units()
        self.unit_combo.blockSignals(True)
        self.unit_combo.clear()
        self.unit_combo.addItems(units or ["kg", "Ci"])
        if "kg" in units:
            self.unit_combo.setCurrentText("kg")
        self.unit_combo.blockSignals(False)

        current_farm = self.farm_combo.currentText()
        self.farm_combo.blockSignals(True)
        self.farm_combo.clear()
        self.farm_combo.addItems(tsci.available_farms_with_all(dataset))
        restore = self.farm_combo.findText(current_farm)
        self.farm_combo.setCurrentIndex(restore if restore >= 0 else 0)
        self.farm_combo.blockSignals(False)
        self._refresh_tank_list()
        try:
            self.mini_map.set_tanks(ss.tank_table(dataset))
        except Exception:
            self.mini_map.set_tanks(dataset.available_tanks())

    def _refresh_tank_list(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            return
        farm = self.farm_combo.currentText()
        tanks = tsci.tanks_in_farm(self.dataset, farm)
        self.tank_list.clear()
        self.tank_list.addItems(tanks)
        self._apply_filter()
        self._update_selection_label()

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        for i in range(self.tank_list.count()):
            item = self.tank_list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _update_selection_label(self) -> None:
        n = len(self.tank_list.selectedItems())
        total = self.tank_list.count()
        if n == 0:
            self.selection_label.setText(f"No tank selected ({total:,} listed)")
        elif n == 1:
            self.selection_label.setText(f"1 of {total:,} selected")
        else:
            self.selection_label.setText(f"{n:,} of {total:,} selected")
        self.clear_btn.setVisible(n > 0)

    def selected_tanks(self) -> List[str]:
        return [item.text() for item in self.tank_list.selectedItems()]

    def show_tank(self, tank_id: str) -> bool:
        """Open one tank: select it, run, and bring its profile into view.
        Returns False when the tank is not in the loaded dataset."""
        return self.show_tanks([tank_id])

    def show_tanks(self, tank_ids: Sequence[str]) -> bool:
        """Open several tanks as a comparison (one tank: its profile)."""
        if self.dataset is None or not self.dataset.is_loaded():
            return False
        wanted = [str(t) for t in tank_ids]
        self.filter_edit.clear()
        if self.farm_combo.currentIndex() != 0:
            self.farm_combo.setCurrentIndex(0)
        self.tank_list.clearSelection()
        first = None
        for i in range(self.tank_list.count()):
            item = self.tank_list.item(i)
            if item.text() in wanted:
                item.setSelected(True)
                first = first or item
        if first is None:
            return False
        self.tank_list.scrollToItem(first, QAbstractItemView.PositionAtCenter)
        self.run()
        return True

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self) -> None:
        if self.dataset is None or not self.dataset.is_loaded():
            QMessageBox.information(self, "Tank explorer", "Load a dataset first.")
            return
        tanks = self.selected_tanks()
        if not tanks:
            QMessageBox.warning(self, "No tanks", "Select one or more tanks first.")
            return
        unit = self.unit_combo.currentText() or None
        top_n = self.top_n_spin.value()
        self._profile_df = tsci.tank_profile(self.dataset, tanks, unit_filter=unit, top_n=_TABLE_TOP_N)
        self._raw_df = tsci.raw_rows_for_tank(self.dataset, tanks[0], unit_filter=unit, limit=5000)
        self.profile_view.set_dataframe(self._profile_df)
        self.raw_view.set_dataframe(self._raw_df)
        self._result_tanks = list(tanks)
        try:
            self._headlines = ss.tank_headlines(self.dataset, tanks)
        except Exception:
            self._headlines = None
        self._fill_profile(tanks)

        value_col = self.value_combo.currentData() or PLOT_VALUES[0]
        # One tank is named in the title; several are named by the legend.
        plot_tank_composition(
            self.plot, self._profile_df, value_col,
            f"Composition of {tanks[0]}" if len(tanks) == 1 else f"Composition of {len(tanks)} tanks compared",
            top_n=top_n, unit=unit or "",
        )
        self._unattributed_note = tsci.unattributed_mass_note(self.dataset, tanks, unit)
        self.notes_banner.set_notes([self._unattributed_note] if self._unattributed_note else [])
        n_elements = min(top_n, self._profile_df["Element"].nunique()) if not self._profile_df.empty else 0
        series = min(len(tanks), MAX_SERIES) if len(tanks) <= MAX_SERIES else 1
        self._chart_rows = n_elements * max(1.0, 0.45 * series + 0.55)
        self._fit_layers_table(len(self._layers_df) if self._layers_df is not None else 1)
        self.result_stack.setCurrentWidget(self.result_scroll)
        self._fit_columns()
        self.result_scroll.verticalScrollBar().setValue(0)
        what = tanks[0] if len(tanks) == 1 else f"{len(tanks)} tanks"
        self.app_window.statusBar().showMessage(f"Tank explorer: {what} ({unit or 'all units'}).")

    def _fill_profile(self, tanks: Sequence[str]) -> None:
        head = self._headlines
        single = len(tanks) == 1
        while self.pill_row.count():
            item = self.pill_row.takeAt(0)
            pill = item.widget()
            if pill is not None:
                pill.hide()
                pill.setParent(None)
                pill.deleteLater()

        def column(name: str) -> pd.Series:
            if head is None or name not in head.columns:
                return pd.Series(dtype=float)
            return head[name]

        def first(name: str):
            values = column(name)
            if values.empty:
                return None
            value = values.iloc[0]
            return None if value is None or (isinstance(value, float) and np.isnan(value)) else value

        if single:
            self.profile_title.setText(tanks[0])
            pills = []
            farm = first("TankFarm")
            if farm:
                pills.append(f"{farm} farm")
            system = first("TankSystem")
            if system in ("SST", "DST"):
                pills.append("Double-shell tank" if system == "DST" else "Single-shell tank")
            tank_type = first("TankType")
            if tank_type and tank_type != system:
                pills.append(f"Type {tank_type}")
            capacity = first("Capacity")
            if capacity is not None:
                units = first("CapacityUnits") or "kgal"
                text = f"Capacity {ss.format_number(capacity, 4)} {units}"
                if str(units).lower() == "kgal":
                    text += f" ({ss.format_number(float(capacity) * 1000.0 / ss.GALLONS_PER_KL, 3)} kL)"
                pills.append(text)
            integrity = first("TankIntegrity")
            if integrity:
                pills.append(f"Integrity: {integrity}")
            status = first("TankStatus")
            if status:
                pills.append(f"Status: {status}")
            service = first("InServiceDate")
            if service:
                pills.append(f"In service {service}")
            for text in pills:
                self.pill_row.addWidget(Pill(text))
        else:
            self.profile_title.setText(f"Comparison of {len(tanks)} tanks")
            farms = sorted({str(f) for f in column("TankFarm").dropna()})
            if farms:
                self.pill_row.addWidget(Pill("Farm " + ", ".join(farms) if len(farms) == 1
                                             else "Farms " + ", ".join(farms)))

        def total(name: str) -> Optional[float]:
            values = pd.to_numeric(column(name), errors="coerce").dropna()
            return float(values.sum()) if len(values) else None

        suffix = "" if single else " in total"
        volume = total("Volume_kL")
        self.tile_volume.set_value(
            ss.BLANK if volume is None else f"{ss.format_number(volume, 4)} kL",
            "not stated in this dataset" if volume is None else ss.format_gallons(volume) + suffix)
        layers = total("N_layers")
        self.tile_layers.set_value(
            ss.BLANK if layers is None else f"{int(layers):,}",
            "phase and waste type" if single else f"across {len(tanks)} tanks")
        sodium = total("Na_kg")
        hint = "not reported" if sodium is None else ("as Na" + suffix)
        share = first("Na_wt_pct")
        if single and sodium is not None and share is not None:
            hint = f"{ss.format_number(share)} wt% of waste"
        self.tile_sodium.set_value(ss.format_mass(sodium), hint)
        activity, split, sentence = ss.format_activity_split(
            [("137Cs", total("Cs137_Ci")), ("90Sr", total("Sr90_Ci"))])
        self.tile_activity.set_value(activity, split or "not reported")
        self.tile_activity.setToolTip(
            f"Activity of {sentence}{suffix}, as reported." if sentence
            else "Neither 137Cs nor 90Sr is reported here.")

        if single:
            self.side_title.setText("Waste layers")
            self._layers_df = ss.tank_layers(self.dataset, tanks[0])
            self.layers_view.set_dataframe(self._layers_df)
            self.map_title.setText("Location")
            self.mini_map.set_categories({tanks[0]: "This tank"}, {"This tank": SERIES_BLUE})
        else:
            self.side_title.setText("Tanks compared")
            self._layers_df = self._comparison_table(head)
            self.layers_view.set_dataframe(self._layers_df)
            self.map_title.setText("Locations")
            if len(tanks) <= MAX_SERIES:
                # The chart orders its series by tank name: same colours here.
                ordered = sorted(tanks)
                colours = dict(zip(ordered, _series_colours(len(ordered))))
                self.mini_map.set_categories({t: t for t in tanks}, colours)
                if self._layers_df is not None and "Tank" in self._layers_df.columns:
                    self.layers_view.set_row_colours(
                        [colours.get(str(t), SERIES_BLUE) for t in self._layers_df["Tank"]])
            else:
                self.mini_map.set_categories({t: "Selected" for t in tanks}, {"Selected": SERIES_BLUE})
        self.mini_map.set_selection(list(tanks))

    @staticmethod
    def _comparison_table(head: Optional[pd.DataFrame]) -> pd.DataFrame:
        if head is None or head.empty:
            return pd.DataFrame()
        columns = {
            "WasteSiteId": "Tank", "Volume_kL": "Volume (kL)", "Na_kg": "Na (kg)",
            "Cs137_Ci": "137Cs (Ci)", "Sr90_Ci": "90Sr (Ci)",
        }
        keep = [c for c in columns if c in head.columns]
        return head[keep].rename(columns=columns)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _export_tables(self) -> None:
        if self._profile_df is None and self._raw_df is None:
            QMessageBox.information(self, "Export", "Run the tank explorer first.")
            return
        tables = {"tank_composition": self._profile_df, "raw_rows_first_selected_tank": self._raw_df}
        if self._headlines is not None and not self._headlines.empty:
            tables["tank_summary"] = self._headlines
        if self._layers_df is not None and not self._layers_df.empty and len(self._result_tanks) == 1:
            tables["waste_layers"] = self._layers_df
        out_dir = export_utils.export_named_tables(self.dataset, "tank_view", tables)
        try:
            self.plot.figure.savefig(out_dir / "tank_plot.png", bbox_inches="tight", dpi=200)
        except Exception:
            pass
        QMessageBox.information(self, "Export complete", f"Tank explorer tables exported:\n{out_dir}")
