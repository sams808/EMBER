"""
qt_figure_composer.py — the Figure Composer workspace: a gallery of plot
"recipes" captured (via each PlotWidget's "Send to Figure Composer" toolbar
button, anywhere in Ember) into composer_store.store, arranged into a
labeled multi-panel grid and exported at a chosen physical size/DPI.

Each panel is a live re-render of its recipe (see composer_store.py /
figure_composer.py), not a frozen snapshot, so title, axis labels, and any
of the recipe's own keyword parameters (color mode, annotate, top-N, ...)
stay editable right up to export.

Layout: a page header, then the panel list and its controls on the left
and the live preview on the right. The actual grid compositing is
figure_composer.compose_figure(); the pure layout/label math it calls is
in compose_science.py.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import compose_science as csci
import figure_composer as fcomp
from composer_store import TEXT_OVERRIDE_FIELDS, store
from data_model import HanfordDataset
from qt_components import PageHeader, secondary_button
from qt_help import make_help_button
from qt_widgets import CollapsibleSection, PlotWidget

PAGE_SUBTITLE = (
    'Combine charts from anywhere in the app into one lettered, multi-panel figure. Use "Send to '
    'Figure Composer" on the toolbar of a chart to add it; every panel stays editable until export.'
)


def _override_color() -> QColor:
    """The colour of an edited parameter value: the theme's "good" tone."""
    try:
        import qt_theme
        return QColor(qt_theme.current_palette().get("good", "#1a7f4b"))
    except Exception:
        return QColor("#1a7f4b")


def _parse_param_text(text: str) -> Any:
    """A typed Python literal (True/False/None/42/3.14/"quoted") if it
    parses as one, otherwise the raw text as a plain string -- lets users
    edit a string-valued parameter (e.g. method=pearson) without having to
    remember to type quotes."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


class FigureComposerPage(QWidget):
    """Rendering discipline (the fix for "a title edit replays every
    panel"): the store's STRUCTURAL `changed` signal triggers a full
    gallery+preview rebuild, while per-item `item_edited` signals take the
    cheapest correct path -- caption edits touch one gallery row, text
    overrides mutate one panel's axes in place, parameter edits re-render
    exactly one SubFigure after a short debounce (falling back to a full
    recompose only when the edit changed that panel's measured preferred
    size). All signal-driven work is skipped while the page is hidden --
    capturing plots from other workspaces just marks the preview stale,
    and showEvent settles the debt when the user actually navigates
    here."""

    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        self._thumbnail_cache: Dict[int, tuple] = {}  # uid -> (render-state token, png bytes)
        self._composed: Optional[fcomp.ComposedPanels] = None
        self._stale_full = False
        self._dirty_panels: set = set()
        self._dirty_thumbs: set = set()
        self._build_ui()
        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(200)
        self._edit_timer.timeout.connect(self._process_dirty)
        store.changed.connect(self._on_store_changed)
        store.item_edited.connect(self._on_item_edited)
        # Unconditional initial sync (unlike the signal-driven paths, which
        # are visibility-guarded): the page may be constructed after panels
        # were already captured, and must reflect them from the start.
        self._refresh_gallery()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        self.header = PageHeader("Figure Composer", PAGE_SUBTITLE)
        self.header.add_action_widget(make_help_button(self, 13, "Figure Composer"))
        root.addWidget(self.header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # The control column scrolls by itself: opening Advanced options
        # must never push the preview (or the Export button) off the page.
        left = QWidget()
        left.setObjectName("Card")
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("ControlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(372)
        self.controls_scroll.setMaximumWidth(440)
        self.controls_scroll.setWidget(left)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(8)

        panels_title = QLabel("Panels")
        panels_title.setObjectName("SectionTitle")
        ll.addWidget(panels_title)

        self.gallery_list = QListWidget()
        self.gallery_list.setIconSize(QSize(110, 82))
        self.gallery_list.setToolTip("The panels of the figure, in reading order. Select one to edit its labels.")
        self.gallery_list.currentRowChanged.connect(self._on_selection_changed)
        self.gallery_list.setMinimumHeight(150)
        ll.addWidget(self.gallery_list, 1)

        row_btns = QHBoxLayout()
        row_btns.setSpacing(6)
        self.up_btn = QPushButton("Up")
        self.up_btn.setToolTip("Move the selected panel one place earlier.")
        self.up_btn.clicked.connect(lambda: self._move(-1))
        row_btns.addWidget(self.up_btn)
        self.down_btn = QPushButton("Down")
        self.down_btn.setToolTip("Move the selected panel one place later.")
        self.down_btn.clicked.connect(lambda: self._move(1))
        row_btns.addWidget(self.down_btn)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip("Take the selected panel out of the figure.")
        self.remove_btn.clicked.connect(self._remove_selected)
        row_btns.addWidget(self.remove_btn)
        row_btns.addStretch(1)
        self.clear_btn = secondary_button("Clear all", "Remove every panel (asks first).")
        self.clear_btn.clicked.connect(self._clear_all)
        row_btns.addWidget(self.clear_btn)
        ll.addLayout(row_btns)

        # Per-panel label editing and grid layout are refinements on top of
        # the core add/reorder/export workflow above -- collapsed by
        # default so a first-time user sees a short, obvious path.
        self.advanced_section = CollapsibleSection("Advanced options (panel labels and layout)")
        adv = self.advanced_section.content_layout()

        panel_title = QLabel("Selected panel")
        panel_title.setObjectName("FieldLabel")
        adv.addWidget(panel_title)
        panel_form = QFormLayout()
        panel_form.setContentsMargins(0, 0, 0, 0)
        panel_form.setHorizontalSpacing(8)
        panel_form.setVerticalSpacing(6)

        self.caption_edit = QLineEdit()
        self.caption_edit.setEnabled(False)
        self.caption_edit.setToolTip("The name of the panel in the list above. It is not drawn on the figure.")
        self.caption_edit.editingFinished.connect(self._apply_caption_edit)
        panel_form.addRow("Caption", self.caption_edit)

        title_row = QHBoxLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setEnabled(False)
        self.title_edit.setPlaceholderText("(from plot)")
        self.title_edit.setToolTip("Replace the panel's title. Leave empty to keep the title of the chart.")
        self.title_edit.editingFinished.connect(self._apply_title_edit)
        title_row.addWidget(self.title_edit, 1)
        self.hide_title_check = QCheckBox("Hide")
        self.hide_title_check.setEnabled(False)
        self.hide_title_check.setToolTip("Draw this panel without a title.")
        self.hide_title_check.toggled.connect(self._apply_title_edit)
        title_row.addWidget(self.hide_title_check)
        panel_form.addRow("Title", title_row)

        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setEnabled(False)
        self.xlabel_edit.setPlaceholderText("(from plot)")
        self.xlabel_edit.setToolTip("Replace the label of the horizontal axis. Leave empty to keep the chart's own.")
        self.xlabel_edit.editingFinished.connect(self._apply_xlabel_edit)
        panel_form.addRow("X axis", self.xlabel_edit)

        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setEnabled(False)
        self.ylabel_edit.setPlaceholderText("(from plot)")
        self.ylabel_edit.setToolTip("Replace the label of the vertical axis. Leave empty to keep the chart's own.")
        self.ylabel_edit.editingFinished.connect(self._apply_ylabel_edit)
        panel_form.addRow("Y axis", self.ylabel_edit)
        adv.addLayout(panel_form)

        params_row = QHBoxLayout()
        params_label = QLabel("Parameters")
        params_label.setObjectName("FieldLabel")
        params_row.addWidget(params_label)
        params_row.addStretch(1)
        self.reset_params_btn = secondary_button("Reset", "Put every parameter of this panel back to the value it was captured with.")
        self.reset_params_btn.setEnabled(False)
        self.reset_params_btn.clicked.connect(self._reset_params)
        params_row.addWidget(self.reset_params_btn)
        adv.addLayout(params_row)
        self.params_table = QTableWidget(0, 2)
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.params_table.horizontalHeader().setStretchLastSection(True)
        self.params_table.verticalHeader().setVisible(False)
        self.params_table.setMaximumHeight(150)
        self.params_table.setEnabled(False)
        self.params_table.setToolTip(
            "The settings the chart was drawn with. Edit a value to redraw this panel with it;\n"
            "edited values are highlighted.")
        self.params_table.itemChanged.connect(self._on_param_item_changed)
        adv.addWidget(self.params_table)

        layout_title = QLabel("Layout")
        layout_title.setObjectName("FieldLabel")
        adv.addWidget(layout_title)
        layout_form = QFormLayout()
        layout_form.setContentsMargins(0, 0, 0, 0)
        layout_form.setHorizontalSpacing(8)
        layout_form.setVerticalSpacing(6)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(0, 8)
        self.cols_spin.setSpecialValueText("Auto")
        self.cols_spin.setToolTip("Number of columns of the grid. Auto chooses the most compact arrangement.")
        self.cols_spin.valueChanged.connect(self._rebuild_preview)
        layout_form.addRow("Columns", self.cols_spin)

        self.label_style_combo = QComboBox()
        self.label_style_combo.addItems(csci.LABEL_STYLES)
        self.label_style_combo.setToolTip("How panels are lettered in their corner.")
        self.label_style_combo.currentTextChanged.connect(self._rebuild_preview)
        layout_form.addRow("Panel labels", self.label_style_combo)

        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(6, 36)
        self.label_size_spin.setValue(14)
        self.label_size_spin.setSuffix(" pt")
        self.label_size_spin.setToolTip("Size of the panel letters.")
        self.label_size_spin.valueChanged.connect(self._rebuild_preview)
        layout_form.addRow("Label size", self.label_size_spin)
        adv.addLayout(layout_form)

        ll.addWidget(self.advanced_section)

        export_title = QLabel("Export")
        export_title.setObjectName("SectionTitle")
        ll.addWidget(export_title)
        export_form = QFormLayout()
        export_form.setContentsMargins(0, 0, 0, 0)
        export_form.setHorizontalSpacing(8)
        export_form.setVerticalSpacing(6)
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        self.export_w_spin = QDoubleSpinBox()
        self.export_w_spin.setRange(1.0, 100.0)
        self.export_w_spin.setValue(18.0)
        self.export_w_spin.setSuffix(" cm")
        self.export_w_spin.setToolTip("Width of the exported figure.")
        size_row.addWidget(self.export_w_spin, 1)
        size_row.addWidget(QLabel("×"))
        self.export_h_spin = QDoubleSpinBox()
        self.export_h_spin.setRange(1.0, 100.0)
        self.export_h_spin.setValue(12.0)
        self.export_h_spin.setSuffix(" cm")
        self.export_h_spin.setToolTip("Height of the exported figure.")
        size_row.addWidget(self.export_h_spin, 1)
        export_form.addRow("Size", size_row)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.setToolTip("Resolution of a PNG or TIFF export. PDF and SVG are vector formats.")
        export_form.addRow("Resolution", self.dpi_spin)
        ll.addLayout(export_form)

        self.export_btn = QPushButton("Export figure...")
        self.export_btn.setObjectName("Primary")
        self.export_btn.setToolTip("Save the figure as PNG, PDF, SVG or TIFF at the size above.")
        self.export_btn.clicked.connect(self.export_figure)
        ll.addWidget(self.export_btn)
        splitter.addWidget(self.controls_scroll)

        self.plot = PlotWidget()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([384, 740])

    # ------------------------------------------------------------------
    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        pass  # Figure Composer works on captured plot recipes, not the dataset directly.

    # ------------------------------------------------------------------
    # Signal routing: structural vs per-item, visible vs hidden
    # ------------------------------------------------------------------
    def _on_store_changed(self) -> None:
        if not self.isVisible():
            # A capture from another workspace used to pay for a full
            # recompose of a page nobody was looking at -- now it just
            # marks the debt, paid once in showEvent.
            self._stale_full = True
            return
        self._refresh_gallery()

    def _on_item_edited(self, index: int, field: str) -> None:
        items = store.items()
        if not (0 <= index < len(items)):
            return
        item = items[index]
        if field == "caption":
            row = self.gallery_list.item(index)
            if row is not None:
                row.setText(item.caption or "(untitled)")
            return
        if index == self._selected_index():
            # The full gallery rebuild used to be what repopulated the
            # editor fields after an edit -- keep them honest directly.
            # Cheap pure-widget sync, so it runs even while hidden;
            # _populate_editors/_populate_params_table block their widgets'
            # signals, so this can't re-enter the edit handlers. kwargs
            # edits refresh only the params table (the override highlight),
            # not every editor widget -- a kwargs commit originates IN that
            # table and rebuilding five unrelated fields per cell edit is
            # pure churn (review finding).
            if field == "kwargs":
                self._populate_params_table(item)
            else:
                self._populate_editors(item)
        if not self.isVisible():
            self._stale_full = True
            return
        in_place = False
        if field in TEXT_OVERRIDE_FIELDS and self._composed is not None and self._composed.matches(items):
            # Pixel-identical to a full recompose (see apply_text_override)
            # at the cost of one canvas repaint instead of N panel replays.
            # False means the override was cleared back to None -- only the
            # recipe knows the original text, so that panel re-renders.
            in_place = fcomp.apply_text_override(self.plot, self._composed, index, item, field)
        if not in_place:
            self._dirty_panels.add(index)
        self._dirty_thumbs.add(index)
        self._edit_timer.start()

    def _process_dirty(self) -> None:
        if not self.isVisible():
            self._stale_full = self._stale_full or bool(self._dirty_panels) or bool(self._dirty_thumbs)
            self._dirty_panels.clear()
            self._dirty_thumbs.clear()
            return
        items = store.items()
        # Bounds-filter (belt and braces on top of _refresh_gallery clearing
        # dirty state on structural changes): a focus-out edit commit
        # followed by a Remove click lands both events in one debounce
        # window, and a stale index must never reach items[idx].
        panels = sorted(i for i in self._dirty_panels if 0 <= i < len(items))
        self._dirty_panels.clear()
        thumbs = sorted(i for i in (self._dirty_thumbs | set(panels)) if 0 <= i < len(items))
        self._dirty_thumbs.clear()
        if panels:
            needs_full = self._composed is None or not self._composed.matches(items)
            if not needs_full:
                for idx in panels:
                    # An edit that changed this panel's content-driven size
                    # request (e.g. top_n 30 -> 80 bars) invalidates the
                    # grid geometry itself -- only then rebuild everything.
                    if fcomp.measure_preferred_size(items[idx]) != self._composed.measured.get(idx):
                        needs_full = True
                        break
            if needs_full:
                self._rebuild_preview()
            else:
                for idx in panels:
                    fcomp.rerender_panel(self.plot, self._composed, idx, items[idx])
        for idx in thumbs:
            if 0 <= idx < len(items):
                self._update_row_thumbnail(idx, items[idx])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stale_full:
            self._stale_full = False
            self._dirty_panels.clear()
            self._dirty_thumbs.clear()
            self._refresh_gallery()
        elif self._dirty_panels or self._dirty_thumbs:
            self._process_dirty()

    # ------------------------------------------------------------------
    def _thumbnail_state(self, item) -> tuple:
        # Value-based render-state token, NOT id(...) of the override
        # dicts: id() can be recycled by the allocator after an item is
        # removed, which made stale thumbnails possible under the old key.
        # render_args/render_kwargs need no token of their own -- they are
        # frozen at capture time and uid already identifies them.
        return (
            fcomp._overrides_token(item),
            item.title_override, item.xlabel_override, item.ylabel_override,
        )

    def _update_row_thumbnail(self, index: int, item) -> None:
        row = self.gallery_list.item(index)
        if row is None:
            return
        pixmap = QPixmap()
        pixmap.loadFromData(self._thumbnail(item))
        row.setIcon(QIcon(pixmap))

    def _thumbnail(self, item) -> bytes:
        # ONE entry per uid, remembering the state it was rendered under --
        # a new state overwrites the old PNG instead of accumulating beside
        # it, so a long editing session can't grow this cache per edit
        # (review finding).
        state = self._thumbnail_state(item)
        cached = self._thumbnail_cache.get(item.uid)
        if cached is not None and cached[0] == state:
            return cached[1]
        png = fcomp.render_thumbnail_png(item)
        self._thumbnail_cache[item.uid] = (state, png)
        return png

    def _refresh_gallery(self) -> None:
        # A full rebuild supersedes any pending per-panel work, and dirty
        # indices refer to the PRE-rebuild item order -- drop them so a
        # structural change landing inside the debounce window can't leave
        # a stale index for the timer to trip over.
        self._edit_timer.stop()
        self._dirty_panels.clear()
        self._dirty_thumbs.clear()
        current = self.gallery_list.currentRow()
        items = store.items()
        live_uids = {item.uid for item in items}
        for stale_uid in [u for u in self._thumbnail_cache if u not in live_uids]:
            del self._thumbnail_cache[stale_uid]

        self.gallery_list.blockSignals(True)
        self.gallery_list.clear()
        for i, item in enumerate(items):
            self.gallery_list.addItem(QListWidgetItem(item.caption or "(untitled)"))
            self._update_row_thumbnail(i, item)
        n = self.gallery_list.count()
        if n:
            self.gallery_list.setCurrentRow(min(max(current, 0), n - 1))
        self.gallery_list.blockSignals(False)
        self._on_selection_changed(self.gallery_list.currentRow())
        self._rebuild_preview()

    def _selected_index(self) -> Optional[int]:
        row = self.gallery_list.currentRow()
        return row if 0 <= row < len(store.items()) else None

    def _on_selection_changed(self, row: int) -> None:
        items = store.items()
        selected = 0 <= row < len(items)
        for widget in (self.caption_edit, self.title_edit, self.hide_title_check,
                       self.xlabel_edit, self.ylabel_edit, self.params_table, self.reset_params_btn,
                       self.remove_btn):
            widget.setEnabled(selected)
        self.up_btn.setEnabled(selected and row > 0)
        self.down_btn.setEnabled(selected and row < len(items) - 1)
        self.clear_btn.setEnabled(bool(items))
        self.export_btn.setEnabled(bool(items))
        if not selected:
            self.caption_edit.clear()
            self.title_edit.clear()
            self.hide_title_check.setChecked(False)
            self.xlabel_edit.clear()
            self.ylabel_edit.clear()
            self.params_table.setRowCount(0)
            return
        self._populate_editors(items[row])

    def _populate_editors(self, item) -> None:
        self.caption_edit.setText(item.caption)
        self.hide_title_check.blockSignals(True)
        self.hide_title_check.setChecked(item.title_override == "")
        self.hide_title_check.blockSignals(False)
        self.title_edit.setText(item.title_override if item.title_override else "")
        self.title_edit.setEnabled(not self.hide_title_check.isChecked())
        self.xlabel_edit.setText(item.xlabel_override if item.xlabel_override else "")
        self.ylabel_edit.setText(item.ylabel_override if item.ylabel_override else "")
        self._populate_params_table(item)

    def _populate_params_table(self, item) -> None:
        self.params_table.blockSignals(True)
        kwargs = item.effective_kwargs()
        keys = sorted(kwargs)
        self.params_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            name_item = QTableWidgetItem(key)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.params_table.setItem(row, 0, name_item)
            value_item = QTableWidgetItem(str(kwargs[key]))
            if key in item.kwarg_overrides:
                value_item.setForeground(_override_color())
            self.params_table.setItem(row, 1, value_item)
        self.params_table.blockSignals(False)

    # ------------------------------------------------------------------
    def _move(self, delta: int) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        store.move(idx, delta)
        self.gallery_list.setCurrentRow(idx + delta)

    def _remove_selected(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        store.remove(idx)

    def _apply_caption_edit(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        store.set_caption(idx, self.caption_edit.text())

    def _apply_title_edit(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        self.title_edit.setEnabled(not self.hide_title_check.isChecked())
        if self.hide_title_check.isChecked():
            store.set_title_override(idx, "")
            return
        text = self.title_edit.text()
        store.set_title_override(idx, text if text else None)

    def _apply_xlabel_edit(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        text = self.xlabel_edit.text()
        store.set_xlabel_override(idx, text if text else None)

    def _apply_ylabel_edit(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        text = self.ylabel_edit.text()
        store.set_ylabel_override(idx, text if text else None)

    def _on_param_item_changed(self, table_item: QTableWidgetItem) -> None:
        if table_item.column() != 1:
            return
        idx = self._selected_index()
        if idx is None:
            return
        row = table_item.row()
        key_item = self.params_table.item(row, 0)
        if key_item is None:
            return
        value = _parse_param_text(table_item.text())
        store.set_kwarg_override(idx, key_item.text(), value)

    def _reset_params(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        store.clear_kwarg_overrides(idx)

    def _clear_all(self) -> None:
        if not store.items():
            return
        reply = QMessageBox.question(
            self, "Clear all panels", "Remove every panel from the Figure Composer?",
        )
        if reply == QMessageBox.Yes:
            store.clear()

    # ------------------------------------------------------------------
    def _rebuild_preview(self) -> None:
        self._composed = fcomp.compose_figure(
            self.plot, store.items(), cols=self.cols_spin.value(),
            label_style=self.label_style_combo.currentText(),
            label_fontsize=self.label_size_spin.value(),
        )

    def _say(self, message: str) -> None:
        bar = getattr(self.app_window, "statusBar", None)
        if callable(bar):
            bar().showMessage(message)

    def export_figure(self) -> None:
        if not store.items():
            self._say('Figure Composer: nothing to export yet. Use "Send to Figure Composer" on a chart first.')
            return
        from PySide6.QtCore import QSettings
        from qt_help import APP_NAME
        settings = QSettings(APP_NAME, APP_NAME)
        last_dir = settings.value("composer_last_export_dir", "", type=str)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", last_dir, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff)",
        )
        if not path:
            return
        settings.setValue("composer_last_export_dir", str(Path(path).resolve().parent))
        self.plot.export_at_size_cm(path, self.export_w_spin.value(), self.export_h_spin.value(), dpi=self.dpi_spin.value())
        self._say(f"Figure exported: {path}")
        QMessageBox.information(self, "Export complete", f"Saved {path}")
