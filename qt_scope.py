"""
qt_scope.py — the global scope bar (Stage 2).

One slim filter strip under the application header: Farm, Waste
phase, Waste type, SST/DST. Every scoped view (Query, migrated
perspectives) honors it, and the active scope is stamped into plot
subtitles and export notes — a figure filtered to "Farm AN ·
Supernatant" must say so on its face, not in the operator's memory.

Two pieces:
- GlobalScope: the shared state object (QObject with a `changed` signal),
  handed to pages by qt_shell. Pages call .as_scope() when querying and
  .summary() when labeling.
- GlobalScopeBar: the widget driving it. Populated from the loaded
  dataset; dimensions whose column is absent (tank-attribute dims without
  Tank_attributes.csv) are disabled rather than offered-and-refused.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from data_model import HanfordDataset

# (column, display label, fixed values or None -> populate from data)
SCOPE_DIMS = [
    ("TankFarm", "Farm", None),
    ("WastePhase", "Waste phase", None),
    ("WasteType", "Waste type", None),
    ("TankSystem", "SST/DST", ["SST", "DST"]),
]

_ALL = "All"


class GlobalScope(QObject):
    """Shared scope state. Only ACTIVE filters are stored — a dimension on
    "All" simply isn't present, so `as_scope()` plugs straight into
    run_query(scope=...) with no sentinel filtering at call sites."""

    changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._filters: Dict[str, List[str]] = {}
        self._deferrals = 0
        self._pending = False

    def set_filter(self, column: str, values: Optional[List[str]]) -> None:
        """The ONE mutator. Emits `changed` unless a coalesce() block is
        open, in which case the emission is deferred to its end."""
        if not self._set(column, values):
            return
        if self._deferrals:
            self._pending = True
        else:
            self.changed.emit()

    @contextmanager
    def coalesce(self) -> Iterator["GlobalScope"]:
        """Defer `changed` until the block ends, then emit at most once.

        A dataset reload repopulates the bar and drops filters whose values
        no longer exist; emitting mid-load would make pages recompute
        against a half-swapped dataset. Deferring (rather than mutating
        silently) means listeners that are NOT dataset-aware -- a status
        stamp, an export caption -- still hear about the change exactly
        once, after the new dataset is in place."""
        self._deferrals += 1
        try:
            yield self
        finally:
            self._deferrals -= 1
            if self._deferrals == 0 and self._pending:
                self._pending = False
                self.changed.emit()

    def _set(self, column: str, values: Optional[List[str]]) -> bool:
        new = list(values) if values else None
        if new is None:
            if column not in self._filters:
                return False
            del self._filters[column]
        else:
            if self._filters.get(column) == new:
                return False
            self._filters[column] = new
        return True

    def clear(self) -> None:
        with self.coalesce():
            for column in list(self._filters):
                self.set_filter(column, None)

    def as_scope(self) -> Dict[str, List[str]]:
        return {col: list(vals) for col, vals in self._filters.items()}

    def is_active(self) -> bool:
        return bool(self._filters)

    def summary(self) -> str:
        """Human-readable stamp, e.g. "AN · Supernatant" — empty when no
        filter is active. Bar dimensions read in UI order (their values
        are self-describing); anything set programmatically beyond the
        bar's own dimensions is named, since its column isn't on screen."""
        bar_dims = [col for col, _label, _fixed in SCOPE_DIMS]
        parts = [", ".join(self._filters[col]) for col in bar_dims if col in self._filters]
        parts += [
            f"{col}: {', '.join(values)}"
            for col, values in self._filters.items() if col not in bar_dims
        ]
        return " · ".join(parts)


#: Selector widths (px), (preferred, minimum): wide enough for the usual
#: values, and independent of what the dataset holds so the strip keeps
#: one layout. In a narrow window a selector gives way down to its minimum
#: rather than setting the window's minimum width. The drop-down list
#: itself grows to its longest entry (see _fit_popup).
_COMBO_WIDTHS = {
    "TankFarm": (84, 64), "WastePhase": (212, 104), "WasteType": (196, 104), "TankSystem": (84, 64),
}


class _ElidedStamp(QLabel):
    """The scope stamp. `text()` is always the full stamp; what is drawn is
    shortened with an ellipsis when the strip has less room than the stamp
    needs, instead of being cut at an edge."""

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        rect = self.contentsRect()
        shown = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, rect.width())
        painter.setPen(self.palette().color(QPalette.WindowText))
        painter.setFont(self.font())
        painter.drawText(rect, int(self.alignment()), shown)
        painter.end()


class _ScopeCombo(QComboBox):
    """A selector whose width is set by the strip, not by its contents."""

    def __init__(self, preferred: int, minimum: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._preferred = preferred
        self._minimum = minimum
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(self._preferred, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(self._minimum, super().minimumSizeHint().height())

#: Height of the strip. The shell reserves it on every workspace, scoped or
#: not, so the page below never moves when the selectors come and go.
SCOPE_STRIP_HEIGHT = 46


class GlobalScopeBar(QWidget):
    """The slim filter strip under the application header: a caption, the
    four selectors with their field labels, "Clear" (enabled only when a
    filter is set), the number of active filters, and at the right the
    stamp that plots and exports will carry."""

    def __init__(self, scope: GlobalScope, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.scope = scope
        self.setObjectName("ScopeBar")
        self._combos: Dict[str, QComboBox] = {}

        row = QHBoxLayout(self)
        row.setContentsMargins(20, 0, 20, 0)
        row.setSpacing(6)
        title = QLabel("Scope")
        title.setObjectName("ScopeTitle")
        title.setToolTip("Filters every workspace that queries the data. Plots and exports state the active scope.")
        row.addWidget(title)
        row.addSpacing(10)

        for column, label, fixed in SCOPE_DIMS:
            field = QLabel(label)
            field.setObjectName("FieldLabel")
            row.addWidget(field)
            combo = _ScopeCombo(*_COMBO_WIDTHS.get(column, (140, 90)))
            combo.addItem(_ALL)
            if fixed:
                combo.addItems(fixed)
            combo.setToolTip(f"Restrict every scoped view to one {label.lower()}.")
            combo.currentTextChanged.connect(
                lambda text, col=column: self.scope.set_filter(col, None if text == _ALL else [text])
            )
            field.setBuddy(combo)
            row.addWidget(combo)
            row.addSpacing(8)
            self._combos[column] = combo

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("Secondary")
        self.clear_btn.setToolTip("Reset every scope filter to All.")
        self.clear_btn.clicked.connect(self._clear)
        self.clear_btn.setEnabled(False)
        row.addWidget(self.clear_btn)

        self.count_label = QLabel("")
        self.count_label.setObjectName("Pill")
        self.count_label.setToolTip("Number of scope filters in force.")
        self.count_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.count_label.setVisible(False)
        row.addWidget(self.count_label, 0, Qt.AlignVCenter)
        row.addSpacing(8)

        # The stamp mirrors what plots/exports will carry, so the user sees
        # the exact wording their figure subtitle will get. It takes the
        # room that is left and never asks for more: a long scope cannot
        # widen the window (the full text is in the tooltip).
        self.stamp_label = _ElidedStamp("")
        self.stamp_label.setObjectName("ScopeStamp")
        self.stamp_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.stamp_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.stamp_label.setMinimumWidth(0)
        row.addWidget(self.stamp_label, 1)
        self.scope.changed.connect(self._update_stamp)

    def _clear(self) -> None:
        for combo in self._combos.values():
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.scope.clear()  # emits changed -> _update_stamp
        self._update_stamp()

    def active_count(self) -> int:
        return len(self.scope.as_scope())

    def _update_stamp(self) -> None:
        summary = self.scope.summary()
        stamp = f"Scope: {summary}" if summary else ""
        self.stamp_label.setText(stamp)
        self.stamp_label.setToolTip(stamp)
        count = self.active_count()
        self.count_label.setText(f"{count} filter" + ("" if count == 1 else "s"))
        self.count_label.setVisible(count > 0)
        self.clear_btn.setEnabled(count > 0)
        self._sync_combos()

    def _sync_combos(self) -> None:
        """Make the selectors show the state. They drive it, but it can
        also be set from elsewhere (opening a project, a cross-link); a
        selector left on "All" above a filtered view would misstate what
        is on screen. A filter the selector cannot express (several
        values, or one it does not list) leaves it untouched: the stamp
        and the count still report it."""
        filters = self.scope.as_scope()
        for column, combo in self._combos.items():
            values = filters.get(column)
            if not values:
                index = 0
            elif len(values) == 1:
                index = combo.findText(values[0])
            else:
                index = -1
            if index >= 0 and index != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

    @staticmethod
    def _fit_popup(combo: QComboBox) -> None:
        """Let the drop-down list be as wide as its longest entry even
        though the closed selector has a fixed width."""
        view = combo.view()
        width = max(combo.width(), view.sizeHintForColumn(0) + 36)
        view.setMinimumWidth(width)

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        """Repopulate the combos from the new dataset and drop filters the
        new data can't honor. The shell runs this inside a
        GlobalScope.coalesce() block spanning its page dispatch, so any
        resulting `changed` lands ONCE, after every page already has the
        new dataset."""
        if not dataset.is_loaded():
            return
        df = dataset.require_df()
        for column, _label, fixed in SCOPE_DIMS:
            combo = self._combos[column]
            if column not in df.columns:
                # Offering a filter the engine would refuse (attribute dims
                # without Tank_attributes.csv) helps nobody -- disable it.
                combo.setEnabled(False)
                combo.setToolTip("Needs Tank_attributes.csv (column not in this dataset).")
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                self.scope.set_filter(column, None)
                continue
            combo.setEnabled(True)
            current = combo.currentText()
            values = sorted({str(v) for v in df.get_column(column).drop_nulls().unique().to_list()})
            if fixed:
                # A fixed list (SST/DST) keeps its entries, but its
                # SELECTION still has to be reconciled -- skipping that
                # let an SST filter survive onto an all-DST dataset, where
                # every query then returned zero rows that read as "no
                # inventory" (review finding).
                if current != _ALL and current not in values:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
                    self.scope.set_filter(column, None)
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_ALL)
            combo.addItems(values)
            restore = combo.findText(current)
            combo.setCurrentIndex(restore if restore >= 0 else 0)
            combo.blockSignals(False)
            if restore < 0 and current != _ALL:
                # The previous selection doesn't exist in the new dataset --
                # the STATE must follow the UI back to "All", or every query
                # silently filters on a value with zero rows.
                self.scope.set_filter(column, None)
        for combo in self._combos.values():
            self._fit_popup(combo)
        self._update_stamp()
