"""
qt_scope.py — the global scope bar (Stage 2).

One always-visible filter row above the workspace stack: Farm, Waste
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

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

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


class GlobalScopeBar(QWidget):
    def __init__(self, scope: GlobalScope, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.scope = scope
        self.setObjectName("ScopeBar")
        self._combos: Dict[str, QComboBox] = {}

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 4, 12, 4)
        title = QLabel("Scope")
        title.setObjectName("ScopeTitle")
        title.setToolTip("Filters every workspace that queries the data. Plots and exports state the active scope.")
        row.addWidget(title)

        for column, label, fixed in SCOPE_DIMS:
            row.addWidget(QLabel(label))
            combo = QComboBox()
            combo.addItem(_ALL)
            if fixed:
                combo.addItems(fixed)
            combo.setToolTip(f"Restrict every scoped view to one {label.lower()}.")
            combo.currentTextChanged.connect(
                lambda text, col=column: self.scope.set_filter(col, None if text == _ALL else [text])
            )
            row.addWidget(combo)
            self._combos[column] = combo

        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Reset every scope filter to All.")
        clear_btn.clicked.connect(self._clear)
        row.addWidget(clear_btn)
        row.addStretch(1)

        # The stamp mirrors what plots/exports will carry, so the user sees
        # the exact wording their figure subtitle will get.
        self.stamp_label = QLabel("")
        self.stamp_label.setObjectName("ScopeStamp")
        row.addWidget(self.stamp_label)
        self.scope.changed.connect(self._update_stamp)

    def _clear(self) -> None:
        for combo in self._combos.values():
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.scope.clear()  # emits changed -> _update_stamp

    def _update_stamp(self) -> None:
        summary = self.scope.summary()
        self.stamp_label.setText(f"Scope: {summary}" if summary else "")

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
            if fixed:
                continue  # fixed lists don't repopulate
            current = combo.currentText()
            values = sorted({str(v) for v in df.get_column(column).drop_nulls().unique().to_list()})
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
        self._update_stamp()
