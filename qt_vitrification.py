"""
qt_vitrification.py — Vitrification workspace: a nav-shell QTabWidget
hosting five sub-tabs (mirrors Correlations' precedent of one nav row +
many internal tabs for a complex domain): Screening, Oxide Chemistry,
Candidate Search, Blend Partners, and Loading rules.

Loading rules is the one sub-tab that is engine-backed and honors the
global scope bar. The page therefore follows the same protocol as
Correlations: it forwards a scope change only to the sub-tabs that honor
it, and tells the shell (scope_applies / scope_relevance_changed) whether
the sub-tab on screen is one of them, so the scope bar is shown above
that tab and nowhere else in this workspace.

The sub-tabs are built on first use, not at start-up. The workspace is
not the one the application opens on, and five tabs of charts and tables
cost a quarter of a second to build: the page starts as five titled,
empty slots, and a tab is constructed the first time it is shown or asked
for (`page.oxide_tab`, `page.tabs.widget(1)`, `page.tabs.currentWidget()`
all return the real tab, building it if need be). A tab built late is
handed the dataset the page already holds.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from data_model import HanfordDataset
from qt_scope import GlobalScope
from qt_vitrification_oxide import OxideChemistryTab
from qt_vitrification_rules import TAB_TITLE as RULES_TAB_TITLE
from qt_vitrification_rules import LoadingRulesTab
from qt_vitrification_screening import BlendPartnersTab, CandidateSearchTab, ScreeningTab

TAB_TOOLTIPS = {
    "screening": "A composition score for every tank, with the terms that built it.",
    "oxide": "The oxide-basis composition of one or more tanks, with optical basicity and NBO/T.",
    "candidate": "Rank the tanks richest in chosen target elements.",
    "blend": "Rank the tanks that would best complement one base tank in a blend.",
    "rules": (
        "Which published LAW glass loading rule would set the sodium loading of each tank's\n"
        "salt waste, and what it costs in glass. Honors the scope bar."
    ),
}


class _LazyTabWidget(QTabWidget):
    """A QTabWidget whose pages are built by their owner on first use.

    widget() and currentWidget() hand back the real tab, never the empty
    slot that stands in for it, so code written against a plain
    QTabWidget keeps working.

    Its size is that of the tab on screen. A plain QTabWidget reports the
    largest of all its pages, hidden ones included, so one tab with its
    Advanced options or its notes open gave every other tab a page scroll
    bar. The height it asks for is the least the visible tab needs at the
    width it is given: the chart and the tables take what the window has
    left, and the page scrolls only when even that does not fit."""

    def __init__(self, materialize: Callable[[int], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._materialize = materialize
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def _bar_height(self) -> int:
        return self.tabBar().sizeHint().height() + 2

    def _visible_page(self) -> Optional[QWidget]:
        return QTabWidget.currentWidget(self)           # as it is: asking for a size builds nothing

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        page = self._visible_page()
        if page is None:
            return super().minimumSizeHint()
        hint = page.minimumSizeHint()
        return QSize(max(hint.width(), 0), max(hint.height(), 0) + self._bar_height())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return self.minimumSizeHint().expandedTo(QSize(640, 420))

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt naming)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt naming)
        page = self._visible_page()
        if page is None:
            return self._bar_height()
        layout = page.layout()
        height = layout.minimumHeightForWidth(width) if layout is not None else -1
        if height < 0:
            height = page.minimumSizeHint().height()
        return max(height, 0) + self._bar_height()

    def widget(self, index: int) -> QWidget:  # noqa: D102 (Qt naming)
        self._materialize(index)
        return super().widget(index)

    def currentWidget(self) -> QWidget:  # noqa: N802 (Qt naming)
        self._materialize(self.currentIndex())
        return super().currentWidget()


class VitrificationPage(QWidget):
    # The visible sub-tab changed, and with it whether the global scope
    # applies to what is on screen (see scope_applies).
    scope_relevance_changed = Signal()

    def __init__(self, app_window, scope: Optional[GlobalScope] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        # The shell's shared scope: passed in, or read off the window that
        # owns it. A bare window (tests, tooling) gets a private, empty one
        # inside the tab.
        if scope is None:
            shared = getattr(app_window, "global_scope", None)
            scope = shared if isinstance(shared, GlobalScope) else None
        self._scope = scope
        self._dataset: Optional[HanfordDataset] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = _LazyTabWidget(self._materialize_index)
        layout.addWidget(self.tabs)

        # key -> (title, the tab's class, how to build it). Loading rules is
        # appended, not inserted: the four tabs before it keep their positions.
        self._specs: Dict[str, tuple] = {
            "screening": ("Screening", ScreeningTab, lambda: ScreeningTab(app_window)),
            "oxide": ("Oxide Chemistry", OxideChemistryTab, lambda: OxideChemistryTab(app_window)),
            "candidate": ("Candidate Search", CandidateSearchTab, lambda: CandidateSearchTab(app_window)),
            "blend": ("Blend Partners", BlendPartnersTab, lambda: BlendPartnersTab(app_window)),
            "rules": (RULES_TAB_TITLE, LoadingRulesTab, lambda: LoadingRulesTab(app_window, scope=self._scope)),
        }
        self._order: List[str] = list(self._specs)
        self._built: Dict[str, QWidget] = {}
        self._slots: Dict[str, QWidget] = {}
        for key in self._order:
            slot = QWidget()
            self._slots[key] = slot
            index = self.tabs.addTab(slot, self._specs[key][0])
            self.tabs.setTabToolTip(index, TAB_TOOLTIPS[key])

        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------
    # Tabs, built on first use
    # ------------------------------------------------------------------
    @property
    def screening_tab(self) -> ScreeningTab:
        return self._tab("screening")

    @property
    def oxide_tab(self) -> OxideChemistryTab:
        return self._tab("oxide")

    @property
    def candidate_tab(self) -> CandidateSearchTab:
        return self._tab("candidate")

    @property
    def blend_tab(self) -> BlendPartnersTab:
        return self._tab("blend")

    @property
    def rules_tab(self) -> LoadingRulesTab:
        return self._tab("rules")

    def is_built(self, key: str) -> bool:
        """Whether the tab `key` (screening, oxide, candidate, blend, rules) exists yet."""
        return key in self._built

    def _materialize_index(self, index: int) -> None:
        if 0 <= index < len(self._order):
            self._tab(self._order[index])

    def _tab(self, key: str) -> QWidget:
        tab = self._built.get(key)
        if tab is not None:
            return tab
        title, _cls, build = self._specs[key]
        tab = build()
        self._built[key] = tab
        slot = self._slots.pop(key)
        tabs = self.tabs
        index = tabs.indexOf(slot)
        was_current = tabs.currentIndex() == index
        blocked = tabs.blockSignals(True)       # the swap is not a change of tab
        try:
            # Insert before the slot, then drop the slot: the tab on screen
            # never changes to a neighbour in between.
            tabs.insertTab(index, tab, title)
            tabs.setTabToolTip(index, TAB_TOOLTIPS[key])
            if was_current:
                tabs.setCurrentIndex(index)
            tabs.removeTab(index + 1)
        finally:
            tabs.blockSignals(blocked)
        slot.deleteLater()
        if key == "screening":
            tab.jump_to_blend_partners.connect(self._jump_to_blend_partners)
        if self._dataset is not None:
            tab.on_dataset_changed(self._dataset)
        return tab

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # The tab on screen exists before the page is painted.
        self._materialize_index(self.tabs.currentIndex())
        super().showEvent(event)

    def _on_tab_changed(self, index: int) -> None:
        self._materialize_index(index)
        self.scope_relevance_changed.emit()

    def _jump_to_blend_partners(self, tank_id: str) -> None:
        """Double-clicking a tank in Screening summary lands here with a
        real tank -- the connective tissue that lets "I found a promising
        tank" flow straight into "what would blend well with it" without
        re-entering the tank ID in a different tab."""
        self.tabs.setCurrentWidget(self.blend_tab)
        self.blend_tab.set_base_tank_and_run(tank_id)

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        """Every tab gets the dataset: the built ones now, the others when they are built."""
        self._dataset = dataset
        for key in self._order:
            tab = self._built.get(key)
            if tab is not None:
                tab.on_dataset_changed(dataset)

    def on_scope_changed(self) -> None:
        """Forward a global scope change to the sub-tabs that honor it
        (derived from the protocol, so a legacy tab joins when it migrates
        onto the engine). Today that is Loading rules alone; the other
        four compute on the whole dataset and are not told. A tab that is
        not built yet has nothing to refresh: it reads the scope when it
        first runs."""
        for key in self._order:
            tab = self._built.get(key)
            if tab is not None and hasattr(tab, "on_scope_changed"):
                tab.on_scope_changed()

    def scope_applies(self) -> bool:
        """Whether the sub-tab on screen honors the global scope. The
        shell shows the scope bar only then: a filter displayed above a
        tab that ignores it would misdescribe what is on screen. Read off
        the tab's class, so asking does not build the tab."""
        index = self.tabs.currentIndex()
        if not (0 <= index < len(self._order)):
            return False
        return hasattr(self._specs[self._order[index]][1], "on_scope_changed")
