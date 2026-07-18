"""
qt_vitrification.py — Vitrification workspace: a nav-shell QTabWidget
hosting four sub-tabs (mirrors Correlations' precedent of one nav row +
many internal tabs for a complex domain). "Oxide Chemistry" is built this
milestone; "Screening", "Candidate Search", and "Blend Partners" (ported
scoring formulas with user-adjustable weights) land in the next milestone.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from data_model import HanfordDataset
from qt_vitrification_oxide import OxideChemistryTab


class _ComingSoonTab(QWidget):
    def __init__(self, name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel(f"{name} — coming in a later milestone.")
        label.setObjectName("SectionNote")
        layout.addWidget(label)


class VitrificationPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.screening_tab = _ComingSoonTab("Screening")
        self.tabs.addTab(self.screening_tab, "Screening")
        self.oxide_tab = OxideChemistryTab(app_window)
        self.tabs.addTab(self.oxide_tab, "Oxide Chemistry")
        self.candidate_tab = _ComingSoonTab("Candidate Search")
        self.tabs.addTab(self.candidate_tab, "Candidate Search")
        self.blend_tab = _ComingSoonTab("Blend Partners")
        self.tabs.addTab(self.blend_tab, "Blend Partners")

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.oxide_tab.on_dataset_changed(dataset)
