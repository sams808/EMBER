"""
qt_vitrification.py — Vitrification workspace: a nav-shell QTabWidget
hosting four sub-tabs (mirrors Correlations' precedent of one nav row +
many internal tabs for a complex domain): Screening, Oxide Chemistry,
Candidate Search, and Blend Partners.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from data_model import HanfordDataset
from qt_vitrification_oxide import OxideChemistryTab
from qt_vitrification_screening import BlendPartnersTab, CandidateSearchTab, ScreeningTab


class VitrificationPage(QWidget):
    def __init__(self, app_window, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_window = app_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.screening_tab = ScreeningTab(app_window)
        self.tabs.addTab(self.screening_tab, "Screening")
        self.oxide_tab = OxideChemistryTab(app_window)
        self.tabs.addTab(self.oxide_tab, "Oxide Chemistry")
        self.candidate_tab = CandidateSearchTab(app_window)
        self.tabs.addTab(self.candidate_tab, "Candidate Search")
        self.blend_tab = BlendPartnersTab(app_window)
        self.tabs.addTab(self.blend_tab, "Blend Partners")

        self.screening_tab.jump_to_blend_partners.connect(self._jump_to_blend_partners)

    def _jump_to_blend_partners(self, tank_id: str) -> None:
        """Double-clicking a tank in Screening summary lands here with a
        real tank -- the connective tissue that lets "I found a promising
        tank" flow straight into "what would blend well with it" without
        re-entering the tank ID in a different tab."""
        self.tabs.setCurrentWidget(self.blend_tab)
        self.blend_tab.base_tank_combo.setCurrentText(tank_id)
        self.blend_tab.run_blend()

    def on_dataset_changed(self, dataset: HanfordDataset) -> None:
        self.screening_tab.on_dataset_changed(dataset)
        self.oxide_tab.on_dataset_changed(dataset)
        self.candidate_tab.on_dataset_changed(dataset)
        self.blend_tab.on_dataset_changed(dataset)
