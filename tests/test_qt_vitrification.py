from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from qt_vitrification import VitrificationPage


class TestVitrificationPage:
    def test_hosts_five_tabs_in_order(self, qtbot):
        # Loading rules is appended: the four earlier tabs keep their
        # titles and their positions.
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        titles = [page.tabs.tabText(i) for i in range(page.tabs.count())]
        assert titles == ["Screening", "Oxide Chemistry", "Candidate Search", "Blend Partners", "Loading rules"]
        assert titles[:4] == ["Screening", "Oxide Chemistry", "Candidate Search", "Blend Partners"]
        assert page.tabs.currentIndex() == 0

    def test_on_dataset_changed_forwards_to_all_tabs(self, qtbot, oxide_dataset):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        page.on_dataset_changed(oxide_dataset)
        assert page.screening_tab.dataset is oxide_dataset
        assert page.oxide_tab.dataset is oxide_dataset
        assert page.candidate_tab.dataset is oxide_dataset
        assert page.blend_tab.dataset is oxide_dataset
        assert page.rules_tab.dataset is oxide_dataset

    def test_a_hidden_loading_rules_tab_computes_nothing_on_load(self, qtbot, oxide_dataset, monkeypatch):
        # The tab is engine-backed and refreshes when it is shown: a
        # dataset arriving while another tab is on screen must not start
        # its screen.
        import glass_rules
        calls = []
        original = glass_rules.screen
        monkeypatch.setattr(glass_rules, "screen", lambda *a, **k: calls.append(1) or original(*a, **k))
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        page.on_dataset_changed(oxide_dataset)
        qtbot.wait(20)
        assert not calls
        assert page.rules_tab.is_stale()


class TestJumpToBlendPartners:
    """Double-clicking a tank in Screening should feel like one continuous
    action (found a promising tank -> see what blends well with it), not a
    trip through re-entering the same tank ID in a different tab."""

    def test_double_clicking_a_screening_row_switches_tabs_and_runs_blend(self, qtbot, vitrification_dataset):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        page.on_dataset_changed(vitrification_dataset)
        page.screening_tab.run_summary()
        top_tank = page.screening_tab._table_views["Screening summary"].dataframe().iloc[0]["WasteSiteId"]

        index = page.screening_tab._table_views["Screening summary"].model().index(0, 0)
        page.screening_tab._on_summary_row_double_clicked(index)
        qtbot.wait(20)

        assert page.tabs.currentWidget() is page.blend_tab
        assert page.blend_tab.base_tank_combo.currentText() == top_tank
        assert not page.blend_tab.table.dataframe().empty


class TestTabsAreBuiltOnFirstUse:
    """The workspace is not the one the application opens on: its five
    tabs are constructed when they are first shown or asked for."""

    KEYS = ("screening", "oxide", "candidate", "blend", "rules")

    def _page(self, qtbot):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        return page

    def test_construction_builds_no_tab_and_keeps_the_five_titles(self, qtbot):
        page = self._page(qtbot)
        assert [key for key in self.KEYS if page.is_built(key)] == []
        assert page.tabs.count() == 5
        assert [page.tabs.tabText(i) for i in range(5)] == [
            "Screening", "Oxide Chemistry", "Candidate Search", "Blend Partners", "Loading rules"]
        assert all(page.tabs.tabToolTip(i) for i in range(5))

    def test_showing_the_page_builds_the_tab_on_screen_only(self, qtbot):
        page = self._page(qtbot)
        page.setAttribute(Qt.WA_DontShowOnScreen, True)
        page.show()
        qtbot.wait(20)
        try:
            assert [key for key in self.KEYS if page.is_built(key)] == ["screening"]
            page.tabs.setCurrentIndex(2)                    # what a click on the tab bar does
            assert [key for key in self.KEYS if page.is_built(key)] == ["screening", "candidate"]
            assert page.tabs.currentWidget() is page.candidate_tab
            assert page.tabs.currentIndex() == 2 and page.tabs.count() == 5
        finally:
            page.hide()

    def test_any_way_of_asking_returns_the_real_tab(self, qtbot):
        from qt_vitrification_oxide import OxideChemistryTab
        from qt_vitrification_rules import LoadingRulesTab
        page = self._page(qtbot)
        assert isinstance(page.tabs.widget(4), LoadingRulesTab)         # by position, before anything else
        assert page.tabs.widget(4) is page.rules_tab
        assert isinstance(page.oxide_tab, OxideChemistryTab)            # by name
        assert page.tabs.indexOf(page.oxide_tab) == 1
        assert page.tabs.currentWidget() is page.screening_tab          # the tab on screen
        assert page.tabs.currentIndex() == 0 and page.tabs.count() == 5
        assert page.oxide_tab is page.oxide_tab                         # built once

    def test_a_tab_built_after_the_load_gets_the_dataset(self, qtbot, vitrification_dataset):
        page = self._page(qtbot)
        page.on_dataset_changed(vitrification_dataset)
        assert not page.is_built("blend")
        assert page.blend_tab.dataset is vitrification_dataset
        assert page.blend_tab.base_tank_combo.count() == len(vitrification_dataset.available_tanks())

    def test_scope_questions_do_not_build_a_tab(self, qtbot):
        page = self._page(qtbot)
        assert page.scope_applies() is False
        page.on_scope_changed()
        assert [key for key in self.KEYS if page.is_built(key)] == []
        page.tabs.setCurrentIndex(4)
        assert page.scope_applies() is True


class TestPageSizeFollowsTheVisibleTab:
    def test_an_open_advanced_section_on_one_tab_does_not_enlarge_the_others(self, qtbot):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        page.resize(1200, 700)
        page.setAttribute(Qt.WA_DontShowOnScreen, True)
        page.show()
        qtbot.wait(20)
        try:
            width = page.tabs.width()
            screening_height = page.tabs.heightForWidth(width)
            screening_minimum = page.tabs.minimumSizeHint().height()
            page.tabs.setCurrentWidget(page.oxide_tab)
            collapsed = page.tabs.heightForWidth(width)
            page.oxide_tab.advanced_section.set_expanded(True)
            qtbot.wait(20)
            assert page.tabs.heightForWidth(width) > collapsed + 100    # the open section needs room here
            page.tabs.setCurrentWidget(page.screening_tab)
            qtbot.wait(20)
            assert page.tabs.heightForWidth(width) == screening_height  # ... and nowhere else
            assert page.tabs.minimumSizeHint().height() == screening_minimum
            assert page.tabs.hasHeightForWidth()
        finally:
            page.hide()
