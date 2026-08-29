from PySide6.QtWidgets import QMainWindow

from qt_vitrification import VitrificationPage


class TestVitrificationPage:
    def test_hosts_four_tabs(self, qtbot):
        app_window = QMainWindow()
        qtbot.addWidget(app_window)
        page = VitrificationPage(app_window)
        qtbot.addWidget(page)
        titles = [page.tabs.tabText(i) for i in range(page.tabs.count())]
        assert titles == ["Screening", "Oxide Chemistry", "Candidate Search", "Blend Partners"]

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
