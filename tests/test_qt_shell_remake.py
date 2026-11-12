"""
The application frame after the interface remake: the header, the grouped
sidebar, the menus, the shell's navigation API (open_tank, open_page,
open_element, the global search), the busy indicator, the cheap refresh of
hidden workspaces, and the guarantee that no page can grow the window.
"""
import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QStackedWidget

import data_model as dm
import qt_shell
from qt_shell import (
    NAV_FIGURE_COMPOSER, NAV_GROUPS, NAV_HEATMAPS, NAV_ITEMS, NAV_OVERVIEW,
    NAV_QUERY, EmberMainWindow,
)


def _text_scale() -> float:
    """How wide UI text is on this Qt platform, relative to the native one.

    The offscreen platform has no Segoe UI and falls back to a font about
    twice as wide, so a pixel bound that holds on screen is scaled by this
    factor before it is asserted (1.0 on the native platform)."""
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QApplication
    reference = "Scope Farm Waste phase Waste type SST/DST Clear"
    return max(1.0, QFontMetrics(QApplication.font()).horizontalAdvance(reference) / 272.0)


CSV = (
    "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units\n"
    "241-A-101,Fe,Liquid,T1,50.0,kg\n"
    "241-A-101,Fe,Solid,T1,10.0,kg\n"
    "241-A-101,137Cs,Liquid,T1,100.0,Ci\n"
    "241-AN-104,Na,Sludge,T2,500.0,kg\n"
    "241-AN-104,137Cs,Sludge,T2,200.0,Ci\n"
)


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "app_base_dir", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    win = EmberMainWindow()
    qtbot.addWidget(win)
    qtbot.wait(50)
    return win


@pytest.fixture
def loaded(window, tmp_path, qtbot):
    path = tmp_path / "inventory.csv"
    path.write_text(CSV)
    window._start_load(lambda: window.dataset.load(path, use_cache=False))
    qtbot.wait(50)
    assert window.dataset.is_loaded()
    return window


@pytest.fixture
def on_screen(loaded, qtbot):
    """The loaded window, shown (off the real screen), which is when the
    shell defers the refresh of hidden workspaces."""
    loaded.setAttribute(Qt.WA_DontShowOnScreen, True)
    loaded.show()
    qtbot.wait(20)
    yield loaded
    loaded.hide()


def _count_calls(monkeypatch, page, method):
    calls = []
    original = getattr(page, method)

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(page, method, counted, raising=False)
    return calls


class TestFrame:
    def test_default_and_minimum_size(self, window):
        available = window.screen().availableGeometry().size()
        assert window.width() == min(1440, max(640, available.width() - 40))
        assert window.height() == min(900, max(480, available.height() - 40))
        assert window.minimumWidth() <= 1100 and window.minimumHeight() <= 700

    def test_header_names_the_product_and_the_version_is_in_the_status_bar(self, window):
        assert window.findChild(QLabel, "AppTitle").text() == "EMBER"
        assert window.findChild(QLabel, "AppSubtitle").text() == "Hanford tank waste explorer"
        assert qt_shell.APP_VERSION in window.status_version_label.text()

    def test_no_agency_name_in_the_frame(self, window):
        # The product must not present itself as an agency's: no such name
        # in the header, the sidebar, the menus or the status bar.
        texts = [label.text() for label in window.findChildren(QLabel)
                 if not any(page.isAncestorOf(label) for page in window._pages_by_nav.values())]
        texts += [action.text() for action in window.menuBar().actions()]
        for menu_action in window.menuBar().actions():
            texts += [a.text() for a in menu_action.menu().actions()]
        blob = " ".join(texts)
        for name in ("DOE", "PNNL", "Department of Energy", "Pacific Northwest"):
            assert name not in blob

    def test_dataset_chip_says_calmly_that_nothing_is_loaded(self, window):
        assert window.dataset_chip.full_text() == "No dataset loaded"
        assert window.status_dataset_label.text() == "No dataset loaded"

    def test_dataset_chip_states_what_is_loaded(self, loaded):
        text = loaded.dataset_chip.full_text()
        assert "inventory.csv" in text
        assert "5 records" in text          # 5 rows in CSV
        assert "2 tanks" in text            # 241-A-101 and 241-AN-104
        assert "5 records" in loaded.status_dataset_label.text()

    def test_dataset_summary_reads_the_date_columns_when_present(self, window, tmp_path, qtbot):
        path = tmp_path / "dated.csv"
        path.write_text(
            "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units,DecayDate,PublishedDate\n"
            "241-A-101,Fe,Liquid,T1,50.0,kg,2022/01/01 12:00:00 AM,2019/09/25 1:04:10 PM\n"
            "241-A-101,Na,Liquid,T1,10.0,kg,2015/07/01 12:00:00 AM,2025/12/31 2:12:33 PM\n"
        )
        window._start_load(lambda: window.dataset.load(path, use_cache=False))
        qtbot.wait(50)
        info = window.dataset_summary()
        assert info["published"] == ["2019-09-25", "2025-12-31"]
        assert info["decay"] == ["2015-07-01", "2022-01-01"]
        assert "published 2019–2025" in window.dataset_chip.full_text()
        assert "decay date 2015 / 2022" in window.dataset_chip.full_text()
        assert "2015-07-01, 2022-01-01" in window.dataset_chip.toolTip()

    def test_dataset_summary_is_empty_without_data(self, window):
        assert window.dataset_summary() == {}

    def test_the_developer_toolbar_is_gone(self, window):
        from PySide6.QtWidgets import QToolBar
        assert [bar for bar in window.findChildren(QToolBar)
                if not any(page.isAncestorOf(bar) for page in window._pages_by_nav.values())] == []


class TestMenus:
    def test_menu_bar_reads_file_data_view_help(self, window):
        titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
        assert titles == ["File", "Data", "View", "Help"]

    def test_cache_options_are_checkable_actions_the_loader_reads(self, window):
        assert window.use_cache_check.isCheckable() and window.use_cache_check.isChecked()
        assert window.refresh_cache_check.isCheckable() and not window.refresh_cache_check.isChecked()

    def test_the_cache_options_reach_the_loader(self, window, monkeypatch):
        seen = {}
        monkeypatch.setattr(window.dataset, "load_local_default",
                            lambda use_cache, refresh_cache: seen.update(use=use_cache, refresh=refresh_cache))
        monkeypatch.setattr(window, "_on_load_done", lambda report: None)
        window.use_cache_check.setChecked(False)
        window.refresh_cache_check.setChecked(True)
        window._load_local()
        assert seen == {"use": False, "refresh": True}

    def test_rebuild_the_cache_is_a_one_shot_not_a_mode(self, window, tmp_path, qtbot):
        path = tmp_path / "inventory.csv"
        path.write_text(CSV)
        window.refresh_cache_check.setChecked(True)
        window._start_load(lambda: window.dataset.load(path, use_cache=False))
        qtbot.wait(50)
        assert not window.refresh_cache_check.isChecked()

    def test_no_developer_vocabulary_in_the_menus(self, window):
        words = []
        for menu_action in window.menuBar().actions():
            words += [a.text().lower() for a in menu_action.menu().actions()]
        blob = " ".join(words)
        assert "parquet" not in blob and "csvs" not in blob

    def test_every_workspace_has_a_view_menu_entry_that_opens_it(self, window):
        assert [a.text() for a in window.workspace_actions] == NAV_ITEMS
        window.workspace_actions[NAV_ITEMS.index(NAV_HEATMAPS)].trigger()
        assert window.stack.currentWidget() is window.heatmaps_page

    def test_export_current_view_calls_the_page_export(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(window.query_page, "_export", lambda: calls.append("query"))
        window.open_page(NAV_QUERY)
        assert window.export_current_page() is True
        assert calls == ["query"]

    def test_export_current_view_prefers_the_visible_sub_tab(self, window, monkeypatch):
        page = window.correlations_page
        calls = []
        monkeypatch.setattr(page.association_tab, "export_tables", lambda: calls.append("association"))
        window.open_page("Correlations")
        page.tabs.setCurrentWidget(page.association_tab)
        assert window.export_current_page() is True
        assert calls == ["association"]


class TestSidebar:
    def test_groups_cover_every_workspace_once_in_nav_order(self):
        flat = [name for _caption, names in NAV_GROUPS for name in names]
        assert flat == NAV_ITEMS

    def test_one_row_per_workspace_whatever_the_grouping(self, window):
        assert window.nav.count() == len(NAV_ITEMS)
        assert [window.nav.item(i).text() for i in range(window.nav.count())] == NAV_ITEMS

    def test_first_row_of_a_group_is_taller_by_its_caption(self, window):
        from PySide6.QtWidgets import QStyleOptionViewItem
        delegate = window.nav.itemDelegate()
        option = QStyleOptionViewItem()
        heights = [delegate.sizeHint(option, window.nav.model().index(row, 0)).height()
                   for row in range(len(NAV_ITEMS))]
        first_rows = {NAV_ITEMS.index(names[0]) for _caption, names in NAV_GROUPS}
        plain = {heights[row] for row in range(len(NAV_ITEMS)) if row not in first_rows}
        assert plain == {delegate.ROW_HEIGHT}
        for row in first_rows:
            assert heights[row] > delegate.ROW_HEIGHT

    def test_collapsing_narrows_the_sidebar_and_keeps_the_rows(self, window):
        assert window.sidebar.maximumWidth() == qt_shell.SIDEBAR_WIDTH
        window.collapse_action.setChecked(True)
        assert window.sidebar.maximumWidth() == qt_shell.SIDEBAR_COLLAPSED_WIDTH
        assert window.nav.count() == len(NAV_ITEMS)
        window.nav.setCurrentRow(3)
        assert window.stack.currentWidget() is window._pages_by_nav[NAV_ITEMS[3]]
        window.collapse_action.setChecked(False)
        assert window.sidebar.maximumWidth() == qt_shell.SIDEBAR_WIDTH

    def test_painting_the_nav_in_both_modes_and_themes_does_not_raise(self, window):
        window.nav.resize(216, 500)
        for collapsed in (False, True):
            window.collapse_action.setChecked(collapsed)
            for dark in (True, False):
                window.dark_mode_action.setChecked(dark)
                assert not window.nav.grab().isNull()


class TestScopeStrip:
    def test_strip_keeps_its_height_on_scoped_and_unscoped_workspaces(self, window, qtbot):
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.show()
        try:
            window.open_page(NAV_QUERY)
            qtbot.wait(10)
            scoped_top = window.page_scroll.mapTo(window, window.page_scroll.rect().topLeft()).y()
            assert window.scope_bar.isVisible() and not window.scope_idle.isVisible()
            window.open_page(NAV_FIGURE_COMPOSER)
            qtbot.wait(10)
            unscoped_top = window.page_scroll.mapTo(window, window.page_scroll.rect().topLeft()).y()
            assert not window.scope_bar.isVisible() and window.scope_idle.isVisible()
            assert scoped_top == unscoped_top            # the page does not jump
            assert window.scope_strip.height() == qt_shell.SCOPE_STRIP_HEIGHT
        finally:
            window.hide()

    def test_clear_is_enabled_only_when_a_filter_is_set(self, loaded, qtbot):
        bar = loaded.scope_bar
        assert not bar.clear_btn.isEnabled()
        assert bar.count_label.isHidden()
        loaded.global_scope.set_filter("WastePhase", ["Sludge"])
        assert bar.clear_btn.isEnabled()
        assert bar.count_label.text() == "1 filter"
        loaded.global_scope.set_filter("TankFarm", ["AN"])
        assert bar.count_label.text() == "2 filters"
        bar._clear()
        assert not bar.clear_btn.isEnabled()
        assert bar.count_label.isHidden()

    def test_selectors_follow_a_scope_set_elsewhere(self, loaded):
        # Opening a project (or a cross-link) sets the state directly; a
        # selector left on "All" above a filtered view would misstate it.
        loaded.global_scope.set_filter("WastePhase", ["Sludge"])
        assert loaded.scope_bar._combos["WastePhase"].currentText() == "Sludge"
        loaded.global_scope.clear()
        assert loaded.scope_bar._combos["WastePhase"].currentText() == "All"


class TestOpenPage:
    def test_open_page_switches_workspace(self, window):
        assert window.open_page(NAV_HEATMAPS) is True
        assert window.nav.currentRow() == NAV_ITEMS.index(NAV_HEATMAPS)
        assert window.stack.currentWidget() is window.heatmaps_page

    def test_unknown_name_is_refused_and_nothing_moves(self, window):
        before = window.stack.currentWidget()
        assert window.open_page("Not A Workspace") is False
        assert window.stack.currentWidget() is before


class TestOpenTank:
    def test_selects_the_tank_and_runs_the_tank_explorer(self, loaded, monkeypatch):
        page = loaded.tank_explorer_page
        if hasattr(page, "show_tank"):
            calls = _count_calls(monkeypatch, page, "show_tank")
            assert loaded.open_tank("241-AN-104") is True
            assert [args[0] for args in calls] == ["241-AN-104"]
        else:
            runs = _count_calls(monkeypatch, page, "run")
            assert loaded.open_tank("241-AN-104") is True
            assert page.selected_tanks() == ["241-AN-104"]
            assert len(runs) == 1
        assert loaded.stack.currentWidget() is page

    def test_fallback_selects_in_the_list_and_runs(self, loaded):
        # The path used while the page has no show_tank of its own.
        from PySide6.QtWidgets import QListWidget, QWidget

        class LegacyPage(QWidget):
            def __init__(self):
                super().__init__()
                self.tank_list = QListWidget(self)
                self.tank_list.setSelectionMode(QListWidget.ExtendedSelection)
                self.tank_list.addItems(["241-A-101", "241-AN-104"])
                self.tank_list.item(0).setSelected(True)
                self.runs = 0

            def run(self):
                self.runs += 1

        page = LegacyPage()
        assert loaded._show_tank_by_selection(page, "241-AN-104") is True
        assert [i.text() for i in page.tank_list.selectedItems()] == ["241-AN-104"]
        assert page.runs == 1

    def test_fallback_reports_an_unknown_tank_and_does_not_run(self, loaded):
        from PySide6.QtWidgets import QListWidget, QWidget

        class LegacyPage(QWidget):
            def __init__(self):
                super().__init__()
                self.tank_list = QListWidget(self)
                self.tank_list.addItems(["241-A-101"])
                self.runs = 0

            def run(self):
                self.runs += 1

        page = LegacyPage()
        assert loaded._show_tank_by_selection(page, "241-ZZ-999") is False
        assert page.runs == 0
        assert "241-ZZ-999" in loaded.statusBar().currentMessage()

    def test_a_page_with_show_tank_is_handed_the_id(self, loaded, monkeypatch):
        seen = []
        monkeypatch.setattr(loaded.tank_explorer_page, "show_tank", seen.append, raising=False)
        assert loaded.open_tank("241-A-101") is True
        assert seen == ["241-A-101"]
        assert loaded.stack.currentWidget() is loaded.tank_explorer_page

    def test_show_tank_returning_false_is_reported(self, loaded, monkeypatch):
        monkeypatch.setattr(loaded.tank_explorer_page, "show_tank", lambda tank_id: False, raising=False)
        assert loaded.open_tank("241-A-101") is False

    def test_without_data_it_does_nothing_and_says_why(self, window):
        before = window.stack.currentWidget()
        assert window.open_tank("241-A-101") is False
        assert window.stack.currentWidget() is before
        assert "dataset" in window.statusBar().currentMessage().lower()

    def test_blank_id_is_refused(self, loaded):
        assert loaded.open_tank("") is False
        assert loaded.open_tank(None) is False


class TestOpenElementAndSearch:
    def test_open_element_fills_the_explorer_and_runs_it(self, loaded, monkeypatch):
        page = loaded.explorer_page
        runs = _count_calls(monkeypatch, page, "run_search")
        assert loaded.open_element("Fe") is True
        assert loaded.stack.currentWidget() is page
        assert page.query_edit.text() == "Fe"
        assert len(runs) == 1

    def test_search_hit_for_a_tank_opens_the_tank(self, loaded, monkeypatch):
        opened = []
        monkeypatch.setattr(loaded, "open_tank", lambda tank_id: opened.append(tank_id) or True)
        hits = loaded.search_index.search("AN-104")
        assert (hits[0].kind, hits[0].key) == ("tank", "241-AN-104")
        assert loaded.open_hit(hits[0]) is True
        assert opened == ["241-AN-104"]

    def test_search_hit_for_an_analyte_goes_to_the_explorer(self, loaded, monkeypatch):
        seen = []
        monkeypatch.setattr(loaded, "open_element", lambda subject, kind="element": seen.append((subject, kind)) or True)
        hit = loaded.search_index.search("137Cs")[0]
        loaded.open_hit(hit)
        assert seen == [("137Cs", "analyte")]

    def test_search_hit_for_a_workspace_switches_to_it(self, loaded):
        hit = loaded.search_index.search("heatmaps")[0]
        assert loaded.open_hit(hit) is True
        assert loaded.stack.currentWidget() is loaded.heatmaps_page

    def test_typing_and_enter_in_the_header_box_opens_the_match(self, loaded, qtbot):
        loaded.search_box.setText("composer")
        loaded.search_box.show_results()
        loaded.search_box.activate_current()
        assert loaded.stack.currentWidget() is loaded.figure_composer_page
        assert loaded.search_box.text() == ""

    def test_the_index_is_refilled_on_load(self, window, loaded):
        counts = loaded.search_index.counts()
        assert counts["tank"] == 2 and counts["workspace"] == len(NAV_ITEMS)

    def test_ctrl_k_is_the_search_shortcut(self, window):
        assert window.search_action.shortcut().toString() == "Ctrl+K"
        window.focus_search()   # must not raise on a window that is not shown


class TestBusy:
    def test_set_busy_shows_and_clears_the_indicator(self, window):
        assert not window.is_busy()
        window.set_busy("Computing correlations…")
        assert window.is_busy()
        assert window.busy_label.text() == "Computing correlations…"
        assert not window.busy_label.isHidden() and not window.busy_bar.isHidden()
        window.set_busy(None)
        assert not window.is_busy()
        assert window.busy_label.isHidden() and window.busy_bar.isHidden()

    def test_a_finished_load_leaves_no_busy_indicator(self, loaded):
        assert not loaded.is_busy()
        assert loaded.load_local_action.isEnabled() and loaded.browse_action.isEnabled()

    def test_busy_bar_paints_in_both_themes(self, window):
        window.set_busy("Working…")
        for dark in (True, False):
            window.dark_mode_action.setChecked(dark)
            assert not window.busy_bar.grab().isNull()
        window.set_busy(None)


class TestCheapRefreshOfHiddenWorkspaces:
    def test_scope_change_refreshes_only_the_workspace_on_screen(self, on_screen, monkeypatch):
        window = on_screen
        window.open_page(NAV_QUERY)
        query = _count_calls(monkeypatch, window.query_page, "on_scope_changed")
        overview = _count_calls(monkeypatch, window.overview_page, "on_scope_changed")
        heatmaps = _count_calls(monkeypatch, window.heatmaps_page, "on_scope_changed")
        window.global_scope.set_filter("WastePhase", ["Sludge"])
        assert len(query) == 1
        assert overview == [] and heatmaps == []
        assert window.overview_page in window._scope_stale

    def test_a_stale_workspace_refreshes_once_when_shown(self, on_screen, monkeypatch):
        window = on_screen
        window.open_page(NAV_QUERY)
        heatmaps = _count_calls(monkeypatch, window.heatmaps_page, "on_scope_changed")
        window.global_scope.set_filter("WastePhase", ["Sludge"])
        window.global_scope.set_filter("TankFarm", ["AN"])      # two changes while hidden
        assert heatmaps == []
        window.open_page(NAV_HEATMAPS)
        assert len(heatmaps) == 1                               # ... one refresh
        window.open_page(NAV_QUERY)
        window.open_page(NAV_HEATMAPS)
        assert len(heatmaps) == 1                               # and not again
        assert window.heatmaps_page not in window._scope_stale

    def test_the_deferred_refresh_gives_the_same_result_as_the_eager_one(self, on_screen, qtbot):
        window = on_screen
        page = window.query_page
        window.open_page(NAV_QUERY)
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("Fe")
        page.basis_combo.setCurrentText("kg")
        page.run()
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(60.0)   # 50 + 10
        window.open_page(NAV_OVERVIEW)
        window.global_scope.set_filter("WastePhase", ["Solid"])
        # Hidden: the standing result is still the unscoped one ...
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(60.0)
        window.open_page(NAV_QUERY)
        # ... and showing the page re-asks it under the new scope.
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(10.0)
        assert "Scope: Solid" in page.plot.ax.get_title()

    def test_a_window_that_is_not_shown_refreshes_every_workspace_at_once(self, loaded, monkeypatch):
        # No "when it is shown" to wait for: scripts and tests read hidden
        # pages right after a change.
        heatmaps = _count_calls(monkeypatch, loaded.heatmaps_page, "on_scope_changed")
        loaded.open_page(NAV_QUERY)
        loaded.global_scope.set_filter("WastePhase", ["Sludge"])
        assert len(heatmaps) == 1
        assert loaded._scope_stale == set()

    def test_lazy_refresh_can_be_switched_off(self, on_screen, monkeypatch):
        window = on_screen
        window.lazy_refresh = False
        window.open_page(NAV_QUERY)
        heatmaps = _count_calls(monkeypatch, window.heatmaps_page, "on_scope_changed")
        window.global_scope.set_filter("WastePhase", ["Sludge"])
        assert len(heatmaps) == 1

    def test_a_new_dataset_reaches_hidden_workspaces_when_they_are_shown(self, on_screen, tmp_path, monkeypatch, qtbot):
        window = on_screen
        window.open_page(NAV_OVERVIEW)
        overview = _count_calls(monkeypatch, window.overview_page, "on_dataset_changed")
        query = _count_calls(monkeypatch, window.query_page, "on_dataset_changed")
        path = tmp_path / "second.csv"
        path.write_text(CSV)
        window._start_load(lambda: window.dataset.load(path, use_cache=False))
        qtbot.wait(50)
        assert len(overview) == 1 and query == []
        assert window.query_page in window._dataset_stale
        window.open_page(NAV_QUERY)
        assert len(query) == 1 and query[0][0] is window.dataset
        assert window.query_page not in window._dataset_stale

    def test_a_cross_link_into_a_stale_workspace_delivers_the_dataset_first(self, on_screen, tmp_path, qtbot):
        window = on_screen
        window.open_page(NAV_OVERVIEW)
        path = tmp_path / "second.csv"
        path.write_text(CSV)
        window._start_load(lambda: window.dataset.load(path, use_cache=False))
        qtbot.wait(50)
        assert window.query_page in window._dataset_stale
        assert window.ask_in_query("element", "Fe", "kg") is True
        assert window.query_page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(60.0)

    def test_saving_a_project_does_not_keep_a_question_from_the_previous_dataset(self, on_screen, tmp_path, qtbot):
        window = on_screen
        window.open_page(NAV_QUERY)
        page = window.query_page
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("Fe")
        page.basis_combo.setCurrentText("kg")
        page.run()
        assert len(window.current_project().questions) == 1
        window.open_page(NAV_OVERVIEW)
        path = tmp_path / "second.csv"
        path.write_text(CSV)
        window._start_load(lambda: window.dataset.load(path, use_cache=False))
        qtbot.wait(50)
        # The Query page is hidden and has not seen the new dataset yet; its
        # standing result belongs to the old one and must not be saved.
        assert window.current_project().questions == []

    def test_ensure_page_current_accepts_a_name(self, on_screen, monkeypatch):
        window = on_screen
        window.open_page(NAV_QUERY)
        heatmaps = _count_calls(monkeypatch, window.heatmaps_page, "on_scope_changed")
        window.global_scope.set_filter("WastePhase", ["Sludge"])
        window.ensure_page_current(NAV_HEATMAPS)
        assert len(heatmaps) == 1


class TestWindowSizeIsNotSetByThePages:
    LONG = "Not shown: " + "; ".join(f"analyte number {i} the parser cannot resolve" for i in range(120))

    def test_stack_lives_in_a_scroll_area_and_still_returns_the_page(self, window):
        assert isinstance(window.stack, QStackedWidget)
        assert isinstance(window.page_scroll, QScrollArea)
        assert window.page_scroll.widget() is window.stack
        assert window.page_scroll.widgetResizable()
        assert window.stack.currentWidget() is window.overview_page

    def test_a_very_long_label_cannot_raise_the_window_minimum(self, window, qtbot):
        label = QLabel(self.LONG)              # one line, several thousand pixels wide
        window.query_page.layout().addWidget(label)
        window.open_page(NAV_QUERY)
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.show()
        try:
            qtbot.wait(30)
            assert label.minimumSizeHint().width() > 3000
            assert window.query_page.minimumSizeHint().width() > 3000
            hint = window.minimumSizeHint()
            # Far below the 3000 px the label asks for, on any platform's font.
            assert hint.width() < min(1200 * _text_scale(), 2800) and hint.height() < 760
            assert window.width() <= 1440 and window.height() <= 900
            # The page area caps what a page may ask for.
            assert window.stack.minimumSizeHint().width() == qt_shell.PAGE_MAX_MIN_WIDTH
        finally:
            window.hide()

    def test_a_very_tall_page_scrolls_instead_of_growing_the_window(self, window, qtbot):
        from PySide6.QtWidgets import QWidget
        tall = QWidget()
        tall.setMinimumSize(QSize(200, 3000))
        window.query_page.layout().addWidget(tall)
        window.open_page(NAV_QUERY)
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.show()
        try:
            qtbot.wait(30)
            assert window.minimumSizeHint().height() < 760
            assert window.height() <= 900
            assert window.page_scroll.verticalScrollBar().maximum() > 0     # it scrolls
            assert window.stack.height() >= 3000
        finally:
            window.hide()

    def test_only_the_page_on_screen_counts(self, window):
        from PySide6.QtWidgets import QWidget
        tall = QWidget()
        tall.setMinimumSize(QSize(200, 3000))
        window.query_page.layout().addWidget(tall)
        window.open_page(NAV_OVERVIEW)
        assert window.stack.minimumSizeHint().height() < 3000
        window.open_page(NAV_QUERY)
        assert window.stack.minimumSizeHint().height() >= 3000

    def test_theme_toggle_leaves_the_minimum_alone(self, window):
        before = window.minimumSizeHint()
        window.dark_mode_action.setChecked(True)
        after = window.minimumSizeHint()
        window.dark_mode_action.setChecked(False)
        assert after.width() < 1200 * _text_scale() and after.height() < 760
        assert abs(after.height() - before.height()) < 40
