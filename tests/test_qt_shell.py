"""
Qt-level tests for the app shell: nav-by-name resolution and the
load -> broadcast-to-pages flow. qt_worker's synchronous-mode autouse
fixture (conftest.py) makes run_in_thread calls execute inline, so no
real background thread or wait-loop is needed here.
"""
import pytest

import data_model as dm
from qt_shell import (
    NAV_DESCRIPTIONS, NAV_EXPLORER, NAV_ITEMS, NAV_OVERVIEW, NAV_QUERY,
    EmberMainWindow,
)


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    # Redirect the startup auto-load to an empty folder: shell/nav tests
    # shouldn't depend on (or be slowed by) the real dev-seed Hanford.csv
    # that happens to sit in this repo. Both search locations matter --
    # load_local_default() also falls back to Path.cwd(), which (unpatched)
    # would still be this repo's own root under pytest.
    monkeypatch.setattr(dm, "app_base_dir", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    win = EmberMainWindow()
    qtbot.addWidget(win)
    qtbot.wait(50)  # let QTimer.singleShot(0, auto-load) fire and fail harmlessly
    return win


class TestNavSwitching:
    def test_starts_on_overview(self, window):
        assert window.nav.currentRow() == 0
        assert window.stack.currentWidget() is window.overview_page

    def test_switching_updates_stack_by_name(self, window):
        explorer_row = NAV_ITEMS.index(NAV_EXPLORER)
        window.nav.setCurrentRow(explorer_row)
        assert window.stack.currentWidget() is window.explorer_page

    def test_all_nav_rows_resolve_to_distinct_pages(self, window):
        seen = set()
        for row, name in enumerate(NAV_ITEMS):
            window.nav.setCurrentRow(row)
            page = window.stack.currentWidget()
            assert page is window._pages_by_nav[name]
            seen.add(id(page))
        assert len(seen) == len(NAV_ITEMS)

    def test_out_of_range_row_is_ignored(self, window):
        current = window.stack.currentWidget()
        window._on_nav_changed(-1)
        window._on_nav_changed(999)
        assert window.stack.currentWidget() is current


class TestNavDescriptions:
    def test_every_nav_item_has_a_description(self):
        # Guards against a new nav workspace being added to NAV_ITEMS
        # without also getting a hover description -- easy to forget since
        # nothing else would break.
        for name in NAV_ITEMS:
            assert name in NAV_DESCRIPTIONS
            assert NAV_DESCRIPTIONS[name].strip()

    def test_no_stale_descriptions_for_removed_items(self):
        assert set(NAV_DESCRIPTIONS) == set(NAV_ITEMS)

    def test_nav_list_items_carry_the_tooltip(self, window):
        for row, name in enumerate(NAV_ITEMS):
            assert window.nav.item(row).toolTip() == NAV_DESCRIPTIONS[name]


class TestDatasetLoadBroadcast:
    def _write_csv(self, tmp_path):
        path = tmp_path / "custom.csv"
        path.write_text(
            "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units\n"
            "241-A-101,137Cs,Liquid,T1,100.0,Ci\n"
        )
        return path

    def test_successful_load_broadcasts_to_overview_page(self, window, tmp_path, qtbot):
        csv_path = self._write_csv(tmp_path)
        window._start_load(lambda: window.dataset.load(csv_path, use_cache=False))
        qtbot.wait(50)
        assert window.dataset.is_loaded()
        assert window.overview_page.dataset is window.dataset
        assert "Loaded" in window.statusBar().currentMessage()

    def test_busy_state_toggles_load_actions(self, window, tmp_path, qtbot):
        csv_path = self._write_csv(tmp_path)
        assert window.load_local_action.isEnabled()
        window._start_load(lambda: window.dataset.load(csv_path, use_cache=False))
        qtbot.wait(50)
        # Synchronous worker mode means the load already completed and
        # re-enabled the actions by the time run_in_thread returns.
        assert window.load_local_action.isEnabled()

    def test_load_failure_shows_status_not_a_blocking_dialog(self, window, tmp_path, qtbot):
        missing = tmp_path / "does_not_exist.csv"
        window._start_load(lambda: window.dataset.load(missing, use_cache=False))
        qtbot.wait(50)
        assert "failed" in window.statusBar().currentMessage().lower()

    def test_reload_without_prior_load_shows_info_message(self, window):
        assert window.dataset.path is None
        window._reload()  # must not raise; QMessageBox.information is neutralized by conftest


class TestGlobalScopeWiring:
    def _write_csv(self, tmp_path):
        path = tmp_path / "scoped.csv"
        path.write_text(
            "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units\n"
            "241-A-101,Fe,Liquid,T1,50.0,kg\n"
            "241-A-101,Fe,Solid,T1,10.0,kg\n"
        )
        return path

    def test_scope_bar_only_visible_on_scoped_workspaces(self, window, qtbot):
        window.show()
        qtbot.waitExposed(window)
        try:
            # Query, Heatmaps and Overview all honor the scope now, so
            # the bar shows above them; Figure Composer does not.
            window.nav.setCurrentRow(NAV_ITEMS.index(NAV_QUERY))
            assert window.scope_bar.isVisible()
            window.nav.setCurrentRow(NAV_ITEMS.index(NAV_OVERVIEW))
            assert window.scope_bar.isVisible()
            window.nav.setCurrentRow(NAV_ITEMS.index("Figure Composer"))
            assert not window.scope_bar.isVisible()
        finally:
            window.hide()

    def test_scope_bar_visibility_is_derived_from_the_page_protocol(self, window):
        # Not a hand-maintained list: a page that honors scope changes gets
        # the bar automatically, and one that doesn't can never have data
        # filtered by a control it never shows.
        scoped = window._scoped_nav_items()
        assert scoped == {
            name for name, page in window._pages_by_nav.items()
            if hasattr(page, "on_scope_changed")
        }
        assert NAV_QUERY in scoped
        assert scoped <= set(NAV_ITEMS)

    def test_ask_in_query_navigates_and_answers(self, window, tmp_path, qtbot):
        window._start_load(lambda: window.dataset.load(self._write_csv(tmp_path), use_cache=False))
        qtbot.wait(50)
        assert window.ask_in_query("element", "Fe", "kg") is True
        qtbot.wait(20)
        assert window.stack.currentWidget() is window.query_page
        assert window.query_page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(60.0)

    def test_explorer_cross_link_uses_the_public_api(self, window, tmp_path, qtbot):
        window._start_load(lambda: window.dataset.load(self._write_csv(tmp_path), use_cache=False))
        qtbot.wait(50)
        window.explorer_page.query_edit.setText("Fe")
        window.explorer_page._ask_in_query()
        qtbot.wait(20)
        assert window.stack.currentWidget() is window.query_page
        assert window.query_page.subject_edit.text() == "Fe"

    def test_query_page_shares_the_shell_scope_object(self, window):
        assert window.query_page.scope is window.global_scope

    def test_load_reconciles_scope_bar_before_page_dispatch(self, window, tmp_path, qtbot):
        window._start_load(lambda: window.dataset.load(self._write_csv(tmp_path), use_cache=False))
        qtbot.wait(50)
        farm_combo = window.scope_bar._combos["TankFarm"]
        assert [farm_combo.itemText(i) for i in range(farm_combo.count())] == ["All", "A"]

    def test_scope_change_fans_out_to_pages(self, window, tmp_path, qtbot):
        window._start_load(lambda: window.dataset.load(self._write_csv(tmp_path), use_cache=False))
        qtbot.wait(50)
        page = window.query_page
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("Fe")
        page.basis_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(60.0)
        window.global_scope.set_filter("WastePhase", ["Solid"])
        qtbot.wait(20)
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(10.0)
        assert "Scope: Solid" in page.plot.ax.get_title()


class TestBrowseAndLoadRemembersLastDirectory:
    def test_cancelled_dialog_does_nothing(self, window, monkeypatch):
        import qt_shell
        monkeypatch.setattr(qt_shell.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
        window._browse_and_load()
        assert window.dataset.path is None

    def test_successful_browse_saves_and_restores_last_directory(self, window, tmp_path, qtbot, monkeypatch):
        import qt_shell
        csv_path = tmp_path / "browsed.csv"
        csv_path.write_text(
            "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units\n"
            "241-A-101,137Cs,Liquid,T1,100.0,Ci\n"
        )
        seen_start_dirs = []

        def fake_dialog(*args, **kwargs):
            seen_start_dirs.append(args[2] if len(args) > 2 else kwargs.get("dir", ""))
            return (str(csv_path), "")

        monkeypatch.setattr(qt_shell.QFileDialog, "getOpenFileName", staticmethod(fake_dialog))
        assert seen_start_dirs == []
        window._browse_and_load()
        qtbot.wait(50)
        assert seen_start_dirs == [""]  # nothing saved yet on the first call

        window._browse_and_load()
        qtbot.wait(50)
        assert seen_start_dirs[-1] == str(tmp_path.resolve())


class TestDarkModeToggle:
    def test_toggling_does_not_raise(self, window, qtbot):
        window.dark_mode_action.setChecked(True)
        window.dark_mode_action.setChecked(False)


class TestHelpMenu:
    """HelpDialog.exec() blocks for real user input, so these patch the
    class qt_shell imported (not qt_help's own copy of the name) rather
    than calling the real modal dialog."""

    def test_show_about_opens_dialog(self, window, monkeypatch):
        import qt_shell

        captured = {}

        class FakeDialog:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def exec(self):
                captured["exec_called"] = True

        monkeypatch.setattr(qt_shell, "HelpDialog", FakeDialog)
        window.show_about()
        assert captured["exec_called"]
        assert "Ember" in captured["title"]

    def test_show_notice_opens_dialog_with_notice_markdown(self, window, monkeypatch):
        import qt_shell

        captured = {}

        class FakeDialog:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def exec(self):
                captured["exec_called"] = True

        monkeypatch.setattr(qt_shell, "HelpDialog", FakeDialog)
        window.show_notice()
        assert captured["exec_called"]
        assert "Ember" in captured["markdown"]
        assert "Detailed Notice" in captured["title"]


class TestDebugMenu:
    """Debug/Export moved from a nav-rail workspace to a Help-menu dialog
    (rarely-needed troubleshooting tool, not something a first-time user
    should see as a 9th equally-weighted workspace). The dialog is shown
    NON-modally -- a live log viewer must not block the very app actions
    the user wants to observe -- so show_debug() never blocks under test
    and needs no exec() neutralization."""

    def test_debug_not_in_nav_items(self):
        assert "Debug / Export" not in NAV_ITEMS

    def test_help_menu_has_debug_export_action(self, window):
        from PySide6.QtGui import QAction
        actions = [a.text() for a in window.findChildren(QAction)]
        assert "Debug / Export…" in actions

    def test_first_call_builds_the_dialog_and_debug_page(self, window):
        assert window.debug_page is None
        window.show_debug()
        assert window.debug_page is not None
        assert window.debug_page in window._dataset_aware_pages()

    def test_dialog_is_not_modal(self, window):
        # Review finding: exec() made the live log viewer application-modal,
        # so it could only ever show events from before it was opened.
        window.show_debug()
        assert window._debug_dialog.isModal() is False
        assert window._debug_dialog.isVisible() is True
        window._debug_dialog.close()

    def test_second_call_reuses_the_same_dialog_and_page(self, window):
        window.show_debug()
        first_dialog, first_page = window._debug_dialog, window.debug_page
        window.show_debug()
        assert window._debug_dialog is first_dialog
        assert window.debug_page is first_page
        assert window._dataset_aware_pages().count(first_page) == 1  # derived, never double-appended

    def test_debug_page_receives_dataset_broadcasts_once_it_exists(self, window, tmp_path):
        window.show_debug()
        csv_path = tmp_path / "custom.csv"
        csv_path.write_text(
            "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units\n"
            "241-A-101,137Cs,Liquid,T1,100.0,Ci\n"
        )
        window._start_load(lambda: window.dataset.load(csv_path, use_cache=False))
        assert window.debug_page.dataset is window.dataset

    def test_opening_after_a_dataset_is_already_loaded_syncs_immediately(self, window, tmp_path):
        csv_path = tmp_path / "custom.csv"
        csv_path.write_text(
            "WasteSiteId,Analyte,WastePhase,WasteType,Inventory,Units\n"
            "241-A-101,137Cs,Liquid,T1,100.0,Ci\n"
        )
        window._start_load(lambda: window.dataset.load(csv_path, use_cache=False))
        window.show_debug()  # first open happens *after* the load
        assert window.debug_page.dataset is window.dataset


class _FakeGettingStartedDialog:
    """GettingStartedDialog.exec() blocks for real user input; these tests
    patch the class qt_shell imported so show_getting_started()/
    _maybe_show_getting_started() can run without a real modal, mirroring
    TestHelpMenu's FakeDialog pattern above."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.workspace_chosen = _FakeSignal()
        self.exec_called = False
        _FakeGettingStartedDialog.instances.append(self)

    def exec(self):
        self.exec_called = True


class _FakeSignal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, value):
        for slot in self._slots:
            slot(value)


class TestGettingStarted:
    def test_show_getting_started_opens_and_wires_the_signal(self, window, monkeypatch):
        import qt_shell
        _FakeGettingStartedDialog.instances.clear()
        monkeypatch.setattr(qt_shell, "GettingStartedDialog", _FakeGettingStartedDialog)

        window.show_getting_started()

        assert len(_FakeGettingStartedDialog.instances) == 1
        dialog = _FakeGettingStartedDialog.instances[0]
        assert dialog.exec_called is True
        assert dialog.workspace_chosen._slots  # something connected

    def test_choosing_a_workspace_navigates_the_nav_rail(self, window, monkeypatch):
        import qt_shell
        _FakeGettingStartedDialog.instances.clear()
        monkeypatch.setattr(qt_shell, "GettingStartedDialog", _FakeGettingStartedDialog)

        window.show_getting_started()
        dialog = _FakeGettingStartedDialog.instances[0]
        dialog.workspace_chosen.emit("Vitrification")

        assert window.nav.currentRow() == NAV_ITEMS.index("Vitrification")

    def test_unknown_workspace_name_is_ignored_not_a_crash(self, window):
        window._go_to_workspace("Not A Real Workspace")  # must not raise

    def test_first_launch_shows_the_dialog_once(self, window, monkeypatch):
        # Uses the shared `window` fixture, which already triggers one
        # auto-show (via QTimer.singleShot during its own qtbot.wait(50))
        # against the REAL GettingStartedDialog before this test body even
        # runs -- that already consumed the "not yet shown" QSettings flag.
        # So the FIRST explicit call here is really the *second* trigger
        # overall, and correctly does nothing; this test's job is just to
        # confirm calling it again on an already-flagged window is a noop.
        import qt_shell
        _FakeGettingStartedDialog.instances.clear()
        monkeypatch.setattr(qt_shell, "GettingStartedDialog", _FakeGettingStartedDialog)

        window._maybe_show_getting_started()
        window._maybe_show_getting_started()
        assert _FakeGettingStartedDialog.instances == []  # flag was already set before this test ran

    def test_does_not_show_again_on_a_fresh_window_after_the_flag_is_set(self, qtbot, tmp_path, monkeypatch):
        import data_model as dm
        import qt_shell
        _FakeGettingStartedDialog.instances.clear()
        monkeypatch.setattr(qt_shell, "GettingStartedDialog", _FakeGettingStartedDialog)
        monkeypatch.setattr(dm, "app_base_dir", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)

        win1 = qt_shell.EmberMainWindow()
        qtbot.addWidget(win1)
        qtbot.wait(50)
        assert len(_FakeGettingStartedDialog.instances) == 1

        win2 = qt_shell.EmberMainWindow()
        qtbot.addWidget(win2)
        qtbot.wait(50)
        assert len(_FakeGettingStartedDialog.instances) == 1  # hermetic QSettings persisted the flag
