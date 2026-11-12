import pytest
from PySide6.QtWidgets import QMainWindow, QPushButton

import data_model as dm
import qt_explorer
from qt_explorer import MODE_LABELS, MODES, PLOT_SOURCES, PLOT_TYPES, ExplorerPage, plain_note

from conftest import requires_real_data


def _make_page(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    page = ExplorerPage(app_window)
    qtbot.addWidget(page)
    return page


def _run(page, qtbot, dataset, query="Cs"):
    page.on_dataset_changed(dataset)
    page.query_edit.setText(query)
    page.run_search()
    qtbot.wait(20)


class TestExplorerPage:
    def test_run_without_dataset_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page.run_search()
        # A sentence in the status bar, not a dialog.
        assert "no dataset" in page.app_window.statusBar().currentMessage()

    def test_run_with_empty_query_shows_message_not_crash(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)  # let the empty state's canvas.draw_idle() flush before teardown
        page.query_edit.setText("   ")
        page.run_search()
        assert "enter an element or an analyte" in page.app_window.statusBar().currentMessage()
        assert not page._results

    def test_on_dataset_changed_populates_unit_combo(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        items = [page.unit_combo.itemText(i) for i in range(page.unit_combo.count())]
        assert items == ["All", "Ci", "kg"]

    def test_run_search_populates_tables(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        target_by_tank = page._table_views["Target by tank"].dataframe()
        assert len(target_by_tank) == 2
        assert set(target_by_tank["WasteSiteId"]) == {"241-A-101", "241-AN-104"}
        co_el = page._table_views["Co-elements"].dataframe()
        assert "Fe" in co_el["Element"].values

    def test_raw_target_rows_table_populated(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        raw = page._table_views["Raw target rows"].dataframe()
        assert len(raw) == 2
        assert set(raw["Analyte"]) == {"137Cs"}

    def test_invalid_element_symbol_in_element_mode_shows_warning_not_crash(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        page.query_edit.setText("Zz")
        page.set_mode("element")
        page.run_search()  # must not raise
        qtbot.wait(20)
        assert page.mode_key() == "element"
        assert not page._results
        # With nothing standing, the refusal is written where the chart will be.
        texts = [t.get_text() for t in page.plot.ax.texts]
        assert any("not an element symbol" in t for t in texts)

    def test_all_plot_types_render_without_crash(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        for plot_type in PLOT_TYPES:
            page.plot_type_combo.setCurrentText(plot_type)
            qtbot.wait(20)

    def test_target_by_waste_phase_actually_plots_phases(self, qtbot, sample_dataset):
        # The pain-point case: this table always existed one tab away from
        # a plot dropdown that couldn't draw it. Now it can -- assert real
        # phase bars, not just no-crash.
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        page.plot_type_combo.setCurrentText("Target by waste phase")
        qtbot.wait(20)
        labels = [t.get_text() for t in page.plot.ax.get_yticklabels()]
        assert set(labels) == {"Liquid", "Sludge"}  # sample_dataset's Cs phases

    def test_target_by_waste_type_actually_plots_types(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        page.plot_type_combo.setCurrentText("Target by waste type")
        qtbot.wait(20)
        labels = [t.get_text() for t in page.plot.ax.get_yticklabels()]
        assert set(labels) == {"T1", "T2"}

    def test_export_without_results_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page._export_bundle()
        assert "nothing to export" in page.app_window.statusBar().currentMessage()

    def test_export_writes_bundle(self, qtbot, sample_dataset, tmp_path):
        page = _make_page(qtbot)
        sample_dataset.output_root = tmp_path
        _run(page, qtbot, sample_dataset)
        page._export_bundle()
        bundles = list(tmp_path.glob("search_Cs_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "Target_by_tank.csv").exists()
        assert (bundles[0] / "plot.png").exists()
        # The notes and the state that produced the tables travel with them.
        assert (bundles[0] / "Notes.csv").exists()
        context = (bundles[0] / "context.csv").read_text(encoding="utf-8")
        assert "global_scope" in context and "Cs (element)" in context

    def test_dataset_changed_clears_stale_results(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        assert len(page._table_views["Target by tank"].dataframe()) == 2
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        assert page._table_views["Target by tank"].dataframe().empty
        assert page.warnings_label.isHidden()


class TestPagePass:
    """What the interface pass added: wording, readable choices, the notes
    banner, the chart-to-table link, completions."""

    def test_header_actions_read_as_the_house_wording(self, qtbot):
        page = _make_page(qtbot)
        assert page.header.title_label.text() == "Element explorer"
        assert page.export_btn.text() == "Export..."
        assert page.ask_query_btn.text() == "Ask in Query"
        assert page.run_btn.text() == "Run" and page.run_btn.objectName() == "Primary"
        for button in page.findChildren(QPushButton):
            assert "→" not in button.text()

    def test_match_modes_are_shown_in_words_and_kept_as_identifiers(self, qtbot):
        page = _make_page(qtbot)
        shown = [page.mode_combo.itemText(i) for i in range(page.mode_combo.count())]
        assert shown == [MODE_LABELS[key] for key in MODES]
        assert not any("_" in text or text == text.lower() for text in shown)
        assert [page.mode_combo.itemData(i) for i in range(page.mode_combo.count())] == MODES
        page.set_mode("analyte_contains")
        assert page.mode_key() == "analyte_contains"

    def test_chart_is_on_the_left_and_tables_on_the_right(self, qtbot):
        page = _make_page(qtbot)
        assert page.splitter.widget(0) is page.plot
        assert page.splitter.widget(1) is page.tables_pane
        assert page.tables_pane.isAncestorOf(page.tables)
        assert not page.splitter.childrenCollapsible()

    def test_tables_are_chosen_by_name_and_the_list_follows_the_table_shown(self, qtbot):
        page = _make_page(qtbot)
        names = [page.table_combo.itemText(i) for i in range(page.table_combo.count())]
        assert names == list(page._table_views)
        page.table_combo.setCurrentText("Co-analytes")
        assert page.tables.currentWidget() is page._table_views["Co-analytes"]
        page.tables.setCurrentWidget(page._table_views["Notes"])
        assert page.table_combo.currentText() == "Notes"

    def test_idle_chart_and_tables_say_what_to_do(self, qtbot):
        page = _make_page(qtbot)
        qtbot.wait(20)
        texts = [t.get_text() for t in page.plot.ax.texts]
        assert "No result yet" in texts
        assert page._table_views["Co-elements"].empty_state.title_label.text() == "No result yet"

    def test_run_button_says_update_when_the_controls_left_the_result(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.run_btn.text() == "Run"
        page.query_edit.setText("Cs")
        page.run_search()
        qtbot.wait(20)
        assert page.run_btn.text() == "Run"
        page.unit_combo.setCurrentText("kg")
        assert page.run_btn.text() == "Update"
        page.run_search()
        qtbot.wait(20)
        assert page.run_btn.text() == "Run"
        page.query_edit.setText("Fe")
        assert page.run_btn.text() == "Update"

    def test_search_field_offers_what_the_dataset_holds(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)

        def offered():
            model = page.query_edit.completer().model()
            return {model.data(model.index(i, 0)) for i in range(model.rowCount())}

        assert {"Cs", "137Cs", "Fe"} <= offered()             # automatic: elements and analytes
        page.set_mode("element")
        assert "Cs" in offered() and "137Cs" not in offered()
        page.set_mode("analyte_exact")
        assert "137Cs" in offered()

    def test_axis_labels_carry_the_unit(self, qtbot, sample_dataset):
        from qt_table_format import unit_from_label
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        # "All" is answered in bridged grams by the engine.
        label = page.plot.ax.get_xlabel().splitlines()[0]
        assert label == "Cs inventory (g)" and unit_from_label(label) == "g"
        page.unit_combo.setCurrentText("Ci")
        page.run_search()
        qtbot.wait(20)
        assert page.plot.ax.get_xlabel().splitlines()[0] == "Cs inventory (Ci)"
        page.plot_type_combo.setCurrentText("Target vs total inventory")
        qtbot.wait(20)
        assert unit_from_label(page.plot.ax.get_xlabel()) == "Ci"
        assert unit_from_label(page.plot.ax.get_ylabel()) == "Ci"
        page.plot_type_combo.setCurrentText("Co-elements presence (%)")
        qtbot.wait(20)
        assert unit_from_label(page.plot.ax.get_xlabel().splitlines()[0]) == "%"

    def test_clicking_a_bar_selects_its_row(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        view = page._table_views["Target by tank"]
        frame = view.dataframe()
        tank = str(frame["WasteSiteId"].iloc[-1])
        page.tables.setCurrentWidget(page._table_views["Notes"])
        page._on_mark_clicked({"label": tank, "value": 1.0})
        assert page.tables.currentWidget() is view
        selected = view.selectedIndexes()
        assert selected
        assert str(frame["WasteSiteId"].iloc[selected[0].row()]) == tank
        page._on_mark_clicked({"label": "not a row"})     # no match: selection untouched, no error

    def test_clicking_a_mixed_unit_bar_selects_the_row_of_that_unit(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        page.plot_type_combo.setCurrentText("Co-elements inventory")
        qtbot.wait(20)
        view = page._table_views["Co-elements"]
        assert page.tables.currentWidget() is view            # the chart's own table comes forward
        frame = view.dataframe()
        last = frame.iloc[-1]
        page._on_mark_clicked({"label": f"{last['Element']} [{last['Units']}]"})
        selected = view.selectedIndexes()
        assert selected
        picked = frame.iloc[selected[0].row()]
        assert (picked["Element"], picked["Units"]) == (last["Element"], last["Units"])

    def test_every_plot_names_the_table_it_is_drawn_from(self, qtbot):
        page = _make_page(qtbot)
        assert set(PLOT_SOURCES) == set(PLOT_TYPES)
        assert {table for table, _column in PLOT_SOURCES.values()} <= set(page._table_views)

    def test_shell_entry_point_fills_the_field_and_runs(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.show_subject("137Cs", "analyte") is True
        qtbot.wait(20)
        assert page.mode_key() == "analyte_exact" and page.query_edit.text() == "137Cs"
        assert len(page._table_views["Raw target rows"].dataframe()) == 2
        page.show_subject("Fe")
        qtbot.wait(20)
        assert page.mode_key() == "element" and page._symbol == "Fe"

    def test_a_late_result_of_a_superseded_run_is_discarded(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        standing = page._table_views["Target by tank"].dataframe()
        late = qt_explorer._Outcome(token=page._run_token - 1, params=dict(page._params), tables={})
        page._on_run_done(late)
        assert page._table_views["Target by tank"].dataframe() is standing


class TestNotes:
    def test_banner_wording_has_no_internal_identifiers(self):
        assert "Inventory_RSD" not in plain_note(
            "3 of 5 contributing row(s) have no assigned uncertainty (Inventory_RSD = 0 means unassigned, not exact).")
        assert "reference_data" not in plain_note(
            "2 kg row(s) of unregistered species excluded from apportionment (1 kg total): Acetone — "
            "add their formulas to reference_data.SPECIES_FORMULAS to include them.")
        assert "harmonizer" not in plain_note("Ci values mix decay baselines A and B (harmonizer: Stage 4).")
        assert plain_note("min inventory is a filter.") == "Min inventory is a filter."

    def test_engine_warnings_reach_the_banner_and_the_notes_tab(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.query_edit.setText("Cs")
        page.unit_combo.setCurrentText("kg")    # 137Cs is reported in Ci: set aside, with a warning
        page.run_search()
        qtbot.wait(20)
        notes = page._table_views["Notes"].dataframe()
        warnings = list(notes.loc[notes["kind"] == "warning", "text"])
        assert warnings, "precondition: this search carries engine warnings"
        assert not page.warnings_label.isHidden()
        assert list(page.warnings_label.notes()) == [plain_note(w) for w in warnings]
        assert "⚠" not in page.warnings_label.summary_label.text()
        assert (notes["kind"] == "disclosure").any()

    def test_a_scope_is_declared_as_not_reaching_the_row_count_tables(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        _run(page, qtbot, sample_dataset)
        unscoped_co = page._table_views["Co-elements"].dataframe()
        assert not page._table_views["Notes"].dataframe()["text"].str.contains("whole dataset").any()

        page.scope.set_filter("WastePhase", ["Sludge"])
        page.on_scope_changed()
        qtbot.wait(20)
        # The engine-backed tables follow the scope ...
        assert set(page._table_views["Target by phase"].dataframe()["WastePhase"]) == {"Sludge"}
        # ... the row-count tables do not, and the page says so in three places.
        assert page._table_views["Co-elements"].dataframe().equals(unscoped_co)
        assert page._table_views["Notes"].dataframe()["text"].str.contains("whole dataset").any()
        assert any("whole dataset" in note for note in page.warnings_label.notes())
        assert "Scope: Sludge" in page.plot.ax.get_title()
        page.plot_type_combo.setCurrentText("Co-elements presence (%)")
        qtbot.wait(20)
        assert "scope does not apply" in page.plot.ax.get_title()


class TestShippedDefault:
    @requires_real_data
    def test_the_default_search_returns_rows_on_the_shipped_dataset(self, qtbot):
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        page = _make_page(qtbot)
        page.on_dataset_changed(dataset)
        page.run_search()                       # the field's default, untouched
        qtbot.wait(50)
        assert len(page._table_views["Target by tank"].dataframe()) > 100
        assert not page._table_views["Co-elements"].dataframe().empty
        assert page.plot.ax.patches, "the default search draws bars"
