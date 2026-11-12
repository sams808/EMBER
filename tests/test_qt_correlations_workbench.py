import pandas as pd
import pytest
from PySide6.QtWidgets import QMainWindow, QMessageBox

import qt_correlations_workbench as qcw
from conftest import REAL_ATTRS_PATH, REAL_CSV_PATH, requires_real_data
from data_model import HanfordDataset
from qt_correlations_workbench import (
    DEFAULT_ELEMENTS, PLOT_TYPES, WorkbenchTab, completion_items, reporting_notes, set_choice,
    unrecognized_entries,
)


def _make_tab(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    tab = WorkbenchTab(app_window)
    qtbot.addWidget(tab)
    return tab


def _status(tab) -> str:
    return tab.app_window.statusBar().currentMessage()


def _capture(monkeypatch, name):
    calls = []
    monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.Ok))
    return calls


def _all_texts(tab):
    """Every piece of text the figure carries (titles included)."""
    from matplotlib.text import Text
    return " | ".join(t.get_text() for t in tab.plot.figure.findobj(Text) if t.get_text())


class TestWorkbenchTab:
    def test_actions_without_dataset_say_so_in_the_status_bar(self, qtbot, monkeypatch):
        info = _capture(monkeypatch, "information")
        tab = _make_tab(qtbot)
        for action in (tab.build_data, tab.plot_selected, tab.export_plot_suite, tab.run):
            tab.app_window.statusBar().clearMessage()
            action()
            assert "no dataset is loaded" in _status(tab)
        tab.export_tables()
        assert "nothing to export" in _status(tab)
        assert not info and not tab.results

    def test_build_data_populates_tables(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.build_data()
        qtbot.wait(20)
        stats = tab._table_views["Element stats"].dataframe()
        assert set(stats["Element"]) == {"Fe", "Na", "Cd"}
        raw = tab._table_views["Raw kg matrix"].dataframe()
        assert set(raw.columns) - {"WasteSiteId"} == {"Fe", "Na", "Cd"}

    def test_build_data_error_shows_warning_not_crash(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe")  # only one valid element -> ValueError
        tab.build_data()  # QMessageBox.warning neutralized by conftest

    def test_top_kg_selection_mode(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.selection_combo.setCurrentText("Top kg elements")
        tab.top_kg_spin.setValue(2)
        tab.build_data()
        qtbot.wait(20)
        assert len(set(tab._table_views["Element stats"].dataframe()["Element"])) == 2

    def test_skip_list_applied(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.skip_edit.setText("Cd")
        tab.build_data()
        qtbot.wait(20)
        elements = set(tab._table_views["Element stats"].dataframe()["Element"])
        assert "Cd" not in elements
        skipped = tab._table_views["Skipped"].dataframe()
        assert "Cd" in set(skipped["ExcludedElement"])

    def test_tank_inventory_bars_uses_its_own_mode_and_sort_controls(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.build_data()
        qtbot.wait(20)
        tab.plot_type_combo.setCurrentText("Tank inventory bars")
        tab.bar_mode_combo.setCurrentText("Grouped")
        tab.bar_sort_combo.setCurrentText("Tank ID (A-Z)")
        tab.plot_selected()
        qtbot.wait(20)
        assert len(tab.plot.ax.containers) == 3  # one per selected element
        assert "grouped" in tab.plot.ax.get_title().lower()

    def test_all_plot_types_render_without_crash(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.build_data()
        qtbot.wait(20)
        for plot_type in PLOT_TYPES:
            tab.plot_type_combo.setCurrentText(plot_type)
            tab.plot_selected()
            qtbot.wait(20)

    def test_plot_selected_without_prior_build_builds_first(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        assert not tab.results
        tab.plot_selected()
        qtbot.wait(20)
        assert tab.results

    def test_coherent_colors_mode(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.color_mode_combo.setCurrentText("Coherent colors")
        tab.build_data()
        qtbot.wait(20)

    def test_export_tables_writes_bundle(self, qtbot, sample_dataset, tmp_path):
        tab = _make_tab(qtbot)
        sample_dataset.output_root = tmp_path
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.build_data()
        qtbot.wait(20)
        tab.export_tables()
        bundles = list(tmp_path.glob("seaborn_correlations_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "element_stats.csv").exists()
        assert (bundles[0] / "settings.csv").exists()
        assert (bundles[0] / "current_seaborn_plot.png").exists()

    def test_export_plot_suite_writes_all_plots_and_manifest(self, qtbot, sample_dataset, tmp_path):
        tab = _make_tab(qtbot)
        sample_dataset.output_root = tmp_path
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.build_data()
        qtbot.wait(20)
        tab.export_plot_suite()
        qtbot.wait(20)  # export_plot_suite ends with its own plot_selected() -> draw_idle()
        bundles = list(tmp_path.glob("seaborn_plot_suite_*"))
        assert len(bundles) == 1
        manifest = bundles[0] / "plot_suite_manifest.csv"
        assert manifest.exists()
        import pandas as pd
        manifest_df = pd.read_csv(manifest)
        assert len(manifest_df) == len(PLOT_TYPES)
        assert (manifest_df["status"] == "ok").all()
        # Restored to whatever was selected before the export loop ran.
        assert tab.plot_type_combo.currentText() == "Corr heatmap + total kg projections"


class TestPagePass:
    def test_opening_state(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab.elements_edit.text() == DEFAULT_ELEMENTS
        # Elements reported as activity only have no place in a kg-only default.
        assert not {"Cs", "Tc", "I"} & set(qcw.csci.parse_element_list(DEFAULT_ELEMENTS))
        assert tab.run_btn is tab.build_btn
        assert tab.build_btn.objectName() == "Primary" and tab.build_btn.text() == "Run"
        assert tab.warnings_label is tab.notes_banner and tab.warnings_label.isHidden()
        assert not tab.advanced_section.is_expanded()
        assert [t.get_text() for t in tab.plot.ax.texts][0] == "No result yet"
        assert tab.header.title_label.text() == "Association workbench"
        assert tab.header.subtitle_label.wordWrap()
        assert [a.text() for a in tab.export_menu.actions()] == ["Tables and current chart", "Every chart type"]
        assert [tab.metric_combo.itemData(i) for i in range(tab.metric_combo.count())] == qcw.METRICS
        assert [tab.method_combo.itemText(i) for i in range(tab.method_combo.count())] == [
            "Pearson", "Spearman", "Kendall"]

    def test_run_button_says_update_when_the_controls_left_the_result(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        assert tab.build_btn.text() == "Run"
        tab.run()
        qtbot.wait(20)
        assert tab.build_btn.text() == "Run" and tab.build_btn.isEnabled()
        set_choice(tab.method_combo, "spearman")
        assert tab.build_btn.text() == "Update"
        set_choice(tab.method_combo, "pearson")
        assert tab.build_btn.text() == "Run"
        tab.color_mode_combo.setCurrentText("Basic")          # a display choice: nothing to update
        tab.top_pairs_spin.setValue(12)
        assert tab.build_btn.text() == "Run"
        tab.skip_edit.setText("Cd")
        assert tab.build_btn.text() == "Update"
        tab.run()
        qtbot.wait(20)
        assert tab.build_btn.text() == "Run"

    def test_the_list_is_not_used_by_the_top_selection(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab.elements_edit.isEnabled() and not tab.top_kg_spin.isEnabled()
        tab.selection_combo.setCurrentText("Top kg elements")
        assert not tab.elements_edit.isEnabled() and tab.top_kg_spin.isEnabled()

    def test_elements_without_a_kg_inventory_are_named_calmly(self, qtbot, sample_dataset, monkeypatch):
        warnings = _capture(monkeypatch, "warning")
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cs, Pu, Zz9")
        tab.run()
        qtbot.wait(20)
        assert not warnings
        notes = list(tab.warnings_label.notes())
        assert notes[0] == "Zz9 is not an element symbol and was left out."
        assert notes[1].startswith(
            "Cs and Pu have no inventory in kg in this dataset: they are reported in Ci only.")
        assert "Cross-correlation tab" in notes[1]
        assert tab.warnings_label.summary_label.text().startswith("2 notes qualify this result")
        # The columns exist and are zero, as before: the numbers are the same, the page says why.
        raw = tab._table_views["Raw kg matrix"].dataframe()
        assert (raw["Cs"] == 0).all() and (raw["Pu"] == 0).all()
        # A clean list clears the banner.
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.run()
        qtbot.wait(20)
        assert tab.warnings_label.isHidden()

    def test_reporting_notes_name_the_unit_an_element_is_reported_in(self, sample_dataset):
        assert reporting_notes(sample_dataset, ["Fe", "Na"], "kg") == []
        assert reporting_notes(sample_dataset, ["Fe", "Cs"], "kg", consequence="Use {units}.") == [
            "Cs has no inventory in kg in this dataset: it is reported in Ci only. Use Ci."]
        assert reporting_notes(sample_dataset, ["Fe"], "Ci") == [
            "Fe has no inventory in Ci in this dataset: it is reported in kg only."]
        assert reporting_notes(sample_dataset, ["Au"], "kg") == [
            "Au has no inventory in this dataset, in any unit."]
        assert unrecognized_entries("Fe, na; Zz9  Qq") == ["Zz9", "Qq"]

    def test_fields_offer_what_the_dataset_holds(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        assert completion_items(tab.elements_edit) == []
        tab.on_dataset_changed(sample_dataset)
        for edit in (tab.elements_edit, tab.skip_edit):
            assert {"Fe", "Na", "Cd", "Cs"} <= set(completion_items(edit))
        completer = tab.elements_edit.completer()
        assert completer.splitPath("Fe, Na, c") == ["c"]
        tab.elements_edit.setText("Fe, Na, c")
        completer.setCompletionPrefix("Fe, Na, c")
        assert completer.currentCompletion() == "Fe, Na, Cd"
        tab.elements_edit.setText("f")
        completer.setCompletionPrefix("f")
        assert completer.currentCompletion() == "Fe"         # the first entry has nothing to keep

    def test_changing_the_plot_redraws_without_recomputing(self, qtbot, sample_dataset, monkeypatch):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.plot_type_combo.setCurrentText("Tank inventory bars")      # no result yet: nothing happens
        assert not tab.results
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.run()
        qtbot.wait(20)
        assert "stacked" in tab.plot.ax.get_title().lower()
        started = []
        monkeypatch.setattr(qcw, "run_in_thread", lambda *a, **k: started.append(a))
        tab.bar_mode_combo.setCurrentText("Grouped")
        qtbot.wait(20)
        assert "grouped" in tab.plot.ax.get_title().lower()
        tab.plot_type_combo.setCurrentText("Top Jaccard co-presence")
        qtbot.wait(20)
        assert "grouped" not in tab.plot.ax.get_title().lower()
        assert not started

    def test_chart_and_export_describe_the_run_not_the_live_controls(self, qtbot, sample_dataset, tmp_path):
        tab = _make_tab(qtbot)
        sample_dataset.output_root = tmp_path
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.run()
        qtbot.wait(20)
        assert "log10(inventory + 1)" in _all_texts(tab)
        set_choice(tab.metric_combo, "presence")              # moved, not run
        tab.elements_edit.setText("Fe, Na")
        tab.annotate_check.setChecked(True)                   # forces a redraw
        qtbot.wait(20)
        texts = _all_texts(tab)
        assert "log10(inventory + 1)" in texts and "resence" not in texts
        tab.export_tables()
        bundle = next(tmp_path.glob("seaborn_correlations_*"))
        settings = pd.read_csv(bundle / "settings.csv").iloc[0]
        assert settings["metric"] == "log10_plus1" and settings["elements_text"] == "Fe, Na, Cd"
        assert "exported" in _status(tab)

    def test_clicking_a_mark_selects_its_row(self, qtbot, sample_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.run()
        qtbot.wait(20)
        tab._on_mark_clicked({"row": "Na", "column": "Fe", "value": 0.2})        # a heat-map cell
        view = tab._table_views["Pair stats"]
        assert tab.tables.currentWidget() is view
        row = view.dataframe().iloc[view.selectedIndexes()[0].row()]
        assert {row["Element_A"], row["Element_B"]} == {"Fe", "Na"}
        tab._on_mark_clicked({"label": "Cd"})                                     # an element bar
        view = tab._table_views["Element stats"]
        assert tab.tables.currentWidget() is view
        assert view.dataframe()["Element"].iloc[view.selectedIndexes()[0].row()] == "Cd"
        tab._on_mark_clicked({"label": "241-AN-104", "tank_id": "241-AN-104"})    # a tank
        view = tab._table_views["Raw kg matrix"]
        assert tab.tables.currentWidget() is view
        assert view.dataframe()["WasteSiteId"].iloc[view.selectedIndexes()[0].row()] == "241-AN-104"
        tab._on_mark_clicked({"label": "Cd-Na  (n=0)", "value": 0.0})             # a ranked-pair bar
        view = tab._table_views["Pair stats"]
        assert tab.tables.currentWidget() is view
        row = view.dataframe().iloc[view.selectedIndexes()[0].row()]
        assert {row["Element_A"], row["Element_B"]} == {"Cd", "Na"}
        tab._on_mark_clicked({"label": "nothing by that name"})                   # no match: no error

    def test_dataset_change_clears_the_previous_result(self, qtbot, sample_dataset, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cs")
        tab.run()
        qtbot.wait(20)
        assert tab.results and not tab.warnings_label.isHidden()
        tab.on_dataset_changed(structure_dataset)
        qtbot.wait(20)
        assert not tab.results and tab._params == {}
        assert all(view.dataframe().empty for view in tab._table_views.values())
        assert tab.warnings_label.isHidden() and tab.build_btn.text() == "Run"
        assert [t.get_text() for t in tab.plot.ax.texts][0] == "No result yet"


class TestWorkerContract:
    def test_results_are_delivered_to_bound_methods_of_the_tab(self, qtbot, sample_dataset, monkeypatch):
        seen = {}
        monkeypatch.setattr(qcw, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: seen.update(fn=fn, on_done=on_done,
                                                                                    on_error=on_error))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.run()
        assert seen["on_done"].__self__ is tab and seen["on_error"].__self__ is tab
        assert not tab.build_btn.isEnabled() and "Running" in tab.build_btn.text()
        # Everything the worker needs was read before it left: moving a control now changes nothing.
        tab.elements_edit.setText("Fe, Na")
        outcome = seen["fn"]()
        assert outcome.token == tab._run_token
        assert set(outcome.results["element_stats"]["Element"]) == {"Fe", "Na", "Cd"}
        tab._on_run_done(outcome)
        qtbot.wait(20)
        assert tab.build_btn.isEnabled() and tab.build_btn.text() == "Update"
        assert set(tab._table_views["Element stats"].dataframe()["Element"]) == {"Fe", "Na", "Cd"}

    def test_a_superseded_answer_is_dropped(self, qtbot, sample_dataset, monkeypatch):
        pending = []
        monkeypatch.setattr(qcw, "run_in_thread", lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na")
        tab.run()
        tab.elements_edit.setText("Fe, Na, Cd")
        tab._start(tab._read_params())
        first, second = pending[0](), pending[1]()
        tab._on_run_done(second)
        tab._on_run_done(first)                   # arrives late: must not overwrite
        qtbot.wait(20)
        assert set(tab._table_views["Element stats"].dataframe()["Element"]) == {"Fe", "Na", "Cd"}

    def test_an_answer_for_the_previous_dataset_is_dropped(self, qtbot, sample_dataset, structure_dataset,
                                                           monkeypatch):
        pending = []
        monkeypatch.setattr(qcw, "run_in_thread", lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na")
        tab.run()
        tab.on_dataset_changed(structure_dataset)
        tab._on_run_done(pending[0]())
        qtbot.wait(20)
        assert not tab.results and tab.build_btn.isEnabled()

    def test_refusal_keeps_the_standing_result(self, qtbot, sample_dataset, monkeypatch):
        warnings = _capture(monkeypatch, "warning")
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.run()
        qtbot.wait(20)
        tab.elements_edit.setText("Fe")
        tab.run()
        qtbot.wait(20)
        assert len(warnings) == 1 and "at least two valid elements" in warnings[0][2]
        assert set(tab._table_views["Element stats"].dataframe()["Element"]) == {"Fe", "Na", "Cd"}
        assert tab._params["elements_text"] == "Fe, Na, Cd" and tab.build_btn.text() == "Update"

    def test_unexpected_failure_is_reported_not_raised(self, qtbot, sample_dataset, monkeypatch):
        criticals = _capture(monkeypatch, "critical")

        def boom(*args, **kwargs):
            raise RuntimeError("science fell over")

        monkeypatch.setattr(qcw.csci, "kg_correlation_workbench", boom)
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.run()
        qtbot.wait(20)
        assert len(criticals) == 1 and "science fell over" in criticals[0][2]
        assert tab.build_btn.isEnabled() and not tab.results

    def test_plot_suite_asked_before_a_run_builds_then_exports(self, qtbot, sample_dataset, tmp_path):
        tab = _make_tab(qtbot)
        sample_dataset.output_root = tmp_path
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na, Cd")
        tab.export_plot_suite()
        qtbot.wait(20)
        assert tab.results and len(list(tmp_path.glob("seaborn_plot_suite_*"))) == 1
        assert tab._after_build is None


@pytest.fixture(scope="module")
def real_dataset():
    dataset = HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=False, attributes_path=REAL_ATTRS_PATH)
    return dataset


@requires_real_data
class TestDefaultsOnTheShippedDataset:
    def test_every_default_element_carries_a_kg_inventory(self, qtbot, real_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(real_dataset)
        tab.run()
        qtbot.wait(20)
        stats = tab._table_views["Element stats"].dataframe()
        assert sorted(stats["Element"]) == sorted(qcw.csci.parse_element_list(DEFAULT_ELEMENTS))
        assert len(stats) == 11 and (stats["Total_inventory_kg"] > 0).all()
        assert tab.warnings_label.isHidden()
        pairs = tab._table_views["Pair stats"].dataframe()
        assert len(pairs) == 55

    def test_the_former_default_is_explained_not_refused(self, qtbot, real_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(real_dataset)
        tab.elements_edit.setText("Cs, Sr, Tc, I, Se, U, Cr, Fe, Al, Na, P, B, Si")
        tab.run()
        qtbot.wait(20)
        assert list(tab.warnings_label.notes()) == [
            "Cs, Tc and I have no inventory in kg in this dataset: they are reported in Ci only. Such a "
            "column is zero in every tank and adds nothing to the result; the Cross-correlation tab can "
            "place a nuclide beside a mass."]
