import pytest
from PySide6.QtWidgets import QMainWindow

import qt_correlations_structure as qcs
from conftest import requires_real_data
from qt_correlations_structure import STRUCTURE_PLOT_TYPES, StructureTab


def _make_tab(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    tab = StructureTab(app_window)
    qtbot.addWidget(tab)
    return tab


class TestStructureTab:
    def test_actions_without_dataset_show_message_not_crash(self, qtbot):
        tab = _make_tab(qtbot)
        tab.build_data()
        tab.plot_selected()
        tab.export_tables()
        tab.export_html_views()

    def test_build_data_populates_tables(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.build_data()
        qtbot.wait(20)
        summary = tab._table_views["Tank summary"].dataframe()
        assert set(summary["WasteSiteId"]) == set(structure_dataset.available_tanks())
        assert "PC1" in summary.columns and "Cluster" in summary.columns
        stats = tab._table_views["Element stats"].dataframe()
        assert set(stats["Element"]) == {"Cs", "Sr", "Mo", "Fe"}

    def test_build_data_error_shows_warning_not_crash(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs")  # only one valid element -> ValueError
        tab.build_data()  # QMessageBox.warning neutralized by conftest

    def test_all_plot_types_render_without_crash(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.build_data()
        qtbot.wait(20)
        for plot_type in STRUCTURE_PLOT_TYPES:
            tab.plot_type_combo.setCurrentText(plot_type)
            tab.plot_selected()
            qtbot.wait(20)

    def test_plot_selected_without_prior_build_builds_first(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        assert not tab.results
        tab.plot_selected()
        qtbot.wait(20)
        assert tab.results

    def test_color_by_options_include_category_fields(self, qtbot):
        import structure_science as ssci
        tab = _make_tab(qtbot)
        # The items read as words; the column each one colors by rides as
        # item data, and the cluster a tank was assigned to is offered too.
        columns = [tab.color_by_combo.itemData(i) for i in range(tab.color_by_combo.count())]
        assert columns == [None] + ssci.CATEGORY_FIELDS + ["Cluster"]
        labels = [tab.color_by_combo.itemText(i) for i in range(tab.color_by_combo.count())]
        assert labels[0] == "None" and "TankFarm" not in labels and "Farm" in labels
        assert tab._current_color_by() is None

    def test_pca_scatter_colored_by_tank_farm(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.color_by_combo.setCurrentIndex(tab.color_by_combo.findData("TankFarm"))
        assert tab._current_color_by() == "TankFarm"
        tab.plot_type_combo.setCurrentText("PCA scatter")
        tab.build_data()
        qtbot.wait(20)
        legend = tab.plot.ax.get_legend()
        assert legend is not None and {t.get_text() for t in legend.get_texts()} == {"A", "AN"}

    def test_network_thresholds_applied(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.min_abs_r_spin.setValue(0.99)
        tab.plot_type_combo.setCurrentText("Element network")
        tab.build_data()
        qtbot.wait(20)
        edges = tab._table_views["Network edges"].dataframe()
        # Only Cs-Sr clears a 0.99 |r| floor under log10_plus1 (see
        # test_structure_science.py's threshold test for the exact numbers).
        assert len(edges) <= 1

    def test_use_partial_for_network_checkbox(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.use_partial_check.setChecked(True)
        tab.plot_type_combo.setCurrentText("Element network")
        tab.build_data()
        qtbot.wait(20)

    def test_coherent_colors_mode(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.color_mode_combo.setCurrentText("Coherent colors")
        tab.build_data()
        qtbot.wait(20)

    def test_export_tables_writes_bundle(self, qtbot, structure_dataset, tmp_path):
        tab = _make_tab(qtbot)
        structure_dataset.output_root = tmp_path
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.build_data()
        qtbot.wait(20)
        tab.export_tables()
        bundles = list(tmp_path.glob("structure_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "tank_summary.csv").exists()
        assert (bundles[0] / "cluster_linkage.csv").exists()
        assert (bundles[0] / "settings.csv").exists()
        assert (bundles[0] / "current_plot.png").exists()

    def test_export_tables_survives_savefig_failure(self, qtbot, structure_dataset, tmp_path, monkeypatch):
        tab = _make_tab(qtbot)
        structure_dataset.output_root = tmp_path
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.build_data()
        qtbot.wait(20)

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(tab.plot.figure, "savefig", boom)
        tab.export_tables()  # must not raise despite the plot PNG save failing
        bundles = list(tmp_path.glob("structure_*"))
        assert len(bundles) == 1
        assert not (bundles[0] / "current_plot.png").exists()

    def test_export_html_views_writes_files_and_opens_folder(self, qtbot, structure_dataset, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        tab = _make_tab(qtbot)
        structure_dataset.output_root = tmp_path
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.build_data()
        qtbot.wait(20)
        tab.export_html_views()
        bundles = list(tmp_path.glob("structure_html_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "correlation_heatmap.html").exists()
        assert (bundles[0] / "pca_scatter.html").exists()
        assert (bundles[0] / "network_graph.html").exists()
        assert len(opened) == 1

    def test_export_html_views_without_dataset_shows_message_not_crash(self, qtbot, monkeypatch):
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        tab = _make_tab(qtbot)
        tab.export_html_views()
        assert opened == []

    def test_plot_selected_when_build_fails_returns_without_crash(self, qtbot, structure_dataset):
        # elements_edit left at the single-element default -> build_data
        # raises internally, self.results stays empty -> _ensure_results()
        # is False -> plot_selected must return quietly, not crash.
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs")
        tab.plot_selected()
        assert not tab.results

    def test_unknown_plot_type_shows_message(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.build_data()
        qtbot.wait(20)
        tab.plot_type_combo.addItem("Bogus plot type")
        tab.plot_type_combo.setCurrentText("Bogus plot type")
        tab.plot_selected()
        qtbot.wait(20)
        assert "Unknown plot type" in tab.plot.ax.texts[-1].get_text()

    def test_export_html_views_when_build_fails_returns_without_crash(self, qtbot, structure_dataset, monkeypatch):
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs")
        tab.export_html_views()
        assert opened == []

    def test_export_html_views_raises_when_pca_too_few_tanks(self, qtbot, sample_dataset, tmp_path, monkeypatch):
        # sample_dataset has only 2 tanks -- kg_correlation_workbench
        # succeeds (>=2 elements) so build_data doesn't warn, but PCA needs
        # >=3 tanks so tank_summary comes back empty -> the PCA HTML export
        # must raise ValueError, caught and shown as a warning, not crash.
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        # The first of the three pages is written before the second one
        # refuses: into the test's own folder, never the working directory.
        sample_dataset.output_root = tmp_path
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na")
        tab.build_data()
        qtbot.wait(20)
        assert tab.results.get("tank_summary", None) is not None
        assert tab.results["tank_summary"].empty
        tab.export_html_views()
        assert opened == []


def _status(tab) -> str:
    return tab.app_window.statusBar().currentMessage()


def _capture(monkeypatch, name):
    from PySide6.QtWidgets import QMessageBox
    calls = []
    monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: calls.append(a) or QMessageBox.Ok))
    return calls


def _built(qtbot, dataset, elements="Cs, Sr, Mo, Fe", clusters=2):
    tab = _make_tab(qtbot)
    tab.on_dataset_changed(dataset)
    tab.elements_edit.setText(elements)
    tab.n_clusters_spin.setValue(clusters)
    tab.run()
    qtbot.wait(20)
    return tab


class TestPagePass:
    def test_opening_state(self, qtbot):
        from qt_correlations_workbench import DEFAULT_ELEMENTS
        tab = _make_tab(qtbot)
        assert tab.elements_edit.text() == DEFAULT_ELEMENTS
        assert tab.run_btn is tab.build_btn
        assert tab.build_btn.objectName() == "Primary" and tab.build_btn.text() == "Run"
        assert tab.warnings_label is tab.notes_banner and tab.warnings_label.isHidden()
        assert not tab.advanced_section.is_expanded()
        assert [t.get_text() for t in tab.plot.ax.texts][0] == "No result yet"
        assert tab.header.title_label.text() == "Structure" and tab.header.subtitle_label.wordWrap()
        # One Export button, two exports behind it.
        assert tab.export_btn.text() == "Export..."
        assert [a.text() for a in tab.export_menu.actions()] == [
            "Tables and current chart", "Interactive HTML views"]

    def test_actions_without_dataset_say_so_in_the_status_bar(self, qtbot, monkeypatch):
        info = _capture(monkeypatch, "information")
        tab = _make_tab(qtbot)
        for action in (tab.build_data, tab.plot_selected, tab.export_html_views, tab.run):
            tab.app_window.statusBar().clearMessage()
            action()
            assert "no dataset is loaded" in _status(tab)
        tab.export_tables()
        assert "nothing to export" in _status(tab)
        assert not info

    def test_export_menu_runs_the_two_exports(self, qtbot, structure_dataset, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        structure_dataset.output_root = tmp_path
        tab = _built(qtbot, structure_dataset)
        tables_action, html_action = tab.export_menu.actions()
        tables_action.trigger()
        html_action.trigger()
        assert len(list(tmp_path.glob("structure_2*"))) == 1
        assert len(list(tmp_path.glob("structure_html_*"))) == 1 and len(opened) == 1

    def test_clicking_a_tank_of_the_pca_selects_it_in_the_tank_summary(self, qtbot, structure_dataset):
        tab = _built(qtbot, structure_dataset)
        tab.tables.setCurrentWidget(tab._table_views["PCA loadings"])
        tab._on_mark_clicked({"label": "241-AN-103", "tank_id": "241-AN-103", "series": "", "value": 0.1})
        view = tab._table_views["Tank summary"]
        assert tab.tables.currentWidget() is view
        selected = view.selectedIndexes()
        assert selected and view.dataframe()["WasteSiteId"].iloc[selected[0].row()] == "241-AN-103"
        tab._on_mark_clicked({"label": "not a tank"})          # no match: selection untouched, no error
        assert view.dataframe()["WasteSiteId"].iloc[view.selectedIndexes()[0].row()] == "241-AN-103"

    def test_a_real_click_on_a_pca_point_reaches_the_table(self, qtbot, structure_dataset):
        """The whole chain: a matplotlib press on a drawn point, the plot's
        markClicked signal, the row of that tank in the table."""
        from matplotlib.backend_bases import MouseEvent
        tab = _built(qtbot, structure_dataset)
        canvas = tab.plot.canvas
        canvas.draw()
        summary = tab._table_views["Tank summary"].dataframe()
        target = summary.iloc[5]
        px, py = tab.plot.ax.transData.transform((float(target["PC1"]), float(target["PC2"])))
        seen = []
        tab.plot.markClicked.connect(seen.append)
        for kind in ("button_press_event", "button_release_event"):
            canvas.callbacks.process(kind, MouseEvent(kind, canvas, px, py, button=1))
        qtbot.wait(20)
        assert seen and seen[-1].get("tank_id") == target["WasteSiteId"]
        view = tab._table_views["Tank summary"]
        assert view.dataframe()["WasteSiteId"].iloc[view.selectedIndexes()[0].row()] == target["WasteSiteId"]

    def test_clicking_an_element_selects_its_row(self, qtbot, structure_dataset):
        tab = _built(qtbot, structure_dataset)
        tab._on_mark_clicked({"label": "Mo"})
        view = tab._table_views["Element stats"]
        assert tab.tables.currentWidget() is view
        assert view.dataframe()["Element"].iloc[view.selectedIndexes()[0].row()] == "Mo"

    def test_color_by_is_a_display_choice(self, qtbot, structure_dataset, monkeypatch):
        tab = _built(qtbot, structure_dataset)
        assert tab.plot.ax.get_legend() is None
        started = []
        monkeypatch.setattr(qcs, "run_in_thread", lambda *a, **k: started.append(a))
        tab.color_by_combo.setCurrentIndex(tab.color_by_combo.findData("Cluster"))
        qtbot.wait(20)
        legend = tab.plot.ax.get_legend()
        assert legend is not None and {t.get_text() for t in legend.get_texts()} == {"1", "2"}
        assert not started and tab.build_btn.text() == "Run"
        # Only the PCA scatter is colored by a tank attribute.
        tab.plot_type_combo.setCurrentText("Dendrogram")
        assert not tab.color_by_combo.isEnabled() and not started
        tab.plot_type_combo.setCurrentText("PCA scatter")
        assert tab.color_by_combo.isEnabled()

    def test_run_button_says_update_when_the_controls_left_the_result(self, qtbot, structure_dataset):
        tab = _built(qtbot, structure_dataset)
        assert tab.build_btn.text() == "Run"
        tab.n_clusters_spin.setValue(3)
        assert tab.build_btn.text() == "Update"
        tab.n_clusters_spin.setValue(2)
        assert tab.build_btn.text() == "Run"
        tab.min_abs_r_spin.setValue(0.5)
        assert tab.build_btn.text() == "Update"
        tab.run()
        qtbot.wait(20)
        assert tab.build_btn.text() == "Run"

    def test_notes_say_what_the_principal_components_left_out(self, qtbot, structure_dataset):
        tab = _built(qtbot, structure_dataset, elements="Cs, Sr, Mo, Fe, Tc, Zz9")
        notes = list(tab.warnings_label.notes())
        assert notes[0] == "Zz9 is not an element symbol and was left out."
        assert notes[1].startswith("Tc has no inventory in this dataset, in any unit.")
        # Iron is the same in every tank of the fixture, technetium is zero in every tank.
        assert notes[2] == ("Fe and Tc have the same value in every tank and were left out of the "
                            "principal components.")
        assert tab.warnings_label.summary_label.text().startswith("3 notes qualify this result")

    def test_too_few_tanks_is_said_instead_of_an_empty_chart_alone(self, qtbot, sample_dataset):
        tab = _built(qtbot, sample_dataset, elements="Fe, Na")
        assert tab.results["tank_summary"].empty
        assert any("Too few tanks" in note for note in tab.warnings_label.notes())

    def test_export_settings_describe_the_run(self, qtbot, structure_dataset, tmp_path):
        import pandas as pd
        structure_dataset.output_root = tmp_path
        tab = _built(qtbot, structure_dataset)
        tab.n_clusters_spin.setValue(5)                    # moved, not run
        tab.elements_edit.setText("Cs, Sr")
        tab.color_by_combo.setCurrentIndex(tab.color_by_combo.findData("TankFarm"))
        tab.export_tables()
        settings = pd.read_csv(next(tmp_path.glob("structure_2*")) / "settings.csv").iloc[0]
        assert settings["n_clusters"] == 2 and settings["elements_text"] == "Cs, Sr, Mo, Fe"
        assert settings["metric"] == "log10_plus1" and settings["cluster_method"] == "ward"
        assert settings["color_pca_by"] == "TankFarm"      # a display choice: as it stands

    def test_fields_offer_what_the_dataset_holds(self, qtbot, structure_dataset):
        from qt_correlations_workbench import completion_items
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        assert {"Cs", "Sr", "Mo", "Fe"} <= set(completion_items(tab.elements_edit))
        assert {"Cs", "Sr", "Mo", "Fe"} <= set(completion_items(tab.skip_edit))

    def test_dataset_change_clears_the_previous_result(self, qtbot, structure_dataset, sample_dataset):
        tab = _built(qtbot, structure_dataset)
        tab.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        assert not tab.results and tab._params == {}
        assert all(view.dataframe().empty for view in tab._table_views.values())
        assert tab.warnings_label.isHidden()
        assert [t.get_text() for t in tab.plot.ax.texts][0] == "No result yet"


class TestWorkerContract:
    def test_results_are_delivered_to_bound_methods_of_the_tab(self, qtbot, structure_dataset, monkeypatch):
        seen = {}
        monkeypatch.setattr(qcs, "run_in_thread",
                            lambda fn, on_done, on_error=None, *a, **k: seen.update(fn=fn, on_done=on_done,
                                                                                    on_error=on_error))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.run()
        assert seen["on_done"].__self__ is tab and seen["on_error"].__self__ is tab
        assert not tab.build_btn.isEnabled() and "Running" in tab.build_btn.text()
        outcome = seen["fn"]()
        assert outcome.token == tab._run_token and not outcome.results["tank_summary"].empty
        tab._on_run_done(outcome)
        qtbot.wait(20)
        assert tab.build_btn.isEnabled() and tab.build_btn.text() == "Run"
        assert len(tab._table_views["Tank summary"].dataframe()) == 8

    def test_a_superseded_answer_is_dropped(self, qtbot, structure_dataset, monkeypatch):
        pending = []
        monkeypatch.setattr(qcs, "run_in_thread", lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr")
        tab.run()
        tab.elements_edit.setText("Cs, Sr, Mo")
        tab._start(tab._read_params())
        first, second = pending[0](), pending[1]()
        tab._on_run_done(second)
        tab._on_run_done(first)                   # arrives late: must not overwrite
        qtbot.wait(20)
        assert set(tab._table_views["Element stats"].dataframe()["Element"]) == {"Cs", "Sr", "Mo"}

    def test_an_answer_for_the_previous_dataset_is_dropped(self, qtbot, structure_dataset, sample_dataset,
                                                           monkeypatch):
        pending = []
        monkeypatch.setattr(qcs, "run_in_thread", lambda fn, on_done, on_error=None, *a, **k: pending.append(fn))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo")
        tab.run()
        tab.on_dataset_changed(sample_dataset)
        tab._on_run_done(pending[0]())
        qtbot.wait(20)
        assert not tab.results and tab.build_btn.isEnabled()

    def test_html_views_asked_before_a_run_build_then_export(self, qtbot, structure_dataset, tmp_path,
                                                             monkeypatch):
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        structure_dataset.output_root = tmp_path
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.export_html_views()
        qtbot.wait(20)
        assert tab.results and len(opened) == 1 and tab._after_build is None

    def test_unexpected_failure_is_reported_not_raised(self, qtbot, structure_dataset, monkeypatch):
        criticals = _capture(monkeypatch, "critical")

        def boom(*args, **kwargs):
            raise RuntimeError("science fell over")

        monkeypatch.setattr(qcs.ssci, "structure_workbench", boom)
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.run()
        qtbot.wait(20)
        assert len(criticals) == 1 and "science fell over" in criticals[0][2]
        assert tab.build_btn.isEnabled() and not tab.results


@pytest.fixture(scope="module")
def real_dataset():
    from conftest import REAL_ATTRS_PATH, REAL_CSV_PATH
    from data_model import HanfordDataset
    dataset = HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=False, attributes_path=REAL_ATTRS_PATH)
    return dataset


@requires_real_data
class TestDefaultsOnTheShippedDataset:
    def test_default_list_gives_a_full_structure_without_notes(self, qtbot, real_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(real_dataset)
        tab.run()
        qtbot.wait(20)
        summary = tab._table_views["Tank summary"].dataframe()
        assert len(summary) == 177 and summary["PC1"].notna().all() and set(summary["Cluster"]) == {1, 2, 3, 4}
        assert len(tab._table_views["PCA loadings"].dataframe()) == 11       # no element dropped
        assert tab.warnings_label.isHidden()
        assert tab.plot.ax.get_title().startswith("Tank PCA")

    def test_the_former_default_is_explained(self, qtbot, real_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(real_dataset)
        tab.elements_edit.setText("Cs, Sr, Tc, I, Se, U, Cr, Fe, Al, Na, P, B, Si")
        tab.run()
        qtbot.wait(20)
        notes = list(tab.warnings_label.notes())
        assert notes[0].startswith("Cs, Tc and I have no inventory in kg in this dataset: they are reported in Ci only.")
        assert notes[1] == ("Cs, Tc and I have the same value in every tank and were left out of the "
                            "principal components.")
