from PySide6.QtWidgets import QMainWindow

from qt_correlations_workbench import PLOT_TYPES, WorkbenchTab


def _make_tab(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    tab = WorkbenchTab(app_window)
    qtbot.addWidget(tab)
    return tab


class TestWorkbenchTab:
    def test_actions_without_dataset_show_message_not_crash(self, qtbot):
        tab = _make_tab(qtbot)
        tab.build_data()
        tab.plot_selected()
        tab.export_tables()
        tab.export_plot_suite()

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
