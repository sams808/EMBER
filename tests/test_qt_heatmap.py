from PySide6.QtWidgets import QMainWindow

from qt_heatmap import MODES, HeatmapPage


def _make_page(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    page = HeatmapPage(app_window)
    qtbot.addWidget(page)
    return page


class TestHeatmapPage:
    def test_run_without_dataset_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page.run()

    def test_on_dataset_changed_defaults_to_kg(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        assert page.unit_combo.currentText() == "kg"

    def test_run_builds_tables_and_plot(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.run()
        qtbot.wait(20)
        wide = page.wide_view.dataframe()
        assert "241-A-101" in wide["WasteSiteId"].values
        assert set(page.long_view.dataframe()["Element"]) == {"Na", "Fe", "Cd"}

    def test_all_modes_render_without_crash(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        for mode in MODES:
            page.mode_combo.setCurrentText(mode)
            page.run()
            qtbot.wait(20)

    def test_max_tanks_subset_restricts_result(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.max_tanks_spin.setValue(1)
        page.run()
        qtbot.wait(20)
        assert len(page.wide_view.dataframe()) == 1

    def test_export_without_data_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page._export_inputs()

    def test_export_writes_inputs_and_plot(self, qtbot, sample_dataset, tmp_path):
        page = _make_page(qtbot)
        sample_dataset.output_root = tmp_path
        page.on_dataset_changed(sample_dataset)
        page.run()
        qtbot.wait(20)
        page._export_inputs()
        bundles = list(tmp_path.glob("heatmap_kg_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "heatmap_long.csv").exists()
        assert (bundles[0] / "heatmap_wide.csv").exists()
        assert (bundles[0] / "heatmap_parameters.csv").exists()
        assert (bundles[0] / "heatmap.png").exists()

    def test_dataset_not_loaded_shows_plot_message(self, qtbot):
        import data_model as dm
        page = _make_page(qtbot)
        page.on_dataset_changed(dm.HanfordDataset())
        qtbot.wait(20)
