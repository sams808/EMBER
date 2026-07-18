from PySide6.QtWidgets import QMainWindow

import qt_correlations_structure as qcs
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
        options = [tab.color_by_combo.itemText(i) for i in range(tab.color_by_combo.count())]
        assert options == ["None"] + ssci.CATEGORY_FIELDS

    def test_pca_scatter_colored_by_tank_farm(self, qtbot, structure_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(structure_dataset)
        tab.elements_edit.setText("Cs, Sr, Mo, Fe")
        tab.n_clusters_spin.setValue(2)
        tab.color_by_combo.setCurrentText("TankFarm")
        tab.plot_type_combo.setCurrentText("PCA scatter")
        tab.build_data()
        qtbot.wait(20)

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

    def test_export_html_views_raises_when_pca_too_few_tanks(self, qtbot, sample_dataset, monkeypatch):
        # sample_dataset has only 2 tanks -- kg_correlation_workbench
        # succeeds (>=2 elements) so build_data doesn't warn, but PCA needs
        # >=3 tanks so tank_summary comes back empty -> the PCA HTML export
        # must raise ValueError, caught and shown as a warning, not crash.
        opened = []
        monkeypatch.setattr(qcs.os, "startfile", lambda p: opened.append(p))
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(sample_dataset)
        tab.elements_edit.setText("Fe, Na")
        tab.build_data()
        qtbot.wait(20)
        assert tab.results.get("tank_summary", None) is not None
        assert tab.results["tank_summary"].empty
        tab.export_html_views()
        assert opened == []
