from PySide6.QtWidgets import QComboBox, QMainWindow, QTableWidgetItem

import qt_vitrification_oxide as qvo
from qt_vitrification_oxide import OxideChemistryTab


def _make_tab(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    tab = OxideChemistryTab(app_window)
    qtbot.addWidget(tab)
    return tab


def _select_tanks(tab, tank_ids):
    tab.tank_list.clearSelection()
    for i in range(tab.tank_list.count()):
        item = tab.tank_list.item(i)
        if item.text() in tank_ids:
            item.setSelected(True)


class TestOxideChemistryTab:
    def test_actions_without_dataset_show_message_not_crash(self, qtbot):
        tab = _make_tab(qtbot)
        tab.build_composition()
        tab.recompute_summary()
        tab.export_tables()
        tab.run_glassnet()

    def test_build_composition_no_tanks_selected_shows_warning(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        tab.build_composition()  # QMessageBox.warning neutralized by conftest

    def test_build_composition_populates_tables(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        assert tab.oxide_map_table.rowCount() == 2  # Na, Si
        assert tab.sample_combo.count() == 1
        composition = tab._table_views["Oxide composition"].dataframe()
        assert set(composition["Component"]) == {"Na2O", "SiO2"}

    def test_blend_row_appears_for_multiple_tanks(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101", "241-A-103"])
        tab.build_composition()
        assert "Blend (selected tanks)" in tab.oxide_tables
        assert tab.sample_combo.findText("Blend (selected tanks)") >= 0

    def test_role_table_defaults_match_science_layer(self, qtbot, oxide_dataset):
        import oxide_science as oxsci
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        role_map = tab._current_role_map()
        assert role_map["SiO2"] == oxsci.default_oxide_role("SiO2")
        assert role_map["Na2O"] == oxsci.default_oxide_role("Na2O")

    def test_multivalent_element_gets_combo_with_alternatives(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-102"])  # includes Pu, a multivalent element
        tab.build_composition()
        pu_row = next(r for r in range(tab.oxide_map_table.rowCount()) if tab.oxide_map_table.item(r, 0).text() == "Pu")
        widget = tab.oxide_map_table.cellWidget(pu_row, 1)
        assert isinstance(widget, QComboBox)
        assert widget.count() == 2  # PuO2, Pu2O3

    def test_recompute_summary_shows_basicity_and_excludes_pu(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-102"])
        tab.build_composition()
        tab.sample_combo.setCurrentText("241-A-102")
        tab.recompute_summary()
        assert "Optical basicity" in tab.summary_label.text()
        assert "PuO2" in tab.summary_label.text()

    def test_editing_oxide_map_cell_changes_conversion(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        na_row = next(r for r in range(tab.oxide_map_table.rowCount()) if tab.oxide_map_table.item(r, 0).text() == "Na")
        tab.oxide_map_table.setItem(na_row, 1, QTableWidgetItem(""))  # force elemental instead of Na2O
        tab._rebuild_and_show()
        composition = tab._table_views["Oxide composition"].dataframe()
        assert "Na" in set(composition["Component"])
        assert "Na2O" not in set(composition["Component"])

    def test_envelope_add_and_remove_rows(self, qtbot):
        tab = _make_tab(qtbot)
        tab._add_envelope_row()
        tab._add_envelope_row()
        assert tab.envelope_table.rowCount() == 2
        tab.envelope_table.setCurrentCell(0, 0)
        tab._remove_envelope_row()
        assert tab.envelope_table.rowCount() == 1

    def test_envelope_wired_into_summary(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        tab._add_envelope_row()
        tab.envelope_table.setItem(0, 0, QTableWidgetItem("SiO2"))
        tab.envelope_table.setItem(0, 1, QTableWidgetItem("0"))
        tab.envelope_table.setItem(0, 2, QTableWidgetItem("100"))
        tab.recompute_summary()
        envelope_view = tab._table_views["Envelope check"].dataframe()
        assert not envelope_view.empty
        assert "SiO2" in set(envelope_view["Oxide"])

    def test_envelope_row_with_invalid_numbers_skipped_not_crashed(self, qtbot):
        tab = _make_tab(qtbot)
        tab._add_envelope_row()
        tab.envelope_table.setItem(0, 0, QTableWidgetItem("SiO2"))
        tab.envelope_table.setItem(0, 1, QTableWidgetItem("not-a-number"))
        assert tab._current_envelope() == {}

    def test_envelope_row_with_blank_oxide_skipped(self, qtbot):
        tab = _make_tab(qtbot)
        tab._add_envelope_row()
        tab.envelope_table.setItem(0, 0, QTableWidgetItem("  "))
        assert tab._current_envelope() == {}

    def test_save_and_load_envelope_json(self, qtbot, tmp_path, monkeypatch):
        tab = _make_tab(qtbot)
        tab._add_envelope_row()
        tab.envelope_table.setItem(0, 0, QTableWidgetItem("SiO2"))
        tab.envelope_table.setItem(0, 1, QTableWidgetItem("40"))
        tab.envelope_table.setItem(0, 2, QTableWidgetItem("60"))
        save_path = tmp_path / "envelope.json"
        monkeypatch.setattr(qvo.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(save_path), "")))
        tab.save_envelope()
        assert save_path.exists()

        tab2 = _make_tab(qtbot)
        monkeypatch.setattr(qvo.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(save_path), "")))
        tab2.load_envelope()
        assert tab2.envelope_table.rowCount() == 1
        assert tab2.envelope_table.item(0, 0).text() == "SiO2"

    def test_save_envelope_with_no_rows_shows_message_not_crash(self, qtbot):
        tab = _make_tab(qtbot)
        tab.save_envelope()

    def test_load_envelope_cancelled_dialog_does_nothing(self, qtbot, monkeypatch):
        tab = _make_tab(qtbot)
        monkeypatch.setattr(qvo.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
        tab.load_envelope()
        assert tab.envelope_table.rowCount() == 0

    def test_save_envelope_cancelled_dialog_does_nothing(self, qtbot, monkeypatch):
        tab = _make_tab(qtbot)
        tab._add_envelope_row()
        tab.envelope_table.setItem(0, 0, QTableWidgetItem("SiO2"))
        monkeypatch.setattr(qvo.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
        tab.save_envelope()  # must not raise

    def test_load_envelope_bad_json_shows_warning_not_crash(self, qtbot, tmp_path, monkeypatch):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json", encoding="utf-8")
        tab = _make_tab(qtbot)
        monkeypatch.setattr(qvo.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad_path), "")))
        tab.load_envelope()

    def test_export_tables_writes_bundle(self, qtbot, oxide_dataset, tmp_path):
        tab = _make_tab(qtbot)
        oxide_dataset.output_root = tmp_path
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        tab.export_tables()
        bundles = list(tmp_path.glob("oxide_chemistry_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "241-A-101.csv").exists()
        assert (bundles[0] / "oxide_map_used.csv").exists()
        assert (bundles[0] / "role_map_used.csv").exists()

    def test_glassnet_predict_flow_mocked(self, qtbot, oxide_dataset, monkeypatch):
        import pandas as pd
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        monkeypatch.setattr(qvo.gscience, "glassnet_predict", lambda df: pd.DataFrame([{"Tg_K": 750.0}]))
        tab.run_glassnet()
        pred = tab._table_views["GlassNet"].dataframe()
        assert not pred.empty
        assert pred.iloc[0]["Tg_K"] == 750.0
        assert tab.gn_btn.isEnabled()

    def test_glassnet_predict_error_shows_critical_not_crash(self, qtbot, oxide_dataset, monkeypatch):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()

        def boom(df):
            raise RuntimeError("boom")

        monkeypatch.setattr(qvo.gscience, "glassnet_predict", boom)
        tab.run_glassnet()
        assert tab.gn_btn.isEnabled()

    def test_glassnet_no_oxide_components_shows_message(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        na_row = next(r for r in range(tab.oxide_map_table.rowCount()) if tab.oxide_map_table.item(r, 0).text() == "Na")
        si_row = next(r for r in range(tab.oxide_map_table.rowCount()) if tab.oxide_map_table.item(r, 0).text() == "Si")
        tab.oxide_map_table.setItem(na_row, 1, QTableWidgetItem(""))
        tab.oxide_map_table.setItem(si_row, 1, QTableWidgetItem(""))
        tab._rebuild_and_show()
        tab.run_glassnet()  # both elemental now -> no oxide rows for GlassNet


class TestCoverageGaps:
    def test_glassnet_disabled_when_unavailable(self, qtbot, monkeypatch):
        monkeypatch.setattr(qvo.gscience, "glassnet_available", lambda: False)
        tab = _make_tab(qtbot)
        assert not tab.gn_btn.isEnabled()
        assert "glasspy" in tab.gn_btn.toolTip()

    def test_refresh_tank_list_before_any_dataset_set_is_a_noop(self, qtbot):
        tab = _make_tab(qtbot)
        tab._refresh_tank_list()  # self.dataset is still None
        assert tab.tank_list.count() == 0

    def test_current_oxide_map_skips_blank_element_cell(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        tab.oxide_map_table.setItem(0, 0, QTableWidgetItem(""))
        out = tab._current_oxide_map()
        assert len(out) == 1  # only the other (non-blanked) element row counted

    def test_current_role_map_skips_row_without_combo(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        tab.role_table.setCellWidget(0, 1, None)
        out = tab._current_role_map()
        assert len(out) == 1

    def test_build_composition_tank_with_no_kg_elements_shows_warning(self, qtbot):
        import data_model as dm
        import polars as pl
        rows = {
            "WasteSiteId": ["241-A-999"], "Analyte": ["137Cs"], "WastePhase": ["Liquid"],
            "WasteType": ["T1"], "Inventory": [10.0], "Units": ["Ci"],  # Ci only, no kg
        }
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame(rows))
        dataset.report = None
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(dataset)
        _select_tanks(tab, ["241-A-999"])
        tab.build_composition()  # hits the "no elements" warning branch

    def test_rebuild_with_no_tanks_selected_shows_warning(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        tab._rebuild_and_show()  # no tanks selected -> tank_oxide_composition returns {}

    def test_sample_changed_signal_fires_on_real_switch(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101", "241-A-103"])
        tab.build_composition()
        assert tab.sample_combo.count() == 3  # 2 tanks + blend
        other = next(tab.sample_combo.itemText(i) for i in range(tab.sample_combo.count()) if tab.sample_combo.itemText(i) != tab.sample_combo.currentText())
        tab.sample_combo.setCurrentText(other)
        assert tab._table_views["Oxide composition"].dataframe() is tab.oxide_tables[other]

    def test_show_current_sample_with_unknown_selection_is_noop(self, qtbot):
        tab = _make_tab(qtbot)
        tab._show_current_sample()  # sample_combo is empty, currentText() "" not in oxide_tables

    def test_export_includes_envelope_check_when_populated(self, qtbot, oxide_dataset, tmp_path):
        tab = _make_tab(qtbot)
        oxide_dataset.output_root = tmp_path
        tab.on_dataset_changed(oxide_dataset)
        _select_tanks(tab, ["241-A-101"])
        tab.build_composition()
        tab._add_envelope_row()
        tab.envelope_table.setItem(0, 0, QTableWidgetItem("SiO2"))
        tab.envelope_table.setItem(0, 1, QTableWidgetItem("0"))
        tab.envelope_table.setItem(0, 2, QTableWidgetItem("100"))
        tab.recompute_summary()
        tab.export_tables()
        bundles = list(tmp_path.glob("oxide_chemistry_*"))
        assert (bundles[0] / "envelope_check.csv").exists()


class TestOnDatasetChanged:
    def test_farm_and_tank_list_populate(self, qtbot, oxide_dataset):
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(oxide_dataset)
        assert tab.farm_combo.count() >= 1
        assert tab.tank_list.count() == 3

    def test_unloaded_dataset_leaves_lists_empty(self, qtbot):
        import data_model as dm
        tab = _make_tab(qtbot)
        tab.on_dataset_changed(dm.HanfordDataset())
        assert tab.tank_list.count() == 0
