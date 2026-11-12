from PySide6.QtWidgets import QMainWindow

from qt_tank_explorer import TankExplorerPage


def _make_page(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    page = TankExplorerPage(app_window)
    qtbot.addWidget(page)
    return page


class TestTankExplorerPage:
    def test_run_without_dataset_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page.run()

    def test_on_dataset_changed_populates_farm_and_tanks(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        farms = [page.farm_combo.itemText(i) for i in range(page.farm_combo.count())]
        assert farms == ["All", "A", "AN"]
        tanks = [page.tank_list.item(i).text() for i in range(page.tank_list.count())]
        assert tanks == ["241-A-101", "241-AN-104"]

    def test_farm_filter_narrows_tank_list(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.farm_combo.setCurrentText("A")
        tanks = [page.tank_list.item(i).text() for i in range(page.tank_list.count())]
        assert tanks == ["241-A-101"]

    def test_run_without_selection_shows_warning_not_crash(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.run()  # nothing selected; QMessageBox.warning neutralized by conftest

    def test_run_populates_tables_and_plot(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.tank_list.item(0).setSelected(True)  # 241-A-101
        page.unit_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)
        profile = page.profile_view.dataframe()
        assert set(profile["Element"]) == {"Fe", "Cd"}
        raw = page.raw_view.dataframe()
        assert set(raw["Units"]) == {"kg"}

    def test_export_without_run_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page._export_tables()

    def test_export_writes_tables_and_plot(self, qtbot, sample_dataset, tmp_path):
        page = _make_page(qtbot)
        sample_dataset.output_root = tmp_path
        page.on_dataset_changed(sample_dataset)
        page.tank_list.item(0).setSelected(True)
        page.run()
        qtbot.wait(20)
        page._export_tables()
        bundles = list(tmp_path.glob("tank_view_*"))
        assert len(bundles) == 1
        assert (bundles[0] / "tank_composition.csv").exists()
        assert (bundles[0] / "tank_plot.png").exists()

    def test_dataset_not_loaded_shows_plot_message(self, qtbot):
        import data_model as dm
        page = _make_page(qtbot)
        empty_dataset = dm.HanfordDataset()
        page.on_dataset_changed(empty_dataset)
        qtbot.wait(20)


class TestTankProfile:
    """A tank is a place that can be opened from anywhere: show_tank
    selects it, runs, and brings a profile into view."""

    def test_show_tank_selects_runs_and_shows_the_profile(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.result_stack.currentWidget() is page.empty_state
        assert page.show_tank("241-AN-104") is True
        qtbot.wait(20)
        assert page.selected_tanks() == ["241-AN-104"]
        assert page.result_stack.currentWidget() is page.result_scroll
        assert page.profile_title.text() == "241-AN-104"
        assert set(page.profile_view.dataframe()["WasteSiteId"]) == {"241-AN-104"}
        assert page.mini_map.colour_of("241-AN-104") is not None
        assert page.mini_map.colour_of("241-A-101") is None

    def test_show_tank_clears_a_farm_filter_and_a_text_filter(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.farm_combo.setCurrentText("A")
        page.filter_edit.setText("zzz")
        assert page.show_tank("241-AN-104") is True
        assert page.farm_combo.currentText() == "All"
        assert page.filter_edit.text() == ""
        assert page.selected_tanks() == ["241-AN-104"]

    def test_show_tank_refuses_an_unknown_tank_or_no_dataset(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        assert page.show_tank("241-AN-104") is False
        page.on_dataset_changed(sample_dataset)
        assert page.show_tank("241-ZZ-999") is False
        assert page.result_stack.currentWidget() is page.empty_state

    def test_profile_header_and_tiles(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.show_tank("241-AN-104")
        qtbot.wait(20)
        pills = [page.pill_row.itemAt(i).widget().text() for i in range(page.pill_row.count())]
        assert "AN farm" in pills
        assert "Double-shell tank" in pills
        assert any(p.startswith("Capacity 1,160 kgal") for p in pills)
        assert "Integrity: Assumed leaker" in pills
        assert page.tile_sodium.value() == "500 kg"
        assert page.tile_activity.value() == "200 Ci"
        assert page.tile_volume.value() == "—"            # the fixture states no volume
        assert page.side_title.text() == "Waste layers"
        assert list(page.layers_view.dataframe()["Waste phase"]) == ["Sludge"]

    def test_several_tanks_are_a_comparison(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.show_tanks(["241-A-101", "241-AN-104"]) is True
        qtbot.wait(20)
        assert page.profile_title.text() == "Comparison of 2 tanks"
        assert page.side_title.text() == "Tanks compared"
        assert list(page.layers_view.dataframe()["Tank"]) == ["241-A-101", "241-AN-104"]
        assert set(page.profile_view.dataframe()["WasteSiteId"]) == {"241-A-101", "241-AN-104"}
        assert page.mini_map.colour_of("241-A-101") != page.mini_map.colour_of("241-AN-104")
        legend = page.plot.ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["241-A-101", "241-AN-104"]

    def test_chart_is_logarithmic_with_element_labels_only(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.show_tank("241-A-101")
        qtbot.wait(20)
        ax = page.plot.ax
        assert ax.get_xscale() == "log"
        assert "Inventory (kg)" in ax.get_xlabel() and "log scale" in ax.get_xlabel()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["Cd", "Fe"]                      # largest at the top, no "[kg] (tank)" suffixes
        assert "241-A-101" in ax.get_title()

    def test_many_tanks_are_drawn_as_a_range_not_as_bars(self, qtbot, structure_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(structure_dataset)
        tanks = [page.tank_list.item(i).text() for i in range(page.tank_list.count())]
        assert len(tanks) == 8
        assert page.show_tanks(tanks) is True
        qtbot.wait(20)
        assert len(page.plot.ax.patches) == 0              # no bars
        legend = [t.get_text() for t in page.plot.ax.get_legend().get_texts()]
        assert legend == ["range across 8 tanks", "median"]

    def test_filter_box_narrows_the_list_and_keeps_the_items(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.filter_edit.setText("an")
        hidden = {page.tank_list.item(i).text(): page.tank_list.item(i).isHidden()
                  for i in range(page.tank_list.count())}
        assert hidden == {"241-A-101": True, "241-AN-104": False}
        page.filter_edit.clear()
        assert not any(page.tank_list.item(i).isHidden() for i in range(page.tank_list.count()))

    def test_selection_count_is_stated(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        assert page.selection_label.text() == "No tank selected (2 listed)"
        page.tank_list.item(0).setSelected(True)
        assert page.selection_label.text() == "1 of 2 selected"
        page.tank_list.item(1).setSelected(True)
        assert page.selection_label.text() == "2 of 2 selected"
        page.clear_btn.click()
        assert page.selected_tanks() == []

    def test_double_click_in_the_mini_map_opens_that_tank(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.show_tank("241-A-101")
        page.mini_map.tankActivated.emit("241-AN-104")
        qtbot.wait(20)
        assert page.profile_title.text() == "241-AN-104"

    def test_note_about_unattributed_mass_is_on_the_page(self, qtbot):
        import data_model as dm
        import polars as pl
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101", "241-A-101"],
            "Analyte": ["Na", "Aroclors (Total PCB)"],
            "WastePhase": ["Sludge", "Sludge"],
            "WasteType": ["T1", "T1"],
            "Inventory": [5.0, 0.4],
            "Units": ["kg", "kg"],
        }))
        dataset.report = None
        page = _make_page(qtbot)
        page.on_dataset_changed(dataset)
        page.show_tank("241-A-101")
        qtbot.wait(20)
        assert "Not shown" in page.notes_banner.text()
        assert not page.notes_banner.isHidden()
