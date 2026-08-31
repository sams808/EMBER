import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
import matrix_query as mq
import reference_data as rd
from qt_heatmap import MODES, HeatmapPage


def _make_page(qtbot):
    app_window = QMainWindow()
    qtbot.addWidget(app_window)
    page = HeatmapPage(app_window)
    qtbot.addWidget(page)
    return page


def _species_dataset() -> dm.HanfordDataset:
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-A-101", "241-AN-104"],
        "Analyte": ["SO4", "Fe", "NO3"],
        "WastePhase": ["Supernatant", "Supernatant", "Sludge"],
        "WasteType": ["T1", "T1", "T2"],
        "Inventory": [100.0, 10.0, 50.0],
        "Units": ["kg", "kg", "kg"],
    }))
    dataset.report = None
    return dataset


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
        page = _make_page(qtbot)
        page.on_dataset_changed(dm.HanfordDataset())
        qtbot.wait(20)


class TestAttributionMigration:
    """The workspace now answers "how much of element X is here",
    apportioned; the older "whose primary element is X" reading stays
    selectable and says which one produced the figure."""

    def test_default_attribution_apportions_species(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.run()
        qtbot.wait(20)
        wide = page.wide_view.dataframe().set_index("WasteSiteId")
        assert wide.loc["241-A-101", "S"] == pytest.approx(
            100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"])
        assert "O" in wide.columns  # oxygen exists only under apportionment

    def test_as_reported_attribution_reproduces_the_old_numbers(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.attribution_combo.setCurrentText(mq.AS_REPORTED)
        page.run()
        qtbot.wait(20)
        wide = page.wide_view.dataframe().set_index("WasteSiteId")
        assert wide.loc["241-A-101", "S"] == pytest.approx(100.0)
        assert "O" not in wide.columns

    def test_attribution_is_stamped_on_the_figure(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.run()
        qtbot.wait(20)
        assert mq.APPORTIONED in page.plot.ax.get_title()

    def test_as_reported_discloses_its_attribution_in_notes(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.attribution_combo.setCurrentText(mq.AS_REPORTED)
        page.run()
        qtbot.wait(20)
        notes = page.notes_view.dataframe()
        assert (notes["kind"] == "disclosure").any()
        assert notes["text"].str.contains("first parsed element").any()

    def test_min_inventory_warns_that_it_does_not_apply_when_apportioned(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.min_inv_spin.setValue(1.0)
        page.run()
        qtbot.wait(20)
        assert not page.warnings_label.isHidden()
        assert "min inventory" in page.warnings_label.text()


class TestGlobalScope:
    def test_scope_restricts_the_matrix_and_is_stamped(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.scope.set_filter("WastePhase", ["Sludge"])
        page.run()
        qtbot.wait(20)
        assert list(page.wide_view.dataframe()["WasteSiteId"]) == ["241-AN-104"]
        assert "Scope: Sludge" in page.plot.ax.get_title()

    def test_scope_applies_to_the_as_reported_path_too(self, qtbot):
        # One scope mechanism for both attributions, or "global" would
        # mean different things depending on a combo.
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.attribution_combo.setCurrentText(mq.AS_REPORTED)
        page.scope.set_filter("WastePhase", ["Sludge"])
        page.run()
        qtbot.wait(20)
        assert list(page.wide_view.dataframe()["WasteSiteId"]) == ["241-AN-104"]

    def test_scope_change_rebuilds_a_standing_matrix(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.run()
        qtbot.wait(20)
        assert len(page.wide_view.dataframe()) == 2
        page.scope.set_filter("WastePhase", ["Sludge"])
        page.on_scope_changed()
        qtbot.wait(20)
        assert len(page.wide_view.dataframe()) == 1

    def test_scope_change_without_a_matrix_does_not_build_one(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.scope.set_filter("WastePhase", ["Sludge"])
        page.on_scope_changed()
        assert page.wide_view.dataframe().empty


class TestFullMatrixIsNotClipped:
    @pytest.mark.skipif(
        not __import__("conftest").REAL_CSV_PATH.exists(),
        reason="Hanford.csv not present locally",
    )
    def test_every_tank_is_drawn_at_the_default_settings(self, qtbot):
        # The cell guardrail must clear the shipped dataset's own matrix
        # (177 tanks x 35 elements); it exists to stop an absurd render,
        # not to clip the data the app ships with.
        dataset = dm.HanfordDataset()
        dataset.load_local_default()
        page = _make_page(qtbot)
        page.on_dataset_changed(dataset)
        page.run()
        qtbot.wait(50)
        assert len(page.wide_view.dataframe()) == 177
        image = page.plot.ax.get_images()[0].get_array()
        assert image.shape == (177, 35)
        assert "rows shown" not in page.plot.ax.get_title()


class TestModesRenderThroughTheGrammar:
    def test_fraction_mode_normalizes_rows(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.mode_combo.setCurrentText("fraction")
        page.run()
        qtbot.wait(20)
        import numpy as np
        image = np.asarray(page.plot.ax.get_images()[0].get_array(), dtype=float)
        rows = np.nansum(image, axis=1)
        assert np.allclose(rows[~np.isnan(rows)], 1.0)
        assert "fraction of row total" in page.plot.figure.axes[1].get_ylabel()

    def test_log_mode_declares_itself_on_the_colorbar(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.mode_combo.setCurrentText("log10_inventory")
        page.run()
        qtbot.wait(20)
        assert "log10" in page.plot.figure.axes[1].get_ylabel()

    def test_switching_mode_redraws_without_rerunning(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_species_dataset())
        page.mode_combo.setCurrentText("inventory")
        page.run()
        qtbot.wait(20)
        page.mode_combo.setCurrentText("fraction")  # signal-driven redraw
        qtbot.wait(20)
        assert "fraction of row total" in page.plot.figure.axes[1].get_ylabel()
