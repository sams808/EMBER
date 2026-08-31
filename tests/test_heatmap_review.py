"""
Repro tests for the Heatmaps-migration correctness review (commit
d1c6d2a). Every test here failed against the pre-fix code.

Theme, again: the figure, the tables beside it, and the file exported
from them must all describe the same computation — and mass that the
engine cannot attribute must be reported, not quietly absent.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
import matrix_query as mq
from qt_heatmap import HeatmapPage

from conftest import REAL_CSV_PATH, requires_real_data


def _page(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    page = HeatmapPage(window)
    qtbot.addWidget(page)
    return page


def _sparse_dataset() -> dm.HanfordDataset:
    """Two tanks; tank 2 never mentions Fe at all, so (tank2, Fe) is a
    cell with NO measurement -- distinct from a reported zero."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101", "241-A-101", "241-AN-104"],
        "Analyte": ["Fe", "Na", "Na"],
        "WastePhase": ["Supernatant", "Supernatant", "Sludge"],
        "WasteType": ["T1", "T1", "T2"],
        "Inventory": [10.0, 20.0, 30.0],
        "Units": ["kg", "kg", "kg"],
    }))
    dataset.report = None
    return dataset


def _unparsed_dataset() -> dm.HanfordDataset:
    """A tank holding an organic the parser cannot resolve to elements:
    real mass the engine cannot attribute to anything."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-C-202", "241-C-202"],
        "Analyte": ["Na", "Aroclor-1254"],
        "WastePhase": ["Sludge", "Sludge"],
        "WasteType": ["T1", "T1"],
        "Inventory": [90.0, 10.0],
        "Units": ["kg", "kg"],
    }))
    dataset.report = None
    return dataset


class TestNoMeasurementIsNotZero:
    """The Wide tab and the exported CSV zero-filled absent cells while
    the figure (correctly) painted them "no data" — the two halves of one
    screen asserting different facts about 1,207 real cells."""

    def test_absent_cell_stays_blank_in_the_wide_matrix(self):
        result = mq.engine_matrix(_sparse_dataset(), basis="kg", top_n_elements=10)
        wide = result.wide_df.set_index("WasteSiteId")
        assert pd.isna(wide.loc["241-AN-104", "Fe"]), (
            "a tank that never mentions Fe was published as containing 0 kg of it"
        )
        assert wide.loc["241-A-101", "Fe"] == pytest.approx(10.0)

    def test_wide_matrix_matches_what_the_figure_paints(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_sparse_dataset())
        page.mode_combo.setCurrentText("inventory")
        page.run()
        qtbot.wait(20)
        wide = page.wide_view.dataframe().set_index("WasteSiteId")
        image = np.asarray(page.plot.ax.get_images()[0].get_array(), dtype=float)
        # Exactly as many blanks in the table as grey cells on the figure.
        assert int(wide.isna().to_numpy().sum()) == int(np.isnan(image).sum())

    @requires_real_data
    def test_no_fabricated_zeros_on_the_real_dataset(self):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        result = mq.engine_matrix(dataset, basis="kg", top_n_elements=35)
        wide = result.wide_df.drop(columns=["WasteSiteId"])
        long_pairs = set(zip(result.long_df["WasteSiteId"], result.long_df["Element"]))
        # Every non-null cell must correspond to a real contribution.
        for tank, row in zip(result.wide_df["WasteSiteId"], wide.to_numpy()):
            for element, value in zip(wide.columns, row):
                if not np.isnan(value):
                    assert (tank, element) in long_pairs, f"{tank}/{element} invented"


class TestUnattributableMassIsReported:
    def test_unparsed_analyte_mass_is_warned_about(self):
        result = mq.engine_matrix(_unparsed_dataset(), basis="kg", top_n_elements=10)
        text = " ".join(result.warnings)
        assert "10" in text and "not attributable" in text.lower(), (
            f"10 kg of unparsed organics vanished silently; warnings were {result.warnings}"
        )

    def test_fully_attributable_data_raises_no_such_warning(self):
        result = mq.engine_matrix(_sparse_dataset(), basis="kg", top_n_elements=10)
        assert not any("not attributable" in w.lower() for w in result.warnings)

    @requires_real_data
    def test_real_dataset_reports_the_unbooked_organics(self):
        dataset = dm.HanfordDataset()
        dataset.load(REAL_CSV_PATH, use_cache=True)
        result = mq.engine_matrix(dataset, basis="kg", top_n_elements=35)
        note = " ".join(w for w in result.warnings if "not attributable" in w.lower())
        assert note, "the 515 kg of unparsed organics is still silent"
        assert "515" in note, f"the warning does not quantify the real loss: {note!r}"


class TestLegacyPathDisclosesItsOwnTrims:
    def test_top_n_trim_is_disclosed_under_as_reported(self):
        result = mq.legacy_matrix(_sparse_dataset(), unit="kg", top_n_elements=1)
        assert any("omitted" in w for w in result.warnings), (
            "the as-reported path trimmed columns with no warning at all"
        )

    def test_min_inventory_filter_is_disclosed(self):
        result = mq.legacy_matrix(_sparse_dataset(), unit="kg", top_n_elements=10,
                                  min_inventory=15.0)
        assert any("min inventory" in w.lower() for w in result.warnings)


class TestFigureNeverOutlivesItsData:
    def test_empty_result_clears_the_previous_figure(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_sparse_dataset())
        page.run()
        qtbot.wait(20)
        assert page.plot.ax.get_images(), "precondition: a matrix was drawn"

        page.scope.set_filter("WastePhase", ["Nonexistent"])
        page.run()
        qtbot.wait(20)
        assert page.wide_view.dataframe().empty
        assert not page.plot.ax.get_images(), (
            "the previous scope's heatmap is still on screen beside empty tables"
        )

    def test_scope_can_recover_after_an_empty_result(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_sparse_dataset())
        page.run()
        qtbot.wait(20)
        page.scope.set_filter("WastePhase", ["Nonexistent"])
        page.on_scope_changed()
        qtbot.wait(20)
        assert page.wide_view.dataframe().empty

        page.scope.set_filter("WastePhase", None)   # back to everything
        page.on_scope_changed()
        qtbot.wait(20)
        assert not page.wide_view.dataframe().empty, (
            "once a scope emptied the matrix, later scope changes stopped rebuilding"
        )


class TestFigureLabelsMatchTheDataDrawn:
    def test_changing_basis_without_rebuilding_does_not_relabel(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_sparse_dataset())
        page.unit_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)
        assert page._result.value_column == "Total_kg"

        page.unit_combo.setCurrentText("Ci")      # not wired to a rebuild
        page.mode_combo.setCurrentText("inventory")  # wired -> redraws
        qtbot.wait(20)
        title = (page.plot.figure.get_suptitle() or "") + page.plot.ax.get_title()
        assert "Ci" not in title, (
            "the figure claims Ci while drawing kg data"
        )
        assert "kg" in page.plot.figure.axes[1].get_ylabel()

    def test_changing_attribution_without_rebuilding_does_not_relabel(self, qtbot):
        page = _page(qtbot)
        page.on_dataset_changed(_sparse_dataset())
        page.attribution_combo.setCurrentText(mq.AS_REPORTED)
        page.run()
        qtbot.wait(20)

        page.attribution_combo.setCurrentText(mq.APPORTIONED)
        page.mode_combo.setCurrentText("inventory")
        qtbot.wait(20)
        assert mq.AS_REPORTED in page.plot.ax.get_title(), (
            "the figure claims apportioned values while drawing as-reported ones"
        )

    def test_export_records_the_settings_that_produced_the_tables(self, qtbot, tmp_path):
        page = _page(qtbot)
        dataset = _sparse_dataset()
        dataset.output_root = tmp_path
        page.on_dataset_changed(dataset)
        page.attribution_combo.setCurrentText(mq.AS_REPORTED)
        page.unit_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)

        page.attribution_combo.setCurrentText(mq.APPORTIONED)  # no rebuild
        page._export_inputs()
        out_dir = next(iter(tmp_path.glob("heatmap_*")))
        params = pd.read_csv(out_dir / "heatmap_parameters.csv").set_index("parameter")
        assert params.loc["attribution", "value"] == mq.AS_REPORTED, (
            "the export labels the tables with an attribution that did not produce them"
        )
        context = (out_dir / "context.csv").read_text(encoding="utf-8")
        assert mq.AS_REPORTED in context


class TestRowCountsAreHonest:
    def test_a_species_row_is_not_counted_once_per_constituent(self):
        # SO4 feeds both S and O, so a basket over both sees one dataset
        # row twice. The count column must not claim two rows exist.
        dataset = dm.HanfordDataset()
        dataset.df = dataset._clean_dataframe(pl.DataFrame({
            "WasteSiteId": ["241-A-101"],
            "Analyte": ["SO4"],
            "WastePhase": ["Supernatant"],
            "WasteType": ["T1"],
            "Inventory": [100.0],
            "Units": ["kg"],
        }))
        dataset.report = None
        from query_engine import Basket, run_query
        result = run_query(dataset, subject=("basket", Basket("t", ("S", "O"))), basis="kg")
        assert "N_contributions" in result.table.columns, (
            "a count of contributions must not be labelled a count of rows"
        )
        assert result.table["N_contributions"].iloc[0] == 2
