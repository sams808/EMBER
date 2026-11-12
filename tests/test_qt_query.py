"""
Query workspace (qt_query.QueryPage) tests: UI wiring over the L1 engine
and the L2 grammar. Engine *numbers* are covered in test_query_engine.py;
here the assertions are about what the page does with a result -- tables,
warnings label, grammar plots, templates, global scope, export -- using
the same hand-built chem dataset so any value that IS checked is still
hand-derivable.
"""
import polars as pl
import pytest
from PySide6.QtWidgets import QMainWindow

import data_model as dm
import reference_data as rd
from qt_query import QueryPage
from query_templates import TEMPLATES


def _make_page(qtbot):
    app_window = QMainWindow()  # only used for .statusBar()
    qtbot.addWidget(app_window)
    page = QueryPage(app_window)
    qtbot.addWidget(page)
    return page


def _chem_dataset() -> dm.HanfordDataset:
    """Mirror of test_query_engine._chem_dataset: one tank, two layers
    (Supernatant/T1 2 kL x 1.5, Sludge/T2 1 kL x 1.0), species + elemental
    + Ci rows so basket runs produce warnings AND disclosures."""
    dataset = dm.HanfordDataset()
    dataset.df = dataset._clean_dataframe(pl.DataFrame({
        "WasteSiteId": ["241-A-101"] * 5 + ["241-A-101"],
        "Analyte": ["SO4", "Sulfide", "Cl", "129I", "Cr", "Fe"],
        "WastePhase": ["Supernatant"] * 5 + ["Sludge"],
        "WasteType": ["T1"] * 5 + ["T2"],
        "Inventory": [100.0, 5.0, 3.0, 2.0, 4.0, 7.0],
        "Units": ["kg", "kg", "kg", "Ci", "kg", "kg"],
        "Volume": [2.0] * 5 + [1.0],
        "ComponentDensity": [1.5] * 5 + [1.0],
    }))
    dataset.report = None
    return dataset


def _run_element(page, symbol, basis="kg", group1=None):
    page.kind_combo.setCurrentText("Element")
    page.subject_edit.setText(symbol)
    page.basis_combo.setCurrentText(basis)
    if group1 is not None:
        page.group1_combo.setCurrentText(group1)
    page.run()


class TestWithoutDataset:
    def test_run_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page.run()  # QMessageBox.information neutralized by conftest
        assert page._result is None

    def test_export_shows_message_not_crash(self, qtbot):
        page = _make_page(qtbot)
        page._export()  # must not raise


class TestSubjectKindToggle:
    def test_basket_kind_shows_basket_combo_hides_subject_edit(self, qtbot):
        page = _make_page(qtbot)  # starts on "Basket"
        assert not page.basket_combo.isHidden()
        assert page.subject_edit.isHidden()

    def test_element_kind_swaps_the_two_widgets(self, qtbot):
        page = _make_page(qtbot)
        page.kind_combo.setCurrentText("Element")
        assert page.basket_combo.isHidden()
        assert not page.subject_edit.isHidden()

    def test_blank_subject_is_refused_not_queried(self, qtbot):
        page = _make_page(qtbot)
        page.dataset = _chem_dataset()
        page.kind_combo.setCurrentText("Element")
        page.subject_edit.setText("   ")
        page.run()
        assert page._result is None


class TestGlobalScope:
    def test_scope_filters_reach_the_engine(self, qtbot, sample_dataset):
        # Fe is 50 kg Solid + 10 kg Liquid on 241-A-101; scoping to Solid
        # must leave exactly the 50.
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.scope.set_filter("WastePhase", ["Solid"])
        _run_element(page, "Fe")
        qtbot.wait(20)
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(50.0)

    def test_scope_is_stamped_on_the_plot_subtitle(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.scope.set_filter("WastePhase", ["Solid"])
        _run_element(page, "Fe")
        qtbot.wait(20)
        assert "Scope: Solid" in page.plot.ax.get_title()

    def test_scope_change_reruns_a_standing_result(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        _run_element(page, "Fe")
        qtbot.wait(20)
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(60.0)
        page.scope.set_filter("WastePhase", ["Solid"])
        page.on_scope_changed()  # the shell fans this out on scope.changed
        qtbot.wait(20)
        assert page.result_view.dataframe()["Total_kg"].sum() == pytest.approx(50.0)

    def test_scope_change_without_a_result_does_not_run(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        page.scope.set_filter("WastePhase", ["Solid"])
        page.on_scope_changed()
        assert page._result is None


class TestRunFillsTheWorkspace:
    def test_element_run_fills_result_table_and_status(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        _run_element(page, "Fe")
        qtbot.wait(20)
        table = page.result_view.dataframe()
        assert table["Total_kg"].sum() == pytest.approx(60.0)  # 50 Solid + 10 Liquid
        assert "Query completed" in page.app_window.statusBar().currentMessage()

    def test_basket_run_fills_member_table_and_notes(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Basket")
        page.basket_combo.setCurrentText("Salt formers")
        page.basis_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)
        members = set(page.member_view.dataframe()["Member"])
        assert {"S", "Cl", "Cr"} <= members
        notes = page.notes_view.dataframe()
        assert (notes["kind"] == "disclosure").any()
        # The iodine Ci row can't count under a kg basis -- that exclusion
        # must surface as a warning row in Notes, not vanish.
        assert (notes["kind"] == "warning").any()

    def test_apportioned_number_reaches_the_table_unmangled(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        _run_element(page, "S", group1="—")
        qtbot.wait(20)
        expected = 100.0 * rd.SPECIES_ELEMENT_MASS_FRACTIONS["SO4"]["S"] + 5.0
        assert page.result_view.dataframe()["Total_kg"].iloc[0] == pytest.approx(expected)


class TestTemplates:
    """The picker itself lives in View -> Guided questions (see
    test_qt_shell); what this page owns is applying one."""

    def test_applying_a_template_sets_controls_and_runs(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        template = next(t for t in TEMPLATES if t.key == "salt_formers_wtpct")
        page.apply_template(template)
        qtbot.wait(20)
        assert page.kind_combo.currentText() == "Basket"
        assert page.basket_combo.currentText() == "Salt formers"
        assert page.basis_combo.currentText() == "wt%"
        assert page.group1_combo.currentText() == "Tank"
        assert page.group2_combo.currentText() == "Waste phase"
        assert page._result is not None

    def test_every_shipped_template_applies_cleanly(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        for template in TEMPLATES:
            page.apply_template(template)   # must not raise for any of them
            qtbot.wait(5)


class TestGrammarPlots:
    def _salt_formers_two_dims(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Basket")
        page.basket_combo.setCurrentText("Salt formers")
        page.basis_combo.setCurrentText("kg")
        page.group1_combo.setCurrentText("Tank")
        page.group2_combo.setCurrentText("Waste phase")
        page.run()
        return page

    def test_ungrouped_run_plots_the_per_member_split(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Basket")
        page.basket_combo.setCurrentText("Salt formers")
        page.basis_combo.setCurrentText("kg")
        page.group1_combo.setCurrentText("—")
        page.group2_combo.setCurrentText("—")
        page.run()
        qtbot.wait(20)
        labels = [t.get_text() for t in page.plot.ax.get_yticklabels()]
        assert "Cl" in labels and "Cr" in labels

    def test_two_dims_bar_uses_hue_legend_not_joined_labels(self, qtbot):
        page = self._salt_formers_two_dims(qtbot)
        qtbot.wait(20)
        legend = page.plot.ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "Waste phase"      # the dimension's readable label, not the column name
        labels = [t.get_text() for t in page.plot.ax.get_yticklabels()]
        assert all("·" not in label for label in labels)

    def test_heatmap_mark_draws_a_matrix(self, qtbot):
        page = self._salt_formers_two_dims(qtbot)
        page.mark_combo.setCurrentText("Heatmap")
        qtbot.wait(20)
        assert len(page.plot.ax.get_images()) == 1

    def test_one_dim_stacked_bar_splits_by_member(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Basket")
        page.basket_combo.setCurrentText("Salt formers")
        page.basis_combo.setCurrentText("kg")
        page.group1_combo.setCurrentText("Waste phase")
        page.group2_combo.setCurrentText("—")
        page.run()
        page.mark_combo.setCurrentText("Stacked bar")
        qtbot.wait(20)
        legend = page.plot.ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "Member"

    def test_box_without_two_dims_explains_instead_of_drawing(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        _run_element(page, "Fe")
        page.mark_combo.setCurrentText("Box (distribution)")
        qtbot.wait(20)
        texts = [t.get_text() for t in page.plot.ax.texts]
        assert any("two grouping dimensions" in t for t in texts)


class TestWarningsLabel:
    def test_visible_with_engine_warnings(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Basket")
        page.basket_combo.setCurrentText("Salt formers")
        page.basis_combo.setCurrentText("kg")  # 129I is Ci -> exclusion warning
        page.run()
        qtbot.wait(20)
        assert not page.warnings_label.isHidden()
        # One calm line above the result, every engine warning one click
        # away and in full in text(): no alarm glyph, nothing dropped.
        warnings = list(page._result.warnings)
        assert warnings
        assert list(page.warnings_label.notes()) == warnings
        summary = page.warnings_label.summary_label.text()
        if len(warnings) == 1:
            assert summary == warnings[0]
        else:
            assert summary.startswith(f"{len(warnings)} notes qualify this result")
            assert not page.warnings_label.toggle.isHidden()
        assert "⚠" not in summary
        for warning in warnings:
            assert warning in page.warnings_label.detail_label.text()

    def test_run_button_says_update_when_the_controls_left_the_result(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        assert page.run_btn.text() == "Run"
        _run_element(page, "Fe")
        qtbot.wait(20)
        assert page.run_btn.text() == "Run"
        page.basis_combo.setCurrentText("g")
        assert page.run_btn.text() == "Update"
        page.run()
        qtbot.wait(20)
        assert page.run_btn.text() == "Run"

    def test_subject_field_offers_what_the_dataset_holds(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Analyte")
        model = page.subject_edit.completer().model()
        offered = {model.data(model.index(i, 0)) for i in range(model.rowCount())}
        assert {"SO4", "129I", "Cr"} <= offered
        page.kind_combo.setCurrentText("Element")
        model = page.subject_edit.completer().model()
        offered = {model.data(model.index(i, 0)) for i in range(model.rowCount())}
        assert "S" in offered and "SO4" not in offered

    def test_clicking_a_bar_selects_its_row(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(sample_dataset)
        _run_element(page, "Fe")
        qtbot.wait(20)
        frame = page.result_view.dataframe()
        tank = str(frame["WasteSiteId"].iloc[-1])
        page._on_mark_clicked({"label": tank, "value": 1.0})
        selected = page.result_view.selectedIndexes()
        assert selected
        assert str(frame["WasteSiteId"].iloc[selected[0].row()]) == tank
        page._on_mark_clicked({"label": "not a row"})     # no match: selection untouched, no error

    def test_hidden_again_after_a_clean_run(self, qtbot):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        page.kind_combo.setCurrentText("Basket")
        page.basket_combo.setCurrentText("Salt formers")
        page.basis_combo.setCurrentText("kg")
        page.run()
        qtbot.wait(20)
        assert not page.warnings_label.isHidden()  # precondition
        _run_element(page, "Fe")  # elemental kg row only: no warnings
        qtbot.wait(20)
        assert page.warnings_label.isHidden()

    def test_engine_valueerror_shows_dialog_and_keeps_last_result(self, qtbot):
        # wt% grouped by a non-layer dimension is a refusal (ValueError) --
        # the page must surface it and leave the previous result standing.
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        _run_element(page, "Fe")
        qtbot.wait(20)
        previous = page._result
        assert previous is not None
        page.basis_combo.setCurrentText("wt%")
        page.group1_combo.setCurrentText("Element")
        page.run()  # QMessageBox.warning neutralized; must not raise
        assert page._result is previous


class TestDatasetReload:
    def test_dataset_change_clears_stale_result_everywhere(self, qtbot, sample_dataset):
        page = _make_page(qtbot)
        page.on_dataset_changed(_chem_dataset())
        _run_element(page, "S")
        qtbot.wait(20)
        assert page._result is not None  # precondition

        page.on_dataset_changed(sample_dataset)
        qtbot.wait(20)
        assert page._result is None
        assert page.result_view.dataframe().empty
        assert page.warnings_label.isHidden()


class TestExport:
    def test_export_writes_tables_plot_and_scope_context(self, qtbot, tmp_path):
        page = _make_page(qtbot)
        dataset = _chem_dataset()
        dataset.output_root = tmp_path
        page.on_dataset_changed(dataset)
        page.scope.set_filter("WastePhase", ["Supernatant"])
        _run_element(page, "S")
        qtbot.wait(20)
        page._export()
        out_dirs = list(tmp_path.glob("query_S_*"))
        assert len(out_dirs) == 1
        assert (out_dirs[0] / "query_result.csv").exists()
        assert (out_dirs[0] / "query_per_member.csv").exists()
        assert (out_dirs[0] / "query_notes.csv").exists()
        assert (out_dirs[0] / "query_plot.png").exists()
        context = (out_dirs[0] / "context.csv").read_text(encoding="utf-8")
        assert "global_scope" in context and "Supernatant" in context
        assert "plot_spec" in context
